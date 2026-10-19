"""Slurm access for Mission Control: cluster snapshots, submit, cancel, signal. Plus FakeSlurm for tests.

Our jobs carry --comment=megaearth|<profile>|<pool>|<role> so the governor can find them statelessly.
"""
from __future__ import annotations

import json
import copy
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- parsing helpers
_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 ** 3}


def mem_mb(s: str | int | float | None) -> int:
    if s is None or s == "":
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    m = re.fullmatch(r"([\d.]+)([KMGTP]?)B?", s.strip(), re.I)
    if not m:
        return 0
    return int(float(m.group(1)) * _UNITS[(m.group(2) or "M").upper()])


def expand_hostlist(s: str | None) -> list[str]:
    """node[01-03,07],gpu01 -> [node01,node02,node03,node07,gpu01]."""
    if not s or s in ("(null)", "None"):
        return []
    out: list[str] = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", s):
        part = part.strip(",")
        m = re.fullmatch(r"([^\[]*)\[([^\]]*)\](.*)", part)
        if not m:
            if part:
                out.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            if "-" in rng:
                a, b = rng.split("-", 1)
                w = len(a)
                out += [f"{pre}{i:0{w}d}{post}" for i in range(int(a), int(b) + 1)]
            else:
                out.append(f"{pre}{rng}{post}")
    return out


def parse_tres(s: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _num(v, default=0):
    if isinstance(v, dict):
        if not v.get("set", True) or v.get("infinite"):
            return default
        return v.get("number", default)
    return default if v is None else v


def secs(t: str) -> int:
    """'1-02:03:04' / '04:00:00' / '30:00' -> seconds."""
    d = 0
    if "-" in t:
        dd, t = t.split("-", 1)
        d = int(dd)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return ((d * 24 + h) * 60 + m) * 60 + s


# --------------------------------------------------------------------------- model
@dataclass
class Node:
    name: str
    partitions: list[str]
    state: str
    cpus: int
    cpus_alloc: int
    mem_mb: int
    mem_alloc_mb: int
    gpus: int = 0
    gpus_alloc: int = 0

    @property
    def up(self) -> bool:
        bad = ("DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPONDING", "POWER", "INVAL", "UNKNOWN", "FUTURE",
               "REBOOT")
        st = self.state.upper()
        return not any(b in st for b in bad)

    @property
    def planned(self) -> bool:
        """Backfill has planned someone's pending job here (or it is reserved): never start new work here."""
        st = self.state.upper()
        return "PLANNED" in st or "RESERVED" in st

    @property
    def usable(self) -> bool:
        return self.up and not self.planned

    @property
    def free_cpus(self) -> int:
        return max(0, self.cpus - self.cpus_alloc) if self.up else 0

    @property
    def free_mem_mb(self) -> int:
        return max(0, self.mem_mb - self.mem_alloc_mb) if self.up else 0

    @property
    def free_gpus(self) -> int:
        return max(0, self.gpus - self.gpus_alloc) if self.up else 0


@dataclass
class Job:
    id: int
    name: str
    user: str
    state: str                 # PENDING / RUNNING / ...
    reason: str
    partitions: list[str]
    cpus: int
    n_nodes: int
    mem_per_node_mb: int
    gpus_per_node: int
    nodes: list[str]           # allocated nodes (running)
    req_nodes: list[str]
    exc_nodes: list[str]
    comment: str = ""
    priority: int = 0
    submit_t: float = 0.0
    start_t: float = 0.0
    time_limit_s: int = 0

    @property
    def cpus_per_node(self) -> int:
        return math.ceil(self.cpus / max(1, self.n_nodes))

    def mc_tag(self) -> tuple[str, str, str] | None:
        """(profile, pool, role) for our jobs, else None."""
        if not self.comment.startswith("megaearth|"):
            return None
        p = self.comment.split("|")
        return (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")


@dataclass
class Snapshot:
    t: float
    nodes: dict[str, Node]
    jobs: list[Job]
    partition_nodes: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.partition_nodes:
            pn: dict[str, list[str]] = {}
            for n in self.nodes.values():
                for p in n.partitions:
                    pn.setdefault(p, []).append(n.name)
            self.partition_nodes = {k: sorted(v) for k, v in pn.items()}

    def nodes_of(self, partitions: Iterable[str]) -> list[str]:
        s: set[str] = set()
        for p in partitions:
            s.update(self.partition_nodes.get(p, []))
        return sorted(s)


# --------------------------------------------------------------------------- real backend
class Slurm:
    def __init__(self, user: str | None = None, dry_run: bool = False):
        self.user = user or os.environ.get("USER") or ""
        self.dry_run = dry_run
        self.log: list[str] = []

    def _run(self, argv: list[str], timeout: float = 60, check: bool = True) -> str:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"{' '.join(argv[:3])}… rc={r.returncode}: {r.stderr.strip()[:500]}")
        return r.stdout

    def snapshot(self) -> Snapshot:
        nodes = parse_scontrol_nodes(self._run(["scontrol", "-o", "show", "node"]))
        jobs = parse_squeue_json(self._run(["squeue", "--json"], timeout=90))
        return Snapshot(t=time.time(), nodes=nodes, jobs=jobs)

    def terminal(self, job_id: int) -> bool:
        """Missing squeue membership alone never authorises claim recovery."""
        text = self._run(["sacct", "-n", "-P", "-j", str(job_id), "-o", "JobIDRaw,State"])
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}
        for line in text.splitlines():
            fields = line.split("|")
            if fields[0] == str(job_id):
                return fields[1].split()[0].rstrip("+") in terminal
        return False

    def submit(self, argv: list[str]) -> int:
        cmd = ["sbatch", "--parsable", *argv]
        self.log.append(" ".join(cmd))
        if self.dry_run:
            return -1
        out = self._run(cmd).strip()
        return int(out.split(";")[0])

    def cancel(self, job_id: int, signal: str | None = None, batch: bool = False) -> None:
        cmd = ["scancel"]
        if signal:
            cmd += [f"--signal={signal}"]
        if batch:
            cmd += ["--batch"]
        cmd.append(str(job_id))
        self.log.append(" ".join(cmd))
        if not self.dry_run:
            self._run(cmd, check=False)


_NODE_KEYS = ("NodeName", "CPUAlloc", "CPUTot", "RealMemory", "AllocMem", "State", "Partitions", "CfgTRES",
              "AllocTRES")


def parse_scontrol_nodes(text: str) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    for line in text.splitlines():
        if not line.startswith("NodeName="):
            continue
        kv = {}
        for k in _NODE_KEYS:
            m = re.search(rf"(?:^| ){k}=(\S*)", line)
            kv[k] = m.group(1) if m else ""
        cfg = parse_tres(kv["CfgTRES"])
        alloc = parse_tres(kv["AllocTRES"])
        nodes[kv["NodeName"]] = Node(
            name=kv["NodeName"], partitions=[p for p in kv["Partitions"].split(",") if p],
            state=kv["State"], cpus=int(kv["CPUTot"] or 0), cpus_alloc=int(kv["CPUAlloc"] or 0),
            mem_mb=int(kv["RealMemory"] or 0), mem_alloc_mb=int(kv["AllocMem"] or 0),
            gpus=int(cfg.get("gres/gpu", 0) or 0), gpus_alloc=int(alloc.get("gres/gpu", 0) or 0))
    return nodes


def _gpus_from(job: dict, n_nodes: int) -> int:
    for key in ("tres_per_node", "tres_per_job", "tres_per_task"):
        s = job.get(key) or ""
        m = re.search(r"gres[/:]gpu(?::[^:=,]+)?[:=](\d+)", s)
        if m:
            g = int(m.group(1))
            if key == "tres_per_job":
                return math.ceil(g / max(1, n_nodes))
            if key == "tres_per_task":
                return math.ceil(g * max(1, _num(job.get("tasks"), 1)) / max(1, n_nodes))
            return g
    req = parse_tres(job.get("tres_req_str"))
    if "gres/gpu" in req:
        return math.ceil(int(req["gres/gpu"]) / max(1, n_nodes))
    return 0


def parse_squeue_json(text: str) -> list[Job]:
    doc = json.loads(text)
    jobs = []
    for j in doc.get("jobs", []):
        st = j.get("job_state")
        state = (st[0] if isinstance(st, list) and st else st) or ""
        n_nodes = max(1, int(_num(j.get("node_count"), 1) or 1))
        cpus = int(_num(j.get("cpus"), 1) or 1)
        mpn = _num(j.get("memory_per_node"), 0)
        mpc = _num(j.get("memory_per_cpu"), 0)
        req = parse_tres(j.get("tres_req_str"))
        if mpn:
            mem = int(mpn)
        elif mpc:
            mem = int(mpc) * math.ceil(cpus / n_nodes)
        else:
            mem = mem_mb(req.get("mem", "0")) // n_nodes
        parts = [p for p in (j.get("partition") or "").split(",") if p]
        jobs.append(Job(
            id=int(j.get("job_id")), name=j.get("name") or "", user=j.get("user_name") or "",
            state=state, reason=j.get("state_reason") or "", partitions=parts, cpus=cpus, n_nodes=n_nodes,
            mem_per_node_mb=mem, gpus_per_node=_gpus_from(j, n_nodes),
            nodes=expand_hostlist(j.get("nodes")), req_nodes=expand_hostlist(j.get("required_nodes")),
            exc_nodes=expand_hostlist(j.get("excluded_nodes")), comment=j.get("comment") or "",
            priority=int(_num(j.get("priority"), 0) or 0), submit_t=float(_num(j.get("submit_time"), 0) or 0),
            start_t=float(_num(j.get("start_time"), 0) or 0),
            time_limit_s=int(_num(j.get("time_limit"), 0) or 0) * 60))
    return jobs


# --------------------------------------------------------------------------- fake backend (tests)
class FakeSlurm:
    """In-memory cluster. Submitted jobs pinned with --nodelist start immediately if they fit."""

    def __init__(self, nodes: dict[str, Node], user: str = "me"):
        self.nodes = nodes
        self.jobs: list[Job] = []
        self.user = user
        self.next_id = 1000
        self.log: list[str] = []
        self.signals: list[tuple[int, str]] = []
        self.t = 1_000_000.0

    def snapshot(self) -> Snapshot:
        return Snapshot(t=self.t, nodes=copy.deepcopy(self.nodes), jobs=copy.deepcopy(self.jobs))

    def _fits(self, n: Node, j: Job) -> bool:
        return (n.usable and n.free_cpus >= j.cpus_per_node and n.free_mem_mb >= j.mem_per_node_mb
                and n.free_gpus >= j.gpus_per_node)

    def _alloc(self, n: Node, j: Job, sign: int) -> None:
        n.cpus_alloc += sign * j.cpus_per_node
        n.mem_alloc_mb += sign * j.mem_per_node_mb
        n.gpus_alloc += sign * j.gpus_per_node

    def add_job(self, user: str, partition: str, cpus: int, mem_mb_: int, gpus: int = 0, state: str = "PENDING",
                reason: str = "Resources", nodes: list[str] | None = None, name: str = "x", comment: str = "") -> Job:
        self.next_id += 1
        j = Job(id=self.next_id, name=name, user=user, state=state, reason=reason, partitions=[partition],
                cpus=cpus, n_nodes=1, mem_per_node_mb=mem_mb_, gpus_per_node=gpus, nodes=[], req_nodes=[],
                exc_nodes=[], comment=comment, submit_t=self.t)
        if state == "RUNNING":
            node = (nodes or [None])[0]
            if node is None:
                raise ValueError("RUNNING job needs a node")
            j.nodes = [node]
            j.start_t = self.t
            self._alloc(self.nodes[node], j, +1)
        self.jobs.append(j)
        return j

    def submit(self, argv: list[str]) -> int:
        opts = {}
        for a in argv:
            if a.startswith("--") and "=" in a:
                k, v = a[2:].split("=", 1)
                opts[k] = v
        self.log.append(" ".join(argv))
        gpus = int(re.search(r"gpu:(\d+)", opts.get("gres", "gpu:0")).group(1)) if "gres" in opts else 0
        j = self.add_job(self.user, opts.get("partition", "short"), int(opts.get("cpus-per-task", 1)),
                         mem_mb(opts.get("mem", "1G")), gpus, name=opts.get("job-name", ""),
                         comment=opts.get("comment", ""))
        node = opts.get("nodelist")
        if node and self._fits(self.nodes[node], j):
            j.state, j.reason, j.nodes, j.start_t = "RUNNING", "None", [node], self.t
            self._alloc(self.nodes[node], j, +1)
        return j.id

    def cancel(self, job_id: int, signal: str | None = None, batch: bool = False) -> None:
        if signal:
            self.signals.append((job_id, signal))
            return
        self.finish(job_id)

    def finish(self, job_id: int) -> None:
        for j in list(self.jobs):
            if j.id == job_id:
                if j.state == "RUNNING":
                    self._alloc(self.nodes[j.nodes[0]], j, -1)
                self.jobs.remove(j)

    def schedule(self) -> None:
        """Start pending foreign jobs that fit (simple FIFO by priority)."""
        for j in sorted([j for j in self.jobs if j.state == "PENDING"], key=lambda j: -j.priority):
            cands = self.nodes.values() if not j.req_nodes else [self.nodes[n] for n in j.req_nodes]
            for n in cands:
                if (j.reason in ("Resources", "Priority", "None", "") and n.name not in j.exc_nodes
                        and j.partitions[0] in n.partitions and self._fits(n, j)):
                    j.state, j.reason, j.nodes, j.start_t = "RUNNING", "None", [n.name], self.t
                    self._alloc(n, j, +1)
                    break


def enucc_nodes() -> dict[str, Node]:
    """ENUCC topology (2026-10) for tests."""
    out = {}
    for i in range(1, 9):
        out[f"node{i:02d}"] = Node(f"node{i:02d}", ["nodes", "short", "long"], "IDLE", 64, 0, 514048, 0)
    for i in (1, 2):
        out[f"himem{i:02d}"] = Node(f"himem{i:02d}", ["himem"], "IDLE", 128, 0, 2051072, 0)
    for i in range(1, 5):
        out[f"gpu{i:02d}"] = Node(f"gpu{i:02d}", ["gpu"], "IDLE", 64, 0, 256000, 0, 2, 0)
    return out
