"""What can run now? Shared by workers (to claim) and the governor (to size pools)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .core import FINALIZE, Store, read_json
from .registry import Stage, critical_path_rank
from .runner import PLAN

GLOBAL_PAUSES = ("STOP", "PAUSE", "PAUSE_DISK")


@dataclass
class Item:
    stage: str
    unit: str
    pool: str
    cpus: int
    mem_gb: float
    gpus: int
    rank: float
    kind: str  # plan | unit | finalize


class WorkView:
    def __init__(self, store: Store, stages: dict[str, Stage], cfg: dict):
        self.store, self.stages, self.cfg = store, stages, cfg
        self.rank = critical_path_rank(stages)
        self._units: dict[str, tuple[float, list[dict]]] = {}
        self.backoff_s = float(cfg.get("mc", {}).get("unit_backoff_s", 60))

    # ------------------------------------------------------------------ stage states
    def _units_of(self, name: str) -> list[dict] | None:
        p = self.store.units_path(name)
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            return None
        c = self._units.get(name)
        if c and c[0] == m:
            return c[1]
        doc = read_json(p)
        units = doc["units"] if doc else None
        if units is not None:
            self._units[name] = (m, units)
        return units

    def status(self, name: str) -> str | None:
        status = (self.store.stage_status(name) or {}).get("status")
        if status == "complete":
            expected = {u["id"] for u in self.store.read_units(name) or []}
            done = self.store.done(name)
            missing = expected - done
            if missing and len(missing) / max(1, len(expected)) > self.stages[name].allow_failed_frac:
                raise ValueError(f"Completed stage has missing outputs: {name}")
        return status

    def state(self, name: str) -> str:
        """complete | failed | blocked (dep not complete) | dep_failed | gated | ready"""
        s = self.status(name)
        if s in ("complete", "failed"):
            return s
        st = self.stages[name]
        for d in st.deps:
            ds = self.status(d)
            if ds == "failed":
                return "dep_failed"
            if ds != "complete":
                return "blocked"
        if st.gate and not self.store.flag(st.gate):
            return "gated"
        return "ready"

    def paused(self) -> str | None:
        for f in GLOBAL_PAUSES:
            if self.store.flag(f):
                return f
        return None

    # ------------------------------------------------------------------ items
    def stage_items(self, name: str, now: float | None = None) -> tuple[list[Item], dict]:
        """Claimable items of one READY stage + a progress summary."""
        now = now or time.time()
        st = self.stages[name]
        rank = self.rank[name]
        units = self._units_of(name)
        fails = self.store.failures(name)
        def retry_ready(uid):
            recs = fails.get(uid, [])
            return (len(recs) < st.max_attempts and
                    not any(r.get("kind") == "skip" for r in recs) and
                    (not recs or now - max(r.get("t", 0) for r in recs) >=
                     self.backoff_s * (2 ** (len(recs) - 1))))
        claimed = self.store.claimed(name)
        if units is None:
            prog = {"planned": False}
            if PLAN in claimed or not retry_ready(PLAN):
                return [], prog
            c, m, g = st.unit_cpus, st.unit_mem_gb, 0
            return [Item(name, PLAN, st.plan_pool, max(1, min(c, 8)), max(2.0, min(m, 32.0)), g, rank + 0.5,
                         "plan")], prog
        done = self.store.done(name)
        fails = self.store.failures(name)
        items: list[Item] = []
        n_dead = n_out = 0
        for u in units:
            uid = u["id"]
            if uid in done:
                continue
            recs = fails.get(uid, [])
            if len(recs) >= st.max_attempts or any(r.get("kind") == "skip" for r in recs):
                n_dead += 1
                continue
            n_out += 1
            if uid in claimed:
                continue
            if recs:
                last = max(r.get("t", 0) for r in recs)
                if now - last < self.backoff_s * (2 ** (len(recs) - 1)):
                    continue
            c, m, g = st.unit_res(u)
            items.append(Item(name, uid, st.pool, c, m, g, rank, "unit"))
        prog = {"planned": True, "n": len(units), "done": len(done & {u["id"] for u in units}), "dead": n_dead,
                "outstanding": n_out, "claimed": len(claimed - {PLAN, FINALIZE})}
        if n_out == 0 and FINALIZE not in claimed and retry_ready(FINALIZE):
            items.append(Item(name, FINALIZE, st.plan_pool, max(1, min(st.unit_cpus, 16)),
                              max(4.0, min(st.unit_mem_gb, 64.0)), 0, rank + 0.25, "finalize"))
        return items, prog

    def all_items(self, pools: set[str] | None = None) -> tuple[list[Item], dict[str, dict]]:
        out: list[Item] = []
        summary: dict[str, dict] = {}
        if self.paused():
            return [], {"_paused": {"flag": self.paused()}}
        for name in self.stages:
            s = self.state(name)
            summary[name] = {"state": s}
            if s != "ready":
                continue
            items, prog = self.stage_items(name)
            summary[name].update(prog)
            out += [i for i in items if pools is None or i.pool in pools]
        out.sort(key=lambda i: -i.rank)
        return out, summary


def pick(items: list[Item], serves: list[str], free_cpus: int, free_mem_gb: float, free_gpus: int) -> Item | None:
    """Highest-priority item that fits; earlier pools in `serves` win ties (himem before cpu on himem nodes)."""
    order = {p: i for i, p in enumerate(serves)}
    best = None
    for it in items:
        if it.pool not in order:
            continue
        if it.cpus > free_cpus or it.mem_gb > free_mem_gb or it.gpus > free_gpus:
            continue
        key = (order[it.pool], -it.rank)
        if best is None or key < best[0]:
            best = (key, it)
    return best[1] if best else None


def node_tmp_ok(cfg: dict, path: str | None = None) -> bool:
    import shutil
    need = float(cfg.get("mc", {}).get("disk", {}).get("node_tmp_min_free_gb", 60))
    p = path or os.environ.get("MEGAEARTH_TMP") or "/tmp"
    try:
        return shutil.disk_usage(p).free / 1e9 >= need
    except FileNotFoundError:
        return False
