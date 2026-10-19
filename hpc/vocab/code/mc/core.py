"""State on the shared filesystem: atomic writes, claims, done markers, checkpoints, events.

Layout under <root>/mc/:
  units/<stage>.json             frozen unit list for a stage (written once by the governor)
  claims/<stage>/<unit>.claim    atomically linked, complete worker identity
  done/<stage>/<unit>.json       commit marker (outputs + sha256 + metrics); written LAST
  failed/<stage>/<unit>.<n>.json failure records (traceback); attempt counter = count
  ckpt/<stage>/<unit>/           mid-unit checkpoint state (survives crashes/yields)
  stages/<stage>.json            stage status after its finalize unit: complete | failed
  workers/<worker>.json          worker heartbeat (atomic rewrite every ~30 s)
  events/<writer>.jsonl          append-only event log, one file per writer (no contention)
  control/{PAUSE,STOP}           operator switches; control/yield/<worker> per-worker yield
  status.json / status.txt       governor's view, read by status.sh
"""
from __future__ import annotations

import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

UNIT_ID_RE = re.compile(r"^[A-Za-z0-9_.=+,-]{1,160}$")
FINALIZE = "__finalize__"


# --------------------------------------------------------------------------- atomic IO
def _tmp_name(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{socket.gethostname().split('.')[0]}.{os.getpid()}")


def atomic_write_bytes(path: os.PathLike | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: os.PathLike | str, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n").encode())


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        # a torn read is impossible with os.replace, so this is corruption: surface it
        raise


@contextlib.contextmanager
def atomic_path(path: os.PathLike | str) -> Iterator[Path]:
    """Yield a temporary sibling path; on clean exit it replaces `path` atomically.

    For writers that need a filename (np.save, pyarrow, rasterio). The temp keeps the final
    suffix so libraries that sniff extensions behave.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{socket.gethostname().split('.')[0]}.{os.getpid()}{path.suffix}")
    try:
        yield tmp
        if tmp.is_file():
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)


def sha256_file(path: os.PathLike | str, bufsize: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- config
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _add_only_merge(base: dict, over: dict, src: str, path: str = "") -> dict:
    """Merge `over` into `base` but refuse to change any key that already has a value.

    config.json holds the pre-registered science parameters; config.d/*.json may only ADD
    engineering keys, never silently override a registered value.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        here = f"{path}.{k}" if path else k
        if k not in out:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _add_only_merge(out[k], v, src, here)
        elif out[k] != v:
            raise ValueError(f"{src}: may not override registered key {here!r} "
                             f"({out[k]!r} -> {v!r}); edit config.json via a PLAN amendment")
    return out


def load_config(code_root: os.PathLike | str, profile: str | None = None) -> dict:
    """config.json + config.d/*.json (add-only) + profiles/<profile>/*.json (overrides).

    Legacy config.<profile>.json is also honoured as a profile overlay.
    """
    code_root = Path(code_root)
    cfg = read_json(code_root / "config.json")
    if cfg is None:
        raise FileNotFoundError(code_root / "config.json")
    sources = ["config.json"]
    for p in sorted((code_root / "config.d").glob("*.json")):
        cfg = _add_only_merge(cfg, read_json(p) or {}, f"config.d/{p.name}")
        sources.append(f"config.d/{p.name}")
    profile = profile or os.environ.get("MEGAEARTH_PROFILE") or "full"
    if profile != "full":
        overlays = sorted((code_root / "profiles" / profile).glob("*.json"))
        legacy = code_root / f"config.{profile}.json"
        if legacy.exists():
            overlays.insert(0, legacy)
        if not overlays:
            raise FileNotFoundError(f"profile {profile!r}: no profiles/{profile}/*.json")
        for p in overlays:
            cfg = deep_merge(cfg, read_json(p) or {})
            sources.append(os.path.relpath(p, code_root))
    cfg["_profile"] = profile
    cfg["_sources"] = sources
    cfg["_fingerprint"] = sha256_json({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


# --------------------------------------------------------------------------- identity
def worker_id() -> str:
    j = os.environ.get("SLURM_ARRAY_JOB_ID")
    if j:
        return f"{j}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    return os.environ.get("SLURM_JOB_ID") or f"local-{socket.gethostname().split('.')[0]}-{os.getpid()}"


def host() -> str:
    return socket.gethostname().split(".")[0]


def check_unit_id(uid: str) -> str:
    if not UNIT_ID_RE.match(uid):
        raise ValueError(f"bad unit id {uid!r}")
    return uid


# --------------------------------------------------------------------------- state store
class Store:
    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)
        self.mc = self.root / "mc"
        self._verified = {}

    # paths ---------------------------------------------------------------
    def units_path(self, stage: str) -> Path:
        return self.mc / "units" / f"{stage}.json"

    def claim_path(self, stage: str, unit: str) -> Path:
        return self.mc / "claims" / stage / f"{unit}.claim"

    def done_path(self, stage: str, unit: str) -> Path:
        return self.mc / "done" / stage / f"{unit}.json"

    def failed_dir(self, stage: str) -> Path:
        return self.mc / "failed" / stage

    def ckpt_dir(self, stage: str, unit: str) -> Path:
        return self.mc / "ckpt" / stage / unit

    def stage_path(self, stage: str) -> Path:
        return self.mc / "stages" / f"{stage}.json"

    def worker_path(self, wid: str) -> Path:
        return self.mc / "workers" / f"{wid}.json"

    def control(self, name: str) -> Path:
        return self.mc / "control" / name

    def yield_path(self, wid: str) -> Path:
        return self.mc / "control" / "yield" / wid

    # units ---------------------------------------------------------------
    def write_units(self, stage: str, units: list[dict], meta: dict | None = None) -> None:
        seen = set()
        for u in units:
            check_unit_id(u["id"])
            if u["id"] in seen or u["id"] in (FINALIZE, "__plan__"):
                raise ValueError(f"{stage}: duplicate/reserved unit id {u['id']}")
            seen.add(u["id"])
        doc = {"stage": stage, "units": units, "meta": meta or {}, "t": time.time()}
        doc["fingerprint"] = sha256_json(units)
        atomic_write_json(self.units_path(stage), doc)

    def read_units(self, stage: str) -> list[dict] | None:
        doc = read_json(self.units_path(stage))
        return None if doc is None else doc["units"]

    # claims --------------------------------------------------------------
    @contextlib.contextmanager
    def claim_lock(self, stage: str, unit: str):
        path = self.mc / "locks" / stage / f"{unit}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def claim(self, stage: str, unit: str, wid: str | None = None) -> bool:
        with self.claim_lock(stage, unit):
            return self._claim(stage, unit, wid)

    def _claim(self, stage: str, unit: str, wid: str | None = None) -> bool:
        p = self.claim_path(stage, unit)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return False
        # Publish only a fully written identity. A crash or filesystem error during
        # an O_EXCL write can otherwise leave an empty claim with no recoverable owner.
        tmp = _tmp_name(p)
        try:
            with open(tmp, "w") as f:
                json.dump({"worker": wid or worker_id(), "host": host(), "pid": os.getpid(), "t": time.time()}, f)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, p)  # Exclusive publication; never overwrite another owner.
            except FileExistsError:
                return False
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        if self.is_done(stage, unit):  # lost a race with a finished worker
            os.remove(p)
            return False
        return True

    def claim_owner(self, stage: str, unit: str) -> dict | None:
        try:
            return read_json(self.claim_path(stage, unit))
        except json.JSONDecodeError:
            return {"worker": "?", "t": 0}

    def release(self, stage: str, unit: str, wid: str | None = None) -> None:
        with self.claim_lock(stage, unit):
            self._release(stage, unit, wid)

    def recover(self, stage: str, unit: str, expected: dict) -> bool:
        """Caller has verified terminal allocation; compare identity under the claim lock."""
        with self.claim_lock(stage, unit):
            if self.claim_owner(stage, unit) != expected:
                return False
            self._release(stage, unit, expected["worker"])
            return True

    def _release(self, stage: str, unit: str, wid: str | None = None) -> None:
        p = self.claim_path(stage, unit)
        owner = self.claim_owner(stage, unit)
        if owner is None:
            return
        if wid is not None and owner.get("worker") != wid:
            return
        with contextlib.suppress(FileNotFoundError):
            os.remove(p)

    def claimed(self, stage: str) -> set[str]:
        d = self.mc / "claims" / stage
        try:
            return {n[:-6] for n in os.listdir(d) if n.endswith(".claim")}
        except FileNotFoundError:
            return set()

    # done ----------------------------------------------------------------
    def mark_done(self, stage: str, unit: str, outputs: Iterable[os.PathLike | str] = (),
                  metrics: dict | None = None, hash_outputs: bool = True) -> dict:
        outs = {}
        for o in outputs:
            o = Path(o)
            if not o.exists():
                raise FileNotFoundError(f"{stage}/{unit}: declared output missing: {o}")
            rel = os.path.relpath(o, self.root)
            if o.is_file():
                st_ = o.stat()
                outs[rel] = {"bytes": st_.st_size, "mtime_ns": st_.st_mtime_ns, "sha256": sha256_file(o)}
            else:
                outs[rel] = {"dir": True, "files": {
                    str(f.relative_to(o)): {"bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns,
                                            "sha256": sha256_file(f)}
                    for f in sorted(o.rglob("*")) if f.is_file()}}
        rec = {"stage": stage, "unit": unit, "worker": worker_id(), "host": host(), "t": time.time(),
               "outputs": outs, "metrics": metrics or {}}
        atomic_write_json(self.done_path(stage, unit), rec)
        self.release(stage, unit)
        return rec

    HASH_BUDGET_BYTES = 256 * 1024 ** 2   # strict passes re-hash files up to this size; larger untouched files rely on size+mtime

    def is_done(self, stage: str, unit: str, strict: bool = False) -> bool:
        """Completion check. Fast path: every declared output exists with its recorded size and mtime.
        A file whose mtime moved is re-hashed; `strict=True` re-hashes everything (finalize, freezes)."""
        marker = self.done_path(stage, unit)
        if not marker.exists():
            return False
        rec = read_json(marker)
        if rec.get("stage") != stage or rec.get("unit") != unit:
            raise ValueError(f"Invalid completion marker: {marker}")
        for rel, meta in rec.get("outputs", {}).items():
            path = self.root / rel
            files = meta.get("files") if meta.get("dir") else {"": meta}
            if files is None:
                raise ValueError(f"Unverifiable legacy directory output: {path}")
            if meta.get("dir"):
                actual = {str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()}
                if not path.is_dir() or actual != set(files):
                    raise ValueError(f"Output directory mismatch: {path}")
            for sub, info in files.items():
                target = path / sub if sub else path
                if not target.is_file() or target.stat().st_size != info["bytes"]:
                    raise ValueError(f"Missing or resized committed output: {target}")
                stat = target.stat()
                key = (str(target), stat.st_size, stat.st_mtime_ns, info.get("sha256"))
                if key in self._verified:
                    continue
                untouched = info.get("mtime_ns") is not None and stat.st_mtime_ns == info["mtime_ns"]
                if (strict and stat.st_size <= self.HASH_BUDGET_BYTES) or not untouched:
                    if not info.get("sha256") or sha256_file(target) != info["sha256"]:
                        raise ValueError(f"Committed output hash mismatch: {target}")
                self._verified[key] = True
        return True

    def verify_done(self, stage: str) -> int:
        """Finalize-time integrity pass: re-hash committed outputs up to HASH_BUDGET_BYTES each; larger
        untouched files (planet parts, imagery shards, area stores) are checked by size and mtime here and
        fully hashed once at the next freeze."""
        d = self.mc / "done" / stage
        try:
            names = [n[:-5] for n in os.listdir(d) if n.endswith(".json") and not n.startswith(".")]
        except FileNotFoundError:
            return 0
        for n in names:
            self.is_done(stage, n, strict=True)
        return len(names)

    def done(self, stage: str) -> set[str]:
        """Names of committed units: a directory listing only. Per-unit output verification happens when
        a unit is about to be reused (`execute` calls `is_done`) and for every unit at finalize
        (`verify_done`); doing it here made every worker poll re-read thousands of markers."""
        d = self.mc / "done" / stage
        try:
            return {n[:-5] for n in os.listdir(d) if n.endswith(".json") and not n.startswith(".")}
        except FileNotFoundError:
            return set()

    def done_records(self, stage: str) -> list[dict]:
        return [read_json(self.done_path(stage, u)) for u in sorted(self.done(stage))]

    # failures --------------------------------------------------------------
    def record_failure(self, stage: str, unit: str, kind: str, detail: str) -> int:
        d = self.failed_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        n = self.attempts(stage, unit) + 1
        atomic_write_json(d / f"{unit}.{n}.json", {"stage": stage, "unit": unit, "attempt": n, "kind": kind,
                                                   "worker": worker_id(), "host": host(), "t": time.time(),
                                                   "detail": detail[-20000:]})
        return n

    def failures(self, stage: str) -> dict[str, list[dict]]:
        d = self.failed_dir(stage)
        out: dict[str, list[dict]] = {}
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            return out
        for n in names:
            if n.endswith(".json") and not n.startswith("."):
                unit = n[:-5].rsplit(".", 1)[0]
                rec = read_json(d / n)
                if rec:
                    out.setdefault(unit, []).append(rec)
        return out

    def attempts(self, stage: str, unit: str) -> int:
        return len(self.failures(stage).get(unit, []))

    # stage status ---------------------------------------------------------
    def stage_status(self, stage: str) -> dict | None:
        return read_json(self.stage_path(stage))

    def set_stage_status(self, stage: str, status: str, report: dict | None = None) -> None:
        assert status in ("complete", "failed")
        atomic_write_json(self.stage_path(stage), {"stage": stage, "status": status, "t": time.time(),
                                                   "worker": worker_id(), "report": report or {}})

    # events ----------------------------------------------------------------
    def event(self, kind: str, **kw: Any) -> None:
        d = self.mc / "events"
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"t": time.time(), "kind": kind, "worker": worker_id(), "host": host(), **kw},
                          default=str)
        with open(d / f"{worker_id()}.jsonl", "a") as f:
            f.write(line + "\n")

    # control ---------------------------------------------------------------
    def flag(self, name: str) -> bool:
        return self.control(name).exists()

    def set_flag(self, name: str, on: bool = True, note: str = "") -> None:
        p = self.control(name)
        if on:
            atomic_write_json(p, {"t": time.time(), "note": note})
        else:
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)


# --------------------------------------------------------------------------- checkpoints
class Yielded(Exception):
    """Raised by a unit after saving its checkpoint because the worker must stop now."""


class Checkpoint:
    """Mid-unit state for long units. Survives yields, requeues, crashes and retries.

    Usage inside a unit:
        done = ckpt.load("progress", {"next": 0})
        for i in range(done["next"], n):
            ...work...
            if ckpt.due():                      # every ~60 s, or immediately when stopping
                ckpt.save("progress", {"next": i + 1})
            ckpt.maybe_stop()                   # raises Yielded (after you saved) if asked to stop
    """

    def __init__(self, store: Store, stage: str, unit: str, stop_flag=None, every_s: float = 60.0):
        self.dir = store.ckpt_dir(stage, unit)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._stop = stop_flag or (lambda: False)
        self._every = every_s
        self._last = time.monotonic()
        self.progress: dict[str, Any] = {}

    def load(self, name: str, default: Any = None) -> Any:
        return read_json(self.dir / f"{name}.json", default)

    def save(self, name: str, obj: Any) -> None:
        atomic_write_json(self.dir / f"{name}.json", obj)
        self._last = time.monotonic()

    def path(self, name: str) -> Path:
        return self.dir / name

    def stopping(self) -> bool:
        return bool(self._stop())

    def due(self) -> bool:
        return self.stopping() or (time.monotonic() - self._last) >= self._every

    def maybe_stop(self) -> None:
        if self.stopping():
            raise Yielded()

    def report(self, **kw: Any) -> None:
        self.progress.update(kw)

    def clear(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


def ensure_dirs(*paths: os.PathLike | str) -> None:
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except OSError as e:  # pragma: no cover
            if e.errno != errno.EEXIST:
                raise
