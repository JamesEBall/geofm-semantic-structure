"""I/O helpers: JSON logging, deterministic/atomic npz+json, fingerprints, freeze, parameters."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io as _io
import json
import math
import os
import socket
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import ENGINE_VERSION

CODE_DIR = Path(__file__).resolve().parent            # .../hpc/vocab/code/megaearth_stats
VOCAB_DIR = CODE_DIR.parent.parent                     # .../hpc/vocab
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class FreezeError(RuntimeError):
    """F3 freeze missing or not matching the files on disk."""


class SplitAccessError(RuntimeError):
    """Raised when a val-stage computation tries to touch test-split rows."""


# --------------------------------------------------------------------------- logging
def log(event: str, **kw: Any) -> None:
    rec = {"t": round(time.time(), 3), "event": event, "host": socket.gethostname().split(".")[0], **kw}
    print(json.dumps(jsonable(rec), sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)


def n_threads() -> int:
    for k in ("SLURM_CPUS_PER_TASK", "MEGAEARTH_THREADS"):
        v = os.environ.get(k)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return max(1, min(8, os.cpu_count() or 1))


def set_thread_env() -> int:
    """Set BLAS thread env vars from SLURM_CPUS_PER_TASK (call before numpy import in CLIs)."""
    n = n_threads()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, str(n))
    return n


# --------------------------------------------------------------------------- json
def jsonable(o: Any) -> Any:
    """Convert numpy scalars/arrays to python; NaN/inf -> None so json allow_nan=False never fails."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


def dumps(obj: Any, indent: int | None = 1) -> str:
    return json.dumps(jsonable(obj), indent=indent, sort_keys=True, allow_nan=False)


def atomic_write_bytes(path: os.PathLike | str, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{socket.gethostname().split('.')[0]}.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
    return path


def write_json(path: os.PathLike | str, obj: Any) -> Path:
    return atomic_write_bytes(path, (dumps(obj) + "\n").encode())


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return default


# --------------------------------------------------------------------------- npz
def npz_bytes(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any] | None = None) -> bytes:
    """Deterministic compressed npz: sorted entries, fixed timestamps, '__meta__' JSON entry."""
    items = dict(arrays)
    if meta is not None:
        items["__meta__"] = np.frombuffer(dumps(meta, indent=None).encode(), dtype=np.uint8)
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(items):
            arr = np.asarray(items[name])
            if arr.dtype == object:
                raise TypeError(f"npz entry {name}: object arrays are not allowed")
            ab = _io.BytesIO()
            np.lib.format.write_array(ab, np.ascontiguousarray(arr), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, ab.getvalue(), compresslevel=6)
    return buf.getvalue()


def write_npz(path: os.PathLike | str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any] | None = None) -> Path:
    return atomic_write_bytes(path, npz_bytes(arrays, meta))


def read_meta(path: os.PathLike | str) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "__meta__" not in z.files:
                return None
            return json.loads(bytes(z["__meta__"]).decode())
    except (FileNotFoundError, zipfile.BadZipFile, ValueError, OSError):
        return None


def read_npz(path: os.PathLike | str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(bytes(z["__meta__"]).decode()) if "__meta__" in z.files else {}
    return arrays, meta


def is_complete(path: os.PathLike | str, fingerprint: str) -> bool:
    meta = read_meta(path)
    return bool(meta) and meta.get("fingerprint") == fingerprint and meta.get("complete", False)


# --------------------------------------------------------------------------- hashing
def sha256_file(path: os.PathLike | str, bufsize: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(dumps(obj, indent=None).encode())


def code_files() -> list[Path]:
    files = sorted(CODE_DIR.glob("*.py"))
    for name in ("analyse_fit.py", "analyse_rsa.py", "analyse_infer.py"):
        p = CODE_DIR.parent / name
        if p.exists():
            files.append(p)
    return files


def code_hash() -> str:
    h = hashlib.sha256()
    h.update(ENGINE_VERSION.encode())
    for p in code_files():
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def fingerprint(*parts: Any) -> str:
    return sha256_json({"parts": list(parts), "code": code_hash()})


# --------------------------------------------------------------------------- freeze
def verify_freeze(root: os.PathLike | str, *, name: str = "F3", code_base: os.PathLike | str | None = None) -> dict:
    """Refuse unless freeze/<name>.json exists and every listed file's sha256 matches.

    Relative paths resolve against ROOT, then the hpc/vocab code base, then the repository root.
    Returns {"freeze": path, "n_files": n, "engine_code_frozen": bool, "unfrozen_engine_files": [...]}.
    """
    root = Path(root)
    fp = root / "freeze" / f"{name}.json"
    if not fp.exists():
        fp = root / f"freeze_{name.lower()}" / "manifest.json"
    doc = read_json(fp)
    if not doc or not isinstance(doc.get("files"), dict) or not doc["files"]:
        raise FreezeError(f"{fp}: freeze record missing or empty; test scoring refused")
    base = Path(code_base) if code_base else VOCAB_DIR
    bases = [root, base, base.parent.parent]
    frozen_hashes = set()
    for rel, want in sorted(doc["files"].items()):
        cand = [Path(rel)] if Path(rel).is_absolute() else [b / rel for b in bases]
        hit = next((c for c in cand if c.is_file()), None)
        if hit is None:
            raise FreezeError(f"freeze {name}: listed file missing: {rel}")
        got = sha256_file(hit)
        if got != str(want).lower():
            raise FreezeError(f"freeze {name}: sha256 mismatch for {rel}: {got} != {want}")
        frozen_hashes.add(got)
    unfrozen = [p.name for p in code_files() if sha256_file(p) not in frozen_hashes]
    if unfrozen:
        raise FreezeError(f"Engine code files not covered by freeze: {unfrozen}")
    return {"freeze": str(fp), "n_files": len(doc["files"]), "engine_code_frozen": not unfrozen,
            "unfrozen_engine_files": unfrozen, "sha256": sha256_file(fp)}


# --------------------------------------------------------------------------- parameters
DEFAULT_TIERS: list[dict] = [
    # Evaluated top to bottom; the first matching tier wins. Gates are per-architecture booleans
    # (None = unavailable = not satisfied). See verdict.py for the rule grammar.
    {"name": "STRONG YES",
     "rule": {"count_archs": {"min": 2, "gates": [
         "h1_pass", "crit1", "crit2", "crit3", "crit4_available", "crit4", "crit5", "crit6",
         "lb_cstar_gt_sesoi_simultaneous", "matched_support_pass", "family_macro_gain_pos",
         "beats_each_seed"]}}},
    {"name": "YES (weak)", "rule": {"count_archs": {"min": 1, "gates": ["h1_pass"]}}},
    {"name": "INCONCLUSIVE–CEILING",
     "rule": {"count_archs": {"min": 2, "gates": ["probe_below_floor"]}}},
    {"name": "OPERATIONAL NEGATIVE",
     "rule": {"all": [{"study": "power_ok"},
                      {"count_archs": {"min": "all", "gates": ["not_h1_pass", "ub_cstar_lt_sesoi_simultaneous"]}}]},
     "sublabels": [{"name": "WORSE",
                    "rule": {"count_archs": {"min": "all", "gates": ["ub_cstar_lt_minus_sesoi_simultaneous"]}}}]},
    {"name": "INCONCLUSIVE/MIXED", "rule": {"always": True}},
]

_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "anchor_families": [],
    "null_routes": "pretrained",          # 'pretrained' | 'all' | list of route names
    "rsa_bootstrap": None,                # None -> min(bootstrap, 2000) draws of the shared indices
    "rsa_qual_sims": 200,
    "rsa_qual_perms": 999,
    "rsa_qual_classes": 150,
    "rsa_qual_bootstrap": 200,
    "rsa_qual_max_type1": 0.075,
    "rsa_qual_coverage": [0.92, 0.995],
    "power_sims": 400,
    "power_bootstrap": 1000,
    "season_bins": 12,
    "ci_level": 0.95,
    "h1_require_lower_bounds": True,      # PLAN §2.1/TEST-PLAN §3.9: all control lower bounds > 0
    "stratum_coverage_min": 0.6,
    "stratum_purity_min": 0.8,
    "stratum_snow_max": 0.05,
    "stratum_source": "planetary_computer",
    "continent_exclude": ["other"],
    "tiers": None,
    "scaling_seed_tag": 17,
    "text_perm_seed_tag": 23,
    "label_shuffle_seed_tag": 29,
    "gram_chunk_rows": 16384,
    "gzsl": True,
}


def params(cfg: Mapping[str, Any]) -> dict:
    """All analysis parameters (config 'analysis' + labels), with documented defaults for optional keys."""
    a = copy.deepcopy(dict(cfg["analysis"]))
    used = []
    for k, v in _OPTIONAL_DEFAULTS.items():
        if k not in a:
            a[k] = copy.deepcopy(v)
            used.append(k)
    lab = cfg.get("labels", {})
    a["tau"] = float(lab.get("tau", 0.3))
    a["multilabel_min"] = float(lab.get("multilabel_min", 0.05))
    if "exclusion_min" not in a:
        a["exclusion_min"] = a["multilabel_min"]
        used.append("exclusion_min")
    a["footprint_label_column"] = dict(lab.get("footprint_label_column", {"r1": "labels_idx"}))
    a["seed"] = int(cfg.get("seed", 51426))
    if a["rsa_bootstrap"] is None:
        a["rsa_bootstrap"] = int(min(int(a["bootstrap"]), 2000))
    if a["tiers"] is None:
        a["tiers"] = copy.deepcopy(DEFAULT_TIERS)
    lo, hi, step = (float(x) for x in a["gzsl_gamma_grid"])
    n = int(round((hi - lo) / step)) + 1
    a["_gamma_grid"] = [round(lo + i * step, 10) for i in range(n)]
    a["_defaults_used"] = sorted(used)
    return a


def params_fingerprint(p: Mapping[str, Any]) -> str:
    return sha256_json({k: v for k, v in p.items() if k != "_defaults_used"})


def load_config(path: os.PathLike | str) -> dict:
    cfg = read_json(path)
    if cfg is None:
        raise FileNotFoundError(path)
    return cfg


def results_dir(root: os.PathLike | str) -> Path:
    d = Path(root) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def route_dir(root: os.PathLike | str, route: str) -> Path:
    d = results_dir(root) / "fit" / route
    d.mkdir(parents=True, exist_ok=True)
    return d


def iter_chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for s in range(0, n, size):
        yield s, min(n, s + size)
