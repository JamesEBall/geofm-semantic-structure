"""Execute one plan / unit / finalize of a stage, with the commit protocol.

Protocol (crash-safe, idempotent):
  * caller holds the claim (O_EXCL file) before calling execute();
  * the unit writes outputs atomically (mc.core.atomic_path) and returns their paths;
  * mark_done() hashes outputs and writes the done marker = the commit point, then drops the claim;
  * Yielded -> claim released, no failure recorded (checkpoint resumes later);
  * any other exception -> failure record (attempt n), claim released, retried with backoff
    until stage.max_attempts, then the unit is quarantined ("dead").
"""
from __future__ import annotations

import os
import signal
import time
import traceback
from pathlib import Path

from .context import Ctx, make_ctx
from .core import FINALIZE, Checkpoint, Store, Yielded, worker_id
from .registry import Stage, StageFailed, UnitSkip, load_stages

PLAN = "__plan__"
RESERVED = (PLAN, FINALIZE)

_STOP = {"flag": False}


def _on_stop(signum, frame):  # noqa: ARG001
    _STOP["flag"] = True


def install_stop_handlers() -> None:
    for s in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(s, _on_stop)


def stop_requested(store: Store | None = None) -> bool:
    if _STOP["flag"]:
        return True
    if store is not None and (any(store.flag(f) for f in ("STOP", "PAUSE", "PAUSE_DISK")) or store.yield_path(worker_id()).exists()):
        return True
    return False


def dead_units(store: Store, st: Stage) -> set[str]:
    out = set()
    for unit, recs in store.failures(st.name).items():
        if len(recs) >= st.max_attempts or any(r.get("kind") == "skip" for r in recs):
            out.add(unit)
    return out


def stage_progress(store: Store, st: Stage) -> dict:
    units = store.read_units(st.name)
    status = store.stage_status(st.name)
    if units is None:
        return {"planned": False, "status": (status or {}).get("status")}
    ids = {u["id"] for u in units}
    done = store.done(st.name) & ids
    dead = dead_units(store, st) & ids - done
    return {"planned": True, "n": len(ids), "done": len(done), "dead": len(dead),
            "claimed": len(store.claimed(st.name) & ids), "status": (status or {}).get("status")}


def finalizable(store: Store, st: Stage) -> tuple[bool, str]:
    p = stage_progress(store, st)
    if not p.get("planned"):
        return False, "not planned"
    if p["done"] + p["dead"] < p["n"]:
        return False, f"{p['n'] - p['done'] - p['dead']} units outstanding"
    if p["n"] and p["dead"] / p["n"] > st.allow_failed_frac:
        return False, f"FAILED: {p['dead']}/{p['n']} units dead > allow_failed_frac={st.allow_failed_frac}"
    return True, "ok"


def _apply_threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "GDAL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
        os.environ[k] = str(n)
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)


def execute(cfg: dict, code: Path, stage: str, unit_id: str, *, threads: int = 1, gpus: int = 0,
            stages: dict[str, Stage] | None = None, ctx: Ctx | None = None) -> str:
    """Returns one of: done, yielded, failed, skipped, gated."""
    stages = stages or load_stages(cfg)
    st = stages[stage]
    ctx = ctx or make_ctx(cfg, code, stage=stage, unit=unit_id, threads=threads, gpus=gpus)
    ctx.stage, ctx.unit, ctx.threads, ctx.gpus = stage, unit_id, threads, gpus
    store = ctx.store
    # Every entry point applies the same gates; direct run-unit must not bypass them.
    wid = worker_id()
    owner = store.claim_owner(stage, unit_id)
    if owner is not None and owner.get("worker") != wid:
        return "blocked"
    def refuse(outcome):
        store.release(stage, unit_id, wid)
        return outcome
    if any(store.flag(f) for f in ("STOP", "PAUSE", "PAUSE_DISK")):
        return refuse("gated")
    if st.gate and not store.flag(st.gate):
        return refuse("gated")
    if any((store.stage_status(d) or {}).get("status") != "complete" for d in st.deps):
        return refuse("blocked")
    if (store.stage_status(stage) or {}).get("status") == "failed":
        return refuse("blocked")
    if store.attempts(stage, unit_id) >= st.max_attempts:
        return refuse("blocked")
    if owner is None and not store.claim(stage, unit_id, wid):
        return "blocked"
    _apply_threads(threads)
    t0 = time.time()
    release = os.environ.get("MEGAEARTH_RELEASE", "dev")
    store.event("unit_start", stage=stage, unit=unit_id, threads=threads, gpus=gpus, release=release)
    try:
        if unit_id == PLAN:
            if store.read_units(stage) is None:
                units = st.plan(ctx)
                if not isinstance(units, list) or any("id" not in u for u in units):
                    raise TypeError(f"{stage}.plan must return list of dicts with 'id'")
                store.write_units(stage, units, meta={"release": release, "cfg": cfg.get("_fingerprint")})
            store.release(stage, PLAN)
            outcome, metrics = "done", {"n_units": len(store.read_units(stage) or [])}
        elif unit_id == FINALIZE:
            ok, why = finalizable(store, st)
            if not ok:
                if why.startswith("FAILED"):
                    store.set_stage_status(stage, "failed", {"reason": why})
                    outcome, metrics = "failed", {"reason": why}
                else:
                    outcome, metrics = "skipped", {"reason": why}
                store.release(stage, FINALIZE)
            else:
                try:
                    store.verify_done(stage)   # full hash check of every unit output, once, here
                    report = (st.finalize(ctx) if st.finalize else None) or {}
                    report.setdefault("progress", stage_progress(store, st))
                    report["release"] = release
                    store.set_stage_status(stage, "complete", report)
                    outcome, metrics = "done", {}
                except StageFailed as e:
                    store.set_stage_status(stage, "failed", {"reason": str(e), "release": release})
                    outcome, metrics = "failed", {"reason": str(e)}
                store.release(stage, FINALIZE)
        else:
            units = {u["id"]: u for u in (store.read_units(stage) or [])}
            if unit_id not in units:
                raise KeyError(f"{stage}: unknown unit {unit_id}")
            if store.is_done(stage, unit_id):
                store.release(stage, unit_id)
                return "done"
            ckpt = Checkpoint(store, stage, unit_id, stop_flag=lambda: stop_requested(store),
                              every_s=float(cfg.get("mc", {}).get("ckpt_every_s", 60)))
            res = st.run(ctx, units[unit_id], ckpt) or {}
            outputs = res.get("outputs", [])
            metrics = dict(res.get("metrics", {}))
            metrics["_s"] = round(time.time() - t0, 2)
            metrics["_release"] = release
            store.mark_done(stage, unit_id, outputs, metrics, hash_outputs=res.get("hash", True))
            ckpt.clear()
            outcome = "done"
    except Yielded:
        store.release(stage, unit_id)
        store.event("unit_yield", stage=stage, unit=unit_id, s=round(time.time() - t0, 1))
        return "yielded"
    except UnitSkip as e:
        store.record_failure(stage, unit_id, "skip", str(e))
        store.release(stage, unit_id)
        store.event("unit_skip", stage=stage, unit=unit_id, reason=str(e)[:500])
        return "skipped"
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        kind = "oom" if isinstance(e, MemoryError) else type(e).__name__
        n = store.record_failure(stage, unit_id, kind, tb)
        store.release(stage, unit_id)
        store.event("unit_fail", stage=stage, unit=unit_id, attempt=n, err=str(e)[:500])
        ctx.log(f"FAILED attempt {n}: {e}\n{tb}")
        return "failed"
    store.event("unit_end", stage=stage, unit=unit_id, outcome=outcome, s=round(time.time() - t0, 1),
                **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
    return outcome


def local_run(cfg: dict, code: Path, only: list[str] | None = None, threads: int = 2,
              approve: bool = False, max_rounds: int = 1000) -> dict[str, dict]:
    """Single-process executor for development/tests: plan -> units -> finalize in DAG order.

    Gates are mandatory. Explicit automatic approval is restricted to the selftest profile.
    """
    from .registry import topo_order
    if approve and cfg.get("_profile") != "selftest":
        raise ValueError("Automatic approval is restricted to the selftest profile")
    stages = load_stages(cfg)
    if not stages:
        raise ValueError("No stages are implemented for this profile")
    if only and set(only) - set(stages):
        raise ValueError(f"Unknown stages: {sorted(set(only) - set(stages))}")
    order = [s for s in topo_order(stages) if not only or s in only]
    ctx = make_ctx(cfg, code, threads=threads)
    store = ctx.store
    for name in order:
        st = stages[name]
        if (store.stage_status(name) or {}).get("status") == "complete":
            continue
        for d in st.deps:
            if (store.stage_status(d) or {}).get("status") != "complete":
                raise RuntimeError(f"{name}: dependency {d} not complete")
        if st.gate and approve:
            store.set_flag(st.gate, True, note="Synthetic selftest approval")
        if st.gate and not store.flag(st.gate):
            raise RuntimeError(f"{name}: gated on {st.gate}")
        for _ in range(max_rounds):
            if store.read_units(name) is None:
                if store.attempts(name, PLAN) >= st.max_attempts:
                    store.set_stage_status(name, "failed", {"reason": "Planning retries exhausted"})
                    break
                if not store.claim(name, PLAN):
                    raise RuntimeError(f"{name}: planning claimed by another worker")
                outcome = execute(cfg, code, name, PLAN, threads=threads, stages=stages, ctx=ctx)
                if outcome in ("blocked", "gated", "yielded"):
                    raise RuntimeError(f"{name}: {outcome}")
                continue
            dead = dead_units(store, st)
            todo = [u["id"] for u in store.read_units(name)
                    if not store.is_done(name, u["id"]) and u["id"] not in dead]
            if not todo:
                break
            for uid in todo:
                if store.claim(name, uid):
                    outcome = execute(cfg, code, name, uid, threads=threads, stages=stages, ctx=ctx)
                    if outcome in ("blocked", "gated", "yielded"):
                        raise RuntimeError(f"{name}/{uid}: {outcome}")
        if (store.stage_status(name) or {}).get("status") == "failed":
            continue
        if not store.claim(name, FINALIZE):
            raise RuntimeError(f"{name}: finalization claimed by another worker")
        execute(cfg, code, name, FINALIZE, threads=threads, stages=stages, ctx=ctx)
    return {n: stage_progress(store, stages[n]) for n in order}
