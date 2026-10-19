"""One sequential unit executor per allocation; subprocesses isolate heavy libraries."""
from __future__ import annotations
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
from .context import make_ctx
from .core import atomic_write_json, worker_id
from .registry import load_stages
from .runner import install_stop_handlers, stop_requested
from .work import WorkView, node_tmp_ok, pick


def main(cfg, code, *, pool, cpus, mem_gb, gpus=0):
    if cfg.get("_profile") != "selftest" and (not os.getenv("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
        raise RuntimeError("Workers must run in a compute-node Slurm allocation")
    stages = load_stages(cfg)
    if not stages:
        raise ValueError("No pipeline stages registered")
    ctx = make_ctx(cfg, Path(code), threads=cpus, gpus=gpus)
    store = ctx.store
    install_stop_handlers()
    view = WorkView(store, stages, cfg)
    idle_since = time.monotonic()
    settings = cfg.get("mc", {})
    serves = settings.get("pools", {}).get(pool, {}).get("serves", [pool])
    heartbeat = store.worker_path(worker_id())
    def beat(item=None):
        atomic_write_json(heartbeat, {"worker": worker_id(), "pool": pool, "t": time.time(),
                                     "stage": item.stage if item else None, "unit": item.unit if item else None})
    try:
        while not stop_requested(store):
            beat()
            if not node_tmp_ok(cfg, str(ctx.tmp)):
                store.event("worker_tmp_low")
                return 75
            items, summary = view.all_items(set(serves))
            # Spread claims: with many workers polling, all choosing the single top-ranked unit means one
            # claim wins and the rest recompute. Ranks are equal within a stage, so a random pick from the
            # top window loses nothing and removes the contention.
            window = items[:max(1, int(settings.get("claim_window", 256)))]
            random.shuffle(window)
            item = pick(window, serves, cpus, mem_gb, gpus)
            if item is None:
                if time.monotonic() - idle_since > settings.get("idle_exit_s", 180):
                    return 0
                time.sleep(2)
                continue
            if not store.claim(item.stage, item.unit):
                continue
            idle_since = time.monotonic()
            cmd = [sys.executable, "-m", "mc.cli", "--profile", ctx.profile, "run-unit", item.stage, item.unit,
                   "--threads", str(item.cpus), "--gpus", str(item.gpus)]
            proc = subprocess.Popen(cmd, start_new_session=True)
            stopping_at = None
            try:
                while proc.poll() is None:
                    beat(item)
                    if stop_requested(store) and stopping_at is None:
                        stopping_at = time.monotonic()
                        proc.send_signal(signal.SIGUSR1)
                    if stopping_at is not None and time.monotonic() - stopping_at > settings.get("yield_kill_s", 330):
                        os.killpg(proc.pid, signal.SIGKILL)
                    time.sleep(1)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if store.claim_owner(item.stage, item.unit):
                # The only unit process in this allocation has now exited.
                store.record_failure(item.stage, item.unit, "process_exit", f"exit={proc.returncode}")
                store.release(item.stage, item.unit, worker_id())
            if stopping_at is not None or proc.returncode == 75:
                return 75
        return 75
    finally:
        beat()
