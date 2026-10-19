"""Synthetic stages that exercise Mission Control itself (fan-out, deps, checkpoint/yield, retries,
quarantine, net and gpu pools). Loaded only by the `selftest` profile (cfg.mc.stage_modules).

Knobs (cfg.selftest): n_units, unit_s (seconds of fake work per unit), fail_once (unit ids that fail on the
first attempt), poison (unit ids that always fail), gpu (bool: include the gpu stage).
"""
from __future__ import annotations

import json
import os
import time

from mc.core import atomic_write_json, read_json
from mc.registry import Stage


def _k(ctx) -> dict:
    return ctx.cfg.get("selftest", {})


def plan_fan(ctx) -> list[dict]:
    n = int(_k(ctx).get("n_units", 8))
    return [{"id": f"u{i:04d}", "i": i} for i in range(n)]


def run_fan(ctx, unit, ckpt):
    k = _k(ctx)
    uid = unit["id"]
    marker = ctx.stage_dir() / f".tried-{uid}"
    if uid in k.get("poison", []):
        raise RuntimeError(f"poison unit {uid}")
    if uid in k.get("fail_once", []) and not marker.exists():
        marker.touch()
        raise RuntimeError(f"transient failure {uid}")
    st = ckpt.load("progress", {"step": 0, "acc": 0})
    steps = int(k.get("steps", 10))
    dt = float(k.get("unit_s", 1.0)) / steps
    for s in range(st["step"], steps):
        time.sleep(dt)
        st = {"step": s + 1, "acc": st["acc"] + unit["i"] * (s + 1)}
        if ckpt.due():
            ckpt.save("progress", st)
        ckpt.maybe_stop()
    out = ctx.stage_dir() / f"{uid}.json"
    atomic_write_json(out, {"unit": uid, "acc": st["acc"], "host": os.uname().nodename,
                            "threads": ctx.threads, "resumed": bool(ckpt.load("progress"))})
    return {"outputs": [out], "metrics": {"acc": st["acc"]}}


def fin_fan(ctx):
    tot = sum(read_json(p)["acc"] for p in sorted(ctx.stage_dir().glob("u*.json")))
    exp = sum(i * sum(range(1, int(_k(ctx).get("steps", 10)) + 1)) for i in range(int(_k(ctx).get("n_units", 8)))
              if f"u{i:04d}" not in _k(ctx).get("poison", []))
    out = ctx.stage_dir() / "total.json"
    atomic_write_json(out, {"total": tot, "expected": exp})
    if tot != exp:
        from mc.registry import StageFailed
        raise StageFailed(f"total {tot} != expected {exp}")
    return {"total": tot}


def plan_one(ctx):
    return [{"id": "only"}]


def run_net(ctx, unit, ckpt):
    url = _k(ctx).get("net_url", "https://planetarycomputer.microsoft.com/api/stac/v1")
    r = ctx.requests_session().get(url, timeout=30)
    out = ctx.stage_dir() / "net.json"
    atomic_write_json(out, {"status": r.status_code, "bytes": len(r.content), "proxy": bool(ctx.proxy)})
    return {"outputs": [out], "metrics": {"status": r.status_code}}


def run_gpu(ctx, unit, ckpt):
    import torch
    ok = torch.cuda.is_available()
    info = {"cuda": ok, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if ok:
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            x = x @ x / 64
        torch.cuda.synchronize()
        info.update(name=torch.cuda.get_device_name(0), tflops=20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12)
    out = ctx.stage_dir() / "gpu.json"
    atomic_write_json(out, info)
    return {"outputs": [out], "metrics": {k: v for k, v in info.items() if not isinstance(v, str)}}


def fin_join(ctx):
    a = read_json(ctx.p("selftest_fan", "total.json"))
    return {"fan_total": a["total"], "deps_ok": True}


STAGES = [
    Stage(name="selftest_fan", phase="P1", pool="cpu", deps=(), plan=plan_fan, run=run_fan, finalize=fin_fan,
          unit_cpus=2, unit_mem_gb=1, est_unit_min=0.2, allow_failed_frac=0.25),
    Stage(name="selftest_net", phase="P1", pool="net", deps=(), plan=plan_one, run=run_net,
          unit_cpus=1, unit_mem_gb=1, est_unit_min=0.1),
    Stage(name="selftest_join", phase="P1", pool="cpu", deps=("selftest_fan", "selftest_net"), plan=plan_one,
          run=lambda ctx, u, c: {"outputs": [], "metrics": {}}, finalize=fin_join, unit_mem_gb=1),
    Stage(name="selftest_gpu", phase="P2", pool="gpu", deps=("selftest_join",), plan=plan_one, run=run_gpu,
          unit_cpus=4, unit_mem_gb=8, gate="APPROVE_P2", est_unit_min=0.2),
]
