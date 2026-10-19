"""Fail-closed disk checks, with explicit synthetic-profile exemptions."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from .core import atomic_write_json


def check(ctx):
    cfg = ctx.cfg.get("mc", {}).get("disk", {})
    if ctx.profile == "selftest":
        return {"ok": True, "synthetic": True}
    root = ctx.root.resolve()
    home = Path.home().resolve()
    reasons = []
    if root == home or home in root.parents:
        reasons.append("Research root is on the home filesystem")
    free = shutil.disk_usage(root).free
    need = float(cfg.get("min_free_tb", 3)) * 1e12
    if free < need:
        reasons.append(f"Scratch free bytes {free} below {int(need)}")
    used = 0
    for parent, _, names in os.walk(ctx.run):
        for name in names:
            try:
                used += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    if used > float(cfg.get("run_budget_tb", 2)) * 1e12:
        reasons.append("Run disk budget exceeded")
    report = {"ok": not reasons, "reasons": reasons, "free_bytes": free, "run_bytes": used}
    atomic_write_json(ctx.store.mc / "disk.json", report)
    ctx.store.set_flag("PAUSE_DISK", bool(reasons), "; ".join(reasons))
    return report
