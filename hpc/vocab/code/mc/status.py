"""Read-only status for a recovered or active MegaEarth run."""
from __future__ import annotations

import os
from pathlib import Path

from .core import Store
from .registry import load_stages, topo_order
from .runner import stage_progress
from .work import WorkView


def render_status(cfg: dict, code: Path):
    root = Path(os.environ.get("MEGAEARTH_ROOT") or cfg["root"])
    profile = cfg.get("_profile", "full")
    run = Path(os.environ.get("MEGAEARTH_RUN") or root / "runs" / profile)
    store = Store(run)
    stages = load_stages(cfg)
    view = WorkView(store, stages, cfg)
    missing = [f"mc/{name}.py" for name in ("governor", "worker", "disk")
               if not (Path(code) / "code" / "mc" / f"{name}.py").exists()]
    rows = [{"stage": name, "state": view.state(name), **stage_progress(store, stages[name])}
            for name in topo_order(stages)]
    doc = {"author": "James Edward Ball", "profile": profile, "run": str(run),
           "exists": run.exists(), "pause": view.paused(), "stages": rows,
           "missing_scheduler_modules": missing,
           "pipeline_registered": bool(stages)}
    text = [f"MegaEarth {profile}: {run}"]
    if not stages:
        text.append("No pipeline stages registered. The production pipeline is incomplete.")
    if missing:
        text.append("Missing scheduler modules: " + ", ".join(missing))
    if doc["pause"]:
        text.append("Paused: " + doc["pause"])
    for row in rows:
        text.append(f"{row['stage']}: {row['state']} "
                    f"done={row.get('done', 0)}/{row.get('n', '?')} dead={row.get('dead', 0)}")
    return doc, "\n".join(text)
