"""Smoke-run analysis_qualify against a real run on a compute node: plan, one plasmode unit, one Mantel
unit, finalize. Outputs land in the run's analysis_qualify directory and are overwritten by the governed
stage later. Author: James Edward Ball.

    PYTHONPATH=<release>/code python validation/qualify_smoke.py <release> [profile]
"""
import sys
import time
from pathlib import Path

release = Path(sys.argv[1]).resolve()
profile = sys.argv[2] if len(sys.argv) > 2 else 'full'
sys.path.insert(0, str(release / 'code'))
from mc.core import Checkpoint, load_config  # noqa: E402
from mc.context import make_ctx  # noqa: E402
from stages import qualify  # noqa: E402

cfg = load_config(release, profile)
ctx = make_ctx(cfg, release, stage='analysis_qualify')
t0 = time.time()
units = qualify.plan(ctx)
print(f'planned {len(units)} units in {time.time() - t0:.1f}s', flush=True)
for uid in ('oc.null.000', 'oc.planted_cal2.000', 'mantel.000'):
    unit = next(u for u in units if u['id'] == uid)
    t = time.time()
    out = qualify.run(ctx, unit, Checkpoint(ctx.store, 'analysis_qualify', uid))
    print(f'{uid}: {out["metrics"]} in {time.time() - t:.1f}s', flush=True)
t = time.time()
report = qualify.finalize(ctx)
print(f'finalize: {report} in {time.time() - t:.1f}s', flush=True)
print((ctx.stage_dir() / 'report.md').read_text())
