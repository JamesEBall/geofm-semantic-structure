"""Supplementary Mantel null simulations beyond the registered rsa_qual_sims budget (descriptive only).

The registered CAL-5 check uses rsa_qual_sims simulations and cannot separate a type-1 rate of 0.05 from
0.08. This runs additional simulations with the identical generator and statistic so the rate can be
reported with a tighter interval. It changes no registered rule. Author: James Edward Ball.

    PYTHONPATH=<release>/code python validation/mantel_extra.py <release> <start> <count> <out.json>
"""
import json
import sys
from pathlib import Path

release = Path(sys.argv[1]).resolve()
start, count = int(sys.argv[2]), int(sys.argv[3])
out = Path(sys.argv[4])
sys.path.insert(0, str(release / 'code'))
from mc.core import load_config  # noqa: E402
from mc.context import make_ctx  # noqa: E402
from megaearth_stats import qualify as Q  # noqa: E402
from megaearth_stats.io import jsonable  # noqa: E402
from stages import qualify  # noqa: E402
from stages.analysis import checked_study, settings  # noqa: E402

cfg = load_config(release, 'full')
ctx = make_ctx(cfg, release, stage='analysis_qualify')
st = checked_study(ctx)
p = settings(ctx)
P, app, geo, keep = qualify._mantel_inputs(ctx, st, p)
records = [Q.mantel_null_sim(P, app, geo, int(p['rsa_qual_classes']), int(p['rsa_qual_perms']), int(p['seed']), sim)
           for sim in range(start, start + count)]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(jsonable({'start': start, 'count': count, 'records': records}), indent=1))
print(f'wrote {out} ({len(records)} simulations)')
