"""Archived-data diagnostic; it does not qualify the new-data verdict."""
from __future__ import annotations
import os
import subprocess
import sys
from mc.core import read_json
from mc.registry import Stage, StageFailed


def plan(ctx):
    return [{'id': 'all', 'cpus': 1, 'mem_gb': 24}]


def run(ctx, unit, ckpt):
    import signal
    import time
    out = ctx.stage_dir() / 'results'
    script = ctx.code / 'code/p0/rescore_archived.py'
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    proc = subprocess.Popen([sys.executable, '-u', str(script), '--base', str(ctx.root.parent),
        '--out', str(out), '--bootstrap', '1000', '--seed', str(ctx.cfg['seed'])], env=env)
    try:
        while proc.poll() is None:
            ckpt.maybe_stop()
            time.sleep(1)
        if proc.returncode:
            raise StageFailed(f'Archived rescore failed with exit {proc.returncode}')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
    files = sorted(p for p in out.rglob('*') if p.is_file())
    if not files:
        raise StageFailed('Archived rescore produced no outputs')
    return {'outputs': files, 'metrics': {'diagnostic_only': True}}


STAGES = [Stage('p0_rescore', 'P0', 'cpu', (), plan, run, unit_cpus=1, unit_mem_gb=24, est_unit_min=180)]
