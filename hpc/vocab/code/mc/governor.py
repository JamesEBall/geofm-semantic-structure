"""Conservative Slurm dispatcher with explicit run ownership and terminal recovery."""
from __future__ import annotations
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time

from .context import make_ctx
from .core import atomic_write_json, read_json
from .disk import check as disk_check
from .registry import load_stages
from .runner import install_stop_handlers, stop_requested
from .slurm import Slurm
from .work import WorkView


def token(ctx):
    return hashlib.sha256(str(ctx.run.resolve()).encode()).hexdigest()[:16]


def owned(job, ctx, ledger):
    record = ledger.get(str(job.id))
    return bool(record and job.user == ctx.cfg['mc']['user'] and
                job.comment == record['comment'] and record['token'] == token(ctx) and
                job.comment == f'megaearth|{ctx.profile}|{record.get("pool", "")}|worker|{token(ctx)}')


def cycle(ctx, backend, stages, dry_run=False):
    settings = ctx.cfg['mc']
    path = ctx.store.mc / 'submitted.json'
    ledger = read_json(path, {})
    snap = backend.snapshot()
    ours = [j for j in snap.jobs if owned(j, ctx, ledger)]
    foreign = [j for j in snap.jobs if not owned(j, ctx, ledger)]
    active_ids = {j.id for j in snap.jobs}
    for job in ours:
        if job.state == 'PENDING':
            rec = ledger[str(job.id)]
            node = snap.nodes.get(rec.get('node'))
            if node:
                node.cpus_alloc += rec.get('cpus', job.cpus_per_node)
                node.mem_alloc_mb += rec.get('mem_mb', job.mem_per_node_mb)
                node.gpus_alloc += rec.get('gpus', job.gpus_per_node)
    # A missing queue record alone never releases a live claim.
    if not dry_run:
        for stage in stages:
            for uid in ctx.store.claimed(stage):
                claim = ctx.store.claim_owner(stage, uid)
                wid = str((claim or {}).get('worker', ''))
                if wid.isdigit() and wid in ledger and int(wid) not in active_ids and backend.terminal(int(wid)):
                    if ctx.store.recover(stage, uid, claim):
                        ctx.store.record_failure(stage, uid, 'allocation_terminal', 'Recovered after confirmed Slurm termination')
        for name, stage in stages.items():
            for uid, failures in ctx.store.failures(name).items():
                if uid in ('__plan__', '__finalize__') and len(failures) >= stage.max_attempts:
                    ctx.store.set_stage_status(name, 'failed', {'reason': uid + ' retries exhausted'})
    items, summary = WorkView(ctx.store, stages, ctx.cfg).all_items()
    ignored = set(settings.get('foreign_ignore_reasons', []))
    demanded = {p for j in foreign if j.state == 'PENDING' and j.reason not in ignored for p in j.partitions}
    # Yielding to foreign pending jobs is a per-pool policy. The GPU pool yields so SI work starts at once.
    # CPU, himem and tunnel pools rely on Slurm's own ordering: our jobs carry --nice=1000, so a pending
    # foreign job is scheduled ahead of ours wherever it fits, and backfill never delays it for us.
    yield_pools = {pool for pool, pcfg in settings['pools'].items() if pcfg.get('yield_to_foreign', True)}
    actions = []
    for j in ours:
        pool_of_j = ledger[str(j.id)].get('pool')
        if (pool_of_j in yield_pools and any(p in demanded for p in j.partitions)) or ctx.store.flag('STOP') or ctx.store.flag('PAUSE'):
            actions.append({'yield': j.id})
            if not dry_run:
                yp = ctx.store.yield_path(str(j.id))
                if j.state == 'PENDING':
                    fresh = next((v for v in backend.snapshot().jobs if v.id == j.id), None)
                    if fresh and owned(fresh, ctx, ledger):
                        backend.cancel(j.id)
                elif not yp.exists():
                    atomic_write_json(yp, {'time': time.time(), 'reason': 'Eligible foreign demand'})
                elif time.time() - yp.stat().st_mtime > settings.get('yield_kill_s', 330):
                    # Check fresh identity immediately before cancellation.
                    fresh = next((v for v in backend.snapshot().jobs if v.id == j.id), None)
                    if fresh and owned(fresh, ctx, ledger):
                        backend.cancel(j.id)
    reserved = {pool: sum(1 for j in ours if ledger[str(j.id)]['pool'] == pool) for pool in settings['pools']}
    submitted = 0
    for pool, pcfg in settings['pools'].items():
        partition = pcfg['partition']
        if partition in demanded and pcfg.get('yield_to_foreign', True):
            continue
        # A pool takes every item it declares it serves (himem nodes also run cpu units), not only
        # items tagged with its own name; surplus workers idle-exit after idle_exit_s.
        relevant = [i for i in items if i.pool in pcfg.get('serves', [pool])]
        if not relevant:
            continue
        if pool == 'gpu':
            foreign_gpu = any('gpu' in j.partitions and j.state in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING') for j in foreign)
            cap = min(4 if foreign_gpu else 8, pcfg.get('max_gpus', 8))
            reserved[pool] = sum(j.gpus_per_node * max(1,j.n_nodes) for j in ours if ledger[str(j.id)]['pool'] == pool)
        elif pool == 'net':
            cap = pcfg.get('start_width', 4)
            if all(i.stage in ('stac_prefetch', 'stac_search') for i in relevant):
                cap = min(cap, settings.get('catalogue_max_workers', 4))
        else:
            cap = pcfg.get('max_nodes', 1)
        slots = min(cap - reserved[pool], len(relevant))
        planned_nodes = {n for j in ours if ledger[str(j.id)]['pool'] == pool for n in (j.nodes or j.req_nodes)}
        for item in relevant[:max(0, slots)]:
            if submitted >= settings.get('max_submits_per_cycle', 6) or len(ours) + submitted >= settings.get('max_our_jobs', 64):
                break
            cpus, mem, gpu = item.cpus, math.ceil(item.mem_gb + pcfg.get('mem_margin_gb', 2)), item.gpus
            if pool == 'gpu' and reserved[pool] + item.gpus > cap:
                break
            candidates = [n for n in snap.nodes.values() if partition in n.partitions and n.usable and
                          n.free_cpus >= cpus and n.free_mem_mb >= mem * 1024 and n.free_gpus >= gpu]
            if not candidates:
                break
            # Fill existing run nodes first; Slurm still enforces every requested resource.
            if pool == 'gpu' and foreign_gpu:
                used_nodes = planned_nodes
                if len(used_nodes) >= 2:
                    candidates = [n for n in candidates if n.name in used_nodes]
                if not candidates:
                    break
            candidates.sort(key=lambda n: (not any(n.name in j.nodes for j in ours), -n.free_gpus, n.name))
            node = candidates[0]
            planned_nodes.add(node.name)
            comment = f'megaearth|{ctx.profile}|{pool}|worker|{token(ctx)}'
            logs = ctx.root / 'logs'
            args = [f'--partition={partition}', f'--nodelist={node.name}', f'--cpus-per-task={cpus}',
                    f'--mem={mem}G', f'--time={pcfg.get("time", "04:00:00")}',
                    f'--nice={settings.get("nice",1000)}', f'--comment={comment}',
                    f'--job-name={settings.get("job_prefix","🌍🤖")}{pool}{settings.get("job_suffix","")}',
                    f'--output={logs}/worker-%j.log',
                    '--export=ALL']
            if gpu:
                args.append(f'--gres=gpu:{gpu}')
            args += [str(ctx.code / 'slurm/mc_worker.sbatch'), str(ctx.code), pool, str(cpus), str(mem), str(gpu), ctx.profile]
            actions.append({'submit': args})
            if not dry_run:
                logs.mkdir(parents=True, exist_ok=True)
                job_id = backend.submit(args)
                ledger[str(job_id)] = {'token': token(ctx), 'comment': comment, 'pool': pool, 'time': time.time(),
                                       'node': node.name, 'cpus': cpus, 'mem_mb': mem * 1024, 'gpus': gpu}
                atomic_write_json(path, ledger)
            node.cpus_alloc += cpus; node.mem_alloc_mb += mem * 1024; node.gpus_alloc += gpu
            submitted += 1
            if pool == 'gpu':
                reserved[pool] += gpu
    report = {'time': time.time(), 'stages': summary, 'actions': actions, 'owned_jobs': [j.id for j in ours]}
    if not dry_run:
        atomic_write_json(ctx.store.mc / 'governor.json', report)
    return report


TERMINAL = ('complete', 'failed', 'dep_failed')


def run_state(report, idle_cycles, idle_limit):
    """Decide whether the governor keeps cycling.

    exit 0: every stage complete, or the only non-terminal stages wait on a human gate with nothing running.
    exit 2: nothing can progress and at least one stage failed.
    None: keep going. Idle means no owned jobs and no ready stage for `idle_limit` consecutive cycles.
    """
    states = {n: s.get('state') for n, s in report['stages'].items() if not n.startswith('_')}
    failed = sorted(n for n, s in states.items() if s in ('failed', 'dep_failed'))
    if states and all(s == 'complete' for s in states.values()):
        return {'exit': 0, 'failed': failed, 'idle_cycles': 0}
    busy = bool(report.get('owned_jobs')) or any(s == 'ready' for s in states.values()) or any(
        a for a in report.get('actions', []) if 'submit' in a)
    idle_cycles = 0 if busy else idle_cycles + 1
    if idle_cycles >= idle_limit:
        return {'exit': 2 if failed else 0, 'failed': failed, 'idle_cycles': idle_cycles}
    return {'exit': None, 'failed': failed, 'idle_cycles': idle_cycles}


def main(cfg, code, once=False, dry_run=False):
    ctx = make_ctx(cfg, Path(code))
    stages = load_stages(cfg)
    if not stages:
        raise ValueError('No stages registered')
    if not dry_run and cfg.get('_profile') != 'selftest' and not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Governor must run in Slurm')
    ctx.store.mc.mkdir(parents=True, exist_ok=True)
    with open(ctx.store.mc / 'governor.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend = Slurm(user=cfg['mc']['user'], dry_run=dry_run)
        install_stop_handlers()
        last_disk = 0
        idle_cycles = 0
        reported_failed: list = []
        while not stop_requested(ctx.store):
            if not dry_run and time.time() - last_disk >= cfg['mc']['disk'].get('check_every_s',300):
                if not disk_check(ctx)['ok']:
                    return 2
                last_disk = time.time()
            report = cycle(ctx, backend, stages, dry_run)
            if once or dry_run:
                print(report)
                return 0
            verdict = run_state(report, idle_cycles, cfg['mc'].get('governor_idle_cycles', 15))
            idle_cycles = verdict['idle_cycles']
            if verdict['failed'] and verdict['failed'] != reported_failed:
                # Failed branches are recorded and left alone; independent branches keep running.
                reported_failed = verdict['failed']
                atomic_write_json(ctx.store.mc / 'failed_stages.json',
                                  {'time': time.time(), 'stages': verdict['failed'],
                                   'note': 'inspect mc/failures/<stage>/ then clear the records to retry'})
            if verdict['exit'] is not None:
                return verdict['exit']
            time.sleep(min(30, cfg['mc'].get('poll_s',20)))
    return 75
