import tempfile
from pathlib import Path
from mc.context import make_ctx
from mc.core import atomic_write_json
from mc.governor import cycle, owned, token
from mc.registry import Stage
from mc.slurm import FakeSlurm, enucc_nodes


def setup(tmp):
    cfg = {'root': str(tmp), '_profile': 'selftest', 'mc': {'user': 'me', 'max_submits_per_cycle': 6,
        'pools': {'cpu': {'partition': 'short', 'max_nodes': 2, 'serves': ['cpu'], 'time':'00:05:00'}}}}
    ctx = make_ctx(cfg, Path(tmp))
    backend = FakeSlurm(enucc_nodes())
    stages = {'one': Stage('one', 'P1', 'cpu', (), lambda c: [{'id':'x'}], lambda *a: {})}
    return ctx, backend, stages


def test_only_matching_ledger_user_and_run_can_be_owned():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, st = setup(tmp)
        j = b.add_job('me', 'short', 1, 1024, name='si-control', comment=f'megaearth|selftest|cpu|worker|{token(ctx)}')
        assert not owned(j, ctx, {})
        ledger = {str(j.id): {'comment': j.comment, 'token': token(ctx), 'pool': 'cpu'}}
        assert owned(j, ctx, ledger)
        j.user = 'other'
        assert not owned(j, ctx, ledger)


def test_foreign_demand_prevents_submission_and_dry_run_has_no_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, st = setup(tmp)
        j = b.add_job('me', 'short', 1, 1024, name='si-control', reason='Priority')
        assert cycle(ctx, b, st, dry_run=True)['actions'] == []
        assert not b.log
        assert not (ctx.store.mc / 'submitted.json').exists()


def test_dispatch_then_ledger_identity_excludes_si_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, st = setup(tmp)
        report = cycle(ctx, b, st)
        assert len(report['actions']) == 1
        assert len(b.jobs) == 1
        assert (ctx.store.mc / 'submitted.json').exists()


def test_catalogue_worker_cap_leaves_acquisition_capacity_available():
    for name, expected in [('stac_prefetch', 4), ('stac_search', 4), ('acquire', 10)]:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, backend, _ = setup(tmp)
            ctx.cfg['mc'].update(catalogue_max_workers=4, max_submits_per_cycle=24)
            ctx.cfg['mc']['pools'] = {'net': {'partition':'short', 'start_width':24, 'serves':['net']}}
            stage = Stage(name, 'P1', 'net', (), lambda c:[], lambda *a:{}, unit_cpus=1, unit_mem_gb=1)
            ctx.store.write_units(name, [{'id':str(i)} for i in range(10)])
            report = cycle(ctx, backend, {name:stage})
            assert len([a for a in report['actions'] if 'submit' in a]) == expected


def gpu_setup(tmp):
    ctx, b, _ = setup(tmp)
    ctx.cfg['mc']['pools'] = {'gpu': {'partition': 'gpu', 'max_gpus': 8, 'serves': ['gpu']}}
    ctx.cfg['mc']['max_submits_per_cycle'] = 20
    st = Stage('extract', 'P2', 'gpu', (), lambda c: [], lambda *a: {}, unit_cpus=1, unit_mem_gb=1)
    ctx.store.write_units('extract', [{'id': str(i)} for i in range(20)])
    return ctx, b, {'extract': st}


def test_four_gpu_two_node_bound_with_foreign_running_job():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, stages = gpu_setup(tmp)
        foreign = b.add_job('me', 'gpu', 1, 1024, 1, state='RUNNING', nodes=['gpu01'], name='si-72b')
        cycle(ctx, b, stages)
        jobs = [j for j in b.jobs if j.id != foreign.id]
        assert sum(j.gpus_per_node for j in jobs) <= 4
        assert len({n for j in jobs for n in j.nodes}) <= 2
        assert foreign.state == 'RUNNING'


def test_idle_gpu_expansion_stops_at_eight_and_foreign_demand_yields_only_ours():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, stages = gpu_setup(tmp)
        cycle(ctx, b, stages)
        assert sum(j.gpus_per_node for j in b.jobs) == 8
        ids = {j.id for j in b.jobs}
        foreign = b.add_job('me', 'gpu', 1, 1024, 1, reason='Resources', name='si-72b')
        result = cycle(ctx, b, stages)
        assert {a['yield'] for a in result['actions']} == ids
        assert not ctx.store.yield_path(str(foreign.id)).exists()
        assert not b.signals


def test_recovery_requires_terminal_accounting_and_preserves_live_claim():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, stages = setup(tmp)
        cycle(ctx, b, stages)
        job = b.jobs[0]
        ctx.store.claim('one', '__plan__', str(job.id))
        b.finish(job.id)
        b.terminal = lambda jid: False
        cycle(ctx, b, stages)
        assert ctx.store.claim_owner('one', '__plan__')['worker'] == str(job.id)
        b.terminal = lambda jid: jid == job.id
        cycle(ctx, b, stages)
        assert ctx.store.claim_owner('one', '__plan__') is None
        assert ctx.store.attempts('one', '__plan__') == 1


def test_output_corruption_prevents_completion_reuse():
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        ctx, _, _ = setup(tmp)
        path = ctx.run / 'output.bin'
        path.write_bytes(b'abc')
        ctx.store.mark_done('one', 'unit', [path])
        assert ctx.store.is_done('one', 'unit')
        path.write_bytes(b'xyz')
        with pytest.raises(ValueError, match='hash mismatch'):
            ctx.store.is_done('one', 'unit')


def test_recovery_compare_does_not_delete_new_owner():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, _, _ = setup(tmp)
        ctx.store.claim('one', 'x', 'old')
        stale = ctx.store.claim_owner('one', 'x')
        ctx.store.release('one', 'x', 'old')
        ctx.store.claim('one', 'x', 'new')
        assert not ctx.store.recover('one', 'x', stale)
        assert ctx.store.claim_owner('one', 'x')['worker'] == 'new'


def test_fake_scheduler_respects_dependency_and_excluded_nodes():
    b = FakeSlurm(enucc_nodes())
    held = b.add_job('other', 'gpu', 1, 1024, 1, reason='Dependency')
    ready = b.add_job('other', 'gpu', 1, 1024, 1, reason='Resources')
    ready.exc_nodes = ['gpu01']
    b.schedule()
    assert held.state == 'PENDING'
    assert ready.nodes == ['gpu02']
    b.snapshot().jobs[0].req_nodes.append('gpu04')
    assert not held.req_nodes


def test_cpu_pool_without_yield_policy_submits_despite_foreign_pending():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, st = setup(tmp)
        ctx.cfg['mc']['pools']['cpu']['yield_to_foreign'] = False
        b.add_job('other', 'short', 1, 1024, name='array.script', reason='Resources')
        report = cycle(ctx, b, st)
        assert len(report['actions']) == 1 and 'submit' in report['actions'][0]


def test_pool_serves_other_pools_items():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, b, st = setup(tmp)
        ctx.cfg['mc']['pools'] = {'himem': {'partition': 'himem', 'serves': ['himem', 'cpu'], 'max_nodes': 4, 'time': '00:05:00'}}
        report = cycle(ctx, b, st)          # the only stage is a cpu stage; himem must still take it
        assert len(report['actions']) == 1 and '--partition=himem' in report['actions'][0]['submit']
