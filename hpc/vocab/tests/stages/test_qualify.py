"""analysis_qualify: plasmode calibration, Mantel calibration and the pre-registered gate table."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc.core import Checkpoint, atomic_write_json
from megaearth_stats import qualify as Q
from megaearth_stats import rsa as R
from megaearth_stats.io import read_npz
from megaearth_stats.bootstrap import PairBoot, make_draws
import stages.extract  # noqa: F401  (import before the fixture stubs _extract_freeze)
from stages import analysis, qualify
from test_analysis_takeover import world  # noqa: F401  (fixture)


def _prepare(world):
    """Readouts, scaling, RSA and inference on the tiny world with two random seeds."""
    world.cfg['extraction']['routes'][0]['models'] += ['random_clay_101', 'random_clay_202']
    world.cfg['analysis']['architectures'] = {'clay': {'pretrained': 'r1-clay',
                                                       'random': ['r1-random_clay_101', 'r1-random_clay_202']}}
    world.cfg['analysis'].update(power_sims=4, power_bootstrap=20, rsa_qual_sims=2, rsa_qual_perms=9, rsa_qual_classes=6)
    src = world.run / 'extract/r1/random_clay_101/mean.npy'
    dst = world.run / 'extract/r1/random_clay_202/mean.npy'
    dst.parent.mkdir(parents=True, exist_ok=True)
    X = np.load(src)
    rng = np.random.default_rng(7)
    np.save(dst, (X + rng.normal(0, .01, X.shape)).astype('float32'))
    atomic_write_json(dst.with_suffix('.meta.json'), json.loads(src.with_suffix('.meta.json').read_text()))
    routes = analysis.routes(world)
    for name, spec in routes.items():
        for scheme in ('family', 'random_class'):
            for fold in range(3):
                unit = {'id': f'{name}.mean.{scheme}.{fold}', 'route': name, 'spec': spec, 'pooling': 'mean', 'scheme': scheme, 'fold': fold}
                world.stage = 'analysis_readouts'
                analysis.readout_run(world, unit, Checkpoint(world.store, world.stage, unit['id']))
        for fold in range(3):
            unit = {'id': f'{name}.qwen_instructed.{fold}', 'route': name, 'spec': spec, 'ref': 'qwen_instructed', 'pooling': 'mean', 'fold': fold}
            world.stage = 'analysis_scaling'
            analysis.auxiliary_run(world, unit, Checkpoint(world.store, world.stage, unit['id']))
    world.stage = 'analysis_rsa'
    analysis.rsa_run(world, {'id': 'r1-clay.qwen_instructed', 'route': 'r1-clay', 'spec': ['r1-clay'], 'ref': 'qwen_instructed', 'pooling': 'mean'},
                     Checkpoint(world.store, world.stage, 'r1-clay.qwen_instructed'))
    world.stage = 'analysis_inference'
    analysis.inference_run(world, {'id': 'aggregate'}, Checkpoint(world.store, world.stage, 'aggregate'))


def test_qualify_stage_end_to_end(world):
    _prepare(world)
    world.stage = 'analysis_qualify'
    units = qualify.plan(world)
    ids = [u['id'] for u in units]
    assert len([u for u in units if u['kind'] == 'oc']) == len(Q.SCENARIOS) and 'mantel.000' in ids
    for u in units:
        out = qualify.run(world, u, Checkpoint(world.store, world.stage, u['id']))
        assert Path(out['outputs'][0]).exists()
    report = qualify.finalize(world)
    ver = json.loads((world.run / 'analysis_qualify/verdict.json').read_text())
    qual = json.loads((world.run / 'analysis_qualify/qualification.json').read_text())
    assert report['tier'] == ver['label'] and isinstance(ver['label'], str)
    assert set(qual['plasmode']['scenarios']) == set(Q.SCENARIOS)
    assert all(s['replicates'] == 4 for s in qual['plasmode']['scenarios'].values())
    g = ver['arch_gates']['clay']
    # gates with no registered estimator are unavailable, never passed
    for name in ('crit4', 'crit6', 'matched_support_pass'):
        assert g[name] is None and name in ver['unavailable_reasons']['clay']
    assert set(ver['unavailable']['clay']) >= {'crit4', 'crit6', 'matched_support_pass'}
    assert ver['rsa_interval_qualified'] is False
    # the H1 of record is recomputed from the same draws as analysis_inference
    assert ver['real']['clay']['matches_inference'] is True
    assert ver['real']['clay']['crit2']['qwen_instructed/random_class']['available'] is True
    assert isinstance(ver['real']['clay']['crit2_pass'], bool)
    assert ver['real']['clay']['crit3_pass'] is None and 'm=6' in ver['real']['clay']['crit3_reason']
    assert isinstance(ver['real']['clay']['probe_below_floor'], bool)
    assert qual['mantel']['simulations'] == 2 and qual['mantel']['interval_qualified'] is False
    assert ver['test_data_used_for_tuning'] is False
    assert (world.run / 'analysis_qualify/report.md').read_text().startswith('# Qualified verdict')
    # booleans survive serialisation as booleans
    assert all(isinstance(v, (bool, type(None))) for v in g.values())


def test_rsa_statistics_match_production_stage(world):
    world.stage = 'analysis_rsa'
    unit = {'id': 'one', 'route': 'r1-clay', 'spec': ['r1-clay'], 'ref': 'qwen_instructed', 'pooling': 'mean'}
    arrays, meta = read_npz(analysis.rsa_run(world, unit, Checkpoint(world.store, world.stage, 'one'))['outputs'][0])
    st = analysis.checked_study(world)
    p = analysis.settings(world)
    P, app, geo, keep = qualify._mantel_inputs(world, st, p)
    assert np.array_equal(st.original_class_idx[keep], arrays['classes'])
    # rebuild the image prototypes exactly as rsa_run does and compare every statistic
    from megaearth_stats.study import FeatureStore
    eng = analysis.engine(world, unit)
    rows = np.flatnonzero(st.split == 2)
    lab = st.labels_for(eng.fp)
    means = []
    for c in keep:
        cmask = np.zeros(st.K, dtype=bool)
        cmask[c] = True
        rr = rows[lab.any_in(cmask)[rows]]
        means.append(np.mean([eng.feats.rows(rr[st.cell_idx[rr] == g]).mean(0) for g in np.unique(st.cell_idx[rr])], axis=0))
    from scipy.spatial.distance import pdist
    r = R.statistics(np.asarray(means), P, [pdist(app, 'cosine'), pdist(geo, 'cosine')], p['mantel_permutations'], p['seed'])
    assert r['spearman'] == pytest.approx(meta['spearman'])
    assert r['mantel_p'] == pytest.approx(meta['mantel_p'])
    assert r['partial_spearman'] == pytest.approx(meta['partial_spearman'])
    np.testing.assert_allclose(r['null'], arrays['null'])


def test_plasmode_world_preserves_pairs_and_truth():
    rng = np.random.default_rng(3)
    n = 400
    cls = rng.integers(0, 20, n)
    cell = rng.integers(0, 15, n)
    family = cls // 4
    vals = {'pre': rng.normal(0.3, 0.1, n), 'ctl': rng.normal(0.2, 0.1, n)}
    sup = Q.Support(cls, cell, family, vals, np.full(n, 0.1))
    new_cls, new_cell, new_family, ex = Q.resample_world(rng, sup, vals)
    assert len(new_cls) == len(new_cell) == len(new_family) == len(ex['pre']) == len(ex['ctl'])
    # every expanded pair keeps its own value and its class-family relation
    fam_of = {}
    for c, f in zip(new_cls, new_family):
        assert fam_of.setdefault(c, f) == f
    # the production bootstrap accepts the resampled world
    b = PairBoot(new_cls, new_cell, new_family, make_draws(new_cell, new_family, 50, 1))
    assert np.isfinite(b.draws_of(ex['pre'] - ex['ctl'])).all()
    shifted, truth, info = Q.plant(sup, 'pre', ['ctl'], 'null', 0.01)
    assert sup.point(shifted) == pytest.approx(sup.point(vals['ctl']))
    assert truth['ctl'] == pytest.approx(0.0)
    shifted, truth, _ = Q.plant(sup, 'pre', ['ctl'], 'planted_sesoi', 0.01)
    assert truth['ctl'] == pytest.approx(0.01)


def test_calibration_verdict_requires_every_check():
    p = {'holm_alpha': 0.05, 'power_min': 0.8}
    summary = {'null': {'replicates': 100, 'fwer': 0.04,
                        'archs': {'a': {'h1_pass_rate': 0.03, 'iut_reject_rate': 0.04, 'crit1_pass_rate': 0.0, 'cstar_lb_gt_sesoi_rate': 0.0,
                                        'contrasts': {'c': {'truth': 0.0, 'reject_rate': 0.05, 'coverage': 0.95}}}}},
               'exchangeable': {'replicates': 100, 'fwer': 0.0,
                                'archs': {'a': {'h1_pass_rate': 0.0, 'iut_reject_rate': 0.0, 'crit1_pass_rate': 0.0, 'cstar_lb_gt_sesoi_rate': 0.0,
                                                'contrasts': {'s': {'truth': 0.001, 'reject_rate': 0.05, 'coverage': 0.94}}}}},
               'planted_cal2': {'replicates': 100, 'fwer': 1.0,
                                'archs': {'a': {'h1_pass_rate': 0.95, 'iut_reject_rate': 0.95, 'crit1_pass_rate': 0.8, 'cstar_lb_gt_sesoi_rate': 0.9,
                                                'contrasts': {'c': {'truth': 0.1, 'reject_rate': 0.95, 'coverage': 0.95}}}}}}
    v = Q.calibration_verdict(summary, p)
    assert v['power_ok'] is True and not v['missing']
    summary['null']['archs']['a']['contrasts']['c']['reject_rate'] = 0.09
    assert Q.calibration_verdict(summary, p)['power_ok'] is False
    del summary['planted_cal2']
    assert Q.calibration_verdict(summary, p)['power_ok'] is None
