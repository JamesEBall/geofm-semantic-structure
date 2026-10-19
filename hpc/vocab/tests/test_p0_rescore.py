"""Tests for hpc/vocab/code/p0/rescore_archived.py (TEST-PLAN P0-R01..P0-R12). Author: James Edward Ball."""
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
VOCAB = HERE.parent
REPO = VOCAB.parent.parent
sys.path.insert(0, str(VOCAB / 'code' / 'p0'))
import rescore_archived as r  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workdir():
    path = HERE / 'work-p0-rescore'
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope='module')
def legacy():
    """The frozen analyse.py and controls.py, if their dependencies import."""
    try:
        api = _load(REPO / 'hpc/global/code/analyse.py', 'frozen_analyse')
        sys.path.insert(0, str(REPO / 'hpc/supplement-controls/code'))
        controls = _load(REPO / 'hpc/supplement-controls/code/controls.py', 'frozen_controls')
    except Exception as exc:  # scipy/sklearn missing or files absent
        pytest.skip(f'legacy code not importable: {exc}')
    return api, controls


def test_p0_r01_selftest_passes(workdir):
    checks = r.selftest(workdir=workdir / 'self', log=lambda *_: None)
    assert checks['passed'], checks
    assert checks['parity_checked_arrays'] == 16 * 5 * 21
    assert checks['runtime_s'] < 30


def test_p0_r02_primitives_match_frozen_analyse(legacy):
    api, _ = legacy
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(50, 8))
    text = rng.normal(size=(8, 16))
    assert np.array_equal(r.language_scores(pred, text), api.language_scores(pred, text))
    scores = np.round(rng.normal(size=(50, 8)), 1)  # forces exact ties
    y = rng.integers(0, 8, 50)
    assert np.array_equal(r.reciprocal_rank(y, scores), api.reciprocal_rank(y, scores))
    blocks = rng.integers(0, 7, 50).astype(str)
    assert r.block_average(scores[:, 0], blocks) == api.block_average(scores[:, 0], blocks)
    for n, d in ((120, 10), (20, 40)):  # primal and dual branches
        x = rng.normal(size=(n, d))
        targets = np.eye(8)[rng.integers(0, 8, n)]
        xv, xt = rng.normal(size=(9, d)), rng.normal(size=(11, d))
        ridge = r.EigenRidge(x, targets)
        for alpha, pv, pt in api.ridge_predictions(x, targets, xv, xt):
            w = ridge.weights(alpha)
            assert np.array_equal(ridge.standardise(xv) @ w + ridge.ym, pv)
            assert np.array_equal(ridge.standardise(xt) @ w + ridge.ym, pt)


def test_p0_r03_fold_matches_frozen_controls(legacy):
    api, controls = legacy
    feats, y, splits, blocks, text = r.synthetic_world(seed=3, n_blocks=36)
    for model in ('clay', 'metadata'):
        x = feats[model]
        for held in r.FOLDS[:3]:
            result, cache = controls.evaluate_fold(x, y, splits, blocks, text, api, held)
            mine = r.evaluate_fold(x, y, splits, blocks, text, held)
            parity = r.parity_fold(mine, (dict(result=result), cache), y, blocks, 0.0)
            assert parity['failures'] == [] and parity['checked'] == 21, parity


def test_p0_r04_real_metadata_parity(legacy):
    """Bitwise parity with the archived metadata control (features rebuilt from the manifest)."""
    _, controls = legacy
    manifest = REPO / 'hpc/reference/global-manifest.json.gz'
    archive = REPO / 'hpc/reference-results/global/controls'
    text_path = REPO / 'hpc/global/data/text.npz'
    if not (manifest.exists() and (archive / 'metadata/fold-0.json').exists() and text_path.exists()):
        pytest.skip('archived reference results not present')
    y, splits, blocks, sha = r.load_manifest(manifest)
    assert sha == r.EXPECTED_MANIFEST_SHA
    raw = json.loads(gzip.decompress(manifest.read_bytes()))
    rows = raw if isinstance(raw, list) else raw['rows']
    met, _ = controls.metadata(rows)
    res = r.run({'metadata': met}, y, splits, blocks, r.load_text(text_path), archive=archive, n_boot=0, dims=(),
                models=['metadata'], log=lambda *_: None)
    assert res['parity']['passed'], res['parity']
    assert res['parity']['checked_arrays'] == 105
    assert res['parity']['per_model']['metadata']['max_rr'] <= 1e-6


def test_p0_r05_tampered_archive_fails(workdir):
    feats, y, splits, blocks, text = r.synthetic_world(seed=5, n_blocks=34)
    r.write_archive(workdir, feats, y, splits, blocks, text, ['prithvi'])
    res = r.run({'prithvi': feats['prithvi']}, y, splits, blocks, text, archive=workdir, n_boot=0, dims=(),
                log=lambda *_: None)
    assert res['parity']['passed']
    meta = json.loads((workdir / 'prithvi/fold-1.json').read_text())
    meta['result']['ridge/qwen_plain']['selected'] = 7.0
    (workdir / 'prithvi/fold-1.json').write_text(json.dumps(meta))
    res = r.run({'prithvi': feats['prithvi']}, y, splits, blocks, text, archive=workdir, n_boot=0, dims=(),
                log=lambda *_: None)
    assert not res['parity']['passed']
    assert any('alpha' in f for f in res['parity']['per_model']['prithvi']['failures'])


@pytest.mark.parametrize('k', [2, 3, 6, 8])
def test_p0_r06_constant_scores_give_exact_chance(k):
    rng = np.random.default_rng(k)
    y = rng.integers(0, k, 400)
    blocks = rng.integers(0, 20, 400).astype(str)
    panel = r.Panel(blocks, 0, 1)
    rr = r.reciprocal_rank(y, np.zeros((400, k)))
    s, n = panel.table(rr, np.arange(400), blocks, y)
    assert abs(panel.class_balanced(s, n, range(k))[0] - r.chance_mrr(k)) < 1e-12


def test_p0_r07_hubfree_removes_prior_hub():
    """A prediction equal to the training-target mean is uninformative: hub-free scoring
    returns exact chance, whereas cosine scoring ranks a fixed hub class first."""
    rng = np.random.default_rng(1)
    text = r.unit(rng.normal(size=(8, 12)) + 2.0)  # anisotropic, like the real Gram
    seen, held = [0, 1, 2, 4, 5, 7], [3, 6]
    ym = np.zeros(8)
    ym[seen] = rng.dirichlet(np.ones(6))
    pred = np.tile(ym, (40, 1))
    y = np.array([0, 1] * 20)
    hub = r.reciprocal_rank(y, r.gallery_scores('hubfree', pred, ym, text, held))
    assert np.allclose(hub, 0.75)
    cos = r.reciprocal_rank(y, r.gallery_scores('cosine', pred, ym, text, held))
    assert set(np.round(cos, 6)) <= {0.5, 1.0}
    # k = 2 hub-free scores are antisymmetric: the decision is a single direction t1 - t2.
    p = rng.normal(size=(5, 8))
    sc = r.gallery_scores('hubfree', p, ym, text, held)
    assert np.allclose(sc[:, 0], -sc[:, 1])


def test_p0_r08_nsc_equals_zsl_for_onehot_predictions():
    rng = np.random.default_rng(2)
    text = r.unit(rng.normal(size=(8, 12)))
    seen, held = [0, 2, 3, 4, 6, 7], [1, 5]
    pred = np.eye(8)[rng.choice(seen, 30)]
    ym = np.full(8, 1 / 8)
    assert np.allclose(r.gallery_scores('cosine', pred, ym, text, held), r.nsc_scores('cosine', pred, ym, text, seen, held))


def test_p0_r09_fe_slope_recovers_parallel_lines():
    rng = np.random.default_rng(4)
    bases = np.repeat(np.arange(5), 6)
    s = rng.uniform(0, 1, size=bases.size) + bases * 0.5
    z = 0.3 * s + rng.normal(size=5)[bases] + 1e-4 * rng.normal(size=bases.size)
    beta = r.fe_slope(z[None], s[None], bases)
    assert abs(beta[0] - 0.3) < 1e-2
    # A pooled line is biased by the base intercepts; the FE slope is not.
    assert abs(r.pooled_line(z[None], s[None])[1][0] - 0.3) > abs(beta[0] - 0.3)


def test_p0_r10_identical_routes_contrast_is_zero_in_every_draw():
    rng = np.random.default_rng(6)
    blocks = rng.integers(0, 30, 600).astype(str)
    y = rng.integers(0, 8, 600)
    panel = r.Panel(blocks, 300, 9)
    s, n = panel.table(rng.uniform(size=600), np.arange(600), blocks, y)
    a = panel.class_balanced(s, n, range(8))
    b = panel.class_balanced(s.copy(), n.copy(), range(8))
    assert np.all((a - b)[np.isfinite(a)] == 0)
    assert len(a) == 301 and np.isfinite(a).mean() > 0.9


def test_p0_r11_null_centred_p_convention():
    rng = np.random.default_rng(8)
    boot = rng.normal(0.0, 1.0, 9999)
    null = r.summarise_draws(np.r_[0.0, boot])
    assert 0.4 < null['null_centred_p_gt0'] < 0.6
    strong = r.summarise_draws(np.r_[5.0, 5.0 + boot])
    assert strong['null_centred_p_gt0'] < 1e-3
    assert strong['null_centred_p_gt0'] >= 1 / 10000  # (1 + count) / (1 + B) never returns 0
    assert strong['basic_95'][0] == pytest.approx(10 - strong['percentile_95'][1])


def test_p0_r12_cli_selftest_exit_code():
    proc = subprocess.run([sys.executable, str(VOCAB / 'code/p0/rescore_archived.py'), '--selftest'],
                          cwd=HERE, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)['passed']
    assert not (HERE / 'work-p0-selftest').exists()


@pytest.mark.parametrize('fault', ['missing_array', 'nan_array', 'shape', 'missing_summary', 'nan_summary'])
def test_parity_rejects_incomplete_or_nonfinite_reference(fault):
    feats, y, splits, blocks, text = r.synthetic_world(seed=17, n_blocks=34)
    result = r.evaluate_fold(feats['metadata'], y, splits, blocks, text, r.FOLDS[1])
    summary, arrays = r.legacy_reference_fold(feats['metadata'], y, splits, blocks, text, r.FOLDS[1])
    key, summary_key = 'ridge__qwen_instructed', 'ridge/qwen_instructed'
    if fault == 'missing_array':
        arrays.pop(key)
    elif fault == 'nan_array':
        arrays[key][0] = np.nan
    elif fault == 'shape':
        arrays[key] = arrays[key][:-1]
    elif fault == 'missing_summary':
        summary.pop(summary_key)
    else:
        summary[summary_key]['test_query_ap'] = np.nan
    assert r.parity_fold(result, (dict(result=summary), arrays), y, blocks, 1e-6)['failures']


def test_missing_archive_fold_cannot_pass(workdir):
    feats, y, splits, blocks, text = r.synthetic_world(seed=21, n_blocks=34)
    r.write_archive(workdir, feats, y, splits, blocks, text, ['metadata'])
    (workdir / 'metadata/fold-4.json').unlink()
    result = r.run({'metadata': feats['metadata']}, y, splits, blocks, text, archive=workdir,
                   n_boot=0, dims=(), models=['metadata'], log=lambda *_: None)
    assert not result['parity']['passed']
    assert 'metadata' in result['parity']['models_failed']
    assert any('missing archived' in f for f in result['parity']['per_model']['metadata']['failures'])


def test_requested_missing_model_fails_before_analysis():
    with pytest.raises(ValueError, match='missing'):
        r.run({}, None, None, None, None, models=['clay'])
