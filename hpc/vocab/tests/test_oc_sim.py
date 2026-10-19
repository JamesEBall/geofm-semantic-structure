"""P0 simulation guards and numerical oracles. Author: James Edward Ball.

These checks qualify implementation details, not the scientific operating characteristics.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/p0'))
import oc_sim as o


def test_null_centred_p_exact_oracles():
    draws = np.array([2., 1., 2., 3.])
    assert o.p_greater(draws, 0.) == .25
    assert o.p_less(draws, 3.) == .5
    assert o.p_greater(np.zeros(10), 0.) == 1.
    assert o.p_less(np.zeros(10), 0.) == 1.


@pytest.mark.parametrize('draws', [[], [1.], [np.nan, 2., 3.], [1., np.inf, 2.], [1., 2., np.nan]])
def test_invalid_inference_cannot_pass(draws):
    assert o.p_greater(draws, 0.) == 1.
    assert o.p_less(draws, 0.) == 1.


def test_holm_stepdown_and_nonfinite():
    assert o.holm([.01, .02, .04]).tolist() == [True, True, True]
    assert o.holm([.04, .01, .30]).tolist() == [False, True, False]
    assert o.holm([np.nan, np.inf, .04]).tolist() == [False] * 3


def test_matching_fails_outside_support_in_both_directions():
    controls = np.array([[.3, .2], [.4, .1]])
    assert o.matched_level([.6, .5], controls) is None
    assert o.matched_level([.09, .02], controls) is None
    assert o.matched_level([.5, .4, .3], controls) == 1
    assert o.matched_level([.1], controls) == 0
    assert o.matched_level([.2], [np.nan]) is None


def test_parallel_lines_slope_and_unidentified_slope():
    s = np.array([[[.1, .2, .3], [.2, .4, .6]]])
    z = .7 * s + np.array([[[4.], [-3.]]])
    assert o.fe_slope(z, s, [0, 1])[0] == pytest.approx(.7)
    assert np.isnan(o.fe_slope(z, np.ones_like(s), [0, 1])[0])


def replica(theta=.05, guard=True, detected=True, strong=True, negative=False):
    rec = dict(theta_hat=theta, matched_support=guard, negative=negative,
               gates={g: True for g in o.GATE_ORDER})
    return dict(arch={a: copy.deepcopy(rec) for a in o.ARCHS},
                detected=[detected]*3, strong_core=[strong]*3)


def test_tier_guard_and_effect_labels():
    r = replica()
    assert o.study_tier(r, o.GATE_ORDER)[0] == 'STRONG_YES'
    assert o.study_tier(replica(guard=False), o.GATE_ORDER)[0] == 'INCONCLUSIVE'
    assert o.study_tier(replica(theta=.01, strong=False), o.GATE_ORDER)[0] == 'DETECTED_BELOW_SESOI'
    assert o.study_tier(replica(detected=False, strong=False, negative=True), o.GATE_ORDER)[0] == 'OPERATIONAL_NEGATIVE'
    assert o.study_tier(replica(detected=False, strong=False), o.GATE_ORDER)[0] == 'INCONCLUSIVE'


def test_demotion_never_overrides_failed_false_positive_requirement():
    # Gate search can expose a false-positive failure. It must preserve that failure.
    planted = {'0': [replica()], 'delta': [replica()], '2delta': [replica()]}
    result = o.demotion_search(planted)
    assert result['checks']['pass_strong']
    assert not result['checks']['pass_false_positive']
    assert not result['checks']['pass_all']


def test_replicate_bootstrap_matches_hand_point_and_keeps_classes():
    p = copy.deepcopy(o.BASE)
    p.update(K=30, F=12, cells=30, cells_per_class=[5, 7], largest_family_frac=.2)
    r = o.Replicate(np.random.default_rng(7), p, o.DEFAULT_COMPONENTS, .05, 15)
    # Point estimate: per-class mean of observed cells, then equal class weighting.
    per_class = r.XM.sum(0) / r.M.sum(0)
    assert np.allclose(r.agg()[0], per_class.mean(-1))
    assert np.isfinite(r.agg()).all()
    assert np.all(r.wk > 0)
    assert r.agg().shape == (16, 15, len(p['trunc_drop']))
    family_means = [per_class[..., r.d['fam'] == f].mean(-1) for f in range(p['F'])]
    assert np.allclose(r.agg(family_macro=True)[0], np.mean(family_means, axis=0))


def test_tiny_replicate_is_reproducible():
    p = copy.deepcopy(o.BASE)
    p.update(K=30, F=12, cells=30, cells_per_class=[5, 7], largest_family_frac=.2)
    a = o.run_replicate(np.random.default_rng(19), p, o.DEFAULT_COMPONENTS, .05, 29)
    b = o.run_replicate(np.random.default_rng(19), p, o.DEFAULT_COMPONENTS, .05, 29)
    assert a == b
    assert np.isfinite(a['beta_hat'])
    assert all(0 <= value <= 1 for value in a['p_detect'])


@pytest.mark.parametrize('reps,n_boot', [(0, 10), (1, 0), (1, 1)])
def test_simulation_requires_estimable_inputs(reps, n_boot):
    with pytest.raises(ValueError):
        o.simulate(o.BASE, o.DEFAULT_COMPONENTS, reps, n_boot, 1)


def test_archive_components_verify_score_checksum(tmp_path):
    import json
    root = tmp_path / 'clay'
    root.mkdir()
    (root / 'fold-1-scores.npz').write_bytes(b'corrupt')
    (root / 'fold-1.json').write_text(json.dumps({'scores_sha256': 'incorrect'}))
    with pytest.raises(ValueError, match='checksum'):
        o.archive_components(tmp_path)
