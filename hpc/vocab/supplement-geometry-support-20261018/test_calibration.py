"""Scientific estimator and sampling invariants. Author: James Edward Ball."""

import copy
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
import calibration as c
import production_rsa as production


def fixture_design():
    cells = []
    k = 6
    for si, split in enumerate(c.SPLITS):
        for g in range(12):
            labs = [list(range(k)), [g % k, (g + 1) % k]]
            counts = {j: 1 + int(j in labs[1]) for j in range(k)}
            cells.append(
                dict(
                    id=f"{split}_{g}",
                    split=split,
                    geo=[g // 3, si],
                    block=[g // 2, si],
                    counts=sorted(counts.items()),
                    window_labels=labs,
                    windows=2,
                )
            )
    return c.prepare_design(
        dict(
            cells=cells,
            classes=[dict(class_idx=i, family_id=i // 2) for i in range(k)],
            admission=dict(train=10, val=5, test=10, test_cells=5, cells=8),
        )
    )


@pytest.fixture
def tiny(monkeypatch):
    ds = fixture_design()
    monkeypatch.setattr(c, "design", lambda: ds)
    cfg = copy.deepcopy(c.read(c.HERE / "CONFIG.json"))
    cfg["dimension"] = 8
    cfg["bootstrap"] = 19
    cfg["bootstrap_batch"] = 5
    return cfg


def test_production_geometry_parity():
    rng = np.random.default_rng(11)
    x = [rng.normal(size=(8, 5)) for _ in range(4)]
    geo = rng.uniform(size=(8, 6))
    got = c.geometry_values(*x[:3], geo, x[3])
    nu = [pdist(x[2], "cosine"), pdist(geo, "cosine")]
    a = production.statistics(x[0], x[3], nu, 0, 0)
    b = production.statistics(x[1], x[3], nu, 0, 0)
    np.testing.assert_allclose(
        got,
        [
            a["spearman"],
            b["spearman"],
            a["partial_spearman"],
            b["partial_spearman"],
            b["spearman"] - a["spearman"],
            b["partial_spearman"] - a["partial_spearman"],
        ],
        atol=1e-12,
    )


def test_known_zero_and_identical_geometries(tiny):
    w = c.population(tiny, tiny["scenarios"][2])
    np.testing.assert_array_equal(w["truth"][-2:], [0, 0])
    rng = np.random.default_rng(1)
    t = rng.normal(size=(8, 5))
    a = rng.normal(size=(8, 5))
    g = rng.uniform(size=(8, 5))
    np.testing.assert_allclose(
        c.geometry_values(t, t, a, g, t), [1, 1, 1, 1, 0, 0], atol=1e-12
    )


def test_analytic_template_means_and_window_histograms(tiny):
    ds = c.design()
    sc = tiny["scenarios"][3]
    w = c.population(tiny, sc)
    ix = ds["split"] == 2
    cnt = ds["count"][ix].toarray()
    L = w["L"][ds["types"][ix]]
    loc = np.array([L[cnt[:, j] > 0].mean(axis=0) for j in range(ds["k"])])
    base = ds["label_mixture"] @ (w["a"] + 0.7 * w["v"]) + 0.6 * loc
    expected = np.zeros_like(w["geo"])
    for row, x in enumerate(ds["cells"]):
        for j, n in x["counts"]:
            expected[j, ds["types"][row]] += n
    np.testing.assert_array_equal(expected, w["geo"])
    np.testing.assert_allclose(
        c.geometry_values(
            base,
            base + sc["beta"] * (ds["label_mixture"] @ w["s"]),
            ds["label_mixture"] @ w["a"] + 0.3 * loc,
            expected,
            w["text"],
        ),
        w["truth"],
    )


def test_shared_window_noise_covariance(tiny):
    s = c.sample(tiny, tiny["scenarios"][10], "development", 0, 10)
    m = s["window_averager"]
    cov = (m @ m.T).toarray()
    rows = s["evaluation_edge_rows"]
    cols = s["evaluation_edge_classes"]
    same = (rows[:, None] == rows[None, :]) & (cols[:, None] != cols[None, :])
    assert np.any(cov[same] > 0)
    assert np.all(cov[rows[:, None] != rows[None, :]] == 0)
    np.testing.assert_allclose(np.asarray(m.sum(axis=1)).ravel(), 1)


def test_equal_cell_means_and_shared_weighted_geo():
    s = {
        "g": 3,
        "k": 2,
        "d": 1,
        "h": 2,
        "features": csr_matrix([[0] * 6, [1, 2, 3, 4, 5, 6], [9, 10, 11, 12, 13, 14]]),
        "present": csr_matrix([[0, 0], [1, 1], [1, 1]]),
        "geo": csr_matrix([[4, 0, 5, 0], [1, 0, 1, 0], [0, 9, 0, 8]]),
    }
    b, se, a, g, v = c.rebuild(s, [1, 2, 1])
    np.testing.assert_allclose(b[0, :, 0], [11 / 3, 20 / 3])
    np.testing.assert_array_equal(g[0], [[6, 9], [7, 8]])
    assert v[0]


def test_positive_weights_preserve_support_and_split_sizes(tiny):
    s = c.sample(tiny, tiny["scenarios"][2], "development", 1, 2)
    for kind in ["cell", "block"]:
        w = c.draw_weights(s, kind, 30, np.random.default_rng(3))
        assert (w > 0).all()
        assert c.rebuild(s, w)[-1].all()
        for split in range(3):
            np.testing.assert_allclose(w[:, s["split"] == split].mean(1), 1)


def test_block_coupling_survives_split_normalisation():
    s = {
        "g": 6,
        "split": np.array([0, 0, 1, 1, 2, 2]),
        "block": np.array([1, 2, 1, 2, 1, 2]),
    }
    w = c.draw_weights(s, "block", 20, np.random.default_rng(4))
    np.testing.assert_allclose(w[:, 0] / w[:, 1], w[:, 2] / w[:, 3])
    np.testing.assert_allclose(w[:, 0] / w[:, 1], w[:, 4] / w[:, 5])


def test_missing_initial_class_is_explicit_failure(tiny, monkeypatch):
    original = c.sample

    def broken(*args):
        s = original(*args)
        s["support"][2, 0] = 0
        return s

    monkeypatch.setattr(c, "sample", broken)
    r = c.one_replicate(tiny, 2, "development", 0)
    assert r["failure"] == "initial_sample_missing_class" and not r["methods"]


def test_basic_interval_inverts_error_and_bounds():
    draws = np.linspace(0.2, 0.6, 1001)
    r = c.interval_result(0.7, draws, 0.9, "basic", "ordinary_baseline")
    np.testing.assert_allclose([r["raw_lower"], r["raw_upper"]], [0.81, 1.19])
    assert r["upper"] == 1 and r["covers"]
    p = c.interval_result(0.7, draws, 0.9, "percentile", "ordinary_baseline")
    assert not p["covers"]
    e = c.interval_result(
        1.0, np.linspace(-1.0, -0.5, 50), 0.8, "basic", "ordinary_baseline"
    )
    assert not e["available"] and not e["covers"]


def test_seed_namespaces_and_determinism(tiny):
    a = c.sample(tiny, tiny["scenarios"][2], "development", 7, 2)
    b = c.sample(tiny, tiny["scenarios"][2], "development", 7, 2)
    z = c.sample(tiny, tiny["scenarios"][2], "confirmation", 7, 2)
    np.testing.assert_array_equal(a["features"].toarray(), b["features"].toarray())
    assert not np.array_equal(a["features"].toarray(), z["features"].toarray())


def test_truth_ties_and_constant_nuisance_are_finite(tiny):
    sc = tiny["scenarios"][2]
    w = c.population(tiny, sc)
    assert np.isfinite(w["truth"]).all()
    x = np.ones((8, 4))
    rng = np.random.default_rng(2)
    im = rng.normal(size=(8, 6))
    txt = rng.normal(size=(8, 6))
    got = c.geometry_values(im, im, x, x, txt)
    np.testing.assert_allclose(got[0], got[2], atol=1e-12)


def test_end_to_end_and_failure_denominators(tiny):
    r = c.one_replicate(tiny, 2, "development", 0)
    assert len(r["methods"]) == 4 and r["bootstrap"] == 19
    cfg = copy.deepcopy(tiny)
    cfg["development_replicates"] = 2
    cfg["scenarios"] = [tiny["scenarios"][2]]
    fail = copy.deepcopy(r)
    fail.update(
        replicate=1,
        methods={},
        failure="initial_sample_missing_class",
        all_admitted=False,
    )
    z = c.summarise([r, fail], cfg, "development")
    v = z["scenarios"][r["scenario"]]["methods"]["cell_basic"]["ordinary_baseline"]
    assert v["availability"]["rate"] == 0.5 and v["coverage"]["n"] == 2
    assert v["coverage_conditional_admission"]["n"] == 1
    with pytest.raises(ValueError):
        c.summarise([r, r], cfg, "development")
    with pytest.raises(ValueError):
        c.summarise([r], cfg, "development")
    bad = copy.deepcopy(fail)
    bad["CONFIG.json_sha256"] = "x"
    with pytest.raises(ValueError):
        c.summarise([r, bad], cfg, "development")


def test_effective_support_and_no_drop_for_rare_class(tiny):
    s = c.sample(tiny, tiny["scenarios"][2], "development", 3, 2)
    p = s["present"].copy().tolil()
    p[:, 0] = 0
    p[np.flatnonzero(s["split"] == 2)[0], 0] = 1
    s["present"] = p.tocsr()
    w = c.draw_weights(s, "cell", 20, np.random.default_rng(5))
    den = np.asarray(w @ s["present"])
    sq = np.asarray(w * w @ s["present"])
    np.testing.assert_allclose((den * den / sq)[:, 0], 1)
    assert c.rebuild(s, w)[-1].all()


def test_frozen_export_support_and_schema():
    raw = c.read(c.HERE / "RETAINED-DESIGN.json")
    ds = c.prepare_design(raw)
    assert ds["k"] == 188 and len(ds["cells"]) == 9394
    test = ds["count"][ds["split"] == 2]
    n = np.asarray((test > 0).sum(0)).ravel()
    assert (n.min(), np.median(n), n.max()) == (5, 22, 311)
    assert (n <= 10).sum() == 25
    assert int(test.sum()) == 8755
    for x in ds["cells"]:
        counts = {
            j: sum(j in labs for labs in x["window_labels"]) for j, n in x["counts"]
        }
        assert counts == dict(x["counts"])


def test_independent_block_ess_aggregates_cell_masses():
    s = {
        "g": 4,
        "split": np.array([0, 1, 2, 2]),
        "block": np.array([0, 0, 1, 1]),
        "present": csr_matrix([[0], [0], [1], [1]]),
    }
    w = np.ones((3, 4))
    cell, block = c.effective_support(s, w, "block")
    np.testing.assert_array_equal(cell, np.full((3, 1), 2.0))
    np.testing.assert_array_equal(block, np.ones((3, 1)))


def test_complete_window_features_reused_for_every_label(tiny):
    s = c.sample(tiny, tiny["scenarios"][10], "development", 1, 10)
    direct = s["window_averager"] @ s["window_features"]
    dense = s["features"].toarray().reshape(s["g"], s["k"], 3 * s["d"])
    np.testing.assert_allclose(
        dense[s["evaluation_edge_rows"], s["evaluation_edge_classes"]], direct
    )
    for edge, row in enumerate(s["evaluation_edge_rows"]):
        windows = s["window_averager"].getrow(edge).indices
        np.testing.assert_allclose(
            direct[edge], s["window_features"][windows].mean(axis=0)
        )


def test_population_mixture_equals_direct_window_then_cell_average(tiny):
    ds = c.design()
    rng = np.random.default_rng(41)
    latent = rng.normal(size=(ds["k"], 5))
    direct = []
    for target in range(ds["k"]):
        cell_means = []
        for x in ds["cells"]:
            if x["split"] != "test":
                continue
            window_means = [
                latent[labs].mean(0) for labs in x["window_labels"] if target in labs
            ]
            if window_means:
                cell_means.append(np.mean(window_means, axis=0))
        direct.append(np.mean(cell_means, axis=0))
    np.testing.assert_allclose(ds["label_mixture"] @ latent, direct)
    np.testing.assert_allclose(ds["label_mixture"].sum(1), 1)


def test_identical_window_support_gives_identical_complete_class_means(
    tiny, monkeypatch
):
    raw = copy.deepcopy(c.design()["raw"])
    for cell in raw["cells"]:
        labs = [
            sorted(set(x) | {0, 1}) if (0 in x or 1 in x) else x
            for x in cell["window_labels"]
        ]
        cell["window_labels"] = labs
        cell["counts"] = sorted(
            (j, sum(j in x for x in labs))
            for j in range(len(raw["classes"]))
            if any(j in x for x in labs)
        )
    ds = c.prepare_design(raw)
    monkeypatch.setattr(c, "design", lambda: ds)
    s = c.sample(tiny, tiny["scenarios"][11], "development", 1, 11)
    v = c.rebuild(s, np.ones(s["g"]))
    for route in v[:3]:
        np.testing.assert_array_equal(route[0, 0], route[0, 1])
    np.testing.assert_array_equal(ds["label_mixture"][0], ds["label_mixture"][1])
    assert not np.array_equal(s["world"]["a"][0], s["world"]["a"][1])


def test_duplicate_vectors_preserve_production_distance_ties():
    from scipy.stats import rankdata
    from geometry import distance_ranks

    rng = np.random.default_rng(150)
    x = rng.normal(size=(10, 7))
    x[1] = x[0]
    x[3] = x[2]
    x[5] = x[4]
    text = rng.normal(size=(10, 7))
    app = rng.normal(size=(10, 7))
    geo = rng.uniform(size=(10, 4))
    np.testing.assert_array_equal(distance_ranks(x), rankdata(pdist(x, "cosine")))
    expected = production.statistics(
        x, text, [pdist(app, "cosine"), pdist(geo, "cosine")], 0, 0
    )
    got = c.geometry_values(x, x, app, geo, text)
    np.testing.assert_allclose(
        got[:4],
        [
            expected["spearman"],
            expected["spearman"],
            expected["partial_spearman"],
            expected["partial_spearman"],
        ],
        atol=1e-12,
        rtol=0,
    )
