"""Boundary tests for source data, study splits and numerical conventions."""

import zipfile

import numpy as np
import pytest
from study.data import PACKAGE, consensus, coverage, read, safe_extract

from geofm_jlens.semantic_retrieval_v1 import retrieval_metrics
from geofm_jlens.semantic_ridge_v1 import fit_ridge


def test_expert_disagreement_stays_unknown():
    stack = np.array([[[1, 1, 10, 0]], [[1, 2, 10, 0]], [[2, 3, 0, 0]]], dtype=np.uint8)
    assert consensus(stack).tolist() == [[1, 0, 10, 0]]
    lower, upper = coverage(consensus(stack))
    assert lower[0] == 0.25 and upper[0] == 1 and upper[1] == 0.75


def test_exact_coverage_boundary():
    p = np.array([1] * 3 + [2] * 7, dtype=np.uint8).reshape(2, 5)
    lower, upper = coverage(p)
    assert lower[0] >= 0.3 and not upper[0] < 0.3


def test_zip_traversal_rejected_before_extraction(tmp_path):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError):
        safe_extract(p, tmp_path / "labels")
    assert not (tmp_path / "outside.txt").exists()


def test_frozen_parent_and_validation_separation():
    bundle = read(PACKAGE / "data/bundle.json")
    learn = read(PACKAGE / "data/learning.json")
    rows = {r["patch_id"]: r for r in bundle["rows"]}
    assert len(rows) == 2644
    assert all(r["role"] == "test" for p, r in rows.items() if p in bundle["geometry_patch_ids"])
    vp = {rows[p]["parent_id"] for p in learn["validation_ids"]}
    for selection in learn["selections"]:
        parents = {rows[p]["parent_id"] for p in selection["patch_ids"]}
        assert len(parents) == len(selection["patch_ids"])
        assert not parents & vp
        assert all(rows[p]["role"] in ["fit", "fit_single_expert"] for p in selection["patch_ids"])


def test_ridge_matches_independent_dual_solution():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(12, 17))
    y = rng.normal(size=(12, 3))
    q = rng.normal(size=(4, 17))
    lam = 2.5
    model = fit_ridge(x, y, lam)
    xx = (x - x.mean(0)) / x.std(0)
    qq = (q - x.mean(0)) / x.std(0)
    expected = qq @ xx.T @ np.linalg.solve(
        xx @ xx.T + lam * np.eye(len(x)), y - y.mean(0)
    ) + y.mean(0)
    np.testing.assert_allclose(model.predict(q), expected, atol=1e-12)


def test_exact_ties_use_expected_ap():
    score = np.zeros((1, 2))
    labels = np.array([[True, False]])
    result = retrieval_metrics(score, labels, ks=(1, 2))
    assert result["per_query_ap"][0] == 0.75


def test_corrupt_cached_download_is_rejected(tmp_path):
    from study.data import download

    path = tmp_path / "asset.bin"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt cached asset"):
        download("https://example.invalid/unused", path, "0" * 64)


def test_geographic_components_do_not_cross_roles():
    from collections import defaultdict

    roles = defaultdict(set)
    for row in read(PACKAGE / "data/bundle.json")["rows"]:
        roles[row["component"]].add(row["role"])
    assert all(len(values) == 1 for values in roles.values())
