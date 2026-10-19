"""Target and provenance regression tests. Author: James Edward Ball."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.distance import pdist
import preflight as p


def test_brute_force_unequal_blocks():
    rng = np.random.default_rng(4)
    blocks = np.array([0, 0, 0, 1, 2, 2])
    vectors = rng.normal(size=(6, 3, 5)) * 0.1 + 2
    present = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 1]], bool
    )
    sums = np.zeros((3, 3, 5))
    counts = np.zeros((3, 3))
    for g, b in enumerate(blocks):
        sums[b] += vectors[g] * present[g, :, None]
        counts[b] += present[g]
    got = p.corrected_gram(sums, counts)
    for c in range(3):
        for d in range(3):
            pairs = [
                vectors[g, c] @ vectors[h, d]
                for g in range(6)
                for h in range(6)
                if blocks[g] != blocks[h] and present[g, c] and present[h, d]
            ]
            assert got[c, d] == pytest.approx(np.mean(pairs), abs=1e-12)


def test_shared_labels_reuse_all_signal():
    proto = np.array([[1.0, 2.0], [100.0, -30.0], [0.0, 3.0]])
    means, present = p.physical_cell_means(
        [[0, 1], [0, 1, 2]], proto, np.array([2.0, 3.0])
    )
    assert np.array_equal(means[0], means[1])
    assert present.all()
    expected = np.mean([proto[[0, 1]].mean(0) + [2, 3], proto.mean(0) + [2, 3]], axis=0)
    np.testing.assert_allclose(means[0], expected)


def test_heterogeneous_block_counterexample_and_homogeneous_control():
    # One class has two cells in the first block and one in the second.
    counts = np.array([[2.0], [1.0]])
    sums = np.array([[[2.0, 0.0]], [[3.0, 1.0]]])
    corrected = p.corrected_gram(sums, counts)[0, 0]
    original = np.sum((sums.sum(0)[0] / 3) ** 2)
    assert corrected == pytest.approx(3.0)
    assert abs(corrected - original) > 0.1
    common = np.array([2.0, 1.0])
    homogeneous = counts[:, :, None] * common
    assert p.corrected_gram(homogeneous, counts)[0, 0] == pytest.approx(common @ common)


@pytest.mark.parametrize(
    "sums,counts,message",
    [
        ([[[1.0, 0.0]], [[-1.0, 0.0]]], [[1.0], [1.0]], "self-norm"),
        ([[[1.0, 0.0]], [[0.0, 1.0]]], [[1.0], [1.0]], "self-norm"),
        ([[[1.0, 0.0]]], [[1.0]], "denominator"),
        ([[[0.0, 0.0]], [[0.0, 0.0]]], [[0.0], [0.0]], "missing class"),
    ],
)
def test_invalid_geometry_fails(sums, counts, message):
    with pytest.raises(ValueError, match=message):
        p.corrected_gram(np.array(sums), np.array(counts))


def test_leave_block_support_failure():
    counts = np.ones((2, 1))
    sums = np.ones((2, 1, 2))
    p.corrected_gram(sums, counts)
    with pytest.raises(ValueError, match="denominator"):
        p.corrected_gram(sums[1:], counts[1:])


def test_production_tied_rank_svd_parity():
    sys.path.insert(0, str(p.FROZEN))
    from geometry import geometry_values

    rng = np.random.default_rng(8)
    arrays = [rng.normal(size=(8, 4)) for _ in range(5)]
    arrays[3] = np.eye(8)  # exact tied nuisance
    expected = geometry_values(*arrays)
    got = p.statistics(*[pdist(x, "cosine") for x in arrays])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_altered_provenance_rejected(tmp_path):
    (tmp_path / "x").write_text("before")
    config = {"frozen_source_sha256": {"x": p.digest(tmp_path / "x")}}
    p.verify_sources(config, tmp_path)
    (tmp_path / "x").write_text("after")
    with pytest.raises(ValueError, match="provenance"):
        p.verify_sources(config, tmp_path)
