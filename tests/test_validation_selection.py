"""Check validation-grid algebra and frozen tie rules independently."""

import importlib.util
from pathlib import Path
import numpy as np
from geofm_jlens.semantic_ridge_v1 import fit_ridge

spec = importlib.util.spec_from_file_location(
    "check_validation", Path(__file__).parents[1] / "scripts/check_validation.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_grid_matches_direct_ridge_and_dual_solution():
    rng = np.random.default_rng(33)
    x, q, y = (
        rng.normal(size=(13, 19)),
        rng.normal(size=(5, 19)),
        rng.normal(size=(13, 4)),
    )
    x[:, -1] = q[:, -1] = 2.0
    scale = np.where(x.std(0) == 0, 1.0, x.std(0))
    xx, qq = (x - x.mean(0)) / scale, (q - x.mean(0)) / scale
    grid = list(validation.ridge_grid(x, q, y))
    assert len(grid) == 7
    for lam, prediction in grid:
        direct = fit_ridge(x, y, lam).predict(q)
        dual = qq @ xx.T @ np.linalg.solve(
            xx @ xx.T + lam * np.eye(len(x)), y - y.mean(0)
        ) + y.mean(0)
        np.testing.assert_array_equal(prediction, direct)
        np.testing.assert_allclose(prediction, dual, rtol=1e-10, atol=1e-10)


def test_equal_scores_select_stronger_penalty_in_both_directions():
    candidates = [{"score": 0.4, "regularisation": n} for n in (10, 1, 100)]
    assert validation.choose(candidates, "score", True)["regularisation"] == 100
    assert validation.choose(candidates, "score", False)["regularisation"] == 100
