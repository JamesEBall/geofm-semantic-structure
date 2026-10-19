"""Recompute conditional precision simulations and text-only rank feasibility.

Author: James Edward Ball. These checks do not establish semantic capability.
Run: python scripts/check_supporting.py --work work
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import t
from scipy.optimize import linprog
from study.data import PACKAGE, read, write, digest


def wilson(count, n):
    z = 1.959963984540054
    p = count / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def precision_check(expected):
    bundle = read(PACKAGE / "data/bundle.json")
    lookup = {r["patch_id"]: r for r in bundle["rows"]}
    # Original simulation sorts numeric component IDs before assigning random draws.
    rows = [
        {**lookup[p], "component": int(lookup[p]["component"])}
        for p in expected["joint_patch_ids"]
    ]
    parents = sorted({r["parent_id"] for r in rows})
    components = sorted({r["component"] for r in rows})
    pc = {p: {r["component"] for r in rows if r["parent_id"] == p} for p in parents}
    assert all(len(v) == 1 for v in pc.values())
    pi = np.array([parents.index(r["parent_id"]) for r in rows])
    ci = np.array([components.index(r["component"]) for r in rows])
    weights = np.zeros((len(rows), len(components)))
    for j, r in enumerate(rows):
        npatch = sum(x["parent_id"] == r["parent_id"] for x in rows)
        nparent = sum(r["component"] in v for v in pc.values())
        weights[j, ci[j]] = 1 / (npatch * nparent)
    assert np.allclose(weights.sum(0), 1)
    ks = np.array([len(r["positive"]) for r in rows])
    pop = {}
    for k in set(ks):
        pop[int(k)] = np.sort(
            [
                sum((i + 1) / rank for i, rank in enumerate(ranks)) / k
                for ranks in itertools.combinations(range(1, 9), k)
            ]
        )
    mu = np.array([pop[k].mean() for k in ks])
    low = np.array([pop[k][0] for k in ks])
    N = 4000
    g = len(components)
    critical = t.ppf(0.975, g - 1)
    arrays = {}
    cells = []
    for di, (dep, group) in enumerate(
        [("patch", np.arange(len(rows))), ("parent", pi), ("component", ci)]
    ):
        for si, shift in enumerate([0.0, 0.025]):
            for ei, delta in enumerate([0.0, 0.025, 0.05, 0.1]):
                rng = np.random.default_rng(
                    np.random.SeedSequence([20261028, di, si, ei])
                )
                common = rng.choice([-1.0, 1.0], N) * shift
                target = delta + common
                uniforms = rng.random((N, int(group.max()) + 1, 3))[:, group, :]
                base = np.empty((N, len(rows)))
                alt = np.empty_like(base)
                for j, k in enumerate(ks):
                    values = pop[k]
                    base[:, j] = values[
                        np.floor(uniforms[:, j, 0] * len(values)).astype(int)
                    ]
                    alt[:, j] = values[
                        np.floor(uniforms[:, j, 1] * len(values)).astype(int)
                    ]
                probability = np.abs(target[:, None]) / np.where(
                    target[:, None] >= 0, 1 - mu, mu - low
                )
                assert probability.max() <= 1
                alt = np.where(
                    uniforms[:, :, 2] < probability,
                    np.where(target[:, None] >= 0, 1, low),
                    alt,
                )
                means = (alt - base) @ weights
                estimate = means.mean(1)
                se = means.std(1, ddof=1) / np.sqrt(g)
                valid = se > 0
                lower = estimate - critical * se
                upper = estimate + critical * se
                p = np.ones(N)
                p[valid] = t.sf(estimate[valid] / se[valid], g - 1)
                covered = valid & (lower <= delta) & (upper >= delta)
                key = f"{dep}-shift{shift}-effect{delta}"
                arrays[key + "-components"] = means
                arrays[key + "-shift"] = common
                c = {
                    "key": key,
                    "dependence": dep,
                    "training_shift": shift,
                    "effect": delta,
                    "trials": N,
                    "degenerate_trials": int((~valid).sum()),
                    "bias": float(estimate.mean() - delta),
                    "coverage": float(covered.mean()),
                    "coverage_wilson": wilson(int(covered.sum()), N),
                    "mean_interval_width": float(np.mean(upper - lower)),
                }
                for alpha in [0.05, 0.0125]:
                    yes = p < alpha
                    c[f"rejection_{alpha}"] = float(yes.mean())
                    c[f"rejection_wilson_{alpha}"] = wilson(int(yes.sum()), N)
                c["coverage_screen_pass"] = (
                    c["coverage_wilson"][0] >= 0.90 and c["degenerate_trials"] == 0
                )
                c["null_screen_pass"] = (
                    c["rejection_wilson_0.05"][1] <= 0.075 if delta == 0 else None
                )
                c["practical_effect_screen_pass"] = (
                    c["rejection_wilson_0.0125"][0] >= 0.8 if delta == 0.05 else None
                )
                cells.append(c)
    for actual, old in zip(cells, expected["cells"]):
        for key, value in old.items():
            if isinstance(value, (float, list)):
                np.testing.assert_allclose(
                    actual[key], value, rtol=0, atol=1e-10, err_msg=key
                )
            elif actual[key] != value:
                raise ValueError(f"Precision mismatch: {key}")
    if len(cells) != len(expected["cells"]):
        raise ValueError("Precision inventory mismatch")
    return {"status": "passed", "cells": cells, "trials": N * len(cells)}


def span_check(work, expected):
    roles = expected["roles"]
    classes = [c["id"] for c in read(PACKAGE / "data/text.json")["classes"]]
    indices = [classes.index(c) for c in roles["fit"]]
    p = work / "cache/text.npz"
    asset = next(
        a for a in read(PACKAGE / "data/cache.json")["files"] if a["name"] == "text.npz"
    )
    if digest(p) != asset["sha256"]:
        raise ValueError("Text cache checksum mismatch")
    results = []
    arrays = {}
    with np.load(p, allow_pickle=False) as z:
        for name in ["qwen_instructed", "qwen_plain", "mpnet_plain"]:
            T = z[name]
            F = T[indices]
            u, s, v = np.linalg.svd(F, full_matrices=False)
            rank = int((s > s[0] * 1e-10).sum())
            Q = v[:rank].T
            coords = T @ Q
            projection = coords @ Q.T
            independently = T @ F.T @ np.linalg.pinv(F @ F.T, rcond=1e-12) @ F
            error = float(abs(projection - independently).max())
            assert error < 1e-10
            rows = []
            directions = []
            for i, c in enumerate(classes):
                others = [j for j in range(8) if j != i]
                difference = coords[i] - coords[others]
                A = np.column_stack([-difference, np.ones(7)])
                objective = np.r_[np.zeros(rank), -1.0]
                opt = linprog(
                    objective,
                    A_ub=A,
                    b_ub=np.zeros(7),
                    bounds=[(-1, 1)] * rank + [(None, None)],
                    method="highs",
                )
                assert opt.success
                w = opt.x[:rank]
                margin = float(opt.x[-1])
                assert np.max(A @ opt.x) <= 1e-8 and np.max(abs(w)) <= 1 + 1e-8
                direction = Q @ w
                norm = float(np.linalg.norm(direction))
                assert norm > 0
                direction /= norm
                cosines = T @ direction
                actual = float(np.min(cosines[i] - cosines[others]))
                positive = margin > 1e-8 and actual > 1e-8
                if margin > 1e-8:
                    assert positive
                role = (
                    "fit"
                    if c in roles["fit"]
                    else ("validation" if c == roles["validation"] else "test")
                )
                rows.append(
                    {
                        "class": c,
                        "role": role,
                        "maximum_attainable_cosine": float(np.linalg.norm(coords[i])),
                        "box_margin": margin,
                        "verified_unit_direction_margin": actual,
                        "top1_direction_exists": bool(positive),
                        "solver_status": int(opt.status),
                        "solver_message": opt.message,
                        "constraint_max_residual": float(np.max(A @ opt.x)),
                        "cosine_scores": cosines.tolist(),
                    }
                )
                directions.append(direction)
            arrays[name + "-basis"] = Q
            arrays[name + "-projections"] = projection
            arrays[name + "-directions"] = np.stack(directions)
            results.append(
                {
                    "reference": name,
                    "rank": rank,
                    "singular_values": s.tolist(),
                    "projection_audit_error": error,
                    "classes": rows,
                }
            )
    for actual, old in zip(results, expected["results"]):
        if actual["reference"] != old["reference"] or actual["rank"] != old["rank"]:
            raise ValueError("Span inventory mismatch")
        for row, saved in zip(actual["classes"], old["classes"]):
            for key in ["class", "role", "top1_direction_exists"]:
                if row[key] != saved[key]:
                    raise ValueError(f"Span mismatch: {key}")
            np.testing.assert_allclose(
                row["maximum_attainable_cosine"],
                saved["maximum_attainable_cosine"],
                rtol=0,
                atol=1e-10,
            )
    return {
        "status": "passed",
        "results": results,
        "scope": "Feasibility within the fitted target span; neither learnability nor an image result.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    work = parser.parse_args().work.resolve()
    expected = read(PACKAGE / "data/supporting-expected.json")
    report = {
        "author": "James Edward Ball",
        "status": "passed",
        "precision": precision_check(expected["precision"]),
        "text_span": span_check(work, expected["text_span"]),
    }
    output = work / "outputs/supporting-verification.json"
    write(output, report)
    print(f"Passed 24 precision cells and 24 text-rank feasibility checks: {output}")


if __name__ == "__main__":
    main()
