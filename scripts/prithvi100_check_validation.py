"""Recompute the exploratory Prithvi100 ridge choices using fitting and validation rows only.

Author: James Edward Ball. Run: python scripts/prithvi100_check_validation.py --work work
Requires the three checksum-verified release assets in work/cache. No choices or
study inputs are changed. The output retains every candidate and any mismatch.
"""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from study.data import PACKAGE, read, write, digest
from geofm_jlens.semantic_geometry_hierarchy_v1 import unit
from geofm_jlens.semantic_evaluation_v1 import aggregate
from geofm_jlens.semantic_retrieval_v1 import retrieval_metrics

MULTIPLIERS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


def ridge_grid(x, q, y):
    """Same arithmetic as fit_ridge, with the input SVD reused across penalties."""
    x, q, y = (np.asarray(a, dtype=np.float64) for a in (x, q, y))
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale == 0, 1.0, scale)
    xx, qq = (x - mean) / scale, (q - mean) / scale
    u, s, vt = np.linalg.svd(xx, full_matrices=False)
    ym = y.mean(0)
    for multiplier in MULTIPLIERS:
        lam = len(x) * multiplier
        coef = (vt.T * (s / (s * s + lam))) @ (u.T @ (y - ym))
        prediction = qq @ coef + ym
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite validation prediction")
        yield lam, prediction


def choose(candidates, score, maximise):
    # Exact ties choose the stronger penalty, as in the frozen study.
    return (max if maximise else min)(
        candidates,
        key=lambda r: (
            r[score],
            r["regularisation"] if maximise else -r["regularisation"],
        ),
    )


def check(work):
    cache = work / "cache"
    manifest = read(PACKAGE / "data/cache.json")
    hashes = {}
    for asset in manifest["files"]:
        path = cache / asset["name"]
        actual = digest(path)
        if actual != asset["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path.name}")
        hashes[asset["name"]] = actual
    bundle = read(PACKAGE / "data/bundle.json")
    learning = read(PACKAGE / "data/learning.json")
    calibration = read(PACKAGE / "data/calibration.json")
    rows = bundle["rows"]
    index = {row["patch_id"]: i for i, row in enumerate(rows)}
    extension = json.loads(
        gzip.decompress((PACKAGE / "data/prithvi100/validation.json.gz").read_bytes())
    )
    learning["records"] = extension["records"]
    selected = extension["nuisance"]["selected"]["regularisation"]
    calibration["fits"] = [
        {
            "model": "prithvi100",
            "selected_regularisation": selected,
            "candidates": extension["nuisance"]["candidates"],
        }
    ]
    manifest = read(PACKAGE / "data/prithvi100/assets.json")
    for asset in manifest["files"]:
        assert digest(cache / asset["name"]) == asset["sha256"]
    with np.load(cache / "prithvi100-features.npz", allow_pickle=False) as z:
        features = {k: z[k].copy() for k in z.files if not k.endswith("_global")}
    with np.load(cache / "features.npz", allow_pickle=False) as z:
        labels = z["membership"].copy()
        nuisance = z["nuisance"].copy()
    with np.load(cache / "text.npz", allow_pickle=False) as z:
        references = {n: z[n].copy() for n in ("qwen_instructed", "qwen_plain", "mpnet_plain")}
    vi = [index[p] for p in learning["validation_ids"]]
    parents = [rows[i]["parent_id"] for i in vi]
    components = [rows[i]["component"] for i in vi]
    frozen = {
        (r["intended_sites"], r["seed"], r["reference"], r["feature"], r["control"]): r
        for r in learning["records"]
        if r["control"] in ("paired", "shuffled")
    }
    records, failures = [], []
    for selection in learning["selections"]:
        ti = [index[p] for p in selection["patch_ids"]]
        if set(ti) & set(vi) or {rows[i]["parent_id"] for i in ti} & set(parents):
            raise ValueError("Fitting/validation leakage")
        if any(rows[i]["role"] not in ("fit", "fit_single_expert") for i in ti):
            raise ValueError("Unexpected fitting role")
        seed, budget = selection["seed"], selection["intended_sites"]
        permutation = np.random.default_rng(2026101400 + seed).permutation(len(ti))
        for reference, text in references.items():
            target = unit(labels[ti].astype(float) @ text)
            for name, matrix in features.items():
                for control, y in (
                    ("paired", target),
                    ("shuffled", target[permutation]),
                ):
                    candidates = []
                    for lam, prediction in ridge_grid(matrix[ti], matrix[vi], y):
                        ap = retrieval_metrics(unit(prediction) @ text.T, labels[vi])[
                            "per_query_ap"
                        ]
                        candidates.append(
                            {
                                "regularisation": lam,
                                "component_map": aggregate(ap, parents, components)[
                                    "component_mean"
                                ],
                            }
                        )
                    best = choose(candidates, "component_map", True)
                    key = (budget, seed, reference, name, control)
                    old = frozen[key]
                    error = max(
                        abs(best["component_map"] - old["component_map"]),
                        max(
                            abs(c["component_map"] - o["component_map"])
                            for c, o in zip(candidates, old["candidates"])
                        ),
                    )
                    assert [c["regularisation"] for c in candidates] == [
                        c["regularisation"] for c in old["candidates"]
                    ]
                    passed = (
                        best["regularisation"] == old["selected_regularisation"] and error <= 1e-10
                    )
                    record = {
                        "key": list(key),
                        "candidates": candidates,
                        "selected_regularisation": best["regularisation"],
                        "expected_regularisation": old["selected_regularisation"],
                        "validation_ap_error": error,
                        "passed": passed,
                    }
                    records.append(record)
                    if not passed:
                        failures.append(list(key))
        print(f"Checked budget {budget}, seed {seed}", flush=True)
    if len(records) != len(frozen):
        raise ValueError("Validation inventory mismatch")
    nuisance_records = []
    ti = [index[p] for p in calibration["fit_ids"]]
    vi = [index[p] for p in calibration["validation_ids"]]
    if set(ti) & set(vi):
        raise ValueError("Nuisance fitting/validation leakage")
    parents, components = ([rows[i][key] for i in vi] for key in ("parent_id", "component"))
    for old in calibration["fits"]:
        name = old["model"]
        candidates = []
        for lam, prediction in ridge_grid(nuisance[ti], nuisance[vi], features[name][ti]):
            mse = ((prediction - features[name][vi]) ** 2).mean(1)
            candidates.append(
                {
                    "regularisation": lam,
                    "component_mse": aggregate(mse, parents, components)["component_mean"],
                }
            )
        best = choose(candidates, "component_mse", False)
        error = max(
            abs(c["component_mse"] - o["component_mse"])
            for c, o in zip(candidates, old["candidates"])
        )
        passed = best["regularisation"] == old["selected_regularisation"] and error <= 1e-10
        nuisance_records.append(
            {
                "model": name,
                "candidates": candidates,
                "selected_regularisation": best["regularisation"],
                "expected_regularisation": old["selected_regularisation"],
                "maximum_candidate_mse_error": error,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(["nuisance", name])
    report = {
        "author": "James Edward Ball",
        "status": "passed" if not failures else "failed",
        "asset_sha256": hashes,
        "readout_selections": len(records),
        "readout_candidates": len(records) * len(MULTIPLIERS),
        "maximum_validation_ap_error": max(r["validation_ap_error"] for r in records),
        "records": records,
        "nuisance": nuisance_records,
        "failures": failures,
        "scope": "Recomputed seven-penalty grids on frozen fitting/validation rows. Final evaluation rows and scores are not used for selection. Existing study choices remain unchanged.",
    }
    output = work / "outputs/prithvi100-validation-selection.json"
    write(output, report)
    print(f"{report['status']}: {output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    check(parser.parse_args().work.resolve())
