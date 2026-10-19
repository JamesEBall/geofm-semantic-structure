"""Fixed main transfer evaluation with saved validation choices. Author: James Edward Ball."""

import os

for k in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[k] = "1"
import hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from geofm_jlens.semantic_ridge_v1 import fit_ridge
from geofm_jlens.semantic_geometry_hierarchy_v1 import unit
from geofm_jlens.semantic_evaluation_v1 import (
    aggregate,
    summarise_retrieval,
    candidate_paired_interval,
)
from geofm_jlens.semantic_primary_family_v1 import primary_family
from geofm_jlens.semantic_retrieval_v1 import retrieval_metrics

R = Path(os.environ["GEOFM_RUN_DIR"]).resolve()
B = R / "results/semantic-structure"
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def nearest_rows(q, x):
    result = []
    for start in range(0, len(q), 8):
        result.extend(
            np.argmin(
                ((q[start : start + 8, None, :] - x[None, :, :]) ** 2).sum(2), axis=1
            ).tolist()
        )
    return np.array(result, int)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def main():
    start = time.monotonic()
    paths = [
        B / "dynamicworld-main-inputs-01/report.json",
        B / "dynamicworld-main-inputs-01/audit.json",
        B / "dynamicworld-learning-readout-01/report.json",
        B / "dynamicworld-learning-readout-01/audit.json",
        B / "dynamicworld-text-reference-01/report.json",
        B / "dynamicworld-evaluation-admission-01/report.json",
        B / "dynamicworld-evaluation-admission-01/audit.json",
        B / "dynamicworld-evaluation-precision-01/report.json",
        B / "dynamicworld-evaluation-precision-01/audit.json",
    ]
    bundle, ba, learning, la, text, admission, aa, precision, pa = [read(p) for p in paths]
    assert all(r["status"] == "passed" for r in [ba, la, aa, pa])
    hashes = {str(p.relative_to(R)): sha(p) for p in paths}
    for r in [bundle, text]:
        assert sha(R / r["arrays_path"]) == r["arrays_sha256"]
        hashes[r["arrays_path"]] = r["arrays_sha256"]
    rows = bundle["rows"]
    ids = [r["patch_id"] for r in rows]
    index = {p: i for i, p in enumerate(ids)}
    classes = bundle["classes"]
    assert classes == [r["id"] for r in text["classes"]] == learning["classes"]
    names = [n for n in bundle["feature_names"] if not n.endswith("_global")]
    assert len(names) == 13 and {"clay", "prithvi", "appearance", "metadata"} <= set(names)
    with np.load(R / bundle["arrays_path"], allow_pickle=False) as z:
        X = {n: z["feature__" + n].copy() for n in names}
        labels = z["membership"].copy()
        clip = {n: z["text__" + n].copy() for n in ["remoteclip_gelu", "remoteclip_quickgelu"]}
    with np.load(R / text["arrays_path"], allow_pickle=False) as z:
        refs = {k: z[k].copy() for k in ["qwen_instructed", "qwen_plain", "mpnet_plain"]}
    val = learning["validation_ids"]
    evaluation = bundle["evaluation_patch_ids"]
    assert not set(val) & set(evaluation)
    query = val + evaluation
    qi = [index[p] for p in query]
    qindex = {p: i for i, p in enumerate(query)}
    vi = [index[p] for p in val]
    vp = [rows[i]["parent_id"] for i in vi]
    vc = [rows[i]["component"] for i in vi]
    known = {"Built_Area", "Flooded_Vegetation", "Scrub", "Trees", "Water"}
    held = {"Bare_Ground", "Grass"}
    cells = {
        f"{kind}_{geo}": sorted(
            p
            for p in evaluation
            if rows[index[p]]["role"] == geo and set(rows[index[p]]["positive"]) & cats
        )
        for kind, cats in [("known", known), ("heldout", held)]
        for geo in ["fit", "test"]
    }
    primary_ids = cells["heldout_test"]
    assert set(primary_ids) == set(precision["joint_patch_ids"])
    selections = learning["selections"]
    assert len(selections) == 20 and {(s["seed"], s["intended_sites"]) for s in selections} == {
        (seed, n) for seed in range(5) for n in [32, 78, 128, 512]
    }
    previous = {
        (r["intended_sites"], r["seed"], r["reference"], r["feature"], r["control"]): r
        for r in learning["records"]
    }
    admitted = {(r["seed"], r["intended_sites"]): r for r in admission["budget_cells"]}
    files = []
    commit = "portable-v1"
    out = B / "dynamicworld-main-readout-01"
    cache = R / ".scratch/semantic-dynamicworld-main-readout-01"
    out.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    arrays = {}
    records = []
    selection_records = []
    primary = {n: {} for n in ["clay", "prithvi", "appearance"]}
    parity = []
    for selection in selections:
        seed = selection["seed"]
        budget = selection["intended_sites"]
        chosen = selection["patch_ids"]
        ti = [index[p] for p in chosen]
        n = len(ti)
        parents = {rows[i]["parent_id"] for i in ti}
        assert n == selection["realised_sites"] == len(parents)
        assert all(rows[i]["role"] in ["fit", "fit_single_expert"] for i in ti)
        assert not parents & {rows[i]["parent_id"] for i in vi}
        selected_cells = {
            k: [
                p for p in ps if not (k.endswith("_fit") and rows[index[p]]["parent_id"] in parents)
            ]
            for k, ps in cells.items()
        }
        assert selected_cells == admitted[(seed, budget)]["cell_patch_ids"]
        selection_records.append(
            {
                "seed": seed,
                "intended_sites": budget,
                "realised_sites": n,
                "patch_ids": chosen,
                "cell_patch_ids": selected_cells,
            }
        )
        permutation = np.random.default_rng(2026101400 + seed).permutation(n)

        def emit(reference, name, control, cosine, regularisation=None):
            assert cosine.shape == (len(query), 8) and np.isfinite(cosine).all()
            key = f"{budget}-{seed}-{reference}-{name}-{control}"
            arrays[key + "-scores"] = cosine
            if reference in refs:
                old = previous[(budget, seed, reference, name, control)]
                aps = retrieval_metrics(cosine[: len(val)], labels[vi], ks=(1, 5, 8))[
                    "per_query_ap"
                ]
                v = aggregate(aps, vp, vc)["component_mean"]
                error = abs(v - old["component_map"])
                parity.append(error)
            for cell, ps in selected_cells.items():
                r = {
                    "seed": seed,
                    "intended_sites": budget,
                    "realised_sites": n,
                    "reference": reference,
                    "feature": name,
                    "control": control,
                    "cell": cell,
                    "array_key": key,
                    "regularisation": regularisation,
                    "query_ids": ps,
                }
                if not ps:
                    r["status"] = "unavailable_empty_cell"
                    records.append(r)
                    continue
                ix = [index[p] for p in ps]
                qix = [qindex[p] for p in ps]
                summary = summarise_retrieval(
                    cosine[qix],
                    labels[ix],
                    [rows[i]["parent_id"] for i in ix],
                    [rows[i]["component"] for i in ix],
                )
                aps = summary.pop("per_query_ap")
                recalls = summary.pop("per_query_recall")
                arrays[key + "-" + cell + "-ap"] = aps
                for k, v in recalls.items():
                    arrays[key + "-" + cell + f"-recall{k}"] = v
                r.update(status="available", summary=serial(summary))
                records.append(r)
                if (
                    budget == 512
                    and reference == "qwen_instructed"
                    and control == "paired"
                    and name in primary
                    and cell == "heldout_test"
                ):
                    primary[name][seed] = aps.copy()

        for reference, T in refs.items():
            Y = unit(labels[ti].astype(float) @ T)
            prior = np.broadcast_to(Y.mean(0), (len(query), Y.shape[1]))
            emit(reference, "none", "prior", unit(prior) @ T.T)
            for name, matrix in X.items():
                x, q = matrix[ti], matrix[qi]
                mean = x.mean(0)
                scale = x.std(0)
                scale[scale == 0] = 1
                xx = (x - mean) / scale
                qq = (q - mean) / scale
                nearest = nearest_rows(qq, xx)
                emit(reference, name, "nearest_example", unit(Y[nearest]) @ T.T)
                categories = [i for i, c in enumerate(classes) if c in known]
                centres = np.stack([xx[labels[ti, c]].mean(0) for c in categories])
                nearest_class = nearest_rows(qq, centres)
                emit(
                    reference,
                    name,
                    "nearest_prototype",
                    unit(T[np.array(categories)[nearest_class]]) @ T.T,
                )
                for control, target in [("paired", Y), ("shuffled", Y[permutation])]:
                    old = previous[(budget, seed, reference, name, control)]
                    lam = old["selected_regularisation"]
                    model = fit_ridge(x, target, lam)
                    prediction = model.predict(q)
                    emit(reference, name, control, unit(prediction) @ T.T, lam)
        for name, phrases in clip.items():
            T = unit(unit(phrases).reshape(8, 4, -1).mean(1))
            emit("native_remoteclip", name, "native", unit(X[name][qi]) @ T.T)
        print("Budget", budget, "seed", seed, "complete", flush=True)
    primary_results = {}
    ix = [index[p] for p in primary_ids]
    for name in ["clay", "prithvi"]:
        if set(primary[name]) == set(primary["appearance"]) == set(range(5)) and primary_ids:
            a = np.stack([primary[name][s] for s in range(5)])
            b = np.stack([primary["appearance"][s] for s in range(5)])
            r = candidate_paired_interval(
                a, b, [rows[i]["parent_id"] for i in ix], [rows[i]["component"] for i in ix]
            )
            arrays["primary-" + name + "-ap"] = a
            arrays["primary-appearance-ap"] = b
            r["admitted_hierarchy_qualified"] = precision["coverage_qualified"] is True
            r["model_mean"] = aggregate(
                a.mean(0), [rows[i]["parent_id"] for i in ix], [rows[i]["component"] for i in ix]
            )["component_mean"]
            r["appearance_mean"] = aggregate(
                b.mean(0), [rows[i]["parent_id"] for i in ix], [rows[i]["component"] for i in ix]
            )["component_mean"]
        else:
            r = {
                "one_sided_p": None,
                "admitted_hierarchy_qualified": False,
                "reason": "missing_primary_cell_or_seed",
            }
        primary_results[name] = r
    family = primary_family(primary_results)
    path = cache / "scores.npz"
    np.savez_compressed(path, **arrays)
    report = {
        "author": "James Edward Ball",
        "source_commit": commit,
        "source_sha256": {p: sha(R / p) for p in files},
        "input_sha256": hashes,
        "classes": classes,
        "score_query_ids": query,
        "validation_ids": val,
        "selections": selection_records,
        "records": records,
        "primary_query_ids": primary_ids,
        "primary_comparisons": primary_results,
        "primary_family": family,
        "validation_parity_count": len(parity),
        "validation_parity_max_error": max(parity),
        "arrays_path": str(path.relative_to(R)),
        "arrays_sha256": sha(path),
        "seconds": time.monotonic() - start,
        "limits": "Fixed eight-category threshold retrieval; conditional inference only under the saved admitted-hierarchy policy. H1 population inference unavailable; seeds are not geographic replications.",
    }
    (out / "report.json").write_text(json.dumps(serial(report), indent=2) + "\n")
    print(json.dumps(serial(primary_results)), flush=True)


if __name__ == "__main__":
    main()
