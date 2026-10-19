"""Frozen finite-inventory main geometry and nuisance panel. Author: James Edward Ball."""

import os

for k in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[k] = "1"
import hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from geofm_jlens.semantic_geometry_hierarchy_v1 import (
    prototypes,
    unit,
    distances,
    rsa,
    linear_cka,
    neighbour_overlap,
    correspondence_distribution,
)
from geofm_jlens.semantic_matching_v1 import matched_pairs

R = Path(os.environ["GEOFM_RUN_DIR"]).resolve()
B = R / "results/semantic-structure"
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    paths = [
        B / "dynamicworld-main-inputs-01/report.json",
        B / "dynamicworld-main-inputs-01/audit.json",
        B / "dynamicworld-text-reference-01/report.json",
        B / "dynamicworld-nuisance-calibration-01/report.json",
        B / "dynamicworld-nuisance-calibration-01/audit.json",
    ]
    bundle, audit, text, cal, ca = [read(p) for p in paths]
    assert audit["status"] == ca["status"] == "passed"
    hashes = {str(p.relative_to(R)): sha(p) for p in paths}
    for rec in [bundle, text, cal]:
        assert sha(R / rec["arrays_path"]) == rec["arrays_sha256"]
        hashes[rec["arrays_path"]] = rec["arrays_sha256"]
    rows = bundle["rows"]
    ids = [r["patch_id"] for r in rows]
    index = {p: i for i, p in enumerate(ids)}
    classes = bundle["classes"]
    assert classes == [r["id"] for r in text["classes"]] and len(classes) == 8
    gi = [index[p] for p in bundle["geometry_patch_ids"]]
    fi = [index[p] for p in bundle["calibration_fit_ids"]]
    assert len(fi) == 285 and all(rows[i]["role"] == "fit" for i in fi)
    assert all(rows[i]["role"] == "test" for i in gi)
    with np.load(R / bundle["arrays_path"], allow_pickle=False) as z:
        membership = z["membership"].copy()
        X = {name: z["feature__" + name].copy() for name in bundle["feature_names"]}
        nuisance = z["nuisance"].copy()
        oracle = z["oracle"].copy()
        oracle_ok = z["oracle_available"].copy()
    with np.load(R / cal["arrays_path"], allow_pickle=False) as z:
        nm = z["fit_mean"].copy()
        ns = z["fit_scale"].copy()
        for model in ["clay", "prithvi"]:
            X[model + "_residual"] = (
                X[model]
                - ((nuisance - nm) / ns) @ z[model + "-coefficients"]
                - z[model + "-intercept"]
            )
    X.update(quality=nuisance[:, -7:], nuisance=nuisance, oracle=oracle)
    with np.load(R / text["arrays_path"], allow_pickle=False) as z:
        refs = {k: z[k].copy() for k in ["qwen_instructed", "qwen_plain", "mpnet_plain"]}
    assert all(v.shape[0] == len(rows) for v in X.values())
    assert oracle_ok[gi].all() and oracle_ok[fi].all()
    files = []
    commit = "portable-v1"
    out = B / "dynamicworld-main-geometry-01"
    cache = R / ".scratch/semantic-dynamicworld-main-geometry-01"
    out.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    y = membership[gi]
    parents = np.array([rows[i]["parent_id"] for i in gi])
    components = np.array([rows[i]["component"] for i in gi])
    arrays = {"membership": y}
    records = []
    views = {}
    strata = [
        [classes.index(c) for c in ["Trees", "Grass", "Flooded_Vegetation", "Crops", "Scrub"]],
        [classes.index(c) for c in ["Water", "Bare_Ground", "Built_Area"]],
    ]
    for name, a in X.items():
        assert np.isfinite(a[gi]).all() and np.isfinite(a[fi]).all()
        mean = a[fi].mean(0)
        scale = a[fi].std(0)
        scale[scale == 0] = 1
        for route in ["native", "fit_standardised", "unit_patch"]:
            key = name + "__" + route
            r = {"feature": name, "route": route, "array_key": key}
            try:
                x = (
                    a[gi]
                    if route == "native"
                    else ((a[gi] - mean) / scale if route == "fit_standardised" else unit(a[gi]))
                )
                p, support = prototypes(x, y, parents, components)
                vd = distances(p)
                arrays[key + "__prototypes"] = p
                arrays[key + "__distance"] = vd
                views[key] = vd
                r.update(status="available", support=support, references={})
            except ValueError as e:
                r.update(status="unavailable", reason=str(e))
                records.append(r)
                continue
            deleted = []
            for component in sorted(set(components)):
                mask = components != component
                try:
                    dp, _ = prototypes(x[mask], y[mask], parents[mask], components[mask])
                    deleted.append((str(component), distances(dp), None))
                except ValueError as e:
                    deleted.append((str(component), None, str(e)))
            for reference, T in refs.items():
                td = distances(T)
                rs = {}
                try:
                    rs["rsa"] = rsa(vd, td)
                    rs["cka"] = linear_cka(p, T)
                    rs["neighbour_overlap"] = {str(k): neighbour_overlap(vd, td, k) for k in [1, 2]}
                    rs["neighbour_random_expectation"] = {"1": 1 / 7, "2": 2 / 7}
                    for kind, groups in [("global", None), ("coarse", strata)]:
                        distribution = correspondence_distribution(vd, td, groups)
                        ak = key + "__" + reference + "__" + kind
                        arrays[ak] = distribution["values"]
                        rs[kind + "_correspondence_reference"] = {
                            "exceedance_fraction": distribution["exceedance_fraction"],
                            "permutations": len(distribution["values"]),
                            "resolution": distribution["resolution"],
                            "array_key": ak,
                            "inferential_status": "diagnostic_reference_distribution",
                        }
                    rs["leave_one_category_out"] = {}
                    for c in range(8):
                        keep = [j for j in range(8) if j != c]
                        try:
                            rs["leave_one_category_out"][classes[c]] = {
                                "rsa": rsa(vd[np.ix_(keep, keep)], td[np.ix_(keep, keep)]),
                                "status": "available",
                            }
                        except ValueError as e:
                            rs["leave_one_category_out"][classes[c]] = {
                                "status": "unavailable",
                                "reason": str(e),
                            }
                    rs["leave_one_component_out"] = []
                    for c, dd, reason in deleted:
                        if dd is None:
                            rs["leave_one_component_out"].append(
                                {"component": c, "status": "unavailable", "reason": reason}
                            )
                            continue
                        try:
                            rs["leave_one_component_out"].append(
                                {"component": c, "status": "available", "rsa": rsa(dd, td)}
                            )
                        except ValueError as e:
                            rs["leave_one_component_out"].append(
                                {"component": c, "status": "unavailable", "reason": str(e)}
                            )
                    rs["status"] = "available"
                except ValueError as e:
                    rs.update(status="unavailable", reason=str(e))
                r["references"][reference] = rs
            records.append(r)
            print(key, "complete", flush=True)
    # Match once per partition, then apply the same edges to every feature condition.
    matched = []
    z = (nuisance - nm) / ns
    z = z[:, cal["matching_indices"]]
    for role in ["fit", "test"]:
        ix = [index[p] for p in bundle["evaluation_patch_ids"] if rows[index[p]]["role"] == role]
        ps = [rows[i]["parent_id"] for i in ix]
        cs = [rows[i]["component"] for i in ix]
        ss = [(rows[i]["platform"], (int(rows[i]["date"][5:7]) - 1) // 3) for i in ix]
        if not ix:
            matched.append({"role": role, "status": "unavailable_empty_partition"})
            continue
        pairs = matched_pairs(
            z[ix], membership[ix], [ids[i] for i in ix], ps, cs, ss, cal["matching_caliper"]
        )
        mr = {"role": role, "pairs": pairs, "features": {}}
        for name, a in X.items():
            try:
                if name in ["appearance", "metadata", "quality", "nuisance"]:
                    mean = a[fi].mean(0)
                    scale = a[fi].std(0)
                    scale[scale == 0] = 1
                    native = unit((a[ix] - mean) / scale)
                    route = "fit_standardised"
                else:
                    native = unit(a[ix])
                    route = "native"
            except ValueError as e:
                mr["features"][name] = {"status": "unavailable", "reason": str(e)}
                continue
            vd = np.zeros((8, 8))
            results = []
            for pair in pairs:
                left, right = pair["category_indices"]
                edges = pair["matched_edges"]
                ds = [float(1 - native[e["left_index"]] @ native[e["right_index"]]) for e in edges]
                score = float(np.mean(ds)) if ds else None
                results.append(
                    {
                        "category_indices": [left, right],
                        "matched_count": len(ds),
                        "mean_cosine_distance": score,
                        "per_edge_distance": ds,
                    }
                )
                if score is not None:
                    vd[left, right] = vd[right, left] = score
            stats = {
                "route": route,
                "pairs": results,
                "complete_28_pairs": all(r["matched_count"] for r in results),
                "references": {},
            }
            if stats["complete_28_pairs"]:
                for ref, T in refs.items():
                    try:
                        stats["references"][ref] = {
                            "rsa": rsa(vd, distances(T)),
                            "status": "available",
                        }
                    except ValueError as e:
                        stats["references"][ref] = {"status": "unavailable", "reason": str(e)}
            else:
                stats["status"] = "incomplete_pair_inventory"
            mr["features"][name] = stats
        matched.append(mr)
    arrays["shared_patch_counts"] = y.astype(int).T @ y.astype(int)
    path = cache / "arrays.npz"
    np.savez_compressed(path, **arrays)
    report = {
        "author": "James Edward Ball",
        "source_commit": commit,
        "source_sha256": {p: sha(R / p) for p in files},
        "input_sha256": hashes,
        "classes": classes,
        "geometry_patch_ids": bundle["geometry_patch_ids"],
        "calibration_fit_ids": bundle["calibration_fit_ids"],
        "records": records,
        "matched": matched,
        "arrays_path": str(path.relative_to(R)),
        "arrays_sha256": sha(path),
        "seconds": time.monotonic() - start,
        "limits": "Fixed finite-inventory geometry; correspondence fractions are diagnostic, population RSA inference unavailable. Oracle uses expert-label information. No causal or broad semantic conclusion.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Main geometry saved", flush=True)


if __name__ == "__main__":
    main()
