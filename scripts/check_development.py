"""Verify supplementary development geometry. Author: James Edward Ball."""

import hashlib, json, subprocess
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from study.releases import repository

R = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.isfinite(x).all() and (n > 1e-12).all()
    return x / n


def dist(x):
    return 1 - unit(x) @ unit(x).T


def corr(a, b):
    ix = np.triu_indices(len(a), 1)
    return float(spearmanr(a[ix], b[ix]).statistic)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    args = parser.parse_args()
    work = args.work.resolve()
    cache = work / "development"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((R / "data/development.json").read_text())
    asset = manifest["asset"]
    p = cache / asset["name"]
    if not p.exists():
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                "semantic-structure-v1",
                "--repo",
                repository(),
                "--pattern",
                asset["name"],
                "--dir",
                str(cache),
            ],
            check=True,
        )
    assert sha(p) == asset["sha256"], "Development data checksum mismatch"
    a = np.load(p)
    t = a
    groups = manifest["groups"]
    membership = a["membership"]
    region = a["regions"]
    assert membership.shape == (7, 1000)
    features = {
        "appearance": a["features"],
        "metadata": a["metadata"],
        "clay": a["clay"].astype(float),
        "prithvi": a["prithvi"].astype(float),
    }
    arrays = {"membership": membership, "regions": region, "scene_ids": a["scene_ids"]}
    visual = {}
    for route in ["standardised", "raw"]:
        for name, original in features.items():
            x = original.astype(float).copy()
            if route == "standardised":
                sd = x.std(0)
                sd[sd == 0] = 1
                x = (x - x.mean(0)) / sd
            x = unit(x)
            prototypes = []
            for mask in membership:
                means = [
                    unit(x[mask & (region == g)].mean(0))
                    for g in sorted(set(region[mask]))
                ]
                prototypes.append(unit(np.mean(means, axis=0)))
            key = route + "_" + name
            arrays[key + "_prototypes"] = np.array(prototypes)
            visual[key] = dist(np.array(prototypes))
            arrays[key + "_distances"] = visual[key]
    overlap = np.array(
        [
            [
                1 - np.logical_and(x, y).sum() / np.logical_or(x, y).sum()
                for y in membership
            ]
            for x in membership
        ]
    )
    visual["membership_overlap"] = overlap
    arrays["membership_overlap_distances"] = overlap
    refs = {}
    for name in [
        n + "__" + v
        for n in ["Qwen3-Embedding-0.6B", "all-mpnet-base-v2"]
        for v in ["constituent_mean", "original"]
    ]:
        v = t[name + "_vectors"]
        td = dist(v)
        arrays[name + "_text_vectors"] = v
        arrays[name + "_text_distances"] = td
        scores = {}
        for method, vd in visual.items():
            loo = {}
            for j, g in enumerate(groups):
                ix = [k for k in range(7) if k != j]
                loo[g] = corr(vd[np.ix_(ix, ix)], td[np.ix_(ix, ix)])
            scores[method] = {"rsa": corr(vd, td), "leave_one_group_out": loo}
        refs[name] = {"scores": scores, "construction": name.split("__")[1]}
    from scipy.stats import rankdata

    gates = []
    for text, ref in refs.items():
        td = arrays[text + "_text_distances"]
        ix = np.triu_indices(7, 1)
        for route in ["standardised", "raw"]:
            X = np.column_stack(
                [np.ones(21)]
                + [
                    rankdata(visual[n][ix])
                    for n in [
                        route + "_appearance",
                        route + "_metadata",
                        "membership_overlap",
                    ]
                ]
            )
            assert np.linalg.matrix_rank(X) == 4
            for model in ["clay", "prithvi"]:
                method = route + "_" + model
                y = rankdata(visual[method][ix])
                z = rankdata(td[ix])
                yr = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                zr = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
                res = float(np.corrcoef(yr, zr)[0, 1])
                ref["scores"][method]["residual"] = res
                if route == "standardised" and text.endswith("__constituent_mean"):
                    score = ref["scores"][method]["rsa"]
                    baselines = {
                        b: ref["scores"][b]["rsa"]
                        for b in [
                            "standardised_appearance",
                            "standardised_metadata",
                            "membership_overlap",
                        ]
                    }
                    gates.append(
                        {
                            "text": text,
                            "model": model,
                            "rsa": score,
                            "residual": res,
                            "baselines": baselines,
                            "passed": score > max(0, *baselines.values()) and res > 0,
                        }
                    )
    errors = []
    for text, ref in refs.items():
        for method, value in ref["scores"].items():
            expected = manifest["text_references"][text]["scores"][method]
            errors.append(abs(value["rsa"] - expected["rsa"]))
            if "residual" in value:
                errors.append(abs(value["residual"] - expected["residual"]))
            errors.extend(
                abs(v - expected["leave_one_group_out"][k])
                for k, v in value["leave_one_group_out"].items()
            )
    cross = {
        variant: corr(
            arrays["Qwen3-Embedding-0.6B__" + variant + "_text_distances"],
            arrays["all-mpnet-base-v2__" + variant + "_text_distances"],
        )
        for variant in ["original", "constituent_mean"]
    }
    assert max(errors) < 1e-10
    for variant, value in cross.items():
        expected = manifest["text_cross_correlations"][
            "Qwen3-Embedding-0.6B__" + variant
        ]["all-mpnet-base-v2__" + variant]
        assert abs(value - expected) < 1e-10
    assert all(g["passed"] for g in gates) == manifest["directional_gate"]["all_passed"]
    result = {
        "author": "James Edward Ball",
        "status": "passed",
        "comparisons": len(errors),
        "maximum_absolute_error": max(errors),
        "text_cross_correlations": cross,
        "directional_gates": gates,
        "scope": "Recomputes the supplementary development geometry from archived per-scene features and text vectors; source imagery extraction is outside this check.",
    }
    (cache / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
