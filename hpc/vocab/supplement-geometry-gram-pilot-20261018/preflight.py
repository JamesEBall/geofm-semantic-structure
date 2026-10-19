"""Deterministic target-fidelity check. Author: James Edward Ball.

The conditional jackknife/Monte Carlo plan is deliberately not implemented
until this cheaper necessary condition passes.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "supplement-geometry-support-20261018"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(config, folder=FROZEN):
    for name, expected in config["frozen_source_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError("Frozen provenance mismatch: " + name)


def corrected_gram(sums, counts):
    """Average ordered cross-products over distinct blocks, retaining cell mass."""
    sums, counts = np.asarray(sums), np.asarray(counts)
    if not np.isfinite(sums).all() or not np.isfinite(counts).all():
        raise ValueError("nonfinite block sufficient statistic")
    if np.any(counts.sum(0) <= 0):
        raise ValueError("missing class")
    total, n = sums.sum(0), counts.sum(0)
    den = np.outer(n, n) - counts.T @ counts
    if np.any(den <= 0):
        raise ValueError("nonpositive cross-block denominator")
    numerator = total @ total.T - np.einsum("bcd,bed->ce", sums, sums)
    gram = numerator / den
    if not np.isfinite(gram).all() or np.any(np.diag(gram) <= 0):
        raise ValueError("nonpositive or nonfinite corrected self-norm")
    return gram


def gram_distances(gram):
    scale = np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    return (1 - gram / scale)[np.triu_indices(len(gram), 1)]


def statistics(db, ds, da, dg, dt):
    """Production average ranks and SVD rule, accepting estimated distances."""
    rb, rs, ra, rg, rt = [rankdata(x) for x in (db, ds, da, dg, dt)]

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        den = np.linalg.norm(x) * np.linalg.norm(y)
        return np.dot(x, y) / den if den > 1e-10 else np.nan

    ob, os = corr(rb, rt), corr(rs, rt)
    u, sv, _ = np.linalg.svd(
        np.column_stack([np.ones_like(ra), ra, rg]), full_matrices=False
    )
    u = u[:, sv > sv.max() * 1e-12]
    tb = rt - u @ (u.T @ rt)

    def partial(v):
        v = v - u @ (u.T @ v)
        den = np.linalg.norm(v) * np.linalg.norm(tb)
        return np.dot(v, tb) / den if den > 1e-10 else np.nan

    pb, ps = partial(rb), partial(rs)
    return np.array([ob, os, pb, ps, os - ob, ps - pb])


def physical_cell_means(labels, prototypes, location):
    """Build each physical-window vector once and reuse it across all its labels."""
    k, d = prototypes.shape
    sums, count = np.zeros((k, d)), np.zeros(k)
    for labs in labels:
        if not labs:
            continue
        vector = prototypes[labs].mean(0) + location
        sums[labs] += vector
        count[labs] += 1
    means = np.divide(
        sums, count[:, None], out=np.zeros_like(sums), where=count[:, None] > 0
    )
    return means, count > 0


def oracle(ds, world, beta):
    """Exact noiseless cell means for the original finite geographic template."""
    k, d = world["a"].shape
    counts = np.zeros((ds["nb"], k))
    sums = np.zeros((3, ds["nb"], k, d))
    prototypes = [
        world["a"] + 0.7 * world["v"],
        world["a"] + 0.7 * world["v"] + beta * world["s"],
        world["a"],
    ]
    for g, cell in enumerate(ds["cells"]):
        if cell["split"] != "test":
            continue
        for r, (proto, locscale) in enumerate(zip(prototypes, [0.6, 0.6, 0.3])):
            means, present = physical_cell_means(
                cell["window_labels"], proto, locscale * world["L"][ds["types"][g]]
            )
            sums[r, ds["block"][g]] += means
        counts[ds["block"][g]] += present
    return sums, counts


def run():
    cfg = json.loads((HERE / "CONFIG.json").read_text())
    verify_sources(cfg)
    sys.path.insert(0, str(FROZEN))
    import calibration as frozen

    oldcfg = frozen.read(FROZEN / "CONFIG.json")
    ds = frozen.design()
    rows = []
    for scenario in cfg["scenarios"]:
        world = frozen.population(oldcfg, scenario)
        sums, counts = oracle(ds, world, scenario["beta"])
        means = sums.sum(1) / counts.sum(0)[None, :, None]
        plugin = frozen.geometry_values(*means, world["geo"], world["text"])
        error = float(np.max(abs(plugin - world["truth"])))
        if error > 1e-10:
            raise ValueError(
                "Original physical-window oracle does not match analytic truth"
            )
        result = {
            "scenario": scenario["id"],
            "original_truth": world["truth"].tolist(),
            "oracle_reconstruction_max_error": error,
        }
        try:
            grams = [corrected_gram(x, counts) for x in sums]
            distances = [gram_distances(x) for x in grams]
            values = statistics(
                *distances,
                pdist(world["geo"], "cosine"),
                pdist(world["text"], "cosine"),
            )
            delta = values - world["truth"]
            result.update(
                candidate=values.tolist(),
                delta=delta.tolist(),
                target_fidelity_pass=bool(
                    np.isfinite(values).all()
                    and np.max(abs(delta)) <= cfg["target_identity_atol"]
                ),
                routes=[],
            )
            for mean, gram, distance in zip(means, grams, distances):
                original = pdist(mean, "cosine")
                result["routes"].append(
                    {
                        "max_gram_delta": float(np.max(abs(gram - mean @ mean.T))),
                        "max_cosine_distance_delta": float(
                            np.max(abs(distance - original))
                        ),
                        "changed_distance_ranks": int(
                            np.count_nonzero(rankdata(distance) != rankdata(original))
                        ),
                        "cosines_outside_unit_range": int(
                            np.count_nonzero(abs(1 - distance) > 1)
                        ),
                    }
                )
        except ValueError as exc:
            result.update(target_fidelity_pass=False, unavailable_reason=str(exc))
        rows.append(result)
    return {
        "author": "James Edward Ball",
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "config_sha256": digest(HERE / "CONFIG.json"),
        "implementation_sha256": digest(__file__),
        "scenario_count": len(rows),
        "all_target_fidelity_pass": all(x["target_fidelity_pass"] for x in rows),
        "monte_carlo_executed": False,
        "empirical_interval_admitted": False,
        "metrics": list(frozen.METRICS),
        "results": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}))
