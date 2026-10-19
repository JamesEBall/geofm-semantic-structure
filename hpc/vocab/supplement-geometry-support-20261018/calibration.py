"""Retained-support calibration amendment. Author: James Edward Ball."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from geometry import geometry_values, normalise, wilson, band_result

HERE = Path(__file__).resolve().parent
METRICS = (
    "ordinary_baseline",
    "ordinary_semantic",
    "partial_baseline",
    "partial_semantic",
    "ordinary_difference",
    "partial_difference",
)
SPLITS = ("train", "val", "test")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def provenance():
    return {
        name + "_sha256": hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in [
            "PROTOCOL.md",
            "PREFLIGHT-AMENDMENT.md",
            "PHYSICAL-WINDOW-AMENDMENT.md",
            "CONFIG.json",
            "RETAINED-DESIGN.json",
            "calibration.py",
            "geometry.py",
        ]
    }


def validate(record, hashes, phase=None, scenario=None, rep=None):
    for k, v in hashes.items():
        if record.get(k) != v:
            raise ValueError("provenance mismatch " + k)
    for k, v in [("phase", phase), ("scenario", scenario), ("replicate", rep)]:
        if v is not None and record.get(k) != v:
            raise ValueError("identity mismatch " + k)


@lru_cache(maxsize=1)
def design():
    return prepare_design(read(HERE / "RETAINED-DESIGN.json"))


def prepare_design(raw):
    cells = raw["cells"]
    k = len(raw["classes"])
    geos = sorted(set(tuple(x["geo"]) for x in cells))
    blocks = sorted(set(tuple(x["block"]) for x in cells))
    regions = sorted(set((x["block"][0] // 4, x["block"][1] // 4) for x in cells))
    geo = {v: i for i, v in enumerate(geos)}
    block = {v: i for i, v in enumerate(blocks)}
    region = {v: i for i, v in enumerate(regions)}
    fam = np.unique([x["family_id"] for x in raw["classes"]], return_inverse=True)[1]
    split = np.array([SPLITS.index(x["split"]) for x in cells])
    types = np.array([geo[tuple(x["geo"])] for x in cells])
    bl = np.array([block[tuple(x["block"])] for x in cells])
    reg = np.array([region[(x["block"][0] // 4, x["block"][1] // 4)] for x in cells])
    rows = []
    cols = []
    counts = []
    for g, x in enumerate(cells):
        for c, n in x["counts"]:
            rows.append(g)
            cols.append(c)
            counts.append(n)
    count = csr_matrix((counts, (rows, cols)), shape=(len(cells), k), dtype=float)
    test_support = np.asarray((count[split == 2] > 0).sum(axis=0)).ravel()
    label_mixture = np.zeros((k, k))
    for x in cells:
        if x["split"] != "test":
            continue
        ns = dict(x["counts"])
        for labs in x["window_labels"]:
            for target in labs:
                label_mixture[target, labs] += 1 / (
                    test_support[target] * ns[target] * len(labs)
                )
    return dict(
        label_mixture=label_mixture,
        test_support=test_support,
        raw=raw,
        cells=cells,
        k=k,
        fam=fam,
        split=split,
        types=types,
        block=bl,
        region=reg,
        h=len(geos),
        nb=len(blocks),
        nr=len(regions),
        count=count,
    )


def population(cfg, sc):
    """Compute exact means over the fixed retained-template population."""
    ds = design()
    k, d, h = ds["k"], cfg["dimension"], ds["h"]
    rng = np.random.default_rng([cfg["world_seed"], 101])
    fam = ds["fam"]
    f = fam.max() + 1

    def latent():
        return np.sqrt(0.6) * rng.normal(size=(f, d))[fam] + np.sqrt(0.4) * rng.normal(
            size=(k, d)
        )

    a, s, v, L = latent(), latent(), rng.normal(size=(k, d)), rng.normal(size=(h, d))
    test = np.flatnonzero(ds["split"] == 2)
    present = ds["count"][test].copy()
    present.data[:] = 1
    loc = (
        np.asarray(present.T @ L[ds["types"][test]])
        / np.asarray(present.sum(axis=0)).ravel()[:, None]
    )
    base = ds["label_mixture"] @ (a + 0.7 * v) + 0.6 * loc
    sem = base + sc["beta"] * (ds["label_mixture"] @ s)
    app = ds["label_mixture"] @ a + 0.3 * loc
    geo = np.zeros((k, h))
    co = ds["count"].tocoo()
    np.add.at(geo, (co.col, ds["types"][co.row]), co.data)
    text = normalise(a + 0.6 * loc + 0.8 * s)
    if sc["independent_text"]:
        text = normalise(
            np.random.default_rng([cfg["world_seed"], 901]).normal(size=(k, d))
        )
    truth = geometry_values(base, sem, app, geo, text)
    gr = pdist(geo, "cosine")
    _, ct = np.unique(gr, return_counts=True)
    return dict(
        a=a,
        s=s,
        v=v,
        L=L,
        text=text,
        truth=truth,
        geo=geo,
        truth_geo_equal_pair_fraction=float(ct[ct > 1].sum() / len(gr)),
        truth_geo_unique_distances=len(ct),
    )


def sample(cfg, sc, phase, rep, scenario_index):
    """Generate one full vector per physical window and reuse it for every label."""
    ds = design()
    world = population(cfg, sc)
    rng = np.random.default_rng([cfg[phase + "_seed"], 201, scenario_index, rep])
    k, d = ds["k"], cfg["dimension"]
    regime = sc["regime"]
    factor = 4 if regime == "support_x4" else 1
    chosen, splits = [], []
    for split in range(3):
        pool = np.flatnonzero(ds["split"] == split)
        take = (
            pool
            if regime == "fixed_design"
            else rng.choice(pool, size=len(pool) * factor, replace=True)
        )
        chosen.extend(take)
        splits.extend([split] * len(take))
    chosen, splits = np.asarray(chosen), np.asarray(splits)
    g = len(chosen)
    counts = ds["count"][chosen].tocsr()
    co = counts.tocoo()
    types, blocks, regions = (
        ds["types"][chosen],
        ds["block"][chosen],
        ds["region"][chosen],
    )
    support, wins = np.zeros((3, k), int), np.zeros((3, k))
    for split in range(3):
        sub = counts[splits == split]
        support[split] = np.asarray((sub > 0).sum(axis=0)).ravel()
        wins[split] = np.asarray(sub.sum(axis=0)).ravel()
    th = ds["raw"]["admission"]
    admit = (
        (wins[0] >= th["train"])
        & (wins[1] >= th["val"])
        & (wins[2] >= th["test"])
        & (support[2] >= th["test_cells"])
        & (support.sum(axis=0) >= th["cells"])
    )
    test_edge = splits[co.row] == 2
    erow, ecol, en = co.row[test_edge], co.col[test_edge], co.data[test_edge]
    test_rows = np.flatnonzero(splits == 2)
    local = np.full(g, -1)
    local[test_rows] = np.arange(len(test_rows))
    pair_index = {(int(rr), int(cc)): i for i, (rr, cc) in enumerate(zip(erow, ecol))}
    wr, wc, wv = [], [], []
    mr, mc, mv = [], [], []
    fr, fc, fv = [], [], []
    window_cells, bundles, window_scales = [], [], []
    bundle_map, family_map = {}, {}
    n_support = ds["test_support"]
    for rr in test_rows:
        for labs in ds["cells"][chosen[rr]]["window_labels"]:
            wi = len(window_cells)
            window_cells.append(int(rr))
            bundle = (int(rr), tuple(sorted(labs)))
            if bundle not in bundle_map:
                bundle_map[bundle] = len(bundle_map)
            bundles.append(bundle_map[bundle])
            for c in labs:
                ei = pair_index[(int(rr), c)]
                wr.append(ei)
                wc.append(wi)
                wv.append(1 / en[ei])
                mr.append(wi)
                mc.append(c)
                mv.append(1 / len(labs))
            families = np.unique(ds["fam"][labs]) if labs else []
            for family in families:
                key = (int(rr), int(family))
                if key not in family_map:
                    family_map[key] = len(family_map)
                fr.append(wi)
                fc.append(family_map[key])
                fv.append(1 / len(families))
            scale = (
                np.sqrt(1 + np.mean(22 / n_support[labs]))
                if labs and regime == "heterogeneous"
                else 1.0
            )
            window_scales.append(scale)
    window_cells = np.asarray(window_cells)
    bundles = np.asarray(bundles)
    nw = len(window_cells)
    wm = csr_matrix((wv, (wr, wc)), shape=(len(erow), nw))
    mixture = csr_matrix((mv, (mr, mc)), shape=(nw, k))
    family_mix = csr_matrix((fv, (fr, fc)), shape=(nw, len(family_map)))
    ng = len(test_rows)
    window_local = local[window_cells]
    common = 0.7 * rng.normal(size=(ng, d))[window_local] + 0.5 * (
        family_mix @ rng.normal(size=(len(family_map), d))
    )
    scale = np.asarray(window_scales)[:, None] * (0.5 if regime == "low_noise" else 1.0)
    spatial, spatial_app = np.zeros((nw, d)), np.zeros((nw, d))
    if regime in ("spatial_five", "spatial_twenty"):
        ids = (
            blocks[window_cells] if regime == "spatial_five" else regions[window_cells]
        )
        n = ds["nb"] if regime == "spatial_five" else ds["nr"]
        spatial = 0.7 * rng.normal(size=(n, d))[ids]
        spatial_app = 0.4 * rng.normal(size=(n, d))[ids]

    def route():
        return (
            scale
            * (
                common
                + 0.5 * rng.normal(size=(len(bundle_map), d))[bundles]
                + rng.normal(size=(nw, d))
            )
            + spatial
        )

    base_mean = (
        mixture @ (world["a"] + 0.7 * world["v"])
        + 0.6 * world["L"][types[window_cells]]
    )
    baseline_windows = base_mean + route()
    semantic_windows = base_mean + sc["beta"] * (mixture @ world["s"]) + route()
    appearance_windows = (
        mixture @ world["a"]
        + 0.3 * world["L"][types[window_cells]]
        + scale
        * (
            0.4 * rng.normal(size=(ng, d))[window_local]
            + 0.3 * (family_mix @ rng.normal(size=(len(family_map), d)))
            + 0.7 * rng.normal(size=(nw, d))
        )
        + spatial_app
    )
    window_features = np.concatenate(
        [baseline_windows, semantic_windows, appearance_windows], axis=1
    )
    values = wm @ window_features
    dd = 3 * d
    feat = csr_matrix(
        (
            values.ravel(),
            (np.repeat(erow, dd), (ecol[:, None] * dd + np.arange(dd)).ravel()),
        ),
        shape=(g, k * dd),
    )
    present = counts.copy()
    present.data[:] = 1
    present = present.multiply((splits == 2)[:, None]).tocsr()
    geo = csr_matrix(
        (co.data, (co.row, co.col * ds["h"] + types[co.row])), shape=(g, k * ds["h"])
    )
    block_support = np.array(
        [len(np.unique(blocks[erow[ecol == c]])) for c in range(k)]
    )
    return dict(
        world=world,
        features=feat,
        present=present,
        geo=geo,
        split=splits,
        block=blocks,
        g=g,
        k=k,
        d=d,
        h=ds["h"],
        admitted=admit,
        support=support,
        window_counts=wins,
        block_support=block_support,
        window_averager=wm,
        evaluation_edge_rows=erow,
        evaluation_edge_classes=ecol,
        window_features=window_features,
    )


def rebuild(s, weights):
    """Rebuild equal-cell prototypes and window-count geography with shared weights."""
    w = np.atleast_2d(weights)
    den = np.asarray(w @ s["present"])
    valid = (den > 0).all(axis=1)
    means = (
        np.asarray(w @ s["features"]).reshape(len(w), s["k"], 3 * s["d"])
        / np.where(den > 0, den, 1)[..., None]
    )
    geo = np.asarray(w @ s["geo"]).reshape(len(w), s["k"], s["h"])
    return (*np.split(means, 3, axis=-1), geo, valid)


def draw_weights(s, kind, b, rng):
    """Draw positive cell or block multipliers and preserve split sample sizes."""
    if kind == "cell":
        w = rng.exponential(size=(b, s["g"]))
    elif kind == "block":
        _, idx = np.unique(s["block"], return_inverse=True)
        w = rng.exponential(size=(b, idx.max() + 1))[:, idx]
    else:
        raise ValueError(kind)
    for split in range(3):
        ix = s["split"] == split
        w[:, ix] /= w[:, ix].mean(axis=1, keepdims=True)
    return w


def effective_support(s, w, kind):
    den = np.asarray(w @ s["present"])
    sq = np.asarray((w * w) @ s["present"])
    cell = den * den / sq
    if kind != "block":
        return cell, None
    test = np.flatnonzero(s["split"] == 2)
    _, inv = np.unique(s["block"][test], return_inverse=True)
    aggregate = (
        csr_matrix((np.ones(len(test)), (inv, test)), shape=(inv.max() + 1, s["g"]))
        @ s["present"]
    )
    first = np.array([test[np.flatnonzero(inv == b)[0]] for b in range(inv.max() + 1)])
    square = aggregate.copy()
    square.data **= 2
    block = den * den / np.asarray((w[:, first] ** 2) @ square)
    return cell, block


def interval_result(point, draws, truth, kind, metric):
    """Invert bootstrap errors for basic intervals; retain bounded/raw endpoints."""
    q = np.quantile(draws, [0.025, 0.05, 0.95, 0.975])
    raw = [2 * point - q[3], 2 * point - q[0]] if kind == "basic" else [q[0], q[3]]
    lower_one = 2 * point - q[2] if kind == "basic" else q[1]
    bound = 2.0 if metric.endswith("difference") else 1.0
    lo = max(-bound, raw[0])
    hi = min(bound, raw[1])
    available = lo <= hi
    return dict(
        raw_lower=float(raw[0]),
        raw_upper=float(raw[1]),
        lower=float(lo),
        upper=float(hi),
        available=bool(available),
        covers=bool(available and lo <= truth <= hi),
        width=float(max(0, hi - lo)),
        one_sided_reject=bool(available and lower_one > 0),
    )


def one_replicate(cfg, index, phase, rep):
    start = time.monotonic()
    sc = cfg["scenarios"][index]
    s = sample(cfg, sc, phase, rep, index)
    world = s["world"]
    truth = world["truth"]
    rec = {
        **provenance(),
        "phase": phase,
        "scenario": sc["id"],
        "replicate": rep,
        "truth": dict(zip(METRICS, truth.tolist())),
        "all_admitted": bool(s["admitted"].all()),
        "admitted_classes": int(s["admitted"].sum()),
        "failed_admission_class_positions": np.flatnonzero(~s["admitted"]).tolist(),
        "test_cell_support": s["support"][2].tolist(),
        "test_block_support": s["block_support"].tolist(),
        "methods": {},
        "bootstrap": cfg["bootstrap"],
    }
    if np.any(s["support"][2] == 0):
        return {
            **rec,
            "failure": "initial_sample_missing_class",
            "elapsed_seconds": time.monotonic() - start,
        }
    v = rebuild(s, np.ones(s["g"]))
    point = geometry_values(*(x[0] for x in v[:4]), world["text"])
    if not np.isfinite(point).all():
        return {
            **rec,
            "failure": "undefined_point_geometry",
            "elapsed_seconds": time.monotonic() - start,
        }
    rec["point"] = dict(zip(METRICS, point.tolist()))
    for wi, weight_kind in enumerate(cfg["weights"]):
        rng = np.random.default_rng([cfg[phase + "_seed"], 301, index, rep, wi])
        draws = []
        ess = []
        block_ess = []
        for start_b in range(0, cfg["bootstrap"], cfg["bootstrap_batch"]):
            b = min(cfg["bootstrap_batch"], cfg["bootstrap"] - start_b)
            w = draw_weights(s, weight_kind, b, rng)
            values = rebuild(s, w)
            if not values[4].all():
                raise AssertionError("Positive weights lost existing support")
            draws.append(geometry_values(*values[:4], world["text"]))
            cell_eff, block_eff = effective_support(s, w, weight_kind)
            ess.append(cell_eff)
            if block_eff is not None:
                block_ess.append(block_eff)
        draws = np.concatenate(draws)
        eff = np.concatenate(ess)
        rec[weight_kind + "_effective_support"] = {
            "cell_weight_minimum": float(eff.min()),
            "cell_weight_per_class_median": np.median(eff, axis=0).tolist(),
            "cell_weight_per_class_q025": np.quantile(eff, 0.025, axis=0).tolist(),
        }
        if block_ess:
            be = np.concatenate(block_ess)
            rec[weight_kind + "_effective_support"].update(
                independent_block_minimum=float(be.min()),
                independent_block_per_class_median=np.median(be, axis=0).tolist(),
                independent_block_per_class_q025=np.quantile(
                    be, 0.025, axis=0
                ).tolist(),
            )
        if not np.isfinite(draws).all():
            rec[weight_kind + "_failure"] = "undefined_bootstrap_geometry"
            continue
        corrected = 2 * point - draws.mean(axis=0)
        for interval_kind in cfg["intervals"]:
            rec["methods"][weight_kind + "_" + interval_kind] = {
                m: {
                    **interval_result(
                        point[j], draws[:, j], truth[j], interval_kind, m
                    ),
                    "corrected_point": float(corrected[j]),
                }
                for j, m in enumerate(METRICS)
            }
    rec["elapsed_seconds"] = time.monotonic() - start
    return rec


def summarise(records, cfg, phase, require_complete=True):
    expected = {
        (s["id"], r)
        for s in cfg["scenarios"]
        for r in range(replicate_count(cfg, phase, s))
    }
    seen = set()
    hashes = provenance()
    groups = {}
    for r in records:
        validate(r, hashes, phase)
        key = (r["scenario"], r["replicate"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate/unregistered identity")
        seen.add(key)
        groups.setdefault(r["scenario"], []).append(r)
    if require_complete and seen != expected:
        raise ValueError("Incomplete phase")
    out = {
        **hashes,
        "phase": phase,
        "complete": seen == expected,
        "total_replicates": len(records),
        "scenarios": {},
        "qualification": {},
    }
    for name, rows in sorted(groups.items()):
        n = len(rows)
        admitted = [r for r in rows if r["all_admitted"]]
        x = {
            "replicates": n,
            "all_admission_rate": wilson(len(admitted), n),
            "initial_missing_class_rate": wilson(
                sum(r.get("failure") == "initial_sample_missing_class" for r in rows), n
            ),
            "methods": {},
            "failure_counts": {},
        }
        for r in rows:
            if r.get("failure"):
                x["failure_counts"][r["failure"]] = (
                    x["failure_counts"].get(r["failure"], 0) + 1
                )
        for method in [
            "cell_basic",
            "cell_percentile",
            "block_basic",
            "block_percentile",
        ]:
            x["methods"][method] = {}
            for m in METRICS:
                truth = rows[0]["truth"][m]
                av = [
                    r
                    for r in rows
                    if r.get("methods", {}).get(method, {}).get(m, {}).get("available")
                ]
                ad = [
                    r
                    for r in admitted
                    if r.get("methods", {}).get(method, {}).get(m, {}).get("available")
                ]
                points = np.array([r["point"][m] for r in rows if "point" in r])
                corr = np.array(
                    [r["methods"][method][m]["corrected_point"] for r in av]
                )
                cover = sum(r["methods"][method][m]["covers"] for r in av)
                rej = sum(r["methods"][method][m]["one_sided_reject"] for r in av)
                v = {
                    "truth": truth,
                    "availability": wilson(len(av), n),
                    "coverage": band_result(wilson(cover, n), cfg["coverage_band"]),
                    "coverage_conditional_admission": wilson(
                        sum(r["methods"][method][m]["covers"] for r in ad),
                        len(admitted),
                    ),
                    "coverage_available_only": wilson(cover, len(av)),
                    "one_sided_rejection": (
                        band_result(wilson(rej, n), cfg["size_band"])
                        if m.endswith("difference") and abs(truth) < 1e-12
                        else wilson(rej, n)
                    ),
                    "mean_width": (
                        float(np.mean([r["methods"][method][m]["width"] for r in av]))
                        if av
                        else None
                    ),
                }
                for label, values in [("point", points), ("corrected_point", corr)]:
                    v[label + "_bias"] = (
                        float(values.mean() - truth) if len(values) else None
                    )
                    v[label + "_bias_mcse"] = (
                        float(values.std(ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else None
                    )
                if m.endswith("difference") and truth > 0:
                    v["power_reference_pass"] = rej / n >= cfg["power_reference"]
                x["methods"][method][m] = v
        out["scenarios"][name] = x
    if phase == "confirmation" and out["complete"]:
        for method in ["cell_basic", "block_basic"]:
            rule = cfg["qualification"][method]
            out["qualification"][method] = {}
            for m in METRICS:
                coverage = all(
                    out["scenarios"][s]["methods"][method][m]["coverage"][
                        "mc_interval_entirely_in_band"
                    ]
                    for s in rule["coverage_scenarios"]
                )
                size = (
                    all(
                        out["scenarios"][s]["methods"][method][m][
                            "one_sided_rejection"
                        ]["mc_interval_entirely_in_band"]
                        for s in rule["size_scenarios"]
                    )
                    if m.endswith("difference")
                    else True
                )
                out["qualification"][method][m] = {
                    "coverage_all_named_scenarios": coverage,
                    "size_all_named_null_scenarios": size,
                    "candidate_scope_qualified": coverage and size,
                    "empirical_interval_admitted": False,
                }
    return out


def replicate_count(cfg, phase, scenario):
    return (
        cfg["confirmation_null_replicates"]
        if phase == "confirmation" and scenario["beta"] == 0
        else cfg[phase + "_replicates"]
    )


def run_item(args):
    cfg, index, phase, rep, out = args
    name = cfg["scenarios"][index]["id"]
    path = Path(out) / "records" / name / f"rep-{rep:05d}.json"
    if path.exists():
        validate(read(path), provenance(), phase, name, rep)
        return
    write(path, one_replicate(cfg, index, phase, rep))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--phase", choices=["development", "confirmation"], required=True)
    run.add_argument("--scenario", type=int)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=1)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--pilot", action="store_true")
    run.add_argument("--output", required=True)
    su = sub.add_parser("summarise")
    su.add_argument("--phase", required=True)
    su.add_argument("--input", required=True)
    su.add_argument("--output", required=True)
    su.add_argument("--allow-incomplete", action="store_true")
    tr = sub.add_parser("truth")
    tr.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = read(HERE / "CONFIG.json")
    if a.command == "truth":
        write(
            a.output,
            {
                s["id"]: {
                    "metrics": dict(zip(METRICS, population(cfg, s)["truth"].tolist())),
                    "geo_equal_pair_fraction": population(cfg, s)[
                        "truth_geo_equal_pair_fraction"
                    ],
                    "geo_unique_distances": population(cfg, s)[
                        "truth_geo_unique_distances"
                    ],
                }
                for s in cfg["scenarios"]
            },
        )
        return
    if a.command == "summarise":
        rec = [read(f) for f in Path(a.input).rglob("rep-*.json")]
        write(a.output, summarise(rec, cfg, a.phase, not a.allow_incomplete))
        return
    if a.pilot:
        if a.phase != "development":
            p.error("pilot must use development")
        tasks = [
            (cfg, i, a.phase, r, a.output)
            for i in range(len(cfg["scenarios"]))
            for r in (0, 1)
        ]
    else:
        if a.scenario is None:
            p.error("scenario required")
        if a.start < 0 or a.start + a.count > replicate_count(
            cfg, a.phase, cfg["scenarios"][a.scenario]
        ):
            p.error("invalid replicate range")
        tasks = [
            (cfg, a.scenario, a.phase, r, a.output)
            for r in range(a.start, a.start + a.count)
        ]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run_item, tasks))
    print(json.dumps({"completed": len(tasks), "output": a.output, "phase": a.phase}))


if __name__ == "__main__":
    main()
