"""Outcome-blind retained training feature diagnostics. Author: James Edward Ball."""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
FEATURES = {
    "r1-clay": "extract/r1/clay/mean.npy",
    "r1-prithvi": "extract/r1/prithvi/mean.npy",
    "appearance_rich": "features/appearance_rich.npy",
}
METADATA = [
    "assemble/windows_qa.parquet",
    "labels_exact/labels.parquet",
    "admit/windows_final.parquet",
    "admit/classes_final.parquet",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(root, manifest, name):
    expected = manifest["files"].get(name)
    actual = digest(root / name)
    if expected is None or expected != actual:
        raise ValueError("Manifest provenance mismatch: " + name)
    return actual


def checked_training_rows(windows, labels, retained):
    if not len(windows) == len(labels) == len(retained):
        raise ValueError("metadata length mismatch")
    out = []
    for i, (w, l, r) in enumerate(zip(windows, labels, retained)):
        if (
            not w["widx"] == l["widx"] == r["widx"] == i
            or not w["window_id"] == l["window_id"] == r["window_id"]
            or type(r.get("retained")) is not bool
        ):
            raise ValueError("retained/label identity mismatch")
        if r["retained"] and w["split"] == "train":
            out.append(i)
    if not out:
        raise ValueError("empty retained training set")
    return np.array(out, dtype=int)


def half(block, seed):
    return (
        int(hashlib.sha256(f"{seed}:{block[0]}:{block[1]}".encode()).hexdigest(), 16)
        % 2
    )


def describe(x):
    x = np.asarray(x, dtype=float)
    return (
        {
            "n": int(x.size),
            "min": float(x.min()),
            "median": float(np.median(x)),
            "max": float(x.max()),
            "mean": float(x.mean()),
        }
        if x.size
        else {"n": 0}
    )


def run(root, expected_manifest_sha256):
    start = time.monotonic()
    cfg = json.loads((HERE / "CONFIG.json").read_text())
    manifest_path = root / "freeze_f3_cpu/manifest.json"
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Pinned F3 manifest provenance mismatch")
    manifest = json.loads(manifest_path.read_text())
    hashes = {name: verify_file(root, manifest, name) for name in METADATA}
    windows, labels, retained, classes = [
        pq.read_table(root / name).to_pylist() for name in METADATA
    ]
    rows = checked_training_rows(windows, labels, retained)
    classids = sorted(r["class_idx"] for r in classes if r["admitted"])
    if len(classids) != 188:
        raise ValueError("fixed inventory mismatch")
    index = {c: i for i, c in enumerate(classids)}
    groups = {}
    blocks = {}
    for j, i in enumerate(rows):
        w = windows[i]
        cell = w["deg_cell"]
        block = (
            int(np.floor((w["lat"] + 90) / 5)),
            int(np.floor((w["lon"] + 180) / 5)),
        )
        if cell in blocks and blocks[cell] != block:
            raise ValueError("cell crosses registered block")
        blocks[cell] = block
        for c in labels[i]["labels_idx"]:
            if c in index:
                groups.setdefault((index[c], cell), []).append(j)
    support = []
    for c in range(188):
        cells = [cell for ci, cell in groups if ci == c]
        unique = set(blocks[x] for x in cells)
        support.append(
            {
                "class_idx": classids[c],
                "cells": len(cells),
                "blocks": len(unique),
                "half_cells": [
                    sum(
                        half(blocks[x], cfg["diagnostic_split_seed"]) == h
                        for x in cells
                    )
                    for h in [0, 1]
                ],
            }
        )
    result = {
        "author": "James Edward Ball",
        "scope": "retained training rows only; no language arrays or test feature values",
        "empirical_interval_admitted": False,
        "manifest_sha256": digest(manifest_path),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "config_sha256": digest(HERE / "CONFIG.json"),
        "implementation_sha256": digest(__file__),
        "training_rows": len(rows),
        "training_row_ids_sha256": hashlib.sha256(
            "\n".join(windows[i]["window_id"] for i in rows).encode()
        ).hexdigest(),
        "class_support": support,
        "features": {},
        "input_sha256": hashes,
    }
    for name, relative in FEATURES.items():
        hashes[relative] = verify_file(root, manifest, relative)
        stored = np.load(root / relative, mmap_mode="r", allow_pickle=False)
        if stored.ndim != 2 or stored.shape[0] != len(windows):
            raise ValueError("feature shape mismatch")
        x = np.asarray(stored[rows], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite training feature")
        norms = np.linalg.norm(x, axis=1)
        centred = x - x.mean(0)
        cov = centred.T @ centred / (len(x) - 1)
        eigen = np.linalg.eigvalsh(cov)
        if eigen.min() < -1e-8 * max(1, eigen.max()):
            raise ValueError("invalid covariance")
        eigen = np.maximum(eigen, 0)
        trace = eigen.sum()
        probs = eigen[eigen > 0] / trace
        perclass = []
        for c in range(188):
            cells = [cell for ci, cell in groups if ci == c]
            means = np.stack([x[groups[(c, cell)]].mean(0) for cell in cells])
            midpoint = means.mean(0)
            h = np.array(
                [half(blocks[cell], cfg["diagnostic_split_seed"]) for cell in cells]
            )
            row = {
                "class_idx": classids[c],
                "mean_norm_squared": float(midpoint @ midpoint),
                "cell_variance_trace": (
                    float(np.sum((means - midpoint) ** 2) / (len(means) - 1))
                    if len(means) > 1
                    else None
                ),
                "split_half_available": bool(np.any(h == 0) and np.any(h == 1)),
            }
            if row["split_half_available"]:
                a, b = means[h == 0].mean(0), means[h == 1].mean(0)
                den = np.linalg.norm(a) * np.linalg.norm(b)
                row.update(
                    split_half_squared_difference=float(np.sum((a - b) ** 2)),
                    split_half_cosine=float(a @ b / den) if den > 0 else None,
                )
            perclass.append(row)
        result["features"][name] = {
            "stored_shape": list(stored.shape),
            "diagnosed_shape": list(x.shape),
            "norm": describe(norms),
            "covariance_trace": float(trace),
            "covariance_participation_rank": float(trace**2 / np.sum(eigen**2)),
            "covariance_entropy_rank": float(np.exp(-np.sum(probs * np.log(probs)))),
            "largest_eigenvalue_fraction": float(eigen.max() / trace),
            "class_diagnostics": perclass,
            "split_half_missing_classes": sum(
                not r["split_half_available"] for r in perclass
            ),
        }
    result["elapsed_seconds"] = time.monotonic() - start
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    value = run(args.root, args.manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "training_rows": value["training_rows"],
                "features": list(value["features"]),
                "elapsed_seconds": value["elapsed_seconds"],
                "empirical_interval_admitted": False,
            }
        )
    )
