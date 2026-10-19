"""Plot retained MegaEarth support from release-verified tables.

Author: James Edward Ball
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "assemble/windows_qa.parquet", "admit/windows_final.parquet",
    "admit/classes_final.parquet", "labels_exact/labels.parquet",
    "families/families.parquet",
)


def load_support(tables, manifest):
    expected = json.loads(manifest.read_text())["files"]
    hashes = {}
    for rel in TABLES:
        digest = hashlib.sha256((tables / rel).read_bytes()).hexdigest()
        if digest != expected["tables/" + rel]:
            raise ValueError(f"Release checksum mismatch: {rel}")
        hashes[rel] = digest
    frames = {rel: pd.read_parquet(tables / rel) for rel in TABLES}
    windows = frames[TABLES[0]].drop(columns="labels_idx").merge(
        frames[TABLES[1]][["widx", "window_id", "retained"]],
        on=["widx", "window_id"], validate="one_to_one", how="inner",
    ).merge(
        frames[TABLES[3]][["widx", "window_id", "labels_idx"]],
        on=["widx", "window_id"], validate="one_to_one", how="inner",
    )
    if len(windows) != 44355:
        raise ValueError("Incomplete window identity join")
    windows = windows.loc[windows.retained].copy()
    counts = windows.groupby("split").size().to_dict()
    if counts != {"train": 24574, "val": 3539, "test": 6879}:
        raise ValueError(f"Retained split mismatch: {counts}")
    classes = frames[TABLES[2]].query("admitted").copy()
    families = frames[TABLES[4]]
    if len(classes) != 188 or families.loc[
        families.class_idx.isin(classes.class_idx), "family_id"
    ].nunique() != 91:
        raise ValueError("Admitted vocabulary mismatch")
    positives = windows.explode("labels_idx")
    positives = positives.loc[positives.labels_idx.isin(classes.class_idx)]
    test = positives.loc[positives.split == "test"]
    if (len(test), test.widx.nunique(), test.deg_cell.nunique()) != (8755, 5942, 1718):
        raise ValueError("Transfer support mismatch")
    support = positives.groupby(["labels_idx", "split"]).deg_cell.nunique().unstack()
    support = support.join(classes.set_index("class_idx")[["class_id"]])
    support = support.sort_values(["test", "class_id"], ascending=[False, True])
    return windows, support, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "hpc/vocab/results/full-20261018/release-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/figures/megaearth-support.pdf")
    args = parser.parse_args()
    windows, support, hashes = load_support(args.tables, args.manifest)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .55, "axes.edgecolor": "#777777",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "text.color": "#252525", "axes.labelcolor": "#252525",
        "axes.titleweight": "normal", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.65), gridspec_kw={"width_ratios": [1.4, 1]})
    roles = ("train", "val", "test")
    colours = ("#b9bdb9", "#a7a49a", "#536f66")
    labels = ("Training", "Validation", "Test")
    ax = axes[0]
    land = json.loads((ROOT / "data/geography/ne_110m_land.geojson").read_text())
    for feature in land["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            for ring in polygon:
                xy = np.asarray(ring)
                ax.plot(xy[:, 0], xy[:, 1], color="#a5aaa5", lw=.4, zorder=0)
    for role, label, colour in zip(roles, labels, colours):
        rows = windows.loc[windows.split == role]
        ax.scatter(rows.lon, rows.lat, s=1.5, color=colour, alpha=.65,
                   linewidths=0, rasterized=True, label=f"{label} ({len(rows):,})")
    ax.set(xlim=(-180, 180), ylim=(-85, 85), xlabel="Longitude", ylabel="Latitude", title="A  Retained window centres")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, 0, 60])
    ax.grid(lw=.3, alpha=.25)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.51, .96), ncol=3,
               frameon=False, markerscale=4, handletextpad=.3, columnspacing=1.5)
    ax = axes[1]
    x = np.arange(1, len(support) + 1)
    # Every category is shown; one shared order prevents split-wise resorting.
    for role, colour in zip(roles, colours):
        ax.scatter(x, support[role], s=5, color=colour, linewidths=0, alpha=.85)
    ax.set(yscale="log", xlim=(0, 189), ylim=(.8, 1500),
           xlabel="Category rank by test support", ylabel="Geographic cells per category",
           title="B  Support across 188 categories")
    ax.set_xticks([1, 50, 100, 150, 188])
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.grid(axis="y", lw=.3, alpha=.3)
    fig.subplots_adjust(left=.075, right=.965, bottom=.16, top=.76, wspace=.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Author": "James Edward Ball", "CreationDate": None, "ModDate": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    support.to_csv(args.output.with_suffix(".csv"), index_label="class_idx")
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "author": "James Edward Ball", "cohort": "retained", "source_sha256": hashes,
        "retained_windows": 34992, "test_pairs": 8755, "scored_test_windows": 5942,
        "positive_test_cells": 1718, "classes": 188, "families": 91,
        "test_cells_per_class_min": int(support.test.min()),
        "test_cells_per_class_max": int(support.test.max()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
