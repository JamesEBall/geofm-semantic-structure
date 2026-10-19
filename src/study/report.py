"""Export every geometry comparison and transfer cell, plus a compact result plot."""

import csv

import numpy as np

from .data import read, write


def export(work):
    base = work / "results/semantic-structure"
    out = work / "outputs"
    out.mkdir(exist_ok=True)
    geometry = read(base / "dynamicworld-main-geometry-01/report.json")
    readout = read(base / "dynamicworld-main-readout-01/report.json")
    rows = []
    for rec in geometry["records"]:
        for ref, value in rec.get("references", {}).items():
            rows.append(
                {
                    "feature": rec["feature"],
                    "route": rec["route"],
                    "reference": ref,
                    "rsa": value.get("rsa"),
                    "status": value["status"],
                }
            )

    def csvwrite(name, rows):
        with (out / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    csvwrite("geometry.csv", rows)
    cells = []
    for r in readout["records"]:
        cells.append(
            {
                k: r[k]
                for k in [
                    "intended_sites",
                    "seed",
                    "reference",
                    "feature",
                    "control",
                    "cell",
                    "status",
                ]
            }
            | {"component_map": r.get("summary", {}).get("component_map")}
        )
    # The accepted function names its component mean in the nested AP summary.
    for row, r in zip(cells, readout["records"]):
        if r["status"] == "available":
            row["component_map"] = r["summary"]["ap"]["component_mean"]
    csvwrite("transfer.csv", cells)
    write(
        out / "primary-results.json",
        {
            "comparisons": readout["primary_comparisons"],
            "family": readout["primary_family"],
            "validation_max_error": readout["validation_parity_max_error"],
            "limits": readout["limits"],
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    names = ["clay", "prithvi", "appearance"]
    for i, name in enumerate(names):
        geo = next(
            r
            for r in rows
            if r["feature"] == name
            and r["reference"] == "qwen_instructed"
            and r["route"] == ("fit_standardised" if name == "appearance" else "native")
        )
        axs[0].bar(i, geo["rsa"], color="#61776f")
        vals = [
            r["component_map"]
            for r in cells
            if r["feature"] == name
            and r["reference"] == "qwen_instructed"
            and r["control"] == "paired"
            and r["intended_sites"] == 512
            and r["cell"] == "heldout_test"
        ]
        axs[1].bar(i, np.mean(vals), color="#61776f")
    for a, title in zip(axs, ["Category-distance agreement", "Joint-transfer AP"]):
        a.set_xticks(range(3), ["Clay", "Prithvi", "Appearance"])
        a.set_title(title)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "main-results.pdf", metadata={"Author": "James Edward Ball"})
    plt.close(fig)
    print("Results:", out, flush=True)
