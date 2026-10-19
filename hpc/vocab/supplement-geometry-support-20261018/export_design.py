"""Export outcome-blind retained incidence. Author: James Edward Ball."""

import hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    src = ROOT / "output/megaearth-review-tables"
    names = [
        "assemble/windows_qa.parquet",
        "labels_exact/labels.parquet",
        "admit/windows_final.parquet",
        "admit/classes_final.parquet",
        "families/families.parquet",
    ]
    w, l, r, c, f = [pd.read_parquet(src / n) for n in names]
    w = w.merge(
        r[["widx", "window_id", "retained"]],
        on=["widx", "window_id"],
        validate="one_to_one",
    ).merge(
        l[["widx", "window_id", "labels_idx"]],
        on=["widx", "window_id"],
        validate="one_to_one",
        suffixes=("_old", ""),
    )
    w = w[w.retained].copy()
    classes = c[c.admitted].sort_values("class_idx")
    ids = classes.class_idx.tolist()
    mapping = {v: i for i, v in enumerate(ids)}
    families = dict(zip(f.class_idx, f.family_id))
    cells = []
    for (split, cell), x in w.groupby(["split", "deg_cell"], sort=True):
        geos = {
            (int(np.floor((lat + 90) / 10)), int(np.floor((lon + 180) / 10)))
            for lat, lon in zip(x.lat, x.lon)
        }
        blocks = {
            (int(np.floor((lat + 90) / 5)), int(np.floor((lon + 180) / 5)))
            for lat, lon in zip(x.lat, x.lon)
        }
        assert len(geos) == len(blocks) == 1
        count = Counter(
            mapping[int(k)] for labs in x.labels_idx for k in labs if int(k) in mapping
        )
        cells.append(
            {
                "id": cell,
                "split": split,
                "geo": list(geos.pop()),
                "block": list(blocks.pop()),
                "counts": sorted(count.items()),
                "windows": len(x),
                "window_labels": [
                    [mapping[int(k)] for k in labs if int(k) in mapping]
                    for labs in x.labels_idx
                ],
            }
        )
    assert len({x["id"] for x in cells}) == len(cells)
    result = {
        "author": "James Edward Ball",
        "scope": "Conditional retained-cell template distribution; no image or text effects used",
        "source_sha256": {
            n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in names
        },
        "classes": [
            {
                "class_idx": int(i),
                "class_id": str(c.loc[c.class_idx == i, "class_id"].iloc[0]),
                "family_id": int(families[i]),
            }
            for i in ids
        ],
        "cells": cells,
        "admission": {"train": 30, "val": 5, "test": 10, "test_cells": 5, "cells": 8},
        "cap_per_target_class": 150,
        "geo_grid_deg": 10,
        "block_grid_deg": 5,
    }
    (HERE / "RETAINED-DESIGN.json").write_text(
        json.dumps(result, separators=(",", ":")) + "\n"
    )
    for split in ["train", "val", "test"]:
        rows = [x for x in cells if x["split"] == split]
        support = np.bincount(
            [c for x in rows for c, n in x["counts"]], minlength=len(ids)
        )
        print(
            split,
            len(rows),
            sum(sum(n for c, n in x["counts"]) for x in rows),
            "support",
            support.min(),
            np.median(support),
            support.max(),
        )


if __name__ == "__main__":
    main()
