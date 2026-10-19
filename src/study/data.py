"""Download pinned assets and merge original annotations with frozen study splits.

Author: James Edward Ball. No labels are inferred from model predictions.
"""

import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def download(url, path, expected):
    """Stream to a temporary file; verify before admitting or reusing it."""
    import requests

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Corrupt cached asset: {path}")
        return path
    temp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temp.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
    if digest(temp) != expected:
        raise ValueError(f"Download hash mismatch: {path.name}")
    temp.replace(path)
    return path


def safe_extract(archive, destination):
    """Reject paths outside the extraction root and ZIP symlinks."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe ZIP member")
        z.extractall(root)


def consensus(stack):
    stack = np.asarray(stack)
    if stack.ndim != 3 or not np.isin(stack, np.arange(11)).all():
        raise ValueError("Invalid expert rasters")
    if len(stack) == 1:
        return stack[0].copy()
    votes = np.stack([(stack == c).sum(0) for c in range(1, 11)])
    return np.where(votes.max(0) >= 2, votes.argmax(0) + 1, 0).astype(np.uint8)


def coverage(patch):
    counts = np.bincount(np.asarray(patch).ravel(), minlength=11)
    lower = counts[1:10] / patch.size
    return lower, np.minimum(1, lower + (counts[0] + counts[10]) / patch.size)


def merge_labels(work, limit=None):
    """Recount raw expert pixels, then attach the declared roles by patch ID."""
    import rasterio

    work = Path(work)
    scenes = read(PACKAGE / "data/scenes.json")
    bundle = read(PACKAGE / "data/bundle.json")
    for asset in read(PACKAGE / "data/downloads.json"):
        p = download(asset["url"], work / "downloads" / asset["name"], asset["sha256"])
        safe_extract(p, work / "labels")
    rows = scenes["patches"][:limit]
    parents = {}
    merged = []
    physical = [
        "Water",
        "Trees",
        "Grass",
        "Flooded_Vegetation",
        "Crops",
        "Scrub",
        "Built_Area",
        "Bare_Ground",
        "Snow_Ice",
    ]
    for row in rows:
        parent = row["parent_id"]
        if parent not in parents:
            images = []
            for source in scenes["parents"][parent]["labels"]:
                path = work / "labels" / source["path"]
                if digest(path) != source["sha256"]:
                    raise ValueError("Source label mismatch")
                with rasterio.open(path) as ds:
                    images.append(ds.read(1))
            parents[parent] = consensus(np.stack(images))
        r, c = map(int, row["patch_id"].split("__r")[1].split("c"))
        y, x = 15 + 120 * r, 15 + 120 * c
        patch = parents[parent][y : y + 120, x : x + 120]
        lower, upper = coverage(patch)
        positive = [c for c, v in zip(physical, lower) if v >= 0.3 and c in bundle["classes"]]
        if set(positive) != set(row["positive"]):
            raise ValueError(f"Label drift: {row['patch_id']}")
        merged.append(
            {
                **{k: row[k] for k in ["patch_id", "parent_id", "component", "role", "positive"]},
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "unknown": float((patch == 0).mean()),
                "cloud": float((patch == 10).mean()),
                "source": "single expert"
                if len(scenes["parents"][parent]["labels"]) == 1
                else "expert agreement",
            }
        )
    assert len({r["patch_id"] for r in merged}) == len(merged)
    for parent in parents:
        if len({r["role"] for r in merged if r["parent_id"] == parent}) != 1:
            raise ValueError("Parent split leakage")
    write(work / "merged-labels.json", merged)
    return merged


def acquire(work, limit=None):
    """Download exact pinned Sentinel-2 windows, preserving original radiometry."""
    import planetary_computer
    import rasterio
    from affine import Affine
    from rasterio.vrt import WarpedVRT

    from geofm_jlens.semantic_s2_inputs_v1 import BANDS, common_grid

    work = Path(work)
    scenes = read(PACKAGE / "data/scenes.json")
    out = work / "imagery"
    out.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(scenes["patches"][:limit]):
        path = out / (row["patch_id"] + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, h in row["array_sha256"].items():
                    if array_digest(z[k]) != h:
                        raise ValueError(f"Corrupt image cache: {path}")
            continue
        items = scenes["parents"][row["parent_id"]]["items"]
        raw = {}
        for item in items:
            download(
                planetary_computer.sign(item["assets"]["product-metadata"]),
                work / "metadata" / (item["id"] + ".xml"),
                item["metadata_sha256"],
            )
            if float(item["properties"]["s2:processing_baseline"]) >= 4:
                raise ValueError("Unsupported radiometry")
        for band in (*BANDS, "SCL"):
            res = 10 if band in BANDS[:3] else 20
            size = 1200 // res
            t = row["transform"]
            tr = Affine(res, 0, t[2], 0, -res, t[5])
            a = np.zeros((size, size), np.uint16)
            for item in items:
                href = planetary_computer.sign(item["assets"][band])
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60"):
                    with rasterio.open(href) as source:
                        with WarpedVRT(
                            source,
                            crs=row["crs"],
                            transform=tr,
                            width=size,
                            height=size,
                            resampling=rasterio.enums.Resampling.nearest,
                            nodata=0,
                        ) as vrt:
                            b = vrt.read(1)
                use = (a == 0) & (b != 0)
                a[use] = b[use]
            raw[band] = a
        if (
            any((raw[b] == 0).any() or (raw[b] == 65535).any() for b in BANDS)
            or np.isin(raw["SCL"], [0, 1, 3, 8, 9, 10, 11]).mean() > 0.1
        ):
            raise ValueError("Source image no longer passes quality gate")
        native, resized = common_grid({b: raw[b] for b in BANDS}, {b: raw[b] != 0 for b in BANDS})
        arrays = {"native20": native, "input64": resized, "scl": raw["SCL"]}
        for k, h in row["array_sha256"].items():
            if array_digest(arrays[k]) != h:
                raise ValueError(f"Source pixel drift: {row['patch_id']} {k}")
        temp = path.with_suffix(".part.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
        print(f"Images {number + 1}/{len(scenes['patches'][:limit])}", flush=True)


def cached_assets(work):
    """Private release requires an authenticated gh session with repository access."""
    from .releases import repository
    spec = read(PACKAGE / "data/cache.json")
    out = Path(work) / "cache"
    out.mkdir(parents=True, exist_ok=True)
    for item in spec["files"]:
        path = out / item["name"]
        if not path.exists():
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    spec["tag"],
                    "--repo",
                    repository(),
                    "--pattern",
                    item["name"],
                    "--dir",
                    str(out),
                ],
                check=True,
            )
        if digest(path) != item["sha256"]:
            raise ValueError(f"Cached analysis asset mismatch: {path}")
