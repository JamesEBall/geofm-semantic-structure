#!/usr/bin/env python3
"""OSM tooling smoke test on a small extract (Slurm only; no network).

1. pyosmium (venv): FileProcessor(...).with_locations('flex_mem')
   .with_areas(KeyFilter('landuse', 'natural')).with_filter(EntityFilter(AREA)); counts
   areas, times it, and checks WKBFactory().create_multipolygon(area) gives valid hex WKB.
2. osmium-tool (env/osm): --version, fileinfo -e, tags-filter, extract -b / -c (smart).
Writes a JSON report; exits 1 if any required check fails.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HEX = re.compile(r"[0-9A-Fa-f]+")


def pyosmium_check(pbf: str) -> dict:
    import osmium
    import osmium.version as ov
    import shapely

    fab = osmium.geom.WKBFactory()
    n = 0
    by_key = collections.Counter()
    from_way = 0
    hex_ok = valid = invalid = errors = 0
    first_error = None
    t0 = time.perf_counter()
    fp = (osmium.FileProcessor(pbf)
          .with_locations("flex_mem")
          .with_areas(osmium.filter.KeyFilter("landuse", "natural"))
          .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA)))
    for area in fp:
        n += 1
        tags = area.tags
        key = "landuse" if "landuse" in tags else "natural" if "natural" in tags else "other"
        by_key[key] += 1
        from_way += int(area.from_way())
        try:
            wkb = fab.create_multipolygon(area)
        except Exception as e:  # noqa: BLE001
            errors += 1
            first_error = first_error or f"{type(e).__name__}: {e}"
            continue
        if isinstance(wkb, str) and HEX.fullmatch(wkb):
            hex_ok += 1
            geom = shapely.from_wkb(wkb)
            if geom.geom_type == "MultiPolygon" and geom.is_valid and not geom.is_empty:
                valid += 1
            else:
                invalid += 1
    secs = time.perf_counter() - t0

    # with_areas() filters only select *relation* candidates; every closed way still becomes
    # an area, so a post-assembly KeyFilter is needed to keep only landuse/natural areas.
    t1 = time.perf_counter()
    n_kf = sum(1 for _ in (osmium.FileProcessor(pbf)
                           .with_locations("flex_mem")
                           .with_areas(osmium.filter.KeyFilter("landuse", "natural"))
                           .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
                           .with_filter(osmium.filter.KeyFilter("landuse", "natural"))))
    secs_kf = time.perf_counter() - t1
    return {
        "pyosmium": ov.pyosmium_release, "libosmium": ov.libosmium_version,
        "chain": "FileProcessor(pbf).with_locations('flex_mem').with_areas(KeyFilter('landuse','natural'))"
                 ".with_filter(EntityFilter(AREA))",
        "areas": n, "areas_by_key": dict(by_key), "areas_from_ways": from_way,
        "areas_from_relations": n - from_way, "seconds": round(secs, 2),
        "wkb_hex_strings": hex_ok, "wkb_valid_multipolygons": valid, "wkb_invalid": invalid,
        "wkb_errors": errors, "first_wkb_error": first_error,
        "chain_plus_keyfilter": "... .with_filter(EntityFilter(AREA)).with_filter(KeyFilter('landuse','natural'))",
        "chain_plus_keyfilter_areas": n_kf, "chain_plus_keyfilter_seconds": round(secs_kf, 2),
        "note": "with_areas(filters) only restricts relation candidates; all closed ways become areas "
                "(areas_by_key.other). Add a post-assembly KeyFilter.",
        "ok": n > 0 and hex_ok == n - errors and valid > 0.99 * n
              and n_kf == by_key["landuse"] + by_key["natural"],
    }


def run(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)),
            "rc": p.returncode, "seconds": round(time.perf_counter() - t0, 2),
            "stdout_tail": p.stdout[-400:], "stderr_tail": p.stderr[-400:]}


def fileinfo(osmium: str, pbf: Path) -> dict:
    p = subprocess.run([osmium, "fileinfo", "-e", "-j", str(pbf)], capture_output=True, text=True)
    if p.returncode:
        return {"rc": p.returncode, "stderr": p.stderr[-300:]}
    j = json.loads(p.stdout)
    data = j.get("data", {})
    return {"bytes": pbf.stat().st_size, "count": data.get("count"),
            "timestamp": data.get("timestamp"),
            "replication_timestamp": j.get("header", {}).get("option", {}).get("osmosis_replication_timestamp")}


def osmium_tool_check(osmium: str, pbf: str, work: Path) -> dict:
    out: dict = {"binary": osmium}
    v = run([osmium, "--version"])
    out["version"] = v["stdout_tail"].splitlines()[0] if v["stdout_tail"] else v["stderr_tail"]
    t0 = time.perf_counter()
    out["input_fileinfo"] = fileinfo(osmium, Path(pbf))
    out["input_fileinfo"]["seconds"] = round(time.perf_counter() - t0, 2)

    tf = work / "tags-filter.osm.pbf"
    out["tags_filter"] = run([osmium, "tags-filter", pbf, "nwr/landuse", "nwr/natural", "-o", str(tf), "-O"])
    if tf.exists():
        out["tags_filter"]["output"] = fileinfo(osmium, tf)
    tfa = work / "tags-filter-areas.osm.pbf"
    out["tags_filter_areas"] = run([osmium, "tags-filter", pbf, "a/landuse", "a/natural", "-o", str(tfa), "-O"])
    if tfa.exists():
        out["tags_filter_areas"]["output"] = fileinfo(osmium, tfa)

    ex = work / "extract-bbox.osm.pbf"
    out["extract_bbox_smart"] = run([osmium, "extract", "-b", "5.9,49.4,6.3,49.9", "--strategy", "smart",
                                     pbf, "-o", str(ex), "-O"])
    if ex.exists():
        out["extract_bbox_smart"]["output"] = fileinfo(osmium, ex)

    cfg = {"directory": str(work), "extracts": [
        {"output": "multi-north.osm.pbf", "bbox": [5.7, 49.75, 6.6, 50.2]},
        {"output": "multi-south.osm.pbf", "bbox": [5.7, 49.4, 6.6, 49.75]},
        {"output": "multi-west.osm.pbf", "bbox": [5.7, 49.4, 6.0, 50.2]},
    ]}
    cfg_path = work / "extracts.json"
    cfg_path.write_text(json.dumps(cfg))
    out["extract_multi_smart"] = run([osmium, "extract", "-c", str(cfg_path), "--strategy", "smart", pbf, "-O"])
    out["extract_multi_smart"]["outputs"] = {e["output"]: fileinfo(osmium, work / e["output"])
                                             for e in cfg["extracts"] if (work / e["output"]).exists()}
    py = Path(osmium).parent / "python"
    if py.exists():
        p = subprocess.run([str(py), "-c", "import osmium.version as v; print(v.pyosmium_release, v.libosmium_version)"],
                           capture_output=True, text=True)
        out["osm_env_pyosmium"] = p.stdout.strip() or p.stderr[-300:]
    out["ok"] = all(out[k]["rc"] == 0 for k in ("tags_filter", "tags_filter_areas", "extract_bbox_smart", "extract_multi_smart"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--osmium", default="")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report: dict = {"pbf": a.pbf, "pbf_bytes": os.path.getsize(a.pbf), "host": os.uname().nodename,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        report["pyosmium"] = pyosmium_check(a.pbf)
    except Exception as e:  # noqa: BLE001
        report["pyosmium"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(report["pyosmium"], indent=1), flush=True)
    if a.osmium and os.access(a.osmium, os.X_OK):
        report["osmium_tool"] = osmium_tool_check(a.osmium, a.pbf, work)
    else:
        report["osmium_tool"] = {"ok": False, "error": f"osmium-tool binary not found: {a.osmium!r}"}
    print(json.dumps(report["osmium_tool"], indent=1), flush=True)
    report["ok"] = bool(report["pyosmium"].get("ok") and report["osmium_tool"].get("ok"))
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"osmium_smoke: {'OK' if report['ok'] else 'FAIL'} -> {a.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
