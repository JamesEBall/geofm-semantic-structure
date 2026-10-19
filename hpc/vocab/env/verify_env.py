#!/usr/bin/env python3
"""Import test and manifest for the vocab venv (env/venv layered on fresh-01).

Run inside Slurm by slurm/env_build.sbatch after `source env/activate.sh`. Writes, in the
env dir: manifest.json, lock.txt (composite `name==version`), manifest.sha256 (sha256 of
every *.dist-info/RECORD on the path), wheels.sha256. Exits 1 on any failed check.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import glob
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import socket
import sys
import sysconfig
import time
from pathlib import Path

MODULES = [
    "torch", "numpy", "scipy", "sklearn", "rasterio", "shapely", "pyproj", "pyarrow",
    "pyarrow.parquet", "pyarrow.dataset", "pyarrow.compute",
    "pandas", "osmium", "requests", "socks", "transformers", "yaml", "zstandard", "pytest",
    "hypothesis", "shapefile", "PIL", "urllib3", "timm", "huggingface_hub", "safetensors", "open_clip",
]
EXPECT = {"numpy": "1.26.4", "torch": "2.6.0+cu124"}
ALLOWED_OVERLAP = {"pip"}  # the new venv's own pip (same version as fresh-01's)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dists_in(path: str) -> dict[str, md.Distribution]:
    return {norm(d.metadata["Name"]): d for d in md.distributions(path=[path])}


def origin(file: str | None, venv_sp: str, fresh_sp: str) -> str:
    if not file:
        return "builtin"
    if file.startswith(venv_sp):
        return "venv"
    if file.startswith(fresh_sp):
        return "fresh-01"
    return "base"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--fresh-venv", required=True)
    ap.add_argument("--wheelhouse", required=True)
    args = ap.parse_args()
    env_dir = Path(args.env_dir)
    fresh = Path(args.fresh_venv)
    venv_sp = sysconfig.get_paths()["purelib"]
    fresh_sp = str(fresh / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    failures: list[str] = []
    checks: dict[str, object] = {}

    # --- interpreter / layering -------------------------------------------------------
    cfg = {}
    for line in (Path(sys.prefix) / "pyvenv.cfg").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    fresh_cfg = {}
    for line in (fresh / "pyvenv.cfg").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fresh_cfg[k.strip()] = v.strip()
    if cfg.get("home") != fresh_cfg.get("home") or cfg.get("version") != fresh_cfg.get("version"):
        failures.append(f"base interpreter differs from fresh-01: {cfg} vs {fresh_cfg}")
    pth = sorted(glob.glob(os.path.join(venv_sp, "*.pth")))
    try:
        i_venv, i_fresh = sys.path.index(venv_sp), sys.path.index(fresh_sp)
        checks["sys_path_order_ok"] = i_venv < i_fresh
    except ValueError as e:
        checks["sys_path_order_ok"] = False
        failures.append(f"sys.path missing site dir: {e}")
    if not checks["sys_path_order_ok"]:
        failures.append("venv site-packages must precede fresh-01 site-packages")
    checks["no_user_site"] = bool(sys.flags.no_user_site)
    cache_keys = (
        "VOCAB_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "PIP_CACHE_DIR", "TORCH_HOME",
        "XDG_CACHE_HOME", "MPLCONFIGDIR", "MAMBA_ROOT_PREFIX", "TMPDIR", "CPL_TMPDIR",
        "PYTHONPYCACHEPREFIX", "HYPOTHESIS_STORAGE_DIRECTORY",
    )
    home = Path.home().resolve()
    checks["cache_paths"] = {key: os.environ.get(key) for key in cache_keys}
    for key, value in checks["cache_paths"].items():
        if not value or Path(value).resolve().is_relative_to(home):
            failures.append(f"cache/temp path is unset or within home: {key}={value!r}")
    if not checks["no_user_site"]:
        failures.append("user site-packages are enabled")

    venv_d, fresh_d = dists_in(venv_sp), dists_in(fresh_sp)
    overlap = sorted(set(venv_d) & set(fresh_d) - ALLOWED_OVERLAP)
    checks["venv_only_packages"] = sorted(f"{venv_d[n].metadata['Name']}=={venv_d[n].version}" for n in set(venv_d) - set(fresh_d))
    checks["shadowed_fresh01_packages"] = overlap
    if overlap:
        failures.append(f"venv shadows fresh-01 packages: {overlap}")

    # --- imports ----------------------------------------------------------------------
    imports = {}
    for name in MODULES:
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", None)
            if name == "osmium":
                import osmium.version as ov
                ver = f"{ov.pyosmium_release} (libosmium {ov.libosmium_version})"
            imports[name] = {"ok": True, "version": str(ver), "origin": origin(getattr(mod, "__file__", None), venv_sp, fresh_sp),
                             "seconds": round(time.perf_counter() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            imports[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            failures.append(f"import {name}: {e}")
        print(f"import {name:16s} {imports[name]}", flush=True)
    for name, want in EXPECT.items():
        got = imports.get(name, {}).get("version")
        if got != want:
            failures.append(f"{name} version {got} != {want}")
        if imports.get(name, {}).get("origin") != "fresh-01":
            failures.append(f"{name} not loaded from fresh-01")

    # --- GDAL / PROJ / libcurl --------------------------------------------------------
    import numpy as np
    checks["numpy_runtime"] = {"__version__": np.__version__, "dist_version": md.version("numpy"),
                               "file": np.__file__, "origin": origin(np.__file__, venv_sp, fresh_sp)}
    print("numpy runtime", checks["numpy_runtime"], flush=True)
    import pyproj
    import rasterio
    with rasterio.Env() as env:
        drivers = env.drivers()
    curl = None
    libs = glob.glob(os.path.join(os.path.dirname(rasterio.__file__) + ".libs", "libcurl*.so*"))
    if libs:
        lib = ctypes.CDLL(libs[0])
        lib.curl_version.restype = ctypes.c_char_p
        curl = lib.curl_version().decode()
    tr = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
    x, y = tr.transform(-3.19, 55.95)
    gdal = {"gdal_version": rasterio.__gdal_version__, "proj_version": pyproj.proj_version_str,
            "rasterio_proj_version": getattr(rasterio, "__proj_version__", None),
            "GTiff": "GTiff" in drivers, "COG": "COG" in drivers, "libcurl": curl,
            "proj_data_dir": pyproj.datadir.get_data_dir(), "utm30n_edinburgh": [round(x, 1), round(y, 1)],
            "rasterio_epsg_32630": rasterio.crs.CRS.from_epsg(32630).to_epsg()}
    if not (gdal["GTiff"] and gdal["COG"]):
        failures.append("GDAL lacks GTiff/COG driver")
    print("gdal", gdal, flush=True)

    # --- functional smoke tests (no network) -------------------------------------------
    work = env_dir / "selftest"
    work.mkdir(exist_ok=True)
    smoke = {}
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = pd.DataFrame({"a": np.arange(1000, dtype=np.int64), "b": np.random.default_rng(0).random(1000),
                           "c": ["x"] * 1000, "d": np.arange(1000, dtype=np.uint16)})
        p = work / "t.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, compression="zstd")
        back = pq.read_table(p).to_pandas()
        smoke["parquet_zstd_roundtrip"] = bool(back.equals(df)) and pq.ParquetFile(p).metadata.row_group(0).column(0).compression == "ZSTD"
        src = np.arange(10.0)
        arr = pa.array(src)
        smoke["pyarrow_numpy_roundtrip"] = bool(np.array_equal(arr.to_numpy(zero_copy_only=True), src))
        import pyarrow.dataset as pads
        smoke["pyarrow_dataset_scan"] = pads.dataset(str(p), format="parquet").count_rows() == 1000
    except Exception as e:  # noqa: BLE001
        smoke["parquet_zstd_roundtrip"] = f"FAIL {type(e).__name__}: {e}"
    try:
        import osmium
        import shapely
        xml = work / "t.osm"
        xml.write_text(
            "<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6' generator='t'>\n"
            "<node id='1' version='1' lat='0.0' lon='0.0'/><node id='2' version='1' lat='0.0' lon='0.01'/>"
            "<node id='3' version='1' lat='0.01' lon='0.01'/><node id='4' version='1' lat='0.01' lon='0.0'/>\n"
            "<way id='10' version='1'><nd ref='1'/><nd ref='2'/><nd ref='3'/><nd ref='4'/><nd ref='1'/>"
            "<tag k='landuse' v='farmland'/></way>\n</osm>\n")
        fab = osmium.geom.WKBFactory()
        areas = [shapely.from_wkb(bytes.fromhex(fab.create_multipolygon(o)))
                 for o in osmium.FileProcessor(str(xml)).with_areas().with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))]
        smoke["osmium_area_from_xml"] = len(areas) == 1 and areas[0].is_valid and areas[0].area > 0
    except Exception as e:  # noqa: BLE001
        smoke["osmium_area_from_xml"] = f"FAIL {e}"
    try:
        import zstandard
        blob = os.urandom(1000) * 10
        smoke["zstandard_roundtrip"] = zstandard.ZstdDecompressor().decompress(zstandard.ZstdCompressor(level=3).compress(blob)) == blob
        import yaml
        smoke["yaml"] = yaml.safe_load("a: [1, 2]") == {"a": [1, 2]}
        import shapefile
        w = shapefile.Writer(str(work / "t"), shapeType=shapefile.POLYGON)
        w.field("name", "C")
        w.poly([[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]])
        w.record("sq")
        w.close()
        r = shapefile.Reader(str(work / "t"))
        smoke["pyshp_roundtrip"] = len(r) == 1 and r.record(0)["name"] == "sq"
        r.close()
        from PIL import Image
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(work / "t.png")
        smoke["pillow_png"] = Image.open(work / "t.png").size == (8, 8)
    except Exception as e:  # noqa: BLE001
        smoke["misc"] = f"FAIL {e}"
    import torch
    checks["torch_cuda_build"] = torch.version.cuda  # informational: CPU nodes have no GPU
    checks["torch_cuda_available_here"] = torch.cuda.is_available()
    for k, v in smoke.items():
        if v is False or (isinstance(v, str) and v.startswith("FAIL")):
            failures.append(f"smoke {k}: {v}")
    print("smoke", smoke, flush=True)

    # --- lock / hashes / manifest ---------------------------------------------------------
    composite = {}
    for path, tag in ((fresh_sp, "fresh-01"), (venv_sp, "venv")):  # venv wins (precedence)
        for n, d in dists_in(path).items():
            composite[n] = (d.metadata["Name"], d.version, tag, d)
    lock_lines = sorted((f"{v[0]}=={v[1]}" for v in composite.values()), key=str.lower)
    (env_dir / "lock.txt").write_text("\n".join(lock_lines) + "\n")
    rec_lines = []
    for sp in (venv_sp, fresh_sp):
        for rec in sorted(glob.glob(os.path.join(sp, "*.dist-info", "RECORD"))):
            rec_lines.append(f"{sha256(Path(rec))}  {rec}")
    (env_dir / "manifest.sha256").write_text("\n".join(rec_lines) + "\n")
    wheels = {Path(w).name: sha256(Path(w)) for w in sorted(glob.glob(os.path.join(args.wheelhouse, "*.whl")))}
    (env_dir / "wheels.sha256").write_text("".join(f"{h}  {n}\n" for n, h in wheels.items()))

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version, "executable": sys.executable, "base_prefix": sys.base_prefix,
        "platform": platform.platform(), "pyvenv_cfg": cfg, "fresh01_pyvenv_cfg": fresh_cfg,
        "venv_site_packages": venv_sp, "fresh01_site_packages": fresh_sp,
        "pth_files": {os.path.basename(p): Path(p).read_text().strip() for p in pth},
        "sys_path": sys.path, "checks": checks, "imports": imports, "gdal": gdal, "smoke": smoke,
        "packages": {v[0]: {"version": v[1], "origin": v[2]} for _, v in sorted(composite.items())},
        "wheels_sha256": wheels,
        "installed_from_wheelhouse": checks["venv_only_packages"],
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
        "failures": failures, "ok": not failures,
    }
    (env_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"venv-only packages: {checks['venv_only_packages']}")
    print(f"wrote {env_dir/'manifest.json'}, lock.txt ({len(lock_lines)} pins), manifest.sha256, wheels.sha256")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("verify_env: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
