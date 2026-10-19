"""Reproduce the exploratory Prithvi EO 1.0 100M extension. James Edward Ball."""

import os

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .data import PACKAGE, acquire, cached_assets, digest, download, merge_labels, read, write
from .run import stage
from .releases import repository


def extract(work, expected):
    """Rebuild only the extension encoder features from downloaded image pixels."""
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = read(PACKAGE / "data/prithvi100/model.json")
    paths = {}
    for item in spec["files"]:
        paths[item["name"]] = download(
            spec["base_url"] + item["name"],
            work / "weights/prithvi100" / item["name"],
            item["sha256"],
        )
    module_spec = importlib.util.spec_from_file_location(
        "pinned_prithvi100", paths["prithvi_mae.py"]
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    cfg = read(paths["config.json"])["pretrained_cfg"]
    cfg.update(img_size=64, num_frames=1)
    mean = np.array(cfg["mean"], np.float32)[:, None, None] / 10000
    std = np.array(cfg["std"], np.float32)[:, None, None] / 10000
    rows = read(PACKAGE / "data/bundle.json")["rows"]
    features = {}
    for seed in [0, 101, 202, 303]:
        torch.manual_seed(seed)
        model = module.PrithviViT(**cfg).eval()
        assert model.pos_embed.shape == (1, 17, 768)
        name = "prithvi100" if seed == 0 else f"random_prithvi100_{seed}"
        if seed == 0:
            state = torch.load(
                paths["Prithvi_EO_V1_100M.pt"], map_location="cpu", weights_only=True
            )
            enc = {
                k.removeprefix("encoder."): v
                for k, v in state.items()
                if k.startswith("encoder.") and "pos_embed" not in k
            }
            assert set(enc) == set(dict(model.named_parameters()))
            loaded = model.load_state_dict(enc, strict=False)
            assert loaded.missing_keys == ["pos_embed"] and not loaded.unexpected_keys
        vectors, global_vectors = [], []
        with torch.inference_mode():
            for row in rows:
                with np.load(
                    work / "imagery" / (row["patch_id"] + ".npz"), allow_pickle=False
                ) as z:
                    image = z["input64"].copy()
                assert image.shape == (6, 64, 64) and np.isfinite(image).all()
                token = model.forward_features(
                    torch.from_numpy((image - mean) / std)[None, :, None, :, :]
                )[-1]
                assert token.shape == (1, 17, 768)
                vectors.append(token[0, 1:].mean(0).numpy().copy())
                global_vectors.append(token[0, 0].numpy().copy())
        features[name] = np.stack(vectors).astype(float)
        features[name + "_global"] = np.stack(global_vectors).astype(float)
        print("Extracted", name, flush=True)
    for name, value in features.items():
        if not np.allclose(value, expected[name], rtol=0, atol=1e-5):
            raise ValueError("Extracted feature drift: " + name)
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work-prithvi100"))
    parser.add_argument(
        "--from-source",
        action="store_true",
        help="Rebuild the additional encoder from source imagery; original study comparator features remain cached.",
    )
    parser.add_argument(
        "--check-validation",
        action="store_true",
        help="Recompute every additional penalty grid and verify the saved choices.",
    )
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "results").exists():
        raise RuntimeError("Use a fresh work directory to preserve completed or failed results.")
    merge_labels(work)
    cached_assets(work)
    manifest = read(PACKAGE / "data/prithvi100/assets.json")
    for item in manifest["files"]:
        target = work / "cache" / item["name"]
        if not target.exists():
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    manifest["tag"],
                    "--repo",
                    repository(),
                    "--pattern",
                    item["name"],
                    "--dir",
                    str(target.parent),
                ],
                check=True,
            )
        if digest(target) != item["sha256"]:
            raise ValueError("Extension cache mismatch: " + item["name"])
    with np.load(work / "cache/prithvi100-features.npz", allow_pickle=False) as z:
        features = {k: z[k].copy() for k in z.files}
    expected_names = {
        "prithvi100",
        "random_prithvi100_101",
        "random_prithvi100_202",
        "random_prithvi100_303",
    }
    expected_names |= {n + "_global" for n in tuple(expected_names)}
    assert set(features) == expected_names
    assert all(v.shape == (2644, 768) and np.isfinite(v).all() for v in features.values())
    if args.from_source:
        acquire(work)
        features = extract(work, features)
    stage(work)
    base = work / "results/semantic-structure"
    bp = base / "dynamicworld-main-inputs-01/report.json"
    bundle = read(bp)
    with np.load(work / "cache/features.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays.update({"feature__" + k: v for k, v in features.items()})
    np.savez(work / "cache/extended-features.npz", **arrays)
    bundle.update(
        arrays_path="cache/extended-features.npz",
        arrays_sha256=digest(work / "cache/extended-features.npz"),
    )
    bundle["feature_names"] += list(features)
    write(bp, bundle)
    cp = base / "dynamicworld-nuisance-calibration-01/report.json"
    calibration = read(cp)
    with np.load(work / "cache/calibration.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(PACKAGE / "data/prithvi100/calibration.npz", allow_pickle=False) as z:
        arrays.update({k: z[k].copy() for k in z.files})
    np.savez(work / "cache/extended-calibration.npz", **arrays)
    calibration.update(
        arrays_path="cache/extended-calibration.npz",
        arrays_sha256=digest(work / "cache/extended-calibration.npz"),
    )
    write(cp, calibration)
    lp = base / "dynamicworld-learning-readout-01/report.json"
    learning = read(lp)
    choices = json.loads(
        gzip.decompress((PACKAGE / "data/prithvi100/validation.json.gz").read_bytes())
    )
    learning["records"] += choices["records"]
    write(lp, learning)
    env = {**os.environ, "GEOFM_RUN_DIR": str(work), "PYTHONPATH": str(PACKAGE / "src")}
    if args.check_validation:
        subprocess.run(
            [
                sys.executable,
                str(PACKAGE / "scripts/prithvi100_check_validation.py"),
                "--work",
                str(work),
            ],
            env=env,
            check=True,
        )
    for name in ["geometry", "readout"]:
        subprocess.run(
            [sys.executable, str(PACKAGE / "scripts" / f"prithvi100_{name}.py")],
            env=env,
            check=True,
        )
    geometry = read(base / "dynamicworld-main-geometry-01/report.json")
    readout = read(base / "dynamicworld-main-readout-01/report.json")
    actual = {
        "geometry": {
            f"{a['feature']}|{a['route']}|{k}": v["rsa"]
            for a in geometry["records"]
            for k, v in a.get("references", {}).items()
            if v["status"] == "available"
        },
        "transfer": {
            f"{a['intended_sites']}|{a['seed']}|{a['reference']}|{a['feature']}|{a['control']}|{a['cell']}": a[
                "summary"
            ]["ap"]["component_mean"]
            for a in readout["records"]
            if a["status"] == "available"
        },
    }
    original = json.loads(gzip.decompress((PACKAGE / "data/expected-results.json.gz").read_bytes()))
    extension = json.loads(
        gzip.decompress((PACKAGE / "data/prithvi100/expected-results.json.gz").read_bytes())
    )
    checks = {}
    for section in actual:
        expected = {**original[section], **extension[section]}
        assert set(actual[section]) == set(expected)
        error = max(abs(actual[section][k] - v) for k, v in expected.items())
        checks[section] = {"count": len(expected), "maximum_absolute_error": error}
        assert error <= 1e-10
    write(
        work / "outputs/prithvi100-verification.json",
        {
            "author": "James Edward Ball",
            "status": "passed",
            "comparisons": checks,
            "scope": "Exploratory checkpoint extension on the existing benchmark.",
        },
    )
    write(
        work / "outputs/prithvi100-summary.json",
        {
            "author": "James Edward Ball",
            "scope": "Prithvi100 is exploratory; the original hypothesis family remains unchanged.",
            "comparisons": readout["primary_comparisons"],
        },
    )


if __name__ == "__main__":
    main()
