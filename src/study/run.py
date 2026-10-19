"""One-command reproduction of the frozen main study. James Edward Ball."""

import os

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import subprocess
import sys
from pathlib import Path

from .data import PACKAGE, acquire, cached_assets, digest, merge_labels, read, write


def stage(work):
    """Build a portable run tree; accepted reports remain untouched."""
    base = work / "results/semantic-structure"
    aliases = {
        "bundle": "dynamicworld-main-inputs-01",
        "text": "dynamicworld-text-reference-01",
        "calibration": "dynamicworld-nuisance-calibration-01",
        "learning": "dynamicworld-learning-readout-01",
        "admission": "dynamicworld-evaluation-admission-01",
        "precision": "dynamicworld-evaluation-precision-01",
    }
    assets = {"bundle": "features.npz", "text": "text.npz", "calibration": "calibration.npz"}
    for key, directory in aliases.items():
        record = read(PACKAGE / "data" / f"{key}.json")
        if key in assets:
            p = work / "cache" / assets[key]
            record.update(arrays_path=str(p.relative_to(work)), arrays_sha256=digest(p))
        write(base / directory / "report.json", record)
        write(
            base / directory / "audit.json",
            {
                "status": "passed",
                "scope": "Pinned inputs verified by portable entry point; see DATA.md for scope.",
            },
        )
    (work / ".scratch").mkdir(exist_ok=True)


def analyse(work):
    stage(work)
    env = {**os.environ, "GEOFM_RUN_DIR": str(work), "PYTHONPATH": str(PACKAGE / "src")}
    for script in ["geometry", "readout"]:
        result = work / "results/semantic-structure" / f"dynamicworld-main-{script}-01/report.json"
        if result.exists():
            continue
        partial = result.parent
        if partial.exists():
            raise RuntimeError(
                f"Incomplete analysis retained at {partial}; use a new --work directory to retry."
            )
        subprocess.run(
            [sys.executable, str(PACKAGE / "scripts" / f"{script}.py")], env=env, check=True
        )
    from .report import export

    export(work)
    from .verify import verify

    verify(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work"))
    p.add_argument(
        "--from-source",
        action="store_true",
        help="Rebuild image features and text references from pinned upstream assets. CPU run can take hours.",
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--limit", type=int, help="Preparation smoke test only; never produces paper results."
    )
    args = p.parse_args()
    if args.limit is not None and (not args.prepare_only or args.limit < 1):
        p.error("--limit requires --prepare-only and a positive count")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    mode = "source" if args.from_source else "preparation" if args.prepare_only else "cached"
    marker = work / "run-mode.json"
    if marker.exists() and read(marker) != {"mode": mode, "limit": args.limit}:
        raise RuntimeError(
            "Use a separate --work directory for a different run mode or sample size."
        )
    write(marker, {"mode": mode, "limit": args.limit})
    merge_labels(work, args.limit)
    if args.from_source or args.prepare_only:
        acquire(work, args.limit)
    if args.prepare_only:
        return
    if args.from_source:
        from .models import rebuild

        rebuild(work)
    else:
        cached_assets(work)
    analyse(work)
    if not args.from_source:
        for name in ("check_validation", "check_supporting", "check_development"):
            subprocess.run(
                [sys.executable, str(PACKAGE / "scripts" / f"{name}.py"), "--work", str(work)],
                check=True,
            )


if __name__ == "__main__":
    main()
