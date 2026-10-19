"""Compare every regenerated endpoint with the accepted fixed-input result."""

import gzip
import importlib.metadata
import json
import sys

from .data import PACKAGE, digest, read, write


def verify(work):
    reference = json.loads(
        gzip.decompress((PACKAGE / "data/expected-results.json.gz").read_bytes())
    )
    base = work / "results/semantic-structure"
    g = read(base / "dynamicworld-main-geometry-01/report.json")
    r = read(base / "dynamicworld-main-readout-01/report.json")
    actual = {
        "geometry": {
            f"{a['feature']}|{a['route']}|{k}": v["rsa"]
            for a in g["records"]
            for k, v in a.get("references", {}).items()
            if v["status"] == "available"
        },
        "transfer": {
            f"{a['intended_sites']}|{a['seed']}|{a['reference']}|{a['feature']}|{a['control']}|{a['cell']}": a[
                "summary"
            ]["ap"]["component_mean"]
            for a in r["records"]
            if a["status"] == "available"
        },
    }
    comparisons = {}
    for key in reference:
        if set(reference[key]) != set(actual[key]):
            raise ValueError("Incomplete comparison inventory")
        errors = {k: abs(actual[key][k] - v) for k, v in reference[key].items()}
        comparisons[key] = {
            "count": len(errors),
            "maximum_absolute_error": max(errors.values()),
            "within_tolerance": all(v <= 1e-10 for v in errors.values()),
        }
    result = {
        "author": "James Edward Ball",
        "status": "passed"
        if all(x["within_tolerance"] for x in comparisons.values())
        else "numerical_drift",
        "comparisons": comparisons,
        "tolerance": 1e-10,
        "python": sys.version,
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "rasterio"]},
        "source_sha256": {
            str(p.relative_to(PACKAGE)): digest(p)
            for folder in ["src", "scripts", "data"]
            for p in sorted((PACKAGE / folder).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "scope": "Main study fixed-input computation; this is not an independent scientific replication.",
    }
    write(work / "outputs/verification.json", result)
    if result["status"] != "passed":
        raise RuntimeError(
            "Results differ from the accepted values. Inspect outputs/verification.json; retained results are not silently substituted."
        )
    return result
