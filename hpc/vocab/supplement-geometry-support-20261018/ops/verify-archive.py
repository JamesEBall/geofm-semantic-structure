"""Verify archived records and recompute the frozen summary.

Author: James Edward Ball. This changes no inference or qualification rule.
"""

import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path

base = Path(__file__).resolve().parents[1]
repo = base.parents[2]
data = repo / "output/geometry-support-confirmation"
sys.path.insert(0, str(base))
import calibration as c

archive = data / "records.tar.gz"
receipt = c.read(base / "results/confirmation-archive-receipt.json")
h = hashlib.sha256()
with archive.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
assert h.hexdigest() == receipt["files"]["confirmation/records.tar.gz"]["sha256"]
assert (
    hashlib.sha256(
        (base / "results/confirmation-summary.json").read_bytes()
    ).hexdigest()
    == receipt["files"]["confirmation/summary.json"]["sha256"]
)
records = []
with tarfile.open(archive) as handle:
    manifest_payload = handle.extractfile("record-sha256.txt").read()
    assert (
        hashlib.sha256(manifest_payload).hexdigest()
        == receipt["files"]["confirmation/record-sha256.txt"]["sha256"]
    )
    manifest = dict(
        line.split("  ", 1)[::-1]
        for line in handle.extractfile("record-sha256.txt").read().decode().splitlines()
    )
    members = [m for m in handle.getmembers() if m.name.startswith("records/")]
    assert len(members) == len(manifest) == 47000
    assert {m.name for m in members} == set(manifest)
    for member in members:
        payload = handle.extractfile(member).read()
        assert hashlib.sha256(payload).hexdigest() == manifest[member.name]
        record = json.loads(payload)
        assert record["bootstrap"] == 499
        assert member.name == "records/%s/rep-%05d.json" % (
            record["scenario"],
            record["replicate"],
        )
        records.append(record)
cfg = c.read(base / "CONFIG.json")
summary = c.summarise(records, cfg, "confirmation")
hpc = c.read(base / "results/confirmation-summary.json")
max_diff = [0.0]


def compare(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            compare(a[key], b[key], path + "/" + key)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + "/" + str(i))
    elif isinstance(a, float):
        max_diff[0] = max(max_diff[0], abs(a - b))
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=2e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


compare(summary, hpc)
out = {
    "author": "James Edward Ball",
    "archive_sha256": h.hexdigest(),
    "records_verified": len(records),
    "manifest_hashes_match": True,
    "summary_and_manifest_file_hashes_match": True,
    "exact_registered_identities_and_provenance": True,
    "all_records_configured_bootstrap": 499,
    "complete_summary_recomputed_locally": True,
    "all_integer_boolean_and_identity_fields_exact": True,
    "maximum_absolute_float_difference": max_diff[0],
    "unavailable_records_preserved": sum(bool(r.get("failure")) for r in records),
}
(base / "results/confirmation-local-verification.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print(json.dumps(out, indent=2))
