"""Saved-feature selection and manifest tests. Author: James Edward Ball."""

import pytest
import native_diagnostics as n


def test_training_only_and_identity():
    windows = [
        {"widx": i, "window_id": str(i), "split": s}
        for i, s in enumerate(["train", "test", "train", "val"])
    ]
    labels = [{"widx": i, "window_id": str(i)} for i in range(4)]
    retained = [dict(x, retained=i != 2) for i, x in enumerate(labels)]
    assert n.checked_training_rows(windows, labels, retained).tolist() == [0]
    labels[1]["window_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        n.checked_training_rows(windows, labels, retained)


def test_manifest_mutation_rejected(tmp_path):
    (tmp_path / "features").write_bytes(b"original")
    manifest = {"files": {"features": n.digest(tmp_path / "features")}}
    n.verify_file(tmp_path, manifest, "features")
    (tmp_path / "features").write_bytes(b"altered")
    with pytest.raises(ValueError, match="provenance"):
        n.verify_file(tmp_path, manifest, "features")


def test_block_half_shared_and_stable():
    assert n.half((10, 25), 2026101892) == n.half((10, 25), 2026101892)
    assert {n.half((i, 25), 2026101892) for i in range(20)} == {0, 1}


def test_changed_manifest_rejected_before_table_or_feature_access(tmp_path):
    folder = tmp_path / "freeze_f3_cpu"
    folder.mkdir()
    (folder / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Pinned F3"):
        n.run(tmp_path, "0" * 64)
