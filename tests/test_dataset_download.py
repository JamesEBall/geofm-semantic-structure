"""Check immutable downloads and path confinement. Author: James Edward Ball."""
import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('dataset_download', Path(__file__).resolve().parents[1] / 'scripts/download_megaearth.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'tables/../../outside', 'tables\\..\\outside'])
def test_path_escape_is_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        d.destination(tmp_path, name)


def test_existing_corruption_blocks_download(tmp_path):
    (tmp_path / 'table').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum'):
        d.download(tmp_path, 'table', hashlib.sha256(b'right').hexdigest(), None)
    assert (tmp_path / 'table').read_bytes() == b'wrong'


def test_matching_file_needs_no_network(tmp_path):
    (tmp_path / 'table').write_bytes(b'right')
    d.download(tmp_path, 'table', hashlib.sha256(b'right').hexdigest(), None)


def test_pinned_manifest():
    assert d.digest(d.MANIFEST) == d.MANIFEST_SHA256
