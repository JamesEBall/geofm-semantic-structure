"""Release metadata must describe and hash the exact files prepared for upload.

Author: James Edward Ball.
"""
import json
from types import SimpleNamespace
from urllib.parse import urlparse

from mc.core import sha256_file
from stages import release


def test_release_card_is_manifested_and_uses_an_absolute_licence_link(tmp_path):
    tables = ('vocab/classes.parquet', 'vocab/taxonomy.yaml', 'select/windows.parquet',
              'assemble/windows_qa.parquet', 'labels_exact/labels.parquet',
              'admit/classes_final.parquet', 'admit/windows_final.parquet',
              'families/families.parquet', 'families/folds.json',
              'text/phrases.parquet', 'text/attribution.json')
    for rel in (*tables, 'freeze_f3/manifest.json'):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    bundle = tmp_path / 'release_build'
    bundle.mkdir()
    archive = bundle / 'data-00000.tar.zst'
    archive.write_bytes(b'unchanged archive')
    original = sha256_file(archive)
    ctx = SimpleNamespace(run=tmp_path, stage_dir=lambda: bundle,
                          p=lambda *parts: tmp_path.joinpath(*parts))
    result = release.finalize(ctx)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    card = (bundle / 'README.md').read_text()
    link = next(line.split(': ', 1)[1] for line in card.splitlines() if line.startswith('license_link:'))
    assert urlparse(link).scheme == 'https' and urlparse(link).netloc
    assert 'Prepared by' not in card
    assert 'RESOLVE' in card and 'CC BY 4.0' in card
    assert manifest['files']['README.md'] == sha256_file(bundle / 'README.md')
    assert set(manifest['files']) == {str(p.relative_to(bundle)) for p in bundle.rglob('*')
                                     if p.is_file() and p.name != 'manifest.json'}
    assert manifest['files'][archive.name] == original == sha256_file(archive)
    assert result == {'shards': 1, 'files': len(tables) + 2, 'uploaded': False}
