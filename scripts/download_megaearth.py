"""Download and verify the frozen public dataset. Author: James Edward Ball."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Mead0w1ark/megaearth-osm-sentinel2'
REVISION = 'de5ec4da7cb2c195948a12fcbb0ab5b719e0cd92'
MANIFEST = ROOT / 'hpc/vocab/results/full-20261018/release-manifest.json'
MANIFEST_SHA256 = '70a1d0c1a16ed9cbc3748ddba215b1b82edb4bd2e5f6dee2d616a7c89a698fef'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def destination(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError('Invalid dataset path: ' + name)
    path = root / name
    if root.resolve() not in path.resolve().parents:
        raise ValueError('Dataset path escapes destination: ' + name)
    return path


def download(root, name, expected, session):
    path = destination(root, name)
    if path.exists():
        if digest(path) != expected:
            raise ValueError('Existing file checksum mismatch: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
    try:
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for block in response.iter_content(1 << 20):
                    f.write(block)
        if digest(temporary) != expected:
            raise ValueError('Downloaded file checksum mismatch: ' + name)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--imagery', action='store_true', help='Also download all image archives')
    a = p.parse_args()
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError('Frozen dataset manifest has changed')
    manifest = json.loads(MANIFEST.read_text())
    selected = {k: v for k, v in manifest['files'].items()
                if k.startswith('tables/') or (a.imagery and k.endswith('.tar.zst'))}
    a.output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for i, (name, expected) in enumerate(sorted(selected.items()), 1):
            download(a.output, name, expected, session)
            print(f'{i}/{len(selected)} verified: {name}', flush=True)
    receipt = {'author': 'James Edward Ball', 'repository': REPO, 'revision': REVISION,
               'manifest_sha256': MANIFEST_SHA256, 'files': selected, 'imagery': a.imagery}
    (a.output / 'download-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
