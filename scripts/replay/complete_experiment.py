"""Verify and replay the complete paired training archive. Author: James Edward Ball."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from verify_archive import verify_archive

EXPECTED_SHA256 = 'd7651dda035cc23d5b79477456dab96011ff73e1c9e63e33454ff90d076b6816'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for extracted records and fresh verification receipts')
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Archive SHA-256 differs from the released collection')
    verify_archive(args.archive)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    collection = out / 'collection'
    collection.mkdir()
    with tarfile.open(args.archive) as archive:
        archive.extractall(collection, filter='data')
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / 'controlled_transfer.py'), str(collection),
                    '--output', str(out / 'transfer-verification.json')], check=True)
    # The frozen verifier writes beside its own source. Work on a copy so the
    # extracted immutable collection and its original receipts remain unchanged.
    native = out / 'native-recheck'
    shutil.copytree(collection / 'native-geometry-20261021-v1', native)
    (native / 'result/verification.json').unlink()
    subprocess.run([sys.executable, str(native / 'verify_result.py'),
                    str(collection / 'controlled-pretraining-20261021-v1')], check=True)
    shutil.copy2(native / 'result/verification.json', out / 'native-verification.json')
    (out / 'verification.json').write_text(json.dumps({
        'author': 'James Edward Ball', 'passed': True,
        'archive_sha256': EXPECTED_SHA256, 'training_repeated': False,
        'population_inference_qualified': False,
        'transfer_receipt': 'transfer-verification.json',
        'native_receipt': 'native-verification.json'}, indent=2) + '\n')
    print('Complete saved-result replay passed:', out / 'verification.json')

if __name__ == '__main__':
    main()
