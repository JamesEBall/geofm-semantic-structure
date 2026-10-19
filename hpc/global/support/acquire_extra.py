#!/usr/bin/env python3
"""Run one frozen additional block through unchanged acquisition. James Edward Ball."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    task = int(sys.argv[1])
    spec = json.loads((ROOT/'data/blocks.json').read_text())
    committed = json.loads((ROOT/'support/committed.json').read_text())
    original = spec['support_augmentation']['original_blocks']
    total = committed['additional_blocks']
    assert len(spec['blocks']) == original + total
    if not 0 <= task < 38:
        raise ValueError('Unexpected array index')
    if task >= total:
        print('Unused reserve array slot', task, flush=True)
        return
    subprocess.run([sys.executable, '-u', str(ROOT/'code/acquire.py'),
                    'acquire', '--block', str(original+task), '--limit', '200'], check=True)


if __name__ == '__main__':
    main()
