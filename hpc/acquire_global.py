#!/usr/bin/env python3
"""Acquire with bounded concurrency on a network-enabled workstation. James Edward Ball."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Run root containing global-01')
    p.add_argument('--workers', type=int, default=4, choices=range(1, 65),
                   help='Concurrent CPU acquisition processes; match network and memory capacity')
    a = p.parse_args()
    g = a.root.resolve() / 'global-01'
    subprocess.run([sys.executable, str(g / 'code/pilot.py')], check=True)
    if not json.loads((g / 'results/pilot-report.json').read_text())['passed']:
        raise RuntimeError('Pilot failed')
    def block(i):
        with (g / 'logs' / f'acquire-{i:03d}.log').open('a') as log:
            subprocess.run([sys.executable, str(g / 'code/acquire.py'), 'acquire', '--block', str(i), '--limit', '200'], stdout=log, stderr=subprocess.STDOUT, check=True)
        return i
    count = len(json.loads((g / 'data/blocks.json').read_text())['blocks'])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i in pool.map(block, range(count)):
            print(f'Block {i} complete', flush=True)
    subprocess.run([sys.executable, str(g / 'staging/verify_staged.py')], check=True)


if __name__ == '__main__':
    main()
