#!/usr/bin/env python3
"""Freeze additive support inventory before image acquisition. James Edward Ball."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'amendments/2026-10-15-support-augmentation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    # This transition is valid only before global extraction begins.
    if (ROOT / 'results/cohort-freeze.json').exists():
        raise RuntimeError('A global cohort has already been frozen')
    for folder in ['features', 'embeddings']:
        if (ROOT / folder).exists() and any(p.is_file() for p in (ROOT / folder).rglob('*')):
            raise RuntimeError('Global image features already exist')
    for name, digest in json.loads((ROOT / 'support/preselection-freeze.json').read_text()).items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Preselection changed: ' + name)
    old = json.loads((ARCHIVE / 'data/blocks.json').read_text())
    spec = json.loads((ROOT / 'data/blocks.json').read_text())
    if spec != old:
        raise RuntimeError('Original inventory changed during support selection')
    selection = json.loads((ROOT / 'support/selection.json').read_text())
    config = json.loads((ROOT / 'support/targets.json').read_text())
    from select_support import Selection, far_from
    replay = Selection(spec['blocks'], config)
    for item in selection['evidence']:
        block = item['block']
        result = json.loads((ROOT / 'support/cache' / (block['id']+'.json')).read_text())
        candidate = {k:v for k,v in block.items() if k != 'split'}
        if not replay.consider(candidate, result) or replay.additions[-1] != block:
            raise RuntimeError('Selection replay failed: ' + block['id'])
    if not replay.complete() or replay.additions != selection['selected']:
        raise RuntimeError('Support target check failed')
    centres = [[b['lon']+.5, b['lat']+.5] for b in replay.additions]
    if not far_from(centres, old['prior_points'], 100).all():
        raise RuntimeError('Prior-study separation failed')
    spec['blocks'] += replay.additions
    spec['target_blocks'] = len(spec['blocks'])
    spec['target_attempted_windows'] = len(spec['blocks']) * 200
    spec['split_counts'] = dict(Counter(b['split'] for b in spec['blocks']))
    spec['support_augmentation'] = dict(protocol='support/protocol.md',
        original_blocks=120, original_inventory_sha256=sha(ARCHIVE / 'data/blocks.json'),
        selected_blocks=len(replay.additions), selection_sha256=sha(ROOT / 'support/selection.json'),
        quality_gates_changed=False)
    assert spec['blocks'][:120] == old['blocks']
    save(ROOT / 'data/blocks.json', spec)
    freeze = json.loads((ROOT / 'code/acquisition-freeze.json').read_text())
    added = ['support/protocol.md','support/targets.json','support/select_support.py',
             'support/commit_inventory.py','support/selection.json',
             'support/preselection-freeze.json', 'support/acquire_extra.py',
             'support/test_support.py', 'support/prepare.sbatch',
             'support/acquire.sbatch', 'support/admit.sbatch']
    save(ROOT / 'code/acquisition-freeze.json',
         {name:sha(ROOT / name) for name in list(freeze) + added})
    # The previous extraction freeze is archived. Re-freeze the same encoder and
    # analysis files with the new acquisition inventory before any extraction.
    frozen = json.loads((ROOT / 'results/extraction-freeze.json').read_text())
    for name, digest in frozen['sha256'].items():
        if name != 'code/acquisition-freeze.json' and sha(ROOT / name) != digest:
            raise RuntimeError('Extraction or analysis changed: ' + name)
    frozen['sha256']['code/acquisition-freeze.json'] = sha(ROOT / 'code/acquisition-freeze.json')
    frozen['support_amendment_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(ROOT / 'results/extraction-freeze.json', frozen)
    subprocess.run([sys.executable, str(ROOT/'code/freeze.py'), '--check'], check=True)
    save(ROOT / 'support/committed.json', dict(author='James Edward Ball',
         blocks=len(spec['blocks']), additional_blocks=len(replay.additions),
         split_counts=spec['split_counts'], inventory_sha256=sha(ROOT/'data/blocks.json'),
         committed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Additional inventory frozen', len(replay.additions), spec['split_counts'], flush=True)


if __name__ == '__main__':
    main()
