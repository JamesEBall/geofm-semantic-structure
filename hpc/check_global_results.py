#!/usr/bin/env python3
"""Check archived global results against scene identities and scores.

Author: James Edward Ball. This audit does not rerun the image encoders.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def check():
    root = Path(__file__).resolve().parent
    results = root / 'reference-results/global'
    raw = gzip.decompress((root / 'reference/global-manifest.json.gz').read_bytes())
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    primary = json.loads((results / 'primary.json').read_text())
    assert primary['manifest_sha256'] == digest
    assert len(manifest['rows']) == 29733
    classes = manifest['classes']
    labels = np.array([classes.index(r['label']) for r in manifest['rows']])
    splits = np.array([r['split'] for r in manifest['rows']])
    blocks = np.array([str(r['block']) for r in manifest['rows']])
    assert int((splits == 'test').sum()) == 13782
    assert len(np.unique(blocks[splits == 'test'])) == 74
    refs = ('qwen_instructed', 'qwen_plain', 'mpnet_plain')
    pairs = (None, (0, 1), (2, 3), (4, 5), (6, 7))
    models = list(primary['models']) + ['rich_appearance', 'metadata', 'rich_metadata']
    summary = json.loads((results / 'controls/summary.json').read_text())
    lookup = {(r['model'], r['track'], r['route']): r for r in summary['rows']}
    assert len(lookup) == len(summary['rows']) == 768
    calculated = {}
    for model in models:
        directory = results / 'controls' / model
        done = json.loads((directory / 'complete.json').read_text())
        assert done['folds'] == 5
        assert done['fingerprint']['manifest'] == digest
        if model in primary['models']:
            assert done['fingerprint']['features'] == primary['models'][model]['feature_sha256']
        fold_values = []
        for fold, held in enumerate(pairs):
            metadata = json.loads((directory / f'fold-{fold}.json').read_text())
            score_path = directory / f'fold-{fold}-scores.npz'
            assert hashlib.sha256(score_path.read_bytes()).hexdigest() == metadata['scores_sha256']
            assert metadata['fingerprint'] == done['fingerprint']
            keep = (splits == 'test') & (True if held is None else np.isin(labels, held))
            expected = np.flatnonzero(keep)
            with np.load(score_path) as saved:
                np.testing.assert_array_equal(saved['test_rows'], expected)
                np.testing.assert_array_equal(saved['test_labels'], labels[keep])
                np.testing.assert_array_equal(saved['test_blocks'], blocks[keep])
                values = {}
                for name in saved.files:
                    if '__' not in name:
                        continue
                    score = saved[name]
                    assert score.shape == expected.shape
                    assert np.isfinite(score).all() and (score >= 1/8).all() and (score <= 1).all()
                    value = float(np.mean([score[blocks[keep] == b].mean() for b in np.unique(blocks[keep])]))
                    np.testing.assert_allclose(value, metadata['result'][name.replace('__', '/')]['test_query_ap'], atol=1e-12, rtol=0)
                    values[name] = value
                assert len(values) == 24
                fold_values.append(values)
        for track, folds in [('known', [0]), ('heldout', [1, 2, 3, 4])]:
            for route in fold_values[0]:
                calculated[model, track, route] = float(np.mean([fold_values[f][route] for f in folds]))
    for key, value in calculated.items():
        row = lookup[key]
        np.testing.assert_allclose(row['query_ap'], value, atol=1e-12, rtol=0)
        ref = key[2].split('__')[1]
        difference = value-calculated['rich_metadata', key[1], 'ridge__'+ref]
        np.testing.assert_allclose(row['difference_from_rich_metadata'], difference, atol=1e-12, rtol=0)
        assert row['valid_resamples'] == 1000
        assert len(row['conditional_interval']) == 2
    for model, values in primary['models'].items():
        for track, field in [('known', 'known'), ('heldout', 'heldout_summary')]:
            for ref in refs:
                np.testing.assert_allclose(values[field][ref]['query_ap'], calculated[model, track, 'ridge__'+ref], atol=1e-10, rtol=0)
    return {'author': 'James Edward Ball', 'manifest_sha256': digest,
            'models': len(models), 'folds_checked': len(models)*5,
            'summary_estimates_checked': len(calculated),
            'status': 'passed',
            'checks': 'Scene identities, split and category eligibility, feature fingerprints, score hashes, geographic weighting, held-out fold aggregation, baseline differences and original-route parity.'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
