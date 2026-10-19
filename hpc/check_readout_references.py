#!/usr/bin/env python3
"""Reproduce post-hoc image-blind references. Author: James Edward Ball.

Uses the frozen manifest, text prototypes and archived per-scene scores.
No imagery, model weights, fitting or GPU is required. This diagnostic does
not alter the frozen primary analysis or select a baseline using test scores.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('global_analysis', ROOT / 'global/code/analyse.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
MODELS = ('clay', 'prithvi', 'prithvi100')


def target_weights(labels, allowed):
    """Constant predictors use permitted classes and fitting labels only."""
    if not len(labels) or not np.all(allowed[labels]):
        raise ValueError('Empty fitting set or forbidden fitting category')
    return {
        'train_frequency': np.bincount(labels, minlength=8) / len(labels),
        'uniform_permitted_categories': allowed / allowed.sum(),
    }


def compute():
    raw = gzip.decompress((ROOT / 'reference/global-manifest.json.gz').read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    primary = json.loads((ROOT / 'reference-results/global/primary.json').read_text())
    if digest != primary['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    rows = json.loads(raw)['rows']
    analysis.validate_manifest(rows)
    labels = np.array([analysis.CLASSES.index(row['label']) for row in rows])
    blocks = np.array([row['block'] for row in rows])
    splits = np.array([row['split'] for row in rows])
    result = {
        'author': 'James Edward Ball',
        'status': 'Post-hoc review diagnostic; not part of frozen study',
        'manifest_sha256': digest,
        'uniform_random_full_gallery_expected_rr': float(np.sum(1 / np.arange(1, 9)) / 8),
        'uniform_random_full_gallery_expected_top1': 0.125,
        'references': {},
    }
    with np.load(ROOT / 'global/data/text.npz') as texts:
        for ref in analysis.REFS:
            folds = []
            for fold, held in enumerate([None] + analysis.FOLDS):
                allowed = np.ones(8, dtype=bool)
                if held is not None:
                    allowed[list(held)] = False
                fit = (splits == 'train') & allowed[labels]
                test = (splits == 'test') & (True if held is None else np.isin(labels, held))
                record = {
                    'held_categories': None if held is None else [analysis.CLASSES[c] for c in held],
                    'n_test': int(test.sum()), 'baselines': {}, 'trained': {},
                }
                for name, weights in target_weights(labels[fit], allowed).items():
                    scores = analysis.language_scores(np.tile(weights, (test.sum(), 1)), texts[ref])
                    reciprocal_ranks = analysis.reciprocal_rank(labels[test], scores)
                    record['baselines'][name] = {
                        'query_ap': analysis.block_average(reciprocal_ranks, blocks[test]),
                        'top1': float(np.mean(scores.argmax(1) == labels[test])),
                        'constant_rank_order': [analysis.CLASSES[c] for c in np.argsort(-scores[0])],
                    }
                for model in MODELS:
                    directory = ROOT / 'reference-results/global/controls' / model
                    score_file = directory / f'fold-{fold}-scores.npz'
                    metadata = json.loads((directory / f'fold-{fold}.json').read_text())
                    if hashlib.sha256(score_file.read_bytes()).hexdigest() != metadata['scores_sha256']:
                        raise ValueError(f'Score checksum mismatch: {score_file}')
                    if metadata['fingerprint']['text'] != analysis.sha(ROOT / 'global/data/text.npz'):
                        raise ValueError('Frozen text checksum mismatch')
                    with np.load(score_file) as saved:
                        np.testing.assert_array_equal(saved['test_rows'], np.flatnonzero(test))
                        np.testing.assert_array_equal(saved['test_labels'], labels[test])
                        np.testing.assert_array_equal(saved['test_blocks'], blocks[test])
                        record['trained'][model] = analysis.block_average(saved['ridge__' + ref], blocks[test])
                folds.append(record)
            result['references'][ref] = {
                'folds': folds,
                'heldout_average': {name: float(np.mean([f['baselines'][name]['query_ap'] for f in folds[1:]]))
                                    for name in folds[0]['baselines']},
                'heldout_trained': {model: float(np.mean([f['trained'][model] for f in folds[1:]]))
                                    for model in MODELS},
            }
        # Verify the ridge identity with the actual Qwen prototype matrix.
        rng = np.random.default_rng(15026)
        x, val, test = (rng.normal(size=(n, 12)) for n in (90, 30, 40))
        onehot = np.eye(8)[rng.integers(0, 6, 90)]
        text = analysis.unit(texts['qwen_instructed'])
        errors = []
        categorical = analysis.ridge_predictions(x, onehot, val, test)
        linguistic = analysis.ridge_predictions(x, onehot @ text, val, test)
        for (_, _, pred), (_, _, direct) in zip(categorical, linguistic):
            errors.append(float(np.max(np.abs(pred @ text - direct))))
        result['ridge_prototype_identity_max_abs_error'] = max(errors)
        if max(errors) > 1e-12:
            raise AssertionError('Ridge target-linearity check failed')
    np.testing.assert_allclose(
        analysis.reciprocal_rank(np.arange(8), np.zeros((8, 8))),
        result['uniform_random_full_gallery_expected_rr'], rtol=0, atol=1e-15)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('work/readout-references.json'))
    args = parser.parse_args()
    result = compute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.out}; frozen scene identities, text and scores verified.')
    for ref, data in result['references'].items():
        print(ref, data['heldout_average'])


if __name__ == '__main__':
    main()
