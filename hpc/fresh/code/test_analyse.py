"""Numerical checks for the frozen evaluation. Author: James Edward Ball."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from analyse import ridge_predictions, language_scores, metrics, validate_manifest, CLASSES, paired_bootstrap, REFS, validate_geography, discover_features, EXPECTED_MODELS, sha, reciprocal_rank, block_average, geometry


class AnalysisTests(unittest.TestCase):
    def test_complete_panel_and_diagnostic_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'features/clay-diagnostic-8').mkdir(parents=True)
            for model in EXPECTED_MODELS:
                folder = root / 'features' / model
                folder.mkdir()
                np.save(folder / 'mean.npy', np.ones((30, 4), dtype='float32'))
                (folder / 'provenance.json').write_text(json.dumps(dict(manifest_sha256='manifest', fingerprint='same', diagnostic_limit=None)))
                (folder / 'complete.json').write_text(json.dumps(dict(shape=[30, 4], fingerprint='same', parity_passed=True, files={'mean.npy': sha(folder / 'mean.npy')})))
            paths = discover_features(root, 'mean', 30, 'manifest')
            self.assertEqual(len(paths), 13)
            self.assertNotIn('clay-diagnostic-8', paths)
            (root / 'features/random_prithvi100_303/complete.json').unlink()
            with self.assertRaisesRegex(ValueError, 'random_prithvi100_303'):
                discover_features(root, 'mean', 30, 'manifest')

    def test_geometry_matches_raw_parent_block_order(self):
        rng = np.random.default_rng(92)
        y = np.tile(np.arange(10), 6)
        blocks = np.repeat(['a', 'a', 'b', 'b', 'c', 'c'], 10)
        x = rng.normal(size=(60, 8)) * np.linspace(.1, 20, 60)[:, None]
        text = {ref: rng.normal(size=(10, 5)) for ref in REFS}
        actual = geometry(x, y, blocks, text)
        means = []
        for category in range(10):
            components = []
            for block in ['a', 'b', 'c']:
                indices = [i for i in range(60) if y[i] == category and blocks[i] == block]
                mean = sum(x[i] for i in indices) / len(indices)
                components.append(mean / np.linalg.norm(mean))
            mean = sum(components) / len(components)
            means.append(mean / np.linalg.norm(mean))
        distances = [1 - np.dot(means[i], means[j]) for i in range(10) for j in range(i + 1, 10)]
        np.testing.assert_allclose(actual[REFS[0]]['image_distances'], distances, atol=1e-12)

    def test_primal_and_dual_match_direct_solver(self):
        rng = np.random.default_rng(1)
        for n, d in [(30, 8), (12, 25)]:
            x = rng.normal(size=(n, d)); x[:, 0] = 5
            y = rng.normal(size=(n, 10))
            v = rng.normal(size=(7, d)); t = rng.normal(size=(5, d))
            scaler = StandardScaler().fit(x)
            for a, pv, pt in ridge_predictions(x, y, v, t):
                baseline = Ridge(alpha=a, solver='svd').fit(scaler.transform(x), y)
                np.testing.assert_allclose(pv, baseline.predict(scaler.transform(v)), atol=1e-10)
                np.testing.assert_allclose(pt, baseline.predict(scaler.transform(t)), atol=1e-10)

    def test_text_factorisation(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(8, 10)); t = rng.normal(size=(10, 12))
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        predicted = p @ t
        expected = predicted / np.linalg.norm(predicted, axis=1, keepdims=True) @ t.T
        np.testing.assert_allclose(language_scores(p, t), expected, atol=1e-12)

    def test_split_checks(self):
        rows = [dict(id=f'{s}-{c}', label=c, block=f'{s}-{c}', split=s, lon=0, lat=0)
                for s in ('train', 'val', 'test') for c in CLASSES]
        validate_manifest(rows)
        rows[-1]['block'] = rows[0]['block']
        with self.assertRaises(ValueError):
            validate_manifest(rows)

    def test_geographic_guard(self):
        rows = [dict(split=s, lon=lon, lat=0) for s, lon in [('train', 0), ('val', 1), ('test', 2)]]
        self.assertGreater(validate_geography(rows)['train_val'], 100)
        rows[-1]['lon'] = 1.001
        with self.assertRaises(ValueError):
            validate_geography(rows)

    def test_tied_rank_against_enumeration(self):
        import itertools
        scores = np.array([[.9, .8, .8, .8, .2], [1., 1., 1., 1., 1.]])
        y = np.array([2, 0])
        expected = []
        for row, target in zip(scores, y):
            orders = [p for p in itertools.permutations(range(5))
                      if all(row[p[i]] >= row[p[i + 1]] for i in range(4))]
            expected.append(np.mean([1 / (p.index(target) + 1) for p in orders]))
        np.testing.assert_allclose(reciprocal_rank(y, scores), expected)

    def test_block_weighting(self):
        self.assertAlmostEqual(block_average(np.array([1., 0., 0.]), np.array(['a', 'b', 'b'])), .5)

    def test_bootstrap_matches_explicit_block_means(self):
        rng = np.random.default_rng(5)
        y = np.tile(np.arange(10), 3)
        b = np.repeat(np.arange(3), 10).astype(str)
        scores = rng.integers(0, 3, size=(30, 10)).astype(float)
        control = np.zeros_like(scores)
        records = {name: {track: {ref: [(y, s, b, list(range(10)))] for ref in REFS}
                          for track in ('known', 'heldout')} for name, s in [('model', scores), ('appearance', control)]}
        actual = paired_bootstrap(records, b, 40, 42)['model']['known'][REFS[0]]
        draws = np.random.default_rng(42).multinomial(3, [1 / 3] * 3, size=40)
        delta = reciprocal_rank(y, scores) - reciprocal_rank(y, control)
        means = np.array([delta[b == str(i)].mean() for i in range(3)])
        expected = [np.average(means, weights=w) for w in draws]
        np.testing.assert_allclose(actual['conditional_query_ap_difference_interval'], np.quantile(expected, [.025, .975]), atol=1e-12)
        self.assertEqual(actual['invalid_resamples'], 0)


if __name__ == '__main__':
    unittest.main()
