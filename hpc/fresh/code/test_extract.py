"""Checks for extraction cache helpers. Author: James Edward Ball."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('extract', Path(__file__).with_name('extract.py'))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class CacheTests(unittest.TestCase):
    def test_atomic_json_replaces_and_hash_tracks_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'record.json'
            extract.atomic_json(p, {'done': False})
            before = extract.sha(p)
            extract.atomic_json(p, {'done': True})
            self.assertEqual(json.loads(p.read_text()), {'done': True})
            self.assertNotEqual(before, extract.sha(p))
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_rejects_invalid_cached_features(self):
        good = np.ones((3, 7), dtype=np.float32)
        extract.validate_features(good, good, 3)
        invalid = [np.zeros_like(good), good.astype(np.float64),
                   np.full_like(good, np.nan), np.full_like(good, np.inf),
                   np.ones((2, 7), dtype=np.float32)]
        for item in invalid:
            with self.subTest(shape=item.shape, dtype=item.dtype):
                with self.assertRaises(ValueError):
                    extract.validate_features(item, good, 3)

    def test_chunk_metadata_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'chunk.npz'
            x = np.ones((2, 4), dtype=np.float32)
            np.savez(p, mean=x, cls=x, fingerprint='frozen-input-hash')
            with np.load(p, allow_pickle=False) as saved:
                self.assertEqual(str(saved['fingerprint']), 'frozen-input-hash')
                extract.validate_features(saved['mean'], saved['cls'], 2)


if __name__ == '__main__':
    unittest.main()
