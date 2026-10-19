"""Scientific safeguards for additive support selection. James Edward Ball."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from select_support import Selection, far_from


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent/'targets.json').read_text())
        self.original = [dict(id='0_0', lon=0, lat=0, region='Africa', split='train')]

    def result(self, counts, n=200):
        return dict(candidate_count=n, target_counts=counts, candidate_sha256='fixture')

    def block(self, lon=20, lat=20):
        return dict(id=f'{lon}_{lat}', lon=lon, lat=lat, region='Africa')

    def test_geographic_guard_handles_dateline_and_prior_points(self):
        result = far_from([[179.9,0], [170,0]], [[-179.9,0]], 100)
        self.assertEqual(result.tolist(), [False, True])
        self.assertTrue(far_from([[1,2]], [], 100).all())

    def test_incomplete_candidates_and_insufficient_class_support_are_rejected(self):
        s = Selection(self.original, self.config)
        self.assertFalse(s.consider(self.block(), self.result({50:100}, 199)))
        self.assertFalse(s.consider(self.block(), self.result({50:9,90:9,60:19})))

    def test_original_and_new_blocks_keep_distance(self):
        s = Selection(self.original, self.config)
        self.assertFalse(s.consider(self.block(1,0), self.result({50:10})))
        self.assertTrue(s.consider(self.block(), self.result({50:10})))
        self.assertFalse(s.consider(self.block(21,20), self.result({50:10})))

    def test_validation_allocation_and_test_overlap(self):
        original = copy.deepcopy(self.original)
        config = copy.deepcopy(self.config)
        config['targets'][0]['blocks'] = 1
        s = Selection(self.original, config)
        self.assertTrue(s.consider(self.block(20), self.result({50:10,90:10,60:20})))
        self.assertEqual(s.additions[0]['split'], 'val')
        self.assertEqual(s.counts[('test',90)], 0)
        self.assertTrue(s.consider(self.block(30), self.result({50:10,90:10,60:20})))
        self.assertEqual(s.additions[1]['split'], 'test')
        self.assertEqual([s.counts[('test',c)] for c in [90,50,60]], [1,1,1])
        self.assertEqual(original, self.original)

    def test_completed_selection_stops_without_changing_targets(self):
        config = copy.deepcopy(self.config)
        for target in config['targets']:
            target['blocks'] = 1
        s = Selection(self.original, config)
        s.consider(self.block(20), self.result({50:10}))
        s.consider(self.block(30), self.result({50:10,90:10,60:20}))
        self.assertTrue(s.complete())
        self.assertFalse(s.consider(self.block(40), self.result({50:200})))


if __name__ == '__main__':
    unittest.main()
