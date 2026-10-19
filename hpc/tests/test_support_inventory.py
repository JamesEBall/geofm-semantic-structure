"""Validate the frozen support amendment and unchanged gates. James Edward Ball."""
import ast
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'global'
ARCHIVE = ROOT / 'amendments/2026-10-15-support-augmentation'


def distance(first, second):
    lon1, lat1, lon2, lat2 = map(math.radians, [*first, *second])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371*math.asin(min(1, math.sqrt(h)))


class FrozenSupportTests(unittest.TestCase):
    def test_original_blocks_and_acquisition_checks_preserved(self):
        original = json.loads((ARCHIVE/'data/blocks.json').read_text())
        current = json.loads((ROOT/'data/blocks.json').read_text())
        self.assertEqual(current['blocks'][:120], original['blocks'])
        self.assertEqual(current['prior_points'], original['prior_points'])
        self.assertEqual(current['classes'], original['classes'])
        frozen = json.loads((ARCHIVE/'code/acquisition-freeze.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'code/acquire.py').read_bytes()).hexdigest(),
                         frozen['code/acquire.py'])

    def test_added_locations_are_independent(self):
        spec = json.loads((ROOT/'data/blocks.json').read_text())
        blocks = spec['blocks']
        self.assertEqual(len(blocks), 158)
        self.assertEqual(spec['split_counts'], {'train':60, 'val':24, 'test':74})
        self.assertEqual(len({b['id'] for b in blocks}), 158)
        for index, block in enumerate(blocks[120:], start=120):
            point = [block['lon']+.5, block['lat']+.5]
            self.assertGreaterEqual(min(distance(point, [b['lon']+.5,b['lat']+.5])
                                        for b in blocks[:index]), 200)
            self.assertGreaterEqual(min(distance(point, p) for p in spec['prior_points']), 100)

    def test_admission_logic_matches_original(self):
        def gate_ast(path):
            tree = ast.parse(path.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
            prefix = []
            for node in function.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='report'
                                                       for t in node.targets):
                    break
                prefix.append(ast.dump(node, include_attributes=False))
            return prefix
        self.assertEqual(gate_ast(ROOT/'code/aggregate.py'), gate_ast(ARCHIVE/'code/aggregate.py'))

    def test_all_acquisition_freeze_files_present_and_exact(self):
        frozen = json.loads((ROOT/'code/acquisition-freeze.json').read_text())
        for name, expected in frozen.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
