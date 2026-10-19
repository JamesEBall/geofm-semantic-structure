"""Safety tests for the derivative namespace and completion receipts.

Author: James Edward Ball. These tests use temporary synthetic files only.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('retained_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_same_nested_and_symlink_namespaces(self):
        source = self.root / 'source'
        source.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(source)
        for target in (source, source / 'child', self.root, alias):
            with self.subTest(target=target), self.assertRaises(ValueError):
                runner.separate(source, target)
        self.assertEqual(runner.separate(source, self.root / 'derivative')[0], source)

    def test_source_absolute_release_hash_checked(self):
        source = self.root / 'source'
        source.mkdir()
        code = self.root / 'old-release.py'
        code.write_text('frozen source code')
        manifest = source / 'manifest.json'
        manifest.write_text(json.dumps({'files': {str(code): runner.digest(code)}}))
        runner.verified_manifest(source, manifest)
        code.write_text('changed source code')
        with self.assertRaisesRegex(ValueError, 'Source freeze mismatch'):
            runner.verified_manifest(source, manifest)

    def test_rejects_relative_manifest_escape_and_empty_freeze(self):
        source = self.root / 'source'
        source.mkdir()
        target = self.root / 'external'
        target.write_text('x')
        manifest = source / 'manifest.json'
        for files in ({'../external': runner.digest(target)}, {}):
            manifest.write_text(json.dumps({'files': files}))
            with self.assertRaises(ValueError):
                runner.verified_manifest(source, manifest)

    def test_write_once_preserves_first_record(self):
        path = self.root / 'freeze.json'
        runner.write_once(path, {'a': 1})
        runner.write_once(path, {'a': 1})
        with self.assertRaises(ValueError):
            runner.write_once(path, {'a': 2})
        self.assertEqual(runner.read(path), {'a': 1})

    def fake_context(self):
        output = self.root / 'derivative'
        output.mkdir()
        runner.write_once(output / '_retained/identity.json', {'source': 'fixed'})
        return SimpleNamespace(run=output, p=lambda *v: output.joinpath(*v),
                               cfg={'analysis_input_freeze': 'f3_cpu'})

    def test_exact_receipts_require_ids_and_unit_parameters(self):
        ctx = self.fake_context()
        out = ctx.p('analysis_readouts/unit.npz')
        out.parent.mkdir()
        out.write_bytes(b'synthetic scores')
        unit = {'id': 'a', 'fold': 1}
        runner.record(ctx, 'analysis_readouts', 'a', unit, [out])
        runner.exact_receipts(ctx, 'analysis_readouts', [unit])
        for units in ([{'id': 'b', 'fold': 1}], [dict(unit, fold=2)], [unit, {'id': 'b'}]):
            with self.subTest(units=units), self.assertRaises(ValueError):
                runner.exact_receipts(ctx, 'analysis_readouts', units)
        out.write_bytes(b'changed scores')
        with self.assertRaisesRegex(ValueError, 'Completed output changed'):
            runner.exact_receipts(ctx, 'analysis_readouts', [unit])

    def test_receipts_cannot_register_source_symlink_outputs(self):
        ctx = self.fake_context()
        original = self.root / 'source.npy'
        original.write_bytes(b'original')
        alias = ctx.p('alias.npy')
        alias.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'outside the derivative'):
            runner.record(ctx, 'extract', 'a', {'id': 'a'}, [alias])
        self.assertEqual(original.read_bytes(), b'original')

    def test_identity_change_invalidates_receipts(self):
        ctx = self.fake_context()
        runner.record(ctx, 'features', 'a', {'id': 'a'}, [])
        ctx.p('_retained/identity.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'different derivative'):
            runner.validate_receipt(ctx, 'features', 'a')

    def test_cpu_plan_excludes_pending_dofa_even_if_refresh_finishes_early(self):
        ctx = self.fake_context()
        ctx.store = SimpleNamespace(write_units=lambda *a: None)
        args = SimpleNamespace(stage='analysis_readouts', phase='cpu')
        units = [{'id': 'a', 'route': 'r1-clay'}, {'id': 'b', 'route': 'r1-dofa'},
                 {'id': 'c', 'route': 'r1rgb224-dofa'}]
        with patch.object(runner, 'dependencies'), patch.object(
                runner, 'spec', return_value=SimpleNamespace(plan=lambda c: units)):
            runner.plan(ctx, args)
        self.assertEqual([u['id'] for u in runner.load_plan(ctx, args.stage, 'cpu')], ['a', 'c'])

    def test_full_plan_requires_unchanged_cpu_units(self):
        ctx = self.fake_context()
        ctx.store = SimpleNamespace(write_units=lambda *a: None)
        args = SimpleNamespace(stage='analysis_readouts', phase='full')
        original = [{'id': 'a', 'route': 'r1-clay', 'fold': 1}]
        runner.write_once(runner.plan_path(ctx, args.stage, 'cpu'), original)
        altered = [{'id': 'a', 'route': 'r1-clay', 'fold': 2}]
        with patch.object(runner, 'dependencies'), patch.object(
                runner, 'spec', return_value=SimpleNamespace(plan=lambda c: altered)):
            with self.assertRaisesRegex(ValueError, 'CPU unit plan changed'):
                runner.plan(ctx, args)

    def test_f2_rejects_changed_inherited_shard_or_checkpoint(self):
        ctx = self.fake_context()
        source = self.root / 'source'
        source.mkdir()
        paths = [source / 'manifest.json', source / 'DOFA.pth']
        for path in paths:
            path.write_bytes(b'frozen input')
        runner.write_once(ctx.p('freeze_f2/manifest.json'), {
            'files': {str(p): runner.digest(p) for p in paths}})
        runner.verify_frozen_paths(ctx, paths)
        for path in paths:
            with self.subTest(path=path):
                path.write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError, 'changed since derivative F2'):
                    runner.verify_frozen_paths(ctx, [path])
                path.write_bytes(b'frozen input')


if __name__ == '__main__':
    unittest.main()
