"""Submission contract tests. James Edward Ball."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('hpc_run', Path(__file__).resolve().parents[1] / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class SubmissionTests(unittest.TestCase):
    def test_initialise_preserves_existing_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'new'
            run.initialise(root)
            self.assertTrue((root / 'fresh-01/package').is_symlink())
            self.assertTrue((root / 'global-01/data/blocks.json').is_file())
            with self.assertRaises(FileExistsError):
                run.initialise(root)

    def test_dependency_graph_and_receipts(self):
        calls = []
        def fake(cmd, **kw):
            calls.append((cmd, kw))
            return str(100 + len(calls)) + '\n'
        with tempfile.TemporaryDirectory() as d, patch.object(run.subprocess, 'check_output', side_effect=fake):
            root = Path(d)
            f = run.submit(root, 'fresh')
            g = run.submit(root, 'global')
            run.submit(root, 'supplements')
            global_gpu = next(c for c, _ in calls if str(root / 'global-01/gpu.sbatch') in c)
            self.assertIn('--dependency=afterok:' + g['admission'] + ':' + f['cuda_gate'], global_gpu)
            first_supp = next(c for c, _ in calls if str(root / 'supplement-controls/prepare.sbatch') in c)
            self.assertIn('--dependency=afterok:' + ':'.join([f['analysis'], f['interventions'], f['internal_analysis'], g['analysis']]), first_supp)
            self.assertEqual(len(calls), 23)
            self.assertEqual(json.loads((root / 'submitted-fresh.json').read_text()), f)
            with self.assertRaises(FileExistsError):
                run.submit(root, 'fresh')

    def test_partial_failure_records_successful_submission(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run.subprocess, 'check_output', side_effect=['123\n', RuntimeError('scheduler unavailable')]):
            root = Path(d)
            with self.assertRaises(RuntimeError):
                run.submit(root, 'fresh')
            self.assertEqual(json.loads((root / 'submitted-fresh.json').read_text()), {'prepare': '123'})

    def test_dry_run_never_submits_or_writes_receipt(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run.subprocess, 'check_output') as call, patch('builtins.print'):
            run.submit(Path(d), 'fresh', dry_run=True)
            call.assert_not_called()
            self.assertFalse((Path(d) / 'submitted-fresh.json').exists())


if __name__ == '__main__':
    unittest.main()
