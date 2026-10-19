"""Synthetic Slurm-controller tests. Author: James Edward Ball."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller


class FakeSlurm:
    def __init__(self):
        self.calls = []
        self.jobs = {}

    def __call__(self, command):
        self.calls.append(command)
        if command[0] == 'sbatch':
            job_id = str(100 + len(self.jobs))
            array = next((s for s in command if s.startswith('--array=')), None)
            ids = None if array is None else [int(x) for x in array.split('=', 1)[1].split('%')[0].split(',')]
            self.jobs[job_id] = ids
            return job_id + '\n'
        if command[0] == 'sacct':
            if '--format=JobID%80,State,ExitCode' not in command:
                raise AssertionError('Array accounting must use JobID, not underlying JobIDRaw')
            job_id = command[command.index('-j') + 1]
            indices = self.jobs[job_id]
            ids = [job_id] if indices is None else [f'{job_id}_{i}' for i in indices]
            return ''.join(f'{i}|COMPLETED|0:0|\n' for i in ids)
        raise AssertionError(command)


class ControllerSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.args = SimpleNamespace(source_run=self.root / 'source', output=self.root / 'corrected',
                                    worker_time_limit='04:00:00', poll_seconds=20, stop_after_bins=True)
        self.code_patch = patch.object(controller.runner, 'code_hashes', return_value={'code.py': 'pinned'})
        self.code_patch.start()

    def tearDown(self):
        self.code_patch.stop()
        self.tmp.cleanup()

    def test_exact_array_accounting_and_failures(self):
        self.assertFalse(controller.accounting_state('100_0|COMPLETED|0:0\n', '100', [0, 2]))
        self.assertFalse(controller.accounting_state('100|COMPLETED|0:0\n', '100', [0, 2]))
        self.assertTrue(controller.accounting_state(
            '100_0|COMPLETED|0:0\n100_2|COMPLETED|0:0\n', '100', [0, 2]))
        for failure in ('FAILED|1:0', 'COMPLETED|1:0', 'CANCELLED by 1|0:0', 'TIMEOUT|0:0'):
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                controller.accounting_state('100_0|' + failure + '\n', '100', [0])

    def test_dispatch_reuses_acknowledged_job(self):
        fake = FakeSlurm()
        ctl = controller.Controller(self.args, execute=fake)
        first = ctl.dispatch('unit', 'analysis_readouts', 'cpu', indices=[0, 2])
        second = ctl.dispatch('unit', 'analysis_readouts', 'cpu', indices=[0, 2])
        self.assertEqual(first, second)
        self.assertEqual(len(fake.jobs), 1)
        with self.assertRaises(ValueError):
            ctl.dispatch('unit', 'analysis_readouts', 'cpu', indices=[0])

    def test_uncertain_submission_blocks_retry(self):
        def fail(_):
            raise RuntimeError('network lost after submission')
        ctl = controller.Controller(self.args, execute=fail)
        with self.assertRaisesRegex(RuntimeError, 'network lost'):
            ctl.dispatch('prepare', 'features_fit', 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'Uncertain submission'):
            ctl.dispatch('prepare', 'features_fit', 'cpu')

    def test_resource_caps_and_inference_memory(self):
        fake = FakeSlurm()
        ctl = controller.Controller(self.args, execute=fake)
        ctl.dispatch('unit', 'analysis_readouts', 'cpu', indices=[0, 1])
        ctl.dispatch('unit', 'extract', 'cpu', indices=[0, 1])
        ctl.dispatch('unit', 'analysis_inference', 'full', indices=[0])
        commands = [c for c in fake.calls if c[0] == 'sbatch']
        self.assertIn('--array=0,1%24', commands[0])
        self.assertIn('--array=0,1%2', commands[1])
        self.assertIn('--gres=gpu:1', commands[1])
        self.assertIn('--mem=160G', commands[2])
        self.assertIn('--cpus-per-task=8', commands[2])
        self.assertTrue(all('--nice=1000' in c for c in commands))

    def test_stop_after_bins_restart_reuses_whole_ledger(self):
        fake = FakeSlurm()
        ctl = controller.Controller(self.args, execute=fake)
        def missing(*args):
            return [] if (ctl.ledger / 'cpu.features_fit.unit/complete.json').exists() else [0]
        with patch.object(ctl, 'context', return_value=SimpleNamespace()), \
             patch.object(controller.runner, 'load_plan', return_value=[{'id': 'fit0'}]), \
             patch.object(controller.runner, 'require_complete'), \
             patch.object(controller, 'missing_indices', side_effect=missing):
            ctl.run()
            submitted = len(fake.jobs)
            ctl.run()
        self.assertEqual(submitted, 4)  # preparation, plan, unit, finalise
        self.assertEqual(len(fake.jobs), submitted)
        self.assertFalse((ctl.ledger / 'cpu.extract.plan').exists())

    def test_no_submit_mode_reports_topology_without_calls_or_writes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(subprocess_proxy, 'run', side_effect=AssertionError):
            controller.main(['--source-run', str(self.args.source_run), '--output', str(self.args.output)])
        doc = json.loads(output.getvalue())
        self.assertEqual(doc['slurm_commands_issued'], 0)
        self.assertEqual(doc['cpu_unit_concurrency'], 24)
        self.assertEqual(doc['gpu_unit_concurrency'], 2)
        self.assertEqual(doc['inference_memory_gib'], 160)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_full_controller_exact_dependency_order(self):
        self.args.stop_after_bins = False
        ctl = controller.Controller(self.args, execute=FakeSlurm())
        events, jobs = [], {}

        def event(kind, action, stage, phase, after=(), finalize=None):
            key = f'{phase}.{stage}.{action}'
            events.append({'kind': kind, 'key': key, 'after': [j['key'] for j in after],
                           'finalize': finalize})
            job = {'key': key, 'job_id': str(1000 + len(events))}
            jobs[key] = job
            return job

        def command(action, stage, phase, after=()):
            return event('command', action, stage, phase, after)

        def stage(name, phase, *, finalize=True, after=()):
            return event('stage', 'stage', name, phase, after, finalize)

        def start_gpu(after):
            return event('parallel', 'unit', 'extract', 'cpu', [after])

        def wait(job):
            events.append({'kind': 'wait', 'key': job['key'], 'after': [], 'finalize': None})
            return job

        with patch.object(ctl, 'context', return_value=SimpleNamespace()), \
             patch.object(ctl, 'command', side_effect=command), \
             patch.object(ctl, 'stage', side_effect=stage), \
             patch.object(ctl, 'start_gpu', side_effect=start_gpu), \
             patch.object(ctl, 'wait', side_effect=wait), \
             patch.object(controller.runner, 'require_complete'), \
             patch.object(controller.runner, 'digest', return_value='identity'):
            ctl.run()

        analysis = list(controller.runner.ANALYSES)
        expected = ['cpu.features_fit.prepare', 'cpu.features_fit.stage', 'cpu.extract.unit',
                    'cpu.features.stage', 'cpu.features.freeze',
                    *[f'cpu.{s}.stage' for s in analysis], 'cpu.extract.unit',
                    'cpu.extract.finalize', 'full.features.freeze',
                    *[f'full.{s}.stage' for s in analysis],
                    'full.analysis_inference.stage', 'full.analysis_qualify.stage']
        self.assertEqual([e['key'] for e in events], expected)
        by_key = {e['key']: e for e in events if e['kind'] != 'wait'}
        for name in analysis:
            self.assertFalse(by_key[f'cpu.{name}.stage']['finalize'])
            self.assertEqual(by_key[f'cpu.{name}.stage']['after'], ['cpu.features.freeze'])
            self.assertEqual(by_key[f'full.{name}.stage']['after'],
                             ['full.features.freeze', *[f'cpu.{s}.stage' for s in analysis]])
        self.assertEqual(by_key['full.features.freeze']['after'],
                         ['cpu.features.stage', 'cpu.extract.finalize'])
        self.assertEqual(by_key['full.analysis_inference.stage']['after'],
                         [f'full.{s}.stage' for s in analysis])
        self.assertEqual(by_key['full.analysis_qualify.stage']['after'],
                         ['full.analysis_inference.stage'])
        self.assertTrue((ctl.ledger / 'complete.json').exists())

    def test_missing_full_indices_validate_existing_cpu_receipts(self):
        receipt = self.root / 'cpu-receipt.json'
        receipt.write_text('{}')
        absent = self.root / 'missing.json'
        units = [{'id': 'cpu', 'fold': 0}, {'id': 'gpu', 'fold': 0}]
        with patch.object(controller.runner, 'receipt_path', side_effect=[receipt, absent]), \
             patch.object(controller.runner, 'validate_receipt', return_value={'unit': units[0]}) as verify:
            self.assertEqual(controller.missing_indices(SimpleNamespace(), 'analysis_readouts', units), [1])
            verify.assert_called_once()


subprocess_proxy = controller.subprocess

if __name__ == '__main__':
    unittest.main()
