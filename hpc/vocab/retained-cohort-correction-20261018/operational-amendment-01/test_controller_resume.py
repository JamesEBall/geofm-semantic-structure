"""Synthetic pending-task handover tests. Author: James Edward Ball."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location('resume_helper', Path(__file__).with_name('controller_resume.py'))
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'}


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if resume.load(path) != value:
            raise ValueError('Immutable record changed')
    else:
        path.write_text(json.dumps(value))


def successful(raw, job_id, indices):
    expected = {f'{job_id}_{i}' for i in indices}
    rows = {x.split('|')[0]: x.split('|')[1:3] for x in raw.strip().splitlines() if x}
    for task, value in rows.items():
        if task in expected and value[0] not in ACTIVE and value != ['COMPLETED', '0:0']:
            raise RuntimeError('Replacement failed')
    return all(rows.get(task) == ['COMPLETED', '0:0'] for task in expected)


class FakeBase:
    def complete(self, job):
        raise AssertionError('This test expects the mapped completion path')


class HandoverSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root / 'original'
        self.here = self.base / 'retained-cohort-correction-20261018'
        self.ledger = self.root / 'output.dispatch'
        self.amend = self.root / 'amendment'
        self.amend.mkdir()
        self.manifest = self.amend / 'manifest.json'
        self.manifest.write_text('{"immutable":"amendment"}')
        self.handover = self.root / 'handover.json'
        self.handover.write_text('{"approved":"pending tasks only"}')
        runner = SimpleNamespace(HERE=self.here, RELEASE=self.base, write_once=write_once, read=resume.load)
        original = SimpleNamespace(Controller=FakeBase, runner=runner, ACTIVE=ACTIVE, accounting_state=successful)
        cls = resume.amended_controller(original)
        self.ctl = cls.__new__(cls)
        self.ctl.ledger = self.ledger
        self.ctl.amendment_root = self.amend
        self.ctl.operation = self.ledger / 'operational-amendments' / resume.sha(self.manifest)
        self.ctl.manifest = self.manifest
        self.ctl.handover = self.handover
        self.ctl.missing_since = {}
        self.job = {'key': 'cpu.extract.unit', 'job_id': '100', 'indices': [0, 1, 2]}
        self.ctl.mapping = {self.job['key']: {'indices': [1, 2], 'original_job_id': '100'}}
        command = ['sbatch', '--nice=1000', '--cpus-per-task=8', '--mem=48G',
                   '--partition=gpu', '--gres=gpu:1', '--array=0,1,2%2',
                   '--dependency=afterok:90', str(self.here / 'worker.sbatch'),
                   str(self.base), '/source', '/output', 'unit', 'extract', 'cpu']
        write_once(self.ledger / self.job['key'] / 'intent.json', {'command': command})
        write_once(self.ledger / self.job['key'] / 'submitted.json', self.job)
        write_once(self.ledger / 'cpu.extract.plan' / 'submitted.json',
                   {'key': 'cpu.extract.plan', 'job_id': '90', 'indices': None})
        self.original_intent_sha = resume.sha(self.ledger / self.job['key'] / 'intent.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_cancelled_running_or_unapproved_or_failed_original_rejected(self):
        cases = [
            '100_0|FAILED|1:0|2026-10-18T00:00:00|1',
            '100_0|CANCELLED|0:0|Unknown|0',
            '100_1|CANCELLED|0:0|2026-10-18T00:00:00|0',
            '100_1|CANCELLED|0:0|Unknown|1|0|None assigned',
            '100_1|CANCELLED|0:0|None|0|8|None assigned',
            '100_1|CANCELLED|0:0|None|0|0|node01',
            '100_1|CANCELLED|0:0|None|0',
            '100_1|CANCELLED|0:0|2026-10-18T18:37:58|0|0|None assigned|2026-10-18T18:37:58',
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                resume.classify_original(self.job, [1, 2], resume.accounting(raw), ACTIVE)

    def test_recorded_pending_cancellation_has_no_start_or_allocation(self):
        # Read-only sacct observation, 18 October 2026. NNodes was 1 (requested).
        raw = '177108_0|CANCELLED by 27800325|0:0|None|0|0|None assigned|2026-10-18T18:37:58|'
        job = {'job_id': '177108', 'indices': [0]}
        rows = resume.accounting(raw)
        self.assertEqual(rows['177108_0']['allocated_cpus'], '0')
        self.assertEqual(rows['177108_0']['nodes'], 'None assigned')
        self.assertEqual(resume.classify_original(job, [0], rows, ACTIVE), ([], [0], []))

    def test_replacement_waits_for_all_uncancelled_originals(self):
        def execute(cmd):
            if cmd[0] == 'sacct':
                return ('100_0|RUNNING|0:0|2026-10-18T00:00:00|10\n'
                        '100_1|CANCELLED|0:0|Unknown|0|0|None assigned\n100_2|CANCELLED|0:0|Unknown|0|0|None assigned\n')
            if cmd[0] == 'squeue':
                return '100_0|RUNNING\n'
            raise AssertionError('Submission before original completion')
        self.ctl.raw_execute = execute
        self.assertFalse(self.ctl.complete(self.job))

    def test_union_completion_exactly_once_and_dependency_translation(self):
        commands = []
        def execute(cmd):
            commands.append(cmd)
            if cmd[0] == 'sbatch':
                return '200\n'
            jid = cmd[cmd.index('-j') + 1]
            if jid == '90':
                return '90|COMPLETED|0:0\n'
            if jid == '100_0,200':
                return '100_0|COMPLETED|0:0\n200_1|COMPLETED|0:0\n200_2|COMPLETED|0:0\n'
            if jid == '100':
                return ('100_0|COMPLETED|0:0|2026-10-18T00:00:00|10\n'
                        '100_1|CANCELLED|0:0|Unknown|0|0|None assigned\n100_2|CANCELLED|0:0|Unknown|0|0|None assigned\n')
            return '200_1|COMPLETED|0:0\n200_2|COMPLETED|0:0\n'
        self.ctl.raw_execute = execute
        self.assertTrue(self.ctl.complete(self.job))
        self.assertTrue(self.ctl.complete(self.job))
        submits = [c for c in commands if c[0] == 'sbatch']
        self.assertEqual(len(submits), 1)
        self.assertIn('--array=1,2%2', submits[0])
        self.assertIn(str(self.amend / 'worker_amended.sbatch'), submits[0])
        self.assertEqual(submits[0][-7:], [str(self.base), '/source', '/output', 'unit', 'extract', 'cpu', str(self.amend)])
        self.assertFalse((self.ledger / self.job['key'] / 'complete.json').exists())
        self.assertEqual(resume.sha(self.ledger / self.job['key'] / 'intent.json'), self.original_intent_sha)
        cmd = ['sbatch', '--dependency=afterok:100', str(self.here / 'worker.sbatch'),
               str(self.base), '/source', '/output', 'finalize', 'extract', 'cpu']
        adapted = self.ctl.adapt(cmd)
        self.assertIn('--dependency=afterok:100_0:200', adapted)
        self.assertNotIn('--dependency=afterok:100', adapted)
        resolved, evidence = self.ctl.resolve_dependencies(adapted)
        self.assertFalse(any(c.startswith('--dependency=') for c in resolved))
        self.assertEqual([e['expected_ids'] for e in evidence], [['100_0'], ['200_1', '200_2']])
        request = resume.load(self.ctl.operation / 'replacements' / self.job['key'] / 'intent.json')
        self.assertFalse(any(c.startswith('--dependency=') for c in submits[0]))
        self.assertEqual(request['dependency_resolution'][0]['decision'], 'omit_satisfied')

    def test_uncertain_replacement_never_resubmitted(self):
        calls = []
        def fail(command):
            if command[0] == 'sacct':
                return '90|COMPLETED|0:0\n'
            calls.append(command)
            raise RuntimeError('Connection lost')
        self.ctl.raw_execute = fail
        with self.assertRaisesRegex(RuntimeError, 'Connection lost'):
            self.ctl.replacement(self.job, [1, 2])
        with self.assertRaisesRegex(RuntimeError, 'Uncertain operational submission'):
            self.ctl.replacement(self.job, [1, 2])
        self.assertEqual(len(calls), 1)

    def test_pending_task_that_completes_is_preserved(self):
        raw = ''.join(f'100_{i}|COMPLETED|0:0|2026-10-18T00:00:00|2\n' for i in range(3))
        success, cancelled, pending = resume.classify_original(self.job, [1, 2], resume.accounting(raw), ACTIVE)
        self.assertEqual(len(success), 3)
        self.assertEqual(cancelled, [])
        self.assertEqual(pending, [])

    def test_future_effective_submission_records_overlay_without_rewriting_original_intent(self):
        key = 'cpu.extract.finalize'
        command = ['sbatch', '--nice=1000', '--cpus-per-task=8', '--mem=48G',
                   str(self.here / 'worker.sbatch'), str(self.base), '/source', '/output',
                   'finalize', 'extract', 'cpu']
        path = self.ledger / key / 'intent.json'
        write_once(path, {'command': command})
        before = resume.sha(path)
        self.ctl.raw_execute = lambda c: '300\n'
        self.assertEqual(self.ctl.effective_execute(command), '300')
        self.assertEqual(resume.sha(path), before)
        effective = resume.load(self.ctl.operation / 'future' / key / 'intent.json')
        self.assertEqual(effective['original_command'], command)
        self.assertIn(str(self.amend / 'worker_amended.sbatch'), effective['effective_command'])
        self.assertEqual(effective['effective_command'][-1], str(self.amend))

    def test_expired_historical_plan_removed_with_persisted_exact_proof(self):
        key = 'historical.plan'
        write_once(self.ledger / key / 'submitted.json',
                   {'key': key, 'job_id': '177230', 'indices': None})
        command = ['sbatch', '--dependency=afterok:177230', str(self.here / 'worker.sbatch'),
                   str(self.base), '/source', '/output', 'unit', 'extract', 'cpu']
        calls = []
        def execute(cmd):
            calls.append(cmd)
            if cmd[0] == 'sacct':
                return '177230|COMPLETED|0:0\n'
            self.assertFalse(any(c.startswith('--dependency=') for c in cmd))
            return '400\n'
        self.ctl.raw_execute = execute
        directory = self.ctl.operation / 'historical_test'
        request = {'original_command': command}
        self.ctl.submit_effective(directory, request)
        self.ctl.submit_effective(directory, request)
        self.assertEqual([c[0] for c in calls], ['sacct', 'sbatch'])
        proof = resume.load(directory / 'intent.json')['dependency_resolution'][0]
        self.assertEqual(proof['expected_ids'], ['177230'])
        self.assertEqual(proof['accounting'], {'177230': {'state': 'COMPLETED', 'exit': '0:0'}})
        self.assertEqual(proof['decision'], 'omit_satisfied')

    def test_array_dependency_requires_every_exact_expected_task(self):
        command = ['sbatch', '--dependency=afterok:100', 'worker']
        cases = [
            '100|COMPLETED|0:0\n',  # Parent summary proves no individual task.
            '100_0|COMPLETED|0:0\n100_1|COMPLETED|0:0\n',
            '100_0|COMPLETED|0:0\n100_1|COMPLETED|0:0\n100_2|FAILED|1:0\n',
            '100_0|COMPLETED|0:0\n100_1|COMPLETED|0:0\n100_2|COMPLETED|1:0\n',
        ]
        for raw in cases:
            self.ctl.raw_execute = lambda cmd, raw=raw: raw
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.ctl.resolve_dependencies(command)
        raw = ''.join(f'100_{i}|COMPLETED|0:0\n' for i in range(3))
        self.ctl.raw_execute = lambda cmd: raw
        effective, proof = self.ctl.resolve_dependencies(command)
        self.assertEqual(effective, ['sbatch', 'worker'])
        self.assertEqual(proof[0]['expected_ids'], ['100_0', '100_1', '100_2'])

    def test_active_dependency_retained_and_unrelated_completion_rejected(self):
        command = ['sbatch', '--dependency=afterok:90', 'worker']
        self.ctl.raw_execute = lambda cmd: '90|RUNNING|0:0\n'
        effective, proof = self.ctl.resolve_dependencies(command)
        self.assertEqual(effective, command)
        self.assertEqual(proof[0]['decision'], 'retain_active')
        self.ctl.raw_execute = lambda cmd: '91|COMPLETED|0:0\n'
        with self.assertRaises(RuntimeError):
            self.ctl.resolve_dependencies(command)

    def test_individual_survivor_requires_exact_task_success(self):
        command = ['sbatch', '--dependency=afterok:100_0', 'worker']
        self.ctl.raw_execute = lambda cmd: '100|COMPLETED|0:0\n'
        with self.assertRaises(RuntimeError):
            self.ctl.resolve_dependencies(command)
        self.ctl.raw_execute = lambda cmd: '100_0|COMPLETED|0:0\n100_1|CANCELLED|0:0\n'
        effective, proof = self.ctl.resolve_dependencies(command)
        self.assertEqual(effective, ['sbatch', 'worker'])
        self.assertEqual(proof[0]['expected_ids'], ['100_0'])


if __name__ == '__main__':
    unittest.main()
