#!/usr/bin/env python3
"""Append-only pending-task reconciliation for the batched F2 lookup amendment.

Author: James Edward Ball. This program never cancels or stops a Slurm job.
The parent must approve the handover and stop the original controller first.
Only cancelled tasks with accounting proof that they never started can be
replaced. Uncancelled original work finishes before replacement submission.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def accounting(raw):
    records = {}
    for line in raw.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 5:
            records[fields[0]] = {'state': fields[1].split()[0].rstrip('+'),
                                   'exit': fields[2], 'start': fields[3], 'elapsed': fields[4],
                                   'allocated_cpus': fields[5] if len(fields) > 5 else None,
                                   'nodes': fields[6] if len(fields) > 6 else None,
                                   'end': fields[7] if len(fields) > 7 else None}
    return records


def classify_original(job, approved_indices, records, active_states):
    """Return success IDs/cancelled indices/pending IDs; every actual failure is fatal."""
    expected = {f'{job["job_id"]}_{i}': i for i in job['indices']}
    if set(approved_indices) - set(job['indices']):
        raise ValueError('Handover approves indices outside the original frozen dispatch')
    successful, cancelled, pending = [], [], []
    for task, index in expected.items():
        row = records.get(task)
        if row is None or row['state'] in active_states:
            pending.append(task)
        elif row['state'] == 'COMPLETED' and row['exit'] == '0:0':
            successful.append(task)
        elif row['state'] == 'CANCELLED' and index in approved_indices:
            # Observed pending cancellation 177108_0 has Start=None, ElapsedRaw=0,
            # AllocCPUS=0 and NodeList="None assigned". NNodes=1 is its requested
            # allocation, so NNodes cannot prove execution. Timestamped Start,
            # including Start=End, remains ambiguous and fails closed.
            if (row['start'] not in ('Unknown', 'None', '', 'N/A') or row['elapsed'] != '0'
                    or row.get('allocated_cpus') != '0'
                    or row.get('nodes') not in ('None assigned', 'None', '(null)')):
                raise RuntimeError(f'Cancelled task lacks proof it never started; replacement forbidden: {task}')
            cancelled.append(index)
        else:
            raise RuntimeError(f'Original task failure is not an approved pending cancellation: {task}: {row}')
    return successful, cancelled, pending


def import_original(base):
    root = base / 'retained-cohort-correction-20261018'
    for name, file in [('run', root / 'run.py'), ('original_retained_controller', root / 'controller.py')]:
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    original = sys.modules['original_retained_controller']
    if Path(original.runner.__file__).resolve() != (root / 'run.py').resolve():
        raise ValueError('Controller imported a different scientific runner')
    return original


def validate_amendment(manifest, base):
    manifest = Path(manifest).resolve()
    sys.path.insert(0, str(manifest.parent))
    spec = importlib.util.spec_from_file_location('retained_amended_worker_validation',
                                                manifest.parent / 'amended_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_manifest(manifest, base)


def amended_controller(original):
    runner = original.runner

    class ResumeController(original.Controller):
        def __init__(self, args, manifest, handover, execute=None):
            super().__init__(args, execute=execute)
            self.raw_execute = self.execute
            self.manifest = Path(manifest).resolve()
            self.handover = Path(handover).resolve()
            self.amendment_root = self.manifest.parent
            self.operation = self.ledger / 'operational-amendments' / sha(self.manifest)
            self.operation.mkdir(parents=True, exist_ok=True)
            doc = load(self.handover)
            if doc['base_release'] != str(runner.RELEASE):
                raise ValueError('Handover scientific release differs')
            if doc['identity_sha256'] != sha(self.output / '_retained/identity.json'):
                raise ValueError('Handover scientific identity differs')
            if doc['configuration_sha256'] != sha(self.ledger / 'configuration.json'):
                raise ValueError('Handover original dispatch contract differs')
            runner.write_once(self.operation / 'handover.json', doc)
            self.mapping = {}
            for item in doc['replacements']:
                key = item['original_key']
                if key in self.mapping:
                    raise ValueError('Duplicate original array in handover')
                intent = self.ledger / key / 'intent.json'
                submitted = self.ledger / key / 'submitted.json'
                if sha(intent) != item['intent_sha256'] or sha(submitted) != item['submitted_sha256']:
                    raise ValueError('Original dispatch records changed')
                job = load(submitted)
                if job['job_id'] != item['original_job_id'] or job['indices'] is None:
                    raise ValueError('Handover names a different original job or a scalar job')
                if len(item['indices']) != len(set(item['indices'])):
                    raise ValueError('Duplicate cancellation index')
                pending = item['pending_snapshot']
                if any(pending.get(f'{job["job_id"]}_{i}') != 'PENDING' for i in item['indices']):
                    raise ValueError('Each approved index needs an explicit PENDING snapshot')
                self.mapping[key] = item
            self.execute = self.effective_execute

        def dependencies_for(self, job_id):
            for key, mapping in self.mapping.items():
                if mapping['original_job_id'] == job_id:
                    path = self.operation / 'reconciled' / key / 'complete.json'
                    if not path.exists():
                        raise RuntimeError('A reconciled original array is not complete')
                    return load(path)['effective_dependency_ids']
            return [job_id]

        def adapt(self, command):
            cmd = list(command)
            old_worker = str(runner.HERE / 'worker.sbatch')
            if cmd.count(old_worker) != 1:
                raise ValueError('Submission does not contain exactly the original worker script')
            pos = cmd.index(old_worker)
            if cmd[pos + 1] != str(runner.RELEASE):
                raise ValueError('Submission changes the scientific release')
            if len(cmd) != pos + 7:
                raise ValueError('Expected the six original worker arguments')
            cmd[pos] = str(self.amendment_root / 'worker_amended.sbatch')
            for i, value in enumerate(cmd):
                if value.startswith('--dependency='):
                    prefix = '--dependency=afterok:'
                    if not value.startswith(prefix):
                        raise ValueError('Unsupported dependency change')
                    ids = value[len(prefix):].split(':')
                    effective = list(dict.fromkeys(x for j in ids for x in self.dependencies_for(j)))
                    cmd[i] = prefix + ':'.join(effective)
            # Slurm executes a spool copy; the script's dirname is not its release path.
            cmd.append(str(self.amendment_root))
            return cmd

        def submit_once(self, directory, request):
            intent, submitted = directory / 'intent.json', directory / 'submitted.json'
            if submitted.exists():
                if load(intent) != request:
                    raise ValueError('Operational submission contract changed')
                return load(submitted)
            if intent.exists():
                raise RuntimeError(f'Uncertain operational submission; reconcile before continuing: {intent}')
            runner.write_once(intent, request)
            result = self.raw_execute(request['effective_command']).strip()
            if not re.fullmatch(r'\d+(;[^\s;]+)?', result):
                raise RuntimeError('Uncertain operational Slurm acknowledgement')
            record = {'job_id': result.split(';')[0], 'manifest_sha256': sha(self.manifest),
                      'request_sha256': sha(intent)}
            runner.write_once(submitted, record)
            return record

        def dependency_contract(self, job_id):
            """Get exact expected IDs from immutable dispatch records, never a parent summary."""
            contracts = []
            for path in sorted(self.ledger.glob('*/submitted.json')):
                job = load(path)
                contracts.append((str(job['job_id']), job['indices'], path, None))
            for path in sorted((self.operation / 'replacements').glob('*/submitted.json')):
                intent = path.with_name('intent.json')
                contracts.append((str(load(path)['job_id']), load(intent)['original_indices'], path, intent))
            matches = []
            for parent, indices, path, intent in contracts:
                tasks = [parent] if indices is None else [f'{parent}_{i}' for i in indices]
                if job_id == parent or job_id in tasks:
                    matches.append({'expected_ids': tasks if job_id == parent else [job_id],
                                    'submitted_path': str(path), 'submitted_sha256': sha(path),
                                    'replacement_intent_sha256': sha(intent) if intent else None})
            if len(matches) != 1:
                raise RuntimeError(f'Dependency lacks one exact dispatch contract: {job_id}')
            return matches[0]

        def resolve_dependencies(self, command):
            """Persist proof before omitting satisfied, possibly expired Slurm dependencies."""
            flags = [c for c in command if c.startswith('--dependency=')]
            if not flags:
                return command, []
            if len(flags) != 1 or not flags[0].startswith('--dependency=afterok:'):
                raise ValueError('Expected one afterok dependency flag')
            ids = list(dict.fromkeys(flags[0].split(':')[1:]))
            contracts = {j: self.dependency_contract(j) for j in ids}
            query = ['sacct', '-X', '--array', '--noheader', '--parsable2', '-j', ','.join(ids),
                     '--format=JobID%80,State%35,ExitCode']
            rows = {}
            for line in self.raw_execute(query).splitlines():
                fields = line.strip().split('|')
                if len(fields) < 3:
                    continue
                row = {'state': fields[1].split()[0].rstrip('+'), 'exit': fields[2]}
                if fields[0] in rows and rows[fields[0]] != row:
                    raise RuntimeError(f'Conflicting dependency accounting: {fields[0]}')
                rows[fields[0]] = row
            retained, evidence = [], []
            for job_id, contract in contracts.items():
                expected = contract['expected_ids']
                if any(task not in rows for task in expected):
                    raise RuntimeError(f'Incomplete exact dependency accounting: {job_id}')
                selected = {task: rows[task] for task in expected}
                for task, row in selected.items():
                    if row != {'state': 'COMPLETED', 'exit': '0:0'} and row['state'] not in original.ACTIVE:
                        raise RuntimeError(f'Dependency failed: {task}: {row}')
                satisfied = all(row == {'state': 'COMPLETED', 'exit': '0:0'} for row in selected.values())
                if not satisfied:
                    retained.append(job_id)
                evidence.append({'dependency_id': job_id, **contract, 'accounting_query': query,
                                 'accounting': selected, 'decision': 'omit_satisfied' if satisfied else 'retain_active'})
            effective = [c for c in command if c != flags[0]]
            if retained:
                effective.insert(command.index(flags[0]), '--dependency=afterok:' + ':'.join(retained))
            return effective, evidence

        def submit_effective(self, directory, request):
            # Reuse the immutable proof on subsequent polls and restarts. An intent
            # without an acknowledgement still fails in submit_once; never resubmit.
            intent = directory / 'intent.json'
            if intent.exists():
                saved = load(intent)
                if {k: saved.get(k) for k in request} != request:
                    raise ValueError('Operational submission contract changed')
                return self.submit_once(directory, saved)
            command, evidence = self.resolve_dependencies(self.adapt(request['original_command']))
            return self.submit_once(directory, {**request, 'effective_command': command,
                                                'dependency_resolution': evidence})

        def effective_execute(self, command):
            if command[0] != 'sbatch':
                return self.raw_execute(command)
            # Base dispatch has already written its original immutable intent.
            old_worker = str(runner.HERE / 'worker.sbatch')
            pos = command.index(old_worker)
            action, stage, phase = command[pos + 4:pos + 7]
            key = f'{phase}.{stage}.{action}'
            request = {'original_command': command,
                       'original_intent_sha256': sha(self.ledger / key / 'intent.json'),
                       'handover_sha256': sha(self.handover)}
            record = self.submit_effective(self.operation / 'future' / key, request)
            return record['job_id']

        def replacement(self, job, indices):
            original_intent = load(self.ledger / job['key'] / 'intent.json')
            command = list(original_intent['command'])
            # All uncancelled original tasks have succeeded before this call.
            # Verify the exact plan completion before omitting its expired dependency.
            limit = 2 if job['key'] == 'cpu.extract.unit' else 24
            command = ['--array=' + ','.join(map(str, indices)) + f'%{limit}'
                       if c.startswith('--array=') else c for c in command]
            request = {'original_job': job, 'original_indices': indices,
                       'original_intent_sha256': sha(self.ledger / job['key'] / 'intent.json'),
                       'original_command': command, 'handover_sha256': sha(self.handover)}
            record = self.submit_effective(self.operation / 'replacements' / job['key'], request)
            return {'key': job['key'], 'job_id': record['job_id'], 'indices': indices}

        def complete(self, job):
            if job['key'] not in self.mapping:
                return super().complete(job)
            record_path = self.operation / 'reconciled' / job['key'] / 'complete.json'
            if record_path.exists():
                if load(record_path)['original_job'] != job:
                    raise ValueError('Reconciled completion names a different job')
                return True
            mapping = self.mapping[job['key']]
            raw = self.raw_execute(['sacct', '-X', '--array', '--noheader', '--parsable2',
                                    '-j', job['job_id'],
                                    '--format=JobID%80,State%35,ExitCode,Start,ElapsedRaw,AllocCPUS,NodeList%80,End'])
            success, cancelled, pending = classify_original(job, mapping['indices'], accounting(raw), original.ACTIVE)
            if pending:
                queue = self.raw_execute(['squeue', '--noheader', '-j', job['job_id'], '--format=%i|%T'])
                if queue.strip():
                    self.missing_since.pop(job['key'], None)
                elif time.monotonic() - self.missing_since.setdefault(job['key'], time.monotonic()) > 300:
                    raise RuntimeError('Original array accounting is incomplete; review required')
                return False
            replacements = []
            if cancelled:
                replacement = self.replacement(job, cancelled)
                raw = self.raw_execute(['sacct', '-X', '--array', '--noheader', '--parsable2',
                                        '-j', replacement['job_id'], '--format=JobID%80,State,ExitCode'])
                if not original.accounting_state(raw, replacement['job_id'], cancelled):
                    watch = 'replacement:' + replacement['job_id']
                    queue = self.raw_execute(['squeue', '--noheader', '-j', replacement['job_id'], '--format=%i|%T'])
                    if queue.strip():
                        self.missing_since.pop(watch, None)
                    elif time.monotonic() - self.missing_since.setdefault(watch, time.monotonic()) > 300:
                        raise RuntimeError('Replacement accounting is incomplete; review required')
                    return False
                replacements.append(replacement)
            # Union semantics are explicit here. Do not write an original complete.json.
            ids = success + [r['job_id'] for r in replacements]
            runner.write_once(record_path, {'original_job': job, 'successful_original_task_ids': success,
                'cancelled_never_started_indices': cancelled, 'replacement_jobs': replacements,
                'effective_dependency_ids': ids, 'manifest_sha256': sha(self.manifest)})
            return True

    return ResumeController


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-release', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--handover', required=True, type=Path)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--poll-seconds', type=int, default=20)
    p.add_argument('--worker-time-limit', default='04:00:00')
    args = p.parse_args(argv)
    args.stop_after_bins = False
    if not args.resume:
        print(json.dumps({'mode': 'review_only', 'base_release': str(args.base_release),
            'manifest': str(args.manifest), 'handover': str(args.handover),
            'replacement_rule': 'approved CANCELLED tasks with no start and zero elapsed only',
            'concurrency': {'cpu_units': 24, 'gpu_units': 2},
            'assessment_dependency': 'afterok on the actual resumed controller Slurm ID',
            'remote_mutations': 0}, indent=2))
        return
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not job_id.isdigit():
        raise ValueError('Resume must run as the actual final Slurm controller job')
    if args.poll_seconds < 10:
        raise ValueError('Poll interval below ten seconds')
    manifest_doc = validate_amendment(args.manifest, args.base_release.resolve())
    original = import_original(args.base_release.resolve())
    if manifest_doc['base_scientific_identity_sha256'] != sha(args.output / '_retained/identity.json'):
        raise ValueError('Manifest scientific identity differs from requested output')
    ctl = amended_controller(original)(args, args.manifest, args.handover)
    with (ctl.ledger / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        doc = {'controller_job_id': job_id, 'dependency': 'afterok:' + job_id,
               'amendment_path': str(args.manifest.resolve().parent),
               'manifest_sha256': sha(args.manifest), 'handover_sha256': sha(args.handover),
               'base_identity_sha256': sha(ctl.output / '_retained/identity.json')}
        original.runner.write_once(ctl.operation / 'controller_runs' / (job_id + '.json'), doc)
        original.runner.write_once(ctl.operation / 'assessment_dependency.json', doc)
        ctl.run()
        original.runner.write_once(ctl.operation / 'controller_success.json', doc)


if __name__ == '__main__':
    main()
