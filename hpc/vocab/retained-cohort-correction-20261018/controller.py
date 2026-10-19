#!/usr/bin/env python3
"""Unattended derivative controller with an immutable dispatch ledger.

Author: James Edward Ball. Submission requires the explicit --submit switch.
CPU arrays run sequentially (global cap 24); one GPU array has cap 2.
An uncertain submission or failed task blocks progression and is never retried.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
          'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}


def accounting_state(stdout, job_id, indices=None):
    """Require successful accounting for every expected array task, not only its parent."""
    expected = {str(job_id)} if indices is None else {f'{job_id}_{i}' for i in indices}
    rows = {}
    for line in stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) < 3 or fields[0] not in expected:
            continue
        state = fields[1].split()[0].rstrip('+')
        rows[fields[0]] = (state, fields[2])
    for task, (state, exitcode) in rows.items():
        if state == 'COMPLETED' and exitcode == '0:0':
            continue
        if state not in ACTIVE:
            raise RuntimeError(f'Slurm task {task}: state={state}, exit={exitcode}')
    return set(rows) == expected and all(v == ('COMPLETED', '0:0') for v in rows.values())


def missing_indices(ctx, stage, units):
    missing = []
    for i, unit in enumerate(units):
        if runner.receipt_path(ctx, stage, unit['id']).exists():
            if runner.validate_receipt(ctx, stage, unit['id'])['unit'] != unit:
                raise ValueError('Existing unit differs from frozen full plan')
        else:
            missing.append(i)
    return missing


class Controller:
    def __init__(self, args, execute=None):
        self.args = args
        self.source, self.output = runner.separate(args.source_run, args.output)
        self.ledger = self.output.with_name(self.output.name + '.dispatch')
        self.ledger.mkdir(parents=True, exist_ok=True)
        self.execute = execute or self._execute
        self.background = []
        self.missing_since = {}
        runner.write_once(self.ledger / 'configuration.json', {
            'source_run': str(self.source), 'output': str(self.output),
            'release': str(runner.RELEASE), 'code': runner.code_hashes(),
            'cpu_concurrency': 24, 'gpu_concurrency': 2,
            'worker_time_limit': args.worker_time_limit,
            'author': 'James Edward Ball'})

    @staticmethod
    def _execute(command):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f'{command[0]} failed ({result.returncode}): {result.stderr.strip()}')
        return result.stdout

    def context(self, phase, stage):
        return runner.context(SimpleNamespace(source_run=self.source, output=self.output,
                                              phase=phase, stage=stage, threads=8))

    def dispatch(self, action, stage, phase, *, indices=None, after=()):
        key = f'{phase}.{stage}.{action}'
        directory = self.ledger / key
        intent = directory / 'intent.json'
        submitted = directory / 'submitted.json'
        gpu = action == 'unit' and stage == 'extract'
        inference = action == 'unit' and stage == 'analysis_inference'
        mem = 160 if inference else 48
        partition = 'gpu' if gpu else ('himem' if inference else 'short')
        token = hashlib.sha256((str(self.output) + key).encode()).hexdigest()[:12]
        cmd = ['sbatch', '--parsable', '--nice=1000', '--account=default',
               '--partition=' + partition, '--cpus-per-task=8', f'--mem={mem}G',
               '--time=' + self.args.worker_time_limit, '--job-name=retained-' + token,
               '--output=' + str(self.ledger / '%x-%A_%a.log')]
        if gpu:
            cmd += ['--gres=gpu:1']
        if indices is not None:
            if not indices or len(set(indices)) != len(indices):
                raise ValueError('Array requires distinct nonempty indices')
            limit = 2 if gpu else 24
            cmd += ['--array=' + ','.join(map(str, indices)) + f'%{limit}']
        if after:
            cmd += ['--dependency=afterok:' + ':'.join(str(j['job_id']) for j in after)]
        cmd += [str(runner.HERE / 'worker.sbatch'), str(runner.RELEASE),
                str(self.source), str(self.output), action, stage, phase]
        request = {'key': key, 'command': cmd, 'indices': indices}
        if submitted.exists():
            if runner.read(intent) != request:
                raise ValueError(f'Dispatch parameters changed: {key}')
            return runner.read(submitted)
        if intent.exists():
            raise RuntimeError(f'Uncertain submission at {intent}; reconcile the Slurm job manually')
        runner.write_once(intent, request)
        # The durable intent deliberately precedes sbatch. Failure/crash leaves a blocked intent.
        result = self.execute(cmd).strip()
        if not re.fullmatch(r'\d+(;[^\s;]+)?', result):
            raise RuntimeError(f'Unrecognised sbatch acknowledgement for {key}: {result!r}')
        job = {'key': key, 'job_id': result.split(';')[0], 'indices': indices}
        runner.write_once(submitted, job)
        print(json.dumps({'submitted': job}), flush=True)
        return job

    def complete(self, job):
        terminal = self.ledger / job['key'] / 'complete.json'
        if terminal.exists():
            if runner.read(terminal) != job:
                raise ValueError('Terminal ledger differs from submission')
            return True
        raw = self.execute(['sacct', '-X', '--array', '--noheader', '--parsable2',
                            '-j', job['job_id'], '--format=JobID%80,State,ExitCode'])
        if accounting_state(raw, job['job_id'], job['indices']):
            runner.write_once(terminal, job)
            self.missing_since.pop(job['key'], None)
            return True
        queue = self.execute(['squeue', '--noheader', '-j', job['job_id'], '--format=%i|%T'])
        if queue.strip():
            self.missing_since.pop(job['key'], None)
        else:
            since = self.missing_since.setdefault(job['key'], time.monotonic())
            if time.monotonic() - since > 300:
                raise RuntimeError(f'No complete accounting or queue record for {job["key"]}; review required')
        return False

    def wait(self, job):
        while True:
            for other in self.background:
                self.complete(other)  # detects a failed GPU refresh while CPU work proceeds
            if self.complete(job):
                return job
            time.sleep(self.args.poll_seconds)

    def command(self, action, stage, phase, after=()):
        return self.wait(self.dispatch(action, stage, phase, after=after))

    def stage(self, stage, phase, *, finalize=True, after=()):
        planned = self.command('plan', stage, phase, after)
        ctx = self.context(phase, stage)
        units = runner.load_plan(ctx, stage, phase)
        array_key = f'{phase}.{stage}.unit'
        intent = self.ledger / array_key / 'intent.json'
        # A restart reuses exactly the original dispatch, even when some receipts now exist.
        indices = runner.read(intent)['indices'] if intent.exists() else missing_indices(ctx, stage, units)
        jobs = [planned]
        if indices:
            job = self.dispatch('unit', stage, phase, indices=indices, after=[planned])
            self.wait(job)
            jobs = [job]
        if missing_indices(ctx, stage, units):
            raise RuntimeError(f'Slurm completed without all required receipts: {stage}')
        if finalize:
            result = self.command('finalize', stage, phase, jobs)
            runner.require_complete(ctx, stage)
            return result
        return jobs[-1]

    def start_gpu(self, after):
        planned = self.command('plan', 'extract', 'cpu', [after])
        ctx = self.context('cpu', 'extract')
        units = runner.load_plan(ctx, 'extract', 'cpu')
        indices = list(range(len(units)))
        array = self.dispatch('unit', 'extract', 'cpu', indices=indices, after=[planned])
        self.background = [array]
        return array

    def run(self):
        prepared = self.command('prepare', 'features_fit', 'cpu')
        self.context('cpu', 'features_fit')  # verify immutable identity before any downstream work
        fitted = self.stage('features_fit', 'cpu', after=[prepared])
        if self.args.stop_after_bins:
            print('Preparation and bins complete; controller stopped at the explicit review boundary.', flush=True)
            return
        gpu_array = self.start_gpu(fitted)
        features = self.stage('features', 'cpu', after=[fitted])
        cpu_freeze = self.command('freeze', 'features', 'cpu', [features])
        cpu_finished = []
        for stage in runner.ANALYSES:
            cpu_finished.append(self.stage(stage, 'cpu', finalize=False, after=[cpu_freeze]))
        self.wait(gpu_array)
        gpu_final = self.command('finalize', 'extract', 'cpu', [gpu_array])
        runner.require_complete(self.context('cpu', 'extract'), 'extract')
        full_freeze = self.command('freeze', 'features', 'full', [features, gpu_final])
        full_finished = []
        for stage in runner.ANALYSES:
            full_finished.append(self.stage(stage, 'full', after=[full_freeze, *cpu_finished]))
        inference = self.stage('analysis_inference', 'full', after=full_finished)
        qualification = self.stage('analysis_qualify', 'full', after=[inference])
        runner.write_once(self.ledger / 'complete.json', {
            'identity': runner.digest(self.output / '_retained/identity.json'),
            'qualification_job': qualification,
            'status': 'computation_complete_scientific_gates_require_review'})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--submit', action='store_true', help='Enable actual Slurm submission')
    p.add_argument('--stop-after-bins', action='store_true', help='Explicit preparation/fit review boundary')
    p.add_argument('--poll-seconds', type=int, default=20)
    p.add_argument('--worker-time-limit', default='04:00:00')
    args = p.parse_args(argv)
    if not args.submit:
        source, output = runner.separate(args.source_run, args.output)
        print(json.dumps({
            'mode': 'no_submission', 'source_run': str(source), 'output': str(output),
            'release': str(runner.RELEASE), 'slurm_commands_issued': 0,
            'ordered_cpu_work': ['prepare', 'features_fit', 'features', 'freeze_f3_cpu',
                                 *runner.ANALYSES, 'extract_finalize', 'freeze_f3',
                                 *['full_missing_' + s for s in runner.ANALYSES],
                                 'analysis_inference', 'analysis_qualify'],
            'gpu_parallel_after': 'features_fit', 'gpu_refresh_chunks': 55,
            'cpu_unit_concurrency': 24, 'gpu_unit_concurrency': 2,
            'unit_cpus': 8, 'normal_unit_memory_gib': 48,
            'inference_memory_gib': 160, 'nice': 1000,
            'stop_after_bins': args.stop_after_bins,
            'resubmission': 'never automatic; uncertain or failed jobs require review'}, indent=2))
        return
    if args.poll_seconds < 10:
        raise ValueError('Poll interval must be at least ten seconds')
    if not os.environ.get('MEGAEARTH_ROOT'):
        raise ValueError('Set MEGAEARTH_ROOT explicitly before submission')
    controller = Controller(args)
    lock = controller.ledger / 'controller.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller.run()


if __name__ == '__main__':
    main()
