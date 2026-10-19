#!/usr/bin/env python3
"""Portable staging and Slurm submission. Author: James Edward Ball."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parent
NAMES = ['fresh', 'global', 'mechanistic', 'internal-layers', 'supplement-controls',
         'supplement-uncertainty', 'supplement-regions']


def verify():
    for name, expected in json.loads((SOURCE / 'package-sha256.json').read_text()).items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package changed: {name}')


def initialise(root):
    verify()
    if root.exists():
        raise FileExistsError('Choose a new run directory; existing runs are preserved')
    root.mkdir(parents=True)
    for name in NAMES:
        target = root / (name + '-01' if name in ['fresh', 'global'] else name)
        shutil.copytree(SOURCE / name, target)
        for folder in ['logs', 'results', 'data']:
            (target / folder).mkdir(exist_ok=True)
    fresh = root / 'fresh-01'
    (fresh / 'package').symlink_to(SOURCE.parent, target_is_directory=True)
    for name in ['.venv', 'package', 'weights']:
        (root / 'global-01' / name).symlink_to(fresh / name, target_is_directory=True)


class Submitter:
    def __init__(self, root, stage, dry_run=False):
        self.root, self.stage, self.dry_run = root, stage, dry_run
        self.receipt = root / f'submitted-{stage}.json'
        if self.receipt.exists():
            raise FileExistsError(f'Jobs already recorded in {self.receipt}; inspect before resubmission')
        self.jobs = {}

    def add(self, name, cwd, script=None, dependencies=(), args=(), env=None, wrap=None):
        cmd = ['sbatch', '--parsable', '--chdir', str(cwd)]
        if dependencies:
            cmd += ['--dependency=afterok:' + ':'.join(dependencies)]
        if wrap:
            cmd += ['--partition=short', '--cpus-per-task=2', '--mem=16G', '--time=02:00:00',
                    '--output=logs/%j.out', '--error=logs/%j.err', '--wrap', wrap]
        else:
            cmd += [str(script), *map(str, args)]
        if self.dry_run:
            print(json.dumps({'name': name, 'command': cmd, 'environment': env or {}}))
            job = str(900000 + len(self.jobs))
        else:
            job = subprocess.check_output(cmd, text=True, env={**os.environ, **(env or {})}).strip().split(';')[0]
            if not job.isdigit():
                raise ValueError(f'Unexpected sbatch result: {job}')
        self.jobs[name] = job
        if not self.dry_run:
            # Write after each submission so partial failures retain submitted IDs.
            self.receipt.write_text(json.dumps(self.jobs, indent=2) + '\n')
        return job


def submit(root, stage, dry_run=False):
    s = Submitter(root, stage, dry_run)
    f, g = root / 'fresh-01', root / 'global-01'
    internal = root / 'internal-layers'
    env = {'FRESH_ROOT': str(f), 'INTERNAL_ROOT': str(internal)}
    if stage == 'fresh':
        prep = s.add('prepare', f, wrap='set -eu; export OPENBLAS_NUM_THREADS=1; .venv/bin/python code/prepare.py; .venv/bin/python code/finalise_admission.py; .venv/bin/pip freeze > results/environment.txt; .venv/bin/python code/freeze.py')
        gate = s.add('cuda_gate', f, f / 'smoke.sbatch', [prep])
        enc = s.add('encoders', f, f / 'extract.sbatch', [gate])
        ana = s.add('analysis', f, f / 'analyse.sbatch', [enc])
        s.add('interventions', f, root / 'mechanistic/run.sbatch', [enc])
        layers = s.add('internal_encoders', f, internal / 'encode.sbatch', [ana], env=env)
        s.add('internal_analysis', f, internal / 'analyse.sbatch', [layers], env=env)
    elif stage == 'global':
        fresh_jobs = json.loads((root / 'submitted-fresh.json').read_text())
        # Verify pilot and staged blocks before admission; freeze source before scoring.
        pre = s.add('admission', g, wrap='set -eu; .venv/bin/python staging/verify_staged.py; .venv/bin/python code/freeze.py; .venv/bin/python code/aggregate.py')
        enc = s.add('encoders', g, g / 'gpu.sbatch', [pre, fresh_jobs['cuda_gate']], env=env)
        s.add('analysis', g, g / 'analysis.sbatch', [enc])
    else:
        fj = json.loads((root / 'submitted-fresh.json').read_text())
        gj = json.loads((root / 'submitted-global.json').read_text())
        deps = [fj['analysis'], fj['interventions'], fj['internal_analysis'], gj['analysis']]
        controls = root / 'supplement-controls'
        uncertainty = root / 'supplement-uncertainty'
        for label, study in [('fresh', f), ('global', g)]:
            out = controls / 'results' / label
            prep = s.add(label + '_controls_prep', f, controls / 'prepare.sbatch', deps, [study, out])
            run = s.add(label + '_controls', f, controls / 'run.sbatch', [prep], [study, out])
            s.add(label + '_controls_report', f, controls / 'report.sbatch', [run], [out])
            e = {'SUPPLEMENT_ROOT': str(uncertainty), 'STUDY_ROOT': str(study), 'STUDY_NAME': label}
            fits = s.add(label + '_fits', f, uncertainty / 'fits.sbatch', deps, env=e)
            s.add(label + '_fit_report', f, uncertainty / 'summary.sbatch', [fits], env=e)
        e = {'SUPPLEMENT_ROOT': str(uncertainty), 'STUDY_ROOT': str(f), 'INTERNAL_RESULTS': str(internal / 'results')}
        s.add('internal_uncertainty', f, uncertainty / 'internal.sbatch', deps, env=e)
        regions = root / 'supplement-regions'
        groups = s.add('regions', regions, regions / 'run.sbatch', deps)
        s.add('regions_report', regions, regions / 'report.sbatch', [groups])
    return s.jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'verify', 'submit'])
    p.add_argument('--root', type=Path)
    p.add_argument('--stage', choices=['fresh', 'global', 'supplements'])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.action == 'verify':
        verify()
        return
    if a.root is None:
        p.error('--root is required')
    root = a.root.expanduser().resolve()
    if a.action == 'init':
        initialise(root)
    else:
        if not a.stage:
            p.error('--stage is required')
        submit(root, a.stage, a.dry_run)


if __name__ == '__main__':
    main()
