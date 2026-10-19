#!/usr/bin/env python3
"""Execute pinned original science with an explicit, hashed operational overlay.
Author: James Edward Ball.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def validate_manifest(manifest_path,base_release):
    manifest_path=Path(manifest_path).resolve();base=Path(base_release).resolve()
    doc=json.loads(manifest_path.read_text())
    if doc['base_release']!=str(base):raise ValueError('Overlay base release differs')
    for name,key in [('run.py','base_runner_sha256'),('controller.py','base_controller_sha256')]:
        if sha(base/'retained-cohort-correction-20261018'/name)!=doc[key]:
            raise ValueError('Original runner/controller changed')
    required={'batched_f2.py','amended_worker.py','worker_amended.sbatch','controller_resume.py'}
    if not required.issubset(doc['files']):raise ValueError('Incomplete effective overlay inventory')
    for rel,expected in doc['files'].items():
        path=manifest_path.parent/rel
        if Path(rel).is_absolute() or manifest_path.parent not in path.resolve().parents:
            raise ValueError('Overlay inventory path escapes amendment')
        if sha(path)!=expected:raise ValueError('Overlay file hash changed: '+rel)
    return doc


def install(runner,helper,manifest,document,output):
    execution={'amendment_id':document['amendment_id'],'overlay_path':str(manifest.parent),
               'overlay_manifest_sha256':sha(manifest),'overlay_files':document['files'],
               'base_release':document['base_release'],'base_runner_sha256':document['base_runner_sha256'],
               'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
               'slurm_array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
               'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
               'change':'Batch F2 paths in one unchanged verifier invocation; all file hashes retained'}
    def check_inputs(ctx):
        if sha(ctx.p('_retained/identity.json'))!=document['base_scientific_identity_sha256']:
            raise ValueError('Original scientific identity changed')
        if sha(ctx.p('freeze_f2/manifest.json'))!=document['base_f2_sha256']:
            raise ValueError('Derivative F2 changed since operational amendment')
    def verify(ctx,unit):
        check_inputs(ctx)
        return helper.verify_raw_inputs(runner,ctx,unit)
    runner.verify_raw_inputs=verify
    old_write=runner.write_once
    receipt_root=(output/'_retained/receipts').resolve()
    def write(path,value):
        if receipt_root in Path(path).resolve().parents:
            if not isinstance(value,dict):raise ValueError('Receipt must be an object')
            value=dict(value,execution=execution)
        return old_write(path,value)
    runner.write_once=write
    return execution,check_inputs


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-release',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args(argv);base=args.base_release.resolve();manifest=args.manifest.resolve()
    doc=validate_manifest(manifest,base)
    raw=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    q=argparse.ArgumentParser(add_help=False)
    q.add_argument('action');q.add_argument('--output',required=True,type=Path)
    q.add_argument('--stage',required=True);q.add_argument('--phase',required=True)
    parsed,_=q.parse_known_args(raw)
    if parsed.action=='prepare':raise ValueError('Operational overlay cannot create or replace original identity')
    output=parsed.output.resolve()
    runner=load_module('run',base/'retained-cohort-correction-20261018/run.py')
    helper=load_module('batched_f2',manifest.parent/'batched_f2.py')
    execution,check=install(runner,helper,manifest,doc,output)
    from types import SimpleNamespace
    check(SimpleNamespace(p=lambda *parts:output.joinpath(*parts)))
    job=os.environ.get('SLURM_JOB_ID')
    if not job or not job.isdigit():raise ValueError('Amended worker requires an allocated Slurm job')
    record=output/'_retained/runtime_amendments'/doc['amendment_id']/(job+'-'+parsed.action+'-'+parsed.stage+'-'+parsed.phase)
    runner.write_once(record.with_suffix('.started.json'),{'execution':execution,'action':parsed.action,'stage':parsed.stage,'phase':parsed.phase,'arguments':raw})
    runner.main(raw)
    runner.write_once(record.with_suffix('.completed.json'),{'execution':execution,'status':'completed'})


if __name__=='__main__':main()
