#!/usr/bin/env python3
"""Post-qualification, hash-verified transfer comparison. Author: James Edward Ball."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import amended_worker as worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-release',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--source-run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--controller-job-id',required=True)
    a=p.parse_args();doc=worker.validate_manifest(a.manifest,a.base_release)
    operation=a.output.with_name(a.output.name+'.dispatch')/'operational-amendments'/worker.sha(a.manifest)
    success=json.loads((operation/'controller_success.json').read_text())
    if success['controller_job_id']!=a.controller_job_id:raise ValueError('Assessment controller differs from successful final controller')
    if success['manifest_sha256']!=worker.sha(a.manifest):raise ValueError('Assessment amendment differs')
    runner=worker.load_module('assessment_base_runner',a.base_release/'retained-cohort-correction-20261018/run.py')
    from types import SimpleNamespace
    ctx=SimpleNamespace(run=a.output,p=lambda *parts:a.output.joinpath(*parts))
    if worker.sha(ctx.p('_retained/identity.json'))!=doc['base_scientific_identity_sha256']:
        raise ValueError('Assessment scientific identity changed')
    if worker.sha(ctx.p('freeze_f2/manifest.json'))!=doc['base_f2_sha256']:
        raise ValueError('Assessment derivative F2 changed')
    checked={}
    for stage in ('analysis_inference','analysis_qualify'):
        receipt=runner.validate_receipt(ctx,stage,'__complete__')
        if receipt.get('execution',{}).get('overlay_manifest_sha256')!=worker.sha(a.manifest):
            raise ValueError('Final stage lacks expected effective-code provenance: '+stage)
        units=runner.read(runner.plan_path(ctx,stage,'full'))
        runner.exact_receipts(ctx,stage,units)
        checked[stage]=worker.sha(runner.receipt_path(ctx,stage,'__complete__'))
    original=a.source_run/'analysis_inference/results.json'
    original_verdict=a.source_run/'analysis_qualify/verdict.json'
    for path,key in ((original,'original_results_sha256'),(original_verdict,'original_verdict_sha256')):
        if worker.sha(path)!=doc[key]:raise ValueError('Archived comparison input changed: '+str(path))
    result=a.output/'claim_assessment'
    if result.exists():raise ValueError('Assessment output exists; inspect instead of overwriting')
    subprocess.run([sys.executable,str(a.manifest.parent/'compare.py'),'--original',str(original),
        '--original-verdict',str(original_verdict),'--corrected',str(a.output/'analysis_inference/results.json'),
        '--corrected-verdict',str(a.output/'analysis_qualify/verdict.json'),'--output',str(result)],check=True)
    runner.write_once(result/'verification.json',{'author':'James Edward Ball','controller_success':success,
        'checked_final_receipts':checked,'base_identity_sha256':doc['base_scientific_identity_sha256'],
        'overlay_manifest_sha256':worker.sha(a.manifest),'outputs':{p.name:worker.sha(p) for p in result.iterdir() if p.is_file()},
        'overleaf_changed':False,'claim_decision':'requires independent scientific review; report retains all gates'})
    print(result/'comparison.md')


if __name__=='__main__':main()
