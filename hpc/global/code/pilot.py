#!/usr/bin/env python3
"""24-window radiometry, quality and staging gate. James Edward Ball."""
import hashlib,json,time,subprocess,sys
from pathlib import Path
import acquire
ROOT=Path(__file__).resolve().parents[1]
start=time.monotonic();reports=[];failures=[]
spec=json.loads((ROOT/'data/blocks.json').read_text())
for idx in [0,20,40,60,80,100]:
 result=subprocess.run([sys.executable,str(ROOT/'code/acquire.py'),'acquire','--block',str(idx),'--limit','4','--output-subdir','pilot-blocks'])
 block=spec['blocks'][idx];p=ROOT/'data/pilot-blocks'/block['id']/'report-4.json'
 if result.returncode:failures.append({'block':block['id'],'exit_code':result.returncode})
 if p.exists():reports.append(json.loads(p.read_text()))
elapsed=time.monotonic()-start
technical=sum(sum(e.get('kind')=='technical_failure' for e in r['exclusions']) for r in reports)
admitted=sum(r['admitted'] for r in reports);attempted=sum(r['attempted'] for r in reports)
passed=not failures and len(reports)==6 and attempted==24 and technical==0 and all(r['admitted']>=1 for r in reports) and admitted>=12
result={'passed':passed,'attempted':attempted,'target_attempted':24,'admitted':admitted,'technical_failures':technical,'process_failures':failures,'elapsed_seconds':elapsed,'seconds_per_attempt':elapsed/max(attempted,1),'estimated_acquisition_hours_4workers':elapsed/max(attempted,1)*24000/3600/4,'regions':[r['block']['region'] for r in reports],'blocks_sha256':hashlib.sha256((ROOT/'data/blocks.json').read_bytes()).hexdigest(),'note':'Local staging pilot; catalogue timing extrapolation is provisional.'}
acquire.save(ROOT/'results/pilot-report.json',result);print(json.dumps(result,indent=2))
if not passed:raise RuntimeError('Global acquisition pilot failed admission gate')
