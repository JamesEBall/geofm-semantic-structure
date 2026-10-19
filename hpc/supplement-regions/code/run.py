#!/usr/bin/env python3
"""Leave-one-geographic-group-out supplement. Author: James Edward Ball."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import frozen_core as core
GROUPS=['Africa','Asia','Europe','North America','South America','Oceania']
MODELS=['appearance']+core.EXPECTED_MODELS
HERE=Path(__file__).resolve().parents[1]
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def split_masks(rows,group,held=None):
 region=np.array([r['region'] for r in rows]);s=np.array([r['split'] for r in rows]);y=np.array([core.CLASSES.index(r['label']) for r in rows]);allowed=np.ones(len(y),bool) if held is None else ~np.isin(y,held)
 return (region!=group)&(s=='train')&allowed,(region!=group)&(s=='val')&allowed,(region==group)&(np.ones(len(y),bool) if held is None else np.isin(y,held))
def support(y,b,mask,classes):return {core.CLASSES[c]:{'scenes':int((mask&(y==c)).sum()),'blocks':int(len(np.unique(b[mask&(y==c)])))} for c in classes}
def gate(root,model):
 freeze=json.loads((HERE/'source-freeze.json').read_text())
 for p,h in freeze.items():
  if core.sha(HERE/p)!=h:raise RuntimeError('Supplement source changed: '+p)
 cohort=json.loads((root/'results/cohort-freeze.json').read_text())
 # Only consumed inputs are hashed; raw imagery is unused in this cached CPU analysis.
 for p in ['data/manifest.json','data/appearance.npy','data/text.npz','data/text-manifest.json','results/aggregate-report.json']:
  if core.sha(root/p)!=cohort[p]:raise RuntimeError('Global cohort changed: '+p)
 if not json.loads((root/'results/aggregate-report.json').read_text())['admitted']:raise RuntimeError('Global admission failed')
 manifest=json.loads((root/'data/manifest.json').read_text());rows=manifest['rows'];core.validate_manifest(rows)
 if manifest['classes']!=core.CLASSES:raise RuntimeError('Class mismatch')
 bgroup={}
 for r in rows:
  if r['region'] not in GROUPS:raise RuntimeError('Unexpected geographic group')
  bgroup.setdefault(r['block'],set()).add(r['region'])
 if any(len(v)>1 for v in bgroup.values()):raise RuntimeError('Block spans groups')
 if model=='appearance':path=root/'data/appearance.npy'
 else:
  folder=root/'features'/model;done=json.loads((folder/'complete.json').read_text());prov=json.loads((folder/'provenance.json').read_text());path=folder/'mean.npy'
  if done['shape'][0]!=len(rows) or not done['parity_passed'] or prov.get('diagnostic_limit') or prov['manifest_sha256']!=cohort['data/manifest.json'] or done['fingerprint']!=prov['fingerprint'] or done['files']['mean.npy']!=core.sha(path):raise RuntimeError('Invalid feature completion: '+model)
 return manifest,path,cohort

def evaluate(rows,x,text,group):
 y=np.array([core.CLASSES.index(r['label']) for r in rows]);blocks=np.array([r['block'] for r in rows]);results={};cache={}
 for fold,held in [('known',None)]+[(f'joint_{i}',p) for i,p in enumerate(core.FOLDS)]:
  fit,val,test=split_masks(rows,group,held);allowed=list(range(8)) if held is None else [c for c in range(8) if c not in held];targets=list(range(8)) if held is None else list(held)
  sup={'fit':support(y,blocks,fit,allowed),'validation':support(y,blocks,val,allowed),'test':support(y,blocks,test,targets)}
  reasons=[]
  for role in ['fit','validation']:
   for c,s in sup[role].items():
    if s['scenes']<2 or s['blocks']<2:reasons.append(f'{role}:{c}:requires2scenes2blocks')
  for c,s in sup['test'].items():
   if s['scenes']<1:reasons.append('test:'+c+':absent')
  if reasons:results[fold]={'status':'unsupported','reasons':reasons,'support':sup};continue
  # Explicit disjointness and complete geographic group exclusion.
  assert not np.any(fit&val) and not np.any(fit&test) and not np.any(val&test)
  assert set(blocks[fit]).isdisjoint(blocks[val]) and set(blocks[fit]).isdisjoint(blocks[test]) and set(blocks[val]).isdisjoint(blocks[test])
  best={r:(-np.inf,None,None) for r in core.REFS};candidates={r:[] for r in core.REFS}
  for alpha,pv,pt in core.ridge_predictions(x[fit],np.eye(8)[y[fit]],x[val],x[test]):
   for ref in core.REFS:
    score=core.block_average(core.reciprocal_rank(y[val],core.language_scores(pv,text[ref])),blocks[val]);candidates[ref].append({'alpha':alpha,'validation_query_ap':score})
    if score>best[ref][0]:best[ref]=(score,alpha,core.language_scores(pt,text[ref]))
  out={}
  for ref,(_,alpha,scores) in best.items():
   rr=core.reciprocal_rank(y[test],scores);ub=np.unique(blocks[test]);bv=np.array([rr[blocks[test]==b].mean() for b in ub]);out[ref]={'alpha':alpha,'query_ap':float(bv.mean()),'test_scenes':int(test.sum()),'test_blocks':len(ub),'validation_candidates':candidates[ref]}
   cache[f'{fold}__{ref}__blocks']=ub;cache[f'{fold}__{ref}__values']=bv
  results[fold]={'status':'complete','support':sup,'references':out}
 joint={}
 for ref in core.REFS:
  if all(results[f'joint_{i}']['status']=='complete' for i in range(4)):joint[ref]=float(np.mean([results[f'joint_{i}']['references'][ref]['query_ap'] for i in range(4)]))
  else:joint[ref]=None
 return {'folds':results,'joint_summary':joint},cache

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--global-root',type=Path,required=True);ap.add_argument('--task',type=int,required=True);a=ap.parse_args();group=GROUPS[a.task//len(MODELS)];model=MODELS[a.task%len(MODELS)]
 manifest,path,cohort=gate(a.global_root,model);dest=HERE/'results'/f'{a.task:02d}-{model}.json'
 fingerprint=hashlib.sha256((core.sha(HERE/'source-freeze.json')+cohort['data/manifest.json']+core.sha(path)).encode()).hexdigest()
 if dest.exists():
  prior=json.loads(dest.read_text())
  if prior['fingerprint']!=fingerprint:raise RuntimeError('Existing result has different provenance')
  if core.sha(dest.with_suffix('.npz'))!=prior['cache_sha256']:raise RuntimeError('Corrupt result cache')
  print('Resume verified',dest);return
 x=np.load(path,mmap_mode='r')
 if x.ndim!=2 or len(x)!=len(manifest['rows']) or not np.isfinite(x).all():raise RuntimeError('Invalid feature matrix')
 text=np.load(a.global_root/'data/text.npz');result,cache=evaluate(manifest['rows'],x,text,group)
 npz=dest.with_suffix('.npz');tmp=npz.with_suffix('.tmp.npz');np.savez_compressed(tmp,**cache);tmp.replace(npz)
 result.update(author='James Edward Ball',geographic_group=group,model=model,task=a.task,fingerprint=fingerprint,manifest_sha256=cohort['data/manifest.json'],feature_sha256=core.sha(path),cache_sha256=core.sha(npz),source_freeze_sha256=core.sha(HERE/'source-freeze.json'))
 save(dest,result);print(json.dumps({'task':a.task,'group':group,'model':model,'joint_summary':result['joint_summary']}))
if __name__=='__main__':main()
