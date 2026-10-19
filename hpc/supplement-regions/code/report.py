#!/usr/bin/env python3
"""Collect all geographic-group results and paired conditional ranges. James Edward Ball."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run

def contrast(model,base,mc,bc,track,ref):
 folds=['known'] if track=='known' else [f'joint_{i}' for i in range(4)]
 if any(model['folds'][f]['status']!='complete' or base['folds'][f]['status']!='complete' for f in folds):return {'status':'unsupported','gain':None}
 groups=[];diffs=[]
 for f in folds:
  key=f'{f}__{ref}';b=mc[key+'__blocks'];assert np.array_equal(b,bc[key+'__blocks']);groups.append(b);diffs.append(mc[key+'__values']-bc[key+'__values'])
 allblocks=np.unique(np.concatenate(groups));rng=np.random.default_rng(51426);draw=[];invalid=0
 for _ in range(1000):
  sampled=rng.choice(allblocks,len(allblocks),replace=True);counts={b:int((sampled==b).sum()) for b in allblocks};means=[]
  for b,d in zip(groups,diffs):
   w=np.array([counts[x] for x in b]);means.append(float(np.average(d,weights=w)) if w.sum() else None)
  if any(v is None for v in means):invalid+=1
  else:draw.append(float(np.mean(means)))
 return {'status':'complete','gain':float(np.mean([d.mean() for d in diffs])),'conditional_ci':np.quantile(draw,[.025,.975]).tolist() if len(allblocks)>=3 and draw else None,'valid_bootstraps':len(draw),'invalid_bootstraps':invalid,'test_blocks':len(allblocks)}

def main():
 results={};cache={}
 for i in range(78):
  model=run.MODELS[i%13];p=run.HERE/'results'/f'{i:02d}-{model}.json'
  d=json.loads(p.read_text());assert d['task']==i and d['model']==model and d['geographic_group']==run.GROUPS[i//13]
  assert d['source_freeze_sha256']==run.core.sha(run.HERE/'source-freeze.json') and run.core.sha(p.with_suffix('.npz'))==d['cache_sha256']
  results[i]=d;cache[i]=np.load(p.with_suffix('.npz'))
 assert len({d['manifest_sha256'] for d in results.values()})==1
 comparisons=[]
 for i,d in results.items():
  if d['model']=='appearance':continue
  j=(i//13)*13
  for track in ['known','joint']:
   for ref in run.core.REFS:comparisons.append(dict(geographic_group=d['geographic_group'],model=d['model'],track=track,reference=ref,**contrast(d,results[j],cache[i],cache[j],track,ref)))
 run.save(run.HERE/'results/summary.json',{'author':'James Edward Ball','analysis':'Secondary leave-one-geographic-group-out evaluation','source_freeze_sha256':run.core.sha(run.HERE/'source-freeze.json'),'comparisons':comparisons,'limits':'Country-based geographic groups; automated annual reference labels; conditional ranges hold fits and validation fixed. Unsupported folds remain NA; joint summary requires every fold.'})
 for track in ['known','joint']:
  fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
  for ax,ref in zip(axes,run.core.REFS):
   for mi,model in enumerate(['clay','prithvi','prithvi100']):
    for gi,g in enumerate(run.GROUPS):
     q=next(c for c in comparisons if c['track']==track and c['reference']==ref and c['model']==model and c['geographic_group']==g)
     if q['status']=='complete':
      y=gi+(mi-1)*.17;ci=q['conditional_ci'];ax.plot(q['gain'],y,'o',color=['#385b70','#81724b','#536c59'][mi],label=model if gi==0 else None)
      if ci:ax.plot(ci,[y,y],color=['#385b70','#81724b','#536c59'][mi],lw=1)
   missing=sum(c['status']=='unsupported' for c in comparisons if c['track']==track and c['reference']==ref and c['model'] in ['clay','prithvi','prithvi100'])
   ax.text(.02,.02,f'Unsupported cells: {missing}/18',transform=ax.transAxes,fontsize=7,color='0.4')
   ax.axvline(0,color='0.6',lw=.8);ax.set_title(ref.replace('_',' '));ax.set_xlabel('Query AP gain over appearance');ax.spines[['top','right']].set_visible(False)
  axes[0].set_yticks(range(6),run.GROUPS);axes[0].invert_yaxis();axes[-1].legend(frameon=False,fontsize=8);fig.suptitle('Held-out geographic groups: '+track+' transfer');fig.tight_layout();fig.savefig(run.HERE/'results'/f'geographic-groups-{track}.png',dpi=180);fig.savefig(run.HERE/'results'/f'geographic-groups-{track}.pdf');plt.close(fig)
 print('All 78 model/group tasks verified; unsupported outcomes retained.')
if __name__=='__main__':main()
