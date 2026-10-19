"""Development-fit stability and fixed-readout intervention uncertainty. James Edward Ball."""
import argparse, importlib.util, json, hashlib, os
from pathlib import Path
import numpy as np
MODELS=['appearance','clay','prithvi','prithvi100']
SEED=20261016

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def draws(blocks,replicate):
 unique=np.unique(blocks);rng=np.random.default_rng(SEED+replicate)
 chosen=rng.choice(unique,len(unique),replace=True)
 return np.concatenate([np.flatnonzero(blocks==b) for b in chosen])
def fit(a):
 f=module(a.root/'code/analyse.py','primary');raw=json.loads((a.root/'data/manifest.json').read_text());rows=raw if isinstance(raw,list) else raw['rows'];f.validate_manifest(rows)
 y=np.array([f.CLASSES.index(r['label']) for r in rows]);s=np.array([r['split'] for r in rows]);b=np.array([str(r['block']) for r in rows]);train=np.flatnonzero(s=='train');indices=train[draws(b[train],a.replicate)]
 provenance={'analysis':sha(a.root/'code/analyse.py'),'manifest':sha(a.root/'data/manifest.json'),'text':sha(a.root/'data/text.npz'),'runner':sha(Path(__file__))}
 target=a.output/f'fit-{a.replicate:02d}.json'
 if target.exists():
  if json.loads(target.read_text())['provenance']!=provenance:raise ValueError('Existing replicate provenance differs')
  return
 # Unsupported draws remain recorded, never replaced with favourable draws.
 if set(y[indices])!=set(range(len(f.CLASSES))):
  save(target,dict(replicate=a.replicate,status='unsupported_training_draw',provenance=provenance,indices=indices.tolist()));return
 with np.load(a.root/'data/text.npz') as z:text={r:z[r] for r in f.REFS}
 rest=np.flatnonzero(s!='train');order=np.r_[indices,rest];paths=f.discover_features(a.root,'mean',len(rows),provenance['manifest'])
 results={}
 for model in MODELS:
  cache_path=a.output/f'fit-{a.replicate:02d}-{model}.json'
  cache_key={**provenance,'feature':sha(paths[model])}
  if cache_path.exists():
   prior=json.loads(cache_path.read_text())
   if prior['provenance']!=cache_key:raise ValueError('Model checkpoint provenance differs')
   results[model]=prior['result'];continue
  x=np.load(paths[model],mmap_mode='r')
  result,_=f.evaluate_model(x[order],y[order],s[order],b[order],text)
  results[model]={'known':result['known'],'heldout':result['heldout'],'heldout_summary':result['heldout_summary'],'feature_sha256':cache_key['feature']}
  save(cache_path,dict(provenance=cache_key,result=results[model]))
 save(target,dict(replicate=a.replicate,status='complete',provenance=provenance,training_indices=indices.tolist(),models=results))

def summarise(a):
 records=[json.loads((a.output/f'fit-{i:02d}.json').read_text()) for i in range(20)]
 complete=[r for r in records if r['status']=='complete'];summary={}
 if not complete:raise ValueError('No supported training resamples')
 refs=list(complete[0]['models']['appearance']['known'])
 for model in MODELS[1:]:
  summary[model]={}
  for track in ['known','heldout_summary']:
   summary[model][track]={}
   for ref in refs:
    v=[r['models'][model][track][ref]['query_ap']-r['models']['appearance'][track][ref]['query_ap'] for r in complete]
    summary[model][track][ref]={'differences':v,'median':float(np.median(v)),'minimum':min(v),'maximum':max(v),'positive_fraction':float(np.mean(np.array(v)>0))}
 save(a.output/'fit-stability.json',dict(scope='20 training-block resamples; fixed validation and test sets. Descriptive fit stability, not a population confidence interval.',supported=len(complete),unsupported=20-len(complete),summary=summary))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(11,6),sharey=True)
 for row,track in enumerate(['known','heldout_summary']):
  for col,ref in enumerate(refs):
   ax=axes[row,col]
   for j,model in enumerate(MODELS[1:]):
    v=summary[model][track][ref]['differences'];ax.scatter(v,np.full(len(v),j),s=13,alpha=.6)
   ax.axvline(0,color='.6',lw=.8);ax.set_title(track+' / '+ref,fontsize=9);ax.set_yticks(range(3),MODELS[1:]);ax.set_xlabel('Query AP difference from appearance')
 fig.suptitle('Training-block fit stability: all 20 predeclared draws');fig.tight_layout();fig.savefig(a.output/'fit-stability.png',dpi=180)

def paired_intervals(values,blocks,repeats=2000,seed=SEED):
 """values: category x scene, NaN outside each target category; common block draws."""
 unique=np.unique(blocks);rng=np.random.default_rng(seed);counts=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=repeats)
 category=[]
 for row in values:
  means=np.array([np.nanmean(row[blocks==b]) if np.isfinite(row[blocks==b]).any() else np.nan for b in unique]);keep=np.isfinite(means);w=counts[:,keep];den=w.sum(1)
  category.append(np.divide(w@means[keep],den,out=np.full(repeats,np.nan),where=den>0))
 samples=np.mean(category,axis=0);valid=samples[np.isfinite(samples)]
 point=float(np.mean([np.mean([np.mean(row[(blocks==b)&np.isfinite(row)]) for b in unique if np.isfinite(row[blocks==b]).any()]) for row in values]))
 return dict(estimate=point,conditional_95_interval=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_resamples=len(valid),invalid_resamples=repeats-len(valid),_replicates=samples)

def intervention(a):
 f=module(a.root/'code/analyse.py','primary');with_text=np.load(a.root/'data/text.npz');text={r:with_text[r] for r in f.REFS};report=json.loads((a.internal/'report.json').read_text());results={};all_draws=[];shared_selection=None
 for model in MODELS[1:]:
  root=a.internal/model;done=json.loads((root/'complete.json').read_text())
  if not done['zero_dose_exact'] or not done['fresh_feature_replay']:raise ValueError('Internal replay gates failed')
  selected_hash=sha(root/'selection.json')
  if shared_selection is None:shared_selection=selected_hash
  elif selected_hash!=shared_selection:raise ValueError('Internal model cohorts differ')
  for name,digest in done['files'].items():
   if sha(root/name)!=digest:raise ValueError('Internal cache changed')
  rows=json.loads((root/'selection.json').read_text())['rows'];y=np.array([f.CLASSES.index(r['label']) for r in rows]);s=np.array([r['split'] for r in rows]);b=np.array([str(r['block']) for r in rows]);test=np.load(root/'test_indices.npy');x=np.load(root/'clean.npy');fitmask=s=='train'
  mean=x[fitmask].mean(0,dtype=np.float64);scale=x[fitmask].std(0,dtype=np.float64);scale[scale<1e-8]=1;t=(x[fitmask]-mean)/scale;target=np.eye(len(f.CLASSES))[y[fitmask]];ym=target.mean(0);eig,u=np.linalg.eigh(t.T@t);rhs=u.T@(t.T@(target-ym));results[model]={}
  for ref in f.REFS:
   alpha=report[model]['readout_selection'][ref]['alpha'];w=u@(rhs/(np.maximum(eig,0)[:,None]+alpha));contrasts={'random':[],'norm-only':[]}
   clean_scores=f.language_scores(((x[test]-mean)/scale)@w+ym,text[ref])
   clean_ap=f.block_average(f.reciprocal_rank(y[test],clean_scores),b[test])
   if abs(clean_ap-report[model]['clean'][ref]['query_ap'])>1e-8:raise ValueError('Clean readout replay differs')
   for label in f.CLASSES:
    available=[c for c in done['conditions'] if c['dose']==1. and c['label']==label];rr={}
    for c in available:
     features=np.load(root/(c['name']+'.npy'));scores=f.language_scores(((features-mean)/scale)@w+ym,text[ref]);rr[c['control']]=f.reciprocal_rank(y[test],scores)
    keep=y[test]==f.CLASSES.index(label)
    for comparator in contrasts:
     control=np.mean([v for k,v in rr.items() if k.startswith('random-')],axis=0) if comparator=='random' else rr['norm-only']
     contrasts[comparator].append(np.where(keep,rr['target']-control,np.nan))
   results[model][ref]={}
   for comparator,values in contrasts.items():
    result=paired_intervals(np.array(values),b[test]);samples=result.pop('_replicates');all_draws.append((model,ref,comparator,samples,result['estimate']));results[model][ref][comparator]=result
 # One predeclared 18-contrast family (3 models x 3 references x 2 controls).
 # Max absolute centred error gives simultaneous conditional intervals in AP units.
 matrix=np.array([v[3] for v in all_draws]);points=np.array([v[4] for v in all_draws]);valid=np.isfinite(matrix).all(0)
 radius=float(np.quantile(np.max(np.abs(matrix[:,valid]-points[:,None]),axis=0),.95)) if valid.any() else None
 for model,ref,comparator,_,estimate in all_draws:
  results[model][ref][comparator]['simultaneous_conditional_95_interval']=[estimate-radius,estimate+radius] if radius is not None else None
 save(a.output/'internal-uncertainty.json',dict(scope='Fixed training directions, clean readouts, validation choices and 300 test images; geographic-block resampling conditional on these. No circuit identification.',family_size=len(all_draws),valid_common_draws=int(valid.sum()),results=results,internal_report_sha256=sha(a.internal/'report.json'),runner_sha256=sha(Path(__file__))))
 np.savez(a.output/'internal-bootstrap.npz',replicates=matrix,estimates=points)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
 for ax,ref in zip(axes,f.REFS):
  for i,model in enumerate(MODELS[1:]):
   for offset,control,color in [(-.12,'random','#355b73'),(.12,'norm-only','#9d7656')]:
    item=results[model][ref][control];ci=item['simultaneous_conditional_95_interval']
    if ci is not None:ax.plot(ci,[i+offset]*2,color=color)
    ax.scatter(item['estimate'],i+offset,color=color,s=20)
  ax.axvline(0,color='.6',lw=.8);ax.set_title(ref);ax.set_yticks(range(3),MODELS[1:]);ax.set_xlabel('Target intervention minus control AP')
 fig.suptitle('Conditional simultaneous intervals: random (blue), norm-only (brown)',fontsize=10)
 fig.tight_layout();fig.savefig(a.output/'internal-uncertainty.png',dpi=180)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','summarise','intervention']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--internal',type=Path);p.add_argument('--replicate',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);globals()[a.mode](a)
