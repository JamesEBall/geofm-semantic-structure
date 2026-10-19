#!/usr/bin/env python3
"""Supplementary readout controls. Author: James Edward Ball."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,json,datetime,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from appearance116 import BANDS,descriptors
SEEDS=(101,202,303)
MODELS=['appearance','clay','prithvi','prithvi100']+[f'random_{a}_{s}' for a in ('clay','prithvi','prithvi100') for s in SEEDS]+['rich_appearance','metadata','rich_metadata']

def load_api(root):
    spec=importlib.util.spec_from_file_location('frozen',root/'code/analyse.py'); api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def metadata(rows):
    lon,lat=np.radians([[r['lon'],r['lat']] for r in rows]).T
    values=[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
    names=['earth_x','earth_y','earth_z']
    dates=[r.get('datetime') for r in rows]
    if all(dates):
        dates=[datetime.datetime.fromisoformat(d.replace('Z','+00:00')) for d in dates]
        phase=np.array([2*np.pi*(d.timetuple().tm_yday-1)/(366 if d.year%4==0 and (d.year%100 or d.year%400==0) else 365) for d in dates])
        values += [np.sin(phase),np.cos(phase)];names += ['season_sin','season_cos']
    elif any(dates):raise ValueError('Partial date availability; freeze a rule before use')
    return np.array(values).T,names

def prepare(root,out,rows,api):
    if (out/'prepared.json').exists():
        prior=json.loads((out/'prepared.json').read_text())
        if prior['manifest_sha256']!=api.sha(root/'data/manifest.json') or prior['images_sha256']!=api.sha(root/'data/images.npy') or prior['code_sha256']!=api.sha(Path(__file__)):raise ValueError('Existing preparation mismatch')
        for name,digest in prior['sha256'].items():
            if api.sha(out/name)!=digest:raise ValueError('Prepared cache checksum mismatch')
        return
    images=np.load(root/'data/images.npy',mmap_mode='r'); n=len(rows)
    if images.shape!=(n,6,64,64) or images.dtype!=np.float32:raise ValueError('Wrong image input')
    features=np.empty((n,116));counts=np.empty((n,116),dtype=np.int64)
    masks={b:np.ones((64,64),bool) for b in BANDS}
    for i,img in enumerate(images):
        names,features[i],counts[i]=descriptors(dict(zip(BANDS,img)),masks)
    met,names_met=metadata(rows)
    np.save(out/'rich_appearance.npy',features);np.save(out/'metadata.npy',met);np.save(out/'rich_metadata.npy',np.column_stack([features,met]));np.save(out/'descriptor-counts.npy',counts)
    save(out/'prepared.json',{'manifest_sha256':api.sha(root/'data/manifest.json'),'images_sha256':api.sha(root/'data/images.npy'),'descriptor_sha256':api.sha(Path(__file__).with_name('appearance116.py')),'code_sha256':api.sha(Path(__file__)),'names':names,'metadata_names':names_met,'sha256':{p.name:api.sha(p) for p in [out/'rich_appearance.npy',out/'metadata.npy',out/'rich_metadata.npy',out/'descriptor-counts.npy']}})

def scaled(train,val,test):
    median=np.nanmedian(train,axis=0);median[~np.isfinite(median)]=0.
    arrays=[np.where(np.isfinite(a),a,median) for a in (train,val,test)]
    mean=arrays[0].mean(0);std=arrays[0].std(0);std[std<1e-8]=1.
    return [(a-mean)/std for a in arrays]

def shuffle_within_blocks(y,blocks,seed):
    shuffled=y.copy();rng=np.random.default_rng(seed)
    for b in np.unique(blocks):
        ix=np.flatnonzero(blocks==b);shuffled[ix]=rng.permutation(y[ix])
    return shuffled

def neighbours(train,query,labels,nclasses,kmax=20):
    # Bounded query chunks; ties use stable training manifest order.
    values=[]
    for start in range(0,len(query),128):
        d=cdist(query[start:start+128],train,'sqeuclidean')
        ix=np.argsort(d,axis=1,kind='stable')[:,:min(kmax,len(train))]
        values.append(np.eye(nclasses)[labels[ix]])
    return np.concatenate(values)

def evaluate_fold(x,y,splits,blocks,text,api,held):
    nclasses=len(api.CLASSES);allowed=np.ones(len(y),bool) if held is None else ~np.isin(y,held)
    train=(splits=='train')&allowed;val=(splits=='val')&allowed;test=(splits=='test')&(True if held is None else np.isin(y,held))
    a,v,t=scaled(*[np.asarray(x[m],float) for m in (train,val,test)])
    fit_labels=[y[train]]+[shuffle_within_blocks(y[train],blocks[train],seed) for seed in SEEDS]
    category_maps=[]
    for seed in SEEDS:
        classes=np.unique(y[train]); perm=np.random.default_rng(seed).permutation(classes); lookup=dict(zip(classes,perm))
        category_maps.append({str(k):int(v) for k,v in lookup.items()})
        fit_labels.append(np.array([lookup[c] for c in y[train]]))
    targets=np.concatenate([np.eye(nclasses)[labels] for labels in fit_labels],axis=1)
    candidates={};best={}
    # Frozen ridge implementation reuses one decomposition across true and null targets.
    for alpha,pv,pt in api.ridge_predictions(a,targets,v,t):
        for route in range(len(fit_labels)):
            key='ridge' if route==0 else (f'block_shuffle_{SEEDS[route-1]}' if route<=3 else f'category_permutation_{SEEDS[route-4]}')
            for ref in api.REFS:
                scores=api.language_scores(pv[:,route*nclasses:(route+1)*nclasses],text[ref]);value=api.block_average(api.reciprocal_rank(y[val],scores),blocks[val]);index=key+'/'+ref
                candidates.setdefault(index,[]).append({'alpha':alpha,'validation_query_ap':value})
                if index not in best or value>best[index][0]:best[index]=(value,alpha,api.language_scores(pt[:,route*nclasses:(route+1)*nclasses],text[ref]))
    nv=neighbours(a,v,y[train],nclasses);nt=neighbours(a,t,y[train],nclasses)
    for k in (1,5,20):
        for ref in api.REFS:
            scores=api.language_scores(nv[:,:k].mean(1),text[ref]);value=api.block_average(api.reciprocal_rank(y[val],scores),blocks[val]);index='nearest/'+ref
            candidates.setdefault(index,[]).append({'k':k,'validation_query_ap':value})
            if index not in best or value>best[index][0]:best[index]=(value,k,api.language_scores(nt[:,:k].mean(1),text[ref]))
    result={};cache={}
    for index,(value,selected,scores) in best.items():
        rr=api.reciprocal_rank(y[test],scores)
        result[index]={'selected':selected,'validation_query_ap':value,'test_query_ap':api.block_average(rr,blocks[test]),'candidates':candidates[index]}
        cache[index.replace('/','__')]=rr
    result['support']={'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum()),'test_blocks':len(np.unique(blocks[test])),'held':held,'category_permutations':category_maps,'shuffle_changed_fraction':{str(s):float(np.mean(fit_labels[i+1]!=fit_labels[0])) for i,s in enumerate(SEEDS)}}
    cache.update(test_rows=np.flatnonzero(test),test_blocks=blocks[test],test_labels=y[test])
    return result,cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=MODELS);args=p.parse_args()
    root=args.root;out=args.out;out.mkdir(parents=True,exist_ok=True);api=load_api(root)
    checks=[sys.executable,str(root/'code/freeze.py'),'--check']
    if (root/'results/extraction-freeze.json').exists():checks += ['--cohort']
    subprocess.run(checks,check=True)
    raw=json.loads((root/'data/manifest.json').read_text());rows=raw if isinstance(raw,list) else raw['rows'];api.validate_manifest(rows);api.validate_geography(rows)
    if args.prepare:prepare(root,out,rows,api);return
    prepared=json.loads((out/'prepared.json').read_text())
    if prepared['manifest_sha256']!=api.sha(root/'data/manifest.json') or prepared['code_sha256']!=api.sha(Path(__file__)):raise ValueError('Preparation freeze mismatch')
    for name,digest in prepared['sha256'].items():
        if api.sha(out/name)!=digest:raise ValueError('Derived feature checksum mismatch')
    paths=api.discover_features(root,'mean',len(rows),prepared['manifest_sha256']);paths.update({n:out/(n+'.npy') for n in MODELS[-3:]})
    model=args.model
    if not model:raise ValueError('Supply --model')
    x=np.load(paths[model],mmap_mode='r'); y=np.array([api.CLASSES.index(r['label']) for r in rows]);splits=np.array([r['split'] for r in rows]);blocks=np.array([str(r['block']) for r in rows])
    with np.load(root/'data/text.npz') as z:text={r:z[r] for r in api.REFS}
    fingerprint={'manifest':prepared['manifest_sha256'],'features':api.sha(paths[model]),'text':api.sha(root/'data/text.npz'),'code':api.sha(Path(__file__)),'frozen_analysis':api.sha(root/'code/analyse.py'),'protocol':api.sha(Path(__file__).with_name('protocol.md'))}
    dest=out/model;dest.mkdir(exist_ok=True)
    for i,held in enumerate([None]+api.FOLDS):
        path=dest/f'fold-{i}.json';scorepath=dest/f'fold-{i}-scores.npz'
        if path.exists():
            prior=json.loads(path.read_text())
            if prior['fingerprint']!=fingerprint or prior['scores_sha256']!=api.sha(scorepath):raise ValueError('Resume mismatch')
            continue
        print(model,i,flush=True);result,cache=evaluate_fold(x,y,splits,blocks,text,api,held)
        np.savez_compressed(scorepath,**cache)
        save(path,{'author':'James Edward Ball','fingerprint':fingerprint,'result':result,'scores_sha256':api.sha(scorepath)})
    save(dest/'complete.json',{'fingerprint':fingerprint,'folds':len(api.FOLDS)+1})
if __name__=='__main__':main()
