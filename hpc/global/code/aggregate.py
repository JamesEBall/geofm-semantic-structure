#!/usr/bin/env python3
"""Validate and assemble all global acquisition blocks. James Edward Ball."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def sphere(points):
 a=np.radians(points);return np.column_stack([np.cos(a[:,1])*np.cos(a[:,0]),np.cos(a[:,1])*np.sin(a[:,0]),np.sin(a[:,1])])
def main():
 for p,h in json.loads((ROOT/'code/acquisition-freeze.json').read_text()).items():
  if sha(ROOT/p)!=h:raise RuntimeError('Acquisition freeze mismatch: '+p)
 spec=json.loads((ROOT/'data/blocks.json').read_text());rows=[];excluded=[];errors=[];attempted=0
 for b in spec['blocks']:
  p=ROOT/'data/blocks'/b['id']/'report-200.json'
  if not p.exists():errors.append('missing_block_report:'+b['id']);continue
  report=json.loads(p.read_text());attempted+=report['attempted'];excluded+=report['exclusions']
  if report['block']!=b:errors.append('block_provenance:'+b['id'])
  if report['attempted']!=200:errors.append('incomplete_candidate_inventory:'+b['id'])
  if report['admitted']!=len(report['rows']):errors.append('row_count:'+b['id'])
  for row in report['rows']:
   if row['block']!=b['id'] or row['split']!=b['split'] or row.get('expert_labels') is not False:errors.append('row_metadata:'+row['id'])
   rows.append(row)
 rows.sort(key=lambda r:r['id']);classes=spec['classes'];support={}
 for split in ['train','val','test']:
  support[split]={}
  for c in classes:
   group=[r for r in rows if r['split']==split and r['label']==c];n=len(group);nb=len({r['block'] for r in group})
   support[split][c]={'scenes':n,'blocks':nb}
   minimum_n,minimum_blocks=(200,15) if split=='test' else (30,3)
   if n<minimum_n or nb<minimum_blocks:errors.append(f'insufficient_support:{split}:{c}:{n}:{nb}')
 technical=[e for e in excluded if e.get('kind')=='technical_failure']
 if technical:errors.append(f'unresolved_technical_failures:{len(technical)}')
 source_splits={}
 for r in rows:source_splits.setdefault(r['source_item'],set()).add(r['split'])
 if any(len(s)>1 for s in source_splits.values()):errors.append('source_item_crosses_splits')
 if len({r['id'] for r in rows})!=len(rows):errors.append('duplicate_ids')
 if len({r['pixel_sha256'] for r in rows})!=len(rows):errors.append('duplicate_image_pixels')
 for a,b in [('train','val'),('train','test'),('val','test')]:
  qa=[r for r in rows if r['split']==a];qb=[r for r in rows if r['split']==b]
  if qa and qb:
   d,_=cKDTree(sphere([[r['lon'],r['lat']] for r in qa])).query(sphere([[r['lon'],r['lat']] for r in qb]));minimum=2*6371*np.arcsin(np.clip(d.min()/2,0,1))
   if minimum<5:errors.append(f'cross_split_distance:{a}:{b}:{minimum}')
 report={'author':'James Edward Ball','admitted':not errors,'scene_count':len(rows),'attempted_windows':attempted,'target_attempted_windows':spec['target_attempted_windows'],'support':support,'technical_failures':len(technical),'quality_exclusions':len(excluded)-len(technical),'errors':errors,'expert_labels':False,'reference_type':'automated_worldcover_2021_v200','planning_test_floor_met':all(support['test'][c]['scenes']>=200 and support['test'][c]['blocks']>=15 for c in classes),'planning_floor_note':'Admission requires at least 200 test scenes across 15 test blocks per category; fitting and validation require 30 scenes across 3 blocks per category.'}
 save(ROOT/'results/aggregate-report.json',report)
 if errors:raise RuntimeError('Global admission failed: '+'; '.join(errors[:20]))
 tmp=ROOT/'data/images.tmp.npy';images=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(rows),6,64,64));appearance=[]
 for i,r in enumerate(rows):
  with np.load(ROOT/r['path']) as z:a=z['image'];mask=z['mask']
  if a.shape!=(6,64,64) or a.dtype!=np.float32 or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=r['pixel_sha256']:raise RuntimeError('Invalid cached image:'+r['id'])
  if hashlib.sha256(mask.tobytes()).hexdigest()!=r['mask_sha256']:raise RuntimeError('Invalid cached mask:'+r['id'])
  images[i]=a
  appearance.append(np.concatenate([a.mean((1,2)),a.std((1,2)),np.quantile(a,[.1,.5,.9],axis=(1,2)).ravel(),np.abs(np.diff(a,axis=1)).mean((1,2)),np.abs(np.diff(a,axis=2)).mean((1,2))]))
 images.flush();del images;tmp.replace(ROOT/'data/images.npy');np.save(ROOT/'data/appearance.npy',np.asarray(appearance,np.float32))
 manifest={'author':'James Edward Ball','dataset':'Global WorldCover2021 automated-reference stress track','classes':classes,'rows':rows,'exclusions':excluded,'input_bands':['B02','B03','B04','B8A','B11','B12'],'radiometry':'Sentinel2 L2A2021 baseline<4 sourceDN*0.0001; explicitscale/offsetcheck','expert_labels':False,'known_limits':'Annual automated labels with single-acquisition imagery; not independent expert validation; checkpoint pretraining overlap unresolved.','split_rule':f"{len(spec['blocks'])} one-degree blocks; centroids>=200km apart; splits={spec['split_counts']}; 100km guard from known prior centres; original 120 plus reference-only support augmentation",'blocks_sha256':sha(ROOT/'data/blocks.json'),'acquisition_freeze_sha256':sha(ROOT/'code/acquisition-freeze.json')}
 save(ROOT/'data/manifest.json',manifest)
 report.update(manifest_sha256=sha(ROOT/'data/manifest.json'),images_sha256=sha(ROOT/'data/images.npy'),appearance_sha256=sha(ROOT/'data/appearance.npy'));save(ROOT/'results/aggregate-report.json',report)
 freeze_paths=['data/images.npy','data/manifest.json','data/appearance.npy','data/text.npz','data/text-manifest.json','results/aggregate-report.json']
 save(ROOT/'results/cohort-freeze.json',{p:sha(ROOT/p) for p in freeze_paths});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
