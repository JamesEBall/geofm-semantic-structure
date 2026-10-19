"""Resolve prior-image identity using pixel hashes before scoring. James Edward Ball."""
from pathlib import Path
import os,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];d=R/'data';manifest=json.loads((d/'manifest.json').read_text());rows=manifest['rows'];x=np.load(d/'images.npy',mmap_mode='r');a=np.load(d/'appearance.npy',mmap_mode='r')
prior=json.loads((d/'prior-pixels.json').read_text())['records'];hashes={r['rgb_sha256'] for r in prior};sortedhashes={r['sorted_rgb_sha256']:r['rgb_sha256'] for r in prior};found=set();locations=[];matches=[]
def match(arr):
 h=hashlib.sha256(arr.tobytes()).hexdigest()
 if h in hashes:return h
 return sortedhashes.get(hashlib.sha256(np.sort(arr.reshape(3,-1),axis=1).tobytes()).hexdigest())
for i,r in enumerate(rows):
 h=match(np.rint(x[i,:3]*10000).astype(np.uint16))
 if h:found.add(h);locations.append([r['lon'],r['lat']]);matches.append(r['id'])
for r in manifest['exclusions']:
 with rasterio.open('/vsizip/'+str(d/'EuroSAT_MS.zip')+'/'+r['id']) as src:
  h=match(src.read([2,3,4]))
  if h:
   found.add(h);u,v=src.transform*(32,32);lon,lat=Transformer.from_crs(src.crs,'EPSG:4326',always_xy=True).transform(u,v);locations.append([lon,lat]);matches.append(r['id'])
assert found==hashes,('Unresolved prior pixel identities',len(found),len(hashes))
def sphere(ll):
 q=np.radians(ll);return np.column_stack((np.cos(q[:,1])*np.cos(q[:,0]),np.cos(q[:,1])*np.sin(q[:,0]),np.sin(q[:,1])))
tree=cKDTree(sphere(locations));radius=2*np.sin(5/6371/2);dist=tree.query(sphere([[r['lon'],r['lat']] for r in rows]))[0];keep=np.flatnonzero(dist>radius);removed=[{**rows[i],'reason':'within_5km_pixel_resolved_prior_image'} for i in np.flatnonzero(dist<=radius)];newrows=[rows[i] for i in keep]
classes=manifest['classes']
for s in ['train','val','test']:
 for c in classes:
  q=[r for r in newrows if r['split']==s and r['label']==c];assert len(q)>=30 and len(set(r['block'] for r in q))>=3,(s,c,len(q))
new=np.lib.format.open_memmap(d/'images-final.npy',mode='w+',dtype='float32',shape=(len(keep),6,64,64))
for j in range(0,len(keep),128):new[j:j+128]=x[keep[j:j+128]]
new.flush();del new,x
np.save(d/'appearance-final.npy',np.asarray(a[keep]));del a
old=d/'admission-v1';old.mkdir(exist_ok=False)
for n in ['images.npy','appearance.npy','manifest.json']:(d/n).rename(old/n)
(d/'images-final.npy').rename(d/'images.npy');(d/'appearance-final.npy').rename(d/'appearance.npy')
manifest.update(rows=newrows,exclusions=manifest['exclusions']+removed,prior_identity_audit={'unique_prior_images':len(hashes),'resolved':len(found),'matching_source_images':len(matches),'additional_scene_exclusions':len(removed),'rule':'Exact blue-green-red DN pixels or per-band sorted pixels, followed by 5km exclusion around original-source centres.'})
(d/'manifest.json').write_text(json.dumps(manifest,indent=2))
report={'author':'James Edward Ball','admitted':len(newrows),'excluded':len(manifest['exclusions']),'prior_identity_audit':manifest['prior_identity_audit'],'support':{s:{c:sum(r['split']==s and r['label']==c for r in newrows) for c in classes} for s in ['train','val','test']},'blocks':{s:len(set(r['block'] for r in newrows if r['split']==s)) for s in ['train','val','test']},'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest()}
(R/'results/final-admission.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
