"""Fixed metadata-only admission and geographic split. James Edward Ball."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,zipfile,math
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];d=R/'data';zpath=d/'EuroSAT_MS.zip'
assert hashlib.md5(zpath.read_bytes()).hexdigest()=='091174add3c8e680a49244acf185b9f0'
classes=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
prior=json.loads((d/'prior-locations.json').read_text())['points']
def sphere(ll):
 a=np.radians(np.asarray(ll));return np.column_stack((np.cos(a[:,1])*np.cos(a[:,0]),np.cos(a[:,1])*np.sin(a[:,0]),np.sin(a[:,1])))
old=cKDTree(sphere(prior));radius=2*np.sin(5/6371/2)
rows=[];excluded=[];seen=set()
with zipfile.ZipFile(zpath) as z:
 files=sorted(n for n in z.namelist() if n.endswith('.tif') and Path(n).parent.name in classes)
assert len(files)==27000,len(files)
for i,name in enumerate(files):
 path='/vsizip/'+str(zpath)+'/'+name
 with rasterio.open(path) as src:
  if src.crs is None or src.transform.is_identity:raise ValueError('Missing geographic metadata '+name)
  assert src.count==13 and src.shape==(64,64)
  x,y=src.transform*(32,32);lon,lat=Transformer.from_crs(src.crs,'EPSG:4326',always_xy=True).transform(x,y)
  arr=src.read([2,3,4,9,12,13]);ph=hashlib.sha256(arr.tobytes()).hexdigest()
  valid=bool(np.isfinite(arr).all() and not (arr==0).all(axis=0).any() and not (arr==65535).any())
 reason=None
 if not valid:reason='invalid_radiometry'
 elif ph in seen:reason='duplicate_pixels'
 elif old.query(sphere([[lon,lat]]))[0][0]<=radius:reason='within_5km_prior_research_location'
 seen.add(ph)
 if reason:excluded.append({'id':name,'reason':reason});continue
 block=f'{math.floor(lon)}_{math.floor(lat)}'
 v=int(hashlib.sha256(('geofm-external-v1:'+block).encode()).hexdigest()[:8],16)%100
 split='train' if v<60 else 'val' if v<80 else 'test'
 rows.append({'id':name,'label':Path(name).parent.name,'block':block,'split':split,'lon':lon,'lat':lat,'pixel_sha256':ph})
 if i%3000==0:print('inventory',i,flush=True)
# Keep test, remove validation near test, then training near retained validation/test.
for role,higher in [('val',['test']),('train',['test','val'])]:
 targets=[r for r in rows if r['split'] in higher];tree=cKDTree(sphere([[r['lon'],r['lat']] for r in targets]));kept=[]
 for r in rows:
  if r['split']==role and tree.query(sphere([[r['lon'],r['lat']]]))[0][0]<=radius:excluded.append({'id':r['id'],'reason':'cross_split_5km_buffer'})
  else:kept.append(r)
 rows=kept
for split in ['train','val','test']:
 for label in classes:
  q=[r for r in rows if r['split']==split and r['label']==label]
  assert len(q)>=30 and len(set(r['block'] for r in q))>=3,(split,label,len(q))
# Freeze IDs before any encoder outputs; metadata and pixels only.
manifest={'author':'James Edward Ball','dataset':'EuroSAT_MS 7711810','classes':classes,'rows':rows,'exclusions':excluded,'split_rule':'1-degree blocks SHA256 geofm-external-v1, 60/20/20 with 5km cross-split guard','input_bands':['B02','B03','B04','B8A','B11','B12'],'radiometry':'source DN / 10000; L1C to L2A/HLS domain shift retained','known_limits':'External dataset track, Europe, source image labels; not new sensor observations or proven absent from checkpoint pretraining.'}
(d/'manifest.json').write_text(json.dumps(manifest,indent=2))
x=np.lib.format.open_memmap(d/'images.npy',mode='w+',dtype='float32',shape=(len(rows),6,64,64));appearance=[]
for i,r in enumerate(rows):
 with rasterio.open('/vsizip/'+str(zpath)+'/'+r['id']) as src:a=src.read([2,3,4,9,12,13]).astype(np.float32)/10000
 x[i]=a
 # Fixed low-order appearance baseline, fitted scaling is handled by analysis.
 stats=np.concatenate([a.mean((1,2)),a.std((1,2)),np.quantile(a,[.1,.5,.9],axis=(1,2)).ravel(),np.abs(np.diff(a,axis=1)).mean((1,2)),np.abs(np.diff(a,axis=2)).mean((1,2))])
 appearance.append(stats)
x.flush();del x
np.save(d/'appearance.npy',np.array(appearance,np.float32))
report={'admitted':len(rows),'excluded':len(excluded),'support':{s:{c:len([r for r in rows if r['split']==s and r['label']==c]) for c in classes} for s in ['train','val','test']},'blocks':{s:len(set(r['block'] for r in rows if r['split']==s)) for s in ['train','val','test']},'manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest()}
(R/'results/admission.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
