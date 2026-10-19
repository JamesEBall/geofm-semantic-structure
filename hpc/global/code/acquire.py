#!/usr/bin/env python3
"""Global, weak-reference Sentinel-2 acquisition. Author: James Edward Ball.

No model outputs inform scene selection. Network reads are confined to selected
COG windows. WorldCover is an automated reference, not an expert annotation.
"""
import argparse, hashlib, json, math, os, time, urllib.request, urllib.parse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.ndimage import uniform_filter

ROOT=Path(__file__).resolve().parents[1]
SEED='geofm-global-worldcover2021-v1'
CLASSES=['Water','Trees','Grass','Flooded_Vegetation','Crops','Scrub','Built_Area','Bare_Ground']
CODES=[80,10,30,90,40,20,50,60]
BANDS=['blue','green','red','nir08','swir16','swir22']
WC='https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/'
COUNTRIES='https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson'

def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def getjson(url):
 for attempt in range(4):
  try:
   with urllib.request.urlopen(url,timeout=120) as f:return json.load(f)
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
def distance(a,b):
 a,b=np.radians(a),np.radians(b);v=np.sin((b[1]-a[1])/2)**2+np.cos(a[1])*np.cos(b[1])*np.sin((b[0]-a[0])/2)**2
 return 6371*2*np.arcsin(np.minimum(1,np.sqrt(v)))
def tile(lat,lon):return f'{"N" if lat>=0 else "S"}{abs(lat):02d}{"E" if lon>=0 else "W"}{abs(lon):03d}'
def wcurl(lon,lat):return WC+'ESA_WorldCover_10m_2021_v200_'+tile(3*math.floor(lat/3),3*math.floor(lon/3))+'_Map.tif'
def priors(paths):
 points=[]
 for p in paths:
  obj=json.loads(Path(p).read_text())
  points.extend(obj.get('points',[]))
  points.extend([[r['lon'],r['lat']] for r in obj.get('rows',[])+obj.get('exclusions',[]) if 'lon' in r and 'lat' in r])
 return points

def inventory(args):
 countries=getjson(COUNTRIES);save(ROOT/'data/countries-source.json',countries)
 regions=['Africa','Asia','Europe','North America','South America','Oceania']
 old=priors(args.prior);selected=[];counts={r:0 for r in regions}
 for ri,region in enumerate(regions):
  polys=[shape(f['geometry']) for f in countries['features'] if f['properties']['CONTINENT']==region]
  land=prep(unary_union(polys))
  candidates=[]
  for lon in range(-179,179):
   for lat in range(-55,75):
    if land.contains(Point(lon+.5,lat+.5)):candidates.append((lon,lat))
  candidates.sort(key=lambda x:digest(SEED+str(x)))
  for lon,lat in candidates:
   center=[lon+.5,lat+.5]
   if any(distance(center,[b['lon']+.5,b['lat']+.5])<200 for b in selected):continue
   # A 100km guard excludes prior scenes and most surrounding parent products.
   if any(distance(center,p)<100 for p in old):continue
   n=counts[region];nv=4 if ri>=4 else 3
   split='train' if n<10 else 'val' if n<10+nv else 'test'
   selected.append(dict(id=f'{lon}_{lat}',lon=lon,lat=lat,region=region,split=split));counts[region]+=1
   if counts[region]==20:break
  if counts[region]!=20:raise RuntimeError(f'Insufficient separated blocks {region}: {counts[region]}')
 save(ROOT/'data/blocks.json',dict(author='James Edward Ball',seed=SEED,classes=CLASSES,reference_type='automated_worldcover_2021_v200',expert_labels=False,target_blocks=120,target_attempted_windows=24000,attempts_per_block=200,blocks=selected,prior_points=old,country_source=COUNTRIES,country_source_sha256=hashlib.sha256((ROOT/'data/countries-source.json').read_bytes()).hexdigest(),split_counts={s:sum(b['split']==s for b in selected) for s in ['train','val','test']}))
 print('Frozen 120-block inventory',flush=True)

def features(block):
 params=dict(collections='sentinel-2-l2a',bbox=f"{block['lon']},{block['lat']},{block['lon']+1},{block['lat']+1}",datetime='2021-01-01T00:00:00Z/2021-12-31T23:59:59Z',limit=100)
 url='https://earth-search.aws.element84.com/v1/search?'+urllib.parse.urlencode(params);items=[]
 while url:
  d=getjson(url);items.extend(d['features']);url=next((l['href'] for l in d.get('links',[]) if l['rel']=='next'),None)
 eligible=[];unsupported=[]
 for f in items:
  p=f['properties'];assets=f['assets']
  if float(p.get('s2:processing_baseline','99'))>=4 or p.get('eo:cloud_cover',100)>30:continue
  if not all(b in assets for b in BANDS+['scl']):continue
  if not all(assets[b]['href'].startswith('https://') and assets[b]['href'].endswith('.tif') for b in BANDS+['scl']):
   unsupported.append({'id':f['id'],'reason':'unsupported_non_https_cog_asset','urls':{b:assets[b]['href'] for b in BANDS+['scl']}});continue
  if not assets['nir08']['href'].endswith('/B8A.tif'):raise ValueError('Unexpected NIR identity')
  eligible.append(f)
 save(ROOT/'data/catalogue-exclusions'/(block['id']+'.json'),unsupported)
 return sorted(eligible,key=scene_priority)

def scene_priority(f):
 p=f['properties']
 return (float(p.get('eo:cloud_cover',100))+float(p.get('s2:snow_ice_percentage',100)),p['datetime'],f['id'])

def candidate_points(block,limit):
 lon,lat=block['lon'],block['lat'];url=wcurl(lon+.5,lat+.5)
 with rasterio.open(url) as src:
  # Stay inside the block and read a bounded overview for model-blind stratification.
  bounds=(lon+.15,lat+.15,lon+.85,lat+.85)
  hi=src.read(1,window=from_bounds(*bounds,src.transform),out_shape=(1536,1536),resampling=Resampling.nearest)
 arr=hi[2::4,2::4]
 # Conservative 900m neighbourhood around each 640m image footprint.
 dy=max(1,math.ceil(900/(111320*.7/1536)))
 dx=max(1,math.ceil(900/(111320*math.cos(math.radians(lat+.5))*.7/1536)))
 pools=[]
 for code in CODES:
  purity=uniform_filter((hi==code).astype('float32'),size=(dy,dx),mode='constant')[2::4,2::4]
  ys,xs=np.where((arr==code)&(purity>=.8));pool=list(zip(ys.tolist(),xs.tolist()));pool.sort(key=lambda p:digest(SEED+block['id']+str(p)));pools.append(pool)
 selected=[];cursor=[0]*8
 while len(selected)<limit:
  added=False
  for c,pool in enumerate(pools):
   while cursor[c]<len(pool):
    y,x=pool[cursor[c]];cursor[c]+=1
    p=[bounds[0]+(x+.5)/384*.7,bounds[3]-(y+.5)/384*.7]
    if any(distance(p,q['center'])<1.0 for q in selected):continue
    selected.append(dict(center=p,target_code=CODES[c]));added=True;break
   if len(selected)>=limit:break
  if not added:break
 return selected

def read_grid(url,crs,transform,nearest=False):
 with rasterio.open(url) as src:
  with WarpedVRT(src,crs=crs,transform=transform,width=64,height=64,resampling=Resampling.nearest if nearest else Resampling.bilinear) as vrt:
   return vrt.read(1,masked=True)

def acquire(args):
 spec=json.loads((ROOT/'data/blocks.json').read_text());block=spec['blocks'][args.block];out=ROOT/'data'/args.output_subdir/block['id'];out.mkdir(parents=True,exist_ok=True)
 catalog=out/'stac.json'
 if catalog.exists():items=json.loads(catalog.read_text())
 else:items=features(block);save(catalog,items)
 items.sort(key=scene_priority)
 points=candidate_points(block,200);save(out/'candidates-200.json',points);points=points[:args.limit];records=[];excluded=[]
 for i,p in enumerate(points):
  pid=f"wc2021_{block['id']}_{i:04d}";dest=out/(pid+'.npz');meta=out/(pid+'.json')
  if dest.exists() and meta.exists():
   record=json.loads(meta.read_text())
   with np.load(dest) as cached: image=cached['image']; cached_mask=cached['mask']
   mask_hash=hashlib.sha256(cached_mask.tobytes()).hexdigest()
   if record.get('mask_sha256',mask_hash)!=mask_hash:raise RuntimeError('Corrupt cached mask: '+pid)
   record['mask_sha256']=mask_hash;save(meta,record)
   if hashlib.sha256(image.tobytes()).hexdigest()!=record['pixel_sha256']:raise RuntimeError('Corrupt cached image: '+pid)
   if record['lon']!=p['center'][0] or record['lat']!=p['center'][1]:raise RuntimeError('Cached candidate mismatch: '+pid)
   records.append(record);continue
  lon,lat=p['center'];zone=int((lon+180)//6)+1;crs=f'EPSG:{32600+zone if lat>=0 else 32700+zone}'
  x,y=Transformer.from_crs(4326,crs,always_xy=True).transform(lon,lat);transform=from_origin(x-320,y+320,10,10)
  try:
   mask=read_grid(wcurl(lon,lat),crs,transform,True);valid=~np.ma.getmaskarray(mask)
   fractions=np.array([np.count_nonzero((mask.data==c)&valid)/4096 for c in CODES]);dominant=int(fractions.argmax())
   if 1-fractions.sum()>.1 or fractions[dominant]<.6:raise ValueError('mixed_or_excluded_reference')
   chosen=None
   for f in (f for f in items if shape(f['geometry']).contains(Point(lon,lat))):
    # At most 24 metadata-selected acquisitions per candidate (staging amendment).
    if chosen is None:chosen=0
    chosen+=1
    if chosen>24:break
    a=f['assets'];scl=read_grid(a['scl']['href'],crs,transform,True)
    good=np.isin(scl.data,[4,5,6])&~np.ma.getmaskarray(scl)
    if good.mean()<.98:continue
    arrays=[]
    for b in BANDS:
     rb=a[b]['raster:bands'][0]
     if rb.get('scale')!=.0001 or rb.get('offset',0)!=0:raise ValueError('unexpected_radiometry')
     v=read_grid(a[b]['href'],crs,transform)
     if np.ma.getmaskarray(v).any() or not np.isfinite(v.data).all():break
     arrays.append(v.data.astype('float32')*.0001)
    if len(arrays)!=6:continue
    image=np.stack(arrays)
    if (image<0).any() or (image>2).any():continue
    record=dict(id=pid,label=CLASSES[dominant],block=block['id'],split=block['split'],region=block['region'],lon=lon,lat=lat,fractions=fractions.tolist(),expert_labels=False,source_item=f['id'],datetime=f['properties']['datetime'],source_urls={b:a[b]['href'] for b in BANDS},reference_url=wcurl(lon,lat),scl_valid_fraction=float(good.mean()),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.data.tobytes()).hexdigest(),path=str(dest.relative_to(ROOT)))
    tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,image=image,mask=mask.data);tmp.replace(dest);save(meta,record);records.append(record);break
   else:pass
   if not dest.exists():raise ValueError('no_clear_valid_acquisition')
  except Exception as e:
   reason=str(e);expected=reason in ('mixed_or_excluded_reference','no_clear_valid_acquisition')
   excluded.append(dict(id=pid,reason=reason,kind='quality_exclusion' if expected else 'technical_failure',center=p['center']))
  if i%10==0:print(block['id'],i,len(records),len(excluded),flush=True)
 save(out/f'report-{args.limit}.json',dict(block=block,attempted=len(points),admitted=len(records),rows=records,exclusions=excluded))
 if not records:raise RuntimeError('No admitted scenes for block')

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('inventory');p.add_argument('--prior',action='append',default=[])
 p=sub.add_parser('acquire');p.add_argument('--block',type=int,required=True);p.add_argument('--limit',type=int,default=200);p.add_argument('--output-subdir',default='blocks')
 args=ap.parse_args()
 freeze=ROOT/'code/acquisition-freeze.json'
 if args.mode=='acquire':
  if not freeze.exists():raise RuntimeError('Missing acquisition freeze')
  for relative,expected in json.loads(freeze.read_text()).items():
   if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError('Acquisition freeze mismatch: '+relative)
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_MAX_RETRY='3',GDAL_HTTP_RETRY_DELAY='2',GDAL_CACHEMAX=128*1024*1024,CPL_VSIL_CURL_CACHE_SIZE=32*1024*1024):
  (inventory if args.mode=='inventory' else acquire)(args)
if __name__=='__main__':main()
