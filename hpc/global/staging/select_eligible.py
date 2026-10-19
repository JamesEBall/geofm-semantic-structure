"""Eligible-land amendment using reference maps only. Author: James Edward Ball."""
import concurrent.futures,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));import acquire as a
CACHE=ROOT/'data/eligibility';CACHE.mkdir(exist_ok=True)
def check(b):
 p=CACHE/(b['id']+'.json')
 if p.exists(): return json.loads(p.read_text())
 points=a.candidate_points(b,200)
 result={'id':b['id'],'eligible':len(points)==200,'count':len(points),'points':points}
 a.save(p,result);print(b['id'],len(points),flush=True);return result
spec=json.loads((ROOT/'data/blocks.json').read_text());original=spec['blocks']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 checks=list(pool.map(check,original))
selected=[b for b,c in zip(original,checks) if c['eligible']];replacements=[]
country=json.loads((ROOT/'data/countries-source.json').read_text())
prior=np.asarray(spec['prior_points'])
def far_prior(center):
 p=np.radians(prior);x=np.radians(center)
 h=np.sin((p[:,1]-x[1])/2)**2+np.cos(p[:,1])*np.cos(x[1])*np.sin((p[:,0]-x[0])/2)**2
 return bool(np.all(2*6371*np.arcsin(np.sqrt(np.clip(h,0,1)))>=100))
for i,(old,c) in enumerate(zip(original,checks)):
 if c['eligible']:continue
 land=a.prep(a.unary_union([a.shape(f['geometry']) for f in country['features'] if f['properties']['CONTINENT']==old['region']]))
 candidates=[(lon,lat) for lon in range(-179,179) for lat in range(-55,75) if land.contains(a.Point(lon+.5,lat+.5))]
 candidates.sort(key=lambda x:a.digest(a.SEED+str(x)))
 for lon,lat in candidates:
  center=[lon+.5,lat+.5]
  if any(a.distance(center,[b['lon']+.5,b['lat']+.5])<200 for b in selected):continue
  if not far_prior(center):continue
  b=dict(id=f'{lon}_{lat}',lon=lon,lat=lat,region=old['region'],split=old['split'])
  if not check(b)['eligible']:continue
  selected.append(b);original[i]=b;replacements.append({'excluded':old,'replacement':b,'reason':'fewer_than_200_eligible_spaced_candidates'});break
 else:raise RuntimeError('No eligible replacement for '+old['id'])
spec.update(blocks=original,eligibility_rule='Exactly200 deterministic candidate centres in allowedWorldCovercategories, >=1km apart; replace ineligible blocks within original region/split before any model scoring',eligibility_replacements=replacements)
a.save(ROOT/'data/blocks.json',spec)
a.save(ROOT/'results/eligibility-report.json',{'checked':len(checks),'replacements':replacements,'blocks':len(original),'passed':len(original)==120,'split_counts':{s:sum(b['split']==s for b in original) for s in ['train','val','test']}})
print('ELIGIBLE INVENTORY COMPLETE',len(replacements),flush=True)
