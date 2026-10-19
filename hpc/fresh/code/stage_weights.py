"""Stage pinned weights on login node; no inference. James Edward Ball."""
from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures
R=Path(__file__).resolve().parents[1];jobs=[]
jobs += [('https://huggingface.co/made-with-clay/Clay/resolve/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt','clay.pt','21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0'),('https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/9eb1b1102806593963daa333bcc491b1c6f8562f/Prithvi_EO_V2_300M.pt','prithvi.pt','faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7')]
p=json.loads((R/'package/data/prithvi100/model.json').read_text())
for f in p['files']:jobs.append((p['base_url']+f['name'],'prithvi100/'+f['name'],f['sha256']))
for m in json.loads((R/'package/data/text-encoders.json').read_text()):
 for f in m['files']:jobs.append((f"https://huggingface.co/{m['model']}/resolve/{m['revision']}/{f['file']}",m['name']+'/'+f['file'],f['sha256']))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def get(job):
 url,name,h=job;p=R/'weights'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and digest(p)==h:return name
 tmp=p.with_suffix(p.suffix+'.download')
 with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
  while True:
   b=r.read(8388608)
   if not b:break
   f.write(b)
 assert digest(tmp)==h,name
 tmp.replace(p);print('Verified',name,flush=True);return name
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(get,jobs))
(R/'results/weights-ready.json').write_text(json.dumps({'files':[{'path':n,'sha256':h} for _,n,h in jobs]},indent=2))
