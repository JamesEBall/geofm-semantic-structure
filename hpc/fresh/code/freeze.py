"""Freeze and verify the external-validation specification. James Edward Ball."""
from pathlib import Path
import hashlib,json,sys,datetime
R=Path(__file__).resolve().parents[1];out=R/'results/frozen-specification.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
if '--check' in sys.argv:
 j=json.loads(out.read_text())
 for n,h in j['sha256'].items():
  assert sha(R/n)==h,('Frozen specification changed',n)
 print('Frozen specification verified')
else:
 assert not out.exists(),'Freeze exists'
 paths=list((R/'code').glob('*.py'))+list((R/'code').glob('*.md'))+list(R.glob('*.sbatch'))
 paths += [R/'data'/n for n in ['manifest.json','prior-pixels.json','prior-locations.json','text.npz','text-manifest.json']]
 paths += [R/'results'/n for n in ['final-admission.json','environment.txt','weights-ready.json']]
 out.write_text(json.dumps({'author':'James Edward Ball','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Frozen before test model scoring','sha256':{str(p.relative_to(R)):sha(p) for p in paths}},indent=2))
 print('Frozen',len(paths),'files')
