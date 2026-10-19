from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parents[1];prior=json.loads((R/'data/prior-pixels.json').read_text())['records'];exact={r['rgb_sha256'] for r in prior};hist={r['sorted_rgb_sha256'] for r in prior}
rows=json.loads((R/'data/manifest.json').read_text())['rows'];x=np.load(R/'data/images.npy',mmap_mode='r');matches=[]
for i,r in enumerate(rows):
 a=np.rint(x[i,:3]*10000).astype(np.uint16)
 e=hashlib.sha256(a.tobytes()).hexdigest() in exact
 h=hashlib.sha256(np.sort(a.reshape(3,-1),axis=1).tobytes()).hexdigest() in hist
 if e or h:matches.append({**r,'exact':e,'histogram':h})
report={'author':'James Edward Ball','matches':matches,'matched':len(matches),'prior_unique':len(exact)}
(R/'results/prior-pixel-audit.json').write_text(json.dumps(report,indent=2));print('Prior pixel matches among retained:',len(matches))
