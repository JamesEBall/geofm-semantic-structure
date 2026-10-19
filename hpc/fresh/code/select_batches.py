from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];out={}
for model in ['clay','prithvi','prithvi100']:
 records=[]
 for batch in [32,128]:
  p=R/'features'/f'{model}-diagnostic-256-b{batch}'/'complete.json';j=json.loads(p.read_text());assert j['parity_passed'];records.append({'batch_size':batch,'images_per_second':j['images_per_second'],'peak_bytes':j['gpu_peak_reserved_bytes']})
 eligible=[r for r in records if r['peak_bytes']<70*1024**3];assert eligible
 out[model]={**max(eligible,key=lambda r:r['images_per_second']),'candidates':records}
(R/'results/batch-sizes.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
