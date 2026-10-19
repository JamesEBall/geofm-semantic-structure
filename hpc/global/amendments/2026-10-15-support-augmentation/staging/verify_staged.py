"""Validate transferred acquisition evidence without network access. James Edward Ball."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'data/blocks.json').read_text())
reports=[]
for idx in [0,20,40,60,80,100]:
 b=spec['blocks'][idx]
 r=json.loads((ROOT/'data/pilot-blocks'/b['id']/'report-4.json').read_text())
 assert r['block']==b and r['attempted']==4 and r['admitted']==len(r['rows'])
 for row in r['rows']:
  with np.load(ROOT/row['path']) as z:
   assert hashlib.sha256(z['image'].tobytes()).hexdigest()==row['pixel_sha256']
   assert hashlib.sha256(z['mask'].tobytes()).hexdigest()==row['mask_sha256']
 reports.append(r)
technical=sum(sum(e.get('kind')=='technical_failure' for e in r['exclusions']) for r in reports)
assert technical==0 and all(r['admitted']>=1 for r in reports) and sum(r['admitted'] for r in reports)>=12,'Pilot admission failed'
for b in spec['blocks']:
 assert (ROOT/'data/blocks'/b['id']/'report-200.json').is_file(),b['id']
print('Transferred pilot passed original gates; all 120 full acquisition reports present. Full cohort checks follow.')
