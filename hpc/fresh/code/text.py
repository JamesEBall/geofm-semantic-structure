"""Frozen external-category language references. James Edward Ball."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
R=Path(__file__).resolve().parents[1]
device=os.environ.get('GEOFM_TEXT_DEVICE','cuda')
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
phrases={
'AnnualCrop':['Annual crops','Fields used to grow crops that complete their life cycle within one year.','Agricultural land planted with annual crops.'],
'Forest':['Forest','Land covered by a dense canopy of trees.','An area of woodland with tall trees.'],
'HerbaceousVegetation':['Herbaceous vegetation','Land covered by grasses and other non-woody plants.','An area dominated by low herbaceous plants.'],
'Highway':['Highway','A major paved road used by motor vehicles.','A broad transport road crossing the landscape.'],
'Industrial':['Industrial area','Land occupied by factories, warehouses and associated paved surfaces.','Large buildings and infrastructure used for industry.'],
'Pasture':['Pasture','Grassland used for grazing livestock.','Agricultural grazing land covered mainly by grass.'],
'PermanentCrop':['Permanent crops','Agricultural land with perennial crops, including orchards and vineyards.','Fields planted with crops that remain for several years.'],
'Residential':['Residential area','Land occupied mainly by homes and associated streets.','Buildings and roads forming a residential settlement.'],
'River':['River','A flowing channel of water crossing the landscape.','An elongated natural watercourse.'],
'SeaLake':['Sea or lake','A broad body of standing or marine water.','An open area of lake or sea water.']}
classes=sorted(phrases);strings=[s for c in classes for s in phrases[c]];refs={}
for name in ['qwen','mpnet']:
 p=R/'weights'/name;tok=AutoTokenizer.from_pretrained(p,trust_remote_code=False,padding_side='left' if name=='qwen' else 'right')
 model=AutoModel.from_pretrained(p,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
 for ref in (['qwen_plain','qwen_instructed'] if name=='qwen' else ['mpnet_plain']):
  prefix='Instruct: Represent the land-cover meaning of this category for semantic similarity.\nQuery: ' if ref=='qwen_instructed' else ''
  batch=tok([prefix+s for s in strings],padding=True,return_tensors='pt').to(device)
  with torch.inference_mode():
   h=model(**batch).last_hidden_state
   v=h[:,-1] if name=='qwen' else (h*batch['attention_mask'][...,None]).sum(1)/batch['attention_mask'].sum(1)[:,None]
   v=torch.nn.functional.normalize(v,dim=1).reshape(10,3,-1).mean(1);v=torch.nn.functional.normalize(v,dim=1)
  refs[ref]=v.cpu().numpy()
 del model;torch.cuda.empty_cache()
np.savez_compressed(R/'data/text.npz',**refs)
(R/'data/text-manifest.json').write_text(json.dumps({'author':'James Edward Ball','device':device,'classes':classes,'phrases':phrases,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_manifest':json.loads((R/'package/data/text-encoders.json').read_text()),'limits':'Fixed names and two definitions per class; no test-driven wording selection.'},indent=2))
print('Saved', {k:v.shape for k,v in refs.items()})
