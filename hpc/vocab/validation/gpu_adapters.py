"""Allocation-only engineering checks of every architecture and declared input size."""
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from stages._extract_models import build_adapter,validate_features
from mc.core import atomic_write_json


SHARED=Path(os.environ.get('MEGAEARTH_ROOT','/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01'))/'shared'
BAND_STATS={'mean':[.08,.1,.12,.25,.2,.12],'std':[.05,.05,.06,.1,.08,.07]}   # engineering stand-in; the run uses features_fit


def check(root,route,model,pixels,gsd,ten_bands,rgb=False):
    rng=np.random.default_rng(51426)
    raw=rng.uniform(.0,1.,size=(2,3,pixels,pixels)).astype('float32') if rgb else \
        rng.uniform(.02,.3,size=(2,10 if ten_bands else 6,pixels,pixels)).astype('float32')
    t=time.monotonic()
    model_obj,forward,provenance=build_adapter(root,model,torch.device('cuda:0'),pixels,gsd,ten_bands,
                                               weights=SHARED/'weights',band_stats=BAND_STATS,rgb=rgb)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        batch=forward(raw)
        single=[forward(raw[i:i+1]) for i in range(2)]
    validate_features(*batch,2)
    errors=[]
    for k in range(2):
        ref=np.concatenate([v[k] for v in single]);np.testing.assert_allclose(batch[k],ref,rtol=1e-4,atol=1e-4)
        errors.append(float(np.max(np.abs(batch[k]-ref))))
    return {'route':route,'model':model,'pixels':pixels,'max_abs_error':errors,'shape':list(batch[0].shape),
            'seconds':time.monotonic()-t,'checkpoint':provenance['checkpoint_sha256'],'passed':True}


def main():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK','4')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    root=Path('/mnt/scratch/users/40021461/geofm-semantic-expansion/fresh-01')
    cases=[('r1',m,128,10,False) for m in ('clay','prithvi','prithvi100','random_clay_101','random_prithvi_101','random_prithvi100_101')]
    cases += [('r2','clay',64,10,False),('r2g1875','clay',64,18.75,False),('r1c64','prithvi100',64,10,False),('r3','clay',256,10,True)]
    # A4 ladder adapters, run only once weights_fetch has placed their files under shared/weights.
    ladder=[('r1','prithvi_tl',128,10,False,False),('r1','dofa',128,10,False,False),('r1','random_dofa_101',128,10,False,False),
            ('r2','dofa',64,10,False,False),('r1rgb224','dinov2',224,10,False,True),('r1rgb224','dofa',224,10,False,True),
            ('r1rgb224','remoteclip',224,10,False,True)]
    present={'prithvi_tl':(SHARED/'weights/prithvi_tl/Prithvi_EO_V2_300M_TL.pt').exists(),'dofa':(SHARED/'weights/dofa/DOFA_ViT_base_e100.pth').exists(),
             'dinov2':(SHARED/'weights/dinov2/model.safetensors').exists(),'remoteclip':(SHARED/'weights/remoteclip/RemoteCLIP-ViT-B-32.pt').exists()}
    skipped=[c for c in ladder if not present[c[1].split('_')[1] if c[1].startswith('random_') else c[1]]]
    cases += [c for c in ladder if c not in skipped]
    report={'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'tests':[],'scientific_results':False,
            'skipped_missing_weights':[c[1] for c in skipped]}
    out=Path(os.environ['MEGAEARTH_ROOT'])/'validation'/('gpu-'+os.environ['SLURM_JOB_ID']+'.json')
    for case in cases:
        result=check(root,*case);report['tests'].append(result);atomic_write_json(out,report)
        print(json.dumps(result),flush=True);gc.collect();torch.cuda.empty_cache()
    report['passed']=True;atomic_write_json(out,report)

if __name__=='__main__':main()
