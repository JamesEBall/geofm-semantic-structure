"""Common full-footprint development grid. Author: James Edward Ball."""
import numpy as np
import torch
import torch.nn.functional as F
from geofm_jlens.semantic_nuisance_inputs_v1 import grid20
BANDS=('B02','B03','B04','B8A','B11','B12')

def common_grid(raw,masks):
    values=[]
    for b in BANDS:
        x,m=grid20(raw[b],masks[b],10 if b in BANDS[:3] else 20,0)
        if x.shape!=(60,60) or not m.all():raise ValueError('Full valid 1200-m footprint required')
        values.append(x)
    native=np.stack(values)
    resized=F.interpolate(torch.from_numpy(native.astype(np.float32))[None],size=(64,64),mode='bilinear',align_corners=False,antialias=False)[0].numpy()
    return native,resized

def scalar_resize(native):
    # Independent coordinate/weight expression, float64 reference.
    c,h,w=native.shape;out=np.empty((c,64,64),dtype=np.float64)
    for i in range(64):
        y=max(0,min(h-1,(i+.5)*h/64-.5));y0=int(np.floor(y));y1=min(y0+1,h-1);wy=y-y0
        for j in range(64):
            x=max(0,min(w-1,(j+.5)*w/64-.5));x0=int(np.floor(x));x1=min(x0+1,w-1);wx=x-x0
            out[:,i,j]=(1-wy)*((1-wx)*native[:,y0,x0]+wx*native[:,y0,x1])+wy*((1-wx)*native[:,y1,x0]+wx*native[:,y1,x1])
    return out
