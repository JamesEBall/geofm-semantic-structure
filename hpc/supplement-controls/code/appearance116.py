"""Fixed, unfitted scene descriptors. Author: James Edward Ball."""
import numpy as np
BANDS=('B02','B03','B04','B8A','B11','B12')
EDGES=np.array([-np.inf,0,.05,.1,.2,.3,.4,.6,1,np.inf])
RATIOS=(('nir_red','B8A','B04'),('nir_swir1','B8A','B11'),('nir_swir2','B8A','B12'),('green_swir1','B03','B11'))
STATS=('mean','std','q10','q50','q90')

def summary(values):
    if values.size==0:return [np.nan]*5
    return [float(values.mean()),float(values.std(ddof=0)),*np.quantile(values,[.1,.5,.9],method='linear').tolist()]

def descriptors(values,masks):
    if set(values)!=set(BANDS) or set(masks)!=set(BANDS):raise ValueError('Exact six bands required')
    shape=np.asarray(values[BANDS[0]]).shape
    if len(shape)!=2 or min(shape)<2:raise ValueError('At least 2 x 2 pixels required')
    for b in BANDS:
        if np.asarray(values[b]).shape!=shape or np.asarray(masks[b]).shape!=shape:raise ValueError('Common grid required')
        if np.asarray(masks[b]).dtype!=np.bool_:raise ValueError('Boolean masks required')
        if not np.isfinite(np.asarray(values[b])[masks[b]]).all():raise ValueError('Nonfinite valid value')
    names=[];features=[];counts=[]
    def add(n,v,c):names.extend(n);features.extend(v);counts.extend([int(c)]*len(n))
    for b in BANDS:
        a=np.asarray(values[b],dtype=np.float64);m=masks[b];v=a[m]
        add([f'{b}_{s}' for s in STATS],summary(v),v.size)
        hist=np.histogram(v,EDGES)[0]/v.size if v.size else np.full(len(EDGES)-1,np.nan)
        add([f'{b}_hist{i}' for i in range(len(hist))],hist.tolist(),v.size)
        for axis in (0,1):
            lo=[slice(None)]*2;hi=list(lo);lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
            lo=tuple(lo);hi=tuple(hi);valid=m[lo]&m[hi]
            delta=a[hi][valid]-a[lo][valid]
            add([f'{b}_adjacent_abs_axis{axis}'],[float(np.abs(delta).mean()) if delta.size else np.nan],delta.size)
    for name,p,q in RATIOS:
        a=np.asarray(values[p],dtype=np.float64);b=np.asarray(values[q],dtype=np.float64)
        valid=masks[p]&masks[q]&(np.abs(a+b)>1e-6)
        v=(a[valid]-b[valid])/(a[valid]+b[valid])
        add([f'ratio_{name}_{s}' for s in STATS],summary(v),v.size)
    return names,np.asarray(features),np.asarray(counts,dtype=np.int64)
