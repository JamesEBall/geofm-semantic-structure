"""Exact integer aggregation before reflectance scaling. Author: James Edward Ball."""
import numpy as np

def grid20(raw,mask,native_metres,offset):
    a=np.asarray(raw);mask=np.asarray(mask)
    if a.ndim!=2 or mask.shape!=a.shape or mask.dtype!=np.bool_:raise ValueError('Array/mask shape or dtype')
    if not np.isfinite(a).all() or (a<0).any() or (a>65535).any() or not np.equal(a,np.floor(a)).all():raise ValueError('Unsigned integer DN required')
    if offset not in (0,-1000) or native_metres not in (10,20):raise ValueError('Unsupported route')
    a=a.astype(np.int64)
    if native_metres==20:return (a+offset)/10000.,mask.copy()
    h,w=a.shape
    if h%2 or w%2:raise ValueError('Even native dimensions required')
    total=a.reshape(h//2,2,w//2,2).sum(axis=(1,3),dtype=np.int64)
    valid=mask.reshape(h//2,2,w//2,2).all(axis=(1,3))
    return (total+4*offset)/40000.,valid
