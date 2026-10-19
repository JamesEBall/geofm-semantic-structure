"""Deterministic descriptors and fitting-split-only feature transforms."""
from __future__ import annotations
import datetime as dt


def bilinear(a, size):
    """Half-pixel bilinear interpolation with edge clamping, align_corners=False."""
    import numpy as np
    h, w = a.shape[-2:]
    yy = np.clip((np.arange(size)+.5)*h/size-.5, 0, h-1)
    xx = np.clip((np.arange(size)+.5)*w/size-.5, 0, w-1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0+1, h-1), np.minimum(x0+1, w-1)
    wy, wx = yy-y0, xx-x0
    top = a[..., y0[:, None], x0] * (1-wx) + a[..., y0[:, None], x1] * wx
    bot = a[..., y1[:, None], x0] * (1-wx) + a[..., y1[:, None], x1] * wx
    return (top*(1-wy[:, None]) + bot*wy[:, None]).astype(np.float32)


def routes(b10, b20, offset, scale):
    import numpy as np
    from ._imagery_io import reflectance
    a, b = reflectance(b10, offset, scale), reflectance(b20, offset, scale)
    r1 = np.concatenate([a[:3,64:192,64:192], bilinear(b[3:,32:96,32:96],128)])
    # Central 120 pixels at 10 m corresponds exactly to central 60 at 20 m.
    vis = a[:3,68:188,68:188].reshape(3,60,2,60,2).mean((2,4))
    r2 = bilinear(np.concatenate([vis,b[3:,34:94,34:94]]),64)
    return r1,r2


def appearance42(a):
    import numpy as np
    return np.concatenate([a.mean((1,2)),a.std((1,2)),np.quantile(a,[.1,.5,.9],axis=(1,2)).ravel(),
                           np.abs(np.diff(a,axis=1)).mean((1,2)),np.abs(np.diff(a,axis=2)).mean((1,2))])


def descriptors(a, edges):
    import numpy as np
    result = [appearance42(a)]
    result += [np.histogram(np.clip(a[i],edges[i][0],edges[i][-1]),bins=edges[i])[0]/a[i].size for i in range(6)]
    def ratio(x,y):
        den=x+y
        return np.divide(x-y,den,out=np.zeros_like(x),where=np.abs(den)>1e-6)
    blue,green,red,nir,swir1,swir2=a
    result += [np.quantile(idx,np.linspace(.1,.9,9)) for idx in
               (ratio(nir,red),ratio(green,nir),ratio(swir1,nir),ratio(nir,swir1))]
    for row in (0,64):
        for col in (0,64):
            patch=a[:,row:row+64,col:col+64]
            result += [patch.mean((1,2)),patch.std((1,2))]
    for band in a:
        gy,gx=np.gradient(band)
        mag=np.hypot(gx,gy)
        result += [np.quantile(mag,[.1,.25,.5,.75,.9])]
        hist=np.histogram(np.arctan2(gy,gx),bins=np.linspace(-np.pi,np.pi,9),weights=mag)[0]
        result += [hist/max(hist.sum(),1e-12)]
    q=np.clip(np.searchsorted(edges[3],nir,side='right')-1,0,15)
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        lhs=q[max(dy,0):q.shape[0]+min(dy,0),max(dx,0):q.shape[1]+min(dx,0)]
        rhs=q[max(-dy,0):q.shape[0]+min(-dy,0),max(-dx,0):q.shape[1]+min(-dx,0)]
        delta=(lhs-rhs).astype(float)
        result += [np.array([np.mean(delta**2),np.mean(1/(1+delta**2))])]
    return np.concatenate(result).astype(np.float32)


def metadata(rows, cfg, train=None):
    import numpy as np
    lon,lat=np.radians(rows[['lon','lat']].to_numpy(float)).T
    xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    days=np.array([dt.datetime.fromisoformat(str(s).replace('Z','+00:00')).timetuple().tm_yday for s in rows['datetime']])
    phase=2*np.pi*(days-1)/365.25
    simple=np.column_stack([xyz,np.sin(phase),np.cos(phase)])
    harmonics=np.column_stack([f(h*phase) for h in range(1,int(cfg['features']['doy_harmonics'])+1) for f in (np.sin,np.cos)])
    if train is None:
        train=rows['split'].to_numpy()=='train'
    else:
        train=np.asarray(train,dtype=bool)
        if train.shape!=(len(rows),) or np.any(train & (rows['split'].to_numpy()!='train')):
            raise ValueError('metadata fitting mask must select training rows only')
    if not train.any():
        raise ValueError('metadata feature fitting requires train rows')
    biomes=sorted(set(rows.loc[train,'biome_id'].astype(str)))
    onehot=np.column_stack([(rows['biome_id'].astype(str)==b).to_numpy() for b in biomes]+[~rows['biome_id'].astype(str).isin(biomes).to_numpy()])
    density=np.log1p(np.maximum(rows['mapping_density'].to_numpy(float),0))[:,None]
    f=cfg['features'];d=int(f['metadata_rich_rff_dim']);scales=f['metadata_rich_rff_scales_km']
    rng=np.random.default_rng(int(f['metadata_rich_rff_seed']))
    chunks=[]
    for i,scale in enumerate(scales):
        width=d//len(scales)+(i<d%len(scales))
        w=rng.normal(size=(3,width))/float(scale)
        phase0=rng.uniform(0,2*np.pi,width)
        chunks.append(np.sqrt(2/d)*np.cos((xyz*6371.)@w+phase0))
    geo=np.concatenate(chunks,axis=1)
    rich=np.column_stack([simple,harmonics,density,onehot,geo])
    return {'metadata':simple.astype('float32'),'metadata_rich':rich.astype('float32'),
            'metadata_rich_rff':geo.astype('float32')}, {'biomes_fit':biomes,'doy_period_days':365.25}


def rff_features(x, train, cfg):
    import numpy as np
    if not np.asarray(train).any():
        raise ValueError('RFF fitting requires train rows')
    mean=x[train].mean(0,dtype=np.float64);sd=x[train].std(0,dtype=np.float64)
    sd[sd<1e-8]=1
    d=int(cfg['features']['appearance_rff_dim'])
    rng=np.random.default_rng(int(cfg['features']['appearance_rff_seed']))
    w=rng.normal(size=(x.shape[1],d))/np.sqrt(x.shape[1]);bias=rng.uniform(0,2*np.pi,d)
    # Chunking avoids a full float64 N by RFF-width temporary for large cohorts.
    out=np.empty((len(x),d),dtype='float32')
    for start in range(0,len(x),1024):
        out[start:start+1024]=np.sqrt(2/d)*np.cos(((x[start:start+1024]-mean)/sd)@w+bias)
    return out,{'mean':mean.tolist(),'std':sd.tolist(),'seed':cfg['features']['appearance_rff_seed'],
                'bandwidth_rule':'standardised Euclidean Gaussian frequencies / sqrt(input_dimension)'}
