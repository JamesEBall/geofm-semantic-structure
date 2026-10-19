"""Pinned image encoders, deterministic route preprocessing and immutable freezes.

Author: James Edward Ball. Feature arrays retain the complete canonical widx order.
"""
from __future__ import annotations
import json
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file, sha256_json
from mc.registry import Stage, StageFailed
from ._extract_freeze import make_freeze, verify_freeze


def single(ctx):
    return [{'id':'all'}]


def freeze1(ctx, unit, ckpt):
    path = make_freeze(ctx,'f1',[ctx.p(n) for n in ('osm_fetch','osm_filter','osm_split','osm_areas','vocab','select','text','families')])
    return {'outputs':[path]}


def freeze2(ctx, unit, ckpt):
    import pandas as pd
    verify_freeze(ctx,'f1')
    classes = pd.read_parquet(ctx.p('admit','classes_final.parquet'))
    if not classes.admitted.any():
        raise StageFailed('No classes pass admission')
    # Shard hashes are checked by the reader and also frozen here.
    from ._imagery_io import iter_shards
    # Weights are frozen here, before embeddings (PLAN S8): a late ladder download cannot hold up F1 or imagery.
    paths = [ctx.p(n) for n in ('assemble','labels_exact','admit','pilot_check','weights_fetch')]
    paths.extend(Path(s) for s in iter_shards(ctx))
    path = make_freeze(ctx,'f2',paths)
    return {'outputs':[path]}


def freeze3(ctx, unit, ckpt):
    verify_freeze(ctx,'f2')
    path = make_freeze(ctx,'f3',[ctx.p(n) for n in ('freeze_f1','freeze_f2','extract','features','text','families','admit','assemble','labels_exact','pilot_check')])
    return {'outputs':[path]}


def route_pixels(b10, b20, offsets, scales, route):
    """Raw arrays -> declared FP32 reflectance route, with no clipping or per-image scaling."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    if b10.shape[1:] != (4,256,256) or b20.shape[1:] != (6,128,128):
        raise ValueError('Unexpected stored band grid')
    off=np.asarray(offsets,dtype=np.float32)[:,None,None,None]
    scale=np.asarray(scales,dtype=np.float32)[:,None,None,None]
    x10=torch.from_numpy((np.asarray(b10,dtype=np.float32)+off)*scale)
    x20=torch.from_numpy((np.asarray(b20,dtype=np.float32)+off)*scale)
    if route in ('r2','r2g1875'):
        hi=F.avg_pool2d(x10[:,:3,68:188,68:188],2)
        lo=x20[:,[3,4,5],34:94,34:94]
        x=F.interpolate(torch.cat((hi,lo),dim=1),size=(64,64),mode='bilinear',align_corners=False)
    elif route=='r3':
        x=torch.cat((x10,F.interpolate(x20,scale_factor=2,mode='bilinear',align_corners=False)),dim=1)
    elif route in ('r1','r1c64'):
        hi=x10[:,:3,64:192,64:192]
        lo=F.interpolate(x20[:,[3,4,5],32:96,32:96],scale_factor=2,mode='bilinear',align_corners=False)
        x=torch.cat((hi,lo),dim=1)
        if route=='r1c64':
            x=x[:,:,32:96,32:96]
    elif route=='r1rgb224':
        # Natural-image references (ladder rungs 3 and 4): the paper's fixed RGB display conversion
        # (reflectance / 0.4, gamma 2.2, 8-bit), then one bilinear resize of the R1 footprint to 224 px.
        rgb=torch.clamp(x10[:,[2,1,0],64:192,64:192]/0.4,0.,1.)**(1/2.2)
        rgb=torch.round(rgb*255.)/255.
        x=F.interpolate(rgb,size=(224,224),mode='bilinear',align_corners=False)
    else:
        raise ValueError('Unknown input route: '+route)
    out=x.numpy()
    if not np.isfinite(out).all():
        raise ValueError('Non-finite model input')
    return out


def extraction_plan(ctx):
    import pandas as pd
    verify_freeze(ctx,'f2')
    n=len(pd.read_parquet(ctx.p('assemble','windows_qa.parquet'),columns=['widx']))
    if n==0:
        raise StageFailed('No QA-passed images')
    return [{'id':f'{r["route"]}-{m}-{start:07d}','route':r['route'],'model':m,'start':start,'stop':min(start+4096,n)}
            for r in ctx.cfg['extraction']['routes'] for m in r['models'] for start in range(0,n,4096)]


def load_batch(ctx, table, route):
    import numpy as np
    if table.empty:
        raise ValueError('Cannot load an empty image batch')
    # Group by shard, retaining batch positions rather than DataFrame index labels.
    # Copy only selected rows and close each mapping before opening the next one.
    # Keeping per-row memmap slices alive exhausts file descriptors at large batches.
    groups = {}
    for position, (shard, row) in enumerate(zip(table.shard_dir, table.row)):
        groups.setdefault(str(shard), []).append((position, int(row)))
    bands = []
    for filename in ('b10.npy', 'b20.npy'):
        batch = None
        for shard, rows in groups.items():
            mapped = np.load(ctx.run / shard / filename, mmap_mode='r', allow_pickle=False)
            try:
                if batch is None:
                    batch = np.empty((len(table), *mapped.shape[1:]), dtype=mapped.dtype)
                if mapped.shape[1:] != batch.shape[1:] or mapped.dtype != batch.dtype:
                    raise ValueError('Inconsistent stored band grid or dtype')
                for position, row in rows:
                    batch[position] = mapped[row]
            finally:
                mapped._mmap.close()
        bands.append(batch)
    return route_pixels(*bands, table.dn_offset.to_numpy(), table.dn_scale.to_numpy(), route)


def extraction_run(ctx, unit, ckpt):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import pandas as pd
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from ._extract_models import build_adapter,validate_features
    if not torch.cuda.is_available():
        raise StageFailed('Image extraction requires a GPU allocation')
    torch.set_num_threads(ctx.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    route,name=unit['route'],unit['model']
    arch=name.split('_')[1] if name.startswith('random_') else name
    px={'r1':128,'r2':64,'r2g1875':64,'r1c64':64,'r3':256,'r1rgb224':224}[route]
    band_stats=(read_json(ctx.p('features_fit','bins.json')) or {}).get('band_stats')
    model,forward,prov=build_adapter(ctx.model_root,name,torch.device('cuda:0'),px,
                                   ctx.cfg['extraction']['clay_gsd'].get(route,10.),route=='r3',
                                   weights=ctx.shared/'weights',band_stats=band_stats,rgb=(route=='r1rgb224'))
    table=pd.read_parquet(ctx.p('assemble','windows_qa.parquet')).sort_values('widx').iloc[unit['start']:unit['stop']]
    if list(table.widx)!=list(range(unit['start'],unit['stop'])):
        raise StageFailed('Image rows differ from frozen widx order')
    batch=int(ctx.cfg['extraction']['batch_size'][route][arch])
    fingerprint=sha256_json({'unit':unit,'provenance':prov,'freeze':sha256_file(ctx.p('freeze_f2','manifest.json'))})
    chunks=[]
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        raw=load_batch(ctx,table.iloc[:min(4,len(table))],route)
        both=forward(raw)
        for k in range(2):
            one=np.concatenate([forward(raw[i:i+1])[k] for i in range(len(raw))])
            np.testing.assert_allclose(both[k],one,rtol=1e-4,atol=1e-4)
        validate_features(*both,len(raw))
        start=0
        while start<len(table):
            ckpt.maybe_stop()
            stop=min(start+batch,len(table))
            cache=ckpt.path(f'batch-{start:07d}-{stop:07d}.npz')
            if cache.exists():
                with np.load(cache,allow_pickle=False) as z:
                    if str(z['fingerprint'])!=fingerprint:
                        raise StageFailed('Extraction checkpoint fingerprint mismatch')
                    feats=z['mean'],z['cls']
            else:
                raw=load_batch(ctx,table.iloc[start:stop],route)
                try:
                    feats=forward(raw)
                except torch.cuda.OutOfMemoryError:
                    if batch==1:
                        raise
                    batch=max(1,batch//2);torch.cuda.empty_cache();continue
                validate_features(*feats,stop-start)
                with atomic_path(cache) as p:
                    np.savez(p,mean=feats[0],cls=feats[1],fingerprint=fingerprint)
            chunks.append(cache);start=stop
        target=ctx.stage_dir()/route/name/'chunks'/(unit['id']+'.npz')
        values={'mean':[],'cls':[]}
        for path in chunks:
            with np.load(path,allow_pickle=False) as z:
                for k in values:
                    values[k].append(z[k])
        with atomic_path(target) as p:
            np.savez(p,**{k:np.concatenate(v) for k,v in values.items()},widx=table.widx.to_numpy())
        meta=target.with_suffix('.json')
        atomic_write_json(meta,dict(prov,fingerprint=fingerprint,start=unit['start'],stop=unit['stop'],batch_size=batch,parity_passed=True))
    return {'outputs':[target,meta],'metrics':{'rows':len(table),'batch_size':batch,'route':route,'model':name}}


def required_models(cfg):
    """Route-model pairs the confirmatory analysis cannot do without (H1 architectures and their seeds)."""
    out=set()
    for arch in cfg['analysis']['architectures'].values():
        out.add(arch['pretrained']);out.update(arch['random'])
    return out


def extraction_finalize(ctx):
    import numpy as np
    import pandas as pd
    table=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    n=len(table);ids_hash=sha256_json(table.window_id.tolist())
    outputs=[];missing=[]
    required=required_models(ctx.cfg)
    for route in ctx.cfg['extraction']['routes']:
        for model in route['models']:
            out=ctx.stage_dir()/route['route']/model
            parts=sorted(out.glob('chunks/*.npz'))
            expected=[u for u in ctx.units() if u['route']==route['route'] and u['model']==model]
            if len(parts)!=len(expected):
                name=f"{route['route']}-{model}"
                if name in required:
                    raise StageFailed('Missing extraction chunks for registered architecture '+name)
                # Descriptive ladder or sensitivity route: record and continue; analysis skips it.
                missing.append({'route':name,'chunks':len(parts),'expected':len(expected)})
                ctx.log('descriptive route incomplete; skipped',route=name)
                continue
            for pooling in ('mean','cls'):
                with np.load(parts[0],allow_pickle=False) as z:
                    dim=z[pooling].shape[1]
                path=out/(pooling+'.npy');seen=np.zeros(n,dtype=bool)
                with atomic_path(path) as p:
                    arr=np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(n,dim))
                    for chunk in parts:
                        with np.load(chunk,allow_pickle=False) as z:
                            idx=z['widx'];x=z[pooling]
                            if seen[idx].any() or not np.isfinite(x).all():
                                raise StageFailed('Duplicated rows or invalid features')
                            arr[idx]=x;seen[idx]=True
                    if not seen.all():
                        raise StageFailed('Incomplete feature rows')
                    arr.flush();del arr
                meta=path.with_suffix('.meta.json')
                atomic_write_json(meta,{'rows':n,'d':dim,'window_ids_sha256':ids_hash,'sha256':sha256_file(path),
                    'route':route['route'],'model':model,'pooling':pooling,'precision':'float32','chunks':[p.name for p in parts]})
                outputs += [str(path.relative_to(ctx.run)),str(meta.relative_to(ctx.run))]
    atomic_write_json(ctx.stage_dir()/'complete.json',{'outputs':outputs,'rows':n,'missing_descriptive_routes':missing})
    return {'rows':n,'arrays':len(outputs)//2,'missing_descriptive_routes':missing}


def audit_pack(ctx, unit, ckpt):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from ._imagery_io import open_shard
    table=pd.read_parquet(ctx.p('assemble','windows_qa.parquet')).sort_values('window_id')
    rng=np.random.default_rng(ctx.cfg['seed']);indices=np.sort(rng.choice(len(table),min(150,len(table)),replace=False))
    outputs=[];rows=[]
    for idx in indices:
        ckpt.maybe_stop();row=table.iloc[int(idx)];raw=load_batch(ctx,table.iloc[[int(idx)]],'r1')[0]
        rgb=np.clip(np.moveaxis(raw[[2,1,0]],0,-1)/0.3,0,1)
        dest=ctx.stage_dir()/(str(row.window_id)+'.png')
        with atomic_path(dest) as p:
            Image.fromarray((rgb*255).astype('uint8')).save(p)
        outputs.append(dest);rows.append({'widx':int(row.widx),'window_id':str(row.window_id),'image':dest.name})
    path=ctx.stage_dir()/'index.json';atomic_write_json(path,{'sample':rows,'purpose':'Quality audit; no labels inferred','display_max_reflectance':0.3})
    return {'outputs':outputs+[path]}


STAGES=[
    Stage('freeze_f1','P1','cpu',('select','families'),single,freeze1,unit_mem_gb=8),
    Stage('freeze_f2','P2','cpu',('admit','weights_fetch'),single,freeze2,unit_mem_gb=8),
    # allow_failed_frac=1: dead units reach finalize, which enforces completeness for the registered
    # architectures only; a descriptive ladder encoder that cannot run is recorded, not fatal.
    Stage('extract','P2','gpu',('assemble','freeze_f2','features_fit','weights_fetch'),extraction_plan,extraction_run,extraction_finalize,
          unit_cpus=8,unit_mem_gb=48,unit_gpus=1,est_unit_min=30,allow_failed_frac=1.0),
    Stage('freeze_f3','P2','cpu',('extract','features'),single,freeze3,unit_mem_gb=8),
    Stage('audit_pack','P2','cpu',('admit',),single,audit_pack,unit_mem_gb=4),
]
