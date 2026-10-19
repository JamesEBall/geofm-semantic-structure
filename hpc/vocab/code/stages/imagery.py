"""MegaEarth STAC, raw-DN acquisition, QA and descriptor stages.

Author: James Edward Ball. Scientific thresholds come from config; all mutations
are confined to the current stage. No model scores enter scene selection.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file
from mc.registry import Stage, StageFailed


def selected(ctx):
    import pandas as pd
    frame=pd.read_parquet(ctx.p('select','windows.parquet')).sort_values('window_id').reset_index(drop=True)
    if frame.window_id.duplicated().any():
        raise StageFailed('duplicate selected window IDs')
    return frame


def interval_for(key, cfg):
    seasons = cfg['imagery']['seasons']
    return seasons[key][0] + 'T00:00:00Z/' + seasons[key][1] + 'T23:59:59Z'


def cell_seasons(cell, cfg):
    """Season keys whose latitude band intersects the cell (a cell straddling the tropic has two)."""
    from ._imagery_catalogue import season
    from ._osm_geometry import cell_bounds
    south, north = cell_bounds(cell, cfg['lattice']['deg_cell'])[1::2]
    edge = cfg['imagery']['seasons']['tropics_abs_lat']
    probes = [south + 1e-9, north - 1e-9] + [x for x in (edge - 1e-6, edge + 1e-6, -edge - 1e-6, -edge + 1e-6) if south < x < north]
    return sorted({season(lat, cfg)[0] for lat in probes})


def catalogue_path(ctx, cell, season_key):
    return ctx.shared / 'stac' / f'{cell}_{season_key}.json'


def fetch_catalogue(ctx, cell, season_key, stop=lambda: None):
    """One frozen scene catalogue per cell and season, searched over the whole 1-degree cell (PLAN S9),
    cached under shared/ so the pre-fetch stage and the search stage share it."""
    from ._imagery_catalogue import search, SOURCES
    from ._catalogue_http import CatalogueSession
    from ._osm_geometry import cell_bounds
    path = catalogue_path(ctx, cell, season_key)
    cached = read_json(path)
    if cached:
        return cached, path
    bbox = list(cell_bounds(cell, ctx.cfg['lattice']['deg_cell']))
    interval = interval_for(season_key, ctx.cfg)
    primary = ctx.cfg['imagery']['primary_source']
    sources, failed = {}, False
    import requests
    # Disable transport-level retries so every actual request, including a retry,
    # passes through the shared per-provider limiter.
    with ctx.requests_session(retries=0) as raw_session:
        session = CatalogueSession(ctx, raw_session, stop)
        for source in (primary, ctx.cfg['imagery']['fallback_source']):
            source_path = ctx.shared / 'stac_sources' / source / path.name
            identity = {'source': source, 'collection': SOURCES[source][1], 'bbox': bbox,
                        'datetime': interval, 'cloud_max': ctx.cfg['imagery']['stac_cloud_max']}
            partial = read_json(source_path)
            if partial is not None:
                if partial.get('request') != identity:
                    raise StageFailed('Cached source catalogue differs from registered query')
                sources[source] = partial['items']
                continue
            try:
                sources[source] = search(session, source, bbox, interval, ctx.cfg, stop)
                atomic_write_json(source_path, {'request': identity, 'items': sources[source]})
            except requests.RequestException as exc:
                status = exc.response.status_code if exc.response is not None else None
                if source != primary:
                    raise StageFailed(f'fallback STAC access failed: source={source}, status={status}, kind={type(exc).__name__}; catalogue remains incomplete') from None
                failed = True; sources[source] = []
    doc = {'cell': cell, 'season': season_key, 'sources': sources, 'primary_search_failed': failed, 'bbox': bbox, 'datetime': interval}
    atomic_write_json(path, doc)
    return doc, path


def prefetch_plan(ctx):
    from ._osm_geometry import connect
    size = int(ctx.cfg.get('imagery', {}).get('stac_cells_per_unit', 16))
    with connect(ctx.p('osm_areas', 'areas.sqlite'), True) as db:
        cells = [c for c, in db.execute('SELECT cell FROM cells ORDER BY cell')]
    pairs = [(c, k) for c in cells for k in cell_seasons(c, ctx.cfg)]
    return [{'id': f'batch-{i // size:05d}', 'pairs': pairs[i:i + size]} for i in range(0, len(pairs), size)]


def prefetch_run(ctx, unit, ckpt):
    outputs = []
    for cell, key in unit['pairs']:
        ckpt.maybe_stop()
        outputs.append(fetch_catalogue(ctx, cell, key, ckpt.maybe_stop)[1])
    return {'outputs': outputs, 'metrics': {'catalogues': len(outputs)}}


def prefetch_final(ctx):
    return {'catalogues': sum(len(u['pairs']) for u in ctx.units()), 'source': 'Planetary Computer; Earth Search technical fallback'}


def stac_plan(ctx):
    from ._imagery_catalogue import season
    frame=selected(ctx)
    frame['season']=[season(float(x),ctx.cfg)[0] for x in frame.lat]
    groups=sorted({(str(c),str(k)) for c,k in zip(frame.deg_cell,frame.season)})
    size=int(ctx.cfg.get('imagery',{}).get('stac_cells_per_unit',16))
    out=[]
    for i in range(0,len(groups),size):
        chunk=groups[i:i+size]
        ident=hashlib.sha256('|'.join(c+':'+k for c,k in chunk).encode()).hexdigest()[:20]
        out.append({'id':ident,'groups':[list(g) for g in chunk]})
    return out


def stac_run(ctx,unit,ckpt):
    from ._imagery_catalogue import season,candidates
    frame=selected(ctx)
    frame['season']=[season(float(x),ctx.cfg)[0] for x in frame.lat]
    outputs=[];windows=0;failed_any=False
    for cell,key in unit['groups']:
        ckpt.maybe_stop()
        rows=frame[(frame.deg_cell.astype(str)==cell)&(frame.season==key)].to_dict('records')
        catalogue,path=fetch_catalogue(ctx,cell,key,ckpt.maybe_stop)
        outputs.append(path);failed=bool(catalogue['primary_search_failed']);failed_any|=failed
        for row in rows:
            ckpt.maybe_stop()
            dest=ctx.stage_dir()/'windows'/(row['window_id']+'.json')
            atomic_write_json(dest,{'window_id':row['window_id'],'primary_search_failed':failed,
                'sources':{source:candidates(items,row,source,ctx.cfg) for source,items in catalogue['sources'].items()}})
            outputs.append(dest);windows+=1
    return {'outputs':outputs,'metrics':{'windows':windows,'groups':len(unit['groups']),'primary_search_failed':failed_any}}


def stac_final(ctx):
    frame=selected(ctx)
    missing=[v for v in frame.window_id if not ctx.p('stac_search','windows',v+'.json').exists()]
    if missing:
        raise StageFailed(f'{len(missing)} windows lack frozen scene candidates')
    return {'windows':len(frame),'source':'Planetary Computer; Earth Search technical fallback'}


def acquisition_plan(ctx):
    from ._imagery_io import write_parquet
    frame=selected(ctx)
    pilot=ctx.stage=='acquire_pilot'
    frame=frame[frame.pilot.astype(bool)==pilot].sort_values(['priority','window_id'])
    if pilot and (frame.empty or (frame.split!='val').any()):
        raise StageFailed('pilot must contain validation windows only')
    units=[]
    width=int(ctx.cfg['imagery']['chunk_windows'])
    for start in range(0,len(frame),width):
        ident=f'shard-{start//width:05d}'
        path=ctx.stage_dir()/'plans'/(ident+'.parquet')
        write_parquet(path,frame.iloc[start:start+width])
        units.append({'id':ident,'plan':str(path.relative_to(ctx.run))})
    return units


def acquisition_run(ctx,unit,ckpt):
    import concurrent.futures as cf
    import pandas as pd
    import numpy as np
    from ._imagery_acquire import Signer,cached_window,acquire_window,read_scene,match_scene,TechnicalFailure,QualityReject
    from ._imagery_io import write_shard,verify_shard,reflectance
    rows=pd.read_parquet(ctx.run/unit['plan']).to_dict('records')
    dest=ctx.stage_dir()/'shards'/unit['id']
    if (dest/'manifest.json').exists():
        manifest=verify_shard(dest)
        return {'outputs':sorted(p for p in dest.iterdir() if p.is_file()),'metrics':manifest}
    from ._catalogue_http import CatalogueSession
    pilot=ctx.stage=='acquire_pilot'
    # Cross-source parity is prescribed by TEST-PLAN S7-T06. IDs are chosen
    # before reading pixels, from the validation-only pilot list.
    parity_ids=set(selected(ctx).query('pilot == True').sort_values('window_id').window_id.head(20)) if pilot else set()
    start=time.monotonic()
    # One token signer per unit, shared by all threads and budgeted like the catalogue requests.
    signer=Signer(CatalogueSession(ctx,ctx.requests_session(retries=0),ckpt.maybe_stop),cache_path=ctx.shared/'pc_sas_token.json')
    primary=ctx.cfg['imagery']['primary_source'];fallback=ctx.cfg['imagery']['fallback_source']
    def parity(result,row,catalogue):
        """Same-product pixel agreement against the other source; the outcome and any error are recorded."""
        source=result['meta']['source'];other_source=fallback if source==primary else primary
        item=next((x for x in catalogue['sources'].get(source,[]) if x['id']==result['meta']['item_id']),None)
        other=match_scene(item,catalogue['sources'].get(other_source,[])) if item else None
        result['meta']['source_pair_status']='unavailable';result['meta'].pop('source_pair_error',None)
        if other is None:
            result['meta']['source_pair_error']='no same-product scene in '+other_source;return result
        try:
            second=read_scene(other,row,other_source,ctx.cfg,signer,ctx.proxy,ckpt.maybe_stop)
        except (TechnicalFailure,QualityReject) as exc:
            result['meta']['source_pair_error']=type(exc).__name__+': '+str(exc)[:200];return result
        pairs=[]
        for band in ('b10','b20'):
            x=reflectance(result[band],result['meta']['dn_offset'],result['meta']['dn_scale'])
            y=reflectance(second[band],second['meta']['dn_offset'],second['meta']['dn_scale'])
            valid=(result[band]>0)&(second[band]>0)
            pairs.append(float((np.abs(x[valid]-y[valid])<=1.0001e-4).mean()) if valid.any() else 0.)
        result['meta']['source_pair_status']='pass' if min(pairs)>=.99 else 'fail'
        result['meta']['source_pair_fraction']=min(pairs);result['meta']['source_pair_item']=other['id']
        return result
    def process(row):
        catalogue=read_json(ctx.p('stac_search','windows',row['window_id']+'.json'))
        if catalogue is None:
            raise StageFailed('missing frozen STAC candidates')
        wants_parity=row['window_id'] in parity_ids
        def fetch():
            result=acquire_window(row,catalogue,ctx.cfg,signer,ctx.proxy,ckpt.maybe_stop)
            if 'reject' not in result and wants_parity:
                parity(result,row,catalogue)
            return result
        def repair(result):
            # A cached window whose parity never resolved is re-checked; pixels are untouched.
            if wants_parity and result['meta'].get('source_pair_status') in (None,'unavailable'):
                return parity(result,row,catalogue),True
            return result,False
        return cached_window(ctx.stage_dir()/'cache'/unit['id'],row['window_id'],fetch,repair)
    results=[]
    # Bounded unit concurrency. Separate raster handles and HTTP sessions per thread.
    # COG reads are latency-bound through the tunnel, so I/O threads exceed the CPU reservation on purpose.
    with cf.ThreadPoolExecutor(max_workers=max(1,min(int(ctx.cfg['imagery']['threads_per_task']),3*ctx.threads))) as executor:
        for result in executor.map(process,rows):
            ckpt.maybe_stop();results.append(result)
    accepted=sorted([r for r in results if 'reject' not in r],key=lambda r:r['meta']['window_id'])
    rejected=sorted([r['reject'] for r in results if 'reject' in r],key=lambda r:r['window_id'])
    outputs=write_shard(dest,accepted,rejected)
    report=ctx.stage_dir()/'timings'/(unit['id']+'.json')
    atomic_write_json(report,{'attempted':len(rows),'accepted':len(accepted),'wall_seconds':time.monotonic()-start,
                             'threads':ctx.threads})
    outputs.append(report)
    return {'outputs':outputs,'metrics':{'attempted':len(rows),'accepted':len(accepted),'rejected':len(rejected)}}


def acquisition_final(ctx):
    import pandas as pd
    from ._imagery_io import verify_shard
    records=[]
    for unit in ctx.units():
        path=ctx.stage_dir()/'shards'/unit['id']
        verify_shard(path)
        records.extend(pd.read_parquet(path/'meta.parquet').window_id.tolist())
        records.extend(pd.read_parquet(path/'rejects.parquet').window_id.tolist())
    frame=selected(ctx)
    expected=set(frame[frame.pilot.astype(bool)==(ctx.stage=='acquire_pilot')].window_id)
    if len(records)!=len(set(records)) or set(records)!=expected:
        raise StageFailed('acquisition accounting differs from frozen selected windows')
    return {'attempted':len(records),'shards':len(ctx.units())}


def one(ctx):
    return [{'id':'all'}]


def water_check_passes(gate, correlation, diagnostics):
    """Registered form by default; A5 form only when `pilot_gate.water_check` selects it."""
    rule = gate.get('water_check', 'all_windows_tau')
    floor = gate['water_coverage_scl_spearman_min']
    if rule == 'all_windows_tau':
        return correlation is not None and correlation >= floor
    if rule == 'labelled_multilabel_and_separation':
        within = diagnostics.get('spearman_within_labelled_water')
        a, b = diagnostics.get('median_scl_water_labelled_ge_tau'), diagnostics.get('median_scl_water_unlabelled')
        return (within is not None and within >= floor and a is not None and b is not None
                and a - b >= float(gate.get('water_separation_min', 0.2)))
    raise StageFailed('unknown pilot_gate.water_check rule: ' + str(rule))


def source_parity_passes(gate, parity):
    rule = gate.get('source_parity_rule', 'all_pass_of_20')
    if rule == 'all_pass_of_20':
        return len(parity) >= 20 and all(v == 'pass' for v in parity)
    if rule == 'resolved_pass_min_15':
        resolved = [v for v in parity if v != 'unavailable']
        return len(parity) >= 20 and len(resolved) >= 15 and all(v == 'pass' for v in resolved)
    raise StageFailed('unknown pilot_gate.source_parity_rule: ' + str(rule))


def pilot_check_run(ctx,unit,ckpt):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from ._imagery_io import open_shard,analysis_crop,reflectance
    rows=selected(ctx);pilot=rows[rows.pilot.astype(bool)]
    if pilot.empty or (pilot.split!='val').any():
        raise StageFailed('pilot includes non-validation rows or is empty')
    classes=pd.read_parquet(ctx.p('vocab','classes.parquet'))
    water=set(classes.loc[(classes.primary_key=='natural')&(classes.primary_value=='water'),'class_idx'].astype(int))
    import collections
    records=[];rejects=[];water_pixels=[];cov=[];cov_multi_list=[];sclwater=[];neg=total=0;nodata=0;min_refl=float('inf')
    byid=pilot.set_index('window_id')
    for path in sorted(ctx.p('acquire_pilot','shards').glob('shard-*')):
        ckpt.maybe_stop();data=open_shard(ctx,path)
        records.extend(data['meta'].to_dict('records'));rejects.extend(data['rejects'].to_dict('records'))
        for i,meta in enumerate(data['meta'].to_dict('records')):
            a,b,s=analysis_crop(data['b10'][i],data['b20'][i],data['scl'][i])
            aa,bb=reflectance(a,meta['dn_offset'],meta['dn_scale']),reflectance(b,meta['dn_offset'],meta['dn_scale'])
            neg+=int((aa<0).sum()+(bb<0).sum());total+=aa.size+bb.size
            nodata+=int((a==0).sum()+(b==0).sum());min_refl=min(min_refl,float(aa.min()),float(bb.min()))
            row=byid.loc[meta['window_id']]
            coverage=sum(float(c) for k,c in zip(row['labels_idx'],row['labels_cov']) if int(k) in water)
            cov.append(coverage);sclwater.append(float((s==6).mean()))
            cov_multi_list.append(sum(float(c) for k,c in zip(row['multi_idx'],row['multi_cov']) if int(k) in water))
            if coverage>=ctx.cfg['labels']['tau']:
                nir=aa[3].reshape(64,2,64,2).mean((1,3))
                water_pixels.extend(nir[s==6].tolist())
    gate=ctx.cfg['pilot_gate'];n=len(pilot);tech=sum(x['reason'].startswith('technical_') for x in rejects)
    median=float(np.median(water_pixels)) if water_pixels else None
    correlation=float(spearmanr(cov,sclwater).statistic) if len(set(cov))>1 and len(set(sclwater))>1 else None
    timings=[read_json(p) for p in sorted(ctx.p('acquire_pilot','timings').glob('*.json'))]
    seconds=sum(t['wall_seconds'] for t in timings)
    projected_rate=len(records)/seconds*3600*ctx.cfg['imagery']['max_concurrent_tasks'] if seconds>0 else 0.
    parity=[r.get('source_pair_status') for r in records if r.get('source_pair_status')]
    parity_errors=collections.Counter(str(r.get('source_pair_error')) for r in records if r.get('source_pair_error'))
    # Diagnostics (A4 review): the registered correlation uses coverage >= tau, which labels only a few
    # percent of a class-stratified pilot as water. These descriptive variants are reported, not gating.
    cov_multi=np.array(cov_multi_list);cov_arr=np.array(cov);scl_arr=np.array(sclwater);labelled=cov_multi>0
    diagnostics={'windows_labelled_water_ge_tau':int((cov_arr>0).sum()),'windows_labelled_water_ge_multilabel_min':int(labelled.sum()),
                 'spearman_multilabel_coverage_vs_scl':float(spearmanr(cov_multi,scl_arr).statistic) if labelled.any() and len(set(scl_arr))>1 else None,
                 'spearman_within_labelled_water':float(spearmanr(cov_multi[labelled],scl_arr[labelled]).statistic) if labelled.sum()>3 else None,
                 'median_scl_water_labelled_ge_tau':float(np.median(scl_arr[cov_arr>0])) if (cov_arr>0).any() else None,
                 'median_scl_water_unlabelled':float(np.median(scl_arr[~labelled])) if (~labelled).any() else None,
                 'mean_scl_water_labelled_ge_tau':float(scl_arr[cov_arr>0].mean()) if (cov_arr>0).any() else None,
                 'mean_scl_water_unlabelled':float(scl_arr[~labelled].mean()) if (~labelled).any() else None}
    checks={
      'validation_only':bool((pilot.split=='val').all()),'accept_rate':len(records)/n>=gate['min_accept_rate'],
      'negative_fraction':total>0 and neg/total<=gate['max_negative_reflectance_fraction'],
      'no_analysis_nodata':nodata==0,'reflectance_minimum':min_refl>=-.100001,
      'water_nir':median is not None and gate['water_nir_median_min']<=median<=gate['water_nir_median_max'],
      'water_scl_correlation':water_check_passes(gate,correlation,diagnostics),
      'alignment':bool(records) and all(r['alignment_error_m']<=gate['max_alignment_error_m'] for r in records),
      'throughput':projected_rate>=gate['min_windows_per_hour'],'technical_fraction':tech/n<=.01,
      'source_parity':source_parity_passes(gate,parity),
    }
    report={'passed':all(checks.values()),'checks':checks,'attempted':n,'accepted':len(records),
            'technical_failures':tech,'negative_fraction':neg/total if total else None,'water_nir_median':median,
            'water_coverage_scl_spearman':correlation,'projected_windows_per_hour':projected_rate,
            'source_pair_checked':len(parity),'source_pair_statuses':dict(collections.Counter(parity)),'source_pair_errors':dict(parity_errors),
            'water_diagnostics_not_gating':diagnostics,'threshold_source':'config pilot_gate; technical and source parity TEST-PLAN S7',
            'note':'Radiometric and geometric pilot; no model scores inspected.'}
    path=ctx.stage_dir()/'report.json';atomic_write_json(path,report)
    if not report['passed']:
        raise StageFailed('pilot quality gate failed: '+', '.join(k for k,v in checks.items() if not v))
    return {'outputs':[path],'metrics':report}


def assemble_run(ctx,unit,ckpt):
    import pandas as pd
    from ._imagery_io import iter_shards,open_shard,write_parquet
    frame=selected(ctx);records={};rejections={};pixels={}
    for path in iter_shards(ctx):
        ckpt.maybe_stop();data=open_shard(ctx,path)
        for idx,r in enumerate(data['meta'].to_dict('records')):
            wid=r['window_id']
            if wid in records:
                if records[wid]['pixel_sha256']!=r['pixel_sha256']:
                    raise StageFailed('conflicting duplicate window')
                continue
            if r['pixel_sha256'] in pixels:
                raise StageFailed('identical pixels under distinct window IDs')
            pixels[r['pixel_sha256']]=wid
            records[wid]={**r,'shard_dir':str(path.relative_to(ctx.run)),'row':idx}
        for r in data['rejects'].to_dict('records'):
            rejections[r['window_id']]=r
    expected=set(frame.window_id)
    if set(records)&set(rejections) or set(records)|set(rejections)!=expected:
        raise StageFailed('assemble accounting mismatch')
    tech=sum(r['reason'].startswith('technical_') for r in rejections.values())
    if tech/max(1,len(frame))>.01:
        raise StageFailed('unresolved technical failures exceed TEST-PLAN one-percent limit')
    if not records:
        raise StageFailed('no accepted imagery')
    result=pd.DataFrame([records[k] for k in sorted(records)])
    result=result.merge(frame,on='window_id',how='left',validate='one_to_one',suffixes=('','_selected')).sort_values('window_id').reset_index(drop=True)
    result.insert(0,'widx',range(len(result)))
    paths=[ctx.stage_dir()/'windows_qa.parquet',ctx.stage_dir()/'rejects.parquet',ctx.stage_dir()/'report.json']
    write_parquet(paths[0],result);write_parquet(paths[1],pd.DataFrame(list(rejections.values()),columns=['window_id','reason','tries']))
    atomic_write_json(paths[2],{'attempted':len(frame),'accepted':len(result),'technical_failures':tech,
                              'quality_rejections':len(rejections)-tech,'window_ids_sha256':hashlib.sha256('\n'.join(result.window_id).encode()).hexdigest()})
    return {'outputs':paths,'metrics':{'accepted':len(result),'rejected':len(rejections)}}


def retained_training_mask(ctx, rows):
    from megaearth_stats.cohort import retained_mask
    return retained_mask(ctx.run, rows) & (rows['split'].to_numpy() == 'train')


def features_fit_plan(ctx):
    """One unit per shard that holds training windows; samples are merged in finalize."""
    import pandas as pd
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    shards=sorted(rows[retained_training_mask(ctx,rows)].shard_dir.unique())
    if not shards:
        raise StageFailed('no fitting images for histogram bins')
    return [{'id':f'part-{i:05d}','shard_dir':str(sh)} for i,sh in enumerate(shards)]


def features_fit_run(ctx,unit,ckpt):
    import numpy as np
    import pandas as pd
    from ._imagery_io import open_shard
    from ._imagery_features import routes
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    train=rows[retained_training_mask(ctx,rows)&(rows.shard_dir==unit['shard_dir'])].sort_values('widx')
    data=open_shard(ctx,unit['shard_dir']);samples=[]
    for row in train.to_dict('records'):
        ckpt.maybe_stop();i=row['row']
        a,_=routes(data['b10'][i],data['b20'][i],row['dn_offset'],row['dn_scale'])
        samples.append(a[:,::16,::16].reshape(6,-1))
    path=ctx.stage_dir()/'samples'/(unit['id']+'.npz')
    with atomic_path(path) as tmp:
        with open(tmp,'wb') as stream:
            np.savez(stream,samples=np.concatenate(samples,axis=1).astype('float32'),widx=train.widx.to_numpy())
    return {'outputs':[path],'metrics':{'train_windows':len(train)}}


def features_fit_final(ctx):
    """Same statistics as the former single unit: quantile edges and band mean/std over every training
    pixel sample, concatenated in unit order (np.quantile is order-independent)."""
    import numpy as np
    import pandas as pd
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'));train=rows[retained_training_mask(ctx,rows)]
    parts=[];seen=[]
    for unit in ctx.units():
        with np.load(ctx.stage_dir()/'samples'/(unit['id']+'.npz'),allow_pickle=False) as z:
            parts.append(z['samples']);seen.extend(z['widx'].tolist())
    if sorted(seen)!=sorted(train.widx.tolist()):
        raise StageFailed('histogram-fit samples do not cover the training windows exactly')
    sampled=np.concatenate(parts,axis=1).astype('float64')
    limits=np.quantile(sampled,[.005,.995],axis=1)
    edges=[np.linspace(limits[0,i],max(limits[1,i],limits[0,i]+1e-5),17).tolist() for i in range(6)]
    std=sampled.std(axis=1)
    if not np.all(np.isfinite(std)) or np.any(std<=0):
        raise StageFailed('degenerate training band statistics')
    path=ctx.stage_dir()/'bins.json'
    atomic_write_json(path,{'edges':edges,'fit_split':'train','fit_windows':len(train),'pixel_stride':16,
                          'band_stats':{'bands':['B02','B03','B04','B8A','B11','B12'],'domain':'R1 reflectance',
                                        'mean':sampled.mean(axis=1).tolist(),'std':std.tolist()},
                          'training_ids_sha256':hashlib.sha256('\n'.join(train.window_id).encode()).hexdigest()})
    return {'train_windows':len(train),'samples':int(sampled.shape[1])}


def features_plan(ctx):
    import pandas as pd
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    return [{'id':f'part-{i:05d}','shard_dir':s} for i,s in enumerate(sorted(rows.shard_dir.unique()))]


def features_run(ctx,unit,ckpt):
    import numpy as np
    import pandas as pd
    from ._imagery_io import open_shard
    from ._imagery_features import routes,appearance42,descriptors
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    rows=rows[rows.shard_dir==unit['shard_dir']].sort_values('widx')
    data=open_shard(ctx,unit['shard_dir']);edges=np.array(read_json(ctx.p('features_fit','bins.json'))['edges'])
    values=[];small=[]
    for row in rows.to_dict('records'):
        ckpt.maybe_stop();i=row['row']
        r1,r2=routes(data['b10'][i],data['b20'][i],row['dn_offset'],row['dn_scale'])
        values.append(descriptors(r1,edges));small.append(appearance42(r2))
    path=ctx.stage_dir()/'parts'/(unit['id']+'.npz')
    with atomic_path(path) as tmp:
        with open(tmp,'wb') as stream:
            np.savez(stream,widx=rows.widx.to_numpy(),appearance_rich=np.array(values,dtype='float32'),appearance42=np.array(small,dtype='float32'))
    return {'outputs':[path],'metrics':{'windows':len(rows)}}


def features_final(ctx):
    import numpy as np
    import pandas as pd
    from ._imagery_features import metadata,rff_features
    from ._imagery_io import save_array
    rows=pd.read_parquet(ctx.p('assemble','windows_qa.parquet')).sort_values('widx')
    values={};seen=set()
    for unit in ctx.units():
        with np.load(ctx.stage_dir()/'parts'/(unit['id']+'.npz'),allow_pickle=False) as part:
            idx=part['widx']
            if seen & set(idx.tolist()):
                raise StageFailed('duplicate feature widx')
            seen.update(idx.tolist())
            for name in ('appearance42','appearance_rich'):
                values.setdefault(name,np.empty((len(rows),part[name].shape[1]),dtype='float32'))[idx]=part[name]
    if seen!=set(range(len(rows))):
        raise StageFailed('feature rows incomplete')
    train=retained_training_mask(ctx,rows)
    met,met_fit=metadata(rows,ctx.cfg,train=train);values.update(met)
    values['appearance_rff'],rff_fit=rff_features(values['appearance_rich'],train,ctx.cfg)
    row_hash=hashlib.sha256('\n'.join(rows.window_id).encode()).hexdigest()
    for name,array in values.items():
        if not np.isfinite(array).all():
            raise StageFailed('non-finite descriptor')
        path=ctx.stage_dir()/(name+'.npy');save_array(path,array)
        atomic_write_json(ctx.stage_dir()/(name+'.meta.json'),{'rows':len(rows),'d':array.shape[1],
            'window_ids_sha256':row_hash,'sha256':sha256_file(path),'author':'James Edward Ball',
            'description':name,'fit_split':'train'})
    atomic_write_json(ctx.stage_dir()/'transforms.json',{'metadata':met_fit,'appearance_rff':rff_fit,
        'histogram_bins':read_json(ctx.p('features_fit','bins.json'))})
    return {'rows':len(rows),'dimensions':{k:v.shape[1] for k,v in values.items()}}


STAGES=[
 Stage('stac_prefetch','P1','net',('osm_areas',),prefetch_plan,prefetch_run,prefetch_final,unit_cpus=1,unit_mem_gb=4,est_unit_min=5),
 # stac_search fetches any catalogue the pre-fetch has not cached, so it does not wait for stac_prefetch.
 Stage('stac_search','P2','net',('freeze_f1',),stac_plan,stac_run,stac_final,unit_cpus=1,unit_mem_gb=4,gate='APPROVE_P2'),
 Stage('acquire_pilot','P2','net',('stac_search',),acquisition_plan,acquisition_run,acquisition_final,unit_cpus=8,unit_mem_gb=12,est_unit_min=20),
 Stage('pilot_check','P2','cpu',('acquire_pilot',),one,pilot_check_run,unit_cpus=2,unit_mem_gb=8),
 Stage('acquire','P2','net',('pilot_check',),acquisition_plan,acquisition_run,acquisition_final,unit_cpus=8,unit_mem_gb=12,est_unit_min=20),
 Stage('assemble','P2','cpu',('acquire',),one,assemble_run,unit_cpus=2,unit_mem_gb=12),
 Stage('features_fit','P2','cpu',('assemble','admit'),features_fit_plan,features_fit_run,features_fit_final,unit_cpus=2,unit_mem_gb=12,est_unit_min=5),
 Stage('features','P2','cpu',('features_fit',),features_plan,features_run,features_final,unit_cpus=4,unit_mem_gb=12,est_unit_min=10),
]
