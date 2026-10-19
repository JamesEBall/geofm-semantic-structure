"""Frozen-input analysis stages. Reports separate measured evidence from unqualified verdicts."""
from __future__ import annotations
from pathlib import Path
from mc.core import atomic_write_json, read_json
from mc.registry import Stage, StageFailed


def settings(ctx):
    from megaearth_stats.io import params
    return params(ctx.cfg)


def routes(ctx):
    """Analysable routes. Registered architectures must exist; descriptive ladder routes whose
    extraction did not complete are skipped and listed in the inference report."""
    from stages.extract import required_models
    required = required_models(ctx.cfg)
    out = {}
    for route in ctx.cfg['extraction']['routes']:
        for model in route['models']:
            key = f"{route['route']}-{model}"
            present = (ctx.run/'extract'/route['route']/model/'mean.npy').exists()
            if not present and key in required:
                raise StageFailed('Registered architecture features missing: '+key)
            if present:
                out[key] = [key]
    out.update(ctx.cfg['analysis']['controls'])
    for spec in ctx.cfg['analysis'].get('descriptive_routes', []):
        out.setdefault('+'.join(spec), spec)
    return out


def checked_study(ctx):
    from stages._extract_freeze import verify_freeze
    from megaearth_stats.study import load_study
    verify_freeze(ctx, ctx.cfg.get('analysis_input_freeze', 'f3'))
    return load_study(ctx.run, ctx.cfg, settings(ctx), 'test')


def plan(ctx):
    from megaearth_stats.study import available_poolings
    st = checked_study(ctx)
    out = []
    for name, spec in routes(ctx).items():
        pools = available_poolings(ctx.run, spec, settings(ctx)['poolings'])
        if settings(ctx)['primary_pooling'] not in pools:
            raise StageFailed(f'Missing primary features for {name}')
        if len(spec)==1 and '-' not in spec[0]:
            pools=[settings(ctx)['primary_pooling']]
        for pool in pools:
            for scheme in ('family','random_class'):
                for fold in range(len(st.heldout_sets(scheme))):
                    out.append({'id':f'{name}.{pool}.{scheme}.{fold}', 'route':name,'spec':spec,
                                'pooling':pool,'scheme':scheme,'fold':fold})
    return out


def engine(ctx, unit):
    from megaearth_stats.study import FeatureStore
    from megaearth_stats.readout import RouteEngine
    st=checked_study(ctx)
    spec=unit['spec']
    feats=FeatureStore(st,spec,unit['pooling'])
    fp=st.footprint(unit['route'])
    return RouteEngine(st,feats,fp,settings(ctx),'test')


def readout_run(ctx, unit, ckpt):
    from megaearth_stats.readout import Results,run_scheme,run_probe
    from megaearth_stats.io import write_npz
    eng=engine(ctx,unit);p=eng.p
    res=Results(eng.pairs)
    refs=list(p['refs'])+[r for r in p.get('text_sensitivity_arrays',[]) if r in eng.study.protos]
    run_scheme(eng,res,unit['scheme'],unit['pooling'],refs,
               {'ridge':{},'constant':{'const':True},'rank256':{'rank':p['pca_dims']}},
               gzsl_ref=p['primary_ref'] if p.get('gzsl') else None,
               ckpt=ckpt,folds=[unit['fold']])
    if unit['fold']==0 and unit['scheme']=='family':
        run_probe(eng,res,unit['pooling'])
    res.meta.update(unit=unit,original_class_idx=eng.study.original_class_idx.tolist(),
                    input_hashes=eng.study.hashes,feature_hash=eng.feats.hash,
                    alpha_selection='inner family validation only')
    path=ctx.stage_dir()/f"{unit['id']}.npz"
    write_npz(path,res.arrays,res.meta)
    return {'outputs':[path],'metrics':{'pairs':eng.pairs.n}}


def primary_routes(ctx):
    a=ctx.cfg['analysis']
    return {v['pretrained'] for v in a['architectures'].values()}


def auxiliary_plan(ctx):
    checked_study(ctx)
    out=[];a=ctx.cfg['analysis']
    names=primary_routes(ctx)
    if ctx.stage=='analysis_scaling':
        names = set(routes(ctx))
    for name in sorted(names):
        for ref in a['refs']:
            for fold in range(a['outer_folds']):
                out.append({'id':f'{name}.{ref}.{fold}','route':name,'spec':routes(ctx)[name],
                            'ref':ref,'pooling':a['primary_pooling'],'fold':fold})
    return out


def auxiliary_run(ctx, unit, ckpt):
    import numpy as np
    from megaearth_stats.readout import run_nulls_fold,run_scaling_fold
    from megaearth_stats.io import write_npz
    eng=engine(ctx,unit);p=eng.p
    if ctx.stage=='analysis_nulls':
        data=run_nulls_fold(eng,unit['fold'],unit['ref'],p['text_permutations'],p['label_shuffles'],
                            p['seed'],p['text_perm_seed_tag'],p['label_shuffle_seed_tag'],ckpt)
        arrays={k:np.asarray(v) for k,v in data.items()}
        meta={'unit':unit}
    else:
        data=run_scaling_fold(eng,unit['fold'],unit['ref'],p['scaling_m'],p['scaling_subsets'],
                              p['seed'],p['scaling_seed_tag'],ckpt)
        arrays={'pair_idx':data.pop('pair_idx'),'rr':np.asarray(data.pop('rr'))}
        meta={'unit':unit,**data}
    path=ctx.stage_dir()/f"{unit['id']}.npz"
    write_npz(path,arrays,meta)
    return {'outputs':[path]}


def rsa_plan(ctx):
    checked_study(ctx)
    return [{'id':n+'.'+ref,'route':n,'spec':spec,'ref':ref,
             'pooling':ctx.cfg['analysis']['primary_pooling']} for n,spec in routes(ctx).items()
            for ref in ctx.cfg['analysis']['refs']]


def rsa_run(ctx,unit,ckpt):
    import numpy as np
    from scipy.spatial.distance import pdist,squareform
    from scipy.stats import rankdata
    from megaearth_stats.io import write_npz
    eng=engine(ctx,unit);st=eng.study
    rows=st.rows_for_split("test");lab=st.labels_for(eng.fp)
    keep=[];means=[]
    for c in range(st.K):
        cmask=np.zeros(st.K,dtype=bool);cmask[c]=True
        rr=rows[lab.any_in(cmask)[rows]]
        if len(rr):
            # Cell-balanced category means prevent dense sampled cells dominating geometry.
            means.append(np.mean([eng.feats.rows(rr[st.cell_idx[rr]==g]).mean(0)
                                  for g in np.unique(st.cell_idx[rr])],axis=0));keep.append(c)
    if len(keep)<4:
        raise StageFailed('RSA requires at least four represented categories')
    X=np.asarray(means);P=st.protos[unit['ref']][keep]
    if np.any(np.linalg.norm(X,axis=1)==0):
        raise StageFailed('RSA has a zero feature prototype')
    dx=pdist(X,'cosine');dt=pdist(P,'cosine')
    rx=rankdata(dx);rt=rankdata(dt)
    def corr(a,b):
        a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
        if den==0:
            raise StageFailed('RSA distance vector is constant')
        return float(a@b/den)
    observed=corr(rx,rt);matrix=squareform(rx);tri=np.triu_indices(len(keep),1)
    # Nuisance-conditioned geometry is descriptive until its calibration gate passes.
    from megaearth_stats.study import FeatureStore
    app=FeatureStore(st,['appearance_rich'],eng.p['primary_pooling'])
    appmeans=[]
    for c in keep:
        cmask=np.zeros(st.K,dtype=bool);cmask[c]=True
        rr=rows[lab.any_in(cmask)[rows]]
        appmeans.append(np.mean([app.rows(rr[st.cell_idx[rr]==g]).mean(0)
                                 for g in np.unique(st.cell_idx[rr])],axis=0))
    geo_hist=st.class_histograms(eng.fp,'geo')[keep]
    nuis=np.column_stack([np.ones(len(rx)),rankdata(pdist(np.asarray(appmeans),'cosine')),
                           rankdata(pdist(geo_hist,'cosine'))])
    if not np.isfinite(nuis).all():
        raise StageFailed('Non-finite RSA nuisance geometry')
    U,D,_=np.linalg.svd(nuis,full_matrices=False);Q=U[:,D>D.max()*1e-12]
    target=rt-Q@(Q.T@rt)
    def partial(v):
        residual=v-Q@(Q.T@v);den=np.linalg.norm(residual)*np.linalg.norm(target)
        return float(residual@target/den) if den>1e-10 else None
    observed_partial=partial(rx)
    B=eng.p['mantel_permutations'];state=ckpt.load('rsa',{'null':[],'partial_null':[]})
    null=list(state['null']);partial_null=list(state['partial_null'])
    for b in range(len(null),B):
        perm=np.random.default_rng([eng.p['seed'],10103,b]).permutation(len(keep))
        permuted=matrix[perm[:,None],perm[None,:]][tri]
        null.append(corr(permuted,rt));partial_null.append(partial(permuted))
        if ckpt.due():ckpt.save('rsa',{'null':null,'partial_null':partial_null})
        ckpt.maybe_stop()
    same=(st.family[np.asarray(keep)][:,None]==st.family[np.asarray(keep)][None,:])[tri]
    ratio=float(dx[same].mean()/dx[~same].mean()) if same.any() and (~same).any() and dx[~same].mean()>0 else None
    path=ctx.stage_dir()/f"{unit['id']}.npz"
    write_npz(path,{'classes':st.original_class_idx[keep],'null':np.asarray(null),'image_distances':dx,'text_distances':dt,
                    'partial_null':np.asarray([np.nan if v is None else v for v in partial_null])},
              {'unit':unit,'spearman':observed,'mantel_p':(1+int((np.asarray(null)>=observed).sum()))/(1+B),
               'within_between_family_distance_ratio':ratio,'partial_spearman':observed_partial,
               'partial_mantel_p':((1+sum(v>=observed_partial for v in partial_null if v is not None))/(1+B)
                                   if observed_partial is not None and all(v is not None for v in partial_null) else None),
               'nuisances':['appearance_rich category geometry','cosine distances of geographic class histograms (frozen geo_grid_deg)'],
               'interval_qualified':False,'interpretation':'descriptive category geometry; no causal or population interval claim'})
    return {'outputs':[path]}


def single(ctx):
    return [{'id':'aggregate'}]


def merge_readouts(ctx):
    import numpy as np
    from megaearth_stats.io import read_npz
    merged={}
    for path in sorted((ctx.run/'analysis_readouts').glob('*.npz')):
        arrays,meta=read_npz(path);u=meta['unit'];key=(u['route'],u['pooling'],u['scheme'])
        dst=merged.setdefault(key,{'pairs':{},'values':{},'meta':[]})
        for k in ('pairs/win','pairs/cls'):
            if k in dst['pairs'] and not np.array_equal(dst['pairs'][k],arrays[k]):
                raise StageFailed('Readout pair supports differ across folds')
            dst['pairs'][k]=arrays[k]
        for name,value in arrays.items():
            if value.ndim==1 and len(value)==len(arrays['pairs/win']) and (name.endswith('/rr') or name.startswith('chance/') or name.endswith('/top1')
                                                                       or name.endswith('/rr_withinkey') or name.startswith('chance_withinkey/')):
                if name not in dst['values']:dst['values'][name]=np.full(len(value),np.nan)
                ok=np.isfinite(value)
                if np.any(np.isfinite(dst['values'][name]) & ok & (dst['values'][name]!=value)):
                    raise StageFailed('Overlapping fold scores disagree')
                dst['values'][name][ok]=value[ok]
        dst['meta'].append(meta)
    return merged


def inference_run(ctx,unit,ckpt):
    import numpy as np
    from megaearth_stats.bootstrap import make_draws,PairBoot,percentile_ci,p_null_centred,holm
    st=checked_study(ctx);p=settings(ctx);merged=merge_readouts(ctx)
    estimates=[];cache={}
    for (route,pooling,scheme),doc in sorted(merged.items()):
        win=doc['pairs']['pairs/win'];cls=doc['pairs']['pairs/cls']
        cell=st.cell_idx[win];family=st.family[cls]
        draws=make_draws(cell,family,p['bootstrap'],p['seed'])
        boot=PairBoot(cls,cell,family,draws)
        for key,values in doc['values'].items():
            if 'withinkey' in key:
                # Undefined where a key has a single held-out class; summarised on its supported pairs only.
                ok=np.isfinite(values)
                if not ok.any():
                    continue
                b=PairBoot(cls[ok],cell[ok],family[ok],make_draws(cell[ok],family[ok],p['bootstrap'],p['seed']))
                dist=b.draws_of(values[ok]);point=float(b.point(values[ok]));ci=percentile_ci(dist)
                estimates.append({'route':route,'pooling':pooling,'scheme':scheme,'metric':key,'value':point,
                                  'ci':list(ci),'pairs':int(ok.sum()),'classes':len(np.unique(cls[ok])),'descriptive':True})
                cache[(route,pooling,scheme,key)]=(doc['pairs'],np.where(ok,values,np.nan),b,dist,point)
                continue
            if not np.isfinite(values).all():
                raise StageFailed(f'Incomplete outer scores: {route}/{key}')
            dist=boot.draws_of(values)
            point=float(boot.point(values));ci=percentile_ci(dist)
            estimates.append({'route':route,'pooling':pooling,'scheme':scheme,'metric':key,'value':point,
                              'ci':list(ci),'pairs':len(win),'classes':len(np.unique(cls))})
            cache[(route,pooling,scheme,key)]=(doc['pairs'],values,boot,dist,point)
        ckpt.maybe_stop()
    strata=summarise_strata(st,cache,p)
    comparisons=[]
    for arch,ac in p['architectures'].items():
        route=ac['pretrained'];pool=p['primary_pooling']
        for ref in p['refs']:
            key=f'family.{ref}.{pool}.ridge/rr';entry=cache.get((route,pool,'family',key))
            if entry is None:raise StageFailed(f'Missing primary readout {route}/{key}')
            pairs,values,boot,dist,point=entry
            chance=cache[(route,pool,'family','chance/family')]
            controls=list(ac['random'])+list(p['controls'])
            cpoints=[];cdraws=[]
            for control in controls:
                other=cache.get((control,pool,'family',key))
                if other is None:raise StageFailed(f'Missing registered control {control}')
                if any(not np.array_equal(pairs[k],other[0][k]) for k in pairs):
                    raise StageFailed('Primary comparison has unmatched window/class supports')
                delta=values-other[1];dd=boot.draws_of(delta);dp=float(boot.point(delta))
                comparisons.append({'architecture':arch,'ref':ref,'control':control,'delta':dp,
                                    'ci':list(percentile_ci(dd)),'p_null_centred':p_null_centred(dp,dd)})
                cpoints.append(other[4]);cdraws.append(boot.draws_of(other[1]))
            cstar=dist-np.max(np.stack(cdraws),axis=0)
            comparisons.append({'architecture':arch,'ref':ref,'control':'best-control-per-draw',
                                'delta':point-max(cpoints),'ci':list(percentile_ci(cstar)),
                                'chance_corrected':point-chance[4]})
    primary=[c for c in comparisons if c['ref']==p['primary_ref'] and c['control']!='best-control-per-draw']
    # Report every seed/control contrast and Holm across architecture-level IUT p-values.
    iuts=[max(c['p_null_centred'] for c in primary if c['architecture']==arch) for arch in p['architectures']]
    adjusted=holm(iuts)
    h1={arch:{'iut_p':float(v),'holm_p':float(q)} for arch,v,q in zip(p['architectures'],iuts,adjusted)}
    auxiliary=summarise_auxiliary(ctx, st, cache)
    ladder=summarise_ladder(st,cache,p)
    skipped=(read_json(ctx.p('extract','complete.json')) or {}).get('missing_descriptive_routes',[])
    report={'estimates':estimates,'comparisons':comparisons,'h1_statistics':h1,'ladder':ladder,'skipped_descriptive_routes':skipped,
            'verdict':'UNQUALIFIED - operating-characteristic and population RSA qualification required',
            'power_qualified':False,'rsa_interval_qualified':False,'test_data_used_for_tuning':False,
            'original_class_idx':st.original_class_idx.tolist(),'auxiliary':auxiliary,'strata':strata,
            'analysis_parameter_defaults':p['_defaults_used'],
            'limits':['Intervals condition on fitted maps; this is not a refitting bootstrap.',
                      'No strong-positive or operational-negative tier is assigned without calibration gates.']}
    path=ctx.stage_dir()/'results.json';atomic_write_json(path,report)
    return {'outputs':[path],'metrics':{'estimates':len(estimates),'contrasts':len(comparisons)}}


def seen_similarity(st, p):
    """Per class: maximum text cosine (primary reference only) between the class's *family* and any class
    seen in fitting when that family is held out. Text-only by construction; no image feature enters."""
    import numpy as np
    P=st.protos[p['primary_ref']];fold=st.fold_of_class('family');out=np.full(st.K,np.nan)
    for f,held in enumerate(st.heldout_sets('family')):
        seen=np.flatnonzero((fold!=f)&(fold>=0))
        if len(held) and len(seen):
            per_class=(P[held]@P[seen].T).max(1)
            for fam in np.unique(st.family[held]):
                members=held[st.family[held]==fam]
                out[members]=per_class[np.isin(held,members)].max()
    return out


def summarise_ladder(st, cache, p):
    """Descriptive gains of every ladder route over each registered control and the per-draw best control."""
    import numpy as np
    from megaearth_stats.bootstrap import percentile_ci
    pool=p['primary_pooling'];out=[]
    for rung,routes_ in sorted((p.get('ladder') or {}).items()):
        for route in routes_:
            for ref in p['refs']:
                for metric in ('rr','rr_withinkey'):
                    key=f'family.{ref}.{pool}.ridge/{metric}';entry=cache.get((route,pool,'family',key))
                    if entry is None:
                        out.append({'rung':rung,'route':route,'ref':ref,'metric':metric,'available':False});continue
                    pairs,values,boot,dist,point=entry;ok=np.isfinite(values)
                    row={'rung':rung,'route':route,'ref':ref,'metric':metric,'available':True,'value':point,
                         'ci':list(percentile_ci(dist)),'controls':{},'descriptive':True}
                    cdraws=[];cpoints=[]
                    for control in p['controls']:
                        other=cache.get((control,pool,'family',key))
                        if other is None or not np.array_equal(np.isfinite(other[1]),ok):
                            row['controls'][control]={'available':False};continue
                        delta=(values-other[1])[ok];dd=boot.draws_of(delta);dp=float(boot.point(delta))
                        row['controls'][control]={'available':True,'delta':dp,'ci':list(percentile_ci(dd))}
                        cdraws.append(boot.draws_of(other[1][ok]));cpoints.append(other[4])
                    if metric=='rr_withinkey':
                        row['conditioning']="given the target's top-level OSM key (oracle-conditioned decomposition)"
                    if cdraws:
                        stack=np.stack(cdraws)
                        row['best_control_delta']=point-max(cpoints)
                        row['best_control_ci']=list(percentile_ci(dist-np.max(stack,axis=0)))
                        # Conservative for a positive, anti-conservative for a null (ST O4): report both ends.
                        row['worst_control_delta']=point-min(cpoints)
                        row['worst_control_ci']=list(percentile_ci(dist-np.min(stack,axis=0)))
                    out.append(row)
    # The one matched contrast in the ladder: same architecture, data and objective, metadata conditioning only.
    for a,b in p.get('matched_contrasts',[]):
        for ref in p['refs']:
            for metric in ('rr','rr_withinkey'):
                key=f'family.{ref}.{pool}.ridge/{metric}';ea=cache.get((a,pool,'family',key));eb=cache.get((b,pool,'family',key))
                row={'rung':'matched_contrast','route':f'{b} - {a}','ref':ref,'metric':metric,'descriptive':True}
                if ea is None or eb is None or not np.array_equal(np.isfinite(ea[1]),np.isfinite(eb[1])):
                    out.append({**row,'available':False});continue
                ok=np.isfinite(ea[1]);delta=(eb[1]-ea[1])[ok];boot=ea[2]
                out.append({**row,'available':True,'delta':float(boot.point(delta)),'ci':list(percentile_ci(boot.draws_of(delta))),
                            'interpretation':'metadata-conditioned pretraining, imagery-only inference; other rung differences are confounded by architecture, corpus, patch size and scale'})
    return out


def summarise_strata(st, cache, p):
    import numpy as np
    from megaearth_stats.bootstrap import PairBoot
    reports=[]
    maxsim=seen_similarity(st,p)
    pretrained={a['pretrained'] for a in p['architectures'].values()}
    for (route,pooling,scheme,key),entry in cache.items():
        if scheme!='family' or not key.endswith('.ridge/rr'):
            continue
        pairs,values,_,_,_=entry;win=pairs['pairs/win'];cls=pairs['pairs/cls']
        fams=np.unique(st.family[np.unique(cls)]);fam_sim=np.array([maxsim[np.flatnonzero(st.family==f)[0]] for f in fams])
        median=float(np.nanmedian(fam_sim)) if np.isfinite(fam_sim).any() else np.nan
        masks={f'continent={v}':st.continent[win]==v for v in sorted(set(st.continent[win]))}
        if np.isfinite(median):
            # A4 S4: transfer that depends on a near-synonym among the seen families is not transfer.
            masks['textsim=low-half']=maxsim[cls]<=median
            masks['textsim=high-half']=maxsim[cls]>median
        masks.update({f'key={v}':st.top_key[cls]==v for v in sorted(set(st.top_key[cls]))})
        masks.update({f'leave-key-out={v}':st.top_key[cls]!=v for v in sorted(set(st.top_key[cls]))})
        masks.update({'purity>=0.8':st.purity[win]>=p['stratum_purity_min'],
                      'coverage>=0.6':st.coverage[win]>=p['stratum_coverage_min'],
                      'snow<=0.05':st.snow[win]<=p['stratum_snow_max']})
        for name,mask in masks.items():
            count=int(mask.sum())
            if not count:
                reports.append({'route':route,'metric':key,'stratum':name,'available':False})
                continue
            b=PairBoot(cls[mask],st.cell_idx[win[mask]],st.family[cls[mask]],None)
            row={'route':route,'pooling':pooling,'metric':key,'stratum':name,
                 'value':float(b.point(values[mask])),'pairs':count,
                 'classes':len(np.unique(cls[mask])),'available':True,'descriptive':True}
            if route in pretrained and name.startswith('textsim='):
                # A4 S4: paired gain over each registered control inside the stratum, with its interval.
                from megaearth_stats.bootstrap import make_draws,percentile_ci
                bb=PairBoot(cls[mask],st.cell_idx[win[mask]],st.family[cls[mask]],
                            make_draws(st.cell_idx[win[mask]],st.family[cls[mask]],p['bootstrap'],p['seed']))
                row['gains']={}
                for control in p['controls']:
                    other=cache.get((control,pooling,scheme,key))
                    if other is None:
                        row['gains'][control]={'available':False};continue
                    delta=(values-other[1])[mask]
                    row['gains'][control]={'available':True,'delta':float(bb.point(delta)),'ci':list(percentile_ci(bb.draws_of(delta)))}
            reports.append(row)
    return reports


def summarise_auxiliary(ctx, st, cache):
    """Join fold-local null/scaling scores on the exact primary pair support."""
    import numpy as np
    from megaearth_stats.io import read_npz
    from megaearth_stats.bootstrap import p_permutation
    p=settings(ctx);groups={};reports={'nulls':[],'scaling':[],'rsa':[]}
    for kind in ('nulls','scaling'):
        for path in sorted((ctx.run/f'analysis_{kind}').glob('*.npz')):
            arr,meta=read_npz(path);u=meta['unit'];k=(kind,u['route'],u['ref'])
            key=f"family.{u['ref']}.{u['pooling']}.ridge/rr"
            entry=cache[(u['route'],u['pooling'],'family',key)]
            n=len(entry[1]);indices=arr['pair_idx'];target=groups.setdefault(k,{'entry':entry,'values':{},'extra':[]})
            if kind=='nulls':
                measures={'text':arr['text_rr'],'shuffle':arr['shuf_rr']}
                target['extra'].append(arr['shuf_unchanged'].tolist())
            else:
                measures={f'm={m}':rr[None,:] for m,rr in zip(meta['m'],arr['rr'])}
                target['extra'].append({'fold':u['fold'],'n_train':meta['n_train'],'n_subsets':meta['n_subsets']})
            for measure,v in measures.items():
                if measure not in target['values']:
                    target['values'][measure]=np.full((len(v),n),np.nan)
                if np.isfinite(target['values'][measure][:,indices]).any():
                    raise StageFailed('Duplicate auxiliary fold support')
                target['values'][measure][:,indices]=v
    for (kind,route,ref),group in groups.items():
        entry=group['entry'];boot=entry[2]
        for measure,v in group['values'].items():
            if not np.isfinite(v).all():
                reports[kind].append({'route':route,'ref':ref,'measure':measure,'available':False,
                                      'reason':'Unsupported subset or incomplete fold support'})
                continue
            values=np.asarray(boot.point(v.T))
            report={'route':route,'ref':ref,'measure':measure,'available':True,'values':values.tolist(),
                    'mean':float(values.mean()),'extra':group['extra']}
            if kind=='nulls':report['permutation_p']=p_permutation(entry[4],values)
            reports[kind].append(report)
    for path in sorted((ctx.run/'analysis_rsa').glob('*.npz')):
        _,meta=read_npz(path);reports['rsa'].append(meta)
    return reports


BASE=('freeze_f3','families','text_embed')
STAGES=[
    Stage('analysis_readouts','P2','cpu',BASE,plan,readout_run,unit_cpus=8,unit_mem_gb=48),
    Stage('analysis_nulls','P2','cpu',BASE,auxiliary_plan,auxiliary_run,unit_cpus=8,unit_mem_gb=48),
    Stage('analysis_scaling','P2','cpu',BASE,auxiliary_plan,auxiliary_run,unit_cpus=8,unit_mem_gb=48),
    Stage('analysis_rsa','P2','cpu',BASE,rsa_plan,rsa_run,unit_cpus=8,unit_mem_gb=48),
    # Completed full run: Slurm MaxRSS 125623124 KiB (~120 GiB), job 176540.
    # Reserve headroom on the high-memory pool for future reproductions.
    Stage('analysis_inference','P2','himem',('analysis_readouts','analysis_nulls','analysis_scaling','analysis_rsa'),
          single,inference_run,unit_cpus=8,unit_mem_gb=160),
]
