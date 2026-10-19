"""Canonical artifact integration and split-isolation regression checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from mc.core import Store,Checkpoint,atomic_write_json
from mc.context import Ctx
from megaearth_stats.io import params,read_npz
from megaearth_stats.study import load_study,FeatureStore
from megaearth_stats.io import SplitAccessError
from stages import analysis


def parquet(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows),path)


@pytest.fixture
def world(tmp_path,monkeypatch):
    cfg=json.loads((Path(__file__).resolve().parents[2]/'config.json').read_text())
    cfg.update(root=str(tmp_path),_profile='selftest')
    cfg['analysis'].update(outer_folds=3,inner_folds=2,alphas=[.1,1.],poolings=['mean'],bootstrap=30,
                           text_permutations=2,label_shuffles=2,mantel_permutations=9,
                           scaling_m=[2,'all'],scaling_subsets=2,pca_dims=2,
                           gzsl_gamma_grid=[-1,1,.5],refs=['qwen_instructed'],text_sensitivity_arrays=[])
    cfg['extraction']['routes']=[{'route':'r1','models':['clay']}]
    cfg['analysis']['architectures']={'clay':{'pretrained':'r1-clay','random':['r1-random_clay_101']}}
    cfg['analysis']['controls']={'appearance_rich':['appearance_rich']}
    cfg['analysis']['descriptive_routes']=[]
    from megaearth_stats.folds import make_folds
    K=12;family=np.arange(K)//2
    folds=make_folds(family,['landuse']*K,3,51426)
    rows=[];labs=[];rng=np.random.default_rng(82)
    for split,offset in [('train',0),('val',100),('test',200)]:
        for c in range(K):
            for j in range(4):
                i=len(rows);w=f'w{i:07d}'
                rows.append(dict(widx=i,window_id=w,split=split,deg_cell=f'E{offset+j}',lon=j,lat=10,
                                 datetime='2025-06-01T00:00:00Z',target_class_idx=c,continent='Europe',
                                 purity=1.,snow_frac=0.,source='synthetic'))
                lab=dict(widx=i,window_id=w,multi_idx=[c],multi_cov=[1.])
                lab.update({col:[c] for col in cfg['labels']['footprint_label_column'].values()});labs.append(lab)
    parquet(tmp_path/'admit/windows_final.parquet',[dict(widx=r['widx'],window_id=r['window_id'],retained=True) for r in rows])
    parquet(tmp_path/'assemble/windows_qa.parquet',rows);parquet(tmp_path/'labels_exact/labels.parquet',labs)
    parquet(tmp_path/'vocab/classes.parquet',[dict(class_idx=c,class_id=f'landuse=c{c}',name=f'c{c}',primary_key='landuse') for c in range(K)])
    parquet(tmp_path/'admit/classes_final.parquet',[dict(class_idx=c,admitted=c!=11) for c in range(K)])
    parquet(tmp_path/'families/families.parquet',[dict(class_idx=c,family_id=int(family[c])) for c in range(K)])
    atomic_write_json(tmp_path/'families/folds.json',folds)
    (tmp_path/'text').mkdir();P=rng.normal(size=(K,5));P/=np.linalg.norm(P,axis=1)[:,None]
    np.save(tmp_path/'text/emb_qwen_instructed.npy',P)
    X=np.stack([P[r['target_class_idx']]+rng.normal(0,.02,5) for r in rows]).astype('float32')
    for path in [tmp_path/'extract/r1/clay/mean.npy',tmp_path/'extract/r1/random_clay_101/mean.npy',tmp_path/'features/appearance_rich.npy']:
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,X)
        atomic_write_json(path.with_suffix('.meta.json'),{'rows':len(rows),'d':5})
    ctx=Ctx(cfg,Path(__file__).resolve().parents[2],tmp_path,tmp_path,tmp_path/'shared',Store(tmp_path))
    from stages import _extract_freeze
    monkeypatch.setattr(_extract_freeze,'verify_freeze',lambda *a: {'test_stub':True})
    return ctx


def test_canonical_mapping_and_test_feature_seal(world):
    st=load_study(world.run,world.cfg,params(world.cfg),'val')
    assert st.K==11
    assert st.original_class_idx.tolist()==list(range(11))
    test=np.flatnonzero(st.split==2)
    assert not st.labels['r1'].lengths()[test].any()
    assert not st.family_sets('r1').lengths()[test].any()
    assert not st.family_sets('r1', exclusion=False).lengths()[test].any()
    fs=FeatureStore(st,['r1-clay'],'mean')
    with pytest.raises(SplitAccessError):fs.rows(test[:1])
    assert not st.class_sel[test].any() or np.all(st.class_sel[test]==-1)


def test_widx_mismatch_refused(world):
    path=world.run/'labels_exact/labels.parquet';rows=pq.read_table(path).to_pylist()
    rows[0]['window_id']='wrong';parquet(path,rows)
    with pytest.raises(ValueError,match='widx/window_id'):
        load_study(world.run,world.cfg,params(world.cfg),'test')


def test_readout_unit_uses_validation_for_alpha_and_maps_original_ids(world):
    unit={'id':'one','route':'r1-clay','spec':['r1-clay'],'pooling':'mean','scheme':'family','fold':0}
    world.stage='analysis_readouts'
    ckpt=Checkpoint(world.store,world.stage,'one')
    result=analysis.readout_run(world,unit,ckpt)
    arrays,meta=read_npz(result['outputs'][0])
    assert meta['original_class_idx']==list(range(11))
    key='family.qwen_instructed.mean.ridge'
    assert np.isfinite(arrays[f'{key}/rr']).any()
    assert meta['fits'][key]['0']['selection']['n_pairs']>0
    # A pure rescaling/translation of test features must not affect chosen alpha.
    source=world.run/'extract/r1/clay/mean.npy';X=np.load(source)
    st=load_study(world.run,world.cfg,params(world.cfg),'test');X[st.split==2]=X[st.split==2]*100+100
    np.save(source,X)
    result=analysis.readout_run(world,unit,ckpt)
    _,second=read_npz(result['outputs'][0])
    assert second['fits'][key]['0']['selection']==meta['fits'][key]['0']['selection']


def test_nulls_and_scaling_small_run(world):
    unit={'id':'one','route':'r1-clay','spec':['r1-clay'],'pooling':'mean','ref':'qwen_instructed','fold':0}
    for stage in ['analysis_nulls','analysis_scaling']:
        world.stage=stage
        result=analysis.auxiliary_run(world,unit,Checkpoint(world.store,stage,'one'))
        arrays,meta=read_npz(result['outputs'][0])
        assert len(arrays['pair_idx'])>0
        assert np.isfinite(arrays['text_rr' if stage=='analysis_nulls' else 'rr']).all()


def test_rsa_reports_no_qualified_population_interval(world):
    unit={'id':'one','route':'r1-clay','spec':['r1-clay'],'pooling':'mean','ref':'qwen_instructed'}
    world.stage='analysis_rsa'
    result=analysis.rsa_run(world,unit,Checkpoint(world.store,world.stage,'one'))
    arrays,meta=read_npz(result['outputs'][0])
    assert len(arrays['null'])==9
    assert not meta['interval_qualified']


def test_primary_inference_merges_fold_support_and_keeps_verdict_unqualified(world):
    # Matched feature controls deliberately tie here. Their paired confidence interval is zero.
    world.cfg['extraction']['routes'][0]['models'].append('random_clay_101')
    for name,spec in analysis.routes(world).items():
        for scheme in ('family','random_class'):
            for fold in range(3):
                unit={'id':f'{name}.{scheme}.{fold}','route':name,'spec':spec,'pooling':'mean','scheme':scheme,'fold':fold}
                world.stage='analysis_readouts'
                analysis.readout_run(world,unit,Checkpoint(world.store,world.stage,unit['id']))
    world.stage='analysis_inference'
    output=analysis.inference_run(world,{'id':'aggregate'},Checkpoint(world.store,world.stage,'aggregate'))
    report=json.loads(Path(output['outputs'][0]).read_text())
    assert report['verdict'].startswith('UNQUALIFIED')
    assert not report['test_data_used_for_tuning']
    assert report['comparisons']
    for contrast in report['comparisons']:
        assert abs(contrast['delta'])<1e-12
        assert np.allclose(contrast['ci'],[0.,0.],atol=1e-12)


def test_within_key_rank_matches_primary_when_all_keys_agree(world):
    # Every synthetic class shares the key 'landuse', so the within-key gallery is the full gallery.
    unit={'id':'one','route':'r1-clay','spec':['r1-clay'],'pooling':'mean','scheme':'family','fold':0}
    world.stage='analysis_readouts'
    arrays,_=read_npz(analysis.readout_run(world,unit,Checkpoint(world.store,world.stage,'one'))['outputs'][0])
    key='family.qwen_instructed.mean.ridge'
    ok=np.isfinite(arrays[f'{key}/rr'])
    np.testing.assert_allclose(arrays[f'{key}/rr_withinkey'][ok],arrays[f'{key}/rr'][ok])
    np.testing.assert_allclose(arrays['chance_withinkey/family'][ok],arrays['chance/family'][ok])


def test_ladder_and_similarity_strata_are_descriptive(world):
    world.cfg['extraction']['routes'][0]['models'].append('random_clay_101')
    world.cfg['analysis']['ladder']={'rung0_random':['r1-random_clay_101'],'rung2_imagery_plus_metadata':['r1-clay'],
                                    'rung4_language_aligned':['r1rgb224-remoteclip']}
    for name,spec in analysis.routes(world).items():
        for fold in range(3):
            unit={'id':f'{name}.family.{fold}','route':name,'spec':spec,'pooling':'mean','scheme':'family','fold':fold}
            world.stage='analysis_readouts'
            analysis.readout_run(world,unit,Checkpoint(world.store,world.stage,unit['id']))
    world.stage='analysis_inference'
    output=analysis.inference_run(world,{'id':'aggregate'},Checkpoint(world.store,world.stage,'aggregate'))
    report=json.loads(Path(output['outputs'][0]).read_text())
    rows=report['ladder'];assert rows
    missing=[r for r in rows if r['route']=='r1rgb224-remoteclip'];assert missing and not missing[0]['available']
    clay=[r for r in rows if r['route']=='r1-clay' and r['metric']=='rr' and r['available']]
    assert clay and clay[0]['descriptive'] and abs(clay[0]['controls']['appearance_rich']['delta'])<1e-12
    within=[r for r in rows if r['route']=='r1-clay' and r['metric']=='rr_withinkey'];assert within and within[0]['available']
    strata={s['stratum'] for s in report['strata']}
    assert {'textsim=low-half','textsim=high-half'}<=strata
    assert any(e['metric'].endswith('rr_withinkey') and e.get('descriptive') for e in report['estimates'])
    assert report['verdict'].startswith('UNQUALIFIED')


def test_routes_skip_missing_descriptive_but_require_registered(world):
    world.cfg['extraction']['routes'].append({'route':'r1','models':['prithvi_tl']})   # no features on disk
    assert 'r1-prithvi_tl' not in analysis.routes(world) and 'r1-clay' in analysis.routes(world)
    world.cfg['analysis']['architectures']['tl']={'pretrained':'r1-prithvi_tl','random':[]}
    with pytest.raises(Exception,match='Registered architecture features missing'):
        analysis.routes(world)


def test_retained_cohort_controls_fit_validation_pairs_geometry_and_histograms(world):
    """Excluded cap rows cannot affect any fitted/scored/nuisance result."""
    from megaearth_stats.readout import RouteEngine
    from megaearth_stats.study import build_pairs
    mask_path = world.run/'admit/windows_final.parquet'
    mask = pq.read_table(mask_path).to_pylist()
    excluded = [0, 48, 96]  # one row in each split, all from an admitted class
    for i in excluded:
        mask[i]['retained'] = False
    parquet(mask_path, mask)
    p = params(world.cfg)
    st = load_study(world.run, world.cfg, p, 'test')
    fs = FeatureStore(st, ['r1-clay'], 'mean')
    eng = RouteEngine(st, fs, 'r1', p, 'test')
    assert st.n == 144 and st.window_ids[96] == 'w0000096'
    assert 0 not in eng.train_rows and 48 not in eng.val_rows
    assert 96 not in eng.pairs.win and eng.pairs.n == 43
    assert build_pairs(st, 'r1', 'train').n == 43
    assert build_pairs(st, 'r1', 'val').n == 43
    assert st.class_histograms('r1', 'geo')[0].sum() == 9
    np.testing.assert_allclose(eng.m0, fs.rows(eng.train_rows).mean(0), rtol=1e-6)
    with pytest.raises(SplitAccessError):
        fs.rows(np.array([96]))
    unit = dict(id='retained', route='r1-clay', spec=['r1-clay'], pooling='mean', scheme='family', fold=0)
    world.stage = 'analysis_readouts'
    before = analysis.readout_run(world, unit, Checkpoint(world.store, world.stage, 'retained-before'))
    arrays, meta = read_npz(before['outputs'][0])
    world.stage = 'analysis_rsa'
    rsa_unit = dict(id='retained-rsa', route='r1-clay', spec=['r1-clay'], pooling='mean', ref='qwen_instructed')
    before_rsa = analysis.rsa_run(world, rsa_unit, Checkpoint(world.store, world.stage, 'rsa-before'))
    rsa_arrays, rsa_meta = read_npz(before_rsa['outputs'][0])
    for name in ('extract/r1/clay/mean.npy', 'features/appearance_rich.npy'):
        path = world.run/name
        x = np.load(path)
        x[excluded] = np.arange(x.shape[1]) * 1e6
        np.save(path, x)
    windows_path = world.run/'assemble/windows_qa.parquet'
    windows = pq.read_table(windows_path).to_pylist()
    for i in excluded:
        windows[i]['lat'], windows[i]['lon'] = -80., 179.
    parquet(windows_path, windows)
    world.stage = 'analysis_readouts'
    after = analysis.readout_run(world, unit, Checkpoint(world.store, world.stage, 'retained-after'))
    arrays2, meta2 = read_npz(after['outputs'][0])
    for name in arrays:
        np.testing.assert_allclose(arrays[name], arrays2[name], rtol=0, atol=0, equal_nan=True)
    assert meta['fits'] == meta2['fits']
    world.stage = 'analysis_rsa'
    after_rsa = analysis.rsa_run(world, rsa_unit, Checkpoint(world.store, world.stage, 'rsa-after'))
    rsa_arrays2, rsa_meta2 = read_npz(after_rsa['outputs'][0])
    for name in rsa_arrays:
        np.testing.assert_allclose(rsa_arrays[name], rsa_arrays2[name], rtol=0, atol=0, equal_nan=True)
    assert rsa_meta['spearman'] == rsa_meta2['spearman']
    assert rsa_meta['partial_spearman'] == pytest.approx(rsa_meta2['partial_spearman'], abs=1e-12)


@pytest.mark.parametrize('error', ['missing', 'identity', 'nullable'])
def test_canonical_retention_mask_fails_closed(world, error):
    path = world.run/'admit/windows_final.parquet'
    if error == 'missing':
        path.unlink()
    else:
        records = pq.read_table(path).to_pylist()
        records[0]['window_id' if error == 'identity' else 'retained'] = 'wrong' if error == 'identity' else None
        parquet(path, records)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_study(world.run, world.cfg, params(world.cfg), 'test')


def test_actual_retained_cohort_has_8755_primary_test_pairs(tmp_path):
    """Independent audit of the shipped support tables through the corrected loader."""
    from megaearth_stats.study import build_pairs
    repo = Path(__file__).resolve().parents[4]
    source = repo/'output/megaearth-review-tables'
    if not source.exists():
        pytest.skip('Downloaded review tables are not present on this host')
    cfg = json.loads((Path(__file__).resolve().parents[2]/'config.json').read_text())
    for stage in ('assemble', 'labels_exact', 'admit', 'vocab', 'families'):
        (tmp_path/stage).symlink_to(source/stage, target_is_directory=True)
    (tmp_path/'text').mkdir()
    nclasses = pq.read_table(source/'vocab/classes.parquet').num_rows
    for ref in cfg['analysis']['refs']:
        np.save(tmp_path/'text'/f'emb_{ref}.npy', np.ones((nclasses, 2)))
    st = load_study(tmp_path, cfg, params(cfg), 'test')
    assert st.n == 44355 and st.active.sum() == 34992 and st.K == 188
    assert len(st.rows_for_split('train')) == 24574
    assert len(st.rows_for_split('val')) == 3539
    assert len(st.rows_for_split('test')) == 6879
    pairs = build_pairs(st, 'r1', 'test')
    assert pairs.n == 8755 and len(np.unique(pairs.win)) == 5942
    assert st.active[pairs.win].all()
    # Full-taxonomy family exclusion removes non-admitted siblings too.
    train = st.rows_for_split('train')
    train = train[st.labels_for('r1').lengths()[train] > 0]
    exact, family = st.exclusion_sets('r1'), st.family_sets('r1')
    additional, retained_fit = [], []
    for H in st.heldout_sets('family'):
        hm = np.zeros(st.K, dtype=bool); hm[H] = True
        fm = np.zeros(family.indices.max()+1, dtype=bool); fm[st.family[H]] = True
        before = train[~exact.any_in(hm)[train]]
        after = before[~family.any_in(fm)[before]]
        additional.append(len(before)-len(after)); retained_fit.append(len(after))
    assert additional == [26, 66, 216, 157, 205]
    assert retained_fit == [17585, 15498, 14359, 7902, 14680]


def test_nonadmitted_sibling_exclusion_for_family_inner_scaling_and_shuffle(world):
    from megaearth_stats.readout import RouteEngine, run_nulls_fold, run_scaling_fold
    # Class 11 is not admitted, but remains in frozen family 5 with class 10.
    path = world.run/'labels_exact/labels.parquet'
    labels = pq.read_table(path).to_pylist()
    labels[0]['multi_idx'] = [0, 11]; labels[0]['multi_cov'] = [1., .05]
    labels[1]['multi_idx'] = [0, 11]; labels[1]['multi_cov'] = [1., .049]
    # A positive non-admitted sibling must also obey scaling subset membership.
    for col in world.cfg['labels']['footprint_label_column'].values():
        labels[2][col] = [0, 11]
    parquet(path, labels)
    p = params(world.cfg)
    st = load_study(world.run, world.cfg, p, 'test')
    eng = RouteEngine(st, FeatureStore(st, ['r1-clay'], 'mean'), 'r1', p, 'test')
    H = np.zeros(st.K, dtype=bool); H[10] = True
    assert 0 not in eng.eligible(eng.train_rows, H, scheme='family')
    assert 1 in eng.eligible(eng.train_rows, H, scheme='family')
    assert 2 not in eng.eligible(eng.train_rows, H, scheme='family')
    assert 0 in eng.eligible(eng.train_rows, H, scheme='random_class')
    allowed = np.zeros(st.K, dtype=bool); allowed[0:2] = True
    assert 2 not in eng.eligible(eng.train_rows, np.zeros(st.K, dtype=bool), allowed, scheme='family')
    f = next(f for f, hs in enumerate(st.heldout_sets('family')) if 10 in hs)
    ctx = eng.outer_context('family', f)
    assert 0 not in ctx.rows and ctx.info['n_additional_family_excluded'] >= 1
    # Check real inner and shuffled-null fit callsites, not only the helper.
    original = eng.inner_contexts
    checked = []
    def audit_inner(*args, **kwargs):
        inner = original(*args, **kwargs)
        for ic in inner:
            if ic is None: continue
            fm = np.zeros(eng.family_excl.indices.max()+1, dtype=bool)
            fm[st.family[ic.H]] = True
            assert not eng.family_excl.any_in(fm)[ic.rows].any()
            if kwargs.get('family_excl') is not None:
                assert not kwargs['family_excl'].any_in(fm)[ic.rows].any()
                checked.append('shuffle')
            else:
                checked.append('plain')
        return inner
    eng.inner_contexts = audit_inner
    other = next(i for i in range(3) if i != f and 0 not in st.heldout_sets('family')[i])
    eng.outer_context('family', other)
    run_nulls_fold(eng, other, 'qwen_instructed', 1, 1, 123, 11, 12)
    run_scaling_fold(eng, other, 'qwen_instructed', [2, 'all'], 1, 123, 13)
    assert 'plain' in checked and 'shuffle' in checked
    assert st.K == 11 and st.original_class_idx.tolist() == list(range(11))
