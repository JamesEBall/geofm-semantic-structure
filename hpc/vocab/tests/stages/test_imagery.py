"""Imagery contracts, native-grid regression and synthetic stage integration."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

VOCAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(VOCAB/'code'))
from stages import imagery as im
from stages._imagery_io import BANDS10,BANDS20,analysis_crop,reflectance,write_shard,open_shard,write_parquet
from stages._imagery_catalogue import radiometry,asset_for,season,bounds_for,search
from stages._imagery_acquire import read_scene,quality_scl,QualityReject,TechnicalFailure,acquire_window,cached_window,match_scene
from stages._imagery_features import routes,bilinear,appearance42,descriptors,rff_features,metadata
from mc.context import make_ctx
from mc.core import sha256_file
from mc.registry import StageFailed


def test_product_identity_matches_provider_timestamp_and_tile_schema_difference():
    product='S2C_MSIL2A_20250801T212541_N0511_R043_T06WWT_20250802T000312.SAFE'
    pc={'properties':{'s2:product_uri':product,'s2:mgrs_tile':'06WWT','datetime':'2025-08-01T21:25:41.025000Z'}}
    es={'properties':{'s2:product_uri':product,'mgrs:utm_zone':6,'mgrs:latitude_band':'W',
                      'mgrs:grid_square':'WT','datetime':'2025-08-01T21:27:29.920000Z'}}
    assert match_scene(pc,[es]) is es
    assert match_scene(es,[pc]) is pc
    different=copy.deepcopy(pc)
    different['properties']['s2:product_uri']=product.replace('20250802T000312','20250803T000312')
    assert match_scene(pc,[different]) is None
    assert match_scene(pc,[{'properties':{'datetime':pc['properties']['datetime'],'s2:mgrs_tile':'06WWT'}}]) is None


def test_legacy_scene_matching_requires_explicit_tile():
    p={'properties':{'datetime':'2025-06-01T10:00:00Z'}}
    assert match_scene(p,[copy.deepcopy(p)]) is None
    p['properties']['s2:mgrs_tile']='30UWH'
    other=copy.deepcopy(p);other['properties']['datetime']='2025-06-01T10:00:00.500Z'
    assert match_scene(p,[other]) is other


@pytest.fixture
def cfg(tmp_path):
    c=json.loads((VOCAB/'config.json').read_text())
    c['root']=str(tmp_path/'root');c['_profile']='synthetic'
    return c


@pytest.fixture
def cog_world(tmp_path,cfg):
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer
    lon,lat=Transformer.from_crs(32630,4326,always_xy=True).transform(500000,6200000)
    row=dict(window_id='w0000000',lon=lon,lat=lat,utm_epsg=32630,split='val',deg_cell='E003N55',pilot=True)
    epsg,x0,y0,corners=bounds_for(row)
    item={'id':'scene1','_local_fixture':True,'properties':{'proj:epsg':epsg,'datetime':'2025-06-01T10:00:00Z',
          's2:processing_baseline':'05.11','s2:mgrs_tile':'30UWH','eo:cloud_cover':0.,'s2:snow_ice_percentage':0.},'assets':{}}
    originals={}
    for i,band in enumerate(BANDS10+BANDS20+('SCL',)):
        px=10 if band in BANDS10 else 20
        side=512 if px==10 else 256
        yy,xx=np.indices((side,side))
        values=(1500+i*200+xx+yy).astype('uint16') if band!='SCL' else np.full((side,side),4,dtype='uint8')
        path=tmp_path/(band+'.tif')
        with rasterio.open(path,'w',driver='GTiff',height=side,width=side,count=1,dtype=values.dtype,
               crs=f'EPSG:{epsg}',transform=from_origin(x0-1280,y0+1280,px,px),tiled=True,blockxsize=128,blockysize=128) as ds:
            ds.write(values,1)
        item['assets'][band]={'href':str(path),'eo:bands':[{'name':band}]}
        originals[band]=values
    signer=SimpleNamespace(sign=lambda href,source,force=False:href)
    return row,item,signer,originals


def test_native_scene_exact_slices_and_no_resampling(cog_world,cfg):
    row,item,signer,originals=cog_world
    result=read_scene(item,row,'planetary_computer',cfg,signer)
    assert result['b10'].dtype==np.uint16
    np.testing.assert_array_equal(result['b10'][0],originals['B02'][128:384,128:384])
    np.testing.assert_array_equal(result['b20'][0],originals['B05'][64:192,64:192])
    assert result['meta']['dn_offset']==-1000
    assert result['meta']['alignment_error_m']==0
    assert result['meta']['bad_frac']==0


def test_scl_boundary_and_snow_recording(cfg):
    a=np.full((128,128),4,dtype='uint8')
    a[32:33,32:72]=9
    assert quality_scl(a,cfg)[0]==40/4096
    a[33,32]=9
    with pytest.raises(QualityReject):quality_scl(a,cfg)
    a[:]=11
    assert quality_scl(a,cfg)==(0.,1.)


def test_radiometry_metadata_and_signed_url_guards(cog_world):
    _,item,_,_=cog_world
    a=item['assets']['B02']
    assert radiometry(item,a,'planetary_computer')==(-1000.,.0001)
    assert reflectance(np.array([0,1000,2000],dtype='uint16'),-1000,.0001).tolist()==pytest.approx([-.1,0,.1])
    es={'raster:bands':[{'scale':.0001,'offset':-.1}]}
    assert radiometry(item,es,'earth_search')==(-1000.,.0001)
    es['raster:bands'][0]['offset']=0
    with pytest.raises(ValueError):radiometry(item,es,'earth_search')
    with pytest.raises(ValueError):radiometry(item,{},'earth_search')
    item=copy.deepcopy(item);item['assets']['B02']['href']+='?sig=secret'
    with pytest.raises(ValueError):asset_for(item,'B02','planetary_computer')


def test_max_six_scenes_and_technical_fallback(cfg):
    row={'window_id':'w0'}
    items=[{'id':str(i),'properties':{'s2:mgrs_tile':'T','datetime':f'2025-06-{i+1:02d}T00:00:00Z'}} for i in range(7)]
    catalogue={'sources':{'planetary_computer':items,'earth_search':[]}}
    seen=[]
    def reader(item,*args):
        seen.append(item['id'])
        if item['id']!='5':raise QualityReject('no_clear_scene')
        return {'meta':{'source':'planetary_computer'}}
    result=acquire_window(row,catalogue,cfg,None,reader=reader)
    assert seen==list('012345') and result['meta']['tries']==6
    def cloudy(item,*args):raise QualityReject('no_clear_scene')
    assert acquire_window(row,catalogue,cfg,None,reader=cloudy)['reject']['tries']==6
    catalogue['sources']['earth_search']=[items[0]]
    def fallback(item,row,source,*args):
        if source=='planetary_computer':raise TechnicalFailure()
        return {'meta':{'source':source}}
    assert acquire_window(row,catalogue,cfg,None,reader=fallback)['meta']['fallback_reason']=='primary_COG_technical_failure_same_acquisition'


def test_retry_cache_and_shard_hash(cog_world,cfg,tmp_path):
    row,item,signer,_=cog_world
    fn=lambda:read_scene(item,row,'planetary_computer',cfg,signer)
    a=cached_window(tmp_path/'cache',row['window_id'],fn)
    b=cached_window(tmp_path/'cache',row['window_id'],lambda:pytest.fail('cache missed'))
    np.testing.assert_array_equal(a['b10'],b['b10'])
    a['meta']['tries']=1
    ctx=make_ctx(cfg,VOCAB,'acquire_pilot')
    path=ctx.stage_dir()/'shards/shard-00000'
    write_shard(path,[a],[])
    loaded=open_shard(ctx,path)
    np.testing.assert_array_equal(loaded['b20'][0],a['b20'])
    with open(path/'b10.npy','r+b') as f:f.seek(-1,2);f.write(b'X')
    with pytest.raises(ValueError,match='checksum'):open_shard(ctx,path)


def test_r1_half_pixel_r2_frozen_golden(cog_world,cfg):
    sys.path.insert(0,str(VOCAB.parents[1]/'src'))
    from geofm_jlens.semantic_s2_inputs_v1 import common_grid
    row,item,signer,_=cog_world
    a=read_scene(item,row,'planetary_computer',cfg,signer)
    r1,r2=routes(a['b10'],a['b20'],-1000,.0001)
    assert r1.shape==(6,128,128) and r2.shape==(6,64,64)
    # The published grid20 consumes raw integer DN and applies the baseline offset itself;
    # common_grid hard-codes offset 0 (baselines < 04.00). Replicate its exact integer
    # aggregation and resize with the -1000 offset used for baseline >= 04.00 scenes.
    import torch
    import torch.nn.functional as F
    from geofm_jlens.semantic_nuisance_inputs_v1 import grid20
    raw={b:a['b10'][i,68:188,68:188] for i,b in enumerate(BANDS10[:3])}
    raw.update({b:a['b20'][i,34:94,34:94] for i,b in enumerate(BANDS20) if b in ('B8A','B11','B12')})
    masks={k:np.ones_like(v,dtype=bool) for k,v in raw.items()}
    native=[]
    for b in list(BANDS10[:3])+['B8A','B11','B12']:
        x,m=grid20(raw[b],masks[b],10 if b in BANDS10 else 20,-1000)
        assert x.shape==(60,60) and m.all()
        native.append(x)
    gold=F.interpolate(torch.from_numpy(np.stack(native).astype(np.float32))[None],size=(64,64),
                       mode='bilinear',align_corners=False,antialias=False)[0].numpy()
    np.testing.assert_allclose(r2,gold,atol=1e-6)
    # The offset-0 published path differs from the >= 04.00 path by exactly -0.1 reflectance.
    _,gold0=common_grid(raw,masks)
    np.testing.assert_allclose(r2,gold0-0.1,atol=1e-6)
    ramp=np.tile(np.arange(64,dtype='float32'),(64,1))[None]
    up=bilinear(ramp,128)
    np.testing.assert_allclose(up[0,20,2:-2],(np.arange(128)[2:-2]+.5)/2-.5)
    a0=np.ones((6,64,64),dtype='float32')
    assert appearance42(a0).shape==(42,)


def test_descriptor_histograms_and_train_only_rff(cfg):
    rng=np.random.default_rng(2)
    a=rng.uniform(.1,.9,(6,128,128)).astype('float32')
    d=descriptors(a,np.tile(np.linspace(0,1,17),(6,1)))
    assert d.shape==(308,) and np.isfinite(d).all()
    assert np.allclose(d[42:138].reshape(6,16).sum(1),1)
    x=rng.normal(size=(8,308)).astype('float32');train=np.arange(8)<4
    a,fit=rff_features(x,train,cfg)
    x[~train]*=100
    b,fit2=rff_features(x,train,cfg)
    assert fit==fit2
    np.testing.assert_array_equal(a[train],b[train])


@pytest.mark.parametrize('excluded_cap', [False, True])
def test_synthetic_assemble_and_features(cog_world,cfg,tmp_path,excluded_cap):
    row,item,signer,_=cog_world
    ctx=make_ctx(cfg,VOCAB,'assemble');ckpt=SimpleNamespace(maybe_stop=lambda:None)
    accepted=[];selected_rows=[]
    for i,split in enumerate(('train','train','val','test')):
        r={**row,'window_id':f'w{i:07d}','split':split,'pilot':split=='val','priority':i,
           'biome_id':i%2,'mapping_density':float(i),'labels_idx':[0],'labels_cov':[.7]}
        image=read_scene(item,r,'planetary_computer',cfg,signer)
        image['b10']=image['b10']+(10000 if excluded_cap and i==1 else i)
        import hashlib
        image['meta']['pixel_sha256']=hashlib.sha256(image['b10'].tobytes()+image['b20'].tobytes()+image['scl'].tobytes()).hexdigest()
        image['meta']['tries']=1
        accepted.append(image);selected_rows.append(r)
    write_parquet(ctx.p('select','windows.parquet'),selected_rows)
    write_shard(ctx.p('acquire','shards','shard-00000'),accepted,[])
    result=im.assemble_run(ctx,{'id':'all'},ckpt)
    table=pd.read_parquet(ctx.p('assemble','windows_qa.parquet'))
    assert table.widx.tolist()==[0,1,2,3]
    assert table.window_id.tolist()==[r['window_id'] for r in selected_rows]
    write_parquet(ctx.p('admit','windows_final.parquet'), [dict(widx=int(r.widx),window_id=r.window_id,retained=not(excluded_cap and r.widx==1)) for r in table.itertuples()])
    ctx.stage='features_fit';fit_units=im.features_fit_plan(ctx);ctx.store.write_units('features_fit',fit_units)
    for u in fit_units:im.features_fit_run(ctx,u,ckpt)
    fit=im.features_fit_final(ctx);assert fit['train_windows']==(1 if excluded_cap else 2)
    bins=json.loads(ctx.p('features_fit','bins.json').read_text());assert len(bins['edges'])==6 and len(bins['band_stats']['mean'])==6
    ctx.stage='features';units=im.features_plan(ctx)
    ctx.units=lambda:units
    for u in units:im.features_run(ctx,u,ckpt)
    report=im.features_final(ctx)
    assert report['rows']==4
    assert report['dimensions']['appearance42']==42
    assert report['dimensions']['appearance_rich']==308
    assert np.load(ctx.p('features','appearance_rff.npy')).shape==(4,cfg['features']['appearance_rff_dim'])
    transforms=json.loads(ctx.p('features','transforms.json').read_text())
    rich=np.load(ctx.p('features','appearance_rich.npy'))
    selected=[0] if excluded_cap else [0,1]
    np.testing.assert_allclose(transforms['appearance_rff']['mean'],rich[selected].mean(0,dtype=np.float64))
    assert transforms['metadata']['biomes_fit']==(['0'] if excluded_cap else ['0','1'])
    if excluded_cap:
        from stages._imagery_features import routes
        image=accepted[0]
        a,_=routes(image['b10'],image['b20'],image['meta']['dn_offset'],image['meta']['dn_scale'])
        sampled=a[:,::16,::16].reshape(6,-1).astype('float64')
        np.testing.assert_allclose(bins['band_stats']['mean'],sampled.mean(1))
        assert bins['training_ids_sha256']==hashlib.sha256(table.window_id.iloc[0].encode()).hexdigest()
    first=ctx.p('features','appearance_rich.npy').read_bytes()
    im.features_final(ctx)
    assert ctx.p('features','appearance_rich.npy').read_bytes()==first


def test_pilot_cannot_use_test_split(cfg):
    ctx=make_ctx(cfg,VOCAB,'acquire_pilot')
    write_parquet(ctx.p('select','windows.parquet'),[dict(window_id='w0',pilot=True,split='test',priority=0)])
    with pytest.raises(StageFailed,match='validation'):im.acquisition_plan(ctx)


def test_stac_pagination_and_season(cfg):
    class Response:
        def __init__(self,data):self.data=data
        def raise_for_status(self):pass
        def json(self):return self.data
    class Session:
        def __init__(self):self.bodies=[]
        def post(self,url,json,timeout):
            self.bodies.append(json)
            if len(self.bodies)==1:
                return Response({'features':[{'id':'b','properties':{'eo:cloud_cover':2,'s2:snow_ice_percentage':0,'datetime':'2025-01-01'}}],
                     'links':[{'rel':'next','href':'https://planetarycomputer.microsoft.com/api/stac/v1/search','method':'POST','body':{'token':'next'},'merge':True}]})
            return Response({'features':[{'id':'a','properties':{'eo:cloud_cover':1,'s2:snow_ice_percentage':0,'datetime':'2025-01-01'}}]})
    s=Session();out=search(s,'planetary_computer',[0,0,1,1],'2025-01-01/2025-12-31',cfg)
    assert [x['id'] for x in out]==['a','b']
    assert s.bodies[1]['token']=='next' and s.bodies[1]['collections']==['sentinel-2-l2a']
    assert season(-30,cfg)[0]=='south'
    assert '2024-11-01' in season(-30,cfg)[1]
    assert season(0,cfg)[0]=='tropics'


def test_cell_seasons_and_prefetch_plan_and_cached_catalogue(cfg,tmp_path):
    from stages import imagery
    from stages._osm_geometry import connect
    from stages._osm_buildareas import SCHEMA
    cfg['lattice']['deg_cell']=1.0
    assert imagery.cell_seasons('x0180y0140',cfg)==['north']          # 0E, 50N
    assert imagery.cell_seasons('x0180y0113',cfg)==['north','tropics'] # 23N..24N straddles 23.5
    assert imagery.cell_seasons('x0180y0060',cfg)==['south']
    run=tmp_path/'run';(run/'osm_areas').mkdir(parents=True);shared=tmp_path/'shared'
    db=connect(run/'osm_areas/areas.sqlite');db.executescript(SCHEMA)
    db.executemany('INSERT INTO cells VALUES (?)',[('x0180y0140',),('x0180y0113',),('x0180y0060',)]);db.commit();db.close()
    class Sess:
        calls=0
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def post(self,url,json,timeout):
            Sess.calls+=1
            class R:
                status_code=200
                headers={}
                def raise_for_status(self):pass
                def json(self):return {'features':[],'links':[]}
            return R()
    from mc.core import Store
    ctx=SimpleNamespace(cfg=cfg,shared=shared,store=Store(run),p=lambda *x:run.joinpath(*x),units=lambda:units,requests_session=lambda **kw:Sess())
    cfg['imagery']['stac_cells_per_unit']=2
    units=imagery.prefetch_plan(ctx)
    assert sum(len(u['pairs']) for u in units)==4 and len(units)==2
    out=imagery.prefetch_run(ctx,units[0],SimpleNamespace(maybe_stop=lambda:None))
    assert len(out['outputs'])==2 and all(p.exists() for p in out['outputs'])
    doc=json.loads(out['outputs'][0].read_text())
    assert doc['bbox']==[0.0,-30.0,1.0,-29.0] and set(doc['sources'])=={'planetary_computer','earth_search'}   # cells sort south-first
    calls=Sess.calls
    imagery.prefetch_run(ctx,units[0],SimpleNamespace(maybe_stop=lambda:None))   # cached: no new network calls
    assert Sess.calls==calls
    assert imagery.prefetch_final(ctx)['catalogues']==4


def test_cached_window_repair_amends_metadata_without_touching_pixels(cog_world,cfg,tmp_path):
    row,item,signer,_=cog_world
    fn=lambda:read_scene(item,row,'planetary_computer',cfg,signer)
    a=cached_window(tmp_path/'cache',row['window_id'],fn)
    assert 'source_pair_status' not in a['meta']
    def repair(result):
        result['meta']['source_pair_status']='pass';return result,True
    b=cached_window(tmp_path/'cache',row['window_id'],lambda:pytest.fail('cache missed'),repair)
    assert b['meta']['source_pair_status']=='pass'
    c=cached_window(tmp_path/'cache',row['window_id'],lambda:pytest.fail('cache missed'))
    assert c['meta']['source_pair_status']=='pass' and np.array_equal(a['b10'],c['b10'])
    meta=json.loads((tmp_path/'cache'/(row['window_id']+'.json')).read_text())
    assert meta['array_sha256']==sha256_file(tmp_path/'cache'/(row['window_id']+'.npz'))


def test_signer_shares_one_cached_token_and_waits_out_429(tmp_path):
    import threading
    from stages._imagery_acquire import Signer, TechnicalFailure
    calls=[];slept=[]
    class R:
        def __init__(self,code,retry=None):self.status_code=code;self.headers={'Retry-After':retry} if retry else {}
        def json(self):return {'token':'t%d'%len(calls),'msft:expiry':'2099-01-01T00:00:00Z'}
        def raise_for_status(self):pass
    class S:
        def __init__(self,codes):self.codes=list(codes)
        def get(self,url,timeout):calls.append(url);return R(self.codes.pop(0) if self.codes else 200,'3')
    cache=tmp_path/'tok.json'
    a=Signer(S([429,429,200]),cache_path=cache,sleep=slept.append)
    out=[];ts=[threading.Thread(target=lambda:out.append(a.sign('https://x/y.tif','planetary_computer'))) for _ in range(16)]
    [t.start() for t in ts];[t.join() for t in ts]
    assert len(calls)==3 and len(slept)==2 and all(5<=w<=11.5 for w in slept)       # two 429s waited out (5 s then 10 s back-off), one token for 16 threads
    assert all(o.endswith('?t3') for o in out)
    b=Signer(S([200]),cache_path=cache,sleep=slept.append)
    assert b.sign('https://x/z.tif','planetary_computer').endswith('?t3') and len(calls)==3   # second worker adopts the cached token
    assert b.sign('https://x/z.tif','earth_search')=='https://x/z.tif'
    c=Signer(S([429]*20),cache_path=tmp_path/'other.json',sleep=slept.append,attempts=3)
    with pytest.raises(TechnicalFailure):c.sign('https://x/y.tif','planetary_computer')
