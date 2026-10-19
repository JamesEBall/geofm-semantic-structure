from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mc.context import make_ctx
from mc.core import load_config, Checkpoint, sha256_json, sha256_file, atomic_write_json
from stages import _osm_pipeline as pipe
from stages import _osm_geometry as geo
from stages._osm_buildareas import build

CODE=Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path):
    c=load_config(CODE)
    c['root']=str(tmp_path/'root');c['_profile']='test'
    c['lattice']['deg_cell']=.05
    c['selection'].update(min_candidate_cells=1,cap_per_class=30,candidates_per_class=100,
        candidates_per_class_per_deg=6,max_per_class_per_deg=4,buffer_km=0,budget=90,
        split_fracs={'train':.5,'val':.25,'test':.25},admission={'train':1,'val':0,'test':0,'test_cells':0,'cells':1})
    c['imagery']['pilot_windows']=6
    c['osm_pipeline']['labels_chunk']=5
    return c


def rectangle_osm(path,extra=()):
    # Several distinct natural classes and an explicitly excluded private area.
    pieces=['<osm version="0.6" generator="unit-test">']; ways=[];nid=1
    for i,(tag,x,y) in enumerate([('wood',.01,50.01),('water',.21,50.01),('grassland',.41,50.01),*extra]):
        coords=[(x,y),(x+.12,y),(x+.12,y+.12),(x,y+.12),(x,y)]
        ids=[]
        for lon,lat in coords[:-1]:
            pieces.append(f'<node id="{nid}" version="1" lon="{lon}" lat="{lat}"/>');ids.append(nid);nid+=1
        ids.append(ids[0]);nds=''.join(f'<nd ref="{n}"/>' for n in ids)
        ways.append(f'<way id="{i+1}" version="2" user="private-name" uid="23" changeset="45">{nds}<tag k="natural" v="{tag}"/><tag k="name" v="private-name"/></way>')
    path.write_text(''.join(pieces+ways+['</osm>']))
    return path


def context(c,stage,tmp):
    ctx=make_ctx(c,CODE,stage=stage,threads=1,tmp=tmp)
    ck=Checkpoint(ctx.store,stage,'test')
    return ctx,ck


def test_taxonomy_alias_and_exclusion(cfg):
    t=cfg['osm_pipeline']['taxonomy']
    assert geo.classification({'landuse':'reservoir'},t)=='natural=water|water=reservoir'
    assert geo.classification({'landuse':'forest','leaf_type':'needleleaved'},t)=='landuse=forest|leaf_type=needleleaved'
    assert geo.classification({'landuse':'forest','boundary':'protected_area'},t) is None
    assert geo.classification({'leisure':'nature_reserve','natural':'wood'},t) is None
    assert geo.classification({'place':'town'},t) is None
    assert geo.classification({'natural':'wood','user':'private'},t)=='natural=wood'


def test_coverage_union_not_sum():
    import shapely
    from shapely.geometry import box
    geoms=[box(0,0,7,10),box(3,0,10,10)]
    index=(shapely.STRtree(geoms),geoms,['tree','tree'],[None,None])
    assert geo.coverage(index,box(0,0,10,10))=={'tree':1.}


def test_dateline_separation_and_buffer(cfg):
    bins=geo.SphericalBins(5)
    a={'lon':179.99,'lat':0};b={'lon':-179.99,'lat':0}
    bins.add(a);assert list(bins.neighbours(b))==[a]
    base={'lat':0,'deg_cell':'x0000y0000','target_class_idx':0}
    data=[{**base,'candidate_id':'a','lon':0,'split':'train'},
          {**base,'candidate_id':'b','lon':.01,'split':'test'},
          {**base,'candidate_id':'c','lon':.1,'split':'val'}]
    cfg['selection']['buffer_km']=5
    selected=geo.select_rows(data,cfg,1)
    assert len(selected)==2
    assert not {'a','b'} <= {r['candidate_id'] for r in selected}
    assert all(not r['pilot'] or r['split']=='val' for r in selected)


def test_antimeridian_polygon():
    from shapely.geometry import Polygon
    g=Polygon([(179,0),(-179,0),(-179,1),(179,1)])
    parts=geo.antimeridian_parts(g)
    assert len(parts)==2
    assert sum(p.area for p in parts)==pytest.approx(2)


def test_streamed_area_store_and_resume(cfg,tmp_path):
    src=rectangle_osm(tmp_path/'input.osm');db=tmp_path/'areas.sqlite'
    settings={**cfg['osm_pipeline'],'deg_cell':cfg['lattice']['deg_cell'],'min_polygon_m2':0}
    a=build(src,db,settings,'same',tmp_path/'node-index')
    b=build(src,db,settings,'same',tmp_path/'node-index')
    assert a==b and a['accepted_parts']==3
    with geo.connect(db,True) as conn:
        assert conn.execute('SELECT COUNT(*) FROM areas').fetchone()[0]==3
        assert 'user' not in [r[1] for r in conn.execute('PRAGMA table_info(areas)')]
        assert conn.execute('SELECT COUNT(*) FROM cells').fetchone()[0]>3
    with pytest.raises(ValueError,match='differs'):
        build(src,db,settings,'changed',tmp_path/'node-index')


def shapefile_zip(path,fields,values):
    import shapefile,zipfile
    stem=path.with_suffix('')
    w=shapefile.Writer(str(stem),shapeType=shapefile.POLYGON)
    for name in fields:w.field(name,'C',80)
    # Clockwise polygon over all synthetic scenes.
    w.poly([[(-1,49),(-1,52),(2,52),(2,49),(-1,49)]])
    w.record(*map(str,values));w.close()
    with zipfile.ZipFile(path,'w') as z:
        for suffix in ('shp','dbf','shx'):z.write(stem.with_suffix('.'+suffix),arcname=stem.name+'.'+suffix)


def test_alias_duplicates_do_not_consume_reservoir_slots(cfg,tmp_path,monkeypatch):
    ctx,ck=context(cfg,'select',tmp_path/'tmp')
    cfg['selection']['candidates_per_class']=3
    classes=[{'class_id':'canonical','class_idx':0}]
    aliases={'a':'canonical','b':'canonical'}
    records=[{'candidate_id':f'c{i}', 'class_id':label, 'multi_classes':['a','b'],
              'multi_cov':[.8,.8]} for i in range(12) for label in ('a','b')]
    monkeypatch.setattr(pipe,'lattice_inputs',lambda c:[tmp_path/'cell.parquet'])
    monkeypatch.setattr(pipe,'rows',lambda p:copy.deepcopy(records))
    result=pipe.selection_candidates(ctx,classes,aliases,ck)
    expected=sorted(((-pipe.seed_for(cfg,'select-cap',f'c{i}','canonical'),f'c{i}') for i in range(12)),reverse=True)[:3]
    assert len(result)==3
    assert {r['candidate_id'] for r in result}=={r[1] for r in expected}
    assert all(r['multi_idx']==[0] and r['multi_cov']==[1.] for r in result)


def test_boundary_index_ignores_unused_legacy_text(tmp_path):
    import shapefile
    import zipfile
    stem=tmp_path/'legacy'
    with shapefile.Writer(str(stem),encoding='cp1252') as writer:
        writer.field('ECO_ID','N');writer.field('BIOME_NUM','N');writer.field('ECO_NAME','C')
        writer.poly([[(-1,49),(-1,52),(2,52),(2,49),(-1,49)]])
        writer.record(12,4,'Alto Paraná Atlantic forests')
    archive=tmp_path/'legacy.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for ext in ['shp','shx','dbf']:
            z.write(stem.with_suffix('.'+ext),arcname='legacy.'+ext)
    index=pipe._boundary_index(archive,tmp_path/'extract',['ECO_ID','BIOME_NUM'])
    assert pipe.boundary_value(index,0,50,['ECO_ID','BIOME_NUM'])=={'ECO_ID':12,'BIOME_NUM':4}
    assert index[2]==[{'ECO_ID':12,'BIOME_NUM':4}]
    # Required text still uses strict decoding; corrupt values cannot be dropped.
    with pytest.raises(Exception,match='decode'):
        pipe._boundary_index(archive,tmp_path/'required-text',['ECO_NAME'])


def test_synthetic_end_to_end(cfg,tmp_path):
    # Real pyosmium -> spatial store -> lattice -> vocab -> selected data -> exact labels
    # -> post-QA admission. Network acquisition is separately tested with mocked streams.
    source=rectangle_osm(tmp_path/'input.osm')
    ctx,ck=context(cfg,'osm_areas',tmp_path/'tmp')
    settings={**cfg['osm_pipeline'],'deg_cell':cfg['lattice']['deg_cell'],'min_polygon_m2':0}
    build(source,ctx.stage_dir()/'areas.sqlite',settings,'synthetic',ctx.tmp/'node-index')
    ctx.stage='osm_lattice'
    units=pipe.lattice_plan(ctx)
    ctx.store.write_units('osm_lattice', units)
    for unit in units:
        ck=Checkpoint(ctx.store,ctx.stage,unit['id']);pipe.lattice_run(ctx,unit,ck)
    # Interrupted and unregistered files must never enter inventory or selection.
    (ctx.stage_dir()/'.x0000y0000.tmp.dead-worker.parquet').write_bytes(b'')
    (ctx.stage_dir()/'unregistered.parquet').write_bytes(b'')
    ctx.stage='inventory';pipe.inventory_run(ctx,{},ck)
    ctx.stage='vocab';pipe.vocab_run(ctx,{},ck)
    classes=pipe.rows(ctx.stage_dir()/'classes.parquet');assert len(classes)==3
    ctx.stage='biome_fetch'
    shapefile_zip(ctx.stage_dir()/'ecoregions.zip',['ECO_ID','BIOME_NUM'],[12,4])
    shapefile_zip(ctx.stage_dir()/'countries.zip',['CONTINENT'],['Europe'])
    ctx.stage='select';pipe.select_run(ctx,{},ck)
    selected=pipe.rows(ctx.stage_dir()/'windows.parquet')
    assert selected and all(r['continent']=='Europe' for r in selected)
    assert all(r['biome_id']==4 for r in selected)
    assert all(not r['pilot'] or r['split']=='val' for r in selected)
    # Simulate successful imagery grid snapping without calling external services.
    from pyproj import Transformer
    for i,r in enumerate(selected):
        x,y=Transformer.from_crs(4326,r['utm_epsg'],always_xy=True).transform(r['lon'],r['lat'])
        r.update(widx=i,epsg=r['utm_epsg'],x0=round((x-1280)/20)*20,y0=round((y+1280)/20)*20)
    ctx.stage='assemble';pipe.parquet(ctx.stage_dir()/'windows_qa.parquet',selected)
    ctx.stage='labels_exact'
    label_units=pipe.labels_plan(ctx)
    for unit in label_units:
        ctx.unit=unit['id'];ck=Checkpoint(ctx.store,ctx.stage,unit['id']);pipe.labels_run(ctx,unit,ck)
    pipe.labels_finalize(ctx)
    label=pipe.rows(ctx.stage_dir()/'labels.parquet')
    assert len(label)==len(selected)
    assert all(r['labels_idx']==sorted(r['labels_idx']) for r in label)
    assert all(0<=r['purity']<=1 for r in label)
    # Re-running a completed chunk uses content-verified per-window checkpoints.
    pipe.labels_run(ctx,label_units[0],Checkpoint(ctx.store,ctx.stage,label_units[0]['id']))
    ctx.stage='admit';pipe.admit_run(ctx,{},ck)
    final=pipe.rows(ctx.stage_dir()/'classes_final.parquet')
    assert any(r['admitted'] for r in final)
    mask=pipe.rows(ctx.stage_dir()/'windows_final.parquet')
    assert [r['widx'] for r in mask]==list(range(len(selected)))


def test_fractional_pixel_mask():
    import shapely,numpy as np
    from shapely.geometry import box
    g=box(0,0,5,10)
    index=(shapely.STRtree([g]),[g],['water'],[None])
    mask=pipe.sparse_masks(index,{'water':0},0,10,1)
    assert len(mask)==1
    assert mask[0][1].tolist()==[0]
    assert mask[0][2].tolist()==pytest.approx([.5])


def test_admission_is_after_cap(cfg):
    cfg['selection']['cap_per_class']=2
    cfg['selection']['admission']={'train':3,'val':0,'test':0,'test_cells':0,'cells':1}
    windows=[dict(widx=i,window_id=f'w{i}',target_class_idx=0,split='train',deg_cell=f'c{i}') for i in range(5)]
    labels=[dict(widx=i,window_id=f'w{i}',target_eligible=True,labels_idx=[0],reason='') for i in range(5)]
    kept={r['widx'] for r in pipe.retention(windows,labels,cfg) if r['retained']}
    assert len(kept)==2
    out=pipe.admission_counts([dict(class_idx=0,class_id='natural=wood')],windows,[r for r in labels if r['widx'] in kept],cfg)
    assert out[0]['n_train']==2 and not out[0]['admitted']


def test_regions_tile_globe_once(cfg):
    specs=pipe.regions(cfg)
    assert len(specs)==24 and len({r['id'] for r in specs})==24 and [r['index'] for r in specs]==list(range(24))
    assert sum((b[2]-b[0])*(b[3]-b[1]) for b in (r['bbox'] for r in specs))==360*180
    assert all(pipe.REGION_ID_SPAN*(r['index']+1)>0 for r in specs)
    cfg['osm_pipeline']['region_lon_deg']=50
    with pytest.raises(Exception,match='divide'):
        pipe.regions(cfg)


def test_regional_build_merge_dedupes_and_preserves_ids(cfg,tmp_path):
    settings={**cfg['osm_pipeline'],'deg_cell':cfg['lattice']['deg_cell'],'min_polygon_m2':0}
    a=rectangle_osm(tmp_path/'a.osm')
    b=rectangle_osm(tmp_path/'b.osm',extra=[('scrub',.61,50.01)])   # same three polygons plus one more
    empty=tmp_path/'empty.osm'; empty.write_text('<osm version="0.6"><node id="1" version="1" lon="1" lat="1"/></osm>')
    ra=build(a,tmp_path/'ra.sqlite',settings,'fa',tmp_path/'na',id_offset=1*pipe.REGION_ID_SPAN,allow_empty=True)
    rb=build(b,tmp_path/'rb.sqlite',settings,'fb',tmp_path/'nb',id_offset=2*pipe.REGION_ID_SPAN,allow_empty=True)
    re=build(empty,tmp_path/'re.sqlite',settings,'fe',tmp_path/'ne',id_offset=3*pipe.REGION_ID_SPAN,allow_empty=True)
    assert (ra['accepted_parts'],rb['accepted_parts'],re['accepted_parts'])==(3,4,0)
    with geo.connect(tmp_path/'ra.sqlite',True) as c:
        ids=[r[0] for r in c.execute('SELECT id FROM areas ORDER BY id')]
        assert ids==[pipe.REGION_ID_SPAN+i for i in range(3)]
    with pytest.raises(ValueError,match='offset'):
        build(a,tmp_path/'ra.sqlite',settings,'fa',tmp_path/'na',id_offset=7,allow_empty=True)
    merged=tmp_path/'areas.sqlite'
    counts=pipe.merge_area_databases(merged,[tmp_path/'ra.sqlite',tmp_path/'rb.sqlite',tmp_path/'re.sqlite'],'merged')
    assert counts['accepted_parts']==4 and counts['duplicates_across_regions']==3 and counts['regions']==3
    with geo.connect(merged,True) as c:
        assert c.execute('SELECT COUNT(*) FROM areas').fetchone()[0]==4
        assert c.execute('SELECT COUNT(*) FROM bounds').fetchone()[0]==4
        assert dict(c.execute('SELECT key,value FROM progress'))['complete']=='true'
        kept=sorted(r[0] for r in c.execute('SELECT id FROM areas'))
        assert kept[:3]==[pipe.REGION_ID_SPAN+i for i in range(3)] and kept[3]>=2*pipe.REGION_ID_SPAN
        assert c.execute('SELECT COUNT(*) FROM cells').fetchone()[0]>3
    # The merged store serves the same bbox queries as a single-pass build.
    idx=geo.projected_index(geo.connect(merged,True),geo.utm_epsg(.3,50.07),(-1,49,2,52))
    assert sorted(set(idx[2]))==['natural=grassland','natural=scrub','natural=water','natural=wood']


def test_areas_stage_plan_run_finalize_from_regions(cfg,tmp_path):
    ctx,ck=context(cfg,'osm_split',tmp_path/'tmp')
    a=rectangle_osm(tmp_path/'a.osm'); b=rectangle_osm(tmp_path/'b.osm',extra=[('scrub',.61,50.01)])
    regions={'r00-0':{'path':str(a),'sha256':sha256_file(a),'bytes':a.stat().st_size,'bbox':[-180,-90,0,90],'index':0},
             'r01-0':{'path':str(b),'sha256':sha256_file(b),'bytes':b.stat().st_size,'bbox':[0,-90,180,90],'index':1}}
    atomic_write_json(ctx.stage_dir()/'regions.json',{'source_sha256':'x','strategy':'smart','regions':regions})
    ctx.stage='osm_areas'
    units=pipe.areas_plan(ctx); assert [u['id'] for u in units]==['r00-0','r01-0']
    ctx.store.write_units('osm_areas',units)
    for u in units:
        out=pipe.areas_run(ctx,u,Checkpoint(ctx.store,'osm_areas',u['id']))
        assert out['metrics']['accepted_parts'] in (3,4)
    again=pipe.areas_run(ctx,units[0],Checkpoint(ctx.store,'osm_areas','r00-0'))   # idempotent rerun
    assert again['metrics']['accepted_parts']==3
    report=pipe.areas_finalize(ctx)
    assert report['accepted_parts']==4 and report['duplicates_across_regions']==3
    assert report==pipe.areas_finalize(ctx)   # finalize is idempotent on the merged store
    ctx.stage='osm_lattice'
    assert len(pipe.lattice_plan(ctx))==report['cells']
    with pytest.raises(Exception,match='missing or changed'):
        regions['r00-0']['sha256']='0'*64
        atomic_write_json(ctx.p('osm_split','regions.json'),{'source_sha256':'x','strategy':'smart','regions':regions})
        pipe.areas_run(ctx,units[0],Checkpoint(ctx.store,'osm_areas','r00-0'))


def test_merge_resumes_from_recorded_regions(cfg,tmp_path):
    settings={**cfg['osm_pipeline'],'deg_cell':cfg['lattice']['deg_cell'],'min_polygon_m2':0}
    a=rectangle_osm(tmp_path/'a.osm'); b=rectangle_osm(tmp_path/'b.osm',extra=[('scrub',.61,50.01)])
    build(a,tmp_path/'ra.sqlite',settings,'fa',tmp_path/'na',id_offset=pipe.REGION_ID_SPAN,allow_empty=True)
    build(b,tmp_path/'rb.sqlite',settings,'fb',tmp_path/'nb',id_offset=2*pipe.REGION_ID_SPAN,allow_empty=True)
    merged=tmp_path/'areas.sqlite'
    # First attempt dies after region a: emulate by merging a alone into the working file, then resuming.
    with pytest.raises(Exception):
        pipe.merge_area_databases(merged,[tmp_path/'ra.sqlite',tmp_path/'missing.sqlite'],'fp')
    working=merged.with_suffix('.merging.sqlite'); assert working.exists() and not merged.exists()
    with geo.connect(working,True) as c:
        import json as _j
        assert _j.loads(dict(c.execute('SELECT key,value FROM progress'))['merged_regions'])==['ra.sqlite']
    counts=pipe.merge_area_databases(merged,[tmp_path/'ra.sqlite',tmp_path/'rb.sqlite'],'fp')
    assert counts['accepted_parts']==4 and counts['regions']==2 and merged.exists() and not working.exists()
    with pytest.raises(Exception,match='fingerprint'):
        (tmp_path/'x.sqlite').unlink(missing_ok=True)
        pipe.merge_area_databases(tmp_path/'x.sqlite',[tmp_path/'ra.sqlite'],'fp')  # fresh dest: fine
        import shutil; shutil.copy(tmp_path/'x.merging.sqlite' if (tmp_path/'x.merging.sqlite').exists() else tmp_path/'x.sqlite', tmp_path/'y.merging.sqlite')
        pipe.merge_area_databases(tmp_path/'y.sqlite',[tmp_path/'ra.sqlite'],'other-fp')
    with geo.connect(merged,True) as c:
        assert c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='areas_dedupe'").fetchone()[0]==1
        assert c.execute('SELECT COUNT(*) FROM areas WHERE wkb_sha IS NULL').fetchone()[0]==0


def test_projected_index_repairs_or_skips_invalid_projected_geometry(tmp_path):
    import shapely, sqlite3, hashlib
    from stages._osm_buildareas import SCHEMA
    db=geo.connect(tmp_path/'a.sqlite'); db.executescript(SCHEMA)
    good=shapely.Polygon([(0,50),(0.1,50),(0.1,50.1),(0,50.1)])
    bowtie=shapely.Polygon([(0.2,50),(0.3,50.1),(0.3,50),(0.2,50.1)])          # self-intersecting, repairable
    def ins(i,g,cls):
        w=shapely.to_wkb(g); db.execute('INSERT INTO areas VALUES (?,?,?,?,?,?,?)',(i,cls,w,'relation',i,1,hashlib.sha1(w).digest()))
        x0,y0,x1,y1=g.bounds; db.execute('INSERT INTO bounds VALUES (?,?,?,?,?)',(i,x0,x1,y0,y1))
    ins(1,good,'natural=wood'); ins(2,bowtie,'natural=water'); db.commit()
    report={}
    tree,geoms,ids,prov=geo.projected_index(db,geo.utm_epsg(0.15,50.05),(-1,49,1,51),report)
    assert len(geoms)==2 and all(g.is_valid for g in geoms) and report=={'repaired':1}
    # A geometry that collapses entirely is skipped and named, not fatal.
    ins(3,shapely.Polygon([(0.5,50),(0.5,50),(0.5,50),(0.5,50)]),'natural=scrub'); db.commit()
    report={}
    tree,geoms,ids,prov=geo.projected_index(db,geo.utm_epsg(0.15,50.05),(-1,49,1,51),report)
    assert len(geoms)==2 and report['skipped']==1 and report['skipped_ids']==['relation3']
    assert len(geo.projected_index(db,geo.utm_epsg(0.15,50.05),(-1,49,1,51))[1])==2    # report optional
