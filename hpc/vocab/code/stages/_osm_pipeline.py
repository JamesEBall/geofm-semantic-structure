"""OSM pipeline stages. Large inputs stream to disk; cell work is independently resumable."""
from __future__ import annotations
import collections
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file, sha256_json
from mc.registry import StageFailed
from mc.util import seed_for, lfs_stripe
from ._osm_geometry import (classification, class_fields, cell_id, cell_bounds, split_for,
    utm_epsg, connect, projected_index, coverage, centre_box, exact_window_index, select_rows)


def knobs(ctx):
    return ctx.cfg['osm_pipeline']


def one(ctx):
    return [{'id':'all'}]


def parquet(path, rows, schema=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not rows and schema is None:
        raise StageFailed(f'No rows for {path}; refusing schema-less empty artefact')
    with atomic_path(path) as tmp:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression='zstd', version='2.6')


def rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def monitor(processes, ckpt):
    try:
        while any(p.poll() is None for p in processes):
            if ckpt.due():
                ckpt.save('processes', {'active':len([p for p in processes if p.poll() is None])})
            ckpt.maybe_stop()
            time.sleep(.5)
        if any(p.returncode for p in processes):
            raise RuntimeError('OSM subprocess failed: '+str([p.returncode for p in processes]))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()


def filter_run(ctx, unit, ckpt):
    report = read_json(ctx.p('osm_fetch','source.json'))
    if not report or not report.get('sha256'):
        raise StageFailed('Missing verified immutable OSM source')
    source = Path(report['path'])
    binary = os.environ.get('OSMIUM_BIN') or shutil.which('osmium')
    if not binary:
        raise StageFailed('OSMIUM_BIN is required for streaming filter')
    dest = ctx.stage_dir() / 'filtered.osm.pbf'; log=ctx.stage_dir()/'filter.log'
    env = dict(os.environ, OSMIUM_POOL_THREADS=str(ctx.threads))
    # Privacy fields are removed in the pipe before any derived PBF is stored.
    with atomic_path(dest) as tmp, open(log,'wb') as err:
        a = subprocess.Popen([binary,'tags-filter',str(source),
            *['a/'+key for key in knobs(ctx)['taxonomy']['primary_keys']],'-f','pbf'],
            stdout=subprocess.PIPE,stderr=err,env=env)
        b = subprocess.Popen([binary,'cat','-F','pbf','-',
            '-c','user','-c','uid','-c','changeset','-c','timestamp',
            '-f','pbf','-o',str(tmp),'--fsync'],stdin=a.stdout,stderr=err,env=env)
        a.stdout.close()
        monitor([a,b],ckpt)
    meta=ctx.stage_dir()/'source.json'
    atomic_write_json(meta,{'path':str(dest),'sha256':sha256_file(dest),'source_sha256':report['sha256'],
        'taxonomy_sha256':sha256_json(knobs(ctx)['taxonomy']),'licence':'ODbL-1.0',
        'privacy_fields_removed':['user','uid','changeset','timestamp']})
    return {'outputs':[dest,meta,log],'metrics':{'bytes':dest.stat().st_size}}


REGION_ID_SPAN = 2**32   # each regional database owns [index*SPAN, (index+1)*SPAN) area ids


def regions(cfg):
    """Fixed longitude/latitude bands that tile the globe once; frozen by config.d knobs."""
    k = cfg['osm_pipeline']
    lon = float(k.get('region_lon_deg', 30)); lat = float(k.get('region_lat_deg', 90))
    nx, ny = round(360/lon), round(180/lat)
    if abs(nx*lon-360) > 1e-9 or abs(ny*lat-180) > 1e-9:
        raise StageFailed('Region steps must divide 360 and 180 degrees exactly')
    out = []
    for iy in range(ny):
        for ix in range(nx):
            w, s_ = -180+ix*lon, -90+iy*lat
            out.append({'id': f'r{ix:02d}-{iy:02d}', 'index': iy*nx+ix,
                        'bbox': [w, s_, min(180., w+lon), min(90., s_+lat)]})
    return out


def split_run(ctx, unit, ckpt):
    """osmium extract: one pass set over the filtered planet, one PBF per region.

    The smart strategy keeps complete ways and multipolygon relations that cross a region edge,
    so boundary polygons appear whole in both neighbours; the merge de-duplicates them.
    """
    report = read_json(ctx.p('osm_filter','source.json'))
    if not report or not report.get('sha256'):
        raise StageFailed('Missing verified filtered OSM source')
    binary = os.environ.get('OSMIUM_BIN') or shutil.which('osmium')
    if not binary:
        raise StageFailed('OSMIUM_BIN is required for regional extraction')
    final = ctx.stage_dir()/'regions'; work = ctx.stage_dir()/'regions.tmp'
    if final.exists():
        shutil.rmtree(final)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    lfs_stripe(work)
    specs = regions(ctx.cfg)
    cfgfile = ctx.stage_dir()/'extracts.json'
    atomic_write_json(cfgfile, {'directory': str(work), 'extracts': [
        {'output': r['id']+'.osm.pbf', 'output_format': 'pbf', 'bbox': r['bbox']} for r in specs]})
    log = ctx.stage_dir()/'split.log'
    env = dict(os.environ, OSMIUM_POOL_THREADS=str(ctx.threads))
    with open(log,'ab') as err:
        p = subprocess.Popen([binary,'extract','-c',str(cfgfile),'-s',knobs(ctx).get('extract_strategy','smart'),
                              '--overwrite','--fsync',report['path']],stdout=err,stderr=err,env=env)
        monitor([p],ckpt)
    files = {}
    for r in specs:
        path = work/(r['id']+'.osm.pbf')
        if not path.is_file():
            raise StageFailed('osmium extract produced no file for region '+r['id'])
        files[r['id']] = {'path': str(final/path.name), 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                          'bbox': r['bbox'], 'index': r['index']}
    os.replace(work, final)
    meta = ctx.stage_dir()/'regions.json'
    atomic_write_json(meta, {'source_sha256': report['sha256'], 'strategy': knobs(ctx).get('extract_strategy','smart'),
                             'regions': files, 'licence': 'ODbL-1.0'})
    outputs = [final/(r['id']+'.osm.pbf') for r in specs] + [meta, cfgfile, log]
    return {'outputs': outputs, 'metrics': {'regions': len(files), 'bytes': sum(f['bytes'] for f in files.values())}}


def areas_plan(ctx):
    meta = read_json(ctx.p('osm_split','regions.json'))
    if not meta:
        raise StageFailed('Missing regional extracts')
    return [{'id': rid, 'region': rid, 'index': int(info['index'])}
            for rid, info in sorted(meta['regions'].items(), key=lambda kv: kv[1]['index'])]


def area_settings(ctx):
    return {**knobs(ctx),'deg_cell':ctx.cfg['lattice']['deg_cell'],'min_polygon_m2':ctx.cfg['labels']['min_polygon_m2']}


def areas_run(ctx, unit, ckpt):
    """One region: a resumable pyosmium scan into areas-<region>.sqlite with a private id range."""
    info = read_json(ctx.p('osm_split','regions.json'))['regions'][unit['region']]
    source = Path(info['path'])
    if not source.is_file() or sha256_file(source) != info['sha256']:
        raise StageFailed('Regional extract missing or changed: '+unit['region'])
    settings = area_settings(ctx)
    offset = (int(unit['index'])+1) * REGION_ID_SPAN
    fingerprint = sha256_json({'source': info['sha256'], 'settings': settings, 'id_offset': offset})
    dest = ctx.stage_dir()/f"areas-{unit['region']}.sqlite"; working = dest.with_suffix('.working.sqlite')
    if dest.exists():
        with connect(dest,True) as db:
            progress = dict(db.execute('SELECT key,value FROM progress'))
        if progress.get('fingerprint') != fingerprint or progress.get('complete') != 'true':
            raise StageFailed('Existing regional area database has incompatible provenance')
    else:
        args = ctx.stage_dir()/f"build-{unit['region']}.json"
        atomic_write_json(args, dict(source=str(source), database=str(working), settings=settings, fingerprint=fingerprint,
                                     node_index=str(ctx.tmp/f"osm-node-index-{unit['region']}.bin"),
                                     id_offset=offset, allow_empty=True))
        with open(ctx.stage_dir()/f"areas-{unit['region']}.log",'ab') as log:
            p = subprocess.Popen([sys.executable,'-m','stages._osm_buildareas',str(args)],stdout=log,stderr=log)
            monitor([p],ckpt)
        os.replace(working,dest)
    with connect(dest,True) as db:
        progress = dict(db.execute('SELECT key,value FROM progress'))
        counts = json.loads(progress['counts'])
        ncells = db.execute('SELECT COUNT(*) FROM cells').fetchone()[0]
    meta = dest.with_suffix('.json')
    atomic_write_json(meta,{'path':str(dest),'sha256':sha256_file(dest),'fingerprint':fingerprint,'region':unit['region'],
        'id_offset':offset,'cells':ncells,'counts':counts,'licence':'ODbL-1.0','attribution':'© OpenStreetMap contributors'})
    return {'outputs':[dest,meta], 'metrics':{'cells':ncells,**counts}}


def merge_area_databases(dest, sources, fingerprint):
    """Union regional databases into one, keeping ids and dropping polygons duplicated across region edges.

    Resumable: the merge accumulates in `<dest>.merging.sqlite`; each region is committed and listed in
    `progress.merged_regions`, so a killed allocation resumes with the regions already merged. The final
    file appears atomically. De-duplication keys on (osm_type, osm_id, sha1(wkb)), 20 bytes per row.
    """
    from ._osm_buildareas import SCHEMA
    dest = Path(dest); working = dest.with_suffix('.merging.sqlite')
    db = connect(working)
    try:
        db.executescript(SCHEMA)
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS areas_dedupe ON areas(osm_type, osm_id, wkb_sha)')
        progress = dict(db.execute('SELECT key,value FROM progress'))
        if progress and progress.get('fingerprint') != fingerprint:
            raise StageFailed('Partial merge has a different fingerprint; remove it before resuming')
        db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)', ('fingerprint', fingerprint)); db.commit()
        merged_regions = json.loads(progress.get('merged_regions', '[]'))
        totals = json.loads(progress.get('counts', '{}'))
        for path in sources:
            key = str(Path(path).name)
            if key in merged_regions:
                continue
            db.execute('ATTACH DATABASE ? AS r', (str(Path(path).absolute()),))
            rp = dict(db.execute('SELECT key,value FROM r.progress'))
            if rp.get('complete') != 'true':
                raise StageFailed('Regional area database incomplete: '+str(path))
            for k, v in json.loads(rp['counts']).items():
                totals[k] = totals.get(k, 0) + int(v)
            db.execute('INSERT OR IGNORE INTO main.areas(id,class_id,wkb,osm_type,osm_id,version,wkb_sha) '
                       'SELECT id,class_id,wkb,osm_type,osm_id,version,wkb_sha FROM r.areas ORDER BY id')
            db.execute('INSERT INTO main.bounds(id,minx,maxx,miny,maxy) SELECT b.id,b.minx,b.maxx,b.miny,b.maxy '
                       'FROM r.bounds b WHERE b.id IN (SELECT id FROM main.areas) '
                       'AND b.id NOT IN (SELECT id FROM main.bounds) ORDER BY b.id')
            db.execute('INSERT OR IGNORE INTO main.cells SELECT cell FROM r.cells')
            merged_regions.append(key)
            db.executemany('INSERT OR REPLACE INTO progress VALUES (?,?)',
                           [('merged_regions', json.dumps(merged_regions)), ('counts', json.dumps(totals, sort_keys=True))])
            db.commit()
            db.execute('DETACH DATABASE r')
        merged = db.execute('SELECT COUNT(*) FROM areas').fetchone()[0]
        if db.execute('SELECT COUNT(*) FROM bounds').fetchone()[0] != merged:
            raise StageFailed('Merged area and bounds row counts differ')
        if not merged:
            raise StageFailed('No valid visual polygons remain after merging regions')
        totals['duplicates_across_regions'] = totals.get('accepted_parts', 0) - merged
        totals['accepted_parts'] = merged
        totals['regions'] = len(merged_regions)
        db.execute('CREATE INDEX IF NOT EXISTS areas_class ON areas(class_id)')
        db.executemany('INSERT OR REPLACE INTO progress VALUES (?,?)',
                       [('counts', json.dumps(totals, sort_keys=True)), ('complete', 'true')])
        db.commit()
    finally:
        db.close()
    os.replace(working, dest)
    return totals


def areas_finalize(ctx):
    units = ctx.units()
    metas = [read_json(ctx.stage_dir()/f"areas-{u['region']}.sqlite".replace('.sqlite','.json')) for u in units]
    if any(m is None for m in metas):
        raise StageFailed('Missing regional area metadata')
    fingerprint = sha256_json({'regions': [m['fingerprint'] for m in metas], 'settings': area_settings(ctx)})
    dest = ctx.stage_dir()/'areas.sqlite'
    if dest.exists():
        with connect(dest,True) as db:
            progress = dict(db.execute('SELECT key,value FROM progress'))
        if progress.get('fingerprint') != fingerprint or progress.get('complete') != 'true':
            raise StageFailed('Existing merged area database has incompatible provenance')
        counts = json.loads(progress['counts'])
    else:
        counts = merge_area_databases(dest, [m['path'] for m in metas], fingerprint)
    with connect(dest,True) as db:
        ncells = db.execute('SELECT COUNT(*) FROM cells').fetchone()[0]
    atomic_write_json(ctx.stage_dir()/'areas.json',{'path':str(dest),'sha256':sha256_file(dest),'fingerprint':fingerprint,
        'cells':ncells,'counts':counts,'regions':[m['region'] for m in metas],
        'licence':'ODbL-1.0','attribution':'© OpenStreetMap contributors'})
    return {'cells':ncells,**counts}


def lattice_plan(ctx):
    with connect(ctx.p('osm_areas','areas.sqlite'),True) as db:
        return [{'id':c,'cell':c} for c, in db.execute('SELECT cell FROM cells ORDER BY cell')]


def candidate_schema():
    import pyarrow as pa
    return pa.schema([('candidate_id',pa.string()),('lon',pa.float64()),('lat',pa.float64()),
        ('utm_epsg',pa.int32()),('deg_cell',pa.string()),('split',pa.string()),
        ('class_id',pa.string()),('coverage',pa.float64()),('multi_classes',pa.list_(pa.string())),
        ('multi_cov',pa.list_(pa.float64())),('purity',pa.float64()),('mapping_density',pa.float64())])


def lattice_run(ctx, unit, ckpt):
    import heapq
    import shapely
    import numpy as np
    from pyproj import Transformer, Geod
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    step=ctx.cfg['lattice']['deg_cell']; size=ctx.cfg['lattice']['cell_m']
    bounds=cell_bounds(unit['cell'],step)
    lon0=(bounds[0]+bounds[2])/2; lat0=(bounds[1]+bounds[3])/2
    epsg=utm_epsg(lon0,lat0)
    forward=Transformer.from_crs(4326,epsg,always_xy=True)
    backward=Transformer.from_crs(epsg,4326,always_xy=True)
    x0,y0,x1,y1=forward.transform_bounds(*bounds,densify_pts=21)
    if not all(math.isfinite(x) for x in (x0,y0,x1,y1)):
        raise StageFailed('Non-finite projected lattice cell')
    x0=math.floor(x0/size)*size; y1=math.ceil(y1/size)*size
    width=max(1,math.ceil((x1-x0)/size)); height=max(1,math.ceil((y1-y0)/size))
    # Expand the query by the full candidate footprint, so class unions include neighbours.
    bbox=backward.transform_bounds(x0-size,y0-size,x1+size,y1+size,densify_pts=21)
    geometry_report={}
    with connect(ctx.p('osm_areas','areas.sqlite'),True) as db:
        index=projected_index(db,epsg,bbox,geometry_report)
        if len(index[1]) > knobs(ctx)['max_cell_polygons']:
            raise StageFailed('Cell exceeds declared geometry-memory bound; split engineering work units before resuming')
        radius=ctx.cfg['features']['mapping_density_radius_cells']*step
        n=db.execute('SELECT COUNT(*) FROM bounds WHERE maxx>=? AND minx<=? AND maxy>=? AND miny<=?',
            (bounds[0]-radius,bounds[2]+radius,bounds[1]-radius,bounds[3]+radius)).fetchone()[0]
    region=centre_box(lon0,lat0,step+2*radius)
    area_km2=abs(Geod(ellps='WGS84').geometry_area_perimeter(region)[0])/1e6
    density=n/max(1,area_km2)
    if index[1]:
        mask=rasterize(((g,1) for g in index[1]),out_shape=(height,width),transform=from_origin(x0,y1,size,size),all_touched=True,dtype='uint8')
        cells=np.argwhere(mask)
    else:
        cells=[]
    keep=int(ctx.cfg['selection']['candidates_per_class_per_deg']); heaps={}
    state=ckpt.load('scan',{'next':0,'heaps':{}})
    heaps={k:[(int(a),b,c) for a,b,c in v] for k,v in state['heaps'].items()}
    for heap in heaps.values():heapq.heapify(heap)
    for i in range(state['next'],len(cells)):
        row,col=map(int,cells[i]); x=x0+(col+.5)*size; y=y1-(row+.5)*size
        lon,lat=backward.transform(x,y)
        if cell_id(lon,lat,step) == unit['cell']:
            cov=coverage(index,centre_box(x,y,ctx.cfg['labels']['footprints_m']['r1']))
            multi={k:v for k,v in cov.items() if v>=ctx.cfg['labels']['multilabel_min']}
            candidate=f'{epsg}-{round(x)}-{round(y)}'
            for label,value in cov.items():
                if value < ctx.cfg['labels']['tau']:continue
                record=dict(candidate_id=candidate,lon=lon,lat=lat,utm_epsg=epsg,deg_cell=unit['cell'],
                    split=split_for(unit['cell'],ctx.cfg['selection']),class_id=label,coverage=value,
                    multi_classes=list(multi),multi_cov=list(multi.values()),purity=value/max(sum(cov.values()),1e-15),mapping_density=density)
                rank=seed_for(ctx.cfg,'osm_lattice',candidate,label)
                heap=heaps.setdefault(label,[]); item=(-rank,candidate,record)
                if len(heap)<keep:heapq.heappush(heap,item)
                elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
        if ckpt.due():ckpt.save('scan',{'next':i+1,'heaps':heaps})
        ckpt.maybe_stop()
    out=ctx.stage_dir()/(unit['id']+'.parquet')
    records=sorted((item[2] for heap in heaps.values() for item in heap),key=lambda r:(r['class_id'],r['candidate_id']))
    parquet(out,records,candidate_schema())
    return {'outputs':[out],'metrics':{'candidates':len(records),'grid_cells':len(cells),'polygons':len(index[1]),
                                       'projected_repaired':geometry_report.get('repaired',0),
                                       'projected_skipped':geometry_report.get('skipped',0),
                                       'projected_skipped_ids':geometry_report.get('skipped_ids',[])}}


def lattice_inputs(ctx):
    # The frozen plan names every completed cell. Directory globs also include
    # temporary Parquet files left by interrupted workers.
    units = ctx.store.read_units('osm_lattice')
    if units is None:
        raise StageFailed('Missing frozen lattice plan')
    return sorted(ctx.p('osm_lattice', unit['id']+'.parquet') for unit in units)


def inventory_run(ctx, unit, ckpt):
    # SQLite performs the cross-cell reduction without holding all candidate rows in RAM.
    out=ctx.stage_dir()/'inventory.parquet'; work=ctx.tmp/'inventory.sqlite'
    work.unlink(missing_ok=True)
    db=connect(work); db.execute('CREATE TABLE support(class_id TEXT, cell TEXT, split TEXT, n INTEGER)')
    for file in lattice_inputs(ctx):
        count=collections.Counter((r['class_id'],r['deg_cell'],r['split']) for r in rows(file))
        db.executemany('INSERT INTO support VALUES (?,?,?,?)',[(k,c,s,n) for (k,c,s),n in count.items()]); db.commit()
        ckpt.maybe_stop()
    result=[]
    for label,total,cells in db.execute('SELECT class_id,SUM(n),COUNT(DISTINCT cell) FROM support GROUP BY class_id ORDER BY class_id'):
        rec={**class_fields(label),'n_cand_windows':total,'n_cells':cells,'n_deg_cells':cells}
        for split in ('train','val','test'):
            rec['n_'+split]=db.execute('SELECT COALESCE(SUM(n),0) FROM support WHERE class_id=? AND split=?',(label,split)).fetchone()[0]
        result.append(rec)
    db.close(); work.unlink(missing_ok=True)
    if not result:raise StageFailed('OSM inventory has no eligible windows at frozen tau')
    parquet(out,result)
    return {'outputs':[out],'metrics':{'classes':len(result),'candidate_rows':sum(r['n_cand_windows'] for r in result)}}


def vocab_run(ctx, unit, ckpt):
    from ._text_phrases import render_name, template_table, normalise_name
    taxonomy=knobs(ctx)['taxonomy']; table=template_table(taxonomy)
    candidates=[r for r in rows(ctx.p('inventory','inventory.parquet')) if r['n_cells']>=ctx.cfg['selection']['min_candidate_cells']]
    if not candidates:raise StageFailed('No classes meet frozen minimum candidate-cell support')
    # Identical normalised names are aliases; keep stable lexical class IDs.
    groups=collections.defaultdict(list)
    for rec in candidates:
        rec['name']=render_name(rec,table); groups[normalise_name(rec['name'])].append(rec)
    aliases={}; classes=[]
    for group in groups.values():
        group.sort(key=lambda r:r['class_id']); keep=dict(group[0])
        for rec in group:aliases[rec['class_id']]=keep['class_id']
        # Support here is provisional. Exact, merged-label admission follows imagery QA.
        keep['alias_class_ids']=[r['class_id'] for r in group]
        keep['is_candidate']=True
        classes.append(keep)
    classes.sort(key=lambda r:r['class_id'])
    for i,rec in enumerate(classes):rec['class_idx']=i
    dest=ctx.stage_dir()/'classes.parquet'; tax=ctx.stage_dir()/'taxonomy.yaml'
    parquet(dest,classes)
    atomic_write_json(tax,{**taxonomy,'aliases':aliases,'minimum_candidate_cells':ctx.cfg['selection']['min_candidate_cells']})
    return {'outputs':[dest,tax],'metrics':{'classes':len(classes),'source_classes':len(candidates)}}


def biome_plan(ctx):
    return [{'id':'ecoregions','url':ctx.cfg['features']['biome_url']},
            {'id':'countries','url':knobs(ctx)['countries_url']}]


def biome_run(ctx, unit, ckpt):
    # Versioned source URL is snapshotted once; retries verify the persisted content digest.
    dest=ctx.stage_dir()/(unit['id']+'.zip'); meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        record=read_json(meta)
        if record['url']!=unit['url'] or sha256_file(dest)!=record['sha256']:
            raise StageFailed('Cached boundary archive changed')
    else:
        with ctx.requests_session() as session:
            with session.get(unit['url'],stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with atomic_path(dest) as temp:
                    with open(temp,'wb') as f:
                        for chunk in response.iter_content(1024**2):f.write(chunk);ckpt.maybe_stop()
                    with zipfile.ZipFile(temp) as archive:
                        if archive.testzip():raise StageFailed('Boundary archive CRC failed')
        atomic_write_json(meta,{'url':unit['url'],'sha256':sha256_file(dest),
            'licence':'CC-BY-4.0' if unit['id']=='ecoregions' else 'public-domain',
            'source':'RESOLVE Ecoregions 2017' if unit['id']=='ecoregions' else 'Natural Earth 1:110m admin-0'})
    return {'outputs':[dest,meta],'metrics':{'bytes':dest.stat().st_size}}


def _boundary_index(archive, work, required):
    import shapefile
    import shapely
    from shapely.geometry import shape
    work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Extract only same-stem shape sidecars, rejecting traversal and unrelated contents.
        names=sorted(n for n in z.namelist() if n.lower().endswith('.shp'))
        if len(names)!=1:raise StageFailed('Expected one shapefile in boundary archive')
        stem=names[0][:-4]
        for name in z.namelist():
            if name.rsplit('.',1)[0]==stem and name.rsplit('.',1)[-1].lower() in ('shp','shx','dbf','prj','cpg'):
                (work/Path(name).name).write_bytes(z.read(name))
    reader=shapefile.Reader(str(work/Path(names[0]).name),encoding='utf-8')
    fields=[f[0] for f in reader.fields[1:]]
    if any(name not in fields for name in required):raise StageFailed(f'Missing boundary fields: {required}')
    geo=[]; props=[]
    # Decode only the registered fields. The ecoregion archive has legacy-encoded
    # descriptive names that are unused; its required IDs are numeric.
    for rec in reader.iterShapeRecords(fields=required):
        g=shape(rec.shape.__geo_interface__)
        if not g.is_valid:g=shapely.make_valid(g)
        geo.append(g);props.append(rec.record.as_dict())
    reader.close()
    return shapely.STRtree(geo),geo,props


def boundary_value(index,lon,lat,fields):
    from shapely.geometry import Point
    tree,geo,props=index; ids=sorted(tree.query(Point(lon,lat),predicate='intersects'))
    if not ids:return {key:None for key in fields}
    p=props[ids[0]]
    return {key:p[key] for key in fields}


def selection_candidates(ctx, classes, alias, ckpt):
    idx={r['class_id']:r['class_idx'] for r in classes}
    # Capped per-class heaps bound memory independently of total lattice size.
    import heapq
    heaps={}; retained={}; maxn=ctx.cfg['selection']['candidates_per_class']
    for path in lattice_inputs(ctx):
        for rec in rows(path):
            label=alias.get(rec['class_id'])
            if label is None:continue
            key=idx[label]
            keep_ids=retained.setdefault(key,set())
            # Aliased labels can describe the same window. Count that window
            # once in the bounded per-class reservoir, before heap comparison.
            if rec['candidate_id'] in keep_ids:continue
            rec['target_class_idx']=idx[label]
            merged=collections.defaultdict(float)
            # Exact union will be recomputed after the final grid snap. Alias overlaps are
            # conservatively retained in observed-label vectors here for fold exclusion.
            for k,v in zip(rec.pop('multi_classes'),rec['multi_cov']):
                if k in alias:merged[idx[alias[k]]]=min(1.,merged[idx[alias[k]]]+v)
            rec['multi_idx']=sorted(merged);rec['multi_cov']=[merged[k] for k in rec['multi_idx']]
            rec['labels_idx']=[k for k in rec['multi_idx'] if merged[k]>=ctx.cfg['labels']['tau']]
            rec['labels_cov']=[merged[k] for k in rec['labels_idx']]
            rank=seed_for(ctx.cfg,'select-cap',rec['candidate_id'],label)
            key=rec['target_class_idx'];item=(-rank,rec['candidate_id'],rec)
            heap=heaps.setdefault(key,[])
            if len(heap)<maxn:
                heapq.heappush(heap,item);keep_ids.add(rec['candidate_id'])
            elif item[:2]>heap[0][:2]:
                old=heapq.heapreplace(heap,item);keep_ids.remove(old[1]);keep_ids.add(rec['candidate_id'])
        ckpt.maybe_stop()
    return [t[2] for heap in heaps.values() for t in heap]


def select_run(ctx, unit, ckpt):
    classes=rows(ctx.p('vocab','classes.parquet'))
    alias=read_json(ctx.p('vocab','taxonomy.yaml'))['aliases']
    selected=select_rows(selection_candidates(ctx,classes,alias,ckpt),ctx.cfg,ctx.cfg['seed'])
    if not selected:raise StageFailed('No windows survive frozen spatial selection')
    eco=_boundary_index(ctx.p('biome_fetch','ecoregions.zip'),ctx.tmp/'eco',['ECO_ID','BIOME_NUM'])
    country=_boundary_index(ctx.p('biome_fetch','countries.zip'),ctx.tmp/'country',['CONTINENT'])
    for rec in selected:
        ev=boundary_value(eco,rec['lon'],rec['lat'],['ECO_ID','BIOME_NUM'])
        rec['ecoregion_id']=int(ev['ECO_ID']) if ev['ECO_ID'] is not None else -1
        rec['biome_id']=int(ev['BIOME_NUM']) if ev['BIOME_NUM'] is not None else -1
        rec['continent']=boundary_value(country,rec['lon'],rec['lat'],['CONTINENT'])['CONTINENT'] or 'unassigned'
    dest=ctx.stage_dir()/'windows.parquet';parquet(dest,selected)
    support=[]
    for c in classes:
        sub=[r for r in selected if c['class_idx'] in r['labels_idx']]
        support.append({'class_idx':c['class_idx'],'n_windows':len(sub),'n_cells':len({r['deg_cell'] for r in sub}),
            'n_continents':len({r['continent'] for r in sub if r['continent']!='unassigned'})})
    supp=ctx.stage_dir()/'support.parquet';parquet(supp,support)
    report=ctx.stage_dir()/'selection.json'
    atomic_write_json(report,{'rows':len(selected),'pilot':sum(r['pilot'] for r in selected),
        'splits':dict(collections.Counter(r['split'] for r in selected)),
        'unassigned_continent':sum(r['continent']=='unassigned' for r in selected),
        'unassigned_biome':sum(r['biome_id']==-1 for r in selected),
        'stored_edge_buffer_km':ctx.cfg['selection']['buffer_km'],
        'within_class_guard':'circumscribed stored-footprint circle non-overlap',
        'mapping_density_units':'source polygons per square kilometre in declared degree-cell neighbourhood'})
    return {'outputs':[dest,supp,report],'metrics':{'windows':len(selected),'classes':len(classes)}}


def labels_plan(ctx):
    import pyarrow.parquet as pq
    table=pq.read_table(ctx.p('assemble','windows_qa.parquet'),columns=['widx','window_id'])
    index=table['widx'].to_pylist()
    if index != list(range(len(index))):raise StageFailed('Assembled widx must be contiguous and sorted')
    size=int(knobs(ctx)['labels_chunk'])
    return [{'id':f'shard-{i//size:05d}','start':i,'stop':min(len(index),i+size)} for i in range(0,len(index),size)]


def sparse_masks(index, class_ids, x0, y0, pixels):
    """Exact covered area per 10m pixel, stored only for nonzero class pixels.

    Rasterisation fills interiors; only boundary pixels require polygon intersection.
    The scalar route labels always use exact polygon union, independent of rasterisation.
    """
    import numpy as np
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    tree,geoms,names,_=index
    outer=centre_box(x0+pixels*5,y0-pixels*5,pixels*10)
    by=collections.defaultdict(list)
    for j in tree.query(outer,predicate='intersects'):
        if names[j] in class_ids:by[names[j]].append(geoms[j].intersection(outer))
    affine=from_origin(x0,y0,10,10)
    result=[]
    for label in sorted(by,key=lambda k:class_ids[k]):
        geom=shapely.union_all(by[label])
        if geom.area==0:continue
        values=rasterize([(geom,1)],out_shape=(pixels,pixels),transform=affine,dtype='float32')
        edges=rasterize([(geom.boundary,1)],out_shape=(pixels,pixels),transform=affine,all_touched=True,dtype='uint8')
        for row,col in np.argwhere(edges):
            box=centre_box(x0+(col+.5)*10,y0-(row+.5)*10,10)
            values[row,col]=geom.intersection(box).area/100
        which=np.flatnonzero(values.ravel()>0)
        result.append((class_ids[label],which.astype(np.int32),values.ravel()[which]))
    return result


def exact_record(db, rec, cfg, alias, idx):
    stored=cfg['lattice']['store_px10']*10
    epsg=int(rec['epsg']); x0=float(rec['x0']); y0=float(rec['y0'])
    if abs(x0/20-round(x0/20))>1e-8 or abs(y0/20-round(y0/20))>1e-8:
        raise StageFailed('Exact labels require the declared snapped 20m grid')
    geometry_report={}
    index=exact_window_index(db,epsg,x0,y0,stored,geometry_report)
    # Merge aliases before geometric union, avoiding double-counted overlaps.
    index=(index[0],index[1],[alias.get(k,k) for k in index[2]],index[3])
    x,y=x0+stored/2,y0-stored/2
    out={'widx':int(rec['widx']),'window_id':rec['window_id']}
    r1={}
    for route,column in cfg['labels']['footprint_label_column'].items():
        cov=coverage(index,centre_box(x,y,cfg['labels']['footprints_m'][route]))
        pairs=sorted((idx[k],v) for k,v in cov.items() if k in idx and v>=cfg['labels']['tau'])
        out[column]=[k for k,v in pairs];out[column+'_cov']=[v for k,v in pairs]
        if route=='r1':r1=cov
    multi=sorted((idx[k],v) for k,v in r1.items() if k in idx and v>=cfg['labels']['multilabel_min'])
    out['multi_idx']=[k for k,v in multi];out['multi_cov']=[v for k,v in multi]
    target=int(rec['target_class_idx'])
    target_cov=sum(v for k,v in r1.items() if idx.get(k)==target)
    out['target_eligible']=target in out[cfg['labels']['footprint_label_column']['r1']]
    out['target_coverage']=target_cov
    out['purity']=target_cov/max(sum(r1.values()),1e-15)
    out['reason']='' if out['target_eligible'] else 'exact_target_coverage_below_tau'
    out['projected_repaired']=geometry_report.get('repaired',0);out['projected_skipped']=geometry_report.get('skipped',0)
    return out,index


def labels_run(ctx, unit, ckpt):
    import numpy as np
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    classes=rows(ctx.p('vocab','classes.parquet'));idx={r['class_id']:r['class_idx'] for r in classes}
    alias=read_json(ctx.p('vocab','taxonomy.yaml'))['aliases']
    input_rows=ds.dataset(ctx.p('assemble','windows_qa.parquet')).to_table(
        filter=(ds.field('widx')>=unit['start']) & (ds.field('widx')<unit['stop'])).to_pylist()
    input_rows.sort(key=lambda r:r['widx'])
    directory=ctx.stage_dir()/'parts'/unit['id'];directory.mkdir(parents=True,exist_ok=True)
    # Per-window committed pieces allow exact-label and mask work to resume after preemption.
    manifests=[]
    with connect(ctx.p('osm_areas','areas.sqlite'),True) as db:
        for rec in input_rows:
            dest=directory/(str(rec['widx'])+'.json'); mask=dest.with_suffix('.npz')
            if dest.exists() and mask.exists():
                old=read_json(dest)
                if old.get('input_sha256')!=sha256_json(rec) or old.get('config_sha256')!=sha256_json(ctx.cfg['labels']) or old.get('mask_sha256')!=sha256_file(mask):
                    raise StageFailed('Exact-label checkpoint provenance mismatch')
            else:
                out,index=exact_record(db,rec,ctx.cfg,alias,idx)
                masks=sparse_masks(index,idx,rec['x0'],rec['y0'],ctx.cfg['lattice']['store_px10'])
                pointers=[0];ci=[];pi=[];vals=[]
                for k,p,v in masks:ci.append(k);pi.extend(p);vals.extend(v);pointers.append(len(pi))
                with atomic_path(mask) as temp:
                    with open(temp,'wb') as f:np.savez_compressed(f,class_idx=np.asarray(ci,np.int32),indptr=np.asarray(pointers,np.int64),pixel_idx=np.asarray(pi,np.int32),coverage=np.asarray(vals,np.float32))
                atomic_write_json(dest,{'record':out,'input_sha256':sha256_json(rec),'config_sha256':sha256_json(ctx.cfg['labels']),'mask_sha256':sha256_file(mask)})
            manifests.append(dest)
            if ckpt.due():ckpt.save('labels',{'completed':len(manifests),'total':len(input_rows)})
            ckpt.maybe_stop()
    records=[read_json(p)['record'] for p in manifests]
    table=ctx.stage_dir()/'parts'/(unit['id']+'.parquet');parquet(table,records,labels_schema(ctx.cfg))
    wi=[];ci=[];pi=[];vals=[];pointers=[0]
    for manifest in manifests:
        widx=read_json(manifest)['record']['widx']
        with np.load(manifest.with_suffix('.npz'),allow_pickle=False) as data:
            for i,c in enumerate(data['class_idx']):
                lo,hi=data['indptr'][i:i+2]
                wi.append(widx);ci.append(c);pi.extend(data['pixel_idx'][lo:hi]);vals.extend(data['coverage'][lo:hi]);pointers.append(len(pi))
    mask=ctx.stage_dir()/'masks'/(unit['id']+'.npz')
    with atomic_path(mask) as temp:
        with open(temp,'wb') as f:np.savez_compressed(f,widx=np.asarray(wi,np.int64),class_idx=np.asarray(ci,np.int32),indptr=np.asarray(pointers,np.int64),pixel_idx=np.asarray(pi,np.int32),coverage=np.asarray(vals,np.float32),shape=np.array([ctx.cfg['lattice']['store_px10']]*2,np.int32))
    return {'outputs':[table,mask],'metrics':{'rows':len(records),'target_eligible':sum(r['target_eligible'] for r in records)}}


def labels_finalize(ctx):
    import pyarrow.parquet as pq
    source=ctx.p('assemble','windows_qa.parquet')
    expected=rows(source);parts=sorted((ctx.stage_dir()/'parts').glob('shard-*.parquet'))
    if not parts:raise StageFailed('No exact-label shards')
    dest=ctx.stage_dir()/'labels.parquet';count=0
    # Final output is streamed and matched to the canonical assembled row identity.
    with atomic_path(dest) as temp:
        writer=None
        try:
            for path in parts:
                table=pq.read_table(path)
                if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                for rec in table.select(['widx','window_id']).to_pylist():
                    if count>=len(expected) or (rec['widx'],rec['window_id'])!=(count,expected[count]['window_id']):
                        raise StageFailed('Label rows differ from assembled order')
                    count+=1
                writer.write_table(table)
        finally:
            if writer:writer.close()
        if count!=len(expected):raise StageFailed('Missing exact-label rows')
    return {'rows':count,'sha256':sha256_file(dest)}


def labels_schema(cfg):
    import pyarrow as pa
    fields=[('widx',pa.int64()),('window_id',pa.string())]
    for column in cfg['labels']['footprint_label_column'].values():
        fields.extend([(column,pa.list_(pa.int32())),(column+'_cov',pa.list_(pa.float32()))])
    fields.extend([('multi_idx',pa.list_(pa.int32())),('multi_cov',pa.list_(pa.float32())),
        ('target_eligible',pa.bool_()),('target_coverage',pa.float64()),('purity',pa.float64()),('reason',pa.string()),
        ('projected_repaired',pa.int64()),('projected_skipped',pa.int64())])
    return pa.schema(fields)


def retention(window_rows, label_rows, cfg):
    by_window={r['widx']:r for r in window_rows}
    eligible=collections.defaultdict(list)
    for rec in label_rows:
        if rec['target_eligible']:
            win=by_window[rec['widx']]
            eligible[win['target_class_idx']].append(win)
    chosen=set()
    for group in eligible.values():
        group.sort(key=lambda r:(seed_for(cfg,'post-qa-cap',r['window_id']),r['window_id']))
        chosen.update(r['widx'] for r in group[:cfg['selection']['cap_per_class']])
    return [{'widx':r['widx'],'window_id':r['window_id'],'retained':r['widx'] in chosen,
        'reason':'' if r['widx'] in chosen else ('post_qa_cap' if r['target_eligible'] else r['reason'])}
        for r in label_rows]


def admission_counts(classes, window_rows, label_rows, cfg):
    by_window={r['widx']:r for r in window_rows}
    support={r['class_idx']:{s:[] for s in ('train','val','test')} for r in classes}
    for rec in label_rows:
        if not rec['target_eligible']:continue
        win=by_window[rec['widx']]
        for k in rec[cfg['labels']['footprint_label_column']['r1']]:
            support[k][win['split']].append(win)
    output=[]; threshold=cfg['selection']['admission']
    for c in classes:
        by=support[c['class_idx']];allrows=sum(by.values(),[])
        record={'class_idx':c['class_idx'],'class_id':c['class_id'],**{'n_'+s:len(by[s]) for s in by},
            'n_test_cells':len({r['deg_cell'] for r in by['test']}),'n_cells':len({r['deg_cell'] for r in allrows})}
        failures=[s for s in by if record['n_'+s]<threshold[s]]
        if record['n_test_cells']<threshold['test_cells']:failures.append('test_cells')
        if record['n_cells']<threshold['cells']:failures.append('cells')
        record.update(admitted=not failures,reason=','.join(failures));output.append(record)
    return output


def admit_run(ctx, unit, ckpt):
    window_rows=rows(ctx.p('assemble','windows_qa.parquet'));labels=rows(ctx.p('labels_exact','labels.parquet'))
    classes=rows(ctx.p('vocab','classes.parquet'))
    if len(labels)!=len(window_rows) or [r['window_id'] for r in labels]!=[r['window_id'] for r in window_rows]:
        raise StageFailed('Admission row identity mismatch')
    window_mask=retention(window_rows,labels,ctx.cfg)
    retained={r['widx'] for r in window_mask if r['retained']}
    result=admission_counts(classes,window_rows,[r for r in labels if r['widx'] in retained],ctx.cfg)
    mask_file=ctx.stage_dir()/'windows_final.parquet';parquet(mask_file,window_mask)
    dest=ctx.stage_dir()/'classes_final.parquet';parquet(dest,result)
    # Preserve all class_idx values; dropped classes are explicit records.
    if not any(r['admitted'] for r in result):raise StageFailed('No class passes frozen post-QA admission')
    report=ctx.stage_dir()/'admission.json'
    atomic_write_json(report,{'classes':len(result),'admitted':sum(r['admitted'] for r in result),
        'thresholds':ctx.cfg['selection']['admission'],'eligible_windows':sum(r['target_eligible'] for r in labels),
        'total_windows':len(labels),'retained_windows':len(retained),'rows_remain_canonical':True})
    return {'outputs':[dest,report,mask_file],'metrics':{'admitted':sum(r['admitted'] for r in result)}}
