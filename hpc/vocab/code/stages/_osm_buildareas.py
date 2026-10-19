"""Restartable streaming pyosmium scan into a disk-backed spatial label index.

The parent process polls checkpoints and terminates this process on worker yield.
SQLite transactions preserve completed batches; resume replays the source stream
through the last committed area, without retaining planet geometry in memory.

Regional builds pass `id_offset` so that every region owns a disjoint id range and the
regional databases can be merged without renumbering (see `_osm_pipeline.merge_area_databases`).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from ._osm_geometry import classification, antimeridian_parts, cell_id, connect

SCHEMA = '''
  CREATE TABLE IF NOT EXISTS areas(id INTEGER PRIMARY KEY, class_id TEXT NOT NULL,
    wkb BLOB NOT NULL, osm_type TEXT NOT NULL, osm_id INTEGER NOT NULL, version INTEGER NOT NULL,
    wkb_sha BLOB);
  CREATE VIRTUAL TABLE IF NOT EXISTS bounds USING rtree(id,minx,maxx,miny,maxy);
  CREATE TABLE IF NOT EXISTS cells(cell TEXT PRIMARY KEY);
  CREATE TABLE IF NOT EXISTS progress(key TEXT PRIMARY KEY,value TEXT NOT NULL);
'''


def build(source, database, settings, fingerprint, node_index, id_offset=0, allow_empty=False):
    import osmium
    import shapely
    from pyproj import Geod
    db = connect(database)
    db.executescript(SCHEMA)
    old = dict(db.execute('SELECT key,value FROM progress'))
    if old and old.get('fingerprint') != fingerprint:
        raise ValueError('Area checkpoint source/taxonomy differs')
    if old and int(old.get('id_offset', id_offset)) != int(id_offset):
        raise ValueError('Area checkpoint id offset differs')
    cursor = int(old.get('seen', 0))
    next_id = int(old.get('next_id', id_offset))
    counts = json.loads(old.get('counts', '{}'))
    counts.setdefault('accepted_parts', 0); counts.setdefault('excluded_taxonomy', 0)
    counts.setdefault('invalid_geometry', 0); counts.setdefault('below_min_area', 0)
    db.executemany('INSERT OR REPLACE INTO progress VALUES (?,?)',
                   [('fingerprint', fingerprint), ('id_offset', str(int(id_offset)))])
    db.commit()
    geod = Geod(ellps='WGS84'); factory = osmium.geom.WKBFactory()
    node_index = Path(node_index); node_index.parent.mkdir(parents=True, exist_ok=True)
    node_index.unlink(missing_ok=True)
    primary = settings['taxonomy']['primary_keys']
    fp = (osmium.FileProcessor(str(source)).with_locations('sparse_file_array,'+str(node_index))
          .with_areas(osmium.filter.KeyFilter(*primary))
          .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
          .with_filter(osmium.filter.KeyFilter(*primary)))
    seen = 0
    def commit():
        db.executemany('INSERT OR REPLACE INTO progress VALUES (?,?)',
                       [('seen', str(seen)), ('next_id', str(next_id)), ('counts', json.dumps(counts, sort_keys=True))])
        db.commit()
    try:
        for seen, area in enumerate(fp, 1):
            if seen <= cursor:
                continue
            label = classification({tag.k:tag.v for tag in area.tags}, settings['taxonomy'])
            if label is None:
                counts['excluded_taxonomy'] += 1
            else:
                try:
                    geometry = shapely.from_wkb(bytes.fromhex(factory.create_multipolygon(area)))
                except Exception as exc:
                    # Broken OSM rings are explicit exclusions, counted in the final report.
                    counts['invalid_geometry'] += 1
                    geometry = None
                if geometry is not None:
                    if not geometry.is_valid or geometry.is_empty:
                        counts['invalid_geometry'] += 1
                    else:
                        for part in antimeridian_parts(geometry):
                            area_m2 = abs(geod.geometry_area_perimeter(part)[0])
                            if area_m2 <= settings['min_polygon_m2']:
                                counts['below_min_area'] += 1
                                continue
                            x0,y0,x1,y1 = part.bounds
                            wkb = shapely.to_wkb(part)
                            db.execute('INSERT INTO areas(id,class_id,wkb,osm_type,osm_id,version,wkb_sha) VALUES (?,?,?,?,?,?,?)',
                                (next_id,label,wkb,'way' if area.from_way() else 'relation',area.orig_id(),area.version,
                                 hashlib.sha1(wkb).digest()))
                            db.execute('INSERT INTO bounds VALUES (?,?,?,?,?)',(next_id,x0,x1,y0,y1))
                            next_id += 1
                            step = settings['deg_cell']
                            xmin,xmax = max(0,math.floor((x0+180)/step)),min(math.ceil(360/step)-1,math.floor((x1+180)/step))
                            ymin,ymax = max(0,math.floor((y0+90)/step)),min(math.ceil(180/step)-1,math.floor((y1+90)/step))
                            db.executemany('INSERT OR IGNORE INTO cells VALUES (?)',
                                ((f'x{x:04d}y{y:04d}',) for x in range(xmin,xmax+1) for y in range(ymin,ymax+1)))
                            counts['accepted_parts'] += 1
            if seen % settings['area_commit_rows'] == 0:
                commit()
                print(json.dumps({'seen':seen,**counts}),flush=True)
        commit()
        if not counts['accepted_parts'] and not allow_empty:
            raise ValueError('No valid visual polygons remain')
        db.execute('CREATE INDEX IF NOT EXISTS areas_class ON areas(class_id)')
        db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)', ('complete','true')); db.commit()
        return {'seen':seen,**counts}
    finally:
        db.close(); node_index.unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('config'); args=p.parse_args()
    c=json.loads(Path(args.config).read_text())
    print(json.dumps(build(**c)),flush=True)
