"""Exact polygon-union coverage, geographic splitting and deterministic selection.

All geometry imports are local so Mission Control can inspect the DAG on login.
"""
from __future__ import annotations
import hashlib
import math
import sqlite3
from pathlib import Path


def classification(tags, taxonomy):
    """One visual primary plus its first declared refinement; personal tags never escape."""
    tags = dict(tags)
    for old, new in taxonomy['synonyms'].items():
        key, value = old.split('=', 1)
        if tags.get(key) == value:
            tags.pop(key)
            tags.update(new)
    if any(k in tags for k in taxonomy['exclude_presence']):
        return None
    if any(tags.get(k) in values for k, values in taxonomy['exclude_values'].items()):
        return None
    for key in taxonomy['primary_keys']:
        value = tags.get(key)
        if not value or value in ('yes', 'no') or ';' in value:
            continue
        if key in taxonomy['only_values'] and value not in taxonomy['only_values'][key]:
            continue
        primary = key + '=' + value
        for rk in taxonomy['refinements'].get(primary, []):
            rv = tags.get(rk)
            if rv and rv not in ('yes', 'no') and ';' not in rv:
                return primary + '|' + rk + '=' + rv
        return primary
    return None


def class_fields(class_id):
    primary, *ref = class_id.split('|')
    key, value = primary.split('=', 1)
    rk, rv = ref[0].split('=', 1) if ref else ('', '')
    return dict(class_id=class_id, primary_key=key, primary_value=value,
                refine_key=rk, refine_value=rv)


def cell_id(lon, lat, step):
    return f'x{math.floor((lon + 180) / step):04d}y{math.floor((lat + 90) / step):04d}'


def cell_bounds(key, step):
    x, y = int(key[1:5]), int(key[6:])
    return (-180 + x * step, -90 + y * step, -180 + (x + 1) * step, -90 + (y + 1) * step)


def split_for(key, cfg):
    frac = cfg['split_fracs']
    if set(frac) != {'train', 'val', 'test'} or abs(sum(frac.values()) - 1) > 1e-10:
        raise ValueError('split_fracs must sum to one')
    u = int.from_bytes(hashlib.sha256((cfg['split_salt'] + '|' + key).encode()).digest()[:8], 'big') / 2**64
    return 'train' if u < frac['train'] else ('val' if u < frac['train'] + frac['val'] else 'test')


def utm_epsg(lon, lat):
    # Polar labels have a supported metric CRS too; unavailable S2 scenes remain QA failures.
    if lat >= 84:
        return 3413
    if lat < -80:
        return 3031
    return (32600 if lat >= 0 else 32700) + min(60, max(1, int((lon + 180) // 6) + 1))


def antimeridian_parts(geometry):
    import shapely
    from shapely.ops import transform, split
    from shapely.geometry import LineString
    from shapely.affinity import translate
    parts = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    out = []
    for part in parts:
        if part.bounds[2] - part.bounds[0] <= 180:
            out.append(part)
            continue
        moved = transform(lambda x, y, z=None: (x + 360 if x < 0 else x, y), part)
        for piece in split(moved, LineString([(180, -90), (180, 90)])).geoms:
            if piece.representative_point().x >= 180:
                piece = translate(piece, xoff=-360)
            out.append(piece)
    return [x for x in out if not x.is_empty and shapely.is_valid(x)]


def connect(path, readonly=False):
    db = sqlite3.connect(f'file:{Path(path).absolute()}?mode=ro' if readonly else str(path), uri=readonly)
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('PRAGMA cache_size=-65536')
    return db


def area_rows(db, bbox):
    x0, y0, x1, y1 = bbox
    # Rtree bounds are rounded outward. Exact geometry intersection follows.
    return db.execute('SELECT a.id,a.class_id,a.wkb,a.osm_type,a.osm_id,a.version FROM areas a '
                      'JOIN bounds b ON a.id=b.id WHERE b.maxx>=? AND b.minx<=? AND b.maxy>=? AND b.miny<=?',
                      (x0, x1, y0, y1))


def _polygonal(g):
    """Keep only the polygonal parts of a make_valid result (it may return a GeometryCollection)."""
    import shapely
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    parts = [p for p in getattr(g, 'geoms', []) if p.geom_type in ('Polygon', 'MultiPolygon')]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def projected_index(db, epsg, bbox, report=None):
    """Project the polygons touching `bbox` into the cell's UTM zone.

    Large relations (coastlines, lakes, forests) that extend far from the zone can become
    self-intersecting after projection. They are repaired with make_valid; a geometry that
    stays invalid, empty or non-finite is skipped. Both counts go into `report` so the stage
    records them, instead of one such relation failing the whole cell.
    """
    import shapely
    from shapely.ops import transform
    from pyproj import Transformer
    forward = Transformer.from_crs(4326, epsg, always_xy=True).transform
    geo, ids, provenance = [], [], []
    for _, class_id, wkb, kind, oid, version in area_rows(db, bbox):
        g = transform(forward, shapely.from_wkb(wkb))
        if not g.is_empty and not g.is_valid:
            g = _polygonal(shapely.make_valid(g))
            if report is not None:
                report['repaired'] = report.get('repaired', 0) + 1
        if g.is_empty or not g.is_valid or not all(math.isfinite(v) for v in g.bounds):
            if report is not None:
                report['skipped'] = report.get('skipped', 0) + 1
                report.setdefault('skipped_ids', [])
                if len(report['skipped_ids']) < 20:
                    report['skipped_ids'].append(f'{kind}{oid}')
            continue
        geo.append(g); ids.append(class_id); provenance.append((kind, oid, version))
    return shapely.STRtree(geo), geo, ids, provenance


def coverage(index, footprint):
    """Union before measuring, including overlapping same-class polygons."""
    import shapely
    tree, geoms, classes, provenance = index
    by = {}
    for i in tree.query(footprint, predicate='intersects'):
        clipped = geoms[i].intersection(footprint)
        if clipped.area > 0:
            by.setdefault(classes[i], []).append(clipped)
    areas = {k: min(1., max(0., shapely.union_all(v).area / footprint.area)) for k, v in by.items()}
    return dict(sorted(areas.items()))


def exact_window_index(db, epsg, x0, y0, stored_m, report=None):
    from pyproj import Transformer
    bounds = Transformer.from_crs(epsg, 4326, always_xy=True).transform_bounds(
        x0, y0 - stored_m, x0 + stored_m, y0, densify_pts=21)
    if bounds[2] < bounds[0]:
        # Dateline-crossing stored windows: query both sides and project as one neighbourhood.
        import shapely
        a = projected_index(db, epsg, (bounds[0], bounds[1], 180., bounds[3]), report)
        b = projected_index(db, epsg, (-180., bounds[1], bounds[2], bounds[3]), report)
        geoms, ids, provenance = a[1] + b[1], a[2] + b[2], a[3] + b[3]
        return shapely.STRtree(geoms), geoms, ids, provenance
    return projected_index(db, epsg, bounds, report)


def centre_box(x, y, size):
    from shapely.geometry import box
    return box(x-size/2, y-size/2, x+size/2, y+size/2)


def haversine_km(lon1, lat1, lon2, lat2):
    a, b = math.radians(lat1), math.radians(lat2)
    d = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., d))))


class SphericalBins:
    """Bounded-neighbour search with seam-safe Cartesian unit-vector bins."""
    def __init__(self, radius_km):
        self.radius = radius_km
        self.chord = 2 * math.sin(radius_km / (2*6371.0088))
        self.bins = {}

    def key(self, row):
        a, b = math.radians(row['lon']), math.radians(row['lat'])
        return tuple(math.floor(v/self.chord) for v in (math.cos(b)*math.cos(a), math.cos(b)*math.sin(a), math.sin(b)))

    def neighbours(self, row):
        from itertools import product
        key = self.key(row)
        for delta in product((-1, 0, 1), repeat=3):
            for other in self.bins.get(tuple(x+d for x,d in zip(key,delta)), []):
                if haversine_km(row['lon'], row['lat'], other['lon'], other['lat']) < self.radius:
                    yield other

    def add(self, row):
        self.bins.setdefault(self.key(row), []).append(row)


def select_rows(candidates, cfg, seed):
    """Bounded candidates -> seeded round-robin selection, class cap and geographic guards."""
    from collections import defaultdict, Counter
    stored_m = cfg['lattice']['store_px10'] * 10
    settings = cfg['selection']
    # Conservative edge separation: circumscribed circles guarantee the requested gap.
    guard = SphericalBins(settings['buffer_km'] + math.sqrt(2)*stored_m/1000)
    same_class = SphericalBins(math.sqrt(2)*stored_m/1000)
    by = defaultdict(list)
    for row in candidates:
        by[row['target_class_idx']].append(row)
    def order(row):
        return hashlib.sha256(f"{seed}|select|{row['candidate_id']}|{row['target_class_idx']}".encode()).hexdigest()
    for key in by:
        by[key].sort(key=order)
        by[key] = by[key][:settings['candidates_per_class']]
    budget = int(settings['budget'])
    # All classes share the same final cap when budget is binding.
    cap = min(int(math.ceil(settings['cap_per_class'] * settings['oversample'])), budget // max(1,len(by)))
    if cap < 1:
        raise ValueError('Budget cannot allocate one candidate to every class')
    counts, per_cell = Counter(), Counter()
    selected, unique = [], {}
    # Interleaving candidates protects less common classes against domination by an early key.
    for rank in range(max(map(len, by.values()), default=0)):
        for k in sorted(by):
            if rank >= len(by[k]) or counts[k] >= cap:
                continue
            row = dict(by[k][rank]); cell = row['deg_cell']
            if per_cell[k,cell] >= settings['max_per_class_per_deg']:
                continue
            if row['candidate_id'] in unique:
                continue
            if any(r['split'] != row['split'] for r in guard.neighbours(row)):
                continue
            if any(r['target_class_idx'] == k for r in same_class.neighbours(row)):
                continue
            guard.add(row); same_class.add(row)
            unique[row['candidate_id']] = row
            counts[k] += 1; per_cell[k,cell] += 1; selected.append(row)
            if len(selected) >= budget:
                break
        if len(selected) >= budget:
            break
    selected.sort(key=lambda r: r['candidate_id'])
    val = sorted((r for r in selected if r['split']=='val'), key=order)
    # Round-robin across class and cell makes the validation-only pilot stratified.
    groups = defaultdict(list)
    for row in val:
        groups[(row['target_class_idx'],row['deg_cell'])].append(row)
    pilot = set()
    for rank in range(max(map(len,groups.values()), default=0)):
        for group in sorted(groups):
            if rank < len(groups[group]) and len(pilot) < cfg['imagery']['pilot_windows']:
                pilot.add(groups[group][rank]['candidate_id'])
    for i,row in enumerate(selected):
        row.update(window_id=f'w{i:07d}', pilot=row['candidate_id'] in pilot, priority=0 if row['candidate_id'] in pilot else 1)
    return selected
