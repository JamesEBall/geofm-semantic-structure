"""Frozen STAC candidate selection and source metadata validation."""
from __future__ import annotations
import datetime as dt
import math
from urllib.parse import urlsplit, urlunsplit

SOURCES = {
    'planetary_computer': ('https://planetarycomputer.microsoft.com/api/stac/v1', 'sentinel-2-l2a'),
    'earth_search': ('https://earth-search.aws.element84.com/v1', 'sentinel-2-c1-l2a'),
}
ASSET_KEYS = {'B02': 'blue', 'B03': 'green', 'B04': 'red', 'B08': 'nir', 'B05': 'rededge1',
              'B06': 'rededge2', 'B07': 'rededge3', 'B8A': 'nir08', 'B11': 'swir16', 'B12': 'swir22', 'SCL': 'scl'}


def season(lat, cfg):
    seasons = cfg['imagery']['seasons']
    key = 'tropics' if abs(lat) < seasons['tropics_abs_lat'] else ('north' if lat >= 0 else 'south')
    return key, seasons[key][0] + 'T00:00:00Z/' + seasons[key][1] + 'T23:59:59Z'


def baseline_number(value):
    text = str(value).replace('N', '')
    v = float(text)
    return v / 100 if v >= 100 else v


def epsg_of(item, asset=None):
    for p in (asset or {}, item.get('properties', {})):
        if p.get('proj:epsg'):
            return int(p['proj:epsg'])
        if p.get('proj:code'):
            return int(p['proj:code'].split(':')[-1])
    raise ValueError('source has no declared projection')


def radiometry(item, asset, source):
    baseline = baseline_number(item['properties']['s2:processing_baseline'])
    if baseline < 4:
        raise ValueError('unexpected old processing baseline for the frozen 2025 cohort')
    bands = asset.get('raster:bands') or []
    declared = bands[0] if bands else {}
    if source == 'earth_search':
        if 'scale' not in declared or 'offset' not in declared:
            raise ValueError('Earth Search lacks asset scale/offset')
        scale, offset = float(declared['scale']), float(declared['offset'])
        if abs(scale - 1e-4) > 1e-12 or abs(offset + .1) > 1e-12:
            raise ValueError('Earth Search radiometry differs from frozen C1 convention')
    elif source == 'planetary_computer':
        if item['properties'].get('earthsearch:boa_offset_applied') is True:
            raise ValueError('unexpected harmonised PC source')
        # PC raw L2A assets omit raster scale/offset. The explicit s2 baseline is
        # the documented conversion convention; contradictory declarations fail.
        scale, offset = 1e-4, -.1
        if ('scale' in declared and abs(float(declared['scale']) - scale) > 1e-12) or ('offset' in declared and abs(float(declared['offset']) - offset) > 1e-12):
            raise ValueError('PC metadata contradicts unharmonised L2A radiometry')
    else:
        raise ValueError('unknown radiometry source')
    return offset / scale, scale


def asset_for(item, band, source):
    key = band if source == 'planetary_computer' else ASSET_KEYS[band]
    asset = item['assets'][key]
    href = asset['href']
    # Direct files are accepted only for explicit local test fixtures.
    if not href.startswith('https://') and not item.get('_local_fixture'):
        raise ValueError('asset is not an HTTPS COG')
    if urlsplit(href).query:
        raise ValueError('catalogue must not contain signed asset URLs')
    names = [b.get('name') for b in asset.get('eo:bands', [])]
    suffix = urlsplit(href).path.rsplit('/', 1)[-1].upper()
    if band != 'SCL' and band not in names and suffix != band + '.TIF':
        raise ValueError('asset band identity is not independently declared')
    return asset


def priority(item):
    p = item['properties']
    return (float(p.get('eo:cloud_cover', 100)) + float(p.get('s2:snow_ice_percentage', 100)), p['datetime'], item['id'])


def search(session, source, bbox, interval, cfg, stop=lambda: None):
    endpoint, collection = SOURCES[source]
    body = {'collections': [collection], 'bbox': list(bbox), 'datetime': interval,
            'query': {'eo:cloud_cover': {'lte': cfg['imagery']['stac_cloud_max']}}, 'limit': 100}
    url, method, payload = endpoint + '/search', 'POST', body
    seen_pages, items = set(), {}
    while url:
        stop()
        token = (url, str(payload))
        if token in seen_pages:
            raise RuntimeError('STAC pagination cycle')
        seen_pages.add(token)
        r = session.post(url, json=payload, timeout=60) if method == 'POST' else session.get(url, timeout=60)
        r.raise_for_status()
        page = r.json()
        if 'features' not in page:
            raise ValueError('invalid STAC feature response')
        for item in page['features']:
            if float(item['properties'].get('eo:cloud_cover', 100)) <= cfg['imagery']['stac_cloud_max']:
                item = {k: v for k, v in item.items() if k != 'links'}
                items[item['id']] = item
        nxt = next((l for l in page.get('links', []) if l.get('rel') == 'next'), None)
        if not nxt:
            break
        url, method = nxt['href'], nxt.get('method', 'GET').upper()
        if urlsplit(url).netloc != urlsplit(endpoint).netloc:
            raise ValueError('STAC pagination changed endpoint host')
        payload = {**body, **nxt.get('body', {})} if nxt.get('merge') else nxt.get('body', {})
    return sorted(items.values(), key=priority)


def bounds_for(row):
    from pyproj import Transformer
    epsg = int(row['utm_epsg'])
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(float(row['lon']), float(row['lat']))
    x0, y0 = round(x / 20) * 20 - 1280, round(y / 20) * 20 + 1280
    back = Transformer.from_crs(epsg, 4326, always_xy=True)
    corners = [back.transform(xx, yy) for xx, yy in [(x0, y0), (x0+2560, y0), (x0+2560, y0-2560), (x0, y0-2560)]]
    return epsg, x0, y0, corners


def candidates(items, row, source, cfg):
    from shapely.geometry import Polygon, shape
    epsg, _, _, corners = bounds_for(row)
    footprint = Polygon(corners)
    result = []
    for item in items:
        try:
            if epsg_of(item, asset_for(item, 'B02', source)) != epsg or not shape(item['geometry']).covers(footprint):
                continue
            for band in ASSET_KEYS:
                a = asset_for(item, band, source)
                if band != 'SCL':
                    radiometry(item, a, source)
        except (KeyError, ValueError):
            continue
        result.append(item)
    return sorted(result, key=priority)[:int(cfg['imagery']['max_candidates_per_window'])]
