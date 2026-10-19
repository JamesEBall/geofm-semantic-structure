"""Native COG reads, QA, and crash-safe per-window caches."""
from __future__ import annotations
import datetime as dt
import fcntl
import hashlib
import random
import time
from pathlib import Path
from urllib.parse import urlsplit
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file
from ._imagery_catalogue import asset_for, baseline_number, bounds_for, radiometry
from ._imagery_io import BANDS10, BANDS20, analysis_crop, reflectance


class QualityReject(Exception):
    pass


class TechnicalFailure(Exception):
    pass


class Signer:
    """One Planetary Computer SAS token for the whole run.

    The token is cached on shared storage (`cache_path`) and refreshed under a file lock, so across
    every worker and thread the token endpoint sees about one request per hour instead of one per
    unit start. A 429 (or the session's RetryError for it) is waited out using Retry-After, with
    bounded exponential back-off, instead of failing the unit.
    """
    TOKEN_URL = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a'
    MIN_REMAINING = 600.0

    def __init__(self, session, cache_path=None, sleep=None, attempts=8):
        import threading
        self.session, self.token, self.expiry = session, None, 0.
        self.cache_path = Path(cache_path) if cache_path else None
        self.lock = threading.Lock()
        self.sleep = sleep or time.sleep
        self.attempts = attempts

    def _valid(self, token, expiry):
        return bool(token) and expiry - time.time() > self.MIN_REMAINING

    def _adopt(self, doc):
        if doc and self._valid(doc.get('token'), float(doc.get('expiry', 0))):
            self.token, self.expiry = doc['token'], float(doc['expiry'])
            return True
        return False

    def _fetch(self):
        import requests
        delay = 5.0
        for attempt in range(self.attempts):
            try:
                r = self.session.get(self.TOKEN_URL, timeout=30)
            except requests.exceptions.RetryError:
                r = None
            if r is not None and r.status_code == 200:
                obj = r.json()
                expiry = dt.datetime.fromisoformat(obj['msft:expiry'].replace('Z', '+00:00')).timestamp()
                if expiry - time.time() < 60:
                    raise TechnicalFailure('source token expires too soon')
                return {'token': obj['token'], 'expiry': expiry, 'fetched': time.time()}
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            wait = delay
            if r is not None and r.headers.get('Retry-After', '').isdigit():
                wait = max(wait, float(r.headers['Retry-After']))
            self.sleep(min(wait, 120.0) + random.uniform(0, 1))
            delay = min(delay * 2, 120.0)
        raise TechnicalFailure('token endpoint kept rate-limiting after %d attempts' % self.attempts)

    def _refresh(self, force):
        if self.cache_path is None:
            doc = self._fetch(); self._adopt(doc); return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.cache_path) + '.lock', 'a') as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                if not force and self._adopt(read_json(self.cache_path)):
                    return                      # another worker refreshed while we waited
                doc = self._fetch()
                atomic_write_json(self.cache_path, doc)
                self._adopt(doc)
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)

    def sign(self, href, source, force=False):
        if source != 'planetary_computer' or not href.startswith('https://'):
            return href
        with self.lock:
            if force or not self._valid(self.token, self.expiry):
                if force or self.cache_path is None or not self._adopt(read_json(self.cache_path)):
                    self._refresh(force)
            return href + '?' + self.token


def read_native(asset, item, band, source, signer, epsg, x0, y0, size, proxy=None):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    px = 10 if band in BANDS10 else 20
    # Fewer, larger range requests: read the whole COG header in one go, fetch tile-sized chunks and
    # coalesce neighbouring ranges. Each round trip costs tens of milliseconds through the tunnel.
    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='20',
                GDAL_HTTP_CONNECTTIMEOUT='10', GDAL_HTTP_MAX_RETRY='2', GDAL_HTTP_RETRY_DELAY='1',
                GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
                GDAL_INGESTED_BYTES_AT_OPEN='65536',
                # Tile-sized chunks: ~1.65 MB deflate tiles for 10 m bands, ~0.45 MB for 20 m bands and SCL.
                CPL_VSIL_CURL_CHUNK_SIZE='2097152' if px == 10 else '524288',
                CPL_VSIL_CURL_CACHE_SIZE='67108864')
    if proxy:
        opts['GDAL_HTTP_PROXY'] = proxy
    for refresh in range(2):
        href = signer.sign(asset['href'], source, force=bool(refresh))
        try:
            with rasterio.Env(**opts), rasterio.open(href) as ds:
                tr = ds.transform
                if ds.crs is None or ds.crs.to_epsg() != epsg or tr.a != px or tr.e != -px or tr.b or tr.d:
                    raise ValueError('source CRS or native band resolution differs')
                col, row = (x0 - tr.c) / px, (tr.f - y0) / px
                if abs(col-round(col)) > 1e-8 or abs(row-round(row)) > 1e-8:
                    raise ValueError('band grid is not aligned to the frozen 20 m grid')
                col, row = round(col), round(row)
                if col < 0 or row < 0 or col+size > ds.width or row+size > ds.height:
                    raise QualityReject('outside_native_tile')
                array = ds.read(1, window=Window(col, row, size, size))
                if array.shape != (size, size):
                    raise ValueError('truncated raster window')
                expected = np.uint8 if band == 'SCL' else np.uint16
                if array.dtype != expected:
                    raise ValueError('raw source DN dtype differs')
                # A non-identity raster conversion must agree with the declared STAC
                # convention. GDAL read returns stored DN; it does not apply scales.
                if band != 'SCL' and (ds.scales[0] != 1. or ds.offsets[0] != 0.):
                    off, scale = radiometry(item, asset, source)
                    if abs(ds.scales[0]-scale) > 1e-12 or abs(ds.offsets[0]-off*scale) > 1e-12:
                        raise ValueError('COG scale/offset contradict STAC convention')
                return array
        except rasterio.errors.RasterioError as exc:
            # Do not persist/log GDAL error strings: they can contain signed URLs.
            if source != 'planetary_computer' or refresh:
                raise TechnicalFailure('COG access failed after retries') from None
    raise TechnicalFailure('COG access failed')


def quality_scl(scl, cfg):
    import numpy as np
    central = scl[32:96, 32:96]
    bad = float(np.isin(central, cfg['imagery']['scl_bad']).mean())
    snow = float(np.isin(central, cfg['imagery']['scl_snow']).mean())
    if bad > cfg['imagery']['scl_max_bad_fraction']:
        raise QualityReject('no_clear_scene')
    return bad, snow


def read_scene(item, row, source, cfg, signer, proxy=None, stop=lambda: None):
    import numpy as np
    epsg, x0, y0, _ = bounds_for(row)
    stop()
    scl = read_native(asset_for(item, 'SCL', source), item, 'SCL', source, signer, epsg, x0, y0, 128, proxy)
    bad, snow = quality_scl(scl, cfg)
    arrays, rad = {}, []
    for band in BANDS10 + BANDS20:
        stop()
        asset = asset_for(item, band, source)
        rad.append(radiometry(item, asset, source))
        arrays[band] = read_native(asset, item, band, source, signer, epsg, x0, y0, 256 if band in BANDS10 else 128, proxy)
    if len(set(rad)) != 1:
        raise ValueError('per-band offsets/scales differ; scalar metadata would be invalid')
    b10, b20 = np.stack([arrays[k] for k in BANDS10]), np.stack([arrays[k] for k in BANDS20])
    a10, a20, _ = analysis_crop(b10, b20, scl)
    if np.any(a10 == 0) or np.any(a20 == 0) or np.any(a10 == 65535) or np.any(a20 == 65535):
        raise QualityReject('nodata_or_saturation_in_analysis')
    off, scale = rad[0]
    p = item['properties']
    digest = hashlib.sha256(b10.tobytes()+b20.tobytes()+scl.tobytes()).hexdigest()
    meta = {'window_id': row['window_id'], 'item_id': item['id'], 'source': source, 'datetime': p['datetime'],
            'epsg': epsg, 'x0': x0, 'y0': y0, 'dn_offset': off, 'dn_scale': scale,
            'baseline': str(p['s2:processing_baseline']), 'bad_frac': bad, 'snow_frac': snow,
            'mgrs_tile': p.get('s2:mgrs_tile', ''), 'pixel_sha256': digest, 'alignment_error_m': 0.0,
            'radiometry_basis': 'raw_PC_L2A_processing_baseline' if source == 'planetary_computer' else 'asset_raster_bands'}
    return {'b10': b10, 'b20': b20, 'scl': scl, 'meta': meta}


def match_scene(item, others):
    p = item['properties']
    product = p.get('s2:product_uri')
    # Providers use different timestamp semantics and MGRS field names. The
    # shared SAFE product identifier binds the acquisition and processing version.
    if product:
        return next((x for x in others if x['properties'].get('s2:product_uri') == product), None)
    tile = p.get('s2:mgrs_tile')
    if not tile:
        return None
    def stamp(v):
        return dt.datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()
    # Allow fractional timestamp precision differences only. Tile and sensing time
    # must identify the same physical acquisition; unrelated scenes are not substitutes.
    return next((x for x in others if x['properties'].get('s2:mgrs_tile') == tile
                 and abs(stamp(x['properties']['datetime'])-stamp(p['datetime'])) < 1), None)


def acquire_window(row, catalogue, cfg, signer, proxy=None, stop=lambda: None, reader=read_scene):
    primary = cfg['imagery']['primary_source']
    fallback = cfg['imagery']['fallback_source']
    sources = catalogue['sources']
    items = sources.get(primary, [])
    source = primary
    fallback_reason = None
    if catalogue.get('primary_search_failed'):
        items, source, fallback_reason = sources.get(fallback, []), fallback, 'primary_search_technical_failure'
    if not items:
        return {'reject': {'window_id': row['window_id'], 'reason': 'no_eligible_scene', 'tries': 0}}
    technical = False
    last_reason, tries = 'no_clear_scene', 0
    for tries, item in enumerate(items[:cfg['imagery']['max_scene_tries']], 1):
        stop()
        try:
            result = reader(item, row, source, cfg, signer, proxy, stop)
        except QualityReject as exc:
            last_reason = str(exc)
            continue
        except TechnicalFailure:
            other = match_scene(item, sources.get(fallback, [])) if source == primary else None
            if other is None:
                technical = True
                continue
            try:
                result = reader(other, row, fallback, cfg, signer, proxy, stop)
                fallback_reason = 'primary_COG_technical_failure_same_acquisition'
            except QualityReject as exc:
                last_reason = str(exc)
                continue
            except TechnicalFailure:
                technical = True
                continue
        result['meta'].update(tries=tries, fallback_reason=fallback_reason)
        return result
    return {'reject': {'window_id': row['window_id'], 'reason': 'technical_unresolved' if technical else last_reason, 'tries': tries}}


def cached_window(directory, key, fn, repair=None):
    """Crash-safe per-window cache. `repair(result) -> (result, changed)` may amend the metadata of a
    cached accepted window (used to re-run an unresolved cross-source parity check); pixels are never
    rewritten and the array checksum is unchanged."""
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, array_path = directory / (key+'.json'), directory / (key+'.npz')
    previous = read_json(meta_path)
    if previous:
        if 'reject' in previous:
            # Retry transient failures on a rerun; stable quality exclusions are cached.
            if not previous['reject']['reason'].startswith('technical_'):
                return previous
        elif array_path.exists() and sha256_file(array_path) == previous.get('array_sha256'):
            with np.load(array_path, allow_pickle=False) as z:
                result = {**{k: z[k] for k in ('b10', 'b20', 'scl')}, 'meta': previous['meta']}
            if repair is not None:
                result, changed = repair(result)
                if changed:
                    atomic_write_json(meta_path, {'meta': result['meta'], 'array_sha256': previous['array_sha256']})
            return result
        else:
            raise ValueError('cached window checksum mismatch')
    result = fn()
    if 'reject' in result:
        atomic_write_json(meta_path, result)
    else:
        with atomic_path(array_path) as tmp:
            with open(tmp, 'wb') as stream:
                np.savez(stream, **{k: result[k] for k in ('b10', 'b20', 'scl')})
        atomic_write_json(meta_path, {'meta': result['meta'], 'array_sha256': sha256_file(array_path)})
    return result
