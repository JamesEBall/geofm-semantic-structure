#!/usr/bin/env python3
"""External data/service access checks for the vocab study, run on a COMPUTE node through
the per-job SOCKS tunnel (lib/tunnel.sh). Writes a JSON report (pass/fail, timings,
versions, capabilities) and exits 1 if (a) PC STAC, (b) PC SAS + COG reads or (e) a dated
OSM planet + md5 fail.

  a pc_stac      Planetary Computer STAC search: sortby/query/fields/CQL2 filter/pagination,
                 property names, asset keys + asset-level proj/raster metadata
  b pc_cog       PC SAS token lifetime; rasterio windowed reads (B02/B11/SCL); GDAL proxy
                 forms (subprocesses); multi-threaded windowed-read benchmark
  c earth_search Element84 sentinel-2-c1-l2a search + blue/scl reads, raster:bands scale/offset
  d taginfo      two API calls with a descriptive User-Agent
  e osm_planet   planet-latest + preferred dated snapshot on several mirrors (+ .md5),
                 single/8-range throughput via curl; RESOLVE ecoregions + Geofabrik HEADs
  f reachability huggingface.co, pypi simple/osmium; negative control without proxy

No credentials are used. SAS tokens are kept in memory only and redacted from all output.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import email.utils
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

UA = "geofm-vocab-study/0.1 (research; contact via github JamesEBall)"
BBOX = [-3.4, 55.85, -3.0, 56.05]
DATETIME = "2025-05-01T00:00:00Z/2025-09-30T23:59:59Z"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
ES_STAC = "https://earth-search.aws.element84.com/v1"
S2_ASSETS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]
PC_PROPS = ["s2:processing_baseline", "proj:epsg", "proj:code", "s2:mgrs_tile", "eo:cloud_cover",
            "s2:snow_ice_percentage", "s2:nodata_pixel_percentage", "s2:product_uri", "platform"]
GDAL_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE",
}
PREFERRED_PLANET = "planet-251229.osm.pbf"
PROXY_VARS = {"all_proxy", "https_proxy", "http_proxy", "gdal_http_proxy", "gdal_https_proxy"}
MB = 1e6

_SAS_URL = re.compile(r"(\.blob\.core\.windows\.net/[^?\s\"']*)\?[^\s\"']*")
_SAS_SIG = re.compile(r"\b(sig|skoid|sktid|skt|ske|sks|skv|st|se)=[^&\s\"']+")


def redact(x) -> str:
    return _SAS_SIG.sub(r"\1=REDACTED", _SAS_URL.sub(r"\1?<SAS>", str(x)))


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))], 1)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_session(retries: int = 3):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.headers["User-Agent"] = UA
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None,
                  respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def rate_headers(headers) -> dict:
    keys = ("ratelimit", "rate-limit", "retry-after", "quota", "throttl")
    return {k: v for k, v in headers.items() if any(x in k.lower() for x in keys)}


# --------------------------------------------------------------------------- (a) PC STAC
def sec_pc_stac(s, ctx: dict) -> dict:
    out: dict = {}
    t0 = time.perf_counter()
    r = s.get(PC_STAC, timeout=30)
    r.raise_for_status()
    out["landing_ms"] = ms(t0)
    conf = r.json().get("conformsTo") or s.get(PC_STAC + "/conformance", timeout=30).json().get("conformsTo", [])

    def has(frag: str) -> bool:
        return any(frag in c for c in conf)

    out["conformance"] = {"item_search": has("item-search"), "fields": has("#fields"), "sort": has("#sort"),
                          "query": has("#query"), "filter": has("#filter"), "cql2_json": has("cql2-json"),
                          "cql2_text": has("cql2-text"), "context": has("context")}
    out["conformance_classes"] = sorted(conf)

    base = {"collections": ["sentinel-2-l2a"], "bbox": BBOX, "datetime": DATETIME,
            "query": {"eo:cloud_cover": {"lt": 20}}, "limit": 100}
    sort_field, fc, lat, errors = None, None, [], {}
    for field in ("eo:cloud_cover", "properties.eo:cloud_cover"):
        body = dict(base, sortby=[{"field": field, "direction": "asc"}])
        t0 = time.perf_counter()
        r = s.post(PC_STAC + "/search", json=body, timeout=60)
        if r.status_code == 200:
            sort_field, fc = field, r.json()
            lat.append(ms(t0))
            break
        errors[field] = f"{r.status_code}: {r.text[:200]}"
    if fc is None:
        raise RuntimeError(f"PC search failed: {errors}")
    body = dict(base, sortby=[{"field": sort_field, "direction": "asc"}])
    for _ in range(2):
        t0 = time.perf_counter()
        s.post(PC_STAC + "/search", json=body, timeout=60).raise_for_status()
        lat.append(ms(t0))
    feats = fc["features"]
    cc = [f["properties"].get("eo:cloud_cover") for f in feats]
    nxt = [l for l in fc.get("links", []) if l.get("rel") == "next"]
    out["search"] = {
        "request": body, "latency_ms": lat, "n_returned": len(feats),
        "number_matched": fc.get("numberMatched", (fc.get("context") or {}).get("matched")),
        "sortby_field_accepted": sort_field, "sortby_errors": errors,
        "sorted_ascending": cc == sorted(cc),
        "sorted_by_integer_part": [int(c) for c in cc] == sorted(int(c) for c in cc),
        "query_respected": all(c is not None and c < 20 for c in cc),
        "has_next_link": bool(nxt), "response_bytes": len(r.content),
    }
    if not feats:
        raise RuntimeError("PC search returned no items")

    props = [f["properties"] for f in feats]
    out["property_presence"] = {k: f"{sum(k in p for p in props)}/{len(props)}" for k in PC_PROPS}
    for k, name in (("s2:processing_baseline", "processing_baselines"), ("s2:mgrs_tile", "mgrs_tiles"),
                    ("proj:epsg", "proj_epsg_values"), ("proj:code", "proj_code_values"), ("platform", "platforms")):
        out[name] = dict(collections.Counter(str(p.get(k)) for p in props).most_common())
    out["datetime_range"] = [min(p["datetime"] for p in props), max(p["datetime"] for p in props)]
    first = feats[0]
    out["example_item_id"] = first["id"]
    out["item_property_keys"] = sorted(first["properties"])
    out["stac_extensions"] = first.get("stac_extensions")
    out["asset_keys"] = sorted(first["assets"])
    out["s2_assets_missing"] = [k for k in S2_ASSETS if k not in first["assets"]]
    assets = {}
    for k in S2_ASSETS:
        a = first["assets"].get(k)
        if a:
            assets[k] = {"type": a.get("type"), "roles": a.get("roles"), "gsd": a.get("gsd"),
                         "proj:shape": a.get("proj:shape"), "proj:transform": a.get("proj:transform"),
                         "proj:bbox": a.get("proj:bbox"), "proj:epsg": a.get("proj:epsg"),
                         "proj:code": a.get("proj:code"), "raster:bands": a.get("raster:bands"),
                         "eo:bands": [b.get("name") for b in a.get("eo:bands", [])],
                         "href_host": urlsplit(a["href"]).netloc}
    out["assets"] = assets

    # Pagination with small pages, following rel=next exactly as advertised.
    pbody = dict(body, limit=10)
    ids, page_lat, link_info = [], [], None
    t0 = time.perf_counter()
    page = s.post(PC_STAC + "/search", json=pbody, timeout=60).json()
    page_lat.append(ms(t0))
    for i in range(4):
        ids += [f["id"] for f in page["features"]]
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if i == 0:
            link_info = None if link is None else {
                "method": link.get("method", "GET"), "merge": link.get("merge"),
                "body_keys": sorted(link.get("body") or {}), "href_has_token": "token" in link.get("href", "")}
        if link is None or i == 3:
            break
        t0 = time.perf_counter()
        if link.get("method", "GET").upper() == "POST":
            b = link.get("body") or {}
            b = {**pbody, **b} if link.get("merge") else b
            page = s.post(link["href"], json=b, timeout=60).json()
        else:
            page = s.get(link["href"], timeout=60).json()
        page_lat.append(ms(t0))
    all_ids = [f["id"] for f in feats]
    out["pagination"] = {"page_limit": 10, "pages_fetched": len(page_lat), "ids": len(ids),
                         "unique_ids": len(set(ids)), "subset_of_limit100_result": set(ids) <= set(all_ids),
                         "same_order_as_limit100": ids == all_ids[:len(ids)], "next_link": link_info,
                         "page_latency_ms": page_lat}

    # fields extension
    t0 = time.perf_counter()
    rf = s.post(PC_STAC + "/search", json=dict(body, limit=5, fields={
        "include": ["id", "properties.eo:cloud_cover", "properties.s2:mgrs_tile"],
        "exclude": ["assets", "links", "geometry"]}), timeout=60)
    f_lat = ms(t0)
    rfull = s.post(PC_STAC + "/search", json=dict(body, limit=5), timeout=60)
    f0 = (rf.json().get("features") or [{}])[0] if rf.status_code == 200 else {}
    out["fields_extension"] = {
        "status": rf.status_code, "latency_ms": f_lat, "returned_keys": sorted(f0),
        "returned_property_keys": sorted(f0.get("properties", {})),
        "assets_excluded": "assets" not in f0, "bytes_with_fields": len(rf.content), "bytes_without": len(rfull.content),
        "supported": rf.status_code == 200 and "assets" not in f0 and "eo:cloud_cover" in f0.get("properties", {}),
    }

    # CQL2 filter (JSON via POST, text via GET)
    tile = collections.Counter(p.get("s2:mgrs_tile") for p in props).most_common(1)[0][0]
    expect = sum(1 for p in props if p.get("s2:mgrs_tile") == tile)
    cql = {"op": "and", "args": [{"op": "<", "args": [{"property": "eo:cloud_cover"}, 20]},
                                 {"op": "=", "args": [{"property": "s2:mgrs_tile"}, tile]}]}
    fb = {"collections": ["sentinel-2-l2a"], "bbox": BBOX, "datetime": DATETIME, "limit": 100,
          "filter-lang": "cql2-json", "filter": cql}
    t0 = time.perf_counter()
    rj = s.post(PC_STAC + "/search", json=fb, timeout=60)
    j_lat = ms(t0)
    jf = rj.json().get("features", []) if rj.status_code == 200 else []
    t0 = time.perf_counter()
    rt = s.get(PC_STAC + "/search", params={
        "collections": "sentinel-2-l2a", "bbox": ",".join(map(str, BBOX)), "datetime": DATETIME, "limit": 100,
        "filter-lang": "cql2-text", "filter": f"\"eo:cloud_cover\" < 20 AND \"s2:mgrs_tile\" = '{tile}'"}, timeout=60)
    t_lat = ms(t0)
    tf = rt.json().get("features", []) if rt.status_code == 200 else []

    def matches(fs):
        return bool(fs) and all(f["properties"].get("s2:mgrs_tile") == tile and f["properties"]["eo:cloud_cover"] < 20 for f in fs)

    out["cql2_filter"] = {
        "tile": tile, "expected_n": expect,
        "json_post": {"status": rj.status_code, "n": len(jf), "all_match": matches(jf), "latency_ms": j_lat,
                      "error": None if rj.status_code == 200 else rj.text[:200]},
        "text_get": {"status": rt.status_code, "n": len(tf), "all_match": matches(tf), "latency_ms": t_lat,
                     "error": None if rt.status_code == 200 else rt.text[:200]},
        "supported": rj.status_code == 200 and matches(jf) and len(jf) == expect,
    }
    ctx["pc_items"] = feats
    present = lambda k: out["property_presence"][k].split("/")[0] != "0"  # noqa: E731
    out["ok"] = (out["search"]["query_respected"] and not out["s2_assets_missing"]
                 and present("s2:mgrs_tile") and present("eo:cloud_cover")
                 and (present("proj:epsg") or present("proj:code")))
    return out


# --------------------------------------------------------------------------- (b) PC SAS + COG
def describe(ds) -> dict:
    st = ds.tags(ns="IMAGE_STRUCTURE")
    return {"width": ds.width, "height": ds.height, "count": ds.count, "dtype": ds.dtypes[0],
            "nodata": ds.nodata, "block_shape": list(ds.block_shapes[0]), "overviews": ds.overviews(1),
            "n_overviews": len(ds.overviews(1)), "compression": ds.profile.get("compress"),
            "predictor": st.get("PREDICTOR"), "layout": st.get("LAYOUT"), "interleave": ds.profile.get("interleave"),
            "tiled": ds.profile.get("tiled"), "crs": ds.crs.to_string() if ds.crs else None,
            "res": list(ds.res), "scales": list(ds.scales), "offsets": list(ds.offsets),
            "units": list(ds.units), "dataset_tags": dict(list(ds.tags().items())[:20]),
            "band_tags": dict(list(ds.tags(1).items())[:20])}


def read_window(url: str, size: int, off: int) -> dict:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    t0 = time.perf_counter()
    with rasterio.open(url) as ds:
        t_open = ms(t0)
        t1 = time.perf_counter()
        a = ds.read(1, window=Window(off, off, size, size))
        t_read = ms(t1)
        t2 = time.perf_counter()
        ds.read(1, window=Window(off + size, off, size, size))
        t_adj = ms(t2)
        info = describe(ds)
    nod = info["nodata"]
    info.update({"window_col_row_w_h": [off, off, size, size], "open_ms": t_open, "read_ms": t_read,
                 "adjacent_window_read_ms": t_adj, "total_ms": ms(t0), "array_shape": list(a.shape),
                 "min": int(a.min()), "max": int(a.max()), "mean": round(float(a.mean()), 2),
                 "nodata_fraction": round(float(np.mean(a == nod)), 4) if nod is not None else None})
    return info


def gdal_probe() -> int:
    """Subprocess mode: read a JSON {"url","window"} from stdin and try one windowed read."""
    req = json.loads(sys.stdin.read())
    t0 = time.perf_counter()
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(req["url"]) as ds:
            a = ds.read(1, window=Window(*req["window"]))
        print(json.dumps({"ok": True, "ms": ms(t0), "shape": list(a.shape)}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "ms": ms(t0), "error": redact(f"{type(e).__name__}: {e}")[:300]}))
    return 0


def proxy_forms(url: str) -> dict:
    port = os.environ.get("VOCAB_SOCKS_PORT")
    if not port:
        return {"skipped": "no VOCAB_SOCKS_PORT (not running behind the tunnel)"}
    base = {k: v for k, v in os.environ.items() if k.lower() not in PROXY_VARS}
    base.update(GDAL_HTTP_TIMEOUT="20", GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_MAX_RETRY="0")
    h = f"127.0.0.1:{port}"
    variants = {
        "GDAL_HTTP_PROXY=socks5h://127.0.0.1:PORT": {"GDAL_HTTP_PROXY": f"socks5h://{h}"},
        "GDAL_HTTP_PROXY=socks5://127.0.0.1:PORT": {"GDAL_HTTP_PROXY": f"socks5://{h}"},
        "GDAL_HTTP_PROXY=127.0.0.1:PORT (no scheme)": {"GDAL_HTTP_PROXY": h},
        "env https_proxy=socks5h:// only": {"https_proxy": f"socks5h://{h}"},
        "env ALL_PROXY=socks5h:// only": {"ALL_PROXY": f"socks5h://{h}"},
        "tunnel.sh exports": {k: v for k, v in os.environ.items() if k.lower() in PROXY_VARS},
        "no proxy (negative control)": {},
    }
    res = {}
    for name, extra in variants.items():
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--gdal-probe"],
                               input=json.dumps({"url": url, "window": [512, 512, 64, 64]}),
                               env={**base, **extra}, capture_output=True, text=True, timeout=120)
            res[name] = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {
                "ok": False, "error": redact(p.stderr[-300:])}
        except subprocess.TimeoutExpired:
            res[name] = {"ok": False, "error": "timeout 120 s"}
    return res


def bench(items_urls: list[str], layouts: list[dict], threads: int, n_windows: int, used: set,
          rng: random.Random, win: int = 256) -> dict:
    import rasterio
    from rasterio.windows import Window

    cands = [(i, r, c) for i, lay in enumerate(layouts) for r in range(lay["nbr"]) for c in range(lay["nbc"])
             if (i, r, c) not in used]
    rng.shuffle(cands)
    tasks = []
    for i, r, c in cands[:n_windows]:
        lay = layouts[i]
        bh, bw = lay["block"]
        h_avail = min(bh, lay["h"] - r * bh)
        w_avail = min(bw, lay["w"] - c * bw)
        if h_avail < win or w_avail < win:
            continue
        sy = rng.choice(range(0, h_avail - win + 1, win))
        sx = rng.choice(range(0, w_avail - win + 1, win))
        used.add((i, r, c))
        tasks.append((i, Window(c * bw + sx, r * bh + sy, win, win), lay["block_bytes"].get((r, c), 0)))
    tl = threading.local()
    lock = threading.Lock()
    opened: list = []

    def work(t):
        i, w, _ = t
        cache = getattr(tl, "ds", None)
        if cache is None:
            cache = tl.ds = {}
        t0 = time.perf_counter()
        ds = cache.get(i)
        if ds is None:
            ds = cache[i] = rasterio.open(items_urls[i])
            with lock:
                opened.append(ds)
        a = ds.read(1, window=w)
        return ms(t0), a.nbytes

    lat, nbytes, errs = [], 0, []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(threads) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in cf.as_completed(futs):
            try:
                l, b = f.result()
                lat.append(l)
                nbytes += b
            except Exception as e:  # noqa: BLE001
                errs.append(redact(f"{type(e).__name__}: {e}")[:200])
    wall = time.perf_counter() - t0
    for ds in opened:
        ds.close()
    net = sum(t[2] for t in tasks)
    return {"threads": threads, "windows": len(tasks), "window_px": win, "items": len(items_urls),
            "dataset_opens": len(opened), "errors": len(errs), "first_errors": errs[:3],
            "wall_s": round(wall, 2), "windows_per_s": round(len(lat) / wall, 1),
            "decoded_MB_per_s": round(nbytes / MB / wall, 1),
            "compressed_block_MB": round(net / MB, 1), "network_MB_per_s_est": round(net / MB / wall, 1),
            "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95), "latency_ms_max": pct(lat, 100)}


def sec_pc_cog(s, ctx: dict, args) -> dict:
    os.environ.update(GDAL_OPTS)
    import rasterio

    out: dict = {"gdal_options": GDAL_OPTS, "gdal_version": rasterio.__gdal_version__,
                 "rasterio_version": rasterio.__version__}
    t0 = time.perf_counter()
    r = s.get(PC_SAS, timeout=30)
    r.raise_for_status()
    lat1 = ms(t0)
    j = r.json()
    tok = ctx["sas_token"] = j["token"]
    q = dict(parse_qsl(tok))
    server_now = email.utils.parsedate_to_datetime(r.headers["Date"])
    expiry = dt.datetime.fromisoformat(j["msft:expiry"].replace("Z", "+00:00"))
    t0 = time.perf_counter()
    r2 = s.get(PC_SAS, timeout=30)
    lat2 = ms(t0)
    out["sas"] = {
        "status": r.status_code, "latency_ms": [lat1, lat2], "msft:expiry": j["msft:expiry"],
        "server_date": r.headers["Date"], "lifetime_minutes": round((expiry - server_now).total_seconds() / 60, 1),
        "second_call_same_expiry": r2.json().get("msft:expiry") == j["msft:expiry"],
        "token_param_names": sorted(q), "sp": q.get("sp"), "sr": q.get("sr"), "sv": q.get("sv"),
        "rate_limit_headers": rate_headers(r.headers), "response_header_names": sorted(r.headers),
    }
    items = ctx.get("pc_items") or []
    if not items:
        raise RuntimeError("no PC items from section a")

    def sign(href: str) -> str:
        return href + ("&" if "?" in href else "?") + tok

    item = items[0]
    out["read_item"] = item["id"]
    out["asset_host"] = urlsplit(item["assets"]["B02"]["href"]).netloc
    try:
        socket.getaddrinfo(out["asset_host"], 443)
        out["compute_node_resolves_external_dns"] = True
    except OSError:
        out["compute_node_resolves_external_dns"] = False
    h = s.head(sign(item["assets"]["B02"]["href"]), timeout=30)
    out["B02_head"] = {"status": h.status_code, "bytes": int(h.headers.get("Content-Length", 0)),
                       "accept_ranges": h.headers.get("Accept-Ranges"), "server": h.headers.get("Server")}
    reads = {}
    for key, size, off in (("B02", 256, 5120), ("B11", 128, 2560), ("SCL", 128, 2560)):
        try:
            reads[key] = read_window(sign(item["assets"][key]["href"]), size, off)
        except Exception as e:  # noqa: BLE001
            reads[key] = {"ok": False, "error": redact(f"{type(e).__name__}: {e}")}
    out["reads"] = reads
    out["proxy_forms"] = proxy_forms(sign(item["assets"]["B02"]["href"]))

    if args.skip_heavy:
        out["benchmark"] = {"skipped": True}
    else:
        bitems = items[:args.bench_items]
        urls = [sign(it["assets"]["B02"]["href"]) for it in bitems]
        layouts = []
        for u in urls:
            with rasterio.open(u) as ds:
                bh, bw = ds.block_shapes[0]
                nbr, nbc = -(-ds.height // bh), -(-ds.width // bw)
                bb = {}
                for rr in range(nbr):
                    for cc in range(nbc):
                        try:
                            bb[(rr, cc)] = ds.block_size(1, rr, cc)
                        except Exception:  # noqa: BLE001
                            pass
                layouts.append({"h": ds.height, "w": ds.width, "block": (bh, bw), "nbr": nbr, "nbc": nbc,
                                "block_bytes": bb})
        out["benchmark_items"] = [it["id"] for it in bitems]
        out["benchmark_block_bytes_mean"] = round(
            sum(sum(l["block_bytes"].values()) for l in layouts) / max(1, sum(len(l["block_bytes"]) for l in layouts)))
        rng = random.Random(20261016)
        used: set = set()
        plan = [tuple(int(x) for x in p.split(":")) for p in args.bench.split(",")]
        out["benchmark"] = {f"threads_{t}": bench(urls, layouts, t, n, used, rng) for t, n in plan}
    b32 = out["benchmark"].get("threads_32", {})
    rec = out["proxy_forms"].get("GDAL_HTTP_PROXY=socks5h://127.0.0.1:PORT", {"ok": True})
    out["ok"] = (all(v.get("ok", True) for v in reads.values()) and rec.get("ok", False)
                 and out["sas"]["lifetime_minutes"] > 0
                 and (args.skip_heavy or (b32.get("windows", 0) > 0 and b32.get("errors", 1) <= 0.05 * b32["windows"])))
    return out


# --------------------------------------------------------------------------- (c) Earth Search
def sec_earth_search(s, ctx: dict) -> dict:
    out: dict = {}
    base = {"collections": ["sentinel-2-c1-l2a"], "bbox": BBOX, "datetime": DATETIME,
            "query": {"eo:cloud_cover": {"lt": 20}}, "limit": 50}
    fc, lat = None, None
    for field in ("eo:cloud_cover", "properties.eo:cloud_cover"):
        t0 = time.perf_counter()
        r = s.post(ES_STAC + "/search", json=dict(base, sortby=[{"field": field, "direction": "asc"}]), timeout=60)
        if r.status_code == 200:
            fc, lat, out["sortby_field_accepted"] = r.json(), ms(t0), field
            break
    if fc is None:
        raise RuntimeError(f"Earth Search search failed: {r.status_code} {r.text[:200]}")
    feats = fc["features"]
    out.update({"latency_ms": lat, "n_returned": len(feats), "number_matched": fc.get("numberMatched")})
    if not feats:
        raise RuntimeError("Earth Search returned no items")
    it = feats[0]
    p = it["properties"]
    out["example_item_id"] = it["id"]
    out["asset_keys"] = sorted(it["assets"])
    out["processing_baselines"] = dict(collections.Counter(str(f["properties"].get("s2:processing_baseline")) for f in feats))
    out["selected_properties"] = {k: p.get(k) for k in ("s2:processing_baseline", "earthsearch:boa_offset_applied",
                                                        "proj:epsg", "proj:code", "mgrs:utm_zone", "mgrs:grid_square",
                                                        "s2:mgrs_tile", "eo:cloud_cover")}
    out["raster_bands"] = {k: (it["assets"].get(k) or {}).get("raster:bands") for k in ("blue", "red", "nir", "swir16", "scl")}
    reads = {}
    for key, size, off in (("blue", 256, 5120), ("scl", 128, 2560)):
        reads[key] = read_window(it["assets"][key]["href"], size, off)
    out["reads"] = reads
    # Same acquisition on PC? Compare raw DN of PC B02 vs ES blue on the same 10 m window.
    pc = ctx.get("pc_items") or []
    tile = p.get("s2:mgrs_tile") or f"{p.get('mgrs:utm_zone')}{p.get('mgrs:latitude_band')}{p.get('mgrs:grid_square')}"
    match = next((f for f in pc if f["properties"].get("s2:mgrs_tile") == tile
                  and f["properties"]["datetime"][:10] == p["datetime"][:10]), None)
    if match is not None and "sas_token" not in ctx:
        ctx["sas_token"] = s.get(PC_SAS, timeout=30).json()["token"]
    if match is not None:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        w = Window(5120, 5120, 256, 256)
        with rasterio.open(match["assets"]["B02"]["href"] + "?" + ctx["sas_token"]) as a, \
                rasterio.open(it["assets"]["blue"]["href"]) as b:
            x, y = a.read(1, window=w).astype("int32"), b.read(1, window=w).astype("int32")
            same_grid = a.transform == b.transform and a.crs == b.crs
        valid = (x > 0) & (y > 0)
        out["pc_vs_es_same_scene"] = {
            "pc_item": match["id"], "es_item": it["id"], "same_grid": same_grid,
            "pc_baseline": match["properties"].get("s2:processing_baseline"),
            "es_baseline": p.get("s2:processing_baseline"),
            "identical_fraction": round(float(np.mean(x[valid] == y[valid])), 4) if valid.any() else None,
            "mean_pc_minus_es": round(float(np.mean(x[valid] - y[valid])), 2) if valid.any() else None}
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- (d) taginfo
def sec_taginfo(s, ctx: dict) -> dict:
    out: dict = {"user_agent": UA}
    calls = {"wiki_pages_landuse_farmland": ("https://taginfo.openstreetmap.org/api/4/tag/wiki_pages",
                                             {"key": "landuse", "value": "farmland"}),
             "key_values_crop": ("https://taginfo.openstreetmap.org/api/4/key/values",
                                 {"key": "crop", "page": 1, "rp": 5})}
    for name, (url, params) in calls.items():
        lats, r = [], None
        for _ in range(3):
            t0 = time.perf_counter()
            r = s.get(url, params=params, timeout=30)
            lats.append(ms(t0))
            time.sleep(0.5)
        j = r.json() if r.status_code == 200 else {}
        data = j.get("data") or []
        sample = ([(d.get("value"), d.get("count")) for d in data[:5]] if name.startswith("key_values")
                  else sorted({d.get("lang") for d in data})[:12])
        out[name] = {"status": r.status_code, "latency_ms": lats, "n_data": len(data), "total": j.get("total"),
                     "data_until": j.get("data_until"), "sample": sample, "rate_limit_headers": rate_headers(r.headers)}
    out["ok"] = all(out[k]["status"] == 200 and out[k]["n_data"] > 0 for k in calls)
    return out


# --------------------------------------------------------------------------- (e) OSM planet etc.
def head(s, url: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = s.head(url, allow_redirects=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    n = r.headers.get("Content-Length")
    return {"url": url, "status": r.status_code, "ok": r.status_code == 200, "latency_ms": ms(t0),
            "final_url": r.url if r.url != url else None, "redirects": [h.status_code for h in r.history],
            "bytes": int(n) if n and n.isdigit() else None, "last_modified": r.headers.get("Last-Modified"),
            "accept_ranges": r.headers.get("Accept-Ranges"), "etag": r.headers.get("ETag")}


def get_md5(s, url: str) -> dict:
    try:
        r = s.get(url, timeout=30)
    except Exception as e:  # noqa: BLE001
        return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    m = re.match(r"^([0-9a-fA-F]{32})\s+\*?(\S+)", r.text.strip()) if r.status_code == 200 else None
    return {"url": url, "status": r.status_code, "ok": bool(m), "md5": m.group(1).lower() if m else None,
            "file": m.group(2) if m else None}


def planet_mirrors(fname: str) -> dict:
    year = "20" + fname[7:9]
    return {
        "planet.openstreetmap.org": f"https://planet.openstreetmap.org/pbf/{fname}",
        "osmf-s3-eu-central-1": f"https://osm-planet-eu-central-1.s3.dualstack.eu-central-1.amazonaws.com/planet/pbf/{year}/{fname}",
        "osmf-s3-us-west-2": f"https://osm-planet-us-west-2.s3.dualstack.us-west-2.amazonaws.com/planet/pbf/{year}/{fname}",
        "ftp.fau.de": f"https://ftp.fau.de/osm-planet/pbf/{fname}",
        "ftp5.gwdg.de": f"https://ftp5.gwdg.de/pub/misc/openstreetmap/planet.openstreetmap.org/pbf/{fname}",
    }


def curl_range(url: str, start: int, length: int, proxy: str | None) -> dict:
    cmd = ["curl", "-s", "-L", "-o", "/dev/null", "-m", "180", "-A", UA, "-r", f"{start}-{start + length - 1}",
           "-w", "%{http_code} %{size_download} %{time_total} %{time_starttransfer}"]
    if proxy:
        cmd += ["--proxy", proxy]
    p = subprocess.run(cmd + [url], capture_output=True, text=True)
    try:
        code, size, total, ttfb = p.stdout.split()
        return {"rc": p.returncode, "http": int(code), "bytes": int(float(size)), "s": float(total), "ttfb_s": float(ttfb)}
    except ValueError:
        return {"rc": p.returncode, "http": 0, "bytes": 0, "s": 0.0, "ttfb_s": None, "stderr": p.stderr[-200:]}


def throughput(url: str, size: int, proxy: str | None, rng: random.Random) -> dict:
    chunk = 32 * 2**20
    start = rng.randrange(2**30, max(2**30 + 1, size - 2**30), 2**20)
    one = curl_range(url, start, 2 * chunk, proxy)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(8) as ex:
        par = list(ex.map(lambda i: curl_range(url, start + (2 + i) * chunk, chunk, proxy), range(8)))
    wall = time.perf_counter() - t0
    pbytes = sum(r["bytes"] for r in par)
    return {"single_stream_64MB": {**one, "MB_per_s": round(one["bytes"] / MB / one["s"], 1) if one["s"] else 0},
            "parallel_8x32MB": {"wall_s": round(wall, 2), "bytes": pbytes, "MB_per_s": round(pbytes / MB / wall, 1),
                                "http_codes": sorted({r["http"] for r in par}),
                                "all_complete": all(r["bytes"] == chunk for r in par)}}


def sec_osm_planet(s, ctx: dict, args) -> dict:
    out: dict = {}
    latest = head(s, "https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf")
    out["planet_latest_head"] = latest
    latest_name = os.path.basename(urlsplit(latest.get("final_url") or "").path) or None
    out["planet_latest_dated"] = latest_name
    snaps = {}
    for fname in dict.fromkeys(n for n in (PREFERRED_PLANET, latest_name) if n):
        mir = {}
        for name, url in planet_mirrors(fname).items():
            mir[name] = {"head": head(s, url), "md5": get_md5(s, url + ".md5")}
        good = {k: v for k, v in mir.items() if v["head"]["ok"] and v["md5"]["ok"]}
        md5s = {v["md5"]["md5"] for v in good.values()}
        sizes = {v["head"]["bytes"] for v in good.values()}
        snaps[fname] = {"mirrors": mir, "available_on": sorted(good), "md5": md5s.pop() if len(md5s) == 1 else sorted(md5s),
                        "bytes": sizes.pop() if len(sizes) == 1 else sorted(s_ for s_ in sizes if s_),
                        "consistent": len({v["md5"]["md5"] for v in good.values()}) <= 1, "usable": bool(good)}
    out["snapshots"] = snaps
    out["not_available_checked"] = {u: head(s, u).get("status") for u in (
        f"https://planet.openstreetmap.org/pbf/2025/{PREFERRED_PLANET}",
        f"https://planet.openstreetmap.org/planet/2025/{PREFERRED_PLANET}",
        f"https://osm-pds.s3.amazonaws.com/2025/{PREFERRED_PLANET}")}
    best = PREFERRED_PLANET if snaps.get(PREFERRED_PLANET, {}).get("usable") else latest_name
    out["best_snapshot"] = best
    if best and snaps.get(best, {}).get("usable"):
        b = snaps[best]
        out["best"] = {"file": best, "md5": b["md5"], "bytes": b["bytes"], "mirrors": b["available_on"],
                       "url": planet_mirrors(best)["planet.openstreetmap.org"]}
    if not args.skip_heavy and best:
        rng = random.Random(1229)
        tp = {}
        proxy = os.environ.get("VOCAB_PROXY")
        for name, url in planet_mirrors(best).items():
            if name == "planet.openstreetmap.org" or name not in snaps[best]["available_on"]:
                continue
            tp[name] = throughput(url, snaps[best]["mirrors"][name]["head"]["bytes"] or 2**34, proxy, rng)
        if latest_name and latest_name != best and snaps[latest_name]["mirrors"]["ftp5.gwdg.de"]["head"]["ok"]:
            tp[f"ftp5.gwdg.de ({latest_name})"] = throughput(
                planet_mirrors(latest_name)["ftp5.gwdg.de"], snaps[latest_name]["mirrors"]["ftp5.gwdg.de"]["head"]["bytes"],
                proxy, rng)
        out["throughput_via_tunnel"] = tp
        if tp:
            out["fastest_mirror_parallel"] = max(tp, key=lambda k: tp[k]["parallel_8x32MB"]["MB_per_s"])
    out["resolve_ecoregions2017"] = head(s, "https://storage.googleapis.com/teow2016/Ecoregions2017.zip")
    lux = "https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf"
    out["geofabrik_luxembourg"] = {"head": head(s, lux), "md5": get_md5(s, lux + ".md5")}
    out["ok"] = any(v["usable"] for v in snaps.values())
    return out


# --------------------------------------------------------------------------- (f) reachability
def sec_reachability(s, ctx: dict) -> dict:
    import requests

    out: dict = {}
    for name, url, okset in (("huggingface", "https://huggingface.co", {200, 401}),
                             ("pypi_osmium", "https://pypi.org/simple/osmium/", {200})):
        t0 = time.perf_counter()
        try:
            r = s.get(url, timeout=30)
            out[name] = {"status": r.status_code, "ok": r.status_code in okset, "latency_ms": ms(t0)}
            if name == "pypi_osmium":
                out[name]["lists_4_3_1"] = "osmium-4.3.1" in r.text
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    direct = requests.Session()
    direct.trust_env = False
    t0 = time.perf_counter()
    try:
        r = direct.get(PC_STAC, timeout=(5, 5))
        out["direct_without_proxy"] = {"reachable": True, "status": r.status_code, "latency_ms": ms(t0)}
    except Exception as e:  # noqa: BLE001
        out["direct_without_proxy"] = {"reachable": False, "error": type(e).__name__, "latency_ms": ms(t0)}
    out["ok"] = out["huggingface"]["ok"] and out["pypi_osmium"]["ok"]
    return out


# --------------------------------------------------------------------------- main
def versions() -> dict:
    import importlib.metadata as md

    v = {"python": platform.python_version(), "executable": sys.executable}
    for p in ("requests", "urllib3", "PySocks", "rasterio", "numpy", "certifi"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    try:
        import rasterio
        v["gdal"] = rasterio.__gdal_version__
    except Exception:  # noqa: BLE001
        pass
    try:
        v["curl_cli"] = subprocess.run(["curl", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    return v


def write(report: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(redact(json.dumps(report, indent=2, default=str)) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="JSON report path")
    ap.add_argument("--sections", default="a,b,c,d,e,f")
    ap.add_argument("--skip-heavy", action="store_true", help="skip read benchmark and planet throughput")
    ap.add_argument("--bench", default="8:400,32:1200,64:1200",
                    help="threads:windows runs of the B02 windowed-read benchmark (one window per COG block)")
    ap.add_argument("--bench-items", type=int, default=12, help="number of PC items used by the benchmark")
    ap.add_argument("--tunnel-selftest", type=Path, help="JSON from access_check.sbatch to merge")
    ap.add_argument("--annotate", type=Path, help="merge --tunnel-selftest into an existing report and exit")
    ap.add_argument("--gdal-probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.gdal_probe:
        return gdal_probe()
    if args.annotate:
        rep = json.loads(args.annotate.read_text())
        extra = json.loads(args.tunnel_selftest.read_text())
        rep.setdefault("tunnel", {}).update(extra)
        write(rep, args.annotate)
        return 0
    if not args.out:
        ap.error("--out is required")

    report: dict = {
        "generated_utc": utcnow(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tunnel": {"socks_port": os.environ.get("VOCAB_SOCKS_PORT"), "proxy": os.environ.get("VOCAB_PROXY"),
                   "gdal_http_proxy": os.environ.get("GDAL_HTTP_PROXY"),
                   "cipher": os.environ.get("VOCAB_TUNNEL_CIPHER_USED"),
                   "login_host": os.environ.get("VOCAB_LOGIN_HOST", "10.10.0.21")},
        "versions": versions(), "sections": {},
    }
    if args.tunnel_selftest and args.tunnel_selftest.exists():
        report["tunnel"].update(json.loads(args.tunnel_selftest.read_text()))
    s = make_session()
    ctx: dict = {}
    funcs = {"a": ("a_pc_stac", lambda: sec_pc_stac(s, ctx)),
             "b": ("b_pc_sas_cog", lambda: sec_pc_cog(s, ctx, args)),
             "c": ("c_earth_search", lambda: sec_earth_search(s, ctx)),
             "d": ("d_taginfo", lambda: sec_taginfo(s, ctx)),
             "e": ("e_osm_planet", lambda: sec_osm_planet(s, ctx, args)),
             "f": ("f_reachability", lambda: sec_reachability(s, ctx))}
    wanted = [x.strip() for x in args.sections.split(",") if x.strip()]
    for key in wanted:
        name, fn = funcs[key]
        print(f"== {name} ({utcnow()})", flush=True)
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": redact(f"{type(e).__name__}: {e}"),
                   "traceback": redact(traceback.format_exc()[-2000:])}
        res["section_seconds"] = round(time.perf_counter() - t0, 1)
        report["sections"][name] = res
        print(f"   {name}: {'PASS' if res.get('ok') else 'FAIL'} in {res['section_seconds']} s"
              + (f" -- {res['error']}" if res.get("error") else ""), flush=True)
        write(report, args.out)

    secs = report["sections"]
    required = [n for k, (n, _) in funcs.items() if k in ("a", "b", "e") and k in wanted]
    report["required_sections"] = required
    report["ok"] = all(secs[n].get("ok") for n in required)
    report["all_sections_ok"] = all(v.get("ok") for v in secs.values())
    report["finished_utc"] = utcnow()
    write(report, args.out)
    print(f"check_access: {'PASS' if report['ok'] else 'FAIL'} (required {required}); "
          f"all sections ok={report['all_sections_ok']} -> {args.out}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
