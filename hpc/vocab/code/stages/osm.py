"""Immutable OSM acquisition, with verified byte ranges and a whole-file digest.

Author: James Edward Ball. A source checksum never substitutes for later label QA.
"""
from __future__ import annotations
import hashlib
import os
import re
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file
from mc.registry import Stage, StageFailed

SNAPSHOT = 'planet-251229.osm.pbf'
EXPECTED_BYTES = 90603886401
EXPECTED_MD5 = 'e9193927d12fe880ee8d6f99afca5eb1'


def fetch_plan(ctx):
    if ctx.cfg['osm']['preferred_snapshot'] != SNAPSHOT:
        raise StageFailed('Snapshot differs from validated immutable source; update provenance first')
    size = int(ctx.cfg['osm']['range_mib']) * 1024**2
    return [{'id': f'part-{i:05d}', 'start': start, 'stop': min(start+size, EXPECTED_BYTES)-1}
            for i, start in enumerate(range(0, EXPECTED_BYTES, size))]


def validate_range(response, start, stop):
    if response.status_code != 206:
        raise RuntimeError(f'Range server returned HTTP {response.status_code}; expected 206')
    got = response.headers.get('Content-Range', '')
    if got != f'bytes {start}-{stop}/{EXPECTED_BYTES}':
        raise RuntimeError('Incorrect Content-Range: ' + got)
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise RuntimeError('Compressed range response cannot be byte-verified')


def fetch_run(ctx, unit, ckpt):
    start, stop = unit['start'], unit['stop']
    dest = ctx.shared / 'osm_fetch' / 'parts' / (unit['id'] + '.bin')
    meta = dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        old = read_json(meta)
        if dest.stat().st_size == stop-start+1 and sha256_file(dest) == old['sha256']:
            return {'outputs': [dest, meta], 'metrics': {'cached': True}}
        raise StageFailed('Cached source range failed verification')
    session = ctx.requests_session()
    errors = []
    try:
        for mirror in ctx.cfg['osm']['mirrors']:
            ckpt.maybe_stop()
            url = mirror.rstrip('/') + '/' + SNAPSHOT
            try:
                with session.get(url, headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'},
                                 stream=True, timeout=(30,90)) as response:
                    validate_range(response, start, stop)
                    digest = hashlib.sha256(); count = 0
                    with atomic_path(dest) as tmp:
                        with open(tmp, 'wb') as out:
                            for data in response.iter_content(1024**2):
                                ckpt.maybe_stop()
                                out.write(data); digest.update(data); count += len(data)
                            if count != stop-start+1:
                                raise RuntimeError('Truncated range')
                    atomic_write_json(meta, {'url':url,'start':start,'stop':stop,'bytes':count,'sha256':digest.hexdigest()})
                return {'outputs':[dest,meta], 'metrics':{'bytes':count}}
            except (OSError, RuntimeError) as e:
                errors.append(type(e).__name__ + ': ' + str(e)[:120])
        raise RuntimeError('All immutable snapshot mirrors failed: ' + '; '.join(errors))
    finally:
        session.close()


def fetch_finalize(ctx):
    dest = ctx.shared / 'osm_fetch' / SNAPSHOT
    md5 = hashlib.md5(); sha = hashlib.sha256(); total = 0
    with atomic_path(dest) as tmp:
        with open(tmp, 'wb') as out:
            for unit in ctx.units():
                part = ctx.shared / 'osm_fetch' / 'parts' / (unit['id'] + '.bin')
                with open(part, 'rb') as src:
                    for data in iter(lambda: src.read(8*1024**2), b''):
                        out.write(data); md5.update(data); sha.update(data); total += len(data)
        if total != EXPECTED_BYTES or md5.hexdigest() != EXPECTED_MD5:
            raise StageFailed('Whole planet snapshot failed size/MD5 verification')
    report = {'snapshot': SNAPSHOT, 'path':str(dest),'bytes':total,'md5':md5.hexdigest(),
              'sha256':sha.hexdigest(), 'licence':'ODbL-1.0','attribution':'© OpenStreetMap contributors'}
    atomic_write_json(ctx.stage_dir() / 'source.json', report)
    return report


STAGES = [Stage('osm_fetch','P1','net',(),fetch_plan,fetch_run,fetch_finalize,
                unit_cpus=1,unit_mem_gb=2,est_unit_min=3)]

# Keep immutable acquisition paths/units stable while adding dependent processing.
from . import _osm_pipeline as pipeline

STAGES.extend([
    Stage('osm_filter','P1','himem',('osm_fetch',),pipeline.one,pipeline.filter_run,
          unit_cpus=8,unit_mem_gb=128,est_unit_min=120),
    Stage('osm_split','P1','himem',('osm_filter',),pipeline.one,pipeline.split_run,
          unit_cpus=8,unit_mem_gb=64,est_unit_min=60),
    Stage('osm_areas','P1','cpu',('osm_split',),pipeline.areas_plan,pipeline.areas_run,pipeline.areas_finalize,
          unit_cpus=2,unit_mem_gb=24,est_unit_min=25),
    Stage('osm_lattice','P1','cpu',('osm_areas',),pipeline.lattice_plan,pipeline.lattice_run,
          unit_cpus=2,unit_mem_gb=12,est_unit_min=15),
    Stage('inventory','P1','cpu',('osm_lattice',),pipeline.one,pipeline.inventory_run,
          unit_cpus=2,unit_mem_gb=8,est_unit_min=15),
    Stage('vocab','P1','cpu',('inventory',),pipeline.one,pipeline.vocab_run,
          unit_cpus=1,unit_mem_gb=4,est_unit_min=1),
    Stage('biome_fetch','P1','net',(),pipeline.biome_plan,pipeline.biome_run,
          unit_cpus=1,unit_mem_gb=2,est_unit_min=3),
    Stage('select','P1','cpu',('vocab','biome_fetch'),pipeline.one,pipeline.select_run,
          unit_cpus=4,unit_mem_gb=16,est_unit_min=20),
    Stage('labels_exact','P2','cpu',('assemble',),pipeline.labels_plan,pipeline.labels_run,pipeline.labels_finalize,
          unit_cpus=2,unit_mem_gb=16,est_unit_min=20),
    Stage('admit','P2','cpu',('labels_exact',),pipeline.one,pipeline.admit_run,
          unit_cpus=2,unit_mem_gb=8,est_unit_min=3),
])
