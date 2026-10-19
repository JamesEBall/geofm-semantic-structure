#!/usr/bin/env python3
"""Reference-only, additive support selection. Author: James Edward Ball."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import acquire as a

REGIONS = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']


def sphere(points):
    p = np.radians(points)
    return np.column_stack((np.cos(p[:, 1]) * np.cos(p[:, 0]),
                            np.cos(p[:, 1]) * np.sin(p[:, 0]), np.sin(p[:, 1])))


def far_from(points, references, km):
    if not len(references):
        return np.ones(len(points), dtype=bool)
    distances, _ = cKDTree(sphere(references)).query(sphere(points))
    return distances >= 2 * np.sin(km / (2 * 6371))


def candidates(spec, config):
    countries = json.loads((ROOT / 'data/countries-source.json').read_text())
    groups = []
    original_centres = [[b['lon'] + .5, b['lat'] + .5] for b in spec['blocks']]
    for region in REGIONS:
        land = a.prep(a.unary_union([a.shape(f['geometry']) for f in countries['features']
                                    if f['properties']['CONTINENT'] == region]))
        coords = [(lon, lat) for lon in range(-179, 179) for lat in range(-55, 75)
                  if land.contains(a.Point(lon + .5, lat + .5))]
        centres = np.asarray(coords, dtype=float) + .5
        allowed = far_from(centres, original_centres, config['minimum_block_distance_km'])
        allowed &= far_from(centres, spec['prior_points'], config['minimum_prior_distance_km'])
        coords = [p for p, ok in zip(coords, allowed) if ok]
        coords.sort(key=lambda p: a.digest(config['seed'] + str(p)))
        groups.append([dict(id=f'{lon}_{lat}', lon=lon, lat=lat, region=region)
                       for lon, lat in coords])
    return [block for row in itertools.zip_longest(*groups) for block in row if block]


class Selection:
    def __init__(self, original, config):
        self.original = original
        self.config = config
        self.additions = []
        self.counts = Counter()
        self.evidence = []

    def complete(self):
        return all(self.counts[(t['split'], t['code'])] >= t['blocks']
                   for t in self.config['targets'])

    def separated(self, block):
        centre = [block['lon'] + .5, block['lat'] + .5]
        return all(a.distance(centre, [b['lon'] + .5, b['lat'] + .5]) >=
                   self.config['minimum_block_distance_km']
                   for b in self.original + self.additions)

    def consider(self, block, result):
        if self.complete() or result['candidate_count'] != 200 or not self.separated(block):
            return False
        counts = {int(k): v for k, v in result['target_counts'].items()}
        matched = [t for t in self.config['targets']
                   if counts.get(t['code'], 0) >= t['minimum_candidates']]
        outstanding = [t for t in matched if self.counts[(t['split'], t['code'])] < t['blocks']]
        if not outstanding:
            return False
        split = outstanding[0]['split']
        added = dict(block, split=split)
        self.additions.append(added)
        supplied = []
        for target in matched:
            if target['split'] == split:
                self.counts[(split, target['code'])] += 1
                supplied.append(target['class'])
        self.evidence.append(dict(block=added, supplied=supplied,
                                  candidate_counts=counts,
                                  candidate_sha256=result['candidate_sha256']))
        if len(self.additions) > self.config['maximum_extra_blocks']:
            raise RuntimeError('Extra-block bound exceeded')
        return True

    def progress(self):
        return {f"{t['split']}:{t['class']}": self.counts[(t['split'], t['code'])]
                for t in self.config['targets']}


def screen(task):
    block, config, code_hash = task
    cache = ROOT / 'support/cache' / (block['id'] + '.json')
    if cache.exists():
        result = json.loads(cache.read_text())
        if result['acquire_sha256'] != code_hash:
            raise RuntimeError('Candidate code changed')
        return result
    with a.rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                        GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2',
                        GDAL_HTTP_CONNECTTIMEOUT='30', GDAL_HTTP_TIMEOUT='120',
                        GDAL_CACHEMAX=128*1024*1024,
                        CPL_VSIL_CURL_CACHE_SIZE=32*1024*1024):
        lon, lat = block['lon'], block['lat']
        with a.rasterio.open(a.wcurl(lon + .5, lat + .5)) as src:
            overview = src.read(1, window=a.from_bounds(lon+.15, lat+.15, lon+.85, lat+.85,
                                                       src.transform),
                                out_shape=(config['coarse_shape'], config['coarse_shape']),
                                resampling=a.Resampling.nearest)
        coarse = {code: int(np.count_nonzero(overview == code)) for code in [90, 50, 60]}
        points = a.candidate_points(block, 200) if any(
            v >= config['coarse_minimum_pixels'] for v in coarse.values()) else []
    result = dict(id=block['id'], acquire_sha256=code_hash, coarse_counts=coarse,
                  candidate_count=len(points), target_counts=dict(Counter(p['target_code'] for p in points)),
                  candidate_sha256=hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest(),
                  points=points)
    a.save(cache, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    for name, digest in json.loads((ROOT / 'support/preselection-freeze.json').read_text()).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Preselection freeze changed: ' + name)
    spec = json.loads((ROOT / 'data/blocks.json').read_text())
    config = json.loads((ROOT / 'support/targets.json').read_text())
    assert len(spec['blocks']) == config['original_blocks']
    code_hash = hashlib.sha256((ROOT / 'code/acquire.py').read_bytes()).hexdigest()
    sequence = candidates(spec, config)
    a.save(ROOT / 'support/candidate-sequence.json', sequence)
    selection = Selection(spec['blocks'], config)
    screened = 0
    # Batch size is fixed independently of worker count for replay.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for start in range(0, len(sequence), 96):
            batch = [b for b in sequence[start:start+96] if selection.separated(b)]
            results = list(pool.map(screen, [(b, config, code_hash) for b in batch]))
            for block, result in zip(batch, results):
                screened += 1
                if selection.consider(block, result):
                    print('ADDED', selection.additions[-1], selection.progress(), flush=True)
            status = dict(screened=screened, sequence_cursor=start+96, available=len(sequence),
                          selected=len(selection.additions), support=selection.progress(),
                          complete=selection.complete())
            a.save(ROOT / 'support/progress.json', status)
            print(json.dumps(status), flush=True)
            if selection.complete():
                break
    if not selection.complete():
        raise RuntimeError('Reference support targets could not be met: ' + str(selection.progress()))
    a.save(ROOT / 'support/selection.json', dict(author='James Edward Ball',
           original_block_count=len(spec['blocks']), screened=screened,
           selected=selection.additions, evidence=selection.evidence,
           support=selection.progress(), quality_gates_changed=False))


if __name__ == '__main__':
    main()
