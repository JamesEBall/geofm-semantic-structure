"""Frozen encoder weights for the pretraining ladder (amendment A4), fetched inside Slurm.

Author: James Edward Ball. Each file streams through the job's SOCKS tunnel into
`shared/weights/<model>/`, is hashed, and is pinned either to a declared sha256 or, where the
upstream repository publishes no digest, to the sha256 and repository commit recorded on first
fetch. The stage manifest is covered by freeze F1, so the weights cannot change after that point.
Nothing is downloaded on the login node or on any machine outside the cluster.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file
from mc.registry import Stage, StageFailed

HF = 'https://huggingface.co'
# (model, filename, url, declared sha256 or None)
FILES = [
    ('prithvi_tl', 'Prithvi_EO_V2_300M_TL.pt',
     f'{HF}/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL/resolve/63adbd39c271da4c42f447e69b1a7c91a338cdc9/Prithvi_EO_V2_300M_TL.pt', None),
    ('prithvi_tl', 'config.json',
     f'{HF}/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL/resolve/63adbd39c271da4c42f447e69b1a7c91a338cdc9/config.json', None),
    ('dofa', 'DOFA_ViT_base_e100.pth', f'{HF}/earthflow/DOFA/resolve/7a5219e48d2f8848511b0fabea7920a8836bc480/DOFA_ViT_base_e100.pth', None),
    ('dinov2', 'config.json', f'{HF}/facebook/dinov2-base/resolve/f9e44c814b77203eaa57a6bdbbd535f21ede1415/config.json', None),
    ('dinov2', 'model.safetensors', f'{HF}/facebook/dinov2-base/resolve/f9e44c814b77203eaa57a6bdbbd535f21ede1415/model.safetensors', None),
    ('dinov2', 'preprocessor_config.json', f'{HF}/facebook/dinov2-base/resolve/f9e44c814b77203eaa57a6bdbbd535f21ede1415/preprocessor_config.json', None),
    ('remoteclip', 'RemoteCLIP-ViT-B-32.pt',
     f'{HF}/chendelong/RemoteCLIP/resolve/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt',
     '60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4'),
]
LICENCES = {'prithvi_tl': 'Apache-2.0 (IBM/NASA)', 'dofa': 'Apache-2.0 (Zhu lab, TUM; repository earthflow/DOFA)',
            'dinov2': 'Apache-2.0 (Meta AI)', 'remoteclip': 'Apache-2.0 (Liu et al.)'}
LADDER = {'prithvi_tl': 2, 'dofa': '1b', 'dinov2': 3, 'remoteclip': 4}


def needed(cfg):
    """Only models that appear in the registered extraction routes are fetched."""
    models = {m.split('_')[1] if m.startswith('random_') else m
              for r in cfg['extraction']['routes'] for m in r['models']}
    return [f for f in FILES if f[0] in models]


def plan(ctx):
    return [{'id': f'{model}.{name}', 'model': model, 'file': name, 'url': url, 'sha256': digest}
            for model, name, url, digest in needed(ctx.cfg)]


def target(ctx, unit):
    return ctx.shared / 'weights' / unit['model'] / unit['file']


def run(ctx, unit, ckpt):
    dest = target(ctx, unit)
    meta = dest.with_suffix(dest.suffix + '.json')
    old = read_json(meta)
    if dest.exists() and old and dest.stat().st_size == old['bytes'] and sha256_file(dest) == old['sha256']:
        if unit['sha256'] and old['sha256'] != unit['sha256']:
            raise StageFailed(f"{unit['id']}: cached file digest differs from the declared pin")
        return {'outputs': [dest, meta], 'metrics': {'cached': True, 'bytes': old['bytes']}}
    record = download(ctx, unit, dest, ckpt)
    if unit['sha256'] and record['sha256'] != unit['sha256']:
        dest.unlink(missing_ok=True)
        raise StageFailed(f"{unit['id']}: downloaded digest {record['sha256'][:12]} differs from the declared pin")
    atomic_write_json(meta, record)
    return {'outputs': [dest, meta], 'metrics': {'bytes': record['bytes'], 'resumed_from': record['resumed_from']}}


MAX_BYTES = 4 * 1024 ** 3   # no registered weight file is anywhere near this; refuse runaway bodies


def download(ctx, unit, dest, ckpt, session=None):
    """Stream to `dest.part`, resuming with a Range request after a tunnel drop; verify the final size."""
    part = dest.with_suffix(dest.suffix + '.part')
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or ctx.requests_session()
    try:
        have = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={have}-'} if have else {}
        headers['Accept-Encoding'] = 'identity'      # byte counts must match Content-Length exactly
        with session.get(unit['url'], headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
            if have and response.status_code != 206:
                # The server ignored the range: start again from zero.
                have = 0; part.unlink(missing_ok=True)
            elif response.status_code not in (200, 206):
                response.raise_for_status()
            total = None
            if response.status_code == 206:
                rng = response.headers.get('Content-Range', '')
                if not rng.startswith(f'bytes {have}-'):
                    raise RuntimeError('Unexpected Content-Range on resume: ' + rng)
                total = int(rng.rsplit('/', 1)[1]) if rng.endswith(tuple('0123456789')) else None
            elif response.headers.get('Content-Length'):
                total = int(response.headers['Content-Length'])
            if total is not None and total > MAX_BYTES:
                raise StageFailed(f"{unit['id']}: declared size {total} exceeds the weight-file ceiling")
            digest = hashlib.sha256()
            if have:
                with open(part, 'rb') as f:
                    for chunk in iter(lambda: f.read(8 * 1024 ** 2), b''):
                        digest.update(chunk)
            count = have
            with open(part, 'ab') as out:
                for chunk in response.iter_content(4 * 1024 ** 2):
                    ckpt.maybe_stop()
                    out.write(chunk); digest.update(chunk); count += len(chunk)
                    if count > MAX_BYTES:
                        raise StageFailed(f"{unit['id']}: body exceeds the weight-file ceiling")
            if total is not None and count != total:
                raise RuntimeError(f'Truncated weight download ({count} of {total} bytes); will resume')
            record = {'url': unit['url'], 'bytes': count, 'sha256': digest.hexdigest(),
                      # The repository commit is on the first (redirecting) hop; the CDN response has none.
                      'repo_commit': next((h.headers.get('x-repo-commit') for h in getattr(response, 'history', []) if h.headers.get('x-repo-commit')),
                                          response.headers.get('x-repo-commit')),
                      'etag': response.headers.get('etag'),
                      'licence': LICENCES.get(unit['model'], 'see upstream'), 'ladder_rung': LADDER.get(unit['model']),
                      'resumed_from': have}
    finally:
        session.close()
    os.replace(part, dest)
    return record


def finalize(ctx):
    manifest = {}
    for unit in ctx.units():
        meta = read_json(target(ctx, unit).with_suffix(Path(unit['file']).suffix + '.json'))
        if not meta:
            raise StageFailed('Missing weight metadata for ' + unit['id'])
        manifest[unit['id']] = {k: meta.get(k) for k in ('bytes', 'sha256', 'repo_commit', 'url', 'licence', 'ladder_rung')}
    path = ctx.stage_dir() / 'manifest.json'
    atomic_write_json(path, {'files': manifest, 'note': 'sha256 pins frozen at F1; weights live under shared/weights'})
    return {'files': len(manifest), 'manifest': str(path)}


STAGES = [Stage('weights_fetch', 'P1', 'net', (), plan, run, finalize, unit_cpus=1, unit_mem_gb=2, est_unit_min=10)]
