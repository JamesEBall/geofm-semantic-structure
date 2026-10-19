"""Native-grid imagery I/O contract. Author: James Edward Ball."""
from __future__ import annotations
from pathlib import Path
from mc.core import atomic_path, atomic_write_json, read_json, sha256_file

BANDS10 = ('B02', 'B03', 'B04', 'B08')
BANDS20 = ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12')


def write_parquet(path, rows):
    import pandas as pd
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    with atomic_path(path) as tmp:
        frame.to_parquet(tmp, index=False, compression='zstd')


def save_array(path, array):
    import numpy as np
    with atomic_path(path) as tmp:
        with open(tmp, 'wb') as stream:
            np.save(stream, array, allow_pickle=False)


def reflectance(dn, offset, scale):
    import numpy as np
    if not np.isfinite(offset) or not np.isfinite(scale) or abs(float(scale) - 1e-4) > 1e-12:
        raise ValueError('invalid declared radiometry')
    return (np.asarray(dn, dtype=np.float32) + np.float32(offset)) * np.float32(scale)


def analysis_crop(b10, b20=None, scl=None):
    """Crop arbitrary leading axes; one-array use also supported."""
    if b10.shape[-2:] != (256, 256):
        raise ValueError('stored 10 m window must be 256 by 256')
    a = b10[..., 64:192, 64:192]
    if b20 is None and scl is None:
        return a
    if b20.shape[-2:] != (128, 128) or scl.shape[-2:] != (128, 128):
        raise ValueError('stored 20 m window must be 128 by 128')
    return a, b20[..., 32:96, 32:96], scl[..., 32:96, 32:96]


def iter_shards(ctx):
    for stage in ('acquire_pilot', 'acquire'):
        for path in sorted(ctx.p(stage, 'shards').glob('shard-*')):
            if (path / 'manifest.json').exists():
                yield path


def verify_shard(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    expected = {'b10.npy', 'b20.npy', 'scl.npy', 'meta.parquet', 'rejects.parquet'}
    if not manifest or set(manifest.get('sha256', {})) != expected:
        raise ValueError(f'incomplete shard: {path.name}')
    for name, digest in manifest['sha256'].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f'shard checksum mismatch: {path.name}/{name}')
    return manifest


def open_shard(ctx, shard):
    import numpy as np
    import pandas as pd
    path = Path(shard)
    if not path.is_absolute():
        path = ctx.run / path
    manifest = verify_shard(path)
    data = {key: np.load(path / f'{key}.npy', mmap_mode='r', allow_pickle=False) for key in ('b10', 'b20', 'scl')}
    data['meta'] = pd.read_parquet(path / 'meta.parquet')
    data['rejects'] = pd.read_parquet(path / 'rejects.parquet')
    n = len(data['meta'])
    for key, shape, dtype in [('b10', (n, 4, 256, 256), 'uint16'), ('b20', (n, 6, 128, 128), 'uint16'), ('scl', (n, 128, 128), 'uint8')]:
        if data[key].shape != shape or data[key].dtype != dtype:
            raise ValueError(f'shard array contract violation: {key}')
    if manifest['accepted'] != n:
        raise ValueError('shard metadata length mismatch')
    return data


def write_shard(path, accepted, rejected):
    import numpy as np
    import pandas as pd
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # The manifest is the commit marker. Remove it before replacing constituent files.
    (path / 'manifest.json').unlink(missing_ok=True)
    for name, shape, dtype in [('b10', (0, 4, 256, 256), 'uint16'), ('b20', (0, 6, 128, 128), 'uint16'), ('scl', (0, 128, 128), 'uint8')]:
        array = np.stack([a[name] for a in accepted]) if accepted else np.empty(shape, dtype=dtype)
        save_array(path / f'{name}.npy', array)
    meta_cols = ['window_id', 'item_id', 'source', 'datetime', 'epsg', 'x0', 'y0', 'dn_offset', 'dn_scale', 'baseline', 'bad_frac', 'snow_frac', 'tries', 'mgrs_tile', 'pixel_sha256']
    frame = pd.DataFrame([a['meta'] for a in accepted]) if accepted else pd.DataFrame(columns=meta_cols)
    write_parquet(path / 'meta.parquet', frame)
    write_parquet(path / 'rejects.parquet', pd.DataFrame(rejected, columns=['window_id', 'reason', 'tries']))
    names = ['b10.npy', 'b20.npy', 'scl.npy', 'meta.parquet', 'rejects.parquet']
    atomic_write_json(path / 'manifest.json', {'accepted': len(accepted), 'rejected': len(rejected),
        'sha256': {name: sha256_file(path / name) for name in names}})
    return [path / n for n in names + ['manifest.json']]
