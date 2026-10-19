#!/usr/bin/env python3
"""Frozen FP32 CUDA extraction with verified inputs and resumable atomic chunks.

Author: James Edward Ball. No network calls or checkpoint downloads occur here.
Input bands: blue, green, red, NIR, SWIR1, SWIR2; reflectance float32 N,6,64,64.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_features(mean, cls, n):
    if mean.shape != cls.shape or mean.ndim != 2 or mean.shape[0] != n:
        raise ValueError('Invalid feature shapes')
    for x in (mean, cls):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Non-finite or non-float32 features')
        norms = np.linalg.norm(x, axis=1)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('Invalid feature norms')


def device_safe_position_encoding(encoder):
    """Keep regenerated positional tensors on the encoder's existing device."""
    original = encoder.interpolate_pos_encoding
    def interpolate(*args, **kwargs):
        return original(*args, **kwargs).to(device=encoder.pos_embed.device,
                                           dtype=encoder.pos_embed.dtype)
    encoder.interpolate_pos_encoding = interpolate


def build_adapter(root, name, device):
    import torch
    import yaml
    sys.path.insert(0, str(root / 'package/src'))
    from study.models import WEIGHTS
    random = name.startswith('random_')
    arch = name.split('_')[1] if random else name
    if random:
        seed = int(name.rsplit('_', 1)[1])
        if seed not in (101, 202, 303):
            raise ValueError('Random control seed must be 101, 202 or 303')
        torch.manual_seed(seed)
    weight = root / 'weights' / (arch + '.pt')
    checkpoint_sha = None
    if not random and arch != 'prithvi100':
        checkpoint_sha = sha(weight)
        if checkpoint_sha != WEIGHTS[arch][1]:
            raise ValueError(f'Checkpoint hash mismatch: {weight}')
    if arch == 'clay':
        from clay_probe.vendor.model import Encoder
        kw = dict(mask_ratio=0., patch_size=8, shuffle=False, dim=1024,
                  depth=24, heads=16, dim_head=64, mlp_ratio=4)
        if random:
            model = Encoder(**kw)
        else:
            with torch.device('meta'):
                model = Encoder(**kw)
            state = torch.load(weight, weights_only=True, mmap=True, map_location='cpu')['state_dict']
            prefix = 'model.encoder.'
            model.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                   if k.startswith(prefix)}, strict=True, assign=True)
        meta_path = root / 'package/src/clay_probe/vendor/metadata.yaml'
        sensor = yaml.safe_load(meta_path.read_text())['sentinel-2-l2a']['bands']
        bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
        means = [sensor['mean'][b] for b in bands]
        stds = [sensor['std'][b] for b in bands]
        waves = torch.tensor([sensor['wavelength'][b] for b in bands], device=device)
        # Pinned Clay constructs spatial positional encoding on CPU internally.
        # Keep scalar GSD on CPU to avoid a needless GPU-to-CPU synchronisation.
        gsd = torch.tensor(10., dtype=torch.float32)
        def tokens(x):
            b = len(x)
            return model(dict(pixels=x, time=torch.zeros(b, 4, device=device),
                              latlon=torch.zeros(b, 4, device=device), gsd=gsd,
                              waves=waves))[0]
        src = root / 'package/src/clay_probe/vendor'
    elif arch == 'prithvi':
        src = root / 'package/vendor/prithvi_eo_300m'
        spec = importlib.util.spec_from_file_location('prithvi_pinned', src / 'prithvi_mae.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = json.loads((src / 'config.json').read_text())['pretrained_cfg']
        model = module.PrithviMAE(**cfg)
        if not random:
            model.load_state_dict(torch.load(weight, weights_only=True, mmap=True,
                                             map_location='cpu'), strict=True)
        # Upstream regenerates resized position embeddings from NumPy on CPU.
        device_safe_position_encoding(model.encoder)
        means, stds = cfg['mean'], cfg['std']
        def tokens(x):
            return model.forward_features(x[:, :, None])[-1]
    elif arch == 'prithvi100':
        src = root / 'weights/prithvi100'
        manifest100 = json.loads((root / 'package/data/prithvi100/model.json').read_text())
        for entry in manifest100['files']:
            if random and entry['name'].endswith('.pt'):
                continue
            if sha(src / entry['name']) != entry['sha256']:
                raise ValueError('Prithvi100 source/checkpoint hash mismatch: ' + entry['name'])
        spec = importlib.util.spec_from_file_location('prithvi100_pinned', src / 'prithvi_mae.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = json.loads((src / 'config.json').read_text())['pretrained_cfg']
        cfg.update(img_size=64, num_frames=1)
        model = module.PrithviViT(**cfg)
        if model.pos_embed.shape != (1, 17, 768):
            raise ValueError('Prithvi100 positional embedding shape mismatch')
        if not random:
            weight = src / 'Prithvi_EO_V1_100M.pt'
            checkpoint_sha = sha(weight)
            state = torch.load(weight, weights_only=True, mmap=True, map_location='cpu')
            enc = {k.removeprefix('encoder.'): v for k, v in state.items()
                   if k.startswith('encoder.') and 'pos_embed' not in k}
            if set(enc) != set(dict(model.named_parameters())):
                raise ValueError('Prithvi100 parameter mismatch')
            loaded = model.load_state_dict(enc, strict=False)
            if loaded.missing_keys != ['pos_embed'] or loaded.unexpected_keys:
                raise ValueError('Prithvi100 checkpoint load mismatch')
        means, stds = cfg['mean'], cfg['std']
        def tokens(x):
            return model.forward_features(x[:, :, None])[-1]
    else:
        raise ValueError(f'Unsupported architecture: {arch}')
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    mean = torch.tensor(means, device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(stds, device=device, dtype=torch.float32)[None, :, None, None]
    def forward(raw):
        x = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True)).pin_memory().to(device, non_blocking=True)
        # Preserve the 100M extension's reflectance-domain operation order.
        x = (x - mean / 10000.) / (std / 10000.) if arch == 'prithvi100' else (x * 10000. - mean) / std
        t = tokens(x)
        return (t[:, 1:].mean(1).cpu().numpy(), t[:, 0].cpu().numpy())
    provenance = dict(checkpoint_sha256=checkpoint_sha, model=name, precision='float32',
                      normalisation_mean=means, normalisation_std=stds,
                      input_bands=['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
                      reflectance_multiplier=10000., clay_gsd=10. if arch == 'clay' else None,
                      metadata='zero time and location for Clay; omitted for Prithvi',
                      pooling=['mean_non_cls_final_tokens', 'cls_final_token'],
                      normalisation_order='reflectance_domain' if arch == 'prithvi100' else 'digital_number_domain',
                      sources={str(p.relative_to(root)): sha(p)
                               for p in sorted(src.rglob('*')) if p.suffix in ('.py', '.json', '.yaml')})
    return model, forward, provenance


def run(args):
    import torch
    root = args.root.resolve()
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    if not torch.cuda.is_available():
        raise RuntimeError('This extractor requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    images_path, manifest_path = root / 'data/images.npy', root / 'data/manifest.json'
    images = np.load(images_path, mmap_mode='r', allow_pickle=False)
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1:] != (6, 64, 64):
        raise ValueError('Expected reflectance float32 N,6,64,64')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest if isinstance(manifest, list) else manifest.get('rows', manifest.get('scenes'))
    if rows is None or len(rows) != len(images):
        raise ValueError('Manifest row count differs from image array')
    n = min(len(images), args.limit) if args.limit else len(images)
    if n < 1:
        raise ValueError('Empty input')
    name = args.model + (f'-diagnostic-{n}' if args.limit else '')
    out = root / 'features' / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model, forward, provenance = build_adapter(root, args.model, device)
        provenance.update(images_sha256=sha(images_path), manifest_sha256=sha(manifest_path),
                          extractor_sha256=sha(Path(__file__)), n=n, batch_size=args.batch_size,
                          torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), diagnostic_limit=args.limit)
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        marker = out / 'complete.json'
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior['fingerprint'] == fingerprint and all(sha(out / k) == v for k, v in prior['files'].items()):
                print(json.dumps(dict(status='already_complete', directory=str(out))), flush=True)
                return
            raise RuntimeError('Completed cache differs from current inputs/settings; use a new root')
        atomic_json(out / 'provenance.json', dict(fingerprint=fingerprint, **provenance))
        # A mixed set across input order, verified both singly and as one batch.
        inds = np.unique(np.linspace(0, n - 1, min(8, n), dtype=int))
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            probe = images[inds]
            if not np.isfinite(probe).all():
                raise ValueError('Non-finite input')
            batched = forward(probe)
            single = [forward(probe[i:i+1]) for i in range(len(probe))]
            errors = []
            for k in range(2):
                reference = np.concatenate([x[k] for x in single])
                np.testing.assert_allclose(batched[k], reference, rtol=1e-4, atol=1e-4)
                errors.append(float(np.max(np.abs(batched[k] - reference))))
            validate_features(*batched, len(probe))
            atomic_json(out / 'parity.json', dict(indices=inds.tolist(), max_abs_error=errors,
                                                rtol=1e-4, atol=1e-4, passed=True))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()
            computed = 0
            chunks = out / 'chunks'
            chunks.mkdir(exist_ok=True)
            dim = batched[0].shape[1]
            # Write final arrays to temporary memmaps; chunks survive interruption.
            dest = [out / 'mean.npy', out / 'cls.npy']
            temp = [p.with_suffix('.npy.tmp') for p in dest]
            maps = [np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(n, dim)) for p in temp]
            for start in range(0, n, args.batch_size):
                stop = min(start + args.batch_size, n)
                chunk = chunks / f'{start:08d}-{stop:08d}.npz'
                if chunk.exists():
                    with np.load(chunk, allow_pickle=False) as saved:
                        if str(saved['fingerprint']) != fingerprint:
                            raise RuntimeError(f'Cache provenance mismatch: {chunk}')
                        feats = saved['mean'], saved['cls']
                else:
                    raw = images[start:stop]
                    if not np.isfinite(raw).all():
                        raise ValueError(f'Non-finite images at rows {start}:{stop}')
                    feats = forward(raw)
                    validate_features(*feats, stop-start)
                    tmp = chunk.with_suffix('.npz.tmp')
                    with open(tmp, 'wb') as f:
                        np.savez(f, mean=feats[0], cls=feats[1], fingerprint=fingerprint)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, chunk)
                    computed += stop-start
                validate_features(*feats, stop-start)
                for mm, values in zip(maps, feats):
                    mm[start:stop] = values
                if start == 0 or stop == n or (start // args.batch_size) % 20 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(json.dumps(dict(model=args.model, completed=stop, total=n,
                                          computed_this_run=computed, elapsed_seconds=elapsed,
                                          images_per_second=computed/max(elapsed, 1e-9))), flush=True)
            for mm in maps:
                mm.flush()
            del maps
            for tmp, dst in zip(temp, dest):
                with open(tmp, 'rb') as f:
                    os.fsync(f.fileno())
                os.replace(tmp, dst)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            result = dict(fingerprint=fingerprint, files={p.name: sha(p) for p in dest},
                          shape=[n, dim], computed_this_run=computed, elapsed_seconds=elapsed,
                          images_per_second=computed/max(elapsed, 1e-9),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), parity_passed=True)
            atomic_json(marker, result)
            print(json.dumps(dict(status='complete', **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=['clay', 'prithvi', 'prithvi100'] +
                        [f'random_{a}_{s}' for a in ('clay', 'prithvi', 'prithvi100') for s in (101, 202, 303)], required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and limit must be positive')
    # Set before CUDA context initialisation for deterministic cuBLAS execution.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run(args)


if __name__ == '__main__':
    main()
