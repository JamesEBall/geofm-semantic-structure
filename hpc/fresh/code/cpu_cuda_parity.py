#!/usr/bin/env python3
"""Compare the fresh-data adapter on CPU and CUDA, without downloading files.

This checks device numerics, not independent agreement with the original study
implementation. Author: James Edward Ball.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from extract import atomic_json, build_adapter, sha, validate_features


def comparison(cpu, cuda, rtol=1e-4, atol=1e-4):
    if cpu.shape != cuda.shape:
        raise ValueError('CPU and CUDA feature shapes differ')
    cpu64, cuda64 = cpu.astype(np.float64), cuda.astype(np.float64)
    difference = np.abs(cpu64 - cuda64)
    scale = atol + rtol * np.abs(cpu64)
    cosine = np.sum(cpu64 * cuda64, axis=1) / (
        np.linalg.norm(cpu64, axis=1) * np.linalg.norm(cuda64, axis=1))
    return dict(passed=bool(np.all(difference <= scale)),
                max_absolute_error=float(difference.max()),
                mean_absolute_error=float(difference.mean()),
                max_tolerance_ratio=float(np.max(difference / scale)),
                cosine_similarity=cosine.tolist(),
                min_cosine_similarity=float(cosine.min()),
                elements_outside_tolerance=int(np.sum(difference > scale)),
                total_elements=int(difference.size), rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['clay', 'prithvi'],
                        choices=['clay', 'prithvi', 'prithvi100'])
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.samples < 2 or args.samples > 4:
        parser.error('--samples must be 2, 3 or 4')
    if args.rtol < 0 or args.atol <= 0 or args.threads < 1:
        parser.error('Use non-negative rtol and positive atol/threads')
    # Must precede CUDA context construction.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for CPU/CUDA parity check')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = args.root.resolve()
    images_path = root / 'data/images.npy'
    manifest_path = root / 'data/manifest.json'
    images = np.load(images_path, mmap_mode='r', allow_pickle=False)
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1:] != (6, 64, 64):
        raise ValueError('Expected float32 N,6,64,64 reflectance')
    if len(images) < args.samples:
        raise ValueError('Not enough images for parity check')
    indices = np.linspace(0, len(images) - 1, args.samples, dtype=int)
    probe = np.array(images[indices], copy=True)
    if not np.isfinite(probe).all():
        raise ValueError('Probe contains non-finite input')
    output = root / 'results/cpu-cuda-parity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(author='James Edward Ball', status='running',
                  check='same pinned fresh-data adapter on CPU versus CUDA',
                  scope='device numerical agreement; not independent original-implementation parity',
                  precision='float32', attention_backend='math',
                  images_sha256=sha(images_path), manifest_sha256=sha(manifest_path),
                  script_sha256=sha(Path(__file__)),
                  extractor_sha256=sha(Path(__file__).with_name('extract.py')),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), indices=indices.tolist(),
                  rtol=args.rtol, atol=args.atol, records=[])
    atomic_json(output, report)
    all_passed = True
    for name in args.models:
        start = time.perf_counter()
        record = dict(model=name)
        try:
            features = {}
            for device_name in ('cpu', 'cuda:0'):
                model, forward, provenance = build_adapter(root, name, torch.device(device_name))
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    values = forward(probe)
                validate_features(*values, len(probe))
                features[device_name] = tuple(v.copy() for v in values)
                del forward, model
                gc.collect()
                torch.cuda.empty_cache()
            record['provenance'] = provenance
            record['mean'] = comparison(features['cpu'][0], features['cuda:0'][0], args.rtol, args.atol)
            record['cls'] = comparison(features['cpu'][1], features['cuda:0'][1], args.rtol, args.atol)
            record['passed'] = record['mean']['passed'] and record['cls']['passed']
        except Exception as exc:
            record.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        record['elapsed_seconds'] = time.perf_counter() - start
        all_passed = all_passed and record['passed']
        report['records'].append(record)
        atomic_json(output, report)
        print(json.dumps(record, sort_keys=True), flush=True)
    report['passed'] = all_passed
    report['status'] = 'passed' if all_passed else 'failed'
    atomic_json(output, report)
    if not all_passed:
        raise SystemExit('CPU/CUDA parity failed; inspect ' + str(output))
    print('CPU/CUDA parity passed: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
