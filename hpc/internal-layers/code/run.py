"""Layer readout and residual-stream perturbations. Author: James Edward Ball."""
import argparse, hashlib, json, os, sys, types
from pathlib import Path
import numpy as np


def select(rows, limit=4000):
    # Equal split/class budgets, then cycle over geographic blocks within each cell.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r['split'], r['label']), {}).setdefault(str(r['block']), []).append(i)
    for blocks in groups.values():
        for b, ids in blocks.items():
            ids.sort(key=lambda i: hashlib.sha256(json.dumps(rows[i], sort_keys=True).encode()).hexdigest())
    chosen = []
    while len(chosen) < min(limit, len(rows)):
        progress = False
        for key in sorted(groups):
            for b in list(groups[key]):
                ids = groups[key][b]
                if ids and len(chosen) < limit:
                    chosen.append(ids.pop()); progress = True
                    break
            # Rotate blocks so large first blocks cannot monopolise a cell.
            blocks = groups[key]
            if blocks:
                first = next(iter(blocks)); blocks[first] = blocks.pop(first)
        # Use insertion order after rotation on next sweep (handled below).
        if not progress: break
    return np.array(sorted(chosen), dtype=np.int64)


def perturb(x, direction, dose, random=None):
    import torch
    if dose == 0: return x
    p = x[:, 1:]
    d = torch.as_tensor(direction, device=x.device, dtype=x.dtype)
    delta = (p @ d)[..., None] * d
    if isinstance(random, str) and random == 'norm-only':
        target = p - dose * delta
        ratio = torch.linalg.vector_norm(target, dim=-1, keepdim=True) / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(1e-12)
        out = x.clone(); out[:, 1:] = p * ratio
        return out
    if random is not None:
        r = torch.as_tensor(random, device=x.device, dtype=x.dtype)
        # Same per-token removed energy, with signed random-direction projection.
        sign = torch.where((p @ r) >= 0, 1., -1.)
        delta = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) * sign[..., None] * r
    out = x.clone(); out[:, 1:] -= dose * delta
    return out


def attach(model, arch, callback):
    """Callback receives the complete post-residual state, before final layer norm."""
    handles = []
    if arch == 'clay':
        transformer = model.transformer
        original = transformer.forward
        def forward(self, x):
            for i, (attn, ff) in enumerate(self.layers, 1):
                x = attn(x) + x
                x = ff(x) + x
                x = callback(i, x)
            return self.norm(x)
        transformer.forward = types.MethodType(forward, transformer)
        return len(transformer.layers), lambda: setattr(transformer, 'forward', original)
    encoder = model.encoder if hasattr(model, 'encoder') else model
    for i, block in enumerate(encoder.blocks, 1):
        handles.append(block.register_forward_hook(lambda m, a, out, index=i: callback(index, out)))
    return len(encoder.blocks), lambda: [h.remove() for h in handles]


def main(a):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sys.path.insert(0, str(a.fresh / 'code'))
    from extract import build_adapter, sha, atomic_json
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = json.loads((a.fresh / 'data/manifest.json').read_text())
    if isinstance(rows, dict): rows = rows.get('rows', rows.get('scenes'))
    ids = select(rows, a.limit)
    selected = [rows[i] for i in ids]
    images = np.load(a.fresh / 'data/images.npy', mmap_mode='r')
    out = a.output / a.model; out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists(): raise RuntimeError('Output already complete; use a new output root')
    atomic_json(out / 'selection.json', dict(indices=ids.tolist(), rows=selected, manifest_sha256=sha(a.fresh / 'data/manifest.json')))
    model, forward, provenance = build_adapter(a.fresh, a.model, torch.device('cuda'))
    capture = {}; state = {'intervention': None}
    def callback(index, x):
        spec = state['intervention']
        if spec is None and index in layers: capture[index] = x[:, 1:].mean(1).cpu().numpy()
        if spec is not None and index == midpoint:
            return perturb(x, *spec)
        return x
    probe = images[ids[:4]]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        original = forward(probe)
        depth, detach = attach(model, a.model, callback)
        layers = sorted({max(1, round(depth*f)) for f in (.25, .5, .75, 1.)})
        midpoint = round(depth*.5)
        hooked = forward(probe)
        for x, y in zip(original, hooked): np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)
        direction = np.zeros(hooked[0].shape[1], np.float32); direction[0] = 1
        state['intervention'] = (direction, 0., None)
        zero = forward(probe)
        for x, y in zip(original, zero): np.testing.assert_array_equal(x, y)
        state['intervention'] = None
        clean = []; layer_cache = {i: [] for i in layers}
        for start in range(0, len(ids), a.batch):
            clean.append(forward(images[ids[start:start+a.batch]])[0])
            for i in layers: layer_cache[i].append(capture[i])
        clean = np.concatenate(clean)
        fresh_dir = a.fresh / 'features' / a.model
        fresh_marker = json.loads((fresh_dir / 'complete.json').read_text())
        reference = np.load(fresh_dir / 'mean.npy', mmap_mode='r')[ids]
        np.testing.assert_allclose(clean, reference, rtol=1e-4, atol=1e-4)
        np.save(out / 'clean.npy', clean)
        fingerprint = hashlib.sha256(json.dumps(dict(runner=sha(Path(__file__)), provenance=provenance,
            fresh_marker=sha(fresh_dir/'complete.json'), fresh_extractor=sha(a.fresh/'code/extract.py'),
            manifest=sha(a.fresh/'data/manifest.json'), indices=ids.tolist(), batch=a.batch,
            test_limit=a.test_limit, clean=sha(out/'clean.npy')),sort_keys=True).encode()).hexdigest()
        for i, arrays in layer_cache.items():
            layer_cache[i] = np.concatenate(arrays)
            np.save(out / f'layer-{i}.npy', layer_cache[i])
        labels = np.array([r['label'] for r in selected]); split = np.array([r['split'] for r in selected])
        train = split == 'train'
        test_pool = np.flatnonzero(split == 'test')
        test = test_pool[select([selected[i] for i in test_pool], a.test_limit)]
        np.save(out / 'test_indices.npy', test)
        blocks = np.array([str(r['block']) for r in selected])
        directions = {}; mid = layer_cache[midpoint]
        def blockmean(mask):
            return np.mean([mid[mask & (blocks == b)].mean(0) for b in np.unique(blocks[mask])], axis=0)
        for label in sorted(set(labels)):
            pos = train & (labels == label); neg = train & (labels != label)
            if not pos.any() or not neg.any(): raise ValueError('Missing development category')
            d = blockmean(pos) - blockmean(neg)
            if np.linalg.norm(d) < 1e-8: raise ValueError('Degenerate category direction')
            directions[label] = d / np.linalg.norm(d)
        conditions = []
        for label, direction in directions.items():
            rng = np.random.default_rng(731)
            controls = [('target', None), ('norm-only', 'norm-only')]
            for seed in range(3):
                r = rng.normal(size=len(direction)); r -= (r @ direction)*direction; r /= np.linalg.norm(r)
                controls.append((f'random-{seed}', r.astype(np.float32)))
            for dose in (.5, 1.):
                for kind, control in controls:
                    name = f'{label}-dose{dose}-{kind}'
                    path = out / (name+'.npy'); receipt = out / (name+'.json')
                    condition = dict(name=name, label=label, dose=dose, control=kind)
                    if path.exists() or receipt.exists():
                        prior = json.loads(receipt.read_text())
                        if prior['fingerprint'] != fingerprint or prior['sha256'] != sha(path):
                            raise RuntimeError('Existing condition cache differs: '+name)
                        cached = np.load(path, mmap_mode='r')
                        if cached.shape != (len(test), clean.shape[1]) or not np.isfinite(cached).all():
                            raise ValueError('Invalid condition cache')
                        conditions.append(condition)
                        continue
                    state['intervention'] = (direction, dose, control)
                    arrays = []
                    for start in range(0, len(test), a.batch):
                        arrays.append(forward(images[ids[test[start:start+a.batch]]])[0])
                    combined = np.concatenate(arrays)
                    if not np.isfinite(combined).all(): raise ValueError('Nonfinite intervention output')
                    tmp = path.with_suffix('.tmp.npy')
                    np.save(tmp, combined); os.replace(tmp, path)
                    atomic_json(receipt, dict(fingerprint=fingerprint, sha256=sha(path)))
                    conditions.append(condition)
                    print(name, flush=True)
        np.savez(out / 'directions.npz', **directions)
        detach()
    atomic_json(out / 'complete.json', dict(model=a.model, layers=layers, midpoint=midpoint,
        n=len(ids), test_n=len(test), conditions=conditions, zero_dose_exact=True, fresh_feature_replay=True, fingerprint=fingerprint,
        fresh_extractor_sha256=sha(a.fresh/'code/extract.py'),
        provenance=provenance, runner_sha256=sha(Path(__file__)), manifest_sha256=sha(a.fresh/'data/manifest.json'),
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in ('.npy', '.npz', '.json')}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fresh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--model', choices=['clay','prithvi','prithvi100'], required=True)
    p.add_argument('--test-limit',type=int,default=300)
    p.add_argument('--limit',type=int,default=4000); p.add_argument('--batch',type=int,default=32)
    main(p.parse_args())
