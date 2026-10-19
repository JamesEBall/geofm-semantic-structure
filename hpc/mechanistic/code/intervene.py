#!/usr/bin/env python3
"""Training-only output-subspace interventions. Author: James Edward Ball."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def load_analysis(path):
    spec = importlib.util.spec_from_file_location('frozen_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standardise(train, val, test):
    mean, scale = train.mean(0), train.std(0)
    scale[scale < 1e-8] = 1.
    return [(x - mean) / scale for x in (train, val, test)]


def direction(x, y, category):
    q = x[y == category].mean(0) - x[y != category].mean(0)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError('Degenerate class contrast')
    return q / norm


def erase(x, q, strength=1.):
    return x - strength * np.outer(x @ q, q)


def refit_weights(w, invq, q):
    """Exact ridge solution constrained to q-perpendicular space, same alpha."""
    return w - np.outer(invq, q @ w) / (q @ invq)


def energy_strengths(x, q, r):
    eq, er = np.mean((x @ q) ** 2), np.mean((x @ r) ** 2)
    if min(eq, er) < 1e-12:
        raise ValueError('Degenerate intervention energy')
    return min(1., np.sqrt(er / eq)), min(1., np.sqrt(eq / er)), eq, er


def summarise(delta, masks, block_indices, draws):
    out = {}; samples = {}
    for name, mask in masks.items():
        indices = np.unique(block_indices[mask])
        means = np.array([delta[mask & (block_indices == b)].mean() for b in indices])
        weights = draws[:, indices]
        denominator = weights.sum(1)
        good = denominator > 0
        values = weights[good] @ means / denominator[good]
        samples[name] = np.full(len(draws), np.nan); samples[name][good] = values
        out[name] = {'change': float(means.mean()), 'interval': np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                     'blocks': len(indices), 'valid_resamples': int(good.sum())}
    selective = samples['target'] - samples['other']
    valid = np.isfinite(selective)
    out['target_minus_other'] = {'change': out['target']['change'] - out['other']['change'],
        'interval': np.quantile(selective[valid], [.025,.975]).tolist() if valid.any() else None,
        'valid_resamples': int(valid.sum())}
    return out


def run_model(x, y, splits, blocks, text, api, seeds, repeats, score_cache=None):
    tr, va, te = [splits == s for s in ('train', 'val', 'test')]
    a, v, t = standardise(np.asarray(x[tr], float), np.asarray(x[va], float), np.asarray(x[te], float))
    target = np.eye(10)[y[tr]]
    ym = target.mean(0)
    eigen, u = np.linalg.eigh(a.T @ a)
    eigen = np.maximum(eigen, 0.)
    rhs = u.T @ (a.T @ (target - ym))
    fits, best = {}, {}
    for alpha in api.ALPHAS:
        w = u @ (rhs / (eigen[:, None] + alpha))
        fits[alpha] = w
        for ref in api.REFS:
            scores = api.language_scores(v @ w + ym, text[ref])
            value = api.block_average(api.reciprocal_rank(y[va], scores), blocks[va])
            if ref not in best or value > best[ref][0]:
                best[ref] = value, alpha
    _, bi = np.unique(blocks[te], return_inverse=True)
    count = len(np.unique(bi))
    draws = np.random.default_rng(51426).multinomial(count, np.full(count, 1/count), size=repeats)
    predictions = {ref: t @ fits[best[ref][1]] + ym for ref in api.REFS}
    out = {'selection': {r: {'validation_query_ap': best[r][0], 'alpha': best[r][1]} for r in api.REFS}, 'categories': {}}
    for c, label in enumerate(api.CLASSES):
        q = direction(a, y[tr], c)
        masks = {'target': y[te] == c, 'other': y[te] != c}
        conditions = []
        # Keep conditions compact: each stores projected test coordinate and direction.
        conditions.append(('target_full', q, 1., 'fixed'))
        conditions.append(('target_refit', q, 1., 'refit'))
        for seed in seeds:
            rng = np.random.default_rng(seed + c * 10000)
            r = rng.normal(size=a.shape[1]); r /= np.linalg.norm(r)
            sq, sr, eq, er = energy_strengths(a, q, r)
            conditions.append((f'random_{seed}_full', r, 1., 'fixed'))
            conditions.append((f'random_{seed}_refit', r, 1., 'refit'))
            for strength in (.25, .5, 1.):
                conditions.append((f'matched_target_{seed}_{strength}', q, sq * strength, 'fixed'))
                conditions.append((f'matched_random_{seed}_{strength}', r, sr * strength, 'fixed'))
        projected = {key: (a @ z, t @ z, u.T @ z) for key, z in {id(z): z for _, z, _, _ in conditions}.items()}
        cat = {'training_target_energy': float(np.mean((a @ q)**2)), 'references': {}}
        for ref in api.REFS:
            alpha = best[ref][1]; w = fits[alpha]
            prediction = predictions[ref]
            rr = api.reciprocal_rank(y[te], api.language_scores(prediction, text[ref]))
            rr_by_name = {}; diagnostics = {}; contrasts = {}
            for name, z, strength, mode in conditions:
                az, tz, uz = projected[id(z)]
                if mode == 'fixed':
                    pred = prediction - strength * np.outer(tz, z @ w)
                else:
                    invq = u @ (uz / (eigen + alpha))
                    pred = prediction - np.outer(t @ invq, z @ w) / (z @ invq)
                after = api.reciprocal_rank(y[te], api.language_scores(pred, text[ref]))
                rr_by_name[name] = after
                diagnostics[name] = {'strength': float(strength), 'mode': mode,
                    'train_removed_energy': float(strength**2 * np.mean(az**2)),
                    'test_removed_energy': float(strength**2 * np.mean(tz**2))}
            for name, vals in rr_by_name.items():
                if name in ('target_full', 'target_refit'):
                    contrasts[name + '_minus_baseline'] = summarise(vals - rr, masks, bi, draws)
            for strength in (.25, .5, 1.):
                # Seed-averaged paired contrast, not seed pseudo-replication.
                delta = np.mean([rr_by_name[f'matched_target_{s}_{strength}'] - rr_by_name[f'matched_random_{s}_{strength}'] for s in seeds], axis=0)
                contrasts[f'energy_matched_target_minus_random_{strength}'] = summarise(delta, masks, bi, draws)
            for mode in ('full', 'refit'):
                delta = rr_by_name['target_' + mode] - np.mean([rr_by_name[f'random_{s}_{mode}'] for s in seeds], axis=0)
                contrasts[f'rank_matched_target_minus_random_{mode}'] = summarise(delta, masks, bi, draws)
            if score_cache is not None:
                for name, values in {'baseline': rr, **rr_by_name}.items():
                    score_cache[f'{label}/{ref}/{name}'] = values
            cat['references'][ref] = {'contrasts': contrasts, 'interventions': diagnostics,
                'baseline': {name: api.block_average(rr[mask], blocks[te][mask]) for name, mask in masks.items()}}
        out['categories'][label] = cat
    return out


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(report['models'])
    fig, axes = plt.subplots(len(names), 3, figsize=(14, len(names)*2.5), squeeze=False)
    for row, model in enumerate(names):
        for col, ref in enumerate(report['references']):
            ax = axes[row, col]
            for c, label in enumerate(report['classes']):
                result = report['models'][model]['categories'][label]['references'][ref]['contrasts']['energy_matched_target_minus_random_1.0']['target']
                ax.plot(result['interval'], [c,c], color='#8394a2')
                ax.scatter(result['change'], c, color='#304e63', s=12)
            ax.axvline(0, color='.6', lw=.6)
            ax.set_yticks(range(10), report['classes'] if col == 0 else [])
            ax.set_title(f'{model} | {ref}', fontsize=8)
    fig.supxlabel('Target-category query AP: targeted minus energy-matched random removal')
    fig.tight_layout(); fig.savefig(out / 'targeted-removal.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fresh-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['appearance','clay','prithvi','prithvi100','random_clay_101','random_prithvi_101','random_prithvi100_101'])
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    root = args.fresh_root; api = load_analysis(root / 'code/analyse.py')
    raw = json.loads((root / 'data/manifest.json').read_text()); rows = raw if isinstance(raw, list) else raw['rows']
    api.validate_manifest(rows); api.validate_geography(rows)
    y = np.array([api.CLASSES.index(r['label']) for r in rows]); splits = np.array([r['split'] for r in rows]); blocks = np.array([str(r['block']) for r in rows])
    paths = api.discover_features(root, 'mean', len(rows), api.sha(root / 'data/manifest.json'))
    with np.load(root / 'data/text.npz') as z:
        text = {r: z[r] for r in api.REFS}
    report = {'author':'James Edward Ball', 'manifest_sha256':api.sha(root/'data/manifest.json'), 'text_sha256':api.sha(root/'data/text.npz'),
              'analysis_sha256':api.sha(Path(__file__)), 'protocol_sha256':api.sha(Path(__file__).with_name('protocol.md')), 'fresh_analysis_sha256':api.sha(root/'code/analyse.py'),
              'classes':api.CLASSES, 'references':api.REFS, 'random_seeds':[1103,2207,3301,4409,5501], 'bootstrap_resamples':args.bootstrap,
              'interpretation':'Exploratory output-feature dependence of a supervised readout; conditional block intervals; no encoder-mechanism or unseen-category claim.', 'models':{}}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        print('Intervening:', name, flush=True)
        x = np.load(paths[name], mmap_mode='r')
        if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all(): raise ValueError(name)
        cache = {}
        report['models'][name] = run_model(x,y,splits,blocks,text,api,report['random_seeds'],args.bootstrap,cache)
        np.savez_compressed(args.out/f'{name}-scene-scores.npz', **cache, test_ids=np.array([r['id'] for r in rows if r['split']=='test']))
        report['models'][name]['feature_sha256'] = api.sha(paths[name])
        tmp = args.out/'interventions.partial.json'; tmp.write_text(json.dumps(report, allow_nan=False)); tmp.replace(args.out/'interventions.json')
    plot(report,args.out)
    (args.out/'complete.json').write_text(json.dumps({'sha256':api.sha(args.out/'interventions.json'), 'models':args.models}, indent=2))

if __name__ == '__main__': main()
