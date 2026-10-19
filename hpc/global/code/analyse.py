#!/usr/bin/env python3
"""Frozen global external validation. Author: James Edward Ball."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

CLASSES = ['Water', 'Trees', 'Grass', 'Flooded_Vegetation', 'Crops', 'Scrub', 'Built_Area', 'Bare_Ground']
N_CLASSES = len(CLASSES)
FOLDS = [(0, 1), (2, 3), (4, 5), (6, 7)]
REFS = ['qwen_instructed', 'qwen_plain', 'mpnet_plain']
ALPHAS = [1., 10., 100., 1000.]
EXPECTED_MODELS = ['clay', 'prithvi', 'prithvi100'] + [
    f'random_{arch}_{seed}' for arch in ('clay', 'prithvi', 'prithvi100')
    for seed in (101, 202, 303)]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def discover_features(root, pooling, nrows, manifest_hash):
    """Admit the frozen complete panel only; diagnostic runs never enter analysis."""
    paths = {'appearance': root / 'data/appearance.npy'}
    for name in EXPECTED_MODELS:
        directory = root / 'features' / name
        path = directory / f'{pooling}.npy'
        marker = directory / 'complete.json'
        provenance = directory / 'provenance.json'
        if not path.exists() or not marker.exists() or not provenance.exists():
            raise ValueError(f'Planned route is incomplete: {name}')
        done = json.loads(marker.read_text())
        source = json.loads(provenance.read_text())
        if done.get('shape', [None])[0] != nrows or not done.get('parity_passed'):
            raise ValueError(f'Wrong row count or failed parity: {name}')
        if source.get('manifest_sha256') != manifest_hash or source.get('diagnostic_limit'):
            raise ValueError(f'Wrong source manifest or diagnostic route: {name}')
        if done.get('fingerprint') != source.get('fingerprint'):
            raise ValueError(f'Provenance fingerprint mismatch: {name}')
        if done.get('files', {}).get(path.name) != sha(path):
            raise ValueError(f'Feature checksum mismatch: {name}')
        paths[name] = path
    return paths


def validate_manifest(rows):
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate scene IDs')
    block_splits = {}
    for r in rows:
        if r['label'] not in CLASSES or r['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid label or split: {r}')
        if not np.isfinite([r['lon'], r['lat']]).all():
            raise ValueError('Nonfinite location')
        block_splits.setdefault(str(r['block']), set()).add(r['split'])
    if any(len(v) > 1 for v in block_splits.values()):
        raise ValueError('A parent block crosses splits')
    for split in ('train', 'val', 'test'):
        missing = set(CLASSES) - {r['label'] for r in rows if r['split'] == split}
        if missing:
            raise ValueError(f'{split} lacks classes: {missing}')



def validate_geography(rows, minimum_km=5.):
    from scipy.spatial import cKDTree
    coordinates = np.radians([[r['lon'], r['lat']] for r in rows])
    lon, lat = coordinates.T
    points = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    splits = np.array([r['split'] for r in rows])
    minima = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        chord = cKDTree(points[splits == a]).query(points[splits == b])[0].min()
        km = float(2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1)))
        minima[a + '_' + b] = km
        if km < minimum_km - 1e-6:
            raise ValueError(f'Cross-split geographic separation {a}/{b}: {km:.6f} km')
    return minima

def ridge_predictions(xfit, yfit, xval, xtest):
    """Fit intercept and scaling on training only; one spectral solve for all refs."""
    mean = xfit.mean(0, dtype=np.float64)
    scale = xfit.std(0, dtype=np.float64)
    scale[scale < 1e-8] = 1.
    x = (np.asarray(xfit, dtype=np.float64) - mean) / scale
    v = (np.asarray(xval, dtype=np.float64) - mean) / scale
    t = (np.asarray(xtest, dtype=np.float64) - mean) / scale
    ym = yfit.mean(0)
    yc = yfit - ym
    if x.shape[1] <= x.shape[0]:
        eigen, u = np.linalg.eigh(x.T @ x)
        rhs = u.T @ (x.T @ yc)
        for alpha in ALPHAS:
            w = u @ (rhs / (np.maximum(eigen, 0)[:, None] + alpha))
            yield alpha, v @ w + ym, t @ w + ym
    else:
        eigen, u = np.linalg.eigh(x @ x.T)
        rhs = u.T @ yc
        for alpha in ALPHAS:
            w = x.T @ (u @ (rhs / (np.maximum(eigen, 0)[:, None] + alpha)))
            yield alpha, v @ w + ym, t @ w + ym


def language_scores(pred, text):
    text = unit(text)
    gram = text @ text.T
    norm = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', pred, gram, pred), 1e-24))
    return (pred @ gram) / norm[:, None]


def reciprocal_rank(y, scores):
    """Expected per-query AP for one positive, averaging uniformly over exact ties."""
    target = scores[np.arange(len(y)), y]
    above = (scores > target[:, None]).sum(1)
    tied = (scores == target[:, None]).sum(1)
    harmonic = np.r_[0., np.cumsum(1. / np.arange(1, scores.shape[1] + 1))]
    return (harmonic[above + tied] - harmonic[above]) / tied


def block_average(values, blocks):
    return float(np.mean([values[blocks == b].mean() for b in np.unique(blocks)]))


def metrics(y, scores, supported, weights=None, blocks=None):
    if weights is None:
        weights = np.ones(len(y))
    if weights.sum() == 0:
        return None
    aps = []
    for c in supported:
        binary = y == c
        if weights[binary].sum() == 0 or weights[~binary].sum() == 0:
            return None
        aps.append(float(average_precision_score(binary, scores[:, c], sample_weight=weights)))
    if blocks is None:
        blocks = np.arange(len(y))
    return {'query_ap': block_average(reciprocal_rank(y, scores), blocks),
            'macro_ap': float(np.mean(aps)),
            'top1': float(np.average(scores.argmax(1) == y, weights=weights)),
            'class_ap': {CLASSES[c]: a for c, a in zip(supported, aps)}}


def geometry(x, y, blocks, text):
    # Each parent block has equal weight within a category, irrespective of scene count.
    x = np.asarray(x, dtype=np.float64)
    means = []
    for c in range(N_CLASSES):
        keep = y == c
        by_block = [unit(x[keep & (blocks == b)].mean(0)) for b in np.unique(blocks[keep])]
        means.append(np.mean(by_block, axis=0))
    means = unit(np.asarray(means))
    tri = np.triu_indices(N_CLASSES, 1)
    image_dist = (1 - means @ means.T)[tri]
    out = {}
    for ref in REFS:
        t = unit(text[ref])
        dist = (1 - t @ t.T)[tri]
        rho = spearmanr(image_dist, dist).statistic
        out[ref] = {'spearman_rho': float(rho) if np.isfinite(rho) else None,
                    'image_distances': image_dist.tolist(), 'text_distances': dist.tolist()}
    return out


def evaluate_model(x, y, splits, blocks, text):
    test = splits == 'test'
    result = {'geometry': geometry(x[test], y[test], blocks[test], text),
              'known': {}, 'heldout': {ref: [] for ref in REFS}}
    cached = {'known': {}, 'heldout': {ref: [] for ref in REFS}}
    for held in [None] + FOLDS:
        allowed = np.ones(len(y), dtype=bool) if held is None else ~np.isin(y, held)
        fit = (splits == 'train') & allowed
        val = (splits == 'val') & allowed
        ev = test if held is None else test & np.isin(y, held)
        supported = list(range(N_CLASSES)) if held is None else list(held)
        val_classes = list(range(N_CLASSES)) if held is None else [c for c in range(N_CLASSES) if c not in held]
        onehot = np.eye(N_CLASSES)[y[fit]]
        best = {ref: (-np.inf, None, None) for ref in REFS}
        candidates = {ref: [] for ref in REFS}
        for alpha, pv, pt in ridge_predictions(x[fit], onehot, x[val], x[ev]):
            for ref in REFS:
                vs = language_scores(pv, text[ref])
                vm = metrics(y[val], vs, val_classes, blocks=blocks[val])
                score = vm['query_ap']
                candidates[ref].append({'alpha': alpha, **vm})
                if score > best[ref][0]:
                    best[ref] = (score, alpha, language_scores(pt, text[ref]))
        for ref in REFS:
            _, alpha, scores = best[ref]
            m = {'alpha': alpha, 'validation_candidates': candidates[ref],
                 'n_train': int(fit.sum()), 'n_validation': int(val.sum()), 'n_test': int(ev.sum()),
                 'test_blocks': int(len(np.unique(blocks[ev]))),
                 'supported_classes': [CLASSES[c] for c in supported],
                 **metrics(y[ev], scores, supported, blocks=blocks[ev])}
            cache = (y[ev], scores, blocks[ev], supported)
            if held is None:
                result['known'][ref] = m
                cached['known'][ref] = [cache]
            else:
                result['heldout'][ref].append(m)
                cached['heldout'][ref].append(cache)
    result['heldout_summary'] = {ref: {
        key: float(np.mean([m[key] for m in result['heldout'][ref]]))
        for key in ('query_ap', 'macro_ap', 'top1')} for ref in REFS}
    return result, cached


def paired_bootstrap(all_cache, test_blocks, repeats, seed):
    """Resample parent blocks; all maps and validation choices remain fixed."""
    unique = np.unique(test_blocks)
    rng = np.random.default_rng(seed)
    # Common draws preserve pairing across every model and language reference.
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=repeats)
    def bootstrap_query_ap(cache):
        values = np.zeros(repeats)
        valid = np.ones(repeats, dtype=bool)
        for y, scores, b, supported in cache:
            rr = reciprocal_rank(y, scores)
            present = np.unique(b)
            indices = np.searchsorted(unique, present)
            means = np.array([rr[b == block].mean() for block in present])
            weights = draws[:, indices]
            totals = weights.sum(1)
            valid &= totals > 0
            values += (weights @ means) / np.maximum(totals, 1) / len(cache)
            for c in supported:
                category_blocks = np.searchsorted(unique, np.unique(b[y == c]))
                valid &= draws[:, category_blocks].sum(1) > 0
        values[~valid] = np.nan
        return values
    values = {model: {track: {ref: bootstrap_query_ap(cache[track][ref])
                for ref in REFS} for track in ('known', 'heldout')}
                for model, cache in all_cache.items()}
    out = {}
    for model in values:
        if model == 'appearance':
            continue
        out[model] = {}
        for track in ('known', 'heldout'):
            out[model][track] = {}
            for ref in REFS:
                diff = values[model][track][ref] - values['appearance'][track][ref]
                finite = diff[np.isfinite(diff)]
                out[model][track][ref] = {
                    'conditional_query_ap_difference_interval': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                    'valid_resamples': int(len(finite)), 'invalid_resamples': int(repeats - len(finite))}
    return out


def plots(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    names = list(results['models'])
    fig, axs = plt.subplots(1, 3, figsize=(13, max(4, len(names) * .32)), sharey=True)
    for ax, ref in zip(axs, REFS):
        vals = [results['models'][m]['geometry'][ref]['spearman_rho'] for m in names]
        ax.scatter(vals, range(len(names)), color='#415d73', s=30)
        ax.axvline(0, color='.75', linewidth=.7)
        ax.set(title=ref.replace('_', ' '), xlabel='Category-distance Spearman correlation', xlim=(-1, 1))
    axs[0].set_yticks(range(len(names)), names)
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'geometry.png', dpi=200)
    plt.close(fig)
    names = [m for m in names if m != 'appearance']
    fig, axs = plt.subplots(2, 3, figsize=(13, max(7, len(names) * .55)), sharey=True)
    for row, track in enumerate(('known', 'heldout')):
        for ax, ref in zip(axs[row], REFS):
            key = 'known' if track == 'known' else 'heldout_summary'
            base = results['models']['appearance'][key][ref]['query_ap']
            for i, model in enumerate(names):
                delta = results['models'][model][key][ref]['query_ap'] - base
                ci = results['bootstrap'][model][track][ref]['conditional_query_ap_difference_interval']
                if ci:
                    ax.plot(ci, [i, i], color='#718494', lw=1.5)
                ax.scatter([delta], [i], color='#233f53', s=20)
            ax.axvline(0, color='.6', lw=.8)
            ax.set(title=f'{track.title()} categories | {ref.replace("_", " ")}', xlabel='Per-query AP difference from appearance')
        axs[row, 0].set_yticks(range(len(names)), names)
    axs[0, 0].invert_yaxis()
    fig.suptitle('Fixed-map geographic block bootstrap: conditional 95% intervals', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'transfer-vs-appearance.png', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    p.add_argument('--seed', type=int, default=51426)
    args = p.parse_args()
    root = args.root
    raw = json.loads((root / 'data/manifest.json').read_text())
    rows = raw if isinstance(raw, list) else raw['rows']
    validate_manifest(rows)
    geographic_minima = validate_geography(rows)
    admission = json.loads((root/'results/aggregate-report.json').read_text())
    if not admission.get('admitted', False):
        raise ValueError('Global aggregate has not passed admission')
    y = np.array([CLASSES.index(r['label']) for r in rows])
    splits = np.array([r['split'] for r in rows])
    blocks = np.array([str(r['block']) for r in rows])
    with np.load(root / 'data/text.npz') as z:
        text = {ref: z[ref] for ref in REFS}
    for ref, values in text.items():
        if values.shape[0] != N_CLASSES or not np.isfinite(values).all():
            raise ValueError(f'Invalid text reference {ref}')
    paths = discover_features(root, args.pooling, len(rows), sha(root / 'data/manifest.json'))
    results = {'author': 'James Edward Ball', 'dataset': 'Global held-out geographic validation with automated WorldCover references',
               'classes': CLASSES, 'manifest_sha256': sha(root / 'data/manifest.json'),
               'text_sha256': sha(root / 'data/text.npz'), 'seed': args.seed,
               'analysis_sha256': sha(Path(__file__)),
               'protocol_sha256': sha(Path(__file__).with_name('global-protocol.md')),
               'cross_split_minimum_km': geographic_minima,
               'bootstrap_resamples': args.bootstrap, 'pooling': args.pooling, 'models': {},
               'reference_type': 'automated_worldcover_2021_v200', 'expert_labels': False,
               'interpretation': 'Automated weak-reference agreement; reference error and temporal mismatch remain possible. Conditional fixed-map uncertainty; no causal, composition, pretraining novelty or general semantic capability claim.'}
    cache = {}
    for name, path in paths.items():
        print(f'Analysing {name}', flush=True)
        x = np.load(path, mmap_mode='r')
        if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
            raise ValueError(f'Invalid features: {path}')
        results['models'][name], cache[name] = evaluate_model(x, y, splits, blocks, text)
        results['models'][name]['feature_sha256'] = sha(path)
        if name != 'appearance':
            results['models'][name]['extraction_metadata'] = json.loads((path.parent / 'complete.json').read_text())
    print('Paired block bootstrap', flush=True)
    results['bootstrap'] = paired_bootstrap(cache, blocks[splits == 'test'], args.bootstrap, args.seed)
    out = root / 'results' / args.pooling
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    plots(results, out)
    print(f'Wrote {out}', flush=True)


if __name__ == '__main__':
    main()
