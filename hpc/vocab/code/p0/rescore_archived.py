#!/usr/bin/env python3
"""P0 rescoring of the archived global features (MegaEarth Amendment A2, O1/O2).

Author: James Edward Ball.

CPU only, NumPy only. Re-fits the frozen global ridge readout exactly as
`hpc/supplement-controls/code/controls.py` did (train-only median imputation and
scaling, then the eigen-ridge of `hpc/global/code/analyse.py`, the genuine target
fitted jointly with the three block-shuffle and three category-permutation nulls,
alpha in {1, 10, 100, 1000} chosen by block-averaged validation GZSL reciprocal rank,
first maximum wins) and then rescores the *same fitted maps* under the A2 rules:

* GZSL (as published, k = 8 gallery). Parity gate: every archived per-image reciprocal
  rank (21 ridge-type routes x 5 folds x 16 models), every archived fold `test_query_ap`
  and every selected alpha must be reproduced within `--parity-tol` (default 1e-6).
* Held-out-only ZSL (k = held-out gallery = 2 per archived fold; chance H_2/2 = 0.75).
* Hub-free variants of both (prediction minus training-target mean, scored against
  gallery-centred prototypes).
* Seen competence: seen-class test windows ranked among the fold's seen classes (k = 6)
  by the same fitted map and scoring rule.
* Nearest-seen-class (NSC) decomposition: held-out classes ranked by the text similarity
  between the predicted seen class and each held-out class.
* Seen-competence-matched contrast (O1): parallel-lines (base fixed-effect) slope of
  ZSL nu on seen nu across training-only PCA truncations of the control bases; the
  adjusted contrast of pretrained vs control c is (z_a - z_c) - beta * (s_a - s_c).

Everything here is descriptive evidence about the *published* cohort. It is not
MegaEarth evidence and cannot change any MegaEarth verdict.

Usage on the HPC (inside a Slurm CPU job; never on the login node):
    python rescore_archived.py --base /mnt/scratch/users/40021461/geofm-semantic-expansion \
        --out <base>/vocab-01/results/p0-rescore
Local synthetic check (< 30 s):
    python rescore_archived.py --selftest
"""
import os

# Same thread pinning as the archived controls run, so BLAS summation order matches.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

AUTHOR = 'James Edward Ball'
CLASSES = ['Water', 'Trees', 'Grass', 'Flooded_Vegetation', 'Crops', 'Scrub', 'Built_Area', 'Bare_Ground']
N_CLASSES = len(CLASSES)
FOLDS = [None, (0, 1), (2, 3), (4, 5), (6, 7)]  # fold 0 = familiar categories (no exclusion)
REFS = ['qwen_instructed', 'qwen_plain', 'mpnet_plain']
ALPHAS = [1., 10., 100., 1000.]
SEEDS = (101, 202, 303)
ARCHS = ('clay', 'prithvi', 'prithvi100')
RANDOMS = [f'random_{a}_{s}' for a in ARCHS for s in SEEDS]
ORIGINAL = ['appearance', 'clay', 'prithvi', 'prithvi100'] + RANDOMS
DERIVED = ['rich_appearance', 'metadata', 'rich_metadata']
MODELS = ORIGINAL + DERIVED
CONTROL_BASES = ['appearance', 'rich_appearance', 'metadata', 'rich_metadata'] + RANDOMS
NAMED_CONTROLS = ['random_mean', 'rich_appearance', 'rich_metadata', 'appearance', 'metadata']
RULES = ('cosine', 'hubfree')
NULL_ROUTES = ['ridge'] + [f'block_shuffle_{s}' for s in SEEDS] + [f'category_permutation_{s}' for s in SEEDS]
DEFAULT_DIMS = (4, 8, 16, 32, 64)
EXPECTED_MANIFEST_SHA = '9ad0d7c52eb8676d790a65217c800e05dff21f1389e80655e956eba55af671e2'
HPC_BASE = '/mnt/scratch/users/40021461/geofm-semantic-expansion'


# ----------------------------------------------------------------------------- legacy numerics
def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def harmonic(k):
    return float(np.sum(1.0 / np.arange(1, k + 1)))


def chance_mrr(k):
    return harmonic(k) / k


def language_scores(pred, text):
    """Verbatim `hpc/global/code/analyse.py::language_scores`."""
    text = unit(text)
    gram = text @ text.T
    norm = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', pred, gram, pred), 1e-24))
    return (pred @ gram) / norm[:, None]


def reciprocal_rank(y, scores):
    """Verbatim legacy expected RR under uniform tie-breaking; y indexes columns of scores."""
    target = scores[np.arange(len(y)), y]
    above = (scores > target[:, None]).sum(1)
    tied = (scores == target[:, None]).sum(1)
    h = np.r_[0., np.cumsum(1. / np.arange(1, scores.shape[1] + 1))]
    return (h[above + tied] - h[above]) / tied


def expected_top1(y, scores):
    target = scores[np.arange(len(y)), y]
    above = (scores > target[:, None]).sum(1)
    tied = (scores == target[:, None]).sum(1)
    return (above == 0) / tied


def block_average(values, blocks):
    """Verbatim legacy: mean over blocks of the within-block mean."""
    return float(np.mean([values[blocks == b].mean() for b in np.unique(blocks)]))


def fit_scaler(train):
    """`controls.py::scaled` split into fit/apply (elementwise, so subsets are exact)."""
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0.
    a = np.where(np.isfinite(train), train, median)
    mean = a.mean(0)
    std = a.std(0)
    std[std < 1e-8] = 1.
    return median, mean, std


def apply_scaler(x, scaler):
    median, mean, std = scaler
    return (np.where(np.isfinite(x), x, median) - mean) / std


def shuffle_within_blocks(y, blocks, seed):
    shuffled = y.copy()
    rng = np.random.default_rng(seed)
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        shuffled[ix] = rng.permutation(y[ix])
    return shuffled


class EigenRidge:
    """`analyse.py::ridge_predictions` with the per-alpha weights exposed.

    Applying `predict` to a row subset reproduces the legacy `v @ w + ym` exactly
    because standardisation is elementwise and each subset is multiplied separately.
    """

    def __init__(self, xfit, yfit):
        self.mean = xfit.mean(0, dtype=np.float64)
        scale = xfit.std(0, dtype=np.float64)
        scale[scale < 1e-8] = 1.
        self.scale = scale
        x = (np.asarray(xfit, dtype=np.float64) - self.mean) / self.scale
        self.ym = yfit.mean(0)
        yc = yfit - self.ym
        self.primal = x.shape[1] <= x.shape[0]
        if self.primal:
            self.eigen, self.u = np.linalg.eigh(x.T @ x)
            self.rhs = self.u.T @ (x.T @ yc)
        else:
            self.eigen, self.u = np.linalg.eigh(x @ x.T)
            self.rhs = self.u.T @ yc
            self.x = x

    def weights(self, alpha):
        if self.primal:
            return self.u @ (self.rhs / (np.maximum(self.eigen, 0)[:, None] + alpha))
        return self.x.T @ (self.u @ (self.rhs / (np.maximum(self.eigen, 0)[:, None] + alpha)))

    def standardise(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale


# ----------------------------------------------------------------------------- A2 scoring rules
def hubfree_scores(pred, ym, text, gallery):
    """A2 hub-free scoring: (prediction - training-target mean) mapped to text space,
    scored against gallery-centred, re-normalised prototypes. Columns follow `gallery`."""
    t = unit(text)
    z = (np.asarray(pred) - ym) @ t
    proto = t[list(gallery)]
    proto = unit(proto - proto.mean(0)) if len(gallery) > 1 else proto
    return z @ proto.T


def gallery_scores(rule, pred, ym, text, gallery):
    if rule == 'cosine':
        return language_scores(pred, text)[:, list(gallery)]
    return hubfree_scores(pred, ym, text, gallery)


def nsc_scores(rule, pred, ym, text, seen, held):
    """Nearest-seen-class scores for the held-out gallery (hard argmax decomposition)."""
    seen = list(seen)
    jhat = np.asarray(seen)[gallery_scores(rule, pred, ym, text, seen).argmax(1)]
    t = unit(text)
    proto = t[list(held)]
    if rule == 'hubfree' and len(held) > 1:
        proto = unit(proto - proto.mean(0))
    return t[jhat] @ proto.T


def positions(y, gallery):
    lookup = {c: i for i, c in enumerate(gallery)}
    return np.array([lookup[c] for c in y], dtype=np.int64)


# ----------------------------------------------------------------------------- one fold
def pca_projector(a, dim):
    """Training-only PCA on standardised training rows; components sorted by variance."""
    cov = (a.T @ a) / len(a)
    eigen, vecs = np.linalg.eigh(cov)
    order = np.argsort(-eigen, kind='stable')[:dim]
    v = vecs[:, order]
    # Deterministic sign: largest-magnitude loading positive.
    signs = np.sign(v[np.abs(v).argmax(0), np.arange(v.shape[1])])
    signs[signs == 0] = 1.
    return v * signs


def evaluate_fold(x, y, splits, blocks, text, held, dim=None, with_nulls=True):
    """Return archived-style results plus A2 rescoring for one fold and one feature variant."""
    allowed = np.ones(len(y), bool) if held is None else ~np.isin(y, held)
    train = (splits == 'train') & allowed
    val = (splits == 'val') & allowed
    test_all = splits == 'test'
    test_held = test_all if held is None else test_all & np.isin(y, held)
    test_seen = test_all & allowed if held is not None else np.zeros(len(y), bool)
    if not train.any() or not val.any() or not test_held.any() or (held is not None and not test_seen.any()):
        raise ValueError('fold requires non-empty fitting, validation, held-out test and seen-test cohorts')
    xa = np.asarray(x[train], float)
    scaler = fit_scaler(xa)
    a = apply_scaler(xa, scaler)
    v = apply_scaler(np.asarray(x[val], float), scaler)
    t = apply_scaler(np.asarray(x[test_held], float), scaler)
    ts = apply_scaler(np.asarray(x[test_seen], float), scaler) if test_seen.any() else None
    if dim is not None:
        proj = pca_projector(a, dim)
        a, v, t = a @ proj, v @ proj, t @ proj
        ts = ts @ proj if ts is not None else None
    fit_labels = [y[train]]
    category_maps = []
    if with_nulls:
        fit_labels += [shuffle_within_blocks(y[train], blocks[train], s) for s in SEEDS]
        for s in SEEDS:
            classes = np.unique(y[train])
            perm = np.random.default_rng(s).permutation(classes)
            lookup = dict(zip(classes, perm))
            category_maps.append({str(k): int(vv) for k, vv in lookup.items()})
            fit_labels.append(np.array([lookup[c] for c in y[train]]))
    targets = np.concatenate([np.eye(N_CLASSES)[lab] for lab in fit_labels], axis=1)
    ridge = EigenRidge(a, targets)
    vs, tsd = ridge.standardise(v), ridge.standardise(t)
    tss = ridge.standardise(ts) if ts is not None else None
    routes = NULL_ROUTES if with_nulls else ['ridge']
    best = {}
    for alpha in ALPHAS:
        w = ridge.weights(alpha)
        pv = vs @ w + ridge.ym
        pt = tsd @ w + ridge.ym
        ps = tss @ w + ridge.ym if tss is not None else None
        for r, route in enumerate(routes):
            sl = slice(r * N_CLASSES, (r + 1) * N_CLASSES)
            for ref in REFS:
                value = block_average(reciprocal_rank(y[val], language_scores(pv[:, sl], text[ref])), blocks[val])
                key = (route, ref)
                if key not in best or value > best[key][0]:
                    best[key] = (value, alpha, pt[:, sl].copy(), None if ps is None else ps[:, sl].copy(),
                                 ridge.ym[sl].copy())
    held_list = None if held is None else list(held)
    seen_list = [c for c in range(N_CLASSES) if held is None or c not in held]
    out = {'support': {'train': int(train.sum()), 'validation': int(val.sum()), 'test_held': int(test_held.sum()),
                       'test_seen': int(test_seen.sum()), 'held': held_list, 'dim': dim,
                       'category_permutations': category_maps},
           'rows_held': np.flatnonzero(test_held), 'rows_seen': np.flatnonzero(test_seen), 'routes': {}}
    yh, ys = y[test_held], y[test_seen]
    for (route, ref), (value, alpha, pt, ps, ym) in best.items():
        rec = {'alpha': alpha, 'validation_query_ap': value,
               'gzsl_rr': reciprocal_rank(yh, language_scores(pt, text[ref]))}
        if held is not None and route == 'ridge':
            for rule in RULES:
                zs = gallery_scores(rule, pt, ym, text[ref], held_list)
                rec[f'zsl_rr_{rule}'] = reciprocal_rank(positions(yh, held_list), zs)
                rec[f'zsl_top1_{rule}'] = expected_top1(positions(yh, held_list), zs)
                ns = nsc_scores(rule, pt, ym, text[ref], seen_list, held_list)
                rec[f'nsc_rr_{rule}'] = reciprocal_rank(positions(yh, held_list), ns)
                if ps is not None and len(ps):
                    ss = gallery_scores(rule, ps, ym, text[ref], seen_list)
                    rec[f'seen_rr_{rule}'] = reciprocal_rank(positions(ys, seen_list), ss)
            if dim is None:
                rec['gzsl_rr_hubfree'] = reciprocal_rank(yh, hubfree_scores(pt, ym, text[ref], list(range(N_CLASSES))))
        out['routes'][(route, ref)] = rec
    return out


# ----------------------------------------------------------------------------- aggregation
class Panel:
    """Cell x class tables and crossed (cell, class) bootstrap aggregation.

    Class-balanced (A1) aggregation: windows -> mean within (cell, class) -> mean over the
    class's cells -> mean over classes. In P0 every class is its own family.
    """

    def __init__(self, test_blocks, n_boot, seed):
        self.blocks = np.unique(test_blocks)
        rng = np.random.default_rng(seed)
        nb, nc = len(self.blocks), N_CLASSES
        ones = np.ones((1, nb)), np.ones((1, nc))
        if n_boot:
            cc = rng.multinomial(nb, np.full(nb, 1 / nb), size=n_boot).astype(float)
            cf = rng.multinomial(nc, np.full(nc, 1 / nc), size=n_boot).astype(float)
            self.cc = np.vstack([ones[0], cc])
            self.cf = np.vstack([ones[1], cf])
        else:
            self.cc, self.cf = ones
        # Ordinary block-only bootstrap (as published): class weights fixed.
        self.cf_block = np.ones_like(self.cf)

    def table(self, values, rows, blocks, labels):
        bi = np.searchsorted(self.blocks, blocks[rows])
        s = np.zeros((len(self.blocks), N_CLASSES))
        n = np.zeros_like(s)
        np.add.at(s, (bi, labels[rows]), values)
        np.add.at(n, (bi, labels[rows]), 1.)
        return s, n

    def class_balanced(self, s, n, classes, block_only=False):
        """Return (B+1,) draws of the class-balanced mean; index 0 is the point estimate."""
        mask = (n > 0).astype(float)
        mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
        num = self.cc @ (mean * mask)
        den = self.cc @ mask
        cls = np.zeros(N_CLASSES, bool)
        cls[list(classes)] = True
        cf = (self.cf_block if block_only else self.cf)[:, cls]
        present = den[:, cls] > 0
        cmean = np.divide(num[:, cls], den[:, cls], out=np.zeros_like(num[:, cls]), where=present)
        support = (mask[:, cls].sum(0) > 0)
        # A draw is invalid if a supported class with family weight > 0 lost all its cells.
        invalid = ((cf > 0) & ~present & support[None, :]).any(1)
        wsum = (cf * present).sum(1)
        out = np.divide((cf * present * cmean).sum(1), wsum, out=np.full(len(wsum), np.nan), where=wsum > 0)
        out[invalid] = np.nan
        return out


def summarise_draws(draws):
    point = float(draws[0])
    boot = draws[1:]
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {'estimate': point, 'valid_draws': 0}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    centred = boot - point
    p_gt0 = float((1 + np.sum(centred >= point)) / (1 + len(boot)))
    return {'estimate': point, 'percentile_95': [float(lo), float(hi)],
            'basic_95': [float(2 * point - hi), float(2 * point - lo)],
            'null_centred_p_gt0': p_gt0, 'valid_draws': int(len(boot))}


def fe_slope(z, s, base_ids):
    """Parallel-lines (base fixed-effect) OLS slope of z on s; z, s shape (B, P)."""
    zt = np.zeros_like(z)
    st = np.zeros_like(s)
    for b in np.unique(base_ids):
        m = base_ids == b
        zt[:, m] = z[:, m] - z[:, m].mean(1, keepdims=True)
        st[:, m] = s[:, m] - s[:, m].mean(1, keepdims=True)
    den = (st * st).sum(1)
    return np.divide((st * zt).sum(1), den, out=np.full(len(den), np.nan), where=den > 0)


def pooled_line(z, s):
    sm, zm = s.mean(1, keepdims=True), z.mean(1, keepdims=True)
    den = ((s - sm) ** 2).sum(1)
    b1 = np.divide(((s - sm) * (z - zm)).sum(1), den, out=np.full(len(den), np.nan), where=den > 0)
    b0 = zm[:, 0] - b1 * sm[:, 0]
    return b0, b1


# ----------------------------------------------------------------------------- IO
def load_manifest(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    rows = data if isinstance(data, list) else data['rows']
    classes = CLASSES if isinstance(data, list) else data.get('classes', CLASSES)
    if list(classes) != CLASSES:
        raise ValueError('Unexpected class order in manifest')
    y = np.array([CLASSES.index(r['label']) for r in rows])
    splits = np.array([r['split'] for r in rows])
    blocks = np.array([str(r['block']) for r in rows])
    return y, splits, blocks, digest


def load_text(path):
    with np.load(path) as z:
        return {ref: np.asarray(z[ref], dtype=np.float64) for ref in REFS}


def feature_paths(global_root, controls_root):
    paths = {'appearance': Path(global_root) / 'data/appearance.npy'}
    for name in ORIGINAL[1:]:
        paths[name] = Path(global_root) / 'features' / name / 'mean.npy'
    for name in DERIVED:
        paths[name] = Path(controls_root) / f'{name}.npy'
    return paths


def load_archive_fold(archive, model, fold):
    d = Path(archive) / model
    meta_path, score_path = d / f'fold-{fold}.json', d / f'fold-{fold}-scores.npz'
    if not meta_path.exists() or not score_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if sha_file(score_path) != meta['scores_sha256']:
        raise ValueError(f'Archived score checksum mismatch: {score_path}')
    with np.load(score_path) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def parity_fold(result, archived, y, blocks, tol):
    """Compare a recomputed fold with its archive; returns a dict of maxima and failures."""
    meta, arrays = archived
    rows = result['rows_held']
    failures = []
    if not np.array_equal(arrays['test_rows'], rows):
        failures.append('test_rows differ')
        return {'max_rr': float('inf'), 'max_query_ap': float('inf'), 'checked': 0, 'failures': failures}
    if not np.array_equal(arrays['test_labels'], y[rows]) or not np.array_equal(arrays['test_blocks'].astype(str), blocks[rows]):
        failures.append('test labels/blocks differ')
    max_rr = max_ap = 0.
    checked = 0
    for (route, ref), rec in result['routes'].items():
        key = f'{route}__{ref}'
        if key not in arrays or f'{route}/{ref}' not in meta.get('result', {}):
            failures.append(f'{key}: missing archived score array or summary')
            continue
        if arrays[key].shape != rec['gzsl_rr'].shape:
            failures.append(f'{key}: score shape differs')
            continue
        if not np.isfinite(arrays[key]).all() or not np.isfinite(rec['gzsl_rr']).all():
            failures.append(f'{key}: non-finite reciprocal ranks')
            continue
        diff = float(np.max(np.abs(rec['gzsl_rr'] - arrays[key]))) if len(rows) else 0.
        archived_ap = meta['result'][f'{route}/{ref}']['test_query_ap']
        ap = block_average(rec['gzsl_rr'], blocks[rows])
        ap_diff = abs(ap - archived_ap)
        if meta['result'][f'{route}/{ref}']['selected'] != rec['alpha']:
            failures.append(f'{key}: alpha {rec["alpha"]} vs archived {meta["result"][f"{route}/{ref}"]["selected"]}')
        if not np.isfinite(ap_diff) or diff > tol or ap_diff > tol:
            failures.append(f'{key}: max|dRR|={diff:.3g} |dAP|={ap_diff:.3g}')
        max_rr, max_ap = max(max_rr, diff), max(max_ap, ap_diff)
        checked += 1
    return {'max_rr': max_rr, 'max_query_ap': max_ap, 'checked': checked, 'failures': failures}


# ----------------------------------------------------------------------------- driver
def run(features, y, splits, blocks, text, archive=None, n_boot=1000, seed=51426, dims=DEFAULT_DIMS,
        parity_tol=1e-6, models=None, log=print):
    """Rescore every model. `features` maps model name -> array-like (n, d)."""
    t0 = time.time()
    if models is not None and any(m not in features for m in models):
        raise ValueError('requested model features are missing')
    if not np.isfinite(parity_tol) or parity_tol < 0:
        raise ValueError('parity tolerance must be finite and non-negative')
    if n_boot < 0 or any(d <= 0 for d in dims):
        raise ValueError('bootstrap count must be non-negative and PCA dimensions positive')
    models = [m for m in (models or MODELS) if m in features]
    if not models:
        raise ValueError('no models available')
    test_blocks = blocks[splits == 'test']
    panel = Panel(test_blocks, n_boot, seed)
    held_folds = [f for f in range(1, len(FOLDS))]
    zsl_tables, seen_tables, nsc_tables, gzsl_tables = {}, {}, {}, {}
    published, parity, fold_meta = {}, {}, {}
    variants = {}

    def store(model, dim, fold, res):
        for (route, ref), rec in res['routes'].items():
            if route != 'ridge' or fold == 0:
                continue
            for rule in RULES:
                key = (model, dim, ref, rule)
                zsl_tables.setdefault(key, [np.zeros((len(panel.blocks), N_CLASSES)), np.zeros((len(panel.blocks), N_CLASSES))])
                s, n = panel.table(rec[f'zsl_rr_{rule}'], res['rows_held'], blocks, y)
                zsl_tables[key][0] += s
                zsl_tables[key][1] += n
                nsc_tables.setdefault(key, [np.zeros_like(s), np.zeros_like(s)])
                s2, n2 = panel.table(rec[f'nsc_rr_{rule}'], res['rows_held'], blocks, y)
                nsc_tables[key][0] += s2
                nsc_tables[key][1] += n2
                if f'seen_rr_{rule}' in rec:
                    seen_tables.setdefault(key, {})[fold] = panel.table(rec[f'seen_rr_{rule}'], res['rows_seen'], blocks, y)
            if dim is None:
                for rule, field in (('cosine', 'gzsl_rr'), ('hubfree', 'gzsl_rr_hubfree')):
                    key = (model, None, ref, rule)
                    gzsl_tables.setdefault(key, [np.zeros((len(panel.blocks), N_CLASSES)), np.zeros((len(panel.blocks), N_CLASSES))])
                    s, n = panel.table(rec[field], res['rows_held'], blocks, y)
                    gzsl_tables[key][0] += s
                    gzsl_tables[key][1] += n

    for model in models:
        x = features[model]
        width = x.shape[1]
        model_dims = [None] + [d for d in dims if d < width]
        variants[model] = model_dims
        published[model] = {ref: {'known': None, 'heldout_folds': []} for ref in REFS}
        parity[model] = {'checked': 0, 'max_rr': 0., 'max_query_ap': 0., 'failures': [], 'archived_folds': 0}
        for fold, held in enumerate(FOLDS):
            for dim in model_dims:
                if dim is not None and held is None:
                    continue
                res = evaluate_fold(x, y, splits, blocks, text, held, dim=dim, with_nulls=dim is None)
                if dim is None:
                    fold_meta.setdefault(model, {})[fold] = {**res['support'], 'alpha': {
                        f'{r}/{ref}': rec['alpha'] for (r, ref), rec in res['routes'].items()}}
                    for ref in REFS:
                        rec = res['routes'][('ridge', ref)]
                        value = block_average(rec['gzsl_rr'], blocks[res['rows_held']])
                        if held is None:
                            published[model][ref]['known'] = value
                        else:
                            published[model][ref]['heldout_folds'].append(value)
                    if archive is not None:
                        archived = load_archive_fold(archive, model, fold)
                        if archived is not None:
                            p = parity_fold(res, archived, y, blocks, parity_tol)
                            parity[model]['archived_folds'] += 1
                            parity[model]['checked'] += p['checked']
                            parity[model]['max_rr'] = max(parity[model]['max_rr'], p['max_rr'])
                            parity[model]['max_query_ap'] = max(parity[model]['max_query_ap'], p['max_query_ap'])
                            parity[model]['failures'] += [f'fold {fold}: {m}' for m in p['failures']]
                        else:
                            parity[model]['failures'].append(f'fold {fold}: missing archived files')
                store(model, dim, fold, res)
        log(f'[p0] {model}: dims {model_dims} done at {time.time() - t0:.1f}s')

    for model in published:
        for ref in REFS:
            folds = published[model][ref]['heldout_folds']
            published[model][ref]['heldout'] = float(np.mean(folds)) if folds else None

    # ---------------------------------------------------------------- aggregate
    ch2, ch6, ch8 = chance_mrr(2), chance_mrr(6), chance_mrr(8)
    to_nu = {'zsl': lambda m: (m - ch2) / (1 - ch2), 'seen': lambda m: (m - ch6) / (1 - ch6),
             'gzsl': lambda m: (m - ch8) / (1 - ch8)}
    agg = {}

    def zsl_draws(key, block_only=False):
        s, n = zsl_tables[key]
        return panel.class_balanced(s, n, range(N_CLASSES), block_only)

    def seen_draws(key, block_only=False):
        per_fold = []
        for fold in held_folds:
            s, n = seen_tables[key][fold]
            seen = [c for c in range(N_CLASSES) if c not in FOLDS[fold]]
            per_fold.append(panel.class_balanced(s, n, seen, block_only))
        return np.mean(per_fold, axis=0)

    for (model, dim, ref, rule) in zsl_tables:
        key = (model, dim, ref, rule)
        z = zsl_draws(key)
        s = seen_draws(key) if key in seen_tables else np.full_like(z, np.nan)
        ns = panel.class_balanced(*nsc_tables[key], range(N_CLASSES))
        agg[key] = {'zsl_mrr': z, 'zsl_nu': to_nu['zsl'](z), 'seen_mrr': s, 'seen_nu': to_nu['seen'](s),
                    'nsc_mrr': ns, 'nsc_nu': to_nu['zsl'](ns)}
        if dim is None:
            g = panel.class_balanced(*gzsl_tables[key], range(N_CLASSES))
            agg[key]['gzsl_class_mrr'] = g
            agg[key]['gzsl_nu'] = to_nu['gzsl'](g)

    def random_mean(arch, ref, rule, field, dim=None):
        return np.mean([agg[(f'random_{arch}_{s}', dim, ref, rule)][field] for s in SEEDS], axis=0)

    # Per-route summaries (point + crossed-bootstrap intervals for the key quantities).
    routes_out = {}
    for model in models:
        routes_out[model] = {}
        for ref in REFS:
            routes_out[model][ref] = {'published_gzsl_block_average': published[model][ref]['heldout'],
                                      'published_known_block_average': published[model][ref]['known']}
            for rule in RULES:
                a = agg[(model, None, ref, rule)]
                z_block = zsl_draws((model, None, ref, rule), block_only=True)
                routes_out[model][ref][rule] = {
                    'zsl_k2_class_balanced_mrr': summarise_draws(a['zsl_mrr']),
                    'zsl_k2_nu': summarise_draws(a['zsl_nu']),
                    'zsl_k2_block_only_bootstrap_mrr': summarise_draws(z_block),
                    'seen_k6_nu': summarise_draws(a['seen_nu']),
                    'nsc_k2_nu': summarise_draws(a['nsc_nu']),
                    'gzsl_k8_class_balanced_mrr': summarise_draws(a['gzsl_class_mrr']),
                }

    # Seen-competence curve and contrasts (O1).
    curve, contrasts = {}, {}
    for ref in REFS:
        curve[ref], contrasts_ref = {}, {}
        for rule in RULES:
            pts = [(b, d) for b in CONTROL_BASES if b in variants for d in variants[b]]
            if not pts:
                continue
            base_ids = np.array([CONTROL_BASES.index(b) for b, _ in pts])
            Z = np.stack([agg[(b, d, ref, rule)]['zsl_nu'] for b, d in pts], 1)
            S = np.stack([agg[(b, d, ref, rule)]['seen_nu'] for b, d in pts], 1)
            beta = fe_slope(Z, S, base_ids)
            b0, b1 = pooled_line(Z, S)
            curve[ref][rule] = {'fe_slope': summarise_draws(beta), 'pooled_intercept': float(b0[0]),
                                'pooled_slope': float(b1[0]), 'points': [
                                    {'base': b, 'dim': d or 'full', 'zsl_nu': float(Z[0, i]), 'seen_nu': float(S[0, i])}
                                    for i, (b, d) in enumerate(pts)],
                                'control_seen_nu_range': [float(np.nanmin(S[0])), float(np.nanmax(S[0]))]}
            for arch in ARCHS:
                if arch not in variants:
                    continue
                pa = agg[(arch, None, ref, rule)]
                za, sa, na = pa['zsl_nu'], pa['seen_nu'], pa['nsc_nu']
                rec = {'pretrained_seen_nu': float(sa[0]), 'pretrained_zsl_nu': float(za[0]),
                       'extrapolation': bool(sa[0] < np.nanmin(S[0]) or sa[0] > np.nanmax(S[0])),
                       'pooled_curve_residual': summarise_draws(za - (b0 + b1 * sa)), 'controls': {}}
                # Pretrained truncation points: matched-support diagnostic.
                rec['pretrained_truncations'] = [
                    {'dim': d or 'full', 'zsl_nu': float(agg[(arch, d, ref, rule)]['zsl_nu'][0]),
                     'seen_nu': float(agg[(arch, d, ref, rule)]['seen_nu'][0])} for d in variants[arch]]
                for control in NAMED_CONTROLS:
                    if control == 'random_mean':
                        if not all(f'random_{arch}_{s}' in variants for s in SEEDS):
                            continue
                        zc, sc, nc = (random_mean(arch, ref, rule, f) for f in ('zsl_nu', 'seen_nu', 'nsc_nu'))
                    elif control in variants:
                        pc = agg[(control, None, ref, rule)]
                        zc, sc, nc = pc['zsl_nu'], pc['seen_nu'], pc['nsc_nu']
                    else:
                        continue
                    raw = za - zc
                    adjusted = raw - beta * (sa - sc)
                    nsc_gain = na - nc
                    share = nsc_gain[0] / raw[0] if abs(raw[0]) > 1e-12 else None
                    rec['controls'][control] = {
                        'raw_zsl_nu_gain': summarise_draws(raw),
                        'competence_adjusted_gain': summarise_draws(adjusted),
                        'seen_nu_gap': float(sa[0] - sc[0]),
                        'nsc_nu_gain': summarise_draws(nsc_gain),
                        'nsc_share_of_raw_gain': None if share is None else float(share),
                        'residual_after_nsc': summarise_draws(raw - nsc_gain)}
                contrasts_ref.setdefault(arch, {})[rule] = rec
        contrasts[ref] = contrasts_ref

    parity_failed = [m for m, p in parity.items() if p['failures']]
    parity_checked = sum(p['checked'] for p in parity.values())
    result = {
        'author': AUTHOR,
        'status': 'P0 descriptive rescoring of the published global cohort; not MegaEarth evidence',
        'chance': {'zsl_k2_mrr': ch2, 'seen_k6_mrr': ch6, 'gzsl_k8_mrr': ch8},
        'settings': {'alphas': ALPHAS, 'refs': REFS, 'rules': list(RULES), 'pca_dims': list(dims),
                     'bootstrap': n_boot, 'seed': seed, 'bootstrap_kind': 'crossed test-block x class multiplicities '
                     '(classes are families in P0); index 0 = point estimate', 'parity_tol': parity_tol,
                     'alpha_selection': 'legacy: block-averaged validation GZSL RR, first maximum wins'},
        'parity': {'passed': (not parity_failed) and parity_checked > 0, 'checked_arrays': parity_checked,
                   'models_failed': parity_failed, 'per_model': parity},
        'fold_support': fold_meta,
        'routes': routes_out,
        'competence_curve': curve,
        'contrasts': contrasts,
        'runtime_s': round(time.time() - t0, 2),
    }
    return result


# ----------------------------------------------------------------------------- reporting
def fmt(d, key='estimate', ci='percentile_95'):
    if d is None:
        return 'n/a'
    if ci in d:
        return f'{d[key]:+.4f} [{d[ci][0]:+.4f}, {d[ci][1]:+.4f}]'
    return f'{d[key]:+.4f}'


def markdown(result, ref='qwen_instructed'):
    lines = [f'# P0 rescoring of the archived global features ({ref})', '',
             f'Status: {result["status"]}. Parity passed: **{result["parity"]["passed"]}** '
             f'({result["parity"]["checked_arrays"]} archived arrays). Chance: ZSL k=2 '
             f'{result["chance"]["zsl_k2_mrr"]:.4f}, GZSL k=8 {result["chance"]["gzsl_k8_mrr"]:.6f}, seen k=6 '
             f'{result["chance"]["seen_k6_mrr"]:.4f}. nu = (MRR - chance)/(1 - chance).', '',
             '| route | GZSL published | ZSL k=2 MRR (cosine) | ZSL nu cosine | ZSL nu hub-free | seen nu (k=6) | NSC nu |',
             '|---|---|---|---|---|---|---|']
    for model, refs in result['routes'].items():
        r = refs[ref]
        c, h = r['cosine'], r['hubfree']
        pub = r['published_gzsl_block_average']
        lines.append(f'| {model} | {pub:.4f} | {c["zsl_k2_class_balanced_mrr"]["estimate"]:.4f} | '
                     f'{fmt(c["zsl_k2_nu"])} | {fmt(h["zsl_k2_nu"])} | {c["seen_k6_nu"]["estimate"]:.3f} | '
                     f'{c["nsc_k2_nu"]["estimate"]:+.3f} |')
    lines += ['', '## Pretrained contrasts (nu units; 95% crossed-bootstrap percentile intervals)', '',
              '| arch | rule | control | raw ZSL gain | competence-adjusted gain | seen gap | NSC share |',
              '|---|---|---|---|---|---|---|']
    for arch, rules in result['contrasts'].get(ref, {}).items():
        for rule, rec in rules.items():
            for control, c in rec['controls'].items():
                share = c['nsc_share_of_raw_gain']
                lines.append(f'| {arch} | {rule} | {control} | {fmt(c["raw_zsl_nu_gain"])} | '
                             f'{fmt(c["competence_adjusted_gain"])} | {c["seen_nu_gap"]:+.3f} | '
                             f'{"n/a" if share is None else f"{share:.2f}"} |')
    for rule, cv in result['competence_curve'].get(ref, {}).items():
        lines.append('')
        lines.append(f'Competence curve ({rule}): FE slope {fmt(cv["fe_slope"])}; pooled line '
                     f'{cv["pooled_intercept"]:+.4f} + {cv["pooled_slope"]:.4f}*s; control seen-nu range '
                     f'[{cv["control_seen_nu_range"][0]:.3f}, {cv["control_seen_nu_range"][1]:.3f}].')
    lines += ['', 'Descriptive only: fitted maps, alphas and vocabulary are conditioned on; intervals omit '
              'fitting/selection uncertainty. These numbers inform A2 design choices; they are not a verdict.', '']
    return '\n'.join(lines)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_outputs(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'p0-rescore.json.tmp'
    tmp.write_text(json.dumps(to_jsonable(result), indent=1, allow_nan=False) + '\n')
    tmp.replace(out / 'p0-rescore.json')
    (out / 'p0-rescore.md').write_text(markdown(result))


# ----------------------------------------------------------------------------- selftest
def legacy_reference_fold(x, y, splits, blocks, text, held):
    """Independent, literal port of the archived controls.py ridge path (no refactoring)."""
    allowed = np.ones(len(y), bool) if held is None else ~np.isin(y, held)
    train = (splits == 'train') & allowed
    val = (splits == 'val') & allowed
    test = (splits == 'test') & (True if held is None else np.isin(y, held))

    def scaled(tr, va, te):
        median = np.nanmedian(tr, axis=0)
        median[~np.isfinite(median)] = 0.
        arrays = [np.where(np.isfinite(q), q, median) for q in (tr, va, te)]
        mean = arrays[0].mean(0)
        std = arrays[0].std(0)
        std[std < 1e-8] = 1.
        return [(q - mean) / std for q in arrays]

    def ridge_predictions(xfit, yfit, xval, xtest):
        mean = xfit.mean(0, dtype=np.float64)
        scale = xfit.std(0, dtype=np.float64)
        scale[scale < 1e-8] = 1.
        xx = (np.asarray(xfit, dtype=np.float64) - mean) / scale
        vv = (np.asarray(xval, dtype=np.float64) - mean) / scale
        tt = (np.asarray(xtest, dtype=np.float64) - mean) / scale
        ym = yfit.mean(0)
        yc = yfit - ym
        eigen, u = np.linalg.eigh(xx.T @ xx)
        rhs = u.T @ (xx.T @ yc)
        for alpha in ALPHAS:
            w = u @ (rhs / (np.maximum(eigen, 0)[:, None] + alpha))
            yield alpha, vv @ w + ym, tt @ w + ym

    a, v, t = scaled(*[np.asarray(x[m], float) for m in (train, val, test)])
    fit_labels = [y[train]] + [shuffle_within_blocks(y[train], blocks[train], s) for s in SEEDS]
    for s in SEEDS:
        classes = np.unique(y[train])
        perm = np.random.default_rng(s).permutation(classes)
        lookup = dict(zip(classes, perm))
        fit_labels.append(np.array([lookup[c] for c in y[train]]))
    targets = np.concatenate([np.eye(N_CLASSES)[lab] for lab in fit_labels], axis=1)
    best = {}
    for alpha, pv, pt in ridge_predictions(a, targets, v, t):
        for route in range(len(fit_labels)):
            key = NULL_ROUTES[route]
            for ref in REFS:
                sc = language_scores(pv[:, route * N_CLASSES:(route + 1) * N_CLASSES], text[ref])
                value = block_average(reciprocal_rank(y[val], sc), blocks[val])
                index = key + '/' + ref
                if index not in best or value > best[index][0]:
                    best[index] = (value, alpha, language_scores(pt[:, route * N_CLASSES:(route + 1) * N_CLASSES], text[ref]))
    result, cache = {}, {}
    for index, (value, selected, scores) in best.items():
        rr = reciprocal_rank(y[test], scores)
        result[index] = {'selected': selected, 'validation_query_ap': value, 'test_query_ap': block_average(rr, blocks[test])}
        cache[index.replace('/', '__')] = rr
    cache.update(test_rows=np.flatnonzero(test), test_blocks=blocks[test], test_labels=y[test])
    return result, cache


def synthetic_world(seed=7, n_blocks=40, per_block=(6, 14), text_dim=24, latent=4):
    """Planted world with low-rank class semantics (held-out text lies near the seen span).

    clay/prithvi carry the latent semantics linearly; prithvi100 is competent on seen
    classes through orthogonal one-hot class codes but carries no semantics (null);
    random encoders and descriptors carry weak random class codes.
    """
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(N_CLASSES, latent))
    text = {ref: unit(lat @ rng.normal(size=(latent, text_dim)) + 0.15 * rng.normal(size=(N_CLASSES, text_dim)))
            for ref in REFS}
    splits_by_block = np.array(['train'] * 24 + ['val'] * 6 + ['test'] * (n_blocks - 30))
    rng.shuffle(splits_by_block)
    y, blocks, splits, loc = [], [], [], []
    for b in range(n_blocks):
        centre = rng.normal(size=3)
        for c in range(N_CLASSES):
            n = rng.integers(per_block[0], per_block[1] + 1)
            y += [c] * n
            blocks += [f'b{b:02d}'] * n
            splits += [splits_by_block[b]] * n
            loc.append(np.repeat(centre[None], n, 0) + 0.05 * rng.normal(size=(n, 3)))
    y, blocks, splits = np.array(y), np.array(blocks), np.array(splits)
    loc = np.concatenate(loc)
    n = len(y)

    def planted(strength, dim, noise):
        proj = rng.normal(size=(latent, dim)) / np.sqrt(latent)
        return strength * lat[y] @ proj + noise * rng.normal(size=(n, dim))

    def coded(strength, dim, noise):
        codes = rng.normal(size=(N_CLASSES, dim))
        return strength * codes[y] + noise * rng.normal(size=(n, dim))

    onehot = np.zeros((n, 40))
    onehot[np.arange(n), y] = 2.5
    feats = {'clay': planted(2.0, 48, 1.0),
             'prithvi': planted(1.0, 40, 1.0) + coded(0.3, 40, 0.0),
             'prithvi100': onehot + rng.normal(size=(n, 40)),
             'appearance': coded(0.5, 12, 1.0),
             'rich_appearance': coded(0.6, 20, 1.0) + planted(0.3, 20, 0.0),
             'metadata': np.c_[unit(loc), rng.normal(size=(n, 2))]}
    feats['rich_metadata'] = np.c_[feats['rich_appearance'], feats['metadata']]
    for arch in ARCHS:
        for s in SEEDS:
            feats[f'random_{arch}_{s}'] = coded(0.4, 32, 1.0)
    return {k: v.astype(np.float32) for k, v in feats.items()}, y, splits, blocks, text


def write_archive(root, feats, y, splits, blocks, text, models):
    for model in models:
        d = Path(root) / model
        d.mkdir(parents=True, exist_ok=True)
        for fold, held in enumerate(FOLDS):
            result, cache = legacy_reference_fold(feats[model], y, splits, blocks, text, held)
            path = d / f'fold-{fold}-scores.npz'
            np.savez_compressed(path, **cache)
            (d / f'fold-{fold}.json').write_text(json.dumps(
                {'author': AUTHOR, 'fingerprint': {'synthetic': True}, 'result': result,
                 'scores_sha256': sha_file(path)}, indent=1))


def selftest(workdir=None, keep=False, n_boot=200, log=print):
    """Synthetic end-to-end check of parity, oracles and the planted O2 world (< 30 s)."""
    t0 = time.time()
    workdir = Path(workdir or (Path.cwd() / 'work-p0-selftest'))
    if workdir.exists():
        shutil.rmtree(workdir)
    checks = {}
    try:
        feats, y, splits, blocks, text = synthetic_world()
        archive = workdir / 'archive'
        write_archive(archive, feats, y, splits, blocks, text, MODELS)
        result = run(feats, y, splits, blocks, text, archive=archive, n_boot=n_boot, dims=(4, 8, 16), log=lambda *_: None)
        write_outputs(result, workdir / 'out')
        checks['parity_passed'] = bool(result['parity']['passed'])
        checks['parity_checked_arrays'] = result['parity']['checked_arrays']
        checks['parity_max_rr'] = max(p['max_rr'] for p in result['parity']['per_model'].values())
        # Tamper detection: a single changed archived RR must fail parity.
        d = archive / 'clay'
        with np.load(d / 'fold-2-scores.npz') as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays['ridge__qwen_instructed'][3] += 1e-3
        meta = json.loads((d / 'fold-2.json').read_text())
        res = evaluate_fold(feats['clay'], y, splits, blocks, text, FOLDS[2])
        checks['tamper_detected'] = bool(parity_fold(res, (meta, arrays), y, blocks, 1e-6)['failures'])
        # Oracles: constant scores give class-balanced held-out-only MRR = H_k/k exactly.
        panel = Panel(blocks[splits == 'test'], 0, 1)
        test = splits == 'test'
        rows = np.flatnonzero(test)
        for k in (2, 3, 8):
            gallery = list(range(k))
            sel = rows[np.isin(y[rows], gallery)]
            const = np.tile(np.arange(k, 0, -1, dtype=float), (len(sel), 1))
            rr = reciprocal_rank(positions(y[sel], gallery), const)
            s, n = panel.table(rr, sel, blocks, y)
            value = panel.class_balanced(s, n, gallery)[0]
            checks[f'constant_oracle_k{k}'] = bool(abs(value - chance_mrr(k)) < 1e-12)
        tied = reciprocal_rank(np.zeros(5, int), np.zeros((5, 7)))
        checks['all_tied_oracle'] = bool(np.allclose(tied, chance_mrr(7), atol=1e-15))
        r = result['routes']
        clay_nu = r['clay']['qwen_instructed']['cosine']['zsl_k2_nu']['estimate']
        null_nu = r['prithvi100']['qwen_instructed']['hubfree']['zsl_k2_nu']['estimate']
        null_seen = r['prithvi100']['qwen_instructed']['hubfree']['seen_k6_nu']['estimate']
        checks['o2_m6_not_forced_to_chance'] = bool(clay_nu > 0.2)
        checks['o2_clay_zsl_nu'] = clay_nu
        checks['null_semantics_near_chance'] = bool(abs(null_nu) < 0.15 and null_seen > 0.3)
        checks['null_zsl_nu'], checks['null_seen_nu'] = null_nu, null_seen
        adj = result['contrasts']['qwen_instructed']['clay']['cosine']['controls']['rich_metadata']['competence_adjusted_gain']
        checks['clay_adjusted_gain_positive'] = bool(adj['estimate'] > 0)
        # One-hot predictions: cosine ZSL scores equal nearest-seen-class scores exactly.
        seen, held = [0, 1, 2, 4, 5, 7], [3, 6]
        onehot = np.eye(N_CLASSES)[np.array(seen)[np.arange(12) % len(seen)]]
        ym = np.full(N_CLASSES, 1 / N_CLASSES)
        a = gallery_scores('cosine', onehot, ym, text['qwen_plain'], held)
        b = nsc_scores('cosine', onehot, ym, text['qwen_plain'], seen, held)
        checks['nsc_equals_zsl_for_onehot'] = bool(np.allclose(a, b, atol=1e-12))
        checks['outputs_written'] = bool((workdir / 'out/p0-rescore.json').exists() and (workdir / 'out/p0-rescore.md').exists())
    finally:
        if not keep and workdir.exists():
            shutil.rmtree(workdir)
    checks['runtime_s'] = round(time.time() - t0, 2)
    checks['fast_enough'] = bool(checks['runtime_s'] < 30)
    flags = [k for k, v in checks.items() if isinstance(v, bool)]
    checks['passed'] = all(checks[k] for k in flags)
    log(json.dumps(to_jsonable(checks), indent=1))
    return checks


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--selftest', action='store_true', help='synthetic local check (<30 s), no HPC data')
    p.add_argument('--keep', action='store_true', help='keep the selftest work directory')
    p.add_argument('--base', type=Path, default=Path(HPC_BASE))
    p.add_argument('--global-root', type=Path, help='default: <base>/global-01')
    p.add_argument('--controls-root', type=Path, help='default: <base>/supplement-controls/results/global')
    p.add_argument('--archive', type=Path, help='archived fold-*.json/-scores.npz (default: controls root)')
    p.add_argument('--manifest', type=Path, help='default: <global-root>/data/manifest.json')
    p.add_argument('--text', type=Path, help='default: <global-root>/data/text.npz')
    p.add_argument('--out', type=Path, help='default: <base>/vocab-01/results/p0-rescore')
    p.add_argument('--models', nargs='*', choices=MODELS)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=51426)
    p.add_argument('--dims', type=int, nargs='*', default=list(DEFAULT_DIMS))
    p.add_argument('--parity-tol', type=float, default=1e-6)
    p.add_argument('--no-feature-hash', action='store_true', help='skip feature sha256 vs archive fingerprint')
    p.add_argument('--allow-parity-failure', action='store_true')
    args = p.parse_args(argv)
    if args.selftest:
        return 0 if selftest(keep=args.keep)['passed'] else 1
    if os.environ.get('SLURM_JOB_ID') is None and os.uname().nodename.startswith('login'):
        print('Refusing to run on a login node; submit inside a Slurm CPU job.', file=sys.stderr)
        return 2
    groot = args.global_root or args.base / 'global-01'
    croot = args.controls_root or args.base / 'supplement-controls/results/global'
    archive = args.archive or croot
    out = args.out or args.base / 'vocab-01/results/p0-rescore'
    y, splits, blocks, manifest_sha = load_manifest(args.manifest or groot / 'data/manifest.json')
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise ValueError(f'Manifest checksum {manifest_sha} != archived {EXPECTED_MANIFEST_SHA}')
    text_path = args.text or groot / 'data/text.npz'
    text = load_text(text_path)
    paths = feature_paths(groot, croot)
    models = args.models or MODELS
    features, hashes = {}, {}
    for m in models:
        features[m] = np.load(paths[m], mmap_mode='r')
        if features[m].shape[0] != len(y):
            raise ValueError(f'Row mismatch for {m}')
        if not args.no_feature_hash:
            hashes[m] = sha_file(paths[m])
            for fold in range(len(FOLDS)):
                archived = load_archive_fold(archive, m, fold)
                if archived is None:
                    raise ValueError(f'Missing archive for {m} fold {fold}')
                fingerprint = archived[0].get('fingerprint', {})
                if fingerprint.get('features') != hashes[m]:
                    raise ValueError(f'Feature checksum differs from archive fingerprint: {m} fold {fold}')
                if fingerprint.get('text') != sha_file(text_path):
                    raise ValueError(f'Text checksum differs from archive fingerprint: {m} fold {fold}')
                if fingerprint.get('manifest') != manifest_sha:
                    raise ValueError(f'Manifest checksum differs from archive fingerprint: {m} fold {fold}')
    result = run(features, y, splits, blocks, text, archive=archive, n_boot=args.bootstrap, seed=args.seed,
                 dims=tuple(args.dims), parity_tol=args.parity_tol, models=models)
    result['inputs'] = {'global_root': str(groot), 'controls_root': str(croot), 'archive': str(archive),
                        'manifest_sha256': manifest_sha, 'text_sha256': sha_file(text_path),
                        'feature_sha256': hashes, 'code_sha256': sha_file(Path(__file__))}
    write_outputs(result, out)
    print(f'Wrote {out}/p0-rescore.json and .md; parity passed={result["parity"]["passed"]}', flush=True)
    if not result['parity']['passed'] and not args.allow_parity_failure:
        print('PARITY FAILED: ' + json.dumps(result['parity']['models_failed']), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
