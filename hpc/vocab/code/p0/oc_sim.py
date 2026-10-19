#!/usr/bin/env python3
"""Operating characteristics of the MegaEarth A2 verdict procedure (P0, CPU only, NumPy only).

Author: James Edward Ball.

This simulates the analysis at the level the verdict uses: cell x class tables of per-entry
headroom-normalised scores, nu = (RR - H_k'/k') / (1 - H_k'/k'), with one table per route
and per truncation level. It plants a competence-adjusted effect theta in {0, delta, 2*delta}
(delta = 0.025 nu) for all three architectures and approximates a proposed A2 procedure.
Amendment A2 has not yet been reconciled with PLAN/config/production analysis. These
simulations are design diagnostics and cannot pass a production qualification gate:

* product-weight (Exp(1)) crossed bootstrap over test cells and families. Weights are never
  zero, so no class is ever dropped. Draw 0 is the point estimate.
* a parallel-lines (base fixed-effect) slope of ZSL nu on seen nu across control bases x
  training-only PCA truncations, giving the competence-adjusted contrast
  D_adj(a, c) = (z_a - z_c) - beta * (s_a - s_c).
* null-centred one-sided p values p = (1 + #{T* - T >= T - theta0}) / (1 + B). Each
  architecture takes the IUT p (maximum over controls, raw H1 and above-chance), then Holm
  across the 3 architectures.
* an equivalence control e_a chosen on (simulated) validation scores before test; the
  headline is theta_a = D_adj(a, e_a).
* STRONG gates, each recorded separately so that the pre-registered demotion search can
  drop them in order.
* architecture tiers STRONG > YES > DETECTED_BELOW_SESOI > NEGATIVE > INCONCLUSIVE, and study
  tiers STRONG YES > YES > DETECTED, BELOW SESOI > OPERATIONAL NEGATIVE > INCONCLUSIVE.

It reports P(study tier | theta), architecture-level probabilities, gate pass rates, the
coverage of the 90% basic interval, the raw-contrast-only false-YES rate (the O1 problem),
and the gate-demotion result against the A2 requirements:
P(STRONG YES | 2 delta) >= 0.5, P(OPERATIONAL NEGATIVE | 0) >= 0.5 and
P(YES or better | 0) <= 0.05.

Variance components are paired-difference SDs on the nu scale. They come from the archived
global fold scores (`--archive`), or from embedded defaults estimated from them: window
0.125, entry interaction 0.058, class 0.034 (split evenly into family and class), test cell
0.058. Several components are assumed rather than measured; they are listed under
`assumptions` in the output and varied in the full run.

    python oc_sim.py --quick                  # one design, < 60 s
    python oc_sim.py --out <dir>              # full grid (CPU job; about 1-2 h)
"""
import os

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '4')

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

AUTHOR = 'James Edward Ball'
DELTA = 0.025
ALPHA = 0.05
ARCHS = ('clay', 'prithvi', 'prithvi100')
SEEDS = (101, 202, 303)
DESC = ('appearance_rich', 'appearance_rich+metadata_rich', 'appearance_rff+metadata_rich')
CONTROLS = ('random_mean',) + DESC
CONTINENTS = ('europe', 'north_america', 'asia', 'africa', 'south_america', 'oceania')
CONTINENT_P = (0.32, 0.22, 0.20, 0.10, 0.10, 0.06)
# Pre-registered demotion order (first demoted first). 'effect_delta' and the
# matched-support guard are never demoted.
GATE_ORDER = ('continents', 'tercile', 'leave_key_out', 'random_class_folds', 'refs', 'text_robustness',
              'family_macro', 'seeds', 'low_similarity', 'within_key')
STUDY_TIERS = ('STRONG_YES', 'YES', 'DETECTED_BELOW_SESOI', 'OPERATIONAL_NEGATIVE', 'INCONCLUSIVE')
ARCH_TIERS = ('STRONG', 'YES', 'DETECTED_BELOW_SESOI', 'NEGATIVE', 'INCONCLUSIVE')
REQUIREMENTS = {'p_strong_yes_given_2delta_min': 0.5, 'p_operational_negative_given_0_min': 0.5,
                'p_yes_or_better_given_0_max': 0.05}

DEFAULT_COMPONENTS = {'window': 0.125, 'entry': 0.058, 'class': 0.034, 'cell': 0.058, 'source': 'embedded defaults '
                      '(archived global fold scores, qwen_instructed, 9 pretrained-control pairs, GZSL k=8 nu)'}

BASE = {
    'K': 220, 'F': 90, 'keys': 10, 'cells': 400, 'cells_per_class': [5, 12], 'windows_per_entry': 10,
    'largest_family_frac': 0.15, 'var_mult': 1.0,
    # Competence and curve. Seen nu at full dimension; truncation drops in seen nu.
    'beta': 0.30, 'curvature': 0.0, 'curve_centre': 0.30,
    's_pretrained': [0.48, 0.47, 0.45], 's_random': [0.28, 0.30, 0.31], 's_desc': [0.36, 0.38, 0.40],
    'trunc_drop': [0.0, 0.03, 0.06, 0.10, 0.15, 0.21], 'seen_se': 0.003,
    # Control intercept gaps above the strongest control (true D_adj(a, c) = theta + gap_c).
    'gap_random': 0.02, 'gap_desc': [0.01, 0.0, 0.005],
    'semantic_retention_truncated': 1.0, 'low_sim_attenuation': 0.7,
    # Correlation structure of route-specific deviations.
    'lambda_pretrained': 0.3, 'lambda_seeds': 0.5, 'lambda_desc': 0.7, 'rho_levels': 0.85,
    'common_class_sd': 0.15, 'common_cell_sd': 0.05,
    # Approximated co-primaries and robustness re-estimates (ratio of true effect, SE inflation).
    'within_key_phi': 0.6, 'within_key_kappa': 1.6, 'refs_phi': [0.9, 0.8], 'refs_noise': 0.6,
    'rcf_phi': 1.2, 'rcf_noise': 0.6, 'text_phi': [0.8, 0.7], 'text_noise': 0.7,
    'val_noise_sd': 0.010, 'tercile_val_noise': 0.05,
}

SCENARIOS = {
    'main': {},
    'var_x2': {'var_mult': 2.0},
    'few_families': {'K': 150, 'F': 60, 'cells': 300},
    'many_families': {'K': 300, 'F': 130, 'cells': 500},
    'convex_curve': {'curvature': 0.8},
    'semantics_lost_when_truncated': {'semantic_retention_truncated': 0.5},
    'no_competence_gap': {'s_pretrained': [0.40, 0.40, 0.40]},
}


# ----------------------------------------------------------------------------- variance components
def archive_components(archive, ref='qwen_instructed'):
    """Median paired-difference components over the 9 pretrained x control pairs."""
    archive = Path(archive)
    ch8 = float(np.sum(1 / np.arange(1, 9)) / 8)
    out = {'window': [], 'entry': [], 'class': [], 'cell': []}

    def load(model, fold):
        path = archive / model / f'fold-{fold}-scores.npz'
        meta = json.loads((archive / model / f'fold-{fold}.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('scores_sha256'):
            raise ValueError(f'archived score checksum mismatch: {path}')
        with np.load(path) as z:
            rr = z[f'ridge__{ref}']
            if not np.isfinite(rr).all() or np.any((rr < 1 / 8) | (rr > 1)):
                raise ValueError('archived reciprocal ranks are invalid')
            return rr, z['test_rows'], z['test_blocks'].astype(str), z['test_labels']

    for arch in ARCHS:
        for control in ('rich_metadata', 'rich_appearance', f'random_{arch}_101'):
            d, bl, lab = [], [], []
            for fold in range(1, 5):
                a, rows, blocks, labels = load(arch, fold)
                c, rows2, blocks2, labels2 = load(control, fold)
                if not all(np.array_equal(a, b) for a, b in ((rows, rows2), (blocks, blocks2), (labels, labels2))):
                    raise ValueError('archived test support differs between routes')
                d.append((a - c) / (1 - ch8))
                bl.append(blocks)
                lab.append(labels)
            d, bl, lab = np.concatenate(d), np.concatenate(bl), np.concatenate(lab)
            ub, bi = np.unique(bl, return_inverse=True)
            key = bi * 8 + lab
            uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            means = np.bincount(inv, d) / cnt
            within = np.sum((d - means[inv]) ** 2) / (len(d) - len(uk))
            eb, ec = uk // 8, uk % 8
            mu, ca, cb = means.mean(), np.zeros(8), np.zeros(len(ub))
            for _ in range(50):
                ca = np.array([np.mean(means[ec == k] - mu - cb[eb[ec == k]]) for k in range(8)])
                cb = np.array([np.mean(means[eb == b] - mu - ca[ec[eb == b]]) if np.any(eb == b) else 0.
                               for b in range(len(ub))])
            resid = means - mu - ca[ec] - cb[eb]
            out['window'].append(np.sqrt(within))
            out['entry'].append(np.sqrt(max(np.var(resid) - np.mean(within / cnt), 0.)))
            out['class'].append(np.std(ca, ddof=1))
            out['cell'].append(np.std(cb[np.unique(eb)], ddof=1))
    comp = {k: float(np.median(v)) for k, v in out.items()}
    if not all(np.isfinite(value) and value >= 0 for value in comp.values()):
        raise ValueError('archive cannot identify finite non-negative variance components')
    comp['source'] = f'archive {archive} ({ref}; median of 9 pretrained-control pairs; GZSL k=8 nu)'
    return comp


# ----------------------------------------------------------------------------- design
def make_design(rng, p):
    K, F, C = p['K'], p['F'], p['cells']
    cap = max(1, int(np.floor(p['largest_family_frac'] * K)))
    fam = np.r_[np.arange(F), np.zeros(K - F, int)]
    weights = 1 / np.arange(1, F + 1) ** 0.8
    sizes = np.ones(F, int)
    for i in range(F, K):
        w = weights * (sizes < cap)
        f = rng.choice(F, p=w / w.sum())
        sizes[f] += 1
        fam[i] = f
    key_of_family = rng.choice(p['keys'], size=F, p=(w := 1 / np.arange(1, p['keys'] + 1)) / w.sum())
    continent = rng.choice(len(CONTINENTS), size=C, p=CONTINENT_P)
    popularity = rng.lognormal(0, 1, C)
    lo, hi = p['cells_per_class']
    cls, cell = [], []
    for k in range(K):
        n = rng.integers(lo, hi + 1)
        chosen = rng.choice(C, size=n, replace=False, p=popularity / popularity.sum())
        cls += [k] * n
        cell += list(chosen)
    cls, cell = np.array(cls), np.array(cell)
    nw = 1 + rng.poisson(p['windows_per_entry'] - 1, size=len(cls))
    low_sim = rng.random(K) < 0.5
    return {'fam': fam, 'key': key_of_family[fam], 'family_size': np.bincount(fam, minlength=F)[fam],
            'continent': continent, 'cls': cls, 'cell': cell, 'nw': nw, 'low_sim': low_sim, 'K': K, 'F': F, 'C': C}


# ----------------------------------------------------------------------------- one replicate
def holm(p, alpha=ALPHA):
    order = np.argsort(p, kind='stable')
    reject = np.zeros(len(p), bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (len(p) - i):
            reject[j] = True
        else:
            break
    return reject


def p_greater(draws, theta0):
    """Null-centred one-sided p for H0: theta <= theta0 (draws[0] is the point)."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or len(draws) < 2 or not np.isfinite(draws).all() or not np.isfinite(theta0):
        return 1.0  # Missing/invalid evidence cannot satisfy an inference gate.
    t, boot = draws[0], draws[1:]
    return (1 + np.sum(boot - t >= t - theta0)) / (1 + len(boot))


def p_less(draws, theta0):
    """Null-centred one-sided p for H0: theta >= theta0."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or len(draws) < 2 or not np.isfinite(draws).all() or not np.isfinite(theta0):
        return 1.0
    t, boot = draws[0], draws[1:]
    return (1 + np.sum(boot - t <= t - theta0)) / (1 + len(boot))


class Replicate:
    def __init__(self, rng, p, comp, theta, n_boot):
        self.rng, self.p, self.theta, self.B = rng, p, theta, n_boot
        d = self.d = make_design(rng, p)
        E, K, F, C = len(d['cls']), d['K'], d['F'], d['C']
        L = len(p['trunc_drop'])
        vm = np.sqrt(p['var_mult'] / 2.0)
        sd = {k: comp[k] * vm for k in ('window', 'entry', 'class', 'cell')}
        entry_sd = np.sqrt(sd['entry'] ** 2 + sd['window'] ** 2 / d['nw'])

        def dev(n):
            return (rng.normal(0, sd['class'] / np.sqrt(2), (n, K))[:, d['cls']]
                    + rng.normal(0, sd['class'] / np.sqrt(2), (n, F))[:, d['fam'][d['cls']]]
                    + rng.normal(0, sd['cell'], (n, C))[:, d['cell']]
                    + rng.normal(size=(n, E)) * entry_sd)

        # Bases: 3 pretrained, 9 random seeds (arch-major), 3 descriptor controls.
        names = list(ARCHS) + [f'random_{a}_{s}' for a in ARCHS for s in SEEDS] + list(DESC)
        groups = [0, 0, 0] + [1 + i // 3 for i in range(9)] + [4, 4, 4]
        lam = [p['lambda_pretrained']] * 3 + [p['lambda_seeds']] * 9 + [p['lambda_desc']] * 3
        s_full = np.array(p['s_pretrained'] + [p['s_random'][i // 3] for i in range(9)] + p['s_desc'])
        gaps = np.array([0.] * 3 + [p['gap_random'] + rng.normal(0, 0.005) for _ in range(9)] + p['gap_desc'])
        nb = len(names)
        s_true = s_full[:, None] - np.array(p['trunc_drop'])[None, :]
        curve = lambda s: 0.05 + p['beta'] * s + p['curvature'] * (s - p['curve_centre']) ** 2  # noqa: E731
        h = np.where(d['low_sim'], p['low_sim_attenuation'], 2 - p['low_sim_attenuation'])
        retention = np.r_[1.0, np.full(L - 1, p['semantic_retention_truncated'])]
        g_dev, b_dev, l_dev = dev(5), dev(nb), dev(nb * L).reshape(nb, L, E)
        common = rng.normal(0, p['common_class_sd'], K)[d['cls']] + rng.normal(0, p['common_cell_sd'], C)[d['cell']]
        X = np.empty((nb, L, E))
        for b in range(nb):
            own = np.sqrt(p['rho_levels']) * b_dev[b][None] + np.sqrt(1 - p['rho_levels']) * l_dev[b]
            X[b] = (curve(s_true[b])[:, None] - gaps[b] + common[None]
                    + np.sqrt(lam[b]) * g_dev[groups[b]][None] + np.sqrt(1 - lam[b]) * own)
            if b < 3:
                X[b] += theta * retention[:, None] * h[d['cls']][None]
        self.names, self.nb, self.L, self.s_true, self.gaps = names, nb, L, s_true, gaps
        self.app_class_dev = X[nb - 3, 0] - common  # appearance_rich route-specific entry deviations
        # Seen competence estimates: observed point + bootstrap perturbations.
        s_hat = s_true + rng.normal(0, p['seen_se'], s_true.shape)
        self.s_draws = np.concatenate([s_hat[None], s_hat[None] + rng.normal(0, p['seen_se'], (n_boot,) + s_true.shape)])
        # Product-weight bootstrap: Exp(1) weights for cells and families; draw 0 = ones.
        wc = np.vstack([np.ones(C), rng.exponential(1.0, (n_boot, C))])
        wf = np.vstack([np.ones(F), rng.exponential(1.0, (n_boot, F))])
        M = np.zeros((C, K))
        M[d['cell'], d['cls']] = 1.
        XM = np.zeros((C, nb * L, K))
        XM[d['cell'], :, d['cls']] = X.reshape(nb * L, E).T
        den = wc @ M
        num = (wc @ XM.reshape(C, -1)).reshape(n_boot + 1, nb * L, K)
        self.cm = (num / den[:, None, :]).reshape(n_boot + 1, nb, L, K)
        self.wk = wf[:, d['fam']]
        self.XM, self.M = XM.reshape(C, nb, L, K), M

    def agg(self, mask=None, family_macro=False, draws=slice(None)):
        w = self.wk[draws]
        if family_macro:
            w = w / self.d['family_size'][None]
        if mask is not None:
            w = w * mask[None]
        return np.einsum('bnlk,bk->bnl', self.cm[draws], w) / w.sum(1)[:, None, None]


def contrasts(Z, S, beta, a):
    """Raw and adjusted contrasts of pretrained a against the 4 controls (axis -1)."""
    zs, ss = Z[:, 3:12, 0], S[:, 3:12, 0]
    seeds = slice(3 * a, 3 * a + 3)
    zc = np.stack([zs[:, seeds].mean(1)] + [Z[:, 12 + i, 0] for i in range(3)], -1)
    sc = np.stack([ss[:, seeds].mean(1)] + [S[:, 12 + i, 0] for i in range(3)], -1)
    raw = Z[:, a, 0][:, None] - zc
    adj = raw - beta[:, None] * (S[:, a, 0][:, None] - sc)
    return raw, adj


def fe_slope(Z, S, bases):
    z, s = Z[:, bases], S[:, bases]
    zt = z - z.mean(-1, keepdims=True)
    st = s - s.mean(-1, keepdims=True)
    den = (st * st).sum((1, 2))
    return np.divide((st * zt).sum((1, 2)), den, out=np.full(len(den), np.nan), where=den > 0)


def matched_level(pretrained_seen, control_seen):
    """First truncation inside the observed control support; None if none qualifies."""
    control_seen = np.asarray(control_seen, dtype=float)
    if not control_seen.size or not np.isfinite(control_seen).all():
        return None
    lo, hi = control_seen.min(), control_seen.max()
    return next((i for i, value in enumerate(pretrained_seen) if np.isfinite(value) and lo <= value <= hi), None)


def run_replicate(rng, p, comp, theta, n_boot):
    r = Replicate(rng, p, comp, theta, n_boot)
    d = r.d
    Z = r.agg()
    S = r.s_draws
    control_bases = list(range(3, r.nb))
    beta = fe_slope(Z, S, control_bases)
    out = {'beta_hat': float(beta[0])}
    arch = {}
    p_detect, p_delta = np.zeros(3), np.zeros(3)
    for a in range(3):
        raw, adj = contrasts(Z, S, beta, a)
        pa0 = [p_greater(adj[:, j], 0.) for j in range(4)]
        pr0 = [p_greater(raw[:, j], 0.) for j in range(4)]
        pabove = p_greater(Z[:, a, 0], 0.)
        p_detect[a] = max(max(pa0), max(pr0), pabove)
        p_delta[a] = max(p_greater(adj[:, j], DELTA) for j in range(4))
        true_adj = theta + np.r_[np.mean(r.gaps[3 + 3 * a:6 + 3 * a]), r.gaps[12:15]]
        e = int(np.argmin(true_adj + rng.normal(0, p['val_noise_sd'] * np.sqrt(p['var_mult']), 4)))
        th = adj[:, e]
        se = float(np.std(th[1:], ddof=1))
        lo90, hi90 = 2 * th[0] - np.quantile(th[1:], 0.95), 2 * th[0] - np.quantile(th[1:], 0.05)
        target = true_adj[e]
        rec = {'theta_hat': float(th[0]), 'se': se, 'e': CONTROLS[e], 'true': float(target),
               'covered90': bool(lo90 <= target <= hi90),
               'negative': bool(p_less(th, DELTA) < ALPHA), 'worse': bool(p_less(th, -DELTA) < ALPHA),
               'raw_point_min': float(raw[0].min()), 'raw_p': max(pr0)}
        g = {}
        # Exact gates computed from the tables (point estimates unless stated).
        g['seeds'] = bool(all((Z[0, a, 0] - Z[0, 3 + 3 * a + i, 0]
                               - beta[0] * (S[0, a, 0] - S[0, 3 + 3 * a + i, 0])) > 0 for i in range(3)))
        Zf = r.agg(family_macro=True, draws=slice(0, 1))
        g['family_macro'] = bool(contrasts(Zf, S[:1], beta[:1], a)[1][0, e] > 0)
        lko = []
        for key in np.unique(d['key']):
            Zk = r.agg(mask=(d['key'] != key).astype(float), draws=slice(0, 1))
            lko.append(contrasts(Zk, S[:1], beta[:1], a)[1][0, e] > 0)
        g['leave_key_out'] = bool(all(lko))
        app_val = (np.bincount(d['cls'], r.app_class_dev, d['K']) / np.bincount(d['cls'], minlength=d['K'])
                   + rng.normal(0, p['tercile_val_noise'], d['K']))
        hard = (app_val <= np.quantile(app_val, 1 / 3)).astype(float)
        Zt = r.agg(mask=hard, draws=slice(0, 1))
        g['tercile'] = bool(contrasts(Zt, S[:1], beta[:1], a)[1][0, 1] > 0)
        Zl = r.agg(mask=d['low_sim'].astype(float))
        g['low_similarity'] = bool(p_greater(contrasts(Zl, S, beta, a)[1][:, e], 0.) < ALPHA)
        # Continents: point gain > 0 in >= 4 continents with >= 20 test classes.
        wins = 0
        for ci in range(len(CONTINENTS)):
            cells = d['continent'] == ci
            n = r.M[cells].sum(0)
            if np.sum(n > 0) < 20:
                continue
            k = n > 0
            cmc = r.XM[cells].sum(0)[:, :, k] / n[k]
            zc = cmc.mean(-1)[None]
            wins += contrasts(zc, S[:1], beta[:1], a)[1][0, e] > 0
        g['continents'] = bool(wins >= 4)
        # Approximated re-estimates.
        g['within_key'] = bool(p['within_key_phi'] * th[0] + rng.normal(0, np.sqrt(max(p['within_key_kappa'] ** 2
                               - p['within_key_phi'] ** 2, 0)) * se) - 1.645 * p['within_key_kappa'] * se > 0)
        g['refs'] = bool(all(phi * th[0] + rng.normal(0, p['refs_noise'] * se) - 1.645 * se > 0 for phi in p['refs_phi']))
        g['random_class_folds'] = bool(p['rcf_phi'] * th[0] + rng.normal(0, p['rcf_noise'] * se) - 1.645 * se > 0)
        g['text_robustness'] = bool(all(phi * th[0] + rng.normal(0, p['text_noise'] * se) > 0 for phi in p['text_phi']))
        # Matched-support guard (never demoted): pretrained truncation within the control range.
        level = matched_level(S[0, a], S[0, 3:])
        zc_all = np.r_[Z[0, 3 + 3 * a:6 + 3 * a, 0].mean(), Z[0, 12:15, 0]]
        sc_all = np.r_[S[0, 3 + 3 * a:6 + 3 * a, 0].mean(), S[0, 12:15, 0]]
        rec['matched_support'] = bool(level is not None and
            Z[0, a, level] - zc_all[e] - beta[0] * (S[0, a, level] - sc_all[e]) > 0)
        rec['matched_level'] = level
        rec['gates'] = g
        arch[ARCHS[a]] = rec
    out['p_detect'], out['p_delta'] = p_detect.tolist(), p_delta.tolist()
    out['detected'] = holm(p_detect).tolist()
    out['strong_core'] = holm(p_delta).tolist()
    out['arch'] = arch
    return out


# ----------------------------------------------------------------------------- tiers
def arch_tier(rep, a, gates):
    rec = rep['arch'][ARCHS[a]]
    detected, guard = rep['detected'][a], rec['matched_support']
    if detected and rep['strong_core'][a] and guard and all(rec['gates'][g] for g in gates):
        return 'STRONG'
    if detected and guard and rec['theta_hat'] >= DELTA:
        return 'YES'
    if detected and rec['theta_hat'] < DELTA:
        return 'DETECTED_BELOW_SESOI'
    if detected:
        return 'INCONCLUSIVE'  # An unsupported above-SESOI estimate is not a below-SESOI effect.
    if rec['negative']:
        return 'NEGATIVE'
    return 'INCONCLUSIVE'


def study_tier(rep, gates):
    tiers = [arch_tier(rep, a, gates) for a in range(3)]
    if tiers.count('STRONG') >= 2:
        return 'STRONG_YES', tiers
    if any(t in ('STRONG', 'YES') for t in tiers):
        return 'YES', tiers
    if 'DETECTED_BELOW_SESOI' in tiers:
        return 'DETECTED_BELOW_SESOI', tiers
    if all(t == 'NEGATIVE' for t in tiers):
        return 'OPERATIONAL_NEGATIVE', tiers
    return 'INCONCLUSIVE', tiers


def tabulate(reps, gates):
    study = {t: 0 for t in STUDY_TIERS}
    arch = {a: {t: 0 for t in ARCH_TIERS} for a in ARCHS}
    for rep in reps:
        s, tiers = study_tier(rep, gates)
        study[s] += 1
        for a, t in zip(ARCHS, tiers):
            arch[a][t] += 1
    n = len(reps)
    return {'study': {k: v / n for k, v in study.items()}, 'arch': {a: {k: v / n for k, v in d.items()} for a, d in arch.items()}}


def summarise(reps_by_theta, gates):
    table = {name: tabulate(reps, gates) for name, reps in reps_by_theta.items()}
    p_sy = table['2delta']['study']['STRONG_YES']
    p_neg = table['0']['study']['OPERATIONAL_NEGATIVE']
    p_yes0 = table['0']['study']['STRONG_YES'] + table['0']['study']['YES']
    checks = {'p_strong_yes_given_2delta': p_sy, 'p_operational_negative_given_0': p_neg,
              'p_yes_or_better_given_0': p_yes0,
              'pass_strong': p_sy >= REQUIREMENTS['p_strong_yes_given_2delta_min'],
              'pass_negative': p_neg >= REQUIREMENTS['p_operational_negative_given_0_min'],
              'pass_false_positive': p_yes0 <= REQUIREMENTS['p_yes_or_better_given_0_max']}
    checks['pass_all'] = checks['pass_strong'] and checks['pass_negative'] and checks['pass_false_positive']
    return table, checks


def demotion_search(reps_by_theta):
    gates = list(GATE_ORDER)
    demoted = []
    table, checks = summarise(reps_by_theta, gates)
    while not checks['pass_strong'] and gates:
        demoted.append(gates.pop(0))
        table, checks = summarise(reps_by_theta, gates)
    return {'retained': gates, 'demoted': demoted, 'table': table, 'checks': checks}


def diagnostics(reps_by_theta):
    out = {}
    for name, reps in reps_by_theta.items():
        recs = [rep['arch'][a] for rep in reps for a in ARCHS]
        gate_rates = {g: float(np.mean([r['gates'][g] for r in recs])) for g in GATE_ORDER}
        out[name] = {
            'coverage90': float(np.mean([r['covered90'] for r in recs])),
            'mean_theta_hat_minus_true': float(np.mean([r['theta_hat'] - r['true'] for r in recs])),
            'mean_se': float(np.mean([r['se'] for r in recs])),
            'beta_hat_mean': float(np.mean([rep['beta_hat'] for rep in reps])),
            'matched_support_rate': float(np.mean([r['matched_support'] for r in recs])),
            'gate_pass_rate': gate_rates,
            'raw_only_yes_rate': float(np.mean([any(rep['arch'][a]['raw_p'] < ALPHA / 3 and rep['arch'][a]['raw_point_min'] >= DELTA
                                                    for a in ARCHS) for rep in reps])),
            'detected_any_rate': float(np.mean([any(rep['detected']) for rep in reps])),
            'worse_rate': float(np.mean([r['worse'] for r in recs])),
            'equivalence_control_chosen': {c: float(np.mean([r['e'] == c for r in recs])) for c in CONTROLS},
        }
    return out


# ----------------------------------------------------------------------------- driver
def simulate(params, comp, reps, n_boot, seed, log=print):
    if reps < 1 or n_boot < 2:
        raise ValueError('simulation requires at least one replicate and two bootstrap draws')
    rng = np.random.default_rng(seed)
    out = {}
    for name, theta in (('0', 0.0), ('delta', DELTA), ('2delta', 2 * DELTA)):
        out[name] = [run_replicate(rng, params, comp, theta, n_boot) for _ in range(reps)]
        log(f'  theta={name}: {reps} reps done')
    return out


def run(scenarios, comp, reps, n_boot, seed, log=print):
    t0 = time.time()
    results = {}
    for i, name in enumerate(scenarios):
        params = copy.deepcopy(BASE)
        params.update(SCENARIOS[name])
        log(f'[oc] scenario {name}')
        reps_by_theta = simulate(params, comp, reps, n_boot, seed + 1000 * i, log)
        full_table, full_checks = summarise(reps_by_theta, list(GATE_ORDER))
        results[name] = {'params': {k: params[k] for k in ('K', 'F', 'cells', 'var_mult', 'beta', 'curvature',
                                                           'semantic_retention_truncated', 's_pretrained')},
                         'all_gates': {'table': full_table, 'checks': full_checks},
                         'demotion': demotion_search(reps_by_theta),
                         'diagnostics': diagnostics(reps_by_theta)}
    return {'author': AUTHOR, 'qualification_status': 'exploratory_not_design_gate',
            'delta_nu': DELTA, 'alpha': ALPHA, 'requirements': REQUIREMENTS,
            'gate_order': list(GATE_ORDER), 'components': comp, 'reps_per_theta': reps, 'bootstrap': n_boot,
            'seed': seed, 'scenarios': results, 'runtime_s': round(time.time() - t0, 1),
            'assumptions': {k: BASE[k] for k in BASE},
            'note': 'Planted theta applies to all three architectures. Study-level CEILING is not simulated '
                    '(visibility assumed adequate). Within-key, reference, random-class-fold and text-robustness '
                    're-estimates are approximated from the primary estimate (phi, kappa, noise in assumptions).'}


def markdown(res):
    lines = [f'# Operating characteristics of the proposed A2 verdict procedure', '',
             'Exploratory design diagnostic. This output does not qualify the production analysis or authorise acquisition.', '',
             f'delta = {res["delta_nu"]} nu; {res["reps_per_theta"]} replicates per theta; B = {res["bootstrap"]}; '
             f'components: {res["components"]["source"]}.', '',
             '| scenario | gates | P(STRONG YES \\| 2δ) | P(YES+ \\| 2δ) | P(YES+ \\| δ) | P(DET-BELOW \\| δ) | '
             'P(OP-NEG \\| 0) | P(YES+ \\| 0) | P(DET-any \\| 0) | raw-only YES \\| 0 | cover90 (0/δ/2δ) | pass |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, sc in res['scenarios'].items():
        for label, block in (('all', sc['all_gates']), ('after demotion', sc['demotion'])):
            t = block['table']
            st = lambda th, *ks: sum(t[th]['study'][k] for k in ks)  # noqa: E731
            dg = sc['diagnostics']
            lines.append(
                f'| {name} | {label} | {st("2delta", "STRONG_YES"):.2f} | {st("2delta", "STRONG_YES", "YES"):.2f} | '
                f'{st("delta", "STRONG_YES", "YES"):.2f} | {st("delta", "DETECTED_BELOW_SESOI"):.2f} | '
                f'{st("0", "OPERATIONAL_NEGATIVE"):.2f} | {st("0", "STRONG_YES", "YES"):.3f} | '
                f'{dg["0"]["detected_any_rate"]:.3f} | {dg["0"]["raw_only_yes_rate"]:.2f} | '
                f'{dg["0"]["coverage90"]:.2f}/{dg["delta"]["coverage90"]:.2f}/{dg["2delta"]["coverage90"]:.2f} | '
                f'{"yes" if block["checks"]["pass_all"] else "NO"} |')
        lines.append(f'| {name} | demoted | {", ".join(sc["demotion"]["demoted"]) or "none"} | | | | | | | | | |')
    lines += ['', 'Gate pass rates at 2δ (architecture level, main scenario):', '']
    if 'main' in res['scenarios']:
        rates = res['scenarios']['main']['diagnostics']['2delta']['gate_pass_rate']
        lines.append(', '.join(f'{g} {v:.2f}' for g, v in rates.items()))
    lines += ['', res['note'], '']
    return '\n'.join(lines)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--quick', action='store_true', help='main scenario only, 40 reps, B=199 (< 60 s)')
    ap.add_argument('--scenarios', nargs='*', choices=sorted(SCENARIOS))
    ap.add_argument('--reps', type=int, default=400)
    ap.add_argument('--bootstrap', type=int, default=999)
    ap.add_argument('--seed', type=int, default=51426)
    ap.add_argument('--archive', type=Path, help='archived global controls dir (fold-*-scores.npz) for components')
    ap.add_argument('--out', type=Path, help='write oc-sim.json and oc-sim.md here')
    args = ap.parse_args(argv)
    comp = archive_components(args.archive) if args.archive else dict(DEFAULT_COMPONENTS)
    if args.quick:
        scenarios, reps, n_boot = args.scenarios or ['main'], min(args.reps, 40), min(args.bootstrap, 199)
    else:
        scenarios, reps, n_boot = args.scenarios or list(SCENARIOS), args.reps, args.bootstrap
    res = run(scenarios, comp, reps, n_boot, args.seed, log=lambda m: print(m, file=sys.stderr, flush=True))
    res['mode'] = 'quick' if args.quick else 'full'
    md = markdown(res)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'oc-sim.json').write_text(json.dumps(to_jsonable(res), indent=1) + '\n')
        (args.out / 'oc-sim.md').write_text(md)
    print(md)
    main_checks = [sc['demotion']['checks']['pass_all'] for sc in res['scenarios'].values()]
    return 0 if all(main_checks) else 4


if __name__ == '__main__':
    sys.exit(main())
