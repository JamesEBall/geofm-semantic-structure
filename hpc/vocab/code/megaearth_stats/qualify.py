"""Qualification of the production inference procedure and the pre-registered verdict gates.

Author: James Edward Ball.

Amendment A3 leaves strong-yes and operational-negative tiers unavailable "until the same production
estimator and decision rules pass a separate calibration check". This module is that check, plus the
gate evaluation the tier table (io.DEFAULT_TIERS, PLAN §2.1) needs.

Calibration (TEST-PLAN §1.5, CAL-1/2/3) is a plasmode study on the real test pair support:

* a replicate resamples the real cells and the real families with replacement (the two-way design the
  production bootstrap assumes), relabelling duplicates as new cells/classes/families;
* the pretrained route is the real pretrained route re-centred so that its class-balanced point on the
  full data sits at a planted level: equal to the best control (null, least favourable for the IUT),
  best control + SESOI, the observed value, or chance + 2 x best control's above-chance signal (CAL-2);
* the production code (`bootstrap.make_draws`, `PairBoot`, `p_null_centred`, `holm`) then runs on the
  replicate with `power_bootstrap` draws and the production H1 rule decides.

Rejection rates under the null give the size of each contrast, of the IUT per architecture and of the
Holm family; rates under planted effects give power; interval coverage of the known truth gives CAL-3.
Nothing here reads a test window or a feature; it reads the frozen per-pair scores only.

Mantel calibration (CAL-5) runs `rsa.statistics` on simulated class prototypes with the real text
prototypes and the real nuisance geometries.

Gates with no registered estimator are reported as unavailable (None), never as passed.
"""
from __future__ import annotations

import numpy as np

from .bootstrap import PairBoot, holm, make_draws, p_null_centred, percentile_ci, quantile
from . import rsa as R

SCENARIOS = ('null', 'planted_sesoi', 'planted_observed', 'planted_cal2', 'exchangeable')


# --------------------------------------------------------------------------- support and truth
class Support:
    """Real pair support (class, cell, family per pair) plus per-route per-pair values."""

    def __init__(self, cls, cell, family, values: dict[str, np.ndarray], chance: np.ndarray):
        self.cls = np.asarray(cls, dtype=np.int64)
        self.cell = np.asarray(cell, dtype=np.int64)
        self.family = np.asarray(family, dtype=np.int64)
        self.values = {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}
        self.chance = np.asarray(chance, dtype=np.float64)
        self.n = len(self.cls)
        for k, v in self.values.items():
            if len(v) != self.n or not np.isfinite(v).all():
                raise ValueError(f'route {k}: incomplete per-pair values')
        self.point_boot = PairBoot(self.cls, self.cell, self.family, None)

    def point(self, x):
        return float(self.point_boot.point(x))


def plant(support: Support, pretrained: str, controls: list[str], scenario: str, sesoi: float, ratio: float = 2.0):
    """Return (values_dict, truth) for one architecture: the pretrained route re-centred to the scenario.

    truth[c] is the class-balanced difference pretrained - c on the full data (what a replicate estimates).
    """
    pre = support.values[pretrained]
    pts = {c: support.point(support.values[c]) for c in controls}
    best = max(pts.values())
    chance = support.point(support.chance)
    p0 = support.point(pre)
    if scenario == 'null':
        target = best
    elif scenario == 'planted_sesoi':
        target = best + sesoi
    elif scenario == 'planted_observed':
        target = p0
    elif scenario == 'planted_cal2':
        target = chance + ratio * max(best - chance, 0.0)
    else:
        raise ValueError(scenario)
    shifted = pre + (target - p0)
    truth = {c: float(target - pts[c]) for c in controls}
    return shifted, truth, {'target': float(target), 'best_control': float(best), 'chance': float(chance),
                            'observed': float(p0)}


def resample_world(rng: np.random.Generator, support: Support, arrays: dict[str, np.ndarray]):
    """Two-way plasmode draw: cells and families with replacement; duplicates become new ids.

    Returns new (cls, cell, family) and each array expanded in the same order.
    """
    cells, cell_inv = np.unique(support.cell, return_inverse=True)
    fams, fam_inv = np.unique(support.family, return_inverse=True)
    v = rng.multinomial(len(cells), np.full(len(cells), 1.0 / len(cells)))
    u = rng.multinomial(len(fams), np.full(len(fams), 1.0 / len(fams)))
    mult = v[cell_inv] * u[fam_inv]
    idx = np.repeat(np.arange(support.n), mult)
    # copy indices within each duplicated pair: k over cell copies, j over family copies
    vk = np.repeat(v[cell_inv], mult)
    pos = np.arange(len(idx)) - np.repeat(np.cumsum(mult) - mult, mult)
    k = pos % np.maximum(vk, 1)
    j = pos // np.maximum(vk, 1)
    mv = int(v.max()) if len(v) else 1
    mu = int(u.max()) if len(u) else 1
    cell = support.cell[idx] * mv + k
    cls = support.cls[idx] * mu + j
    family = support.family[idx] * mu + j
    out = {name: a[idx] for name, a in arrays.items()}
    return cls, cell, family, out


def h1_decision(boot: PairBoot, pre: np.ndarray, controls: dict[str, np.ndarray], alpha: float,
                require_lower_bounds: bool, chance: np.ndarray | None = None, ratio: float = 1.5,
                sesoi: float | None = None, n_arch_simultaneous: int = 1) -> dict:
    """Production H1 statistics for one architecture on one pair support.

    Returns per-control delta/CI/p, the IUT p (max over controls), the best-control-per-draw contrast
    and, when `chance` is given, the strong-yes criterion-1 contrast A_pre - ratio*max(A_c*, 0).
    """
    dist_pre = boot.draws_of(pre)
    point_pre = float(boot.point(pre))
    per = {}
    cdraws = []
    cpoints = []
    for name, vals in controls.items():
        delta = pre - vals
        dd = boot.draws_of(delta)
        dp = float(boot.point(delta))
        per[name] = {'delta': dp, 'ci': list(percentile_ci(dd)), 'p': p_null_centred(dp, dd)}
        cdraws.append(boot.draws_of(vals))
        cpoints.append(float(boot.point(vals)))
    iut_p = max(c['p'] for c in per.values())
    cstack = np.stack(cdraws)
    cstar = dist_pre - cstack.max(axis=0)
    out = {'controls': per, 'iut_p': float(iut_p), 'lower_bounds_positive': all(c['ci'][0] > 0 for c in per.values()),
           'cstar': {'delta': point_pre - max(cpoints), 'ci': list(percentile_ci(cstar))}}
    if sesoi is not None:
        a2 = 0.05 / max(1, int(n_arch_simultaneous))
        lo, hi = quantile(cstar, a2 / 2), quantile(cstar, 1 - a2 / 2)
        out['cstar']['simultaneous_ci'] = [lo, hi]
        out['cstar']['lb_gt_sesoi_simultaneous'] = bool(lo > sesoi)
        out['cstar']['ub_lt_sesoi_simultaneous'] = bool(hi < sesoi)
        out['cstar']['ub_lt_minus_sesoi_simultaneous'] = bool(hi < -sesoi)
    if chance is not None:
        ch = boot.draws_of(chance)
        a_pre = dist_pre - ch
        a_best = cstack.max(axis=0) - ch
        crit1 = a_pre - ratio * np.maximum(a_best, 0.0)
        out['crit1'] = {'point': float(point_pre - boot.point(chance) - ratio * max(max(cpoints) - boot.point(chance), 0.0)),
                        'ci': list(percentile_ci(crit1)), 'pass': bool(quantile(crit1, 0.025) > 0)}
    return out


def h1_pass(holm_p: float, decision: dict, alpha: float, require_lower_bounds: bool) -> bool:
    return bool(holm_p <= alpha and (decision['lower_bounds_positive'] or not require_lower_bounds))


# --------------------------------------------------------------------------- plasmode replicates
def replicate(support: Support, archs: dict[str, dict], scenario: str, rep: int, p: dict) -> dict:
    """One plasmode replicate across all registered architectures; the truth per contrast is recorded."""
    seed = int(p['seed'])
    rng = np.random.default_rng([seed, 40301, SCENARIOS.index(scenario), int(rep)])
    sesoi = float(p['sesoi'])
    ratio = float(p['strong_yes_ratio'])
    planted = {}
    arrays = {'chance': support.chance}
    truths = {}
    for arch, ac in archs.items():
        controls = list(ac['random']) + list(p['controls'])
        if scenario == 'exchangeable':
            # Two exchangeable routes: the first two random seeds of the architecture, true difference as
            # measured on the full data. Coverage of that truth is the CAL-3 check.
            if len(ac['random']) < 2:
                continue
            a, b = ac['random'][0], ac['random'][1]
            arrays[f'{arch}:pre'] = support.values[a]
            arrays[f'{arch}:{b}'] = support.values[b]
            truths[arch] = {b: support.point(support.values[a]) - support.point(support.values[b])}
            planted[arch] = {'exchangeable': [a, b]}
            continue
        shifted, truth, info = plant(support, ac['pretrained'], controls, scenario, sesoi)
        arrays[f'{arch}:pre'] = shifted
        for c in controls:
            arrays[f'{arch}:{c}'] = support.values[c]
        truths[arch] = truth
        planted[arch] = info
    cls, cell, family, ex = resample_world(rng, support, arrays)
    draws = make_draws(cell, family, int(p['power_bootstrap']), int(rng.integers(0, 2**31 - 1)))
    boot = PairBoot(cls, cell, family, draws)
    boot.cache_weights()
    decisions = {}
    for arch, ac in archs.items():
        names = [k.split(':', 1)[1] for k in ex if k.startswith(arch + ':') and not k.endswith(':pre')]
        if not names:
            continue
        controls = {n: ex[f'{arch}:{n}'] for n in names}
        decisions[arch] = h1_decision(boot, ex[f'{arch}:pre'], controls, float(p['holm_alpha']),
                                      bool(p['h1_require_lower_bounds']), chance=ex['chance'], ratio=ratio,
                                      sesoi=sesoi, n_arch_simultaneous=len(archs))
    adjusted = holm([decisions[a]['iut_p'] for a in decisions])
    out = {'scenario': scenario, 'rep': int(rep), 'planted': planted, 'n_pairs': int(len(cls)), 'archs': {}}
    for (arch, dec), hp in zip(decisions.items(), adjusted):
        rec = {'iut_p': dec['iut_p'], 'holm_p': float(hp),
               'h1_pass': h1_pass(float(hp), dec, float(p['holm_alpha']), bool(p['h1_require_lower_bounds'])),
               'iut_reject': bool(dec['iut_p'] <= float(p['holm_alpha'])),
               'crit1_pass': bool(dec.get('crit1', {}).get('pass', False)),
               'cstar_lb_gt_sesoi_simultaneous': dec['cstar'].get('lb_gt_sesoi_simultaneous'),
               'cstar_ub_lt_sesoi_simultaneous': dec['cstar'].get('ub_lt_sesoi_simultaneous'),
               'contrasts': {}}
        for c, d in dec['controls'].items():
            t = truths[arch].get(c)
            rec['contrasts'][c] = {'delta': d['delta'], 'ci': d['ci'], 'p': d['p'], 'truth': t,
                                   'reject': bool(d['p'] <= float(p['holm_alpha'])),
                                   'covers_truth': None if t is None else bool(d['ci'][0] <= t <= d['ci'][1])}
        out['archs'][arch] = rec
    return out


def summarise_replicates(records: list[dict], p: dict) -> dict:
    """Operating characteristics per scenario from replicate records."""
    alpha = float(p['holm_alpha'])
    out = {}
    by = {}
    for r in records:
        by.setdefault(r['scenario'], []).append(r)
    for scenario, reps in sorted(by.items()):
        n = len(reps)
        archs = sorted(reps[0]['archs'])
        s = {'replicates': n, 'archs': {}, 'fwer': float(np.mean([any(r['archs'][a]['h1_pass'] for a in archs) for r in reps]))}
        for a in archs:
            rows = [r['archs'][a] for r in reps]
            rec = {'h1_pass_rate': float(np.mean([x['h1_pass'] for x in rows])),
                   'iut_reject_rate': float(np.mean([x['iut_reject'] for x in rows])),
                   'crit1_pass_rate': float(np.mean([x['crit1_pass'] for x in rows])),
                   'cstar_lb_gt_sesoi_rate': float(np.mean([bool(x['cstar_lb_gt_sesoi_simultaneous']) for x in rows])),
                   'contrasts': {}}
            for c in rows[0]['contrasts']:
                cs = [x['contrasts'][c] for x in rows]
                truths = [x['truth'] for x in cs if x['truth'] is not None]
                cov = [x['covers_truth'] for x in cs if x['covers_truth'] is not None]
                rec['contrasts'][c] = {'truth': float(np.mean(truths)) if truths else None,
                                       'reject_rate': float(np.mean([x['reject'] for x in cs])),
                                       'coverage': float(np.mean(cov)) if cov else None,
                                       'mean_delta': float(np.mean([x['delta'] for x in cs])),
                                       'sd_delta': float(np.std([x['delta'] for x in cs], ddof=1)) if len(cs) > 1 else None,
                                       'mean_ci_halfwidth': float(np.mean([(x['ci'][1] - x['ci'][0]) / 2 for x in cs]))}
            s['archs'][a] = rec
        out[scenario] = s
    return out


def calibration_verdict(summary: dict, p: dict) -> dict:
    """CAL-1 (size), CAL-2 (power), CAL-3 (coverage) against TEST-PLAN §1.5 and config power_min."""
    size_band = (0.03, 0.07)
    size_max = 0.07
    cov_band = (0.92, 0.995)
    checks = {}
    null = summary.get('null')
    if null:
        exact = {}
        for a, rec in null['archs'].items():
            # the contrast whose truth is exactly zero is the planted best control
            zero = [c for c, v in rec['contrasts'].items() if v['truth'] is not None and abs(v['truth']) < 1e-9]
            exact[a] = {c: rec['contrasts'][c]['reject_rate'] for c in zero}
        checks['cal1_contrast_size_in_band'] = all(size_band[0] <= r <= size_band[1] for v in exact.values() for r in v.values()) and bool(exact)
        checks['cal1_contrast_size'] = exact
        checks['cal1_iut_size_max'] = max(rec['iut_reject_rate'] for rec in null['archs'].values())
        checks['cal1_iut_size_ok'] = checks['cal1_iut_size_max'] <= size_max
        checks['cal1_h1_rate'] = {a: rec['h1_pass_rate'] for a, rec in null['archs'].items()}
        checks['cal1_fwer'] = null['fwer']
        checks['cal1_fwer_ok'] = null['fwer'] <= size_max
        covs = [v['coverage'] for rec in null['archs'].values() for v in rec['contrasts'].values() if v['coverage'] is not None]
        checks['cal3_null_coverage_min'] = min(covs) if covs else None
    ex = summary.get('exchangeable')
    if ex:
        covs = [v['coverage'] for rec in ex['archs'].values() for v in rec['contrasts'].values() if v['coverage'] is not None]
        checks['cal3_exchangeable_coverage'] = {a: {c: v['coverage'] for c, v in rec['contrasts'].items()} for a, rec in ex['archs'].items()}
        checks['cal3_coverage_ok'] = all(cov_band[0] <= c <= cov_band[1] for c in covs) and bool(covs)
    cal2 = summary.get('planted_cal2')
    if cal2:
        checks['cal2_h1_power'] = {a: rec['h1_pass_rate'] for a, rec in cal2['archs'].items()}
        checks['cal2_power_ok'] = all(r >= float(p['power_min']) for r in checks['cal2_h1_power'].values())
        checks['cal2_crit1_power'] = {a: rec['crit1_pass_rate'] for a, rec in cal2['archs'].items()}
        checks['cal2_crit1_power_ok'] = all(r >= 0.70 for r in checks['cal2_crit1_power'].values())
    for name in ('planted_sesoi', 'planted_observed'):
        sc = summary.get(name)
        if sc:
            checks[f'{name}_h1_power'] = {a: rec['h1_pass_rate'] for a, rec in sc['archs'].items()}
    required = ['cal1_iut_size_ok', 'cal1_fwer_ok', 'cal1_contrast_size_in_band', 'cal3_coverage_ok', 'cal2_power_ok']
    missing = [k for k in required if k not in checks]
    power_ok = None if missing else all(bool(checks[k]) for k in required)
    return {'checks': checks, 'required': required, 'missing': missing, 'power_ok': power_ok,
            'rule': 'power_ok = CAL-1 size (per-contrast size in [0.03,0.07], IUT and Holm FWER <= 0.07) and '
                    'CAL-3 coverage in [0.92,0.995] and CAL-2 power >= power_min'}


# --------------------------------------------------------------------------- Mantel calibration
def mantel_null_sim(P: np.ndarray, appearance: np.ndarray, geo: np.ndarray, n_classes: int, perms: int,
                    seed: int, sim: int, d: int = 64) -> dict:
    """One CAL-5 simulation on `n_classes` classes sampled from the real set.

    independent: image prototypes pure noise (unrestricted Mantel should reject at rate alpha);
    control_plus_noise: image prototypes = appearance geometry + noise (partial RSA should be centred at 0).
    """
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng([int(seed), 50501, int(sim)])
    K = len(P)
    pick = np.sort(rng.choice(K, size=min(int(n_classes), K), replace=False))
    Pk = P[pick]
    app = appearance[pick]
    nuis = [pdist(app, 'cosine'), pdist(geo[pick], 'cosine')]
    out = {'sim': int(sim), 'n_classes': int(len(pick))}
    X = rng.normal(size=(len(pick), d))
    r = R.statistics(X, Pk, nuis, perms, int(rng.integers(0, 2**31 - 1)), keep_null=False)
    out['independent'] = {'mantel_p': r['mantel_p'], 'spearman': r['spearman'],
                          'partial_mantel_p': r['partial_mantel_p'], 'partial_spearman': r['partial_spearman']}
    # control + noise: appearance prototypes (unit norm) with isotropic noise of comparable scale
    A = app / np.maximum(np.linalg.norm(app, axis=1, keepdims=True), 1e-12)
    X2 = A + rng.normal(scale=0.5 / np.sqrt(A.shape[1]), size=A.shape)
    r2 = R.statistics(X2, Pk, nuis, perms, int(rng.integers(0, 2**31 - 1)), keep_null=False)
    out['control_plus_noise'] = {'mantel_p': r2['mantel_p'], 'spearman': r2['spearman'],
                                 'partial_mantel_p': r2['partial_mantel_p'], 'partial_spearman': r2['partial_spearman']}
    return out


def summarise_mantel(records: list[dict], p: dict) -> dict:
    alpha = 0.05
    ind = [r['independent'] for r in records]
    cpn = [r['control_plus_noise'] for r in records]
    t1 = float(np.mean([x['mantel_p'] <= alpha for x in ind])) if ind else None
    t1p = float(np.mean([x['partial_mantel_p'] <= alpha for x in cpn if x['partial_mantel_p'] is not None])) if cpn else None
    mean_partial = float(np.mean([x['partial_spearman'] for x in cpn if x['partial_spearman'] is not None])) if cpn else None
    mx = float(p['rsa_qual_max_type1'])
    out = {'simulations': len(records), 'permutations': int(p['rsa_qual_perms']),
           'independent_mantel_type1': t1, 'control_plus_noise_partial_type1': t1p,
           'control_plus_noise_mean_partial_spearman': mean_partial,
           'mantel_type1_ok': None if t1 is None else bool(t1 <= mx),
           'partial_type1_ok': None if t1p is None else bool(t1p <= mx),
           'partial_centred_ok': None if mean_partial is None else bool(abs(mean_partial) < 0.02),
           'interval_qualified': False,
           'interval_note': 'the production RSA stage reports no population interval (interval_qualified=False), '
                            'so the registered coverage band rsa_qual_coverage has nothing to check'}
    oks = [out['mantel_type1_ok'], out['partial_type1_ok'], out['partial_centred_ok']]
    out['mantel_p_qualified'] = None if any(o is None for o in oks) else all(oks)
    return out


# --------------------------------------------------------------------------- gates on the real data
def unavailable(reason: str) -> dict:
    return {'value': None, 'reason': reason}


def gate_table(real: dict, calib: dict, mantel: dict, p: dict) -> tuple[dict, dict, dict]:
    """Per-architecture and study gates for verdict.evaluate, with reasons for unavailable gates.

    `real` holds, per architecture, the production decisions on the frozen data (see stages.qualify)."""
    arch_gates = {}
    reasons = {}
    for arch, r in real.items():
        g = {}
        why = {}
        g['h1_pass'] = r['h1_pass']
        g['beats_each_seed'] = r['beats_each_seed']
        g['crit1'] = r['crit1_pass']
        g['crit2'] = r['crit2_pass']
        g['crit3'] = r.get('crit3_pass')
        if g['crit3'] is None:
            why['crit3'] = r.get('crit3_reason', 'scaling curves unavailable for this architecture')
        g['crit4_available'] = False
        g['crit4'] = None
        why['crit4'] = ('no registered interval estimator for the pretrained-minus-random partial RSA difference; '
                        f"pretrained partial Mantel p = {r.get('partial_mantel_p')} (strong threshold {p['mantel_alpha_strong']})")
        g['crit5'] = r.get('crit5_pass')
        if g['crit5'] is None:
            why['crit5'] = r.get('crit5_reason', 'strata unavailable')
        g['crit6'] = None
        why['crit6'] = 'hardest tercile must be defined on val-split ranks (TEST-PLAN S11-T07); readouts store no val-split scores'
        g['lb_cstar_gt_sesoi_simultaneous'] = r['cstar']['lb_gt_sesoi_simultaneous']
        g['ub_cstar_lt_sesoi_simultaneous'] = r['cstar']['ub_lt_sesoi_simultaneous']
        g['ub_cstar_lt_minus_sesoi_simultaneous'] = r['cstar']['ub_lt_minus_sesoi_simultaneous']
        g['matched_support_pass'] = None
        why['matched_support_pass'] = 'A2 competence adjustment unreconciled (A3); no production matched-support estimator'
        g['family_macro_gain_pos'] = r['family_macro_gain_pos']
        g['probe_below_floor'] = r.get('probe_below_floor')
        if g['probe_below_floor'] is None:
            why['probe_below_floor'] = 'supervised probe top-1 unavailable'
        arch_gates[arch] = g
        reasons[arch] = why
    study = {'power_ok': calib.get('power_ok'), 'mantel_p_qualified': mantel.get('mantel_p_qualified'),
             'rsa_interval_qualified': False}
    return arch_gates, study, reasons
