"""analysis_qualify: calibration of the production inference procedure and the pre-registered verdict.

Author: James Edward Ball.

Runs after analysis_inference on the frozen per-pair scores only (no feature, no test window):
  * oc.<scenario>.<chunk> units: plasmode replicates (megaearth_stats.qualify.replicate);
  * mantel.<chunk> units: CAL-5 Mantel null simulations with the production RSA statistic;
  * finalize: operating characteristics, the real-data gates, verdict.evaluate on the registered tier
    table, written to analysis_qualify/{qualification.json, verdict.json, report.md}.
Gates with no registered estimator are unavailable, never passed (A3).
"""
from __future__ import annotations

import json
import numpy as np
from mc.core import atomic_write_json as _write_json, read_json
from mc.registry import Stage, StageFailed
from .analysis import checked_study, settings, primary_routes


def atomic_write_json(path, obj):
    from megaearth_stats.io import jsonable
    _write_json(path, jsonable(obj))

OC_CHUNK = 25
MANTEL_CHUNK = 20


# --------------------------------------------------------------------------- frozen score loading
def load_values(ctx, routes, pooling, scheme, names):
    """Merge fold files of analysis_readouts for the given routes into full-support arrays.

    Returns (pairs, values) with values[(route, name)] finite on every pair, or raises."""
    from megaearth_stats.io import read_npz
    base = ctx.p('analysis_readouts')
    pairs = None
    values = {}
    for route in routes:
        files = sorted(base.glob(f'{route}.{pooling}.{scheme}.*.npz'))
        if not files:
            raise StageFailed(f'No readout files for {route}/{pooling}/{scheme}')
        for f in files:
            arrays, meta = read_npz(f)
            if pairs is None:
                pairs = {k: arrays[k] for k in ('pairs/win', 'pairs/cls')}
            for k in pairs:
                if not np.array_equal(pairs[k], arrays[k]):
                    raise StageFailed('Readout pair supports differ across routes or folds')
            for name in names:
                if name not in arrays:
                    continue
                v = arrays[name]
                dst = values.setdefault((route, name), np.full(len(v), np.nan))
                ok = np.isfinite(v)
                if np.any(np.isfinite(dst) & ok & (dst != v)):
                    raise StageFailed('Overlapping fold scores disagree')
                dst[ok] = v[ok]
    return pairs, values


def support_for(ctx, st, ref, scheme='family'):
    """Support object for the registered architectures: pretrained, seeds and controls on one reference."""
    from megaearth_stats.qualify import Support
    p = settings(ctx)
    pool = p['primary_pooling']
    routes = sorted({a['pretrained'] for a in p['architectures'].values()}
                    | {s for a in p['architectures'].values() for s in a['random']} | set(p['controls']))
    key = f'{scheme}.{ref}.{pool}.ridge/rr'          # readout.key_name(scheme, ref, pooling, 'ridge')
    pairs, vals = load_values(ctx, routes, pool, scheme, [key, f'chance/{scheme}'])
    win, cls = pairs['pairs/win'], pairs['pairs/cls']
    values = {}
    for route in routes:
        v = vals.get((route, key))
        if v is None or not np.isfinite(v).all():
            raise StageFailed(f'Incomplete primary scores for {route} ({ref}, {scheme})')
        values[route] = v
    chance = vals[(routes[0], f'chance/{scheme}')]
    return Support(cls, st.cell_idx[win], st.family[cls], values, chance), pairs


# --------------------------------------------------------------------------- plan / run
def plan(ctx):
    st = checked_study(ctx)
    p = settings(ctx)
    from megaearth_stats.qualify import SCENARIOS
    support_for(ctx, st, p['primary_ref'])          # fail early if scores are missing
    out = []
    n = int(p['power_sims'])
    for scenario in SCENARIOS:
        for c, start in enumerate(range(0, n, OC_CHUNK)):
            out.append({'id': f'oc.{scenario}.{c:03d}', 'kind': 'oc', 'scenario': scenario,
                        'reps': list(range(start, min(start + OC_CHUNK, n)))})
    m = int(p['rsa_qual_sims'])
    for c, start in enumerate(range(0, m, MANTEL_CHUNK)):
        out.append({'id': f'mantel.{c:03d}', 'kind': 'mantel', 'sims': list(range(start, min(start + MANTEL_CHUNK, m)))})
    return out


def _mantel_inputs(ctx, st, p):
    """Real text prototypes, cell-balanced appearance-rich class prototypes and geo histograms for the
    classes represented in the test split (the same class set as analysis_rsa)."""
    from megaearth_stats.study import FeatureStore
    rows = st.rows_for_split("test")
    fp = st.footprint(next(iter(primary_routes(ctx))))
    lab = st.labels_for(fp)
    app = FeatureStore(st, ['appearance_rich'], p['primary_pooling'])
    keep, means = [], []
    for c in range(st.K):
        cmask = np.zeros(st.K, dtype=bool)
        cmask[c] = True
        rr = rows[lab.any_in(cmask)[rows]]
        if len(rr):
            means.append(np.mean([app.rows(rr[st.cell_idx[rr] == g]).mean(0) for g in np.unique(st.cell_idx[rr])], axis=0))
            keep.append(c)
    keep = np.asarray(keep)
    return st.protos[p['primary_ref']][keep], np.asarray(means), st.class_histograms(fp, 'geo')[keep], keep


def run(ctx, unit, ckpt):
    from megaearth_stats import qualify as Q
    st = checked_study(ctx)
    p = settings(ctx)
    path = ctx.stage_dir() / f"{unit['id']}.json"
    if unit['kind'] == 'oc':
        support, _ = support_for(ctx, st, p['primary_ref'])
        archs = {a: ac for a, ac in p['architectures'].items() if ac['random']}
        state = ckpt.load('oc', {'records': []})
        records = list(state['records'])
        done = {r['rep'] for r in records}
        for rep in unit['reps']:
            if rep in done:
                continue
            records.append(Q.replicate(support, archs, unit['scenario'], rep, p))
            if ckpt.due():
                ckpt.save('oc', {'records': records})
            ckpt.maybe_stop()
        atomic_write_json(path, {'unit': unit, 'records': records, 'n_pairs': support.n,
                                 'assumptions': ['plasmode: cells and families resampled with replacement from the real test support',
                                                 'pretrained route = real pretrained scores re-centred to the planted level',
                                                 'controls keep their real scores; nothing is simulated from a parametric model']})
        return {'outputs': [path], 'metrics': {'replicates': len(records)}}
    if unit['kind'] == 'mantel':
        P, app, geo, keep = _mantel_inputs(ctx, st, p)
        state = ckpt.load('mantel', {'records': []})
        records = list(state['records'])
        done = {r['sim'] for r in records}
        for sim in unit['sims']:
            if sim in done:
                continue
            records.append(Q.mantel_null_sim(P, app, geo, int(p['rsa_qual_classes']), int(p['rsa_qual_perms']),
                                             int(p['seed']), sim))
            if ckpt.due():
                ckpt.save('mantel', {'records': records})
            ckpt.maybe_stop()
        atomic_write_json(path, {'unit': unit, 'records': records, 'classes_available': int(len(keep))})
        return {'outputs': [path], 'metrics': {'simulations': len(records)}}
    raise StageFailed('Unknown qualification unit kind')


# --------------------------------------------------------------------------- finalize: real gates
def _real_gates(ctx, st, p, results):
    """Production decisions on the frozen scores for every registered architecture."""
    from megaearth_stats.bootstrap import PairBoot, make_draws, holm, percentile_ci, quantile
    from megaearth_stats.qualify import h1_decision, h1_pass
    from megaearth_stats.io import read_npz
    pool = p['primary_pooling']
    alpha = float(p['holm_alpha'])
    req = bool(p['h1_require_lower_bounds'])
    sesoi = float(p['sesoi'])
    archs = p['architectures']
    out = {a: {} for a in archs}
    # primary reference, family scheme: the H1 of record (same draws as analysis_inference)
    support, pairs = support_for(ctx, st, p['primary_ref'])
    draws = make_draws(support.cell, support.family, int(p['bootstrap']), int(p['seed']))
    boot = PairBoot(support.cls, support.cell, support.family, draws)
    boot.cache_weights()
    decisions = {}
    for a, ac in archs.items():
        controls = {c: support.values[c] for c in list(ac['random']) + list(p['controls'])}
        decisions[a] = h1_decision(boot, support.values[ac['pretrained']], controls, alpha, req, chance=support.chance,
                                   ratio=float(p['strong_yes_ratio']), sesoi=sesoi, n_arch_simultaneous=len(archs))
    adjusted = holm([decisions[a]['iut_p'] for a in archs])
    inference_h1 = (results or {}).get('h1_statistics', {})
    for (a, dec), hp in zip(decisions.items(), adjusted):
        r = out[a]
        r['iut_p'] = dec['iut_p']
        r['holm_p'] = float(hp)
        r['h1_pass'] = h1_pass(float(hp), dec, alpha, req)
        r['controls'] = dec['controls']
        r['cstar'] = dec['cstar']
        r['crit1'] = dec['crit1']
        r['crit1_pass'] = dec['crit1']['pass']
        r['beats_each_seed'] = all(dec['controls'][s]['ci'][0] > 0 for s in archs[a]['random'])
        ref_h1 = inference_h1.get(a, {})
        r['matches_inference'] = (abs(ref_h1.get('iut_p', np.nan) - dec['iut_p']) < 1e-3 and
                                  abs(ref_h1.get('holm_p', np.nan) - float(hp)) < 1e-3) if ref_h1 else None
        # family-macro gain over the point-best control
        best = max(dec['controls'], key=lambda c: boot.point(support.values[c]))
        r['family_macro_gain'] = float(boot.family_macro(support.values[archs[a]['pretrained']] - support.values[best]))
        r['family_macro_gain_pos'] = bool(r['family_macro_gain'] > 0)
        r['point_best_control'] = best
    # criterion 2: H1 rule under every reference, and under the random-class folds with the primary reference
    crit2 = {a: {} for a in archs}
    for ref in p['refs']:
        for scheme in (('family', 'random_class') if ref == p['primary_ref'] else ('family',)):
            try:
                sup, _ = support_for(ctx, st, ref, scheme)
            except StageFailed as e:
                for a in archs:
                    crit2[a][f'{ref}/{scheme}'] = {'available': False, 'reason': str(e)}
                continue
            if scheme == 'family' and ref == p['primary_ref']:
                b = boot
            else:
                b = PairBoot(sup.cls, sup.cell, sup.family, make_draws(sup.cell, sup.family, int(p['bootstrap']), int(p['seed'])))
                b.cache_weights()
            decs = {}
            for a, ac in archs.items():
                controls = {c: sup.values[c] for c in list(ac['random']) + list(p['controls'])}
                decs[a] = h1_decision(b, sup.values[ac['pretrained']], controls, alpha, req)
            adj = holm([decs[a]['iut_p'] for a in archs])
            for (a, d), hp in zip(decs.items(), adj):
                crit2[a][f'{ref}/{scheme}'] = {'available': True, 'iut_p': d['iut_p'], 'holm_p': float(hp),
                                               'pass': h1_pass(float(hp), d, alpha, req)}
    for a in archs:
        cells = crit2[a]
        out[a]['crit2'] = cells
        out[a]['crit2_pass'] = None if any(not c['available'] for c in cells.values()) else all(c['pass'] for c in cells.values())
    # criterion 3: scaling separation gap(all) - gap(6) on the primary reference
    sc_dir = ctx.p('analysis_scaling')
    for a, ac in archs.items():
        routes = [ac['pretrained']] + list(ac['random']) + list(p['controls'])
        curves = {}
        reason = None
        for route in routes:
            files = sorted(sc_dir.glob(f"{route}.{p['primary_ref']}.*.npz"))
            if not files:
                reason = f'no scaling curve for {route}'
                break
            acc = {}
            for f in files:
                arr, meta = read_npz(f)
                ms = [str(m) for m in meta['m']]
                for want in ('6', 'all'):
                    if want not in ms:
                        reason = f'm={want} absent from scaling curve of {route}'
                        break
                    dst = acc.setdefault(want, np.full(support.n, np.nan))
                    dst[arr['pair_idx']] = arr['rr'][ms.index(want)]
                if reason:
                    break
            if reason:
                break
            if any(not np.isfinite(v).all() for v in acc.values()):
                reason = f'scaling curve of {route} has unsupported subsets'
                break
            curves[route] = acc
        if reason:
            out[a]['crit3_pass'] = None
            out[a]['crit3_reason'] = reason
            continue
        pre6, prea = curves[ac['pretrained']]['6'], curves[ac['pretrained']]['all']
        c6 = np.stack([boot.draws_of(curves[c]['6']) for c in routes[1:]]).max(axis=0)
        ca = np.stack([boot.draws_of(curves[c]['all']) for c in routes[1:]]).max(axis=0)
        diff = (boot.draws_of(prea) - ca) - (boot.draws_of(pre6) - c6)
        p6 = boot.point(pre6) - max(boot.point(curves[c]['6']) for c in routes[1:])
        pa = boot.point(prea) - max(boot.point(curves[c]['all']) for c in routes[1:])
        out[a]['crit3'] = {'gap_m6': float(p6), 'gap_all': float(pa), 'difference': float(pa - p6), 'ci': list(percentile_ci(diff))}
        out[a]['crit3_pass'] = bool(quantile(diff, 0.025) > 0)
    # criterion 5: leave-one-key-out and continents, from the inference strata (point gains over c*)
    strata = (results or {}).get('strata', [])
    key = f"family.{p['primary_ref']}.{pool}.ridge/rr"
    excl = set(p.get('continent_exclude', [])) | {'unassigned'}
    for a, ac in archs.items():
        rows = {}
        for s in strata:
            if s.get('metric') == key and s.get('pooling', pool) == pool and s.get('available'):
                rows.setdefault(s['stratum'], {})[s['route']] = s
        if not rows:
            out[a]['crit5_pass'] = None
            out[a]['crit5_reason'] = 'inference strata missing'
            continue

        def gain(stratum):
            r = rows.get(stratum, {})
            pre = r.get(ac['pretrained'])
            seeds = [r[s]['value'] for s in ac['random'] if s in r]
            ctrls = [r[c]['value'] for c in p['controls'] if c in r]
            if pre is None or not seeds or len(ctrls) != len(p['controls']):
                return None
            return float(pre['value'] - max([float(np.mean(seeds))] + ctrls))
        keys = sorted(s for s in rows if s.startswith('leave-key-out='))
        lko = {s: gain(s) for s in keys}
        conts = {s: (gain(s), rows[s][ac['pretrained']]['classes']) for s in rows
                 if s.startswith('continent=') and s.split('=', 1)[1] not in excl and ac['pretrained'] in rows[s]}
        eligible = {s: g for s, (g, k) in conts.items() if k >= int(p['continent_min_classes'])}
        out[a]['crit5'] = {'leave_key_out_gains': lko, 'continent_gains': {s: g for s, (g, _) in conts.items()},
                           'eligible_continents': sorted(eligible), 'positive_eligible': sum(1 for g in eligible.values() if g is not None and g > 0)}
        if not keys or any(g is None for g in lko.values()) or any(g is None for g in eligible.values()):
            out[a]['crit5_pass'] = None
            out[a]['crit5_reason'] = 'a stratum lacks the pretrained route or a registered control'
        else:
            out[a]['crit5_pass'] = bool(all(g > 0 for g in lko.values()) and
                                        out[a]['crit5']['positive_eligible'] >= int(p['continents_required']))
    # supervised probe and partial Mantel p (descriptive inputs to the tier table)
    est = (results or {}).get('estimates', [])
    for a, ac in archs.items():
        pr = [e for e in est if e['route'] == ac['pretrained'] and e['metric'] == f'probe.{pool}/top1' and e.get('scheme') == 'family']
        out[a]['probe_top1'] = pr[0]['value'] if pr else None
        out[a]['probe_below_floor'] = None if not pr else bool(pr[0]['value'] < float(p['probe_top1_floor']))
        f = ctx.p('analysis_rsa', f"{ac['pretrained']}.{p['primary_ref']}.npz")
        if f.exists():
            _, meta = read_npz(f)
            out[a]['partial_mantel_p'] = meta.get('partial_mantel_p')
            out[a]['mantel_p'] = meta.get('mantel_p')
            out[a]['partial_spearman'] = meta.get('partial_spearman')
    return out


def finalize(ctx):
    from megaearth_stats import qualify as Q
    from megaearth_stats.verdict import evaluate, unavailable_gates
    from megaearth_stats import ESTIMAND, CONDITIONAL_LABEL, ENGINE_VERSION
    st = checked_study(ctx)
    p = settings(ctx)
    d = ctx.stage_dir()
    oc, mantel = [], []
    for f in sorted(d.glob('oc.*.json')):
        oc += read_json(f)['records']
    for f in sorted(d.glob('mantel.*.json')):
        mantel += read_json(f)['records']
    if not oc:
        raise StageFailed('No plasmode replicates found')
    summary = Q.summarise_replicates(oc, p)
    calib = Q.calibration_verdict(summary, p)
    mant = Q.summarise_mantel(mantel, p) if mantel else {'mantel_p_qualified': None, 'interval_qualified': False,
                                                         'note': 'no Mantel simulations found'}
    results = read_json(ctx.p('analysis_inference', 'results.json'))
    real = _real_gates(ctx, st, p, results)
    arch_gates, study_gates, reasons = Q.gate_table(real, calib, mant, p)
    verdict = evaluate(p['tiers'], arch_gates, study_gates)
    strong_gates = next(t for t in p['tiers'] if t['name'] == 'STRONG YES')['rule']['count_archs']['gates']
    qual = {'engine_version': ENGINE_VERSION, 'estimand': ESTIMAND, 'conditional': CONDITIONAL_LABEL,
            'plasmode': {'replicates': len(oc), 'scenarios': summary}, 'calibration': calib, 'mantel': mant,
            'parameters': {k: p[k] for k in ('power_sims', 'power_bootstrap', 'power_min', 'sesoi', 'strong_yes_ratio',
                                               'holm_alpha', 'h1_require_lower_bounds', 'rsa_qual_sims', 'rsa_qual_perms',
                                               'rsa_qual_classes', 'rsa_qual_max_type1', 'probe_top1_floor',
                                               'mantel_alpha_strong', 'continent_min_classes', 'continents_required')}}
    atomic_write_json(d / 'qualification.json', qual)
    ver = {'engine_version': ENGINE_VERSION, 'tier': verdict['tier'], 'label': verdict['label'], 'trace': verdict['trace'],
           'arch_gates': arch_gates, 'study_gates': study_gates, 'unavailable': unavailable_gates(arch_gates, strong_gates),
           'unavailable_reasons': reasons, 'real': real, 'power_qualified': calib['power_ok'],
           'rsa_interval_qualified': False, 'mantel_p_qualified': mant.get('mantel_p_qualified'),
           'test_data_used_for_tuning': False, 'estimand': ESTIMAND, 'conditional': CONDITIONAL_LABEL,
           'reading': 'STRONG YES needs every listed gate True in at least two architectures; an unavailable gate '
                      'never satisfies it. YES (weak) needs one architecture with h1_pass.'}
    atomic_write_json(d / 'verdict.json', ver)
    lines = [f"# Qualified verdict: {verdict['label']}", '', f"Estimand. {ESTIMAND}", '',
             f"power_ok = {calib['power_ok']} ({calib['rule']})", f"Mantel p qualified = {mant.get('mantel_p_qualified')}; RSA interval qualified = False", '',
             '| architecture | ' + ' | '.join(strong_gates) + ' |', '|---|' + '---|' * len(strong_gates)]
    for a, g in arch_gates.items():
        lines.append(f'| {a} | ' + ' | '.join({True: 'yes', False: 'no', None: 'n/a'}[g.get(k)] for k in strong_gates) + ' |')
    lines += ['', 'Unavailable gates and why:'] + [f'- {a}: ' + '; '.join(f'{k}: {v}' for k, v in r.items()) for a, r in reasons.items()]
    (d / 'report.md').write_text('\n'.join(lines) + '\n')
    return {'tier': verdict['label'], 'power_ok': calib['power_ok'], 'replicates': len(oc), 'mantel_sims': len(mantel)}


STAGES = [Stage('analysis_qualify', 'P2', 'cpu', ('analysis_inference',), plan, run, finalize,
                unit_cpus=8, unit_mem_gb=32, est_unit_min=10,
                description='operating-characteristic and Mantel calibration; pre-registered verdict tiers')]
