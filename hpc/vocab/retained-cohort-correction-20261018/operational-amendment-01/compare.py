#!/usr/bin/env python3
"""Build an outcome comparison without modifying archived inputs.
Author: James Edward Ball.
"""
import argparse
import hashlib
import json
from pathlib import Path

ARCHES = ('clay', 'prithvi', 'prithvi100')
REFS = ('qwen_instructed', 'qwen_plain', 'mpnet')
CONTROLS = ('appearance_rff_metadata_rich', 'appearance_rich_metadata_rich', 'appearance_rich')
LIMITS = [
    'Both corrections are applied together: retained-window sampling and full frozen-taxonomy family exclusion. Their separate effects cannot be identified from this comparison.',
    'Before/after differences are descriptive differences between analyses with different supports. They have no paired confidence interval or significance test.',
    'Reported transfer intervals condition on fitted maps, tuning, vocabulary and seed panel. They do not include refitting uncertainty.',
    'The scoring gallery remains 188 admitted classes; family folds exclude all related frozen-taxonomy classes, including non-admitted siblings. Random-class sensitivity excludes the exact held-out classes.',
    'Five multispectral DOFA routes require refreshed retained-training normalisation. Original features and results remain archived.',
    'RSA remains descriptive unless the separate support-aware calibration and empirical estimator qualify. A partial Mantel p-value alone does not establish semantic or causal information.',
    'All original failed or unavailable gates remain visible. Scheduler completion and a positive conditional gain do not establish a strong research claim.',
    'The earlier eight-class experiment has a different cohort and estimand; its metrics are outside this before/after comparison.',
]


def read(path):
    return json.loads(path.read_text())


def provenance(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def extract(path, expected_pairs):
    data = read(path)
    estimates = {}
    for row in data['estimates']:
        if row['pooling'] != 'mean':
            continue
        scheme = row['scheme']
        if row['metric'] not in [f'{scheme}.{ref}.mean.ridge/rr' for ref in REFS]:
            continue
        complete_route = row['route'] in CONTROLS or row['route'] in tuple('r1-'+v for v in ARCHES) or row['route'].startswith('r1-random_')
        if complete_route and (row['pairs'] != expected_pairs or row['classes'] != 188):
            raise ValueError('Unexpected complete-support MRR pair/class count: ' + str(row))
        key = '|'.join((row['route'], scheme, row['metric']))
        if key in estimates:
            raise ValueError('Duplicate estimate: ' + key)
        estimates[key] = row
    if not estimates:
        raise ValueError('No complete-support MRR estimates')
    comparisons = { '|'.join((v['architecture'], v['ref'], v['control'])): v for v in data['comparisons'] }
    return {'provenance': provenance(path), 'mrr_estimates': estimates,
            'paired_control_comparisons': comparisons, 'h1_statistics': data['h1_statistics'],
            'auxiliary': data['auxiliary'], 'verdict': data['verdict'],
            'power_qualified': data['power_qualified'], 'rsa_interval_qualified': data['rsa_interval_qualified'],
            'limits': data['limits'], 'skipped_descriptive_routes': data['skipped_descriptive_routes']}


def fmt(row, value='value'):
    if row is None:
        return 'pending'
    ci = row.get('ci')
    return f"{row[value]:.4f}" + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--original-verdict', type=Path, required=True)
    p.add_argument('--corrected', type=Path)
    p.add_argument('--corrected-verdict', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.corrected_verdict and not a.corrected:
        p.error('Corrected verdict requires corrected results')
    old = extract(a.original, 11495)
    new = extract(a.corrected, 8755) if a.corrected else None
    ov = read(a.original_verdict)
    nv = read(a.corrected_verdict) if a.corrected_verdict else None
    counts = {'accepted_stored_windows': 44355, 'retained_analysis_windows': 34992,
              'excluded_post_qa_cap': 9363, 'original_test_pairs': 11495, 'corrected_test_pairs': 8755,
              'original_scored_test_windows': 7751, 'corrected_scored_test_windows': 5942,
              'retained_split_windows': {'train':24574, 'val':3539, 'test':6879},
              'gallery_classes':188, 'gallery_families':91,
              'additional_family_exclusions_by_outer_fold':[26,66,216,157,205],
              'corrected_outer_fit_rows':[17585,15498,14359,7902,14680]}
    report = {'author':'James Edward Ball', 'status':'corrected inference available; qualification ' + ('available' if nv else 'pending') if new else 'corrected outputs pending',
              'counts':counts, 'original':old, 'corrected':new,
              'original_qualification':{'provenance':provenance(a.original_verdict),'verdict':ov},
              'corrected_qualification':{'provenance':provenance(a.corrected_verdict),'verdict':nv} if nv else None,
              'scope_limits':LIMITS}
    report['descriptive_changes'] = None if new is None else {
        'mrr_point_changes': {key: new['mrr_estimates'][key]['value'] - row['value']
                              for key, row in old['mrr_estimates'].items() if key in new['mrr_estimates']},
        'paired_control_gain_changes': {key: new['paired_control_comparisons'][key]['delta'] - row['delta']
                                        for key, row in old['paired_control_comparisons'].items()
                                        if key in new['paired_control_comparisons']},
        'interpretation': 'Point-estimate changes only; no inferential test comparing the two analyses.'}
    report['claim_review'] = {
        'status': 'pending independent scientific review',
        'original_false_or_unavailable_gate_fields': {arch: {key:value for key,value in gates.items() if value is False or value is None}
                                               for arch,gates in ov['arch_gates'].items()},
        'corrected_false_or_unavailable_gate_fields': None if nv is None else {
            arch: {key:value for key,value in gates.items() if value is False or value is None}
            for arch,gates in nv['arch_gates'].items()},
        'conditional_primary_holm': None if new is None else new['h1_statistics'],
        'strong_claim_requires_full_registered_gates': True,
        'support_calibration_alone_establishes_semantics': False}

    lines = ['# Retained-cohort comparison', '', 'Author: James Edward Ball', '',
             report['status'] + '. Original outputs remain archived. Overleaf has not been edited.', '',
             'The primary result is class-balanced, filtered reciprocal rank under family transfer, mean pooling and the instructed Qwen text reference. Brackets show 95% conditional intervals.', '',
             '| Model | Original MRR | Corrected MRR | Original gain over best control per draw | Corrected gain | Original Holm p | Corrected Holm p |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arch in ARCHES:
        key=f'r1-{arch}|family|family.qwen_instructed.mean.ridge/rr'
        ckey=f'{arch}|qwen_instructed|best-control-per-draw'
        lines.append(f"| {arch} | {fmt(old['mrr_estimates'][key])} | {fmt(new['mrr_estimates'][key] if new else None)} | {fmt(old['paired_control_comparisons'][ckey], 'delta')} | {fmt(new['paired_control_comparisons'][ckey] if new else None, 'delta')} | {old['h1_statistics'][arch]['holm_p']:.6g} | {format(new['h1_statistics'][arch]['holm_p'], '.6g') if new else 'pending'} |")
    lines += ['', '| Appearance control | Original MRR | Corrected MRR |', '|---|---:|---:|']
    for route in CONTROLS:
        key=f'{route}|family|family.qwen_instructed.mean.ridge/rr'
        lines.append(f"| {route} | {fmt(old['mrr_estimates'][key])} | {fmt(new['mrr_estimates'][key] if new else None)} |")
    lines += ['', 'Support changes from 11,495 to 8,755 test window-class pairs and from 7,751 to 5,942 scored test windows. The gallery remains 188 classes in 91 families. Of 44,355 stored accepted windows, 34,992 are retained; 9,363 were excluded by the post-QA cap.', '',
              'The JSON includes all three reference sensitivities, random-class results, every random-weight/control contrast, scaling, nulls and descriptive RSA. The original qualification label is ' + ov['label'] + '; the corrected qualification is ' + (nv['label'] if nv else 'pending') + '.', '',
              'Claim assessment is pending corrected inference and qualification. Conditional Holm results will be reported separately from the full gate assessment.']
    lines += ['', 'Scope limitations:', ''] + ['- '+v for v in LIMITS]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
