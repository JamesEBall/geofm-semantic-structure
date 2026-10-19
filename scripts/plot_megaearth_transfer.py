#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Plot corrected retained-cohort MegaEarth transfer estimates. Author: James Edward Ball.

Run with `uv run --script scripts/plot_megaearth_transfer.py` from the repository,
or pass its absolute path from any directory. Reads the completed analysis without fitting models or
recomputing intervals. Panel B uses the archived paired, best-control-per-draw
intervals; it does not subtract the marginal intervals in panel A.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'hpc/vocab/results/retained-20261018/results.json'
MODELS = [('clay', 'Clay'), ('prithvi', 'Prithvi 300M'), ('prithvi100', 'Prithvi 100M')]
CONTROL = 'appearance_rff_metadata_rich'
METRIC = 'family.qwen_instructed.mean.ridge/rr'
GREEN, INK, GREY = '#61776F', '#303534', '#929693'


def select_one(rows, **criteria):
    matches = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one archived estimate for {criteria}; found {len(matches)}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/figures')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    data = json.loads(raw)
    estimates = {
        route: select_one(data['estimates'], route=route, metric=METRIC,
                          scheme='family', pooling='mean')
        for route in [CONTROL, *(f'r1-{name}' for name, _ in MODELS)]
    }
    comparisons = {
        name: select_one(data['comparisons'], architecture=name,
                         control='best-control-per-draw', ref='qwen_instructed')
        for name, _ in MODELS
    }
    random_order = select_one(data['estimates'], route='r1-clay', metric='chance/family',
                              scheme='family', pooling='mean')['value']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'font.weight': 'normal',
        'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#777777',
        'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .45,
        'xtick.major.width': .45, 'ytick.major.width': .45,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    fig = plt.figure(figsize=(6.45, 2.85))
    left = fig.add_axes([.205, .27, .295, .57])
    right = fig.add_axes([.72, .27, .265, .57])
    fig.text(.025, .94, 'A   Filtered retrieval', fontsize=8.5, weight='normal')
    fig.text(.555, .94, 'B   Paired improvement', fontsize=8.5, weight='normal')

    def point(ax, row, y, field, colour=GREEN):
        lo, hi = row['ci']
        ax.plot([lo, hi], [y, y], color=colour, lw=.75)
        ax.scatter(row[field], y, color=colour, s=12, linewidth=.5, zorder=3)

    left.axvline(random_order, color=GREY, ls='--', lw=.55)
    for y, (name, _) in enumerate(MODELS):
        point(left, estimates[f'r1-{name}'], y, 'value')
        point(right, comparisons[name], y, 'delta')
    point(left, estimates[CONTROL], 3, 'value', INK)
    left.set_yticks(range(4), [label for _, label in MODELS] + ['Appearance +\nmetadata'])
    right.set_yticks(range(3), [label for _, label in MODELS])
    for ax in (left, right):
        ax.set_ylim(3.55, -.55)
        ax.tick_params(axis='y', length=0, labelsize=7.5, pad=6)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(axis='x', color='#DADDD9', lw=.3)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    left.set_xlim(.095, .31)
    left.set_xticks([.10, .15, .20, .25, .30])
    left.set_xlabel('Class-balanced filtered MRR', fontsize=7.5, labelpad=7)
    left.text(random_order + .004, 3.42, f'Random order: {random_order:.3f}', color='#68716D', fontsize=6.5)
    right.axvline(0, color=GREY, ls='--', lw=.55)
    right.set_xlim(-.012, .064)
    right.set_xticks([0, .02, .04, .06])
    right.set_xlabel('MRR difference from best control', fontsize=7.5, labelpad=7)
    right.text(.033, 3.0, 'Control reselected per draw', ha='center',
               color='#68716D', fontsize=6.5)
    fig.text(.50, .055, 'Instructed Qwen; 95% conditional intervals',
             ha='center', fontsize=7, color=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / 'megaearth-transfer'
    fig.savefig(stem.with_suffix('.pdf'), metadata={
        'Author': 'James Edward Ball',
        'Title': 'MegaEarth transfer: filtered retrieval and paired improvements',
        'Subject': 'Archived family-split mean-pooling instructed-Qwen estimates',
        'CreationDate': None, 'ModDate': None,
    })
    fig.savefig(stem.with_suffix('.png'), dpi=300, metadata={'Author': 'James Edward Ball'})
    plt.close(fig)
    provenance = {
        'author': 'James Edward Ball',
        'cohort': 'retained-20261018',
        'source': str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT) else str(args.source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'metric': METRIC, 'random_order_reference': random_order,
        'estimates': estimates, 'comparisons': comparisons,
    }
    encoded = json.dumps(provenance, indent=2) + '\n'
    stem.with_suffix('.provenance.json').write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
