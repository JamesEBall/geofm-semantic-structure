#!/usr/bin/env python3
"""Render the completed global experiment. Author: James Edward Ball.

Uses the archived primary and supplementary estimates without refitting.
Run from any directory; outputs are written to output/figures/ unless --out is given.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
MODELS = [('clay', 'Clay'), ('prithvi', 'Prithvi 300M'), ('prithvi100', 'Prithvi 100M')]
REFS = [('qwen_instructed', 'Instructed Qwen'), ('qwen_plain', 'Plain Qwen'), ('mpnet_plain', 'MPNet')]
SEEDS = (101, 202, 303)
GREEN, GREY, REFERENCE, INK = '#61776F', '#929693', '#687680', '#252525'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT.parent / 'output/figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'reference-results/global'
    primary = json.loads((source / 'primary.json').read_text())
    control = json.loads((source / 'controls/summary.json').read_text())
    rows = {(r['model'], r['track'], r['route']): r for r in control['rows']}
    assert len(rows) == 768
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'font.weight': 'normal', 'axes.titleweight': 'normal',
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#777777',
                         'axes.linewidth': .45, 'xtick.major.width': .45,
                         'ytick.major.width': .45, 'pdf.fonttype': 42})

    def save(fig, name):
        fig.savefig(args.out / f'{name}.pdf', metadata={'Author': 'James Edward Ball', 'Title': name,
                                                     'CreationDate': None, 'ModDate': None})
        fig.savefig(args.out / f'{name}.png', dpi=240)
        plt.close(fig)

    legend = [Line2D([], [], marker='o', color=GREEN, ls='none', markersize=3.3, label='Pretrained'),
              Line2D([], [], marker='D', color=GREY, mfc='white', mew=.5, ls='none', markersize=2.8, label='Random weights (3 seeds)'),
              Line2D([], [], color=REFERENCE, ls='--', lw=.55, label='Appearance + metadata')]
    fig, axes = plt.subplots(2, 3, figsize=(6.45, 4.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.985, top=.79, bottom=.13, hspace=.63, wspace=.18)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.52, 1.0), ncol=3, frameon=False, fontsize=7)
    fig.text(.03, .875, 'A   Familiar categories, new geography', weight='normal', fontsize=8.5)
    fig.text(.03, .455, 'B   Excluded categories, new geography', weight='normal', fontsize=8.5)
    for ri, track in enumerate(('known', 'heldout')):
        for ax, (ref, label) in zip(axes[ri], REFS):
            ax.axvline(0, color=REFERENCE, ls='--', lw=.55)
            for y, (model, _) in enumerate(MODELS):
                entries = [(model, y-.14, True)] + [(f'random_{model}_{s}', y+o, False) for s, o in zip(SEEDS, (.03, .16, .29))]
                for name, pos, trained in entries:
                    r = rows[name, track, 'ridge__'+ref]
                    lo, hi = r['conditional_interval']
                    ax.plot([lo, hi], [pos, pos], color=GREEN if trained else GREY,
                            lw=.75 if trained else .4)
                    ax.scatter(r['difference_from_rich_metadata'], pos, marker='o' if trained else 'D',
                               s=12 if trained else 7, facecolor=GREEN if trained else 'white',
                               edgecolor=GREEN if trained else GREY, linewidth=.5, zorder=3)
            ax.set_yticks(range(3), [label for _, label in MODELS])
            ax.set_ylim(2.45, -.48)
            ax.set_xlim(-.08, .09)
            ax.set_xticks([-.05, 0, .05])
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelbottom=True, labelsize=7)
            ax.grid(axis='x', color='#DADDD9', lw=.3)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(label, fontsize=8, pad=9)
    fig.supxlabel('AP difference from appearance + metadata', y=.035, fontsize=8)
    save(fig, 'global-transfer-controls')

    fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.985, top=.72, bottom=.24, wspace=.18)
    geo_legend = legend[:2]+[Line2D([], [], color=REFERENCE, ls='--', lw=.55, label='Simple appearance')]
    fig.legend(handles=geo_legend, loc='upper center', bbox_to_anchor=(.52, 1.01), ncol=3, frameon=False, fontsize=7)
    for ax, (ref, label) in zip(axes, REFS):
        ax.axvline(primary['models']['appearance']['geometry'][ref]['spearman_rho'], color=REFERENCE, ls='--', lw=.55)
        for y, (model, _) in enumerate(MODELS):
            ax.scatter(primary['models'][model]['geometry'][ref]['spearman_rho'], y-.12, color=GREEN, s=12, linewidth=.5, zorder=3)
            for seed, offset in zip(SEEDS, (.03, .16, .29)):
                value = primary['models'][f'random_{model}_{seed}']['geometry'][ref]['spearman_rho']
                ax.scatter(value, y+offset, marker='D', s=7, facecolor='white', edgecolor=GREY, linewidth=.5, zorder=3)
        ax.set_yticks(range(3), [label for _, label in MODELS])
        ax.set_ylim(2.45, -.48)
        ax.set_xlim(0, .5)
        ax.set_xticks([0, .2, .4])
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#DADDD9', lw=.3)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=8, pad=8)
    fig.supxlabel('Category-distance Spearman correlation', y=.035, fontsize=8)
    save(fig, 'global-category-structure')

    ref = 'qwen_instructed'
    fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.22, right=.985, top=.86, bottom=.25, wspace=.17)
    tracks = [('ridge', 'True pairing'), ('block_shuffle', 'Within-block shuffle'),
              ('category_permutation', 'Permuted categories'), ('nearest', 'Nearest example')]
    for ax, (model, label) in zip(axes, MODELS):
        base = rows['rich_metadata', 'heldout', 'ridge__'+ref]['query_ap']
        ax.axvline(base, color=REFERENCE, ls='--', lw=.55)
        for y, (route, _) in enumerate(tracks):
            route_names = [route] if route in ('ridge', 'nearest') else [f'{route}_{s}' for s in SEEDS]
            offsets = [0] if len(route_names) == 1 else [-.12, 0, .12]
            for rn, offset in zip(route_names, offsets):
                value = rows[model, 'heldout', rn+'__'+ref]['query_ap']
                ax.scatter(value, y+offset, marker='o' if route == 'ridge' else 'D',
                           s=12 if route == 'ridge' else 8,
                           facecolor=GREEN if route == 'ridge' else 'white',
                           edgecolor=GREEN if route == 'ridge' else GREY, linewidth=.5, zorder=3)
        ax.set_yticks(range(4), [label for _, label in tracks])
        ax.set_ylim(3.5, -.5)
        ax.set_xlim(.225, .285)
        ax.set_xticks([.23, .25, .27])
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#DADDD9', lw=.3)
        ax.set_title(label, fontsize=8.5, pad=8)
    fig.supxlabel('Excluded-category AP - instructed Qwen', y=.07, fontsize=8)
    save(fig, 'global-readout-controls')

    with (args.out / 'global-control-values.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(control['rows'][0]))
        writer.writeheader()
        writer.writerows(control['rows'])
    print(f'Wrote three figures and all 768 control estimates to {args.out}')


if __name__ == '__main__':
    main()
