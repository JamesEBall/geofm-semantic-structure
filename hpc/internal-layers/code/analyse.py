"""Frozen clean readout under internal perturbation. Author: James Edward Ball."""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np


def main(a):
    spec=importlib.util.spec_from_file_location('fresh_analysis', a.fresh/'code/analyse.py')
    f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    with np.load(a.fresh/'data/text.npz') as z: text={r:z[r] for r in f.REFS}
    reports={}
    for arch in ('clay','prithvi','prithvi100'):
        root=a.output/arch
        complete=json.loads((root/'complete.json').read_text())
        if complete['runner_sha256'] != f.sha(Path(__file__).with_name('run.py')) or complete['fresh_extractor_sha256'] != f.sha(a.fresh/'code/extract.py'):
            raise ValueError('Source changed after extraction')
        for name, digest in complete['files'].items():
            if f.sha(root/name)!=digest: raise ValueError('Feature checksum mismatch')
        rows=json.loads((root/'selection.json').read_text())['rows']
        labels=np.array([f.CLASSES.index(r['label']) for r in rows]); blocks=np.array([str(r['block']) for r in rows])
        splits=np.array([r['split'] for r in rows]); fit=splits=='train'; val=splits=='val'; test=np.load(root/'test_indices.npy')
        layers={}
        for layer in complete['layers']:
            x=np.load(root/f'layer-{layer}.npy')
            layers[str(layer)]={'geometry':f.geometry(x[splits=='test'],labels[splits=='test'],blocks[splits=='test'],text)}
        x=np.load(root/'clean.npy'); mean=x[fit].mean(0,dtype=np.float64); scale=x[fit].std(0,dtype=np.float64); scale[scale<1e-8]=1
        train=(x[fit]-mean)/scale; y=np.eye(10)[labels[fit]]; ym=y.mean(0)
        eig,u=np.linalg.eigh(train.T@train); rhs=u.T@(train.T@(y-ym))
        clean={}; selections={}; conditions=[]
        for ref in f.REFS:
            best=(-np.inf,None,None)
            for alpha in f.ALPHAS:
                w=u@(rhs/(np.maximum(eig,0)[:,None]+alpha))
                scores=f.language_scores(((x[val]-mean)/scale)@w+ym,text[ref])
                score=f.block_average(f.reciprocal_rank(labels[val],scores),blocks[val])
                if score>best[0]:best=(score,alpha,w)
            _,alpha,w=best; selections[ref]={'alpha':alpha,'validation_query_ap':best[0]}
            base=f.language_scores(((x[test]-mean)/scale)@w+ym,text[ref])
            clean[ref]=f.metrics(labels[test],base,list(range(10)),blocks=blocks[test])
            for cond in complete['conditions']:
                changed=np.load(root/(cond['name']+'.npy'))
                scores=f.language_scores(((changed-mean)/scale)@w+ym,text[ref])
                target=labels[test]==f.CLASSES.index(cond['label'])
                rr=f.reciprocal_rank(labels[test],scores)-f.reciprocal_rank(labels[test],base)
                conditions.append({**cond,'reference':ref, 'metrics':f.metrics(labels[test],scores,list(range(10)),blocks=blocks[test]),
                    'target_delta_query_ap':f.block_average(rr[target],blocks[test][target]),
                    'other_delta_query_ap':f.block_average(rr[~target],blocks[test][~target])})
        reports[arch]={'layers':layers,'clean':clean,'readout_selection':selections,'interventions':conditions,
            'interpretation':'Exploratory directional perturbation; selectivity and random-control comparison required. No unique circuit identification.'}
    (a.output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,(arch,report) in zip(axes,reports.items()):
        nums=sorted(int(k) for k in report['layers'])
        for ref in f.REFS:
            ax.plot(np.array(nums)/max(nums),[report['layers'][str(i)]['geometry'][ref]['spearman_rho'] for i in nums],marker='o',label=ref)
        ax.set_title(arch);ax.set_xlabel('Fraction of encoder depth');ax.axhline(0,color='.7',linewidth=.7)
    axes[0].set_ylabel('Category distance correspondence');axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.output/'layer-geometry.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,(arch,report) in zip(axes,reports.items()):
        for ix,ref in enumerate(f.REFS):
            selected=[c for c in report['interventions'] if c['reference']==ref and c['dose']==1.]
            target=np.mean([c['target_delta_query_ap'] for c in selected if c['control']=='target'])
            norm=np.mean([c['target_delta_query_ap'] for c in selected if c['control']=='norm-only'])
            random=np.mean([c['target_delta_query_ap'] for c in selected if c['control'].startswith('random')])
            ax.scatter([ix-.15,ix,ix+.15],[target,random,norm],c=['#355b73','#9d7656','#777777'])
        ax.axhline(0,color='.7',linewidth=.7);ax.set_title(arch)
        ax.set_xticks(range(3),['Qwen instructed','Qwen plain','MPNet'],rotation=25,ha='right',fontsize=8)
    axes[0].set_ylabel('Target-category query AP change')
    fig.suptitle('Full dose: category direction (blue), random (brown), norm-only (grey)',fontsize=10)
    fig.tight_layout();fig.savefig(a.output/'interventions.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fresh',type=Path,required=True);p.add_argument('--output',type=Path,required=True); main(p.parse_args())
