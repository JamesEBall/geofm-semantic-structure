#!/usr/bin/env python3
"""Paired fixed-fit control summary. Author: James Edward Ball."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from controls import MODELS,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
    info={m:json.loads((out/m/'complete.json').read_text()) for m in MODELS}
    folds=info[MODELS[0]]['folds']; cache={};allblocks=[]
    for m in MODELS:
        if info[m]['folds']!=folds:raise ValueError('fold mismatch')
        for f in range(folds):
            saved=json.loads((out/m/f'fold-{f}.json').read_text())
            if saved['fingerprint']!=info[m]['fingerprint'] or hashlib.sha256((out/m/f'fold-{f}-scores.npz').read_bytes()).hexdigest()!=saved['scores_sha256']:raise ValueError('Fold checksum/fingerprint mismatch')
            with np.load(out/m/f'fold-{f}-scores.npz') as z:cache[m,f]={k:z[k] for k in z.files}
            allblocks.extend(cache[m,f]['test_blocks'])
    blocks=np.unique(allblocks);draws=np.random.default_rng(51426).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=1000)
    estimates={};samples={}
    for m in MODELS:
        for track,fs in [('known',[0]),('heldout',range(1,folds))]:
            routes=[k for k in cache[m,0] if '__' in k]
            for route in routes:
                points=[];reps=[]
                for f in fs:
                    data=cache[m,f];present=np.unique(data['test_blocks']);weights=draws[:,np.searchsorted(blocks,present)]
                    means=np.array([data[route][data['test_blocks']==b].mean() for b in present]);den=weights.sum(1);good=den>0
                    for c in np.unique(data['test_labels']):good &= draws[:,np.searchsorted(blocks,np.unique(data['test_blocks'][data['test_labels']==c]))].sum(1)>0
                    rep=np.full(1000,np.nan);rep[good]=weights[good]@means/den[good];reps.append(rep);points.append(means.mean())
                key=(m,track,route);estimates[key]=float(np.mean(points));samples[key]=np.mean(reps,axis=0)
    rows=[]
    for (m,track,route),point in estimates.items():
        ref=route.split('__')[1];base=('rich_metadata',track,'ridge__'+ref);delta=samples[m,track,route]-samples[base];good=np.isfinite(delta)
        rows.append(dict(model=m,track=track,route=route,query_ap=point,difference_from_rich_metadata=point-estimates[base],conditional_interval=np.quantile(delta[good],[.025,.975]).tolist() if good.any() else None,valid_resamples=int(good.sum())))
    save(out/'summary.json',{'author':'James Edward Ball','rows':rows,'interpretation':'Supplementary conditional fixed-fit block intervals; no familywise significance claim.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs=sorted({r['route'].split('__')[1] for r in rows});fig,axes=plt.subplots(1,3,figsize=(14,7))
    for ax,ref in zip(axes,refs):
        subset=[r for r in rows if r['track']=='heldout' and r['route']=='ridge__'+ref]
        for i,r in enumerate(subset):
            if r['conditional_interval']:ax.plot(r['conditional_interval'],[i,i],color='#879aa8')
            ax.scatter(r['difference_from_rich_metadata'],i,color='#304e63',s=15)
        ax.set_yticks(range(len(subset)),[r['model'] for r in subset]);ax.axvline(0,color='.6',lw=.6);ax.set_title(ref)
    fig.supxlabel('Held-out query AP difference from rich appearance plus metadata');fig.tight_layout();fig.savefig(out/'stronger-controls.png',dpi=180)
if __name__=='__main__':main()
