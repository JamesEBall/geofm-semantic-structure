"""Build a portable review dataset; publication is a separate explicit operation.

Author: James Edward Ball. No account tokens, signed URLs or acquisition logs are packaged.
"""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
from mc.core import atomic_path,atomic_write_json,sha256_file
from mc.registry import Stage,StageFailed
from ._extract_freeze import verify_freeze


def plan(ctx):
    from ._imagery_io import iter_shards
    verify_freeze(ctx,'f3')
    return [{'id':f'data-{i:05d}','shard':str(p.relative_to(ctx.run))}
            for i,p in enumerate(iter_shards(ctx))]


def add_bytes(tar,name,data):
    info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
    info.uname=info.gname='';info.uid=info.gid=0
    tar.addfile(info,io.BytesIO(data))


def run(ctx,unit,ckpt):
    import numpy as np
    import zstandard
    from ._imagery_io import open_shard
    source=open_shard(ctx,ctx.run/unit['shard'])
    path=ctx.stage_dir()/(unit['id']+'.tar.zst')
    # Public fields form an allowlist. Acquisition artefacts and URLs stay outside this export.
    keep=('window_id','item_id','source','datetime','epsg','x0','y0','dn_offset','dn_scale','baseline','bad_frac','snow_frac','mgrs_tile')
    with atomic_path(path) as tmp:
        with open(tmp,'wb') as out,zstandard.ZstdCompressor(level=3,threads=max(0,ctx.threads-1)).stream_writer(out) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for i,row in source['meta'].iterrows():
                    ckpt.maybe_stop();wid=str(row.window_id)
                    for band in ('b10','b20','scl'):
                        buffer=io.BytesIO();np.save(buffer,source[band][i],allow_pickle=False)
                        add_bytes(tar,f'{wid}/{band}.npy',buffer.getvalue())
                    meta={k:row[k].item() if hasattr(row[k],'item') else row[k] for k in keep if k in row}
                    add_bytes(tar,f'{wid}/metadata.json',(json.dumps(meta,sort_keys=True,default=str)+'\n').encode())
    return {'outputs':[path],'metrics':{'windows':len(source['meta']),'bytes':path.stat().st_size}}


def finalize(ctx):
    import shutil
    paths=[]
    tables=('vocab/classes.parquet','vocab/taxonomy.yaml','select/windows.parquet','assemble/windows_qa.parquet',
            'labels_exact/labels.parquet','admit/classes_final.parquet','admit/windows_final.parquet',
            'families/families.parquet','families/folds.json','text/phrases.parquet','text/attribution.json')
    for rel in tables:
        src=ctx.run/rel
        if not src.is_file():raise StageFailed('Missing release table: '+rel)
        dest=ctx.stage_dir()/'tables'/rel
        with atomic_path(dest) as tmp:shutil.copyfile(src,tmp)
        paths.append(dest)
    shards=sorted(ctx.stage_dir().glob('data-*.tar.zst'));paths+=shards
    manifest={'author':'James Edward Ball','source_snapshot':'planet-251229.osm.pbf','files':{str(p.relative_to(ctx.stage_dir())):sha256_file(p) for p in paths},
              'publication_status':'prepared locally; not uploaded','licences':{'OSM':'ODbL-1.0','OSM wiki':'CC-BY-SA-2.0','Sentinel-2':'Copernicus Sentinel data terms','RESOLVE ecoregion fields':'CC-BY-4.0'},
              'frozen_manifest_sha256':sha256_file(ctx.p('freeze_f3','manifest.json'))}
    card='''---
license: other
license_name: mixed-source-terms
license_link: https://huggingface.co/datasets/Mead0w1ark/megaearth-osm-sentinel2/blob/main/README.md
---
# MegaEarth OSM × Sentinel-2

Each archive contains a directory per stable window ID, with native uint16 Sentinel-2 bands, the scene classification layer and acquisition metadata. Convert digital numbers to reflectance using `(DN + dn_offset) * dn_scale`. The stored footprint is 2.56 km; the primary analysis uses its central 1.28 km. Class and split tables preserve the study identifiers. Use the retained-window mask and class admission flags for the reported analysis.

## Source licences

OSM-derived labels are licensed under ODbL 1.0, © OpenStreetMap contributors. Wiki descriptions are CC BY-SA 2.0, © OpenStreetMap Wiki contributors; the phrase table gives each source. Imagery contains modified Copernicus Sentinel data (2024–2025), subject to the Copernicus Sentinel data terms. Ecoregion attributes derive from RESOLVE Ecoregions 2017 (Dinerstein and colleagues), under CC BY 4.0. These source terms apply to their respective contents.

## Use and limitations

The labels describe mapped land-cover and land-use categories. They can be incomplete or out of date, and some categories are difficult to distinguish from satellite imagery. Split controls and image quality filters do not remove every source of bias. This dataset provides a research benchmark, with no independent claim of operational suitability.

The adjacent manifest records file hashes. This build is prepared for review; uploading it is a separate release step.
'''
    path=ctx.stage_dir()/'README.md';path.write_text(card)
    manifest['files']['README.md']=sha256_file(path)
    atomic_write_json(ctx.stage_dir()/'manifest.json',manifest)
    return {'shards':len(shards),'files':len(manifest['files']),'uploaded':False}


STAGES=[Stage('release_build','P3','cpu',('analysis_inference','audit_pack'),plan,run,finalize,
              gate='APPROVE_RELEASE',unit_cpus=4,unit_mem_gb=12,est_unit_min=10)]
