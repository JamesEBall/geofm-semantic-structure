"""Frozen wiki phrases, offline language embeddings and leakage-resistant families.

Author: James Edward Ball. Only text and taxonomy enter family construction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mc.core import atomic_path, atomic_write_json, read_json, sha256_file
from mc.registry import Stage, StageFailed
from . import _text_phrases as phrases
from . import _text_wiki as wiki


def single(ctx):
    return [{"id": "all"}]


def class_rows(ctx):
    import pandas as pd
    rows = pd.read_parquet(ctx.p("vocab", "classes.parquet")).sort_values("class_idx").to_dict("records")
    if [int(r["class_idx"]) for r in rows] != list(range(len(rows))) or not rows:
        raise StageFailed("Class indices must be nonempty and contiguous")
    return rows


def parquet(path, rows):
    import pandas as pd
    with atomic_path(path) as p:
        pd.DataFrame(rows).to_parquet(p, index=False)


def fetch_plan(ctx):
    return [{"id": hashlib.sha256(i.encode()).hexdigest()[:24], "item": i}
            for i in wiki.item_list(class_rows(ctx))]


def fetch_run(ctx, unit, ckpt):
    session = ctx.requests_session()
    rec, fetched = wiki.load_or_fetch(ctx.shared / "text", lambda: session,
        "https://taginfo.openstreetmap.org/api/4", unit["item"], 60, wiki.Throttle(0.25))
    ckpt.maybe_stop()
    dest = ctx.stage_dir() / (unit["id"] + ".json")
    atomic_write_json(dest, rec)
    session.close()
    return {"outputs": [dest], "metrics": {"fetched": fetched}}


def build_run(ctx, unit, ckpt):
    records = [wiki.parse_record(read_json(ctx.p("text_fetch", u["id"] + ".json")))
               for u in ctx.units("text_fetch")]
    tax = phrases.load_taxonomy(ctx.p("vocab", "taxonomy.yaml"))
    pp, cc = phrases.build_phrases(class_rows(ctx), {r["item"]: r for r in records},
                                   phrases.template_table(tax), {})
    # Canonical interface intentionally places all text artefacts in R/text.
    out = ctx.stage_dir("text")
    outputs = [out / "phrases.parquet", out / "classes_text.parquet", out / "wiki.parquet", out / "attribution.json"]
    for path, rows in zip(outputs, (pp, cc, records)):
        parquet(path, rows)
    atomic_write_json(outputs[3], phrases.attribution())
    return {"outputs": outputs, "metrics": {"classes": len(cc), "phrases": len(pp),
            "name_only": sum(r["no_description"] for r in cc)}}


def embed_plan(ctx):
    return [{"id": e, "encoder": e} for e in ctx.cfg["text"]["encoders"]]


def normalise(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norm <= 0).any():
        raise ValueError("Invalid text embedding")
    return x / norm


def pool_phrases(rows, vectors, n_classes):
    """Mean normalised phrase vectors; sensitivities keep explicit missing descriptions."""
    import numpy as np
    main, names, desc, missing = [], [], [], []
    vectors = normalise(vectors)
    for c in range(n_classes):
        ids = [i for i, r in enumerate(rows) if int(r["class_idx"]) == c]
        ni = [i for i in ids if rows[i]["phrase_kind"] == "P1"]
        di = [i for i in ids if rows[i]["phrase_kind"] != "P1"]
        if len(ni) != 1:
            raise ValueError("Each class requires exactly one name phrase")
        main.append(vectors[ids].mean(0)); names.append(vectors[ni[0]])
        # Fallback is flagged for exclusion from description-only sensitivity.
        desc.append(vectors[di].mean(0) if di else vectors[ni[0]])
        missing.append(not di)
    return {"main": normalise(main), "name": normalise(names), "desc": normalise(desc)}, missing


def embed_run(ctx, unit, ckpt):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(ctx.threads)
    ref = unit["encoder"]
    name = "qwen" if ref.startswith("qwen") else "mpnet"
    root = ctx.model_root / "weights" / name
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if not files:
        raise StageFailed("Offline text weights missing")
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                                       padding_side="left" if name == "qwen" else "right")
    model = AutoModel.from_pretrained(root, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    rows = pd.read_parquet(ctx.p("text", "phrases.parquet")).to_dict("records")
    prefix = ctx.cfg["text"]["instruction"] if ref == "qwen_instructed" else ""
    # Description-only variants re-encode the description without the category name.
    strings = [prefix + r["text"] for r in rows]
    drows = [dict(r, text=r.get("desc_text") or r["text"]) for r in rows]
    dstrings = [prefix + r["text"] for r in drows]
    def encode(texts):
        batches = []
        for start in range(0, len(texts), 16):
            ckpt.maybe_stop()
            batch = tok(texts[start:start+16], padding=True, truncation=False, return_tensors="pt")
            maxlen = int(getattr(model.config, "max_position_embeddings", 512))
            if batch["input_ids"].shape[1] > maxlen:
                raise StageFailed("Phrase exceeds encoder context; no silent truncation")
            with torch.inference_mode():
                h = model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                v = h[:, -1] if name == "qwen" else (h * mask[..., None]).sum(1) / mask.sum(1)[:, None]
            batches.append(v.numpy())
        return np.concatenate(batches)
    n = len(class_rows(ctx))
    pooled, missing = pool_phrases(rows, encode(strings), n)
    dpooled, _ = pool_phrases(drows, encode(dstrings), n)
    pooled["desc"] = dpooled["desc"]
    outputs = []
    for kind, x in pooled.items():
        path = ctx.p("text", "emb_" + ref + ("" if kind == "main" else "_" + kind) + ".npy")
        with atomic_path(path) as p:
            np.save(p, x, allow_pickle=False)
        outputs.append(path)
    path = ctx.p("text", "emb_" + ref + ".meta.json")
    atomic_write_json(path, {"encoder": ref, "weights": files, "phrases_sha256": sha256_file(ctx.p("text", "phrases.parquet")),
        "description_missing": missing, "precision": "float32", "pooling": "last token" if name == "qwen" else "masked mean",
        "prototype": "normalised mean of unit phrase embeddings", "rows": n})
    return {"outputs": outputs + [path], "metrics": {"classes": n}}


def make_families(rows, embeddings, threshold, aliases=()):
    """Connected components of primary-tag, identical-name, explicit and ANY-reference edges."""
    import numpy as np
    n = len(rows)
    parent = list(range(n)); edges = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(i, j, reason):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
        edges.append((int(i), int(j), reason))
    groups = {}
    for i, row in enumerate(rows):
        for key in (("primary", row["primary_key"], row["primary_value"]),
                    ("name", phrases.normalise_name(row["name"]))):
            if key in groups:
                join(i, groups[key], key[0])
            else:
                groups[key] = i
    ids = {r["class_id"]: i for i, r in enumerate(rows)}
    for a, b in aliases:
        if a in ids and b in ids:
            join(ids[a], ids[b], "alias")
    for ref, E in embeddings.items():
        E = normalise(E)
        if E.shape[0] != n:
            raise ValueError("Text/class alignment mismatch")
        for start in range(0, n, 256):
            sim = E[start:start+256] @ E.T
            for i, j in np.argwhere(sim > threshold):
                i = int(i) + start; j = int(j)
                if i < j:
                    join(i, j, ref)
    roots = sorted({find(i) for i in range(n)})
    mapping = {r: i for i, r in enumerate(roots)}
    return np.array([mapping[find(i)] for i in range(n)], dtype=np.int32), edges


def families_run(ctx, unit, ckpt):
    import numpy as np
    from megaearth_stats.folds import make_folds
    rows = class_rows(ctx)
    tax = phrases.load_taxonomy(ctx.p("vocab", "taxonomy.yaml"))
    emb = {ref: np.load(ctx.p("text", f"emb_{ref}.npy"), allow_pickle=False) for ref in ctx.cfg["text"]["encoders"]}
    fam, edges = make_families(rows, emb, ctx.cfg["text"]["family_merge_cosine"], tax.get("family_aliases", []))
    folds = make_folds(fam, [r["primary_key"] for r in rows], ctx.cfg["analysis"]["outer_folds"], ctx.cfg["seed"])
    foldmap = {f: i for i, group in enumerate(folds["family_folds"]) for f in group}
    out = ctx.stage_dir()
    parquet(out / "families.parquet", [{"class_idx": i, "family_id": int(f), "fold": foldmap[int(f)]} for i, f in enumerate(fam)])
    atomic_write_json(out / "folds.json", folds)
    atomic_write_json(out / "edges.json", edges)
    sizes = np.bincount(fam)
    report = {"classes": len(rows), "families": len(sizes), "largest_family": int(sizes.max()),
              "largest_fraction": float(sizes.max() / len(rows)), "rule": "ANY reference cosine plus primary and alias edges"}
    atomic_write_json(out / "report.json", report)
    return {"outputs": [out / n for n in ("families.parquet", "folds.json", "edges.json", "report.json")], "metrics": report}


def prefetch_plan(ctx):
    """Wiki pages for every tag that could become a class, from the merged polygon store, before the
    class list exists. Rare tags (below the polygon floor) cannot meet admission and are skipped."""
    from ._osm_geometry import connect, class_fields
    knobs = ctx.cfg.get("text_prefetch", {})
    floor = int(knobs.get("min_polygons", 100)); max_units = max(1, int(knobs.get("max_units", 6)))
    with connect(ctx.p("osm_areas", "areas.sqlite"), True) as db:
        ids = [c for c, in db.execute("SELECT class_id FROM areas GROUP BY class_id HAVING COUNT(*) >= ? ORDER BY class_id", (floor,))]
    items = wiki.item_list(class_fields(c) for c in ids)
    size = -(-len(items) // max_units) if items else 1
    return [{"id": f"batch-{i // size:04d}", "items": items[i:i + size]} for i in range(0, len(items), size)]


def prefetch_run(ctx, unit, ckpt):
    throttle = wiki.Throttle(float(ctx.cfg.get("text_prefetch", {}).get("throttle_s", 0.5)))
    session = ctx.requests_session()
    outputs = []; fetched = 0
    try:
        for item in unit["items"]:
            ckpt.maybe_stop()
            _, hit = wiki.load_or_fetch(ctx.shared / "text", lambda: session, "https://taginfo.openstreetmap.org/api/4", item, 60, throttle)
            fetched += int(hit); outputs.append(wiki.cache_path(ctx.shared / "text", item))
    finally:
        session.close()
    return {"outputs": outputs, "metrics": {"items": len(unit["items"]), "fetched": fetched}}


def prefetch_final(ctx):
    return {"items": sum(len(u["items"]) for u in ctx.units()), "source": "taginfo wiki_pages, CC BY-SA 2.0"}


STAGES = [
    Stage("text_prefetch", "P1", "net", ("osm_areas",), prefetch_plan, prefetch_run, prefetch_final, unit_cpus=1, unit_mem_gb=4, est_unit_min=20),
    Stage("text_fetch", "P1", "net", ("vocab", "text_prefetch"), fetch_plan, fetch_run, unit_cpus=1, unit_mem_gb=2),
    Stage("text_build", "P1", "cpu", ("text_fetch",), single, build_run),
    Stage("text_embed", "P1", "cpu", ("text_build",), embed_plan, embed_run, unit_cpus=8, unit_mem_gb=16, est_unit_min=30),
    Stage("families", "P1", "cpu", ("text_embed",), single, families_run, unit_cpus=4, unit_mem_gb=8),
]
