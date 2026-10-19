"""Class names (frozen template table) and phrases P1/P2/P3 (text_build helpers). Stdlib-only at import."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ._text_wiki import ATTRIBUTION, LICENCE, _present, is_meta, strip_meta

NAME_LICENCE = "n/a (name generated from OSM tags by the frozen template table)"


# --------------------------------------------------------------------------- taxonomy / templates
def load_taxonomy(path: Path) -> dict:
    """taxonomy.yaml (YAML, or JSON which is valid YAML). Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    txt = path.read_text()
    try:
        import yaml  # type: ignore
        doc = yaml.safe_load(txt)
    except ImportError:
        doc = json.loads(txt)
    return doc or {}


def template_table(tax: dict) -> dict[str, dict[str, str]]:
    """Normalise the taxonomy's template section to {'refine': {key: tpl}, 'primary': {k=v: name},
    'primary_key': {key: tpl}}. Accepts a flat {refine_key: tpl} map, or nested sections."""
    t = tax.get("templates") or tax.get("name_templates") or {}
    out: dict[str, dict[str, str]] = {"refine": {}, "primary": {}, "primary_key": {}}
    if not isinstance(t, dict):
        return out
    nested = any(k in t for k in ("refine", "primary", "primary_key", "refinement", "primary_tag"))
    if nested:
        out["refine"].update({str(k): str(v) for k, v in (t.get("refine") or t.get("refinement") or {}).items()})
        out["primary"].update({str(k): str(v) for k, v in (t.get("primary") or t.get("primary_tag") or {}).items()})
        out["primary_key"].update({str(k): str(v) for k, v in (t.get("primary_key") or {}).items()})
    else:
        for k, v in t.items():
            (out["primary"] if "=" in str(k) else out["refine"])[str(k)] = str(v)
    return out


def humanise(v: str) -> str:
    return re.sub(r"\s+", " ", str(v).replace("_", " ").replace(";", " / ")).strip()


def _fill(tpl: str, **kw: str) -> str:
    out = tpl
    for k, v in kw.items():
        out = out.replace("{" + k + "}", v)
    return re.sub(r"\s+", " ", out).strip()


def render_name(row: dict, table: dict[str, dict[str, str]]) -> str:
    pk, pv = str(row["primary_key"]), str(row["primary_value"])
    p = table["primary"].get(f"{pk}={pv}")
    if p is None:
        p = _fill(table["primary_key"].get(pk, "{v}"), v=humanise(pv), value=humanise(pv), k=humanise(pk))
    rk, rv = row.get("refine_key"), row.get("refine_value")
    if not (_present(rk) and _present(rv)):
        return p
    tpl = table["refine"].get(str(rk), "{v} {p}")
    return _fill(tpl, v=humanise(rv), value=humanise(rv), p=p, primary=p, k=humanise(rk))


def normalise_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# --------------------------------------------------------------------------- descriptions
def usable_description(rec: dict | None, meta_policy: str) -> tuple[str, list[str]]:
    """(cleaned description or '', flags) for one parsed wiki row."""
    if rec is None:
        return "", ["no_wiki_row"]
    if not rec.get("has_en"):
        return "", ["no_en_page"]
    d = rec.get("description") or ""
    flags: list[str] = []
    if not d:
        return "", ["empty_description"]
    if meta_policy == "strip":
        d2, dropped = strip_meta(d)
        if dropped:
            flags.append("meta_stripped" if d2 else "meta_only")
        d = d2
    elif is_meta(d):
        flags.append("meta_kept")
    if str(rec.get("status") or "").lower() in ("deprecated", "obsolete", "discardable"):
        flags.append(f"status_{str(rec['status']).lower()}")
    return d, flags


def source_string(rec: dict) -> str:
    return f"taginfo:{rec['item']}:en | {rec.get('wiki_url') or ''}".strip(" |")


def build_phrases(rows: list[dict], wiki: dict[str, dict], table: dict[str, dict[str, str]],
                  knobs: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Return (phrase rows, class rows). `wiki` maps item ('tag:k=v' / 'key:k') -> parsed wiki row.

    P1 = name; P2 = name + primary-tag description; P3 = name + refinement-tag description, falling back
    to the refinement-key description (`p3_policy` decides what unrefined classes get).
    """
    fmt = knobs.get("phrase_template", "{name}: {desc}")
    meta_policy = knobs.get("meta_policy", "strip")
    p3_policy = knobs.get("p3_policy", "refine_tag_else_refine_key")
    phrases: list[dict] = []
    classes: list[dict] = []
    for r in sorted(rows, key=lambda r: int(r["class_idx"])):
        cidx = int(r["class_idx"])
        rendered = render_name(r, table)
        given = r.get("name")
        name = str(given).strip() if _present(given) and str(given).strip() else rendered
        name_src = "classes.parquet:name" if name != rendered or (_present(given) and str(given).strip()) else "template"
        pk, pv = str(r["primary_key"]), str(r["primary_value"])
        rk, rv = r.get("refine_key"), r.get("refine_value")
        refined = _present(rk) and _present(rv)
        cflags: list[str] = []
        out = [dict(class_idx=cidx, phrase_kind="P1", text=name, desc_text=None, source=f"name:{name_src}",
                    licence=NAME_LICENCE, retrieved_at=None, flags="")]
        # P2: primary tag description
        p2rec = wiki.get(f"tag:{pk}={pv}")
        d2, f2 = usable_description(p2rec, meta_policy)
        if d2:
            out.append(dict(class_idx=cidx, phrase_kind="P2", text=_fill(fmt, name=name, desc=d2), desc_text=d2,
                            source=source_string(p2rec), licence=LICENCE, retrieved_at=p2rec.get("retrieved_at"),
                            flags=",".join(f2)))
        cflags += [f"p2_{f}" for f in f2]
        # P3: refinement description (or key description)
        p3kind, d3, f3, p3rec = "none", "", [], None
        if refined:
            cand = [("refine_tag", f"tag:{rk}={rv}"), ("refine_key", f"key:{rk}")]
            if p3_policy == "refine_tag_only":
                cand = cand[:1]
        elif p3_policy == "refine_tag_else_refine_key_else_primary_key":
            cand = [("primary_key", f"key:{pk}")]
        else:
            cand = []
        for kind, item in cand:
            rec = wiki.get(item)
            d, f = usable_description(rec, meta_policy)
            f3 += [f"{kind}_{x}" for x in f]
            if d:
                p3kind, d3, p3rec = kind, d, rec
                break
        if d3:
            t3 = _fill(fmt, name=name, desc=d3)
            if any(p["text"] == t3 for p in out):
                cflags.append("p3_duplicate_dropped")
            else:
                out.append(dict(class_idx=cidx, phrase_kind="P3", text=t3, desc_text=d3, source=source_string(p3rec),
                                licence=LICENCE, retrieved_at=p3rec.get("retrieved_at"),
                                flags=",".join([f"p3_from_{p3kind}"] + [x for x in f3 if x])))
        cflags += [f"p3_{f}" for f in f3]
        phrases += out
        n_desc = sum(p["phrase_kind"] != "P1" for p in out)
        statuses = [str((wiki.get(i) or {}).get("status") or "") for i in
                    ([f"tag:{pk}={pv}"] + ([f"tag:{rk}={rv}"] if refined else []))]
        classes.append(dict(
            class_idx=cidx, class_id=str(r["class_id"]), name=name, name_rendered=rendered, name_source=name_src,
            refined=bool(refined), n_phrases=len(out), has_p2=any(p["phrase_kind"] == "P2" for p in out),
            has_p3=any(p["phrase_kind"] == "P3" for p in out), p3_source=p3kind if d3 else "none",
            no_description=n_desc == 0, primary_status=statuses[0] or None,
            refine_status=(statuses[1] or None) if refined else None,
            deprecated=any(s.lower() == "deprecated" for s in statuses),
            flags=",".join(sorted(set(cflags)))))
    _flag_templated(phrases, classes, rows, int(knobs.get("templated_min_group", 3)))
    return phrases, classes


def _flag_templated(phrases: list[dict], classes: list[dict], rows: list[dict], min_group: int) -> None:
    """Flag P3 refinement descriptions that are one template filled with the value (strawman O6):
    masking the value leaves a string shared by >= min_group siblings of the same refinement key."""
    by_idx = {int(r["class_idx"]): r for r in rows}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in phrases:
        if p["phrase_kind"] != "P3" or "p3_from_refine_tag" not in p["flags"]:
            continue
        r = by_idx[p["class_idx"]]
        rv = str(r["refine_value"])
        masked = p["desc_text"]
        for tok in sorted({rv, humanise(rv), rv.replace("_", " ")}, key=len, reverse=True):
            masked = re.sub(re.escape(tok), "<V>", masked, flags=re.I)
        groups[(str(r["refine_key"]), masked.lower())].append(p)
    templated = set()
    for (_, masked), ps in groups.items():
        if len(ps) >= min_group and "<v>" in masked:
            for p in ps:
                p["flags"] = ",".join(x for x in [p["flags"], "templated"] if x)
                templated.add(p["class_idx"])
    for c in classes:
        c["templated_p3"] = c["class_idx"] in templated


def desc_only_texts(phr: list[dict], class_idx: int) -> list[str]:
    seen, out = set(), []
    for p in phr:
        if p["class_idx"] == class_idx and p["phrase_kind"] in ("P2", "P3") and p.get("desc_text"):
            if p["desc_text"] not in seen:
                seen.add(p["desc_text"])
                out.append(p["desc_text"])
    return out


def attribution() -> dict:
    return {"descriptions": {"licence": LICENCE, "attribution": ATTRIBUTION,
                             "licence_url": "https://creativecommons.org/licenses/by-sa/2.0/",
                             "note": "per-phrase source column carries the wiki page URL"},
            "names": {"licence": NAME_LICENCE}}
