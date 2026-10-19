"""OSM-wiki descriptions via the taginfo API: fetch, cache, parse, clean (text_fetch / text_build helpers).

Stdlib-only at import time. Network access only through a `requests.Session` handed in by the caller
(`ctx.requests_session()` on the net pool), so tests can inject a fake session.

Cache layout (profile-independent, under `S/text/`):
  taginfo/tag/<quote(key)>=<quote(value)>.json   one /api/4/tag/wiki_pages response
  taginfo/key/<quote(key)>.json                  one /api/4/key/wiki_pages response
  wiki/redirects-<sha12>.json                     one OSM-wiki redirect query batch
Each cache file stores the full parsed body plus url, params, UTC retrieval time and body sha256, so the
snapshot is reproducible and re-runs never hit the network again.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

LICENCE = "CC BY-SA 2.0"
ATTRIBUTION = "OpenStreetMap Wiki contributors (wiki.openstreetmap.org), via taginfo.openstreetmap.org"
WIKI_BASE = "https://wiki.openstreetmap.org/wiki/"


# --------------------------------------------------------------------------- tags
def class_tags(row: dict) -> dict[str, list[str]]:
    """Wiki items needed for one class row: tags ('k=v') and keys ('k')."""
    pk, pv = row["primary_key"], row["primary_value"]
    rk, rv = row.get("refine_key"), row.get("refine_value")
    tags, keys = [f"{pk}={pv}"], [pk]
    if _present(rk) and _present(rv):
        tags.append(f"{rk}={rv}")
        keys.append(str(rk))
    return {"tags": tags, "keys": keys}


def _present(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and x != x:  # NaN
        return False
    return str(x) != ""


def item_list(rows: Iterable[dict], extra_tags: Iterable[str] = ()) -> list[str]:
    """Sorted, de-duplicated wiki items: 'tag:k=v' and 'key:k'."""
    out: set[str] = set()
    for r in rows:
        t = class_tags(r)
        out.update(f"tag:{x}" for x in t["tags"])
        out.update(f"key:{x}" for x in t["keys"])
    for x in extra_tags:
        out.add(f"tag:{x}" if "=" in x else f"key:{x}")
    return sorted(out)


def split_item(item: str) -> tuple[str, str, str]:
    kind, rest = item.split(":", 1)
    if kind == "tag":
        k, v = rest.split("=", 1)
        return kind, k, v
    return kind, rest, ""


def cache_path(shared_text: Path, item: str) -> Path:
    kind, k, v = split_item(item)
    if kind == "tag":
        return shared_text / "taginfo" / "tag" / f"{quote(k, safe='')}={quote(v, safe='')}.json"
    return shared_text / "taginfo" / "key" / f"{quote(k, safe='')}.json"


def request_spec(base: str, item: str) -> tuple[str, dict]:
    kind, k, v = split_item(item)
    if kind == "tag":
        return f"{base.rstrip('/')}/tag/wiki_pages", {"key": k, "value": v}
    return f"{base.rstrip('/')}/key/wiki_pages", {"key": k}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Throttle:
    """Minimum interval between requests (per unit/process)."""

    def __init__(self, min_interval_s: float):
        self.dt = float(min_interval_s)
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if self.last and now - self.last < self.dt:
            time.sleep(self.dt - (now - self.last))
        self.last = time.monotonic()


def fetch_item(session, base: str, item: str, timeout: float, throttle: Throttle) -> dict:
    """GET one taginfo wiki_pages endpoint. Raises on non-200 (the unit is retried)."""
    url, params = request_spec(base, item)
    throttle.wait()
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"taginfo {item}: HTTP {r.status_code}")
    body = r.content
    doc = json.loads(body)
    if not isinstance(doc, dict) or "data" not in doc:
        raise RuntimeError(f"taginfo {item}: unexpected body {body[:200]!r}")
    return {"item": item, "endpoint": url, "params": params, "url": getattr(r, "url", url),
            "retrieved_at": utc_now(), "http_status": r.status_code, "body_sha256": sha256_bytes(body),
            "data_until": doc.get("data_until"), "body": doc}


def load_or_fetch(shared_text: Path, session_factory, base: str, item: str, timeout: float,
                  throttle: Throttle, refresh: bool = False) -> tuple[dict, bool]:
    """Return (cache record, fetched_now). `session_factory` is only called when a fetch is needed."""
    from mc.core import atomic_write_json, read_json
    p = cache_path(shared_text, item)
    rec = None if refresh else read_json(p)
    if rec is not None:
        return rec, False
    rec = fetch_item(session_factory(), base, item, timeout, throttle)
    atomic_write_json(p, rec)
    return rec, True


# --------------------------------------------------------------------------- parsing
def english_entry(body: dict, lang: str = "en") -> dict | None:
    for e in body.get("data") or []:
        if e.get("lang") == lang:
            return e
    return None


def wiki_url(title: str | None) -> str | None:
    if not title:
        return None
    return WIKI_BASE + quote(title.replace(" ", "_"), safe=":=/_()-,.'")


def parse_record(rec: dict, lang: str = "en") -> dict:
    """Flatten one cache record into a wiki.parquet row (English entry only)."""
    kind, k, v = split_item(rec["item"])
    body = rec.get("body") or {}
    e = english_entry(body, lang)
    langs = sorted({str(x.get("lang")) for x in body.get("data") or []})
    raw = (e or {}).get("description") or ""
    clean = clean_markup(raw)
    status = (e or {}).get("status") or None
    return {
        "item": rec["item"], "kind": kind, "key": k, "value": v,
        "tag": f"{k}={v}" if kind == "tag" else k,
        "has_en": e is not None, "title": (e or {}).get("title"), "wiki_url": wiki_url((e or {}).get("title")),
        "description_raw": raw, "description": clean,
        "status": status, "on_area": bool((e or {}).get("on_area", False)),
        "tags_implies": sorted(map(str, (e or {}).get("tags_implies") or [])),
        "tags_combination": sorted(map(str, (e or {}).get("tags_combination") or [])),
        "tags_linked": sorted(map(str, (e or {}).get("tags_linked") or [])),
        "replacement": deprecated_replacement(raw),
        "n_langs": len(langs), "url": rec.get("url"), "retrieved_at": rec.get("retrieved_at"),
        "body_sha256": rec.get("body_sha256"), "data_until": rec.get("data_until"),
        "licence": LICENCE,
    }


_TAG_REF = re.compile(r"(?:tag:)?([A-Za-z_][\w:.-]*=[\w:.;/+-]+)", re.I)
_REPLACE = re.compile(r"\buse\s+(.+?)\s+instead\b", re.I)


def deprecated_replacement(desc: str) -> list[str]:
    """'Using this tag is discouraged; use tag:natural=water + tag:water=reservoir instead.' ->
    ['natural=water', 'water=reservoir'] (sorted). Empty when no 'use ... instead' clause."""
    if not desc:
        return []
    m = _REPLACE.search(clean_markup(desc))
    if not m:
        return []
    return sorted({t.rstrip(".,;") for t in _TAG_REF.findall(m.group(1))})


# --------------------------------------------------------------------------- markup cleaning
_TEMPLATE_TAG = re.compile(r"\{\{\s*(?:tag|tagvalue|key)\s*\|([^{}|]*)(?:\|([^{}|]*))?(?:\|[^{}]*)?\}\}", re.I)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_LINK_PIPE = re.compile(r"\[\[(?:[^\[\]|]*\|)([^\[\]]*)\]\]")
_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_EXT_LINK_TEXT = re.compile(r"\[(?:https?|ftp)://\S+\s+([^\]]+)\]")
_EXT_LINK = re.compile(r"\[(?:https?|ftp)://[^\]\s]+\]")
_URL = re.compile(r"(?:https?|ftp)://\S+")
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.I | re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_BOLD_ITALIC = re.compile(r"'{2,5}")
_WS = re.compile(r"\s+")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?)])")
_OPEN_PAREN = re.compile(r"\(\s+")
_EMPTY_PAREN = re.compile(r"\(\s*\)")


def clean_markup(s: str | None) -> str:
    """Deterministic MediaWiki/HTML markup stripping. Idempotent; never rewrites wording."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = _COMMENT.sub(" ", s)
    s = _REF.sub(" ", s)
    s = html.unescape(s)
    for _ in range(3):  # nested templates resolve inside-out
        s = _TEMPLATE_TAG.sub(lambda m: m.group(1).strip() + (f"={m.group(2).strip()}" if m.group(2) else ""), s)
        s2 = _TEMPLATE.sub(" ", s)
        if s2 == s:
            break
        s = s2
    s = _LINK_PIPE.sub(r"\1", s)
    s = _LINK.sub(lambda m: m.group(1).split(":", 1)[-1] if m.group(1).lower().startswith(("w:", "wikipedia:"))
                  else m.group(1), s)
    s = _EXT_LINK_TEXT.sub(r"\1", s)
    s = _EXT_LINK.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    s = _BOLD_ITALIC.sub("", s)
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = _WS.sub(" ", s).strip()
    s = _EMPTY_PAREN.sub("", s)
    s = _OPEN_PAREN.sub("(", s)
    s = _SPACE_PUNCT.sub(r"\1", s)
    s = _WS.sub(" ", s).strip()
    return s


_META = re.compile(
    r"\b(discouraged|deprecated|do not use|don't use|please use|should not be used|"
    r"use\s+.+?\s+instead|this tag is (?:obsolete|outdated)|not recommended)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_sentences(s: str) -> list[str]:
    return [x.strip() for x in _SENT.split(s) if x.strip()]


def strip_meta(desc: str) -> tuple[str, bool]:
    """Drop editorial/meta sentences ('Using this tag is discouraged; use ... instead.').
    Returns (remaining text, any_dropped)."""
    if not desc:
        return "", False
    sents = split_sentences(desc)
    keep = [x for x in sents if not _META.search(x)]
    return " ".join(keep).strip(), len(keep) != len(sents)


def is_meta(desc: str) -> bool:
    return bool(desc) and bool(_META.search(desc))


# --------------------------------------------------------------------------- OSM-wiki redirects
def redirect_titles(items: Iterable[str]) -> list[str]:
    out = []
    for it in items:
        kind, k, v = split_item(it)
        out.append(f"Tag:{k}={v}" if kind == "tag" else f"Key:{k}")
    return sorted(set(out))


def fetch_redirects(session, api: str, titles: list[str], timeout: float, throttle: Throttle) -> dict:
    """One MediaWiki `action=query&redirects=1` call for <= 50 titles."""
    params = {"action": "query", "format": "json", "redirects": "1", "titles": "|".join(titles)}
    throttle.wait()
    r = session.get(api, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"wiki redirects: HTTP {r.status_code}")
    doc = json.loads(r.content)
    return {"titles": titles, "url": api, "retrieved_at": utc_now(), "body_sha256": sha256_bytes(r.content),
            "body": doc}


def parse_redirects(rec: dict) -> list[dict]:
    """[{'from': 'Tag:a=b', 'to': 'Tag:c=d'}] after MediaWiki title normalisation (spaces->_ undone)."""
    q = (rec.get("body") or {}).get("query") or {}
    norm = {n["to"]: n["from"] for n in q.get("normalized") or []}
    out = []
    for r in q.get("redirects") or []:
        frm = norm.get(r["from"], r["from"])
        out.append({"from": frm.replace(" ", "_"), "to": str(r["to"]).replace(" ", "_")})
    return sorted(out, key=lambda d: (d["from"], d["to"]))


def title_to_item(title: str) -> str | None:
    if title.startswith("Tag:") and "=" in title:
        return "tag:" + title[4:]
    if title.startswith("Key:"):
        return "key:" + title[4:]
    return None
