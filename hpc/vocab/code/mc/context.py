"""Ctx: what a stage's plan/run/finalize functions receive."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Store, host, worker_id


@dataclass
class Ctx:
    cfg: dict
    code: Path            # deployed code root (the hpc/vocab directory of this release)
    root: Path            # cfg["root"] (vocab-01)
    run: Path             # root/runs/<profile>: ALL stage outputs live here
    shared: Path          # root/shared: profile-independent caches (planet, ecoregions, ...)
    store: Store
    stage: str = ""
    unit: str = ""
    threads: int = 1
    gpus: int = 0
    tmp: Path = field(default_factory=lambda: Path("/tmp"))
    profile: str = "full"

    # -- paths -------------------------------------------------------------
    def p(self, *parts: str) -> Path:
        return self.run.joinpath(*parts)

    def stage_dir(self, stage: str | None = None) -> Path:
        d = self.run / (stage or self.stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_root(self) -> Path:
        return Path(os.environ.get("MEGAEARTH_MODEL_ROOT") or self.cfg["model_root"])

    @property
    def tiny(self) -> bool:
        return self.profile != "full"

    # -- state helpers -----------------------------------------------------
    def units(self, stage: str | None = None) -> list[dict]:
        return self.store.read_units(stage or self.stage) or []

    def done_records(self, stage: str | None = None) -> list[dict]:
        return self.store.done_records(stage or self.stage)

    def stage_report(self, stage: str) -> dict:
        st = self.store.stage_status(stage) or {}
        return st.get("report", {})

    # -- environment -------------------------------------------------------
    @property
    def proxy(self) -> str | None:
        return os.environ.get("VOCAB_PROXY")

    def requests_session(self, retries: int = 6, backoff: float = 1.0, pool: int = 32):
        """requests.Session routed through the job's SOCKS tunnel, with retry/backoff on 429/5xx."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        r = Retry(total=retries, connect=retries, read=retries, backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=True,
                  allowed_methods=frozenset({"GET", "HEAD", "POST"}))
        a = HTTPAdapter(max_retries=r, pool_connections=pool, pool_maxsize=pool)
        s.mount("https://", a)
        s.mount("http://", a)
        if self.proxy:
            s.proxies = {"http": self.proxy, "https": self.proxy}
        s.headers["User-Agent"] = "megaearth-research/1.0 (Edinburgh Napier University; contact via GitHub JamesEBall)"
        return s

    def log(self, msg: str, **kw: Any) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        extra = (" " + json.dumps(kw, default=str)) if kw else ""
        print(f"{ts} [{host()}:{worker_id()}] {self.stage}/{self.unit}: {msg}{extra}", file=sys.stderr, flush=True)

    def event(self, kind: str, **kw: Any) -> None:
        self.store.event(kind, stage=self.stage, unit=self.unit, **kw)

    def free_gb(self, path: Path | None = None) -> float:
        u = shutil.disk_usage(path or self.run)
        return u.free / 1e9


def make_ctx(cfg: dict, code: Path, stage: str = "", unit: str = "", threads: int | None = None,
             gpus: int = 0, tmp: Path | None = None) -> Ctx:
    root = Path(os.environ.get("MEGAEARTH_ROOT") or cfg["root"])
    profile = cfg.get("_profile", "full")
    run = Path(os.environ.get("MEGAEARTH_RUN") or (root / "runs" / profile))
    shared = Path(os.environ.get("MEGAEARTH_SHARED") or (root / "shared"))
    run.mkdir(parents=True, exist_ok=True)
    shared.mkdir(parents=True, exist_ok=True)
    if tmp is None:
        base = Path(os.environ.get("MEGAEARTH_TMP") or os.environ.get("TMPDIR") or "/tmp")
        tmp = base / f"megaearth-{worker_id()}-{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    return Ctx(cfg=cfg, code=Path(code), root=root, run=run, shared=shared, store=Store(run),
               stage=stage, unit=unit,
               threads=threads or int(os.environ.get("MEGAEARTH_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1),
               gpus=gpus, tmp=tmp, profile=profile)
