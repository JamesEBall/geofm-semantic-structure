"""~🌍 MegaEarth 🛰️~ Mission Control CLI.

    python -m mc.cli dag        [--profile P]            # print the stage DAG
    python -m mc.cli local      [--profile P] [--only a,b] [--threads N]
    python -m mc.cli run-unit   [--profile P] STAGE UNIT [--threads N] [--gpus G]
    python -m mc.cli status     [--profile P] [--json]
    python -m mc.cli governor   [--profile P] [--once] [--dry-run]
    python -m mc.cli worker     [--profile P] --pool POOL --cpus N --mem-gb M [--gpus G]
    python -m mc.cli flag       [--profile P] NAME on|off [--note TEXT]
    python -m mc.cli amend-freeze [--profile P] f1|f2|f3 AMENDMENT [--note TEXT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]      # hpc/vocab/code
CODE_ROOT = CODE_DIR.parent                          # hpc/vocab (config.json lives here)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mc.core import load_config  # noqa: E402


def _cfg(args) -> dict:
    return load_config(CODE_ROOT, args.profile)


def cmd_dag(args) -> int:
    from mc.registry import critical_path_rank, load_stages, topo_order
    cfg = _cfg(args)
    st = load_stages(cfg)
    if not st:
        raise ValueError("No stages are implemented for this profile")
    rank = critical_path_rank(st)
    for n in topo_order(st):
        s = st[n]
        print(f"{s.phase} {n:<22} pool={s.pool:<5} cpus={s.unit_cpus:<3} mem={s.unit_mem_gb:<6g} "
              f"gate={s.gate or '-':<16} cp={rank[n]:8.0f}  deps={','.join(s.deps) or '-'}")
    return 0


def cmd_local(args) -> int:
    from mc.runner import local_run
    cfg = _cfg(args)
    only = args.only.split(",") if args.only else None
    res = local_run(cfg, CODE_ROOT, only=only, threads=args.threads, approve=args.approve_selftest)
    print(json.dumps(res, indent=1))
    bad = [n for n, p in res.items() if p.get("status") != "complete"]
    return 1 if bad else 0


def cmd_run_unit(args) -> int:
    from mc.runner import execute, install_stop_handlers
    install_stop_handlers()
    cfg = _cfg(args)
    out = execute(cfg, CODE_ROOT, args.stage, args.unit, threads=args.threads, gpus=args.gpus)
    print(out)
    return {"done": 0, "skipped": 0, "yielded": 75, "gated": 2, "blocked": 2}.get(out, 1)


def cmd_status(args) -> int:
    from mc.status import render_status
    cfg = _cfg(args)
    doc, text = render_status(cfg, CODE_ROOT)
    print(json.dumps(doc, indent=1, default=str) if args.json else text)
    return 0


def cmd_governor(args) -> int:
    from mc.governor import main as gmain
    return gmain(_cfg(args), CODE_ROOT, once=args.once, dry_run=args.dry_run)


def cmd_worker(args) -> int:
    from mc.worker import main as wmain
    return wmain(_cfg(args), CODE_ROOT, pool=args.pool, cpus=args.cpus, mem_gb=args.mem_gb, gpus=args.gpus)


def cmd_amend_freeze(args) -> int:
    from mc.context import make_ctx
    from stages._extract_freeze import amend_freeze
    ctx = make_ctx(_cfg(args), CODE_ROOT)
    entry = amend_freeze(ctx, args.freeze, args.amendment, args.note)
    print(json.dumps(entry, indent=1))
    return 0


def cmd_flag(args) -> int:
    from mc.context import make_ctx
    ctx = make_ctx(_cfg(args), CODE_ROOT)
    ctx.store.set_flag(args.name, args.state == "on", note=args.note)
    print(f"{args.name}={'on' if ctx.store.flag(args.name) else 'off'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="megaearth")
    p.add_argument("--profile", default=os.environ.get("MEGAEARTH_PROFILE", "full"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dag")
    a = sub.add_parser("local")
    a.add_argument("--only", default="")
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--honour-gates", action="store_true", help="Compatibility option; gates are always enforced")
    a.add_argument("--approve-selftest", action="store_true", help="Approve gates only in the synthetic selftest profile")
    a = sub.add_parser("run-unit")
    a.add_argument("stage")
    a.add_argument("unit")
    a.add_argument("--threads", type=int, default=int(os.environ.get("MEGAEARTH_THREADS", "1")))
    a.add_argument("--gpus", type=int, default=0)
    a = sub.add_parser("status")
    a.add_argument("--json", action="store_true")
    a = sub.add_parser("governor")
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("worker")
    a.add_argument("--pool", required=True)
    a.add_argument("--cpus", type=int, required=True)
    a.add_argument("--mem-gb", type=float, required=True)
    a.add_argument("--gpus", type=int, default=0)
    a = sub.add_parser("amend-freeze", help="record an accepted protocol amendment against a freeze (files unchanged)")
    a.add_argument("freeze", choices=("f1", "f2", "f3"))
    a.add_argument("amendment")
    a.add_argument("--note", default="")
    a = sub.add_parser("flag")
    a.add_argument("name")
    a.add_argument("state", choices=("on", "off"))
    a.add_argument("--note", default="")
    args = p.parse_args(argv)
    fn = {"dag": cmd_dag, "local": cmd_local, "run-unit": cmd_run_unit, "status": cmd_status,
          "governor": cmd_governor, "worker": cmd_worker, "flag": cmd_flag, "amend-freeze": cmd_amend_freeze}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
