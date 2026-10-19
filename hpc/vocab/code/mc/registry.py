"""Stage registry: the pipeline DAG, assembled from code/stages/<module>.py `STAGES` lists.

Stage modules MUST import only the standard library at module level (heavy imports go inside
functions) so the governor can load the DAG with the system python and no third-party packages.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

POOLS = ("cpu", "himem", "net", "gpu")
PHASES = ("P0", "P1", "P2", "P3")


class StageFailed(Exception):
    """Raise from finalize (or a unit) to fail the whole stage deliberately (e.g. a gate)."""


class UnitSkip(Exception):
    """Raise from a unit to mark it permanently unusable without retries (counted as failed)."""


@dataclass
class Stage:
    name: str
    phase: str
    pool: str
    deps: tuple[str, ...]
    plan: Callable[..., list[dict]]
    run: Callable[..., Any]
    finalize: Callable[..., dict] | None = None
    plan_pool: str | None = None      # where __plan__/__finalize__ run; default: pool, but gpu->cpu
    unit_cpus: int = 1
    unit_mem_gb: float = 4.0
    unit_gpus: int = 0
    max_attempts: int = 3
    est_unit_min: float = 5.0
    priority: int = 0
    gate: str | None = None
    allow_failed_frac: float = 0.0
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.pool not in POOLS:
            raise ValueError(f"{self.name}: pool {self.pool!r} not in {POOLS}")
        if self.phase not in PHASES:
            raise ValueError(f"{self.name}: phase {self.phase!r} not in {PHASES}")
        if self.pool == "gpu" and self.unit_gpus < 1:
            self.unit_gpus = 1
        if self.plan_pool is None:
            self.plan_pool = "cpu" if self.pool == "gpu" else self.pool
        self.deps = tuple(self.deps)

    def unit_res(self, unit: dict) -> tuple[int, float, int]:
        """(cpus, mem_gb, gpus) for a unit; units may override with cpus/mem_gb keys."""
        return (int(unit.get("cpus", self.unit_cpus)), float(unit.get("mem_gb", self.unit_mem_gb)),
                int(unit.get("gpus", self.unit_gpus)))


def stage_modules(cfg: dict) -> list[str]:
    """Auto-discover code/stages/*.py (no shared registry file for agents to fight over).

    Modules starting with '_' are private helpers; 'selftest*' modules load only when listed in
    cfg.mc.extra_stage_modules (the selftest profile) or cfg.mc.stage_modules is set explicitly.
    """
    import pkgutil
    import stages as pkg
    mc = cfg.get("mc", {})
    if mc.get("stage_modules"):
        return list(mc["stage_modules"])
    found = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                   if not m.name.startswith("_") and not m.name.startswith("selftest") and not m.ispkg)
    extra = mc.get("extra_stage_modules", [])
    return found + [m for m in extra if m not in found]


def load_stages(cfg: dict) -> dict[str, Stage]:
    """Import every stage module, validate the DAG, apply profile filters."""
    stages: dict[str, Stage] = {}
    for mod in stage_modules(cfg):
        m = importlib.import_module(f"stages.{mod}")
        for st in getattr(m, "STAGES", []):
            if st.name in stages:
                raise ValueError(f"duplicate stage {st.name} (module {mod})")
            stages[st.name] = st
    mc = cfg.get("mc", {})
    only = mc.get("only_stages")
    skip = set(mc.get("skip_stages", []))
    if only:
        keep = set(only)
        stages = {k: v for k, v in stages.items() if k in keep}
    stages = {k: v for k, v in stages.items() if k not in skip}
    for st in stages.values():
        missing = [d for d in st.deps if d not in stages]
        if missing:
            if mc.get("prune_missing_deps"):
                st.deps = tuple(d for d in st.deps if d in stages)
            else:
                raise ValueError(f"{st.name}: unknown deps {missing}")
    topo_order(stages)  # raises on cycles
    return stages


def topo_order(stages: dict[str, Stage]) -> list[str]:
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str, path: tuple[str, ...]) -> None:
        s = state.get(n, 0)
        if s == 2:
            return
        if s == 1:
            raise ValueError(f"cycle: {' -> '.join(path + (n,))}")
        state[n] = 1
        for d in stages[n].deps:
            visit(d, path + (n,))
        state[n] = 2
        order.append(n)

    for n in sorted(stages):
        visit(n, ())
    return order


def critical_path_rank(stages: dict[str, Stage]) -> dict[str, float]:
    """Longest downstream chain (in est minutes) — schedule the critical path first."""
    children: dict[str, list[str]] = {n: [] for n in stages}
    for n, st in stages.items():
        for d in st.deps:
            children[d].append(n)
    memo: dict[str, float] = {}

    def down(n: str) -> float:
        if n not in memo:
            memo[n] = stages[n].est_unit_min + max((down(c) for c in children[n]), default=0.0)
        return memo[n]

    return {n: down(n) + 1000.0 * stages[n].priority for n in stages}
