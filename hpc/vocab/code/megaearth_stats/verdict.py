"""Table-driven study verdict.

A tier table (config analysis.tiers, else io.DEFAULT_TIERS) is evaluated top to bottom; the first tier whose
rule holds is the verdict; its first matching sublabel (if any) is appended. Rule grammar:

  {"always": true}
  {"study": "<gate>"}                         study-level gate is True
  {"count_archs": {"min": n | "all", "gates": [g, ...]}}
                                              >= n architectures (or all) have every listed gate True
  {"all": [rule, ...]} / {"any": [rule, ...]} / {"not": rule}

Gate values are True / False / None; None means UNAVAILABLE and never satisfies a gate (so e.g. an
unavailable RSA criterion blocks STRONG YES). A gate name "not_<g>" is True iff <g> is exactly False.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def gate_value(gates: Mapping[str, Any], name: str):
    if name in gates:
        v = gates[name]
        return None if v is None else bool(v)
    if name.startswith("not_"):
        v = gates.get(name[4:])
        return None if v is None else (not bool(v))
    return None


def _all_true(gates: Mapping[str, Any], names: Sequence[str]) -> bool:
    return all(gate_value(gates, g) is True for g in names)


def eval_rule(rule: Mapping[str, Any], arch_gates: Mapping[str, Mapping[str, Any]],
              study_gates: Mapping[str, Any]) -> bool:
    if "always" in rule:
        return bool(rule["always"])
    if "study" in rule:
        return gate_value(study_gates, rule["study"]) is True
    if "count_archs" in rule:
        spec = rule["count_archs"]
        n_ok = sum(1 for g in arch_gates.values() if _all_true(g, spec["gates"]))
        mn = spec["min"]
        if mn == "all":
            return len(arch_gates) > 0 and n_ok == len(arch_gates)
        return n_ok >= int(mn)
    if "all" in rule:
        return all(eval_rule(r, arch_gates, study_gates) for r in rule["all"])
    if "any" in rule:
        return any(eval_rule(r, arch_gates, study_gates) for r in rule["any"])
    if "not" in rule:
        return not eval_rule(rule["not"], arch_gates, study_gates)
    raise ValueError(f"unknown verdict rule {rule!r}")


def evaluate(tiers: Sequence[Mapping[str, Any]], arch_gates: Mapping[str, Mapping[str, Any]],
             study_gates: Mapping[str, Any]) -> dict:
    trace = []
    for t in tiers:
        ok = eval_rule(t["rule"], arch_gates, study_gates)
        trace.append({"tier": t["name"], "holds": ok})
        if ok:
            sub = None
            for s in t.get("sublabels", []):
                if eval_rule(s["rule"], arch_gates, study_gates):
                    sub = s["name"]
                    break
            label = t["name"] + (f" — {sub}" if sub else "")
            return {"tier": t["name"], "sublabel": sub, "label": label, "trace": trace}
    return {"tier": None, "sublabel": None, "label": "UNDETERMINED", "trace": trace}


def unavailable_gates(arch_gates: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> dict:
    return {a: [g for g in names if gate_value(gs, g) is None] for a, gs in arch_gates.items()}
