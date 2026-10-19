"""~🌍 MegaEarth 🛰️~ S11 analysis engine (pre-registered statistics; numpy/scipy only).

Modules
-------
io        logging, deterministic npz/json writers, fingerprints, freeze verification, parameters
study     windows/classes/folds/prototypes loading, label CSR, pair tables, split isolation
folds     outer/inner/scaling fold construction (deterministic, stratified by top_key)
metrics   tie-aware reciprocal rank / top-k / chance, filtered ranking, multi-label metrics,
          cell->class->study aggregation, GZSL gamma sweeps
ridge     exact ridge from float64 sufficient statistics (one eigh per fit, all alphas)
readout   per-route fitting/scoring engine (alpha selection, ZSL/GZSL, nulls, scaling, probe)
bootstrap two-way (cell x family) bootstrap draws, aggregation, null-centred p, Holm, IUT
infer     estimands/contrasts, H1, strong-tier criteria, strata, verdict inputs
rsa       RSA, Mantel (unrestricted + family-block), partial RSA, bootstrap, qualification
power     variance components from val and IUT+Holm power simulation
verdict   table-driven verdict tiers
report    verdict.json / report.md / summary-small.json
"""

ESTIMAND = ("Estimand: frozen-feature transfer through a supervised linear text readout beyond the "
            "specified controls — not semantic understanding or causal acquisition from imagery alone.")
CONDITIONAL_LABEL = "conditional on fitted weights, tuning, vocabulary and seed panel"
ENGINE_VERSION = "s11-a1.0"
