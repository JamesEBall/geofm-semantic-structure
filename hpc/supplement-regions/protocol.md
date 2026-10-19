---
title: Leave-one-geographic-group-out supplement
author: James Edward Ball
date: 2026-10-15
status: Frozen secondary protocol before scoring
---

# Question and scope

Does category-to-language transfer persist when an entire geographic group is excluded from fitting and validation? This secondary analysis reuses the global study's admitted image features, automated WorldCover references and frozen language prototypes. It creates no new independent data and does not replace the original evaluation.

The six groups retain the Natural Earth country assignment: Africa, Asia, Europe, North America, South America and Oceania. These are geographic groups rather than strict continent boundaries; a country such as Russia can be assigned to Europe in its entirety. Results concern these fixed groups and the sampled blocks.

# Frozen operations

For each group, test on every admitted scene in that group, including scenes assigned to fitting, validation or testing in the original global protocol. Use only original fitting rows from the other five groups for fitting, and only their original validation rows for penalty selection. Original test rows from the other groups remain unused. A held-out group contributes no fitting, standardisation or validation data. Each scene and geographic block appears in one role within a run. The six runs share data across their different roles, so their results are dependent and cannot be treated as six independent replications.

Evaluate appearance, Clay, Prithvi 300M, Prithvi 100M and the three fixed random initialisations per architecture. Use all eligible fitting rows and mean-pooled features. Retain Qwen instructed, Qwen plain and MPNet plain. Their prototypes, class order and calculations come from the frozen global analysis, copied into this package and hashed.

Run both eight-category geographic transfer and four joint category/geographic transfer folds: Water/Trees, Grass/Flooded Vegetation, Crops/Scrub and Built Area/Bare Ground. A joint fold excludes its two categories from fitting and validation and evaluates those categories in the held-out group against the complete eight-category gallery. Standardisation and the ridge intercept use fitting rows only. Penalties are 1, 10, 100 and 1000, selected by validation block-weighted query AP; a tied maximum selects the smaller penalty.

Each required fitting or validation class needs at least two scenes from two blocks. Every evaluated class must occur in the test group. A failed support condition produces an explicit unsupported result with class counts and reasons. Known-category transfer requires all eight test classes. A joint summary requires all four folds; missing folds never disappear from its denominator. Report each group separately without selecting favourable groups or averaging only successful results.

The endpoint is query AP, equal to reciprocal rank for a single positive category, with expected reciprocal rank for exact score ties. Average scenes within each test block and blocks equally. The joint summary averages all four fold values equally. Report gains over the appearance baseline for every language reference and every model, including all random controls.

For each group, use 1,000 paired bootstrap draws over its test blocks with seed 51426. Within a joint draw, reuse block multiplicities across all four folds; any absent fold makes that draw invalid. Report valid and invalid draw counts. Suppress intervals with fewer than three distinct blocks. These are conditional sensitivity ranges with the fitting, penalty choice, reference construction and sampled dataset held fixed. They do not represent global population uncertainty or support causal conclusions.

# Execution and provenance

Run 78 CPU tasks, one per model and geographic group, with at most three concurrent tasks and one BLAS thread per task. Cache each completed task atomically. A resumed task verifies the source, input and result-cache fingerprints. No GPU allocation or new model extraction is required.

Before running, verify the global cohort freeze, its admission report, original manifest, appearance cache, text prototypes and text metadata. For each encoder, verify its completion marker, parity result, manifest fingerprint and feature checksum. Raw image files are unused and are not repeatedly hashed by this CPU supplement. The final collector requires all 78 tasks, retains unsupported outcomes, checks shared cohort provenance, and produces full JSON results and compact graphs for the three trained models; all random-control results remain in the result files.

Submit the task array only after successful completion of all current analyses: successful fresh, output-intervention, internal-intervention and global analysis jobs. Submit `report.sbatch` with `afterok` on the returned task-array job ID. This package submits no jobs by itself and changes no paper or existing experiment.
