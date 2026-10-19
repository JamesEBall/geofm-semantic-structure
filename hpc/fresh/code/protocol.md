---
title: Frozen external geographic validation protocol
author: James Edward Ball
date: 2026-10-15
status: Freeze before model scoring
---

# External validation on EuroSAT

This run implements the external-dataset track of the larger HPC proposal. It tests whether category relationships and language-guided readouts recur in a new labelled dataset after geographic separation. The 27,000-image EuroSAT source provides ten scene categories and multispectral inputs. The final sample is the eligible subset after prior-study and split exclusions; it does not fulfil the earlier proposal for 24,000 new parent scenes with eight expert-mask categories. EuroSAT has scene labels, so this run cannot test category proportions, spatial composition or mask-derived relations. Its locations are new to the current evaluation after the declared exclusion, but checkpoint pretraining overlap remains unresolved.

## Data and freeze

Use the sorted categories AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River and SeaLake. Preserve manifest order in every feature array. Record IDs, coordinates, parent geographic blocks and split membership. Exclude previous development and main-study locations within 5 km. Resolve all 768 known earlier EuroSAT images by RGB pixel identity against the original source, then exclude those scenes and all scenes within 5 km of their source coordinates. Preserve the first admission as a superseded audit record. Assign one-degree geographic blocks with the fixed 60/20/20 train/validation/test hash rule, then exclude cross-split neighbours within 5 km, preserving test before validation and validation before train. The data-preparation implementation and manifest provide the exact hash and boundary conventions. This conservative separation concerns scene centres; it does not establish independent sampling or quantify all spatial dependence.

Freeze the manifest, source identifiers, text-reference prompts and embeddings, model checkpoints, preprocessing, feature pooling, random seeds and analysis code before inspecting test results. SHA-256 hashes accompany analysis results. Report the retained category and block counts in every split. Fail the primary run if any category is absent from train, validation or test, or if a parent block crosses splits.

## Models and references

Run Clay, Prithvi 300M and Prithvi 100M alongside architecture-matched random encoders initialised with seeds 101, 202 and 303, and a fixed appearance descriptor. Use mean pooling as the primary output. CLS pooling, when available, is a separately labelled sensitivity analysis with separate result files. These are comparisons of the complete supplied pipelines; checkpoint size is not an isolated treatment. All models use the same eligible rows and the same training and validation choices.

Keep Qwen instructed, Qwen plain and MPNet plain as distinct language references, with ten text prototypes in the declared category order. Text prototypes are unit-normalised before scoring. Report every reference, including disagreements. Ten categories generate 45 related distances, not 45 independent experimental units.

## Category structure

Use only test scenes. Treat each image as a parent scene, average its raw features within each category and geographic block, unit-normalise each block mean, then average the normalised block means with equal block weight within the category. This preserves the original study's parent/component aggregation order; ridge standardisation applies only to the separate readout. Unit-normalise the resulting category means and compute their pairwise cosine distances. Report Spearman correlation between the 45 image distances and the corresponding text-prototype distances. Do not attach conventional independent-pair p-values. This measures category organisation within the tested labels and geographic support.

## Transfer

Standardise feature columns with means and standard deviations fitted on training rows only; constant columns retain scale one. Fit a ridge map with an unpenalised intercept from image features to normalised language prototypes. Candidate penalties are 1, 10, 100 and 1000, selected by validation geographic-block-weighted per-query average precision, with the smaller penalty winning an exact tie. Spectral factorisation reuses the ridge solution across references by fitting class indicators and multiplying by the text prototypes, which is algebraically equivalent to fitting each prototype target directly. Choose the primal or dual system according to feature and sample count.

For geographic transfer, fit all ten categories on training blocks and score all ten on test blocks. For joint category and geographic transfer, use five fixed excluded pairs: AnnualCrop/Forest, HerbaceousVegetation/Highway, Industrial/Pasture, PermanentCrop/Residential and River/SeaLake. Exclude both categories from training and validation in each fold. Select the penalty using the remaining eight validation categories. Score test images from the excluded pair against a gallery containing all ten prototypes, including the excluded pair.

The primary endpoint is per-query average precision over the complete ten-category gallery. With one positive label per scene this equals reciprocal rank. Exact score ties use the expected reciprocal rank across all positions occupied by the tied group. Average the scene scores within each geographic block and then average blocks equally. For joint transfer calculate this block-weighted mean separately within each held-out pair and average the five folds equally. Validation uses the same block-weighted per-query endpoint and the full ten-category gallery for both known and held-out fits; excluded categories have no fitting or validation images, but their prototypes remain in the gallery. Thus all comparisons retain the same retrieval operation as the original study, while the candidate categories and reference data differ.

Report sample-weighted ten-way top-1 accuracy and class-wise macro-average precision as secondary diagnostics. Known-category macro-AP averages all ten classes. Each held-out fold averages AP over its two supported categories, with negatives drawn from the other category in that fold; the joint summary averages five folds equally. This secondary AP differs from the primary per-query endpoint and cannot be compared directly across known and held-out tracks. Record every penalty candidate and its validation score, together with the selected penalty and test support.

## Conditional uncertainty and reporting

Use 1,000 paired bootstrap draws of the test geographic blocks with seed 51426, sharing each draw across models and references. Keep the fitted maps, scaling and validation selections fixed. Report the 2.5th and 97.5th percentiles of each model's primary block-weighted per-query AP difference from appearance, separately for known and held-out transfer. A resample is invalid if a supported category is absent; report valid and invalid counts. For joint transfer every fold must remain valid. These ranges describe conditional resampling sensitivity, and omit uncertainty from map fitting, model selection, text-reference construction and dataset selection. They do not by themselves support a population-level significance claim.

Produce category-geometry graphs and model-minus-appearance transfer graphs, with all references shown together. Report matched random encoders alongside trained models rather than selecting favourable seeds. Preserve source scores and the complete metadata. Review results and support before deciding whether they strengthen the existing paper. The paper remains unchanged during this run. Representation interventions, nuisance residualisation, larger model admission, temporal shifts and mask-based composition remain separate work from this external validation.

## HPC operation and checks

Extract frozen features in batched GPU jobs and analyse cached features on CPU. Cap each BLAS/OpenMP process at one thread to avoid node oversubscription. The analysis reuses one ridge factorisation per model and fold across language references and penalties; per-query reciprocal ranks and block means are computed once, then bootstrap draws reweight the fixed block means. Numerical tests compare primal and dual ridge solutions with an independent direct solver, verify prototype-factorisation equivalence, check split isolation and compare tied reciprocal ranks with exhaustive permutation enumeration and paired bootstrap estimates with direct block means. All twelve encoder routes must have completion markers, matching manifest provenance, passed numerical parity and matching feature checksums before analysis starts. The thirteenth route is appearance. Diagnostic extraction directories are excluded. Missing planned routes stop the analysis.
