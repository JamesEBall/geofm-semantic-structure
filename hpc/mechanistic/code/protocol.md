---
title: Output-feature intervention protocol
author: James Edward Ball
date: 2026-10-15
status: Frozen before fresh-data test results
---

# Which feature directions support category readout?

This exploratory extension tests selective dependence of the fitted geographic-transfer readout on training-defined class-contrast directions. It uses the existing fresh EuroSAT manifest, labels, all three text references and primary mean-pooled feature caches. It requires the complete twelve-encoder extraction panel and its manifest, parity and checksum gates before starting. The evaluated panel is appearance, trained Clay, Prithvi 300M and Prithvi 100M, and the first predetermined random encoder seed (101) for each architecture. This resource-bounded diagnostic does not select random models by performance.

Standardise features on training data only. Select the baseline ridge penalty from 1, 10, 100 and 1000 using validation block-weighted query AP and the existing tie rule. Fit all ten categories, preserving the original ten-prototype gallery. This experiment concerns geographic transfer of known categories; it makes no claim about category-held-out transfer. For each category, form a unit direction from the difference between its training feature mean and the mean of all other training scenes. The direction is supervised and rank one. Its removal is expected to damage a supervised readout; the falsifiable question is whether this damage exceeds controlled perturbations and is concentrated in the target category.

For each category draw five isotropic rank-one random directions using fixed seeds 1103, 2207, 3301, 4409 and 5501, offset by 10000 times the category index. Random directions may overlap the target direction; removing them from its complement would make the control artificially weak. Keep all seeds and average their per-scene contrasts before block resampling.

First remove the entire target direction and compare it with full random rank-one removals. Second match the expected squared removed feature energy on training scenes: reduce the stronger-energy intervention until it equals the weaker one. Both strengths stay within [0,1]. Repeat these matched pairs at predetermined strengths 0.25, 0.5 and 1.0. The target removal amplitude can therefore differ across random controls; record every amplitude and training/test removed energy. Matching is performed on training scenes and need not hold exactly on test scenes. Report full rank-matched results alongside energy-matched results so weak matched doses remain visible.

The principal diagnostic is the seed-averaged change in target-category block-weighted query AP under targeted versus energy-matched random removal at strength 1. Report the same comparison on all other categories, and the 0.25 and 0.5 doses. A target loss together with widespread non-target loss is not evidence of selective dependence. Full target removal versus the intact baseline is a separate descriptive check.

Keep the baseline map fixed for the main intervention. As a sensitivity check, refit the ridge map after full target or random projection, with the same baseline-selected penalty and training mean. Compute this exactly as a rank-one constrained ridge solution using the existing spectral factorisation. Refit results test whether remaining features can recover the readout; they are not an energy-matched intervention. The sham is algebraically the baseline; numerical tests verify zero-strength identity and direct-solver equivalence.

Use 1000 geographic block draws with seed 51426, common to every contrast. Compute target and other-category block means separately; exclude draws with no supporting block and report their count. Intervals describe conditional fixed-map test-block sensitivity. They do not include direction fitting, validation selection or population sampling uncertainty. This is an exploratory many-comparison panel with no confirmatory significance declaration or cherry-picked category/reference claim. Save per-scene reciprocal ranks for every condition and baseline in compressed model-specific files so alternative summaries can be independently checked.

The intervention acts on cached final image features. It establishes dependence of this fitted readout on a specified output subspace, rather than identifying a causal mechanism inside the encoder. Layer-wise extraction and activation exchange require new GPU adapters, downstream replay parity, layer selection using development data, matched scene-pair rules and separate freezes. They are not silently represented by this CPU experiment. No manuscript changes follow automatically from completion.

# Runtime and audit

One CPU core, 16 GB RAM and a two-hour limit are sufficient as an initial allocation; the job uses cached features and no GPUs. BLAS threads are capped at one. Reuse one spectral decomposition per model, all penalties across references and rank-one inverse updates for refits. Publish hashes of the manifest, text embeddings, extraction features, analysis scripts and protocol. Write partial results atomically after each model, and a completion marker only after the full requested panel and plot succeed. The scene-score cache retains manifest test ID order. Run the numerical and synthetic tests before submission.
