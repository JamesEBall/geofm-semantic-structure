# Stronger appearance controls and readout diagnostics

Author: James Edward Ball. Frozen 15 October 2026 before supplemental scoring.

Run separately on fresh EuroSAT and global data after all current analyses finish. Retain each study's immutable manifest, class order, category-held-out folds, mean-pooled encoders, three text references, geographic split gates and label limitations. Global WorldCover labels remain a weak reference. These are supplementary diagnostics planned after the existing experiment; they do not replace its frozen primary controls.

## Features

Copy the exact original 116-dimensional descriptor implementation from `geofm-semantic-structure/src/geofm_jlens/semantic_nuisance_v1.py` as `appearance116.py`. It contains six-band quantiles, means, standard deviations, fixed-bin histograms, adjacent-pixel differences and four normalised-difference ratio summaries. The source is byte-identical and hashed. Apply it to the six-band float32 64-pixel inputs used by these studies. This changes the sampling grid from the original paper's native 60-pixel 20 m inputs; it preserves the descriptor algorithm, not the original spatial support. All pixels admitted into these prepared inputs are used. Undefined ratio summaries are median-imputed using fitting rows only; a completely undefined column receives zero. Retain descriptor valid-value counts and the frozen source image hash.

Add Earth-centred unit-sphere coordinates from manifest longitude/latitude. Add sine/cosine of acquisition day-of-year only when every row has a genuine recorded datetime, as in the global manifest. EuroSAT has no acquisition dates in its manifest and receives location features only. Reject partial date availability. The three added feature routes are rich appearance, available metadata, and their concatenation; the existing 13 routes remain included.

## Readouts and nulls

For each model, evaluate geographic transfer and every existing category-held-out fold. Fit imputation/scaling on allowed training scenes only. Preserve the ridge penalties 1,10,100,1000, block-weighted validation query AP selection, all-category prototype gallery and exact tie handling of the frozen analysis.

Nearest-example transfer uses squared Euclidean distance in training-standardised feature space. Select k from 1,5,20 on the same allowed validation scenes, retaining the smaller k on ties. Average the unit language prototypes associated with the k nearest allowed training examples and score against the complete gallery through the original language-score function. Exact distance ties use stable training-manifest order. If fewer than k examples exist, use every available training example and record the requested k. No test examples enter the retrieval gallery.

Include two distinct nulls, each with fixed seeds101,202,303 shared across models/references. Within-block shuffling permutes labels among allowed fitting scenes inside each geographic block, preserving block-level class counts. It can do nothing in homogeneous blocks; report the fraction changed and never interpret an ineffective shuffle as a successful null challenge. The second null permutes the allowed category-to-target assignments consistently across all fitting blocks. This preserves visual and geographic category grouping while breaking the named target correspondence. Store the full bijection, including any fixed points. Excluded categories stay excluded. Neither null modifies validation/test labels or scenes. Select each null's ridge penalty using the same true-label validation rule; this is a validation-selected null pipeline rather than an exact permutation significance test. No p-values are assigned.

Fit the genuine and six null target matrices together to reuse one ridge factorisation per model/fold. Record all validation candidates, selected parameters, test reciprocal ranks and block identities. Primary supplemental plots compare actual ridge routes against rich appearance plus available metadata; report individual null seeds and nearest-example routes without choosing favourable results.

## Uncertainty and operation

Aggregate held-out folds equally. Use1000 shared test-block bootstrap draws with seed51426 and fixed maps/choices, invalidating any draw missing a supported category in a fold. Report conditional95% percentile intervals and valid counts. These intervals omit fitting and model-selection uncertainty and do not provide a confirmatory multiple-testing claim.

Prepare descriptors once in a one-CPU job; run one array task per16 model/feature routes, capped at two concurrent CPU tasks. Each task uses one CPU and16GB, with a four-hour limit, checkpointing completed folds. Nearest-neighbour distances use128-query chunks. No GPU is requested. A final report job depends on the entire array. All supplemental jobs begin after the fresh, output-intervention, internal-intervention and global analyses succeed.

Validate the complete cached encoder panel through original manifest, numerical parity, provenance and feature-checksum admission. Verify derived-feature checksums and preparation/source hashes. Resume only matching manifest, feature, text, analysis and protocol fingerprints. Failure preserves completed folds. Keep the current paper unchanged until the results and graphs have been reviewed.
