# Internal features and directional interventions

Author: James Edward Ball

Exploratory follow-up on the frozen fresh-data study; the original split, source images and text references remain unchanged. Run only after the fresh analysis job succeeds.

The deterministic 4,000-image maximum subset balances split/category cells and cycles across available geographic blocks, using metadata alone. All models share this selection. One forward pass extracts post-residual mean patch features at 25%, 50%, 75% and 100% depth. Intermediate features are before the final layer normalisation, so the final layer geometry is distinct from the normalised clean-output geometry.

A midpoint direction for each category is estimated from training-only block-averaged category versus remaining-image means. The direction is removed from patch tokens at doses 0.5 and 1, leaving CLS tokens unchanged. Three random orthogonal directions use the same per-token perturbation magnitude as each target direction. A norm-only control scales each original patch token to the target-perturbed norm while preserving its direction. Directions are not selected using test effects.

Interventions use at most 300 test images selected by the same metadata-only rule. All 100 conditions per model share these images. A ridge readout uses clean training features, with its alpha selected on clean validation images for each language reference. The unchanged readout measures target-category and other-category query AP under intervention. These are exploratory paired effects; the first report includes no population confidence intervals or multiplicity-adjusted significance claims. A later block-bootstrap and category-specific control comparison is required before manuscript claims.

Clay's wrapper reproduces the complete attention-plus-feedforward residual recurrence; hooking only the next attention input would change only one branch and is deliberately avoided. Prithvi hooks operate on complete block outputs. An original-versus-hooked gate and exact zero-dose replay run before scored extraction. Float32, deterministic algorithms and math attention match the fresh extractor. Helpers test energy matching, CLS preservation, selection balance and Clay recurrence parity.

Submit `encode.sbatch` with `--dependency=afterok:<fresh-analysis-job-id>` and exported absolute `FRESH_ROOT` and `INTERNAL_ROOT`; submit `analyse.sbatch` with `--dependency=afterok:<array-job-id>`. CPU analysis uses the ENUCC short partition. Extraction uses one GPU per model with array concurrency one, four CPU cores, 32 GB RAM and one hour. Cached layer features and all intervention outputs support CPU replay. The completed-output checksum manifest prevents silent mixed results; incomplete runs recompute clean layer features and resume completed conditions only after fingerprint, checksum and shape validation. Clean features must match the completed fresh extraction within 1e-4 before any scored perturbation.

Outputs: per-model selection, intermediate and final features, direction vectors, condition arrays, checksum manifest, combined `report.json`, `layer-geometry.png` and `interventions.png`. Selective damage relative to random controls would support a local functional role for the tested direction; it does not identify a unique circuit or general semantic capability.
