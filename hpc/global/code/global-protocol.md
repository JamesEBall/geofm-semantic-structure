# Global weak-reference geographic validation: extraction and readout

Author: James Edward Ball

The class order is Water, Trees, Grass, Flooded_Vegetation, Crops, Scrub, Built_Area, Bare_Ground. Each scene has one dominant category assigned from automated ESA WorldCover 2021 v200 by the separately frozen acquisition and aggregation protocol. These are weak reference labels, not independent expert observations. Disagreement can reflect reference error and temporal mismatch as well as representation differences. Cross-split geographic separation must pass the source gate and a minimum 5 km centroid-distance check before analysis. This is geographic separation within the acquired sample; encoder pretraining overlap remains unknown.

Extract mean patch and CLS features from trained Clay, Prithvi 300M and Prithvi 100M, and the same architectures initialised with seeds 101, 202 and 303. Use the pinned fresh-study adapters, checkpoints, normalisation and deterministic float32 math attention. Each route passes single-image versus batched replay and produces checksummed resumable feature chunks. Batch sizes come from the successful fresh-study A100 smoke gate. Two GPU tasks can run concurrently.

Mean pooling is the primary analysis. Compare category-mean cosine-distance ordering against the existing eight-class Qwen instructed, Qwen plain and MPNet references. Each geographic block has equal weight within a category mean. Compare trained and random features with the fixed appearance baseline.

Fit ridge mappings using training data only, with alpha in {1, 10, 100, 1000} selected on validation query AP. Keep fitting, validation and test blocks separate. Evaluate known-category geographic transfer and four held-out category pairs: (Water,Trees), (Grass,Flooded_Vegetation), (Crops,Scrub), (Built_Area,Bare_Ground). Held-out classes are excluded from fitting and validation; all eight text candidates remain available at test time. Each held-out pair contributes equally to the final transfer summary.

Report block-averaged per-query AP, macro category AP and top-1 accuracy. Use 1,000 shared geographic-block bootstrap draws for paired query-AP differences from appearance. These intervals are conditional on fixed fitted maps and validation choices; unsupported resamples are counted explicitly. Do not interpret the descriptive category correlations as population intervals or causal mechanisms. Results remain separate from the paper pending review.
