# Results and interpretation

Author: James Edward Ball

## Central finding

The category supervision used to fit a language readout changes what a GeoFM benchmark reveals about a fixed representation. Broader fitting vocabularies improve held-out-family retrieval at fixed image counts, including under matched recorded-continent and biome composition and one positive category embedding per fitting image. Random encoders and appearance-and-metadata controls also improve.

At 512 fitting images, expanding from 12 to all eligible fitting families raises mean reciprocal rank by 0.0371 for Clay, 0.0318 for Prithvi 300M and 0.0347 for Prithvi 100M. The pretrained models retain their mean ranking under matched vocabularies. Deliberately unequal vocabularies reverse 23 of 72 primary pairwise comparisons at the same image budget. Category identity, frequency and local geography remain coupled to vocabulary selection, so this measures readout-protocol sensitivity.

The completed-extension release stores the underlying score arrays and replays 720 endpoints, 180 contrasts, 34,020 ranking rows and 540 influence rows. See [reproduction instructions](REPRODUCIBILITY.md).

## Paired image-only training

Six random Prithvi 100M initialisations each receive 100 epochs of masked-image reconstruction on the same 4,012 images. Both initial and trained branches are frozen before evaluation. The training objective uses imagery alone; OSM-informed acquisition and evaluation categories define the corpus and category comparisons.

| Endpoint | Initial mean | Trained mean | Difference | Positive runs |
| --- | ---: | ---: | ---: | ---: |
| Native category correlation, instructed Qwen | 0.26672 | 0.27998 | +0.01326 | 6/6 |
| Native category correlation, plain Qwen | 0.20598 | 0.19740 | -0.00858 | 1/6 |
| Native category correlation, MPNet | 0.25287 | 0.25686 | +0.00399 | 5/6 |
| Primary supervised retrieval MRR | 0.17550 | 0.17504 | -0.00046 | 2/6 |

Native geometry uses raw feature prototypes in this table. Feature scaling and adjusted analyses are retained as sensitivities. The primary transfer condition uses single-positive targets, 512 fitting images, all eligible families and instructed Qwen. The other 47 condition means increase by 0.00034 to 0.00436. The primary result therefore does not imply that training never helps transfer.

These paired runs identify an effect of the specified image-only training procedure within one architecture and corpus. They do not identify the causes of the released Clay and Prithvi checkpoints' representations. The six seeds are the replication units; readout repetitions are nested within them. The cohort is development-informed and provides no independent geographic confirmation.

[Machine-readable tables and verification receipts](../results/controlled-pretraining/) preserve all references and conditions. The full collection includes frozen execution code, training summaries, saved scores and feature/prototype arrays. Replaying scores verifies numerical aggregation; it does not repeat GPU training.

## Scope and validation

The main experiment contains 34,992 retained windows and 188 admitted categories. Its fixed-count test set has 8,755 image-category pairs from 5,942 unique windows. The failed transfer-uncertainty and geometry-interval qualification gates remain part of the record. Point estimates describe these evaluated data; population inference remains unresolved.

Visual review covered 376 pairs from 370 windows. Source reconstruction traced 1,915 OSM objects and reproduced stored polygon coverage within 2.93e-8. The model-assisted follow-up used the completed human ratings. These checks support provenance and sample interpretation. They do not estimate population label accuracy. All original labels and ratings are retained.

Archived full-cohort estimates and sampling warnings remain under `hpc/vocab/results/`. They answer a different question from the fixed-count intervention and must not be substituted for its endpoints.
