# Reproduction guide

Author: James Edward Ball

Choose the level that matches the question. Numerical replay uses saved observations and verifies calculations. Full reconstruction also repeats acquisition, feature extraction and fitting. Neither operation changes the status of the failed population-calibration checks.

## 1. Inspect the results

The [result summary](RESULTS.md) explains the endpoints. Small CSV tables and recorded verification receipts are in [`results/controlled-pretraining`](../results/controlled-pretraining/). `manifest.json` records their hashes and links them to the complete collection. The original archived summaries are unchanged; fresh local verification receipts record subsequent successful checks.

## 2. Replay saved results on a CPU

Use Python 3.12. Install dependencies in a clean environment:

```sh
python -m venv .venv-replay
. .venv-replay/bin/activate
python -m pip install numpy==2.5.3 scipy==1.18.1
```

### Vocabulary and label-sensitivity analyses

Download `geofm-completed-extensions-20261021.tar.gz` from the [completed-extension release](https://github.com/JamesEBall/geofm-semantic-structure/releases/tag/completed-extensions-20261021-v1). The archive is about 109 MB.

```sh
gh release download completed-extensions-20261021-v1 --repo JamesEBall/geofm-semantic-structure --pattern geofm-completed-extensions-20261021.tar.gz
tar -xzf geofm-completed-extensions-20261021.tar.gz
python geofm-completed-extensions/replay.py --report vocabulary-verification.json
```

Expected archive SHA-256: `e8a5ee53dfcaac7dc2c6c63e3d4b23712a156bf01684ed860556893140ff2b2a`.

The replay checks 720 endpoint estimates, 180 contrasts, 34,020 ranking rows and 540 influence rows.

### Complete paired image-only training analysis

Download `complete-results.tar` from the [controlled-pretraining release](https://github.com/JamesEBall/geofm-semantic-structure/releases/tag/controlled-pretraining-20261021-v1). The archive is 1,836,236,800 bytes. Allow at least 5 GB of free disk space and 8 GB of RAM for download, extraction and replay; these are practical allowances, not measured minimum requirements.

```sh
gh release download controlled-pretraining-20261021-v1 --repo JamesEBall/geofm-semantic-structure --pattern complete-results.tar
python scripts/replay/complete_experiment.py complete-results.tar --output work-complete-replay
```

The output directory must be new. The command verifies the archive hash and every member, reaggregates 50,428,800 saved query scores, then independently reconstructs native category prototypes and their geometry. It preserves the extracted source records and writes fresh verification receipts separately. A failed check exits with an error.

Expected archive SHA-256: `d7651dda035cc23d5b79477456dab96011ff73e1c9e63e33454ff90d076b6816`.

The archive contains six completed training summaries, frozen execution scripts and configurations, feature arrays, score arrays, native prototypes, reference inputs and analysis tables. It does not bundle the training imagery or final model checkpoint weights. The public MegaEarth dataset supplies the imagery; upstream models retain their own distribution terms.

## 3. Rebuild the experiments

Follow the [HPC guide](../hpc/README.md) and [MegaEarth guide](../hpc/vocab/README.md) for source acquisition, canonical input construction, model staging and Slurm execution. [DATA.md](../DATA.md) identifies pinned downloads. CUDA extraction/training uses the separately recorded GPU environment; keep it separate from the CPU test environment.

For the controlled training experiment, the complete archive's `controlled-pretraining-20261021-v1/PROTOCOL.md`, freeze files and execution scripts are the record of the run. It uses six seeds, 4,012 training windows, 100 epochs, batch size eight and 50,200 optimiser steps per seed. It was executed on ENUCC. The source records retain cluster paths for provenance; rebuilding on another system requires mapping those paths to the same verified inputs. Full end-to-end portability to a new cluster has not been tested.

## Code checks

From the repository root, use a separate Python 3.12 environment:

```sh
python -m pip install -e '.[test,hpc-test]'
python scripts/test_all.py --megaearth
python hpc/check_global_results.py
```

The code tests check behaviour on controlled inputs. The global-results audit reconciles saved fold-score records. Numerical checks do not establish semantic truth, independent geographic confirmation or a qualified population confidence interval.
