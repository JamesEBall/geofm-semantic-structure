# MegaEarth reproduction

Author: James Edward Ball

`code/` contains data preparation, encoder extraction and statistical analysis. `config.json` fixes the scientific settings. The analysis applies the retained-window mask and excludes all observed siblings of each held-out family, including siblings outside the admitted retrieval gallery. The current code includes both corrections.

## Check the published data and results

From the repository root:

```sh
python -m pip install -e . pandas pyarrow
python scripts/download_megaearth.py --output work-megaearth
python scripts/plot_megaearth_support.py --tables work-megaearth/tables
python scripts/plot_megaearth_transfer.py
```

The first command fetches the published tables at a fixed revision and checks every file hash. The support plot verifies the retained split counts, all 188 classes, 91 families and 8,755 test pairs. The transfer plot reads the corrected estimates and paired intervals without fitting models. Outputs go to `output/figures/`.

`results/retained-20261018/results.json` holds the corrected estimates; `qualify/` holds their statistical qualification. Earlier estimates remain under `results/full-20261018/` solely for the before/after comparison. The frozen source files are unchanged. A completed computation does not make a failed statistical gate pass.

## Rebuild features and analyses

Use Linux and Python 3.12. `env/lock.txt` records the complete study environment, including PyTorch 2.6 and CUDA 12.4. Install the CUDA build of PyTorch before the remaining dependencies. The source-preparation stages also require `osmium-tool`; its environment is specified in `env/osm-lock.txt`. Model extraction needs CUDA; the largest inference job used eight CPUs and 160 GiB RAM.

The full pipeline can be inspected and run through the stage CLI:

```sh
export PYTHONPATH="$PWD/hpc/vocab/code:$PWD/src"
export MEGAEARTH_ROOT=/path/to/scratch/megaearth
export MEGAEARTH_MODEL_ROOT=/path/to/fresh-study
python -m mc.cli --profile full dag
python -m mc.cli --profile full local --threads 8
```

`MEGAEARTH_MODEL_ROOT` points to the fresh-study model staging tree built with the [HPC guide](../README.md). The full pipeline reconstructs source acquisition and requires the frozen scientific gates to be reviewed. It stops at each unapproved gate. Dataset archives alone are not a completed feature cache. Their tables and image bytes can be checked with the download command above; full reconstruction also needs model weights and source preparation.

For Slurm, `slurm/` contains submission scripts and `config.d/mc.json` contains site resource settings. Set the account, partitions, paths and resource caps for your cluster before submitting. Preserve the scientific configuration and use a new output directory. The scripts keep immutable file manifests and reject conflicting or incomplete results.

The historical corrected-run entry point is `retained-cohort-correction-20261018/run.py`. It takes a complete frozen source run with F1/F2/F3 manifests, checks its source hashes, and creates a separate retained-cohort run. Its controller prints a plan by default; `--submit` enables Slurm submission. The operational overlay records the batching used in the completed study. These paths support exact reruns from the original frozen run, rather than treating the public imagery export as that run.

## Calibration and tests

The `supplement-geometry-*` directories contain calibration implementations, fixed simulation designs and expected summaries. The Markdown protocol inputs in these directories are hashed by the programs and form part of their numerical provenance.

```sh
python scripts/test_all.py --megaearth
```

This runs unit and synthetic integration checks without submitting cluster work. Repeating the geometry confirmation requires its full fixed simulation design; retain missing-class failures and the original thresholds. The complete confirmation records are in the `geometry-validation-v1` release. Full transfer calibration and population geometry intervals remain unqualified.
