# Reproduce the HPC extension

James Edward Ball

These experiments extend the paper with fresh EuroSAT scenes, a global Sentinel-2/WorldCover cohort and internal interventions. The global experiment and its appearance, metadata and shuffled-label controls are complete. The [archived global results](reference-results/global/README.md) include the scores used in the paper and commands to audit them and redraw its figures. Use the top-level `reproduce.sh` for the original expert-labelled study.

## Set up

Use Linux, Python 3.12 and Slurm. The supplied jobs target ENUCC A100 80 GB GPUs, the `gpu` and `short` partitions and the `default` account. Check these resource settings against your cluster before submitting. Each encoder uses one GPU; the main arrays allow two concurrent tasks and the internal array allows one. CPU array limits apply to each array separately.

From the repository root, choose a new directory on scratch storage:

```sh
export RUN="$SCRATCH/geofm-reproduction"
python3 hpc/run.py init --root "$RUN"
python3.12 -m venv "$RUN/fresh-01/.venv"
"$RUN/fresh-01/.venv/bin/pip" install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$RUN/fresh-01/.venv/bin/pip" install -r hpc/environment-enucc.txt
```

`environment-enucc.txt` records the Linux environment used for this extension. Keep it separate from the original paper's environment. The runtime links to this checkout for model definitions; keep the checkout available at the same location. Do not install the top-level package dependencies into this environment, as they describe the original study environment.

Stage downloads on a network-enabled host. The pinned model downloader checks every weight file against its SHA256:

```sh
"$RUN/fresh-01/.venv/bin/python" "$RUN/fresh-01/code/stage_weights.py"
curl -fL --retry 3 https://zenodo.org/records/7711810/files/EuroSAT_MS.zip -o "$RUN/fresh-01/data/EuroSAT_MS.zip"
```

The preparation job checks the archive MD5. Fixed language embeddings, prior-scene exclusion records and global block locations are included. Model weights and source imagery remain downloads. Their source licences apply; see the repository's `NOTICE.md` and `global/source-selection.md`.

## Run the fresh-data experiment

```sh
python3 hpc/run.py submit --root "$RUN" --stage fresh --dry-run
python3 hpc/run.py submit --root "$RUN" --stage fresh
```

This submits CPU preparation, a CPU/CUDA parity and batch-size gate, 12 encoders, transfer analysis and both intervention studies. Successful dependencies are required at each stage. Preparation resolves prior image identities and enforces geographic separation before freezing the inputs and code. The expected admitted manifest is in `reference/fresh-manifest.json`; the reference cohort contains 18,124 scenes.

Job IDs are saved immediately in `submitted-fresh.json`. A second submission is refused when this file exists. After a partial submission failure, inspect the receipt and Slurm state before submitting missing jobs manually. Keep failed evidence and the original receipt. Jobs with failed prerequisites remain blocked.

## Acquire and run the global experiment

On a workstation with network access, initialise a separate run directory with `hpc/run.py init`. Use a Python environment containing NumPy, SciPy, Rasterio, PyProj and Shapely (versions in the environment file), then run:

```sh
python hpc/acquire_global.py --root /path/to/local-run --workers 4
```

This runs the 24-window pilot and then attempts 200 candidates in each of the 158 fixed blocks (31,600 attempts). Restarting uses the acquisition code's validated cache. Review any failed block log before retrying. Downloading time depends on the public catalogue and object store.

The initial 120 blocks failed category-support admission. Before any global model scoring, the fixed expansion added 38 separated blocks using reference maps alone, retaining the original blocks and every quality threshold. The resulting split has 60 training, 24 validation and 74 test blocks. The original failed report and inventory are in `global/amendments/2026-10-15-support-augmentation/`; the selection rule and evidence are in `global/support/`. Replication uses the included final inventory directly. The expansion changes sampling proportions and does not estimate global class prevalence.

The expanded cohort passed all admission checks with 29,733 images from 31,600 attempts, 1,867 quality exclusions and zero unresolved technical failures. `reference/global-manifest.json.gz` contains every retained scene, source URL, class, split and image/mask checksum, plus exclusions. The admission report and cohort checksums are alongside it. Completed estimates, per-scene retrieval scores and their audit are in `reference-results/global/`.

On Slurm nodes with working HTTPS access, run the pilot first, then submit `global-01/acquire.sbatch` from that directory with `sbatch --array=0-157%32 acquire.sbatch`. This overrides the historical array range and uses up to 32 CPU workers. A network-enabled workstation can instead use `--workers 32`. Start with fewer workers to check the available network capacity. Use your cluster's approved proxy where required; this repository contains no proxy credentials or site-specific relay configuration.

Transfer the resulting data and pilot report to the HPC runtime, using your own SSH destination:

```sh
rsync -a /path/to/local-run/global-01/data/ USER@HOST:/path/to/hpc-run/global-01/data/
rsync -a /path/to/local-run/global-01/results/pilot-report.json USER@HOST:/path/to/hpc-run/global-01/results/
```

On the HPC:

```sh
python3 hpc/run.py submit --root "$RUN" --stage global --dry-run
python3 hpc/run.py submit --root "$RUN" --stage global
```

The admission job verifies the transferred pilot and all block reports. It then checks image hashes, quality, geographic separation and category support before permitting the encoder array. The global encoders also require the fresh CUDA gate. A failed admission gate stops this experiment; do not lower thresholds to obtain results.

The final sampling uses a 900 m reference-map neighbourhood to screen candidate footprints, then retains the original full-resolution quality gates. Up to 24 eligible dates are ranked by cloud plus snow percentage. Unsupported JP2 catalogue assets are recorded and omitted. The fixed inventory includes these sampling amendments. Reuse it to replicate this study; regenerating the inventory defines a new cohort. Public catalogues can change, so compare new scene identities and pixel hashes with the supplied manifest before claiming an exact replay. The support-selection scripts record the original expansion procedure and are unnecessary for direct acquisition of the frozen 158-block inventory.

## Run the supplements

After submitting both main chains:

```sh
python3 hpc/run.py submit --root "$RUN" --stage supplements --dry-run
python3 hpc/run.py submit --root "$RUN" --stage supplements
```

The helper queues richer appearance and geographic controls, nearest-neighbour comparisons, shuffled references, training-block fit stability, intervention uncertainty and geographic-group transfer. All supplements wait for both main analyses and the two intervention analyses. There are no dependencies on other research projects. Inspect `submitted-*.json` with `squeue` or `sacct`.

## Inspect and test

Fresh and global results are written to `fresh-01/results/mean/` and `global-01/results/mean/`. Intervention and supplement outputs are in each sibling directory's `results/`. Reports include JSON and PNG figures. Review admission reports and completion checks before interpreting results.

To inspect the completed global experiment without downloading imagery or using a GPU, install NumPy and Matplotlib, then run from the repository root:

```sh
python3 hpc/check_global_results.py
python3 hpc/plot_global_results.py --out work/global-figures
```

The audit recomputes all 768 summary estimates from 80 saved fold-score files and checks scene identities, hashes, geographic weighting and agreement with the primary analysis. It does not rerun the encoders or regenerate the bootstrap intervals. The plotting command writes three PDF/PNG figures and a CSV containing every control estimate.

```sh
python3 hpc/run.py verify
python3 -m unittest discover -s hpc/tests
"$RUN/fresh-01/.venv/bin/pip" install pytest
"$RUN/fresh-01/.venv/bin/python" hpc/test_all.py
```

Unit tests check submission dependencies and numerical safeguards. A full rerun also requires the source data, weights and cluster resources. CPU/CUDA parity is checked on the target machine before full extraction. Source checksums accompany the methods. The submission helper preserves frozen numerical scripts; portable documentation and wrapper changes have their own package checksum record.

WorldCover provides automated labels. Geographic groups follow Natural Earth country metadata. These results test correspondence, transfer and specific interventions in fitted readouts; they do not establish general semantic competence or absence from pretraining. Keep these cohorts separate from the paper's original expert-labelled sample.
