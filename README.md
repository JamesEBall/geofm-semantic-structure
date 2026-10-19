# Do Geospatial Foundation Models Learn Semantic Structure from Imagery Alone?

James Edward Ball · [ORCID](https://orcid.org/0000-0002-1052-9747)

Reproduction code and reference results for the [paper](https://www.overleaf.com/project/6ace291cb9243e3d4b794c17), using [MegaEarth](https://huggingface.co/datasets/Mead0w1ark/megaearth-osm-sentinel2) and supporting datasets.

## Setup

Use Python 3.12. Install the [GitHub CLI](https://cli.github.com/) and run `gh auth login` to download the feature assets.

```sh
git clone https://github.com/JamesEBall/geofm-semantic-structure.git
cd geofm-semantic-structure
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pandas pyarrow
```

## Reproduce the results

To verify the published MegaEarth tables and regenerate the corrected support and transfer figures:

```sh
python scripts/download_megaearth.py --output work-megaearth
python scripts/plot_megaearth_support.py --tables work-megaearth/tables
python scripts/plot_megaearth_transfer.py
```

The downloader uses a fixed dataset revision and checks each file hash. These commands verify cohort support and plot the saved estimates; figures are written to `output/figures/`. Add `--imagery` to the download command to fetch the image archives.

To reproduce the expert-reference experiment:

```sh
./reproduce.sh
```

This downloads the original labels and cached features, reconstructs the study splits, reruns the analyses and compares them with the frozen results. Add `--from-source` to rebuild features from imagery. The [Prithvi-100 extension](data/prithvi100/README.md) has a separate entry point.

Full feature extraction and MegaEarth analysis require Linux, CUDA and substantial storage. Follow the [MegaEarth guide](hpc/vocab/README.md) and [HPC guide](hpc/README.md). [DATA.md](DATA.md) lists data sources and licences.

## Checks

```sh
python -m pip install -e '.[test,hpc-test]'
python scripts/test_all.py --megaearth
python hpc/check_global_results.py
```

These run unit and synthetic integration tests and verify the archived global estimates. Full transfer calibration and population geometry intervals remain unqualified; the package preserves those outcomes. Keep this test environment separate from the pinned HPC environment.
