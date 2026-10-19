<h1 align="center">Do Geospatial Foundation Models Learn Semantic Structure from Imagery Alone?</h1>

<p align="center">
  James Edward Ball<br>
  <a href="https://orcid.org/0000-0002-1052-9747">ORCID</a>
</p>

<p align="center">
  <a href="https://www.overleaf.com/project/6ace291cb9243e3d4b794c17">Paper</a> ·
  <a href="https://huggingface.co/datasets/Mead0w1ark/megaearth-osm-sentinel2">Dataset</a> ·
  <a href="https://github.com/JamesEBall/geofm-semantic-structure/releases">Release assets</a>
</p>

## Highlights

Geospatial foundation models learn representations from satellite imagery. This study asks whether the relationships between land-feature categories in those representations resemble relationships expressed in language, and whether a fitted image-to-language map transfers to categories excluded from training.

- **Models:** frozen Clay, Prithvi 300M and Prithvi 100M representations, evaluated against matched controls.
- **Data:** the corrected MegaEarth cohort contains 34,992 retained windows, 188 admitted classes and 91 admitted families, with 8,755 primary test pairs. Supporting experiments use expert-labelled scenes, EuroSAT and a global Sentinel-2 sample.
- **Reproduction:** pinned data downloads, saved feature assets and numerical reference results accompany the analysis code and tests.

[![Experiment overview showing satellite imagery, frozen encoders, category-distance comparisons, a fitted language map, transfer tests and controls.](assets/experiment-overview.png)](assets/experiment-overview.png)

*Frozen image features support category-distance comparisons and fitted language readouts, evaluated on held-out categories and geography. Select the diagram to view it at full size.*

## Experiments

| Experiment | Question |
| --- | --- |
| Category structure | Do distances between image-category representations agree with distances between language representations? |
| Language readout | How well can a linear ridge map connect frozen image features to language embeddings? |
| Transfer | Does that map work for categories and category families excluded from fitting? |
| Controls and calibration | How much can appearance, metadata and random encoders explain, and do the uncertainty procedures meet their declared requirements? |

## Results

Corrected MegaEarth family-transfer results use mean pooling, a ridge readout and the instructed Qwen language reference. Class-balanced filtered mean reciprocal rank (MRR) measures how highly the correct category is ranked; higher values indicate better retrieval.

| Representation | MRR |
| --- | ---: |
| Clay | 0.2417 |
| Prithvi 300M | 0.2301 |
| Prithvi 100M | 0.2126 |
| Appearance + metadata control | 0.2076 |

Clay and Prithvi 300M show positive gains in the paired conditional analysis; the Prithvi 100M comparison remains unresolved. Full transfer calibration fails its declared requirements, and population geometry intervals remain unqualified. These findings support a bounded claim about the tested representations and evaluation setup; general semantic capability remains an open question.

The [corrected results](hpc/vocab/results/retained-20261018/results.json) and [qualification records](hpc/vocab/results/retained-20261018/qualify/) preserve the estimates and statistical checks.

## Installation

Use Python 3.12. The commands below set up a CPU environment for data checks, saved-result plots and the expert-reference analysis. Install the [GitHub CLI](https://cli.github.com/) and run `gh auth login` to download the feature assets.

```sh
git clone https://github.com/JamesEBall/geofm-semantic-structure.git
cd geofm-semantic-structure
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pandas pyarrow
```

Full encoder extraction requires the separate Linux/CUDA environment described in the [HPC guide](hpc/README.md).

## Data

Download the published MegaEarth tables at the pinned revision:

```sh
python scripts/download_megaearth.py --output work-megaearth
```

Each file is checked against the frozen SHA-256 manifest. Add `--imagery` to include the image archives. Cached features for the expert-reference experiments and the 47,000 geometry-confirmation records are available under [Releases](https://github.com/JamesEBall/geofm-semantic-structure/releases). See [DATA.md](DATA.md) for sources and data licences, and [NOTICE.md](NOTICE.md) for third-party notices.

## Reproduction

### MegaEarth figures

Verify the retained cohort and regenerate the support and transfer figures:

```sh
python scripts/plot_megaearth_support.py --tables work-megaearth/tables
python scripts/plot_megaearth_transfer.py
```

Outputs are saved in `output/figures/`. These commands plot the archived estimates. To rebuild features and rerun the statistical analysis, follow the [MegaEarth guide](hpc/vocab/README.md), which covers the frozen inputs, resource requirements and admission checks.

### Expert-reference experiment

```sh
./reproduce.sh
```

This downloads the original labels and cached features, reconstructs the study splits, reruns the analyses and checks agreement with the frozen results. Outputs are saved in `work/`. Add `--from-source` to rebuild features from imagery.

For the [Prithvi 100M extension](data/prithvi100/README.md):

```sh
python -m study.prithvi100 --work work-prithvi100 --check-validation
```

The [HPC guide](hpc/README.md) covers the fresh-data, global and intervention experiments, including model staging and Slurm submission.

## Validation

```sh
python -m pip install -e '.[test,hpc-test]'
python scripts/test_all.py --megaearth
python hpc/check_global_results.py
```

The tests check analysis behaviour on controlled inputs. The global-results audit verifies 768 saved estimates from 80 fold-score files. These checks preserve the study's failed statistical gates. Keep this test environment separate from the pinned HPC environment.

## Repository structure

```text
src/             Model interfaces and expert-reference analysis
scripts/         Dataset download, plots and validation commands
data/            Frozen inputs, splits and reference results
hpc/             HPC runners and supporting experiments
  vocab/         MegaEarth pipeline, corrected results and calibration
tests/           Package tests
vendor/          Third-party model code and notices
```

## Citation

Repository citation metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{ball2026geofm_code,
  author = {James Edward Ball},
  title = {Do Geospatial Foundation Models Learn Semantic Structure from Imagery Alone?},
  year = {2026},
  howpublished = {Code and reference results},
  url = {https://github.com/JamesEBall/geofm-semantic-structure}
}
```
