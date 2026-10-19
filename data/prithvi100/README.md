# Prithvi EO 1.0 100M extension

Run from the repository root after installing the package:

```sh
python -m study.prithvi100 --work work-prithvi100 --check-validation
```

This repeats the full geometry and transfer analysis with Prithvi EO 1.0 100M and three matched random encoders, alongside the original models and controls. It uses the same images, labels, splits, language references, fitting budgets and validation-selected penalties. The derived feature archive is downloaded from the repository release. `--check-validation` recomputes all additional validation penalty grids and checks their selected settings. Results and numerical verification are saved in the chosen work directory.

Add `--from-source` to download the original imagery and rebuild the additional encoder features from the pinned checkpoint. Original comparator features remain cached; use the main reproduction command with `--from-source` to rebuild those as well. Source access needs network connectivity and may take hours.

The additional checkpoint is an exploratory extension of the existing benchmark. It preserves the original hypothesis family and does not establish a model-size effect or general semantic competence. `validation.json.gz` contains all additional penalty candidates and selected settings; `calibration.npz` contains the fitted nuisance map; `expected-results.json.gz` contains the additional reference endpoints.

The source model is IBM and NASA's [Prithvi EO 1.0 100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M), pinned at revision `f3a9ea7a1723621b0aeceb4de993093704d712c5`. Its model card declares Apache-2.0. The downloaded encoder source carries IBM's 2024 copyright and Apache-2.0 notice. Model weights and upstream source are downloaded from that revision and checked against `model.json`; they are not included in this repository.
