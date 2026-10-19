# Data

Author: James Edward Ball

## MegaEarth

[Mead0w1ark/megaearth-osm-sentinel2](https://huggingface.co/datasets/Mead0w1ark/megaearth-osm-sentinel2) provides the published imagery, labels, taxonomy and split tables. The reproduction script pins revision `de5ec4da7cb2c195948a12fcbb0ab5b719e0cd92` and verifies files against the frozen release manifest. The retained mask selects 34,992 of the 44,355 stored windows. Use all 188 admitted classes and preserve failed samples.

```sh
python scripts/download_megaearth.py --output work-megaearth
# Include the imagery archives when rebuilding encoder features:
python scripts/download_megaearth.py --output work-megaearth --imagery
```

OSM-derived labels use ODbL 1.0; OpenStreetMap Wiki text uses CC BY-SA 2.0. Imagery contains modified Copernicus Sentinel data. RESOLVE ecoregion attributes use CC BY 4.0. The dataset card gives source attribution and access terms.

## Expert-reference experiment

The Dynamic World validation tiles and expert rasters come from [PANGAEA dataset 933475](https://doi.org/10.1594/PANGAEA.933475), under CC BY 4.0. `data/downloads.json` pins both downloads. The preparation code merges expert votes, retains disagreement as unknown, rebuilds patch coverage and verifies the frozen splits. `data/scenes.json` records the Sentinel-2 source scenes and pixel hashes.

Cached feature and text arrays are release assets in this repository: `semantic-structure-v1` and `prithvi100-v1`. The downloader follows the current GitHub remote, including repository renames. A source ZIP uses the public repository name; set the mirror variable below when using a private source export. All arrays are checked against their recorded SHA-256 hashes. An authenticated GitHub CLI session is required while the repository is private; `GEOFM_ASSET_REPOSITORY=owner/repository` selects an authorised mirror. Source reconstruction uses the checkpoint revisions in `src/study/models.py` and `data/prithvi100/model.json`.

## Supporting experiments and calibration

The HPC guide identifies the EuroSAT and global-source downloads. Frozen manifests, geographic splits and reference scores are included with their analysis code. The earlier 1,000-scene development arrays are a separate diagnostic and do not provide independent human reference labels.

The `geometry-validation-v1` release contains the complete 47,000-record confirmation archive and its checksum. The calibration code and frozen protocol inputs are under `hpc/vocab/supplement-geometry-support-20261018`. Both tested interval procedures failed qualification. These records support verification of that failure.

The coastline data in `data/geography/ne_110m_land.geojson` are from Natural Earth, in the public domain. Third-party code and checkpoint notices are in [NOTICE.md](NOTICE.md).
