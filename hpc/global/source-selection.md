---
title: Global automated-reference source selection
author: James Edward Ball
date: 2026-10-15
---

# Source and scope

Use 2021 Sentinel-2 Level-2A single acquisitions from the public Earth Search catalogue and ESA WorldCover 2021 v200 masks. This is a global product-agreement stress test. WorldCover labels are automated; their independent validation does not make every pixel an expert label. Annual labels and single acquisition imagery can disagree through seasonality or within-year change. Pretraining overlap is unresolved.

The six image bands are B02, B03, B04, B8A, B11 and B12, in that order, reprojected to a local UTM grid of 64 by 64 pixels at 10 metres. B8A is explicitly checked from the asset URL. Annual WorldCover composites were considered and rejected because their NIR is B08, whereas the frozen encoders use B8A. No band is silently substituted.

The inventory contains 120 one-degree blocks, with 20 in each of Africa, Asia, Europe, North America, South America and Oceania. Block centres have at least 200 km separation. A 100 km guard excludes known earlier-study and current EuroSAT evaluation coordinates. The geographic split contains exactly 60 fitting, 20 validation and 40 test blocks. The target is 200 attempted windows per block, or 24,000 attempted windows. Admission exclusions reduce the final count and must be reported. Each window is 640 metres wide. Candidate centres are at least 1,000 metres apart within each block, so selected windows have no common pixels.

WorldCover nearest-neighbour overviews provide deterministic class-stratified candidate locations. Their eight-class crosswalk is Water=80, Trees=10, Grass=30, Flooded Vegetation=90, Crops=40, Scrub=20, Built Area=50, Bare Ground=60. Snow/ice, mangroves, moss/lichen and nodata are excluded. Wetlands are an approximate crosswalk to the original flooded-vegetation concept. Categories are not guaranteed in every block.

Admission requires at least 60% cover for one of the eight categories and at most 10% excluded-class or missing reference pixels. Up to 24 Sentinel-2 acquisitions containing each point are considered, ordered by catalogue cloud plus snow percentage, timestamp and item ID. The eligible catalogue contains 2021 acquisitions with less than or equal to 30% cloud and processing baseline below 4. Each retained window requires at least 98% SCL vegetation, bare-ground or water pixels and valid reflectance in all six bands. Scale 0.0001 and offset zero are checked from the asset metadata. Bilinear image resampling and nearest-neighbour categorical resampling are explicit.

# Primary sources

- [ESA WorldCover data access, licence, reference year and validation](https://esa-worldcover.org/en/data-access)
- [WorldCover 2021 v200 dataset DOI](https://doi.org/10.5281/zenodo.7254221)
- [Earth Search STAC API](https://earth-search.aws.element84.com/v1)
- [Sentinel-2 COG registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
- [Natural Earth geographic strata source](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson)

WorldCover is CC BY 4.0. Copernicus Sentinel-2 is open data, with the source conditions recorded by the source registry. Natural Earth is public domain. Preserve the source attribution with exports. The retrieved Natural Earth file is stored and hashed. Each retained image records its source item, date, URLs and exact float-array SHA-256. Source COGs are fetched by HTTP range windows; a multipart S3 ETag is not treated as a content checksum.

# Alternatives considered

[DFC2020](https://www.grss-ieee.org/community/technical-committees/2020-ieee-grss-data-fusion-contest/) has high-resolution semi-manual reference masks, but the official contest description specifies undisclosed geolocation. It cannot immediately support the declared global geographic-block evaluation. [SEN12MS](https://arxiv.org/abs/1906.07789) is worldwide and georeferenced, but its labels are coarse MODIS products and its archive acquisition is much larger. Neither is silently substituted for independent expert validation.

# Operation

Run `code/acquire.py inventory --prior path/to/prior-locations.json --prior path/to/fresh/manifest.json` once, before scoring. Run `code/acquire.py acquire --block INDEX --limit 200` for each of the 120 blocks, with no more than four simultaneous CPU acquisition jobs. A small pilot must verify outbound access, label support, scaling, runtime and storage before dependent acquisition jobs start. Compute-node network restrictions require a documented staging route, rather than policy circumvention.

Per-window `.npz` files contain `image` in float32 and `mask`; adjacent JSON files contain provenance. `report-200.json` gives every admitted row and every failed candidate. Downstream aggregation must require all 120 reports, freeze the final manifest, audit class/block support and fail if the predeclared analysis support is inadequate. Small geographic blocks are sampling clusters; this design does not establish independent global population sampling.

## Eligible-land amendment

Before scoring models, each block must provide 200 candidate centres at least 1 km apart. Candidate discovery requires at least 80% of one allowed WorldCover category in a conservative 900 m neighbourhood, measured on a 1536-pixel overview of the block interior. The exact 640 m image footprint then passes the original full-resolution reference and image-quality checks. Blocks without sufficient candidates are replaced within the same geographic group and split using the original deterministic ordering, 200 km separation and 100 km prior-data guard. Failed inventories and pilots remain archived. This sampling targets relatively homogeneous eligible land-cover patches; conclusions do not cover arbitrary global locations or excluded snow and ice.
