# Buffalo Creek buckthorn mapping

This repository is an implementation scaffold for a real-data geospatial classifier. **It does not contain verified field observations or study-area NAIP imagery, so it cannot currently produce defensible model metrics or management maps.** The previously committed map, synthetic labels, coordinates, and reported metrics were fabricated by the prototype and have been removed from the production workflow. They must not be cited as findings.

## Inputs

Place a georeferenced four-band NAIP GeoTIFF in `data/buffalo_creek_naip.tif`. Bands must be red, green, blue, near infrared, and the raster must be in EPSG:26916. Retain source URL, NAIP acquisition year, download date, license, CRS, transform, resolution and nodata metadata in a provenance record alongside the file. Do not substitute simulated imagery.

Create `data/verified_observations.csv` with columns `observation_id,longitude,latitude,label,verification_status,observation_date,positional_accuracy_m`. Coordinates are WGS84; labels are 1 for independently confirmed buckthorn and 0 for independently surveyed controls. iNaturalist can help identify candidate locations, but research-grade records are not independently verified field labels, and lack of a record is not absence. Record source IDs and taxonomic/quality metadata. Obtain controls through a documented, independent field protocol.

The optional candidate-record fetcher writes iNaturalist records with provenance and marks them `candidate_unverified`; it does not create training labels. Example: `python fetch_live_naip_and_inaturalist.py --bbox -88.02 42.15 -87.99 42.18`. Inspect and verify each candidate before any field dataset is assembled.

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python buckthorn_spatial_pipeline.py --imagery data/buffalo_creek_naip.tif --observations data/verified_observations.csv
pytest -q
```

Missing inputs, invalid CRS/bands, unverified labels, and unusable spatial folds stop execution. There is no synthetic fallback. Outputs are `outputs/metrics.json`, `outputs/buckthorn_probability.tif`, and `outputs/observations.gpkg`. Probability surfaces are model predictions, not confirmed locations or ecological abundance.

The baseline is Random Forest. Evaluation includes stratified CV as a comparison and grouped 1-km spatial folds with a configurable 45-m exclusion distance. The latter is sample/geography-specific; it does not establish broad real-world generalization. Metrics are only produced when real input data meet the validation requirements. Imagery acquisition date versus observation date, positional uncertainty, accessibility bias, class imbalance, mixed pixels, canopy occlusion and phenological mismatch require study-specific assessment before interpretation. No performance results are currently available.

## Provenance and limitations

NAIP is a USDA aerial imagery program; obtain imagery from an authoritative USDA/USGS endpoint and record the actual product/version, acquisition year, terms and metadata. iNaturalist observations are community observations under their own licenses and quality controls. This repository does not bundle either dataset, assert that a current API response covers the preserve, establish native controls, or provide field validation. Confirm dataset licensing and all acquisition metadata for the specific files used. Management boundaries are not included; no management-unit summaries are generated.

