# Requirement traceability

Status vocabulary: **Done (verified)** = implemented and covered by passing tests; **Code only** = written but not executed against real inputs; **Pending** = needs your data/credentials.

| Requirement (Phase I) | Implementation | Verification | Status |
|---|---|---|---|
| Acquire/preprocess wheat yield & weather data (Kaggle) | Upload + schema/units/duplicate/missing checks, `DATA_PROVENANCE.md` | Tests; real data not supplied | Tooling done; **dataset pending** |
| Retrieve Sentinel-2 via GEE | `scripts/export_sentinel2_gee.py` (SCL mask, 0–1 reflectance, direct column names) | Compiles, `--help` runs; never executed | **Code only; needs credentials + polygons** |
| Filter poor scenes, compute NDVI | SCL masking (GEE); local exclusion of invalid/negative/zero-denominator/out-of-range rows, counted | 12 processing tests | Done (verified on fixtures); real-scene check pending |
| Seasonal NDVI statistics (mean, max, min, amplitude, peak timing, count) | `build_seasonal_ndvi` (+std, days-to-peak) | Unit tests incl. year-boundary case | Done (verified on fixtures) |
| Fuse satellite, weather, yield | `fuse_with_report`, key normalisation, one-to-one join | Tests | Done (verified); real key reconciliation pending |
| Handle missing values, outliers, duplicates with documented rules | Counted exclusions, same-day merge, duplicate-key rejection, imputation in training fold, leakage-name warning | Tests; `DataQualityReport` | Done; project-specific thresholds (e.g. min observations) to be chosen with data |
| Train Random Forest and XGBoost + baseline | `build_models` | Tests; XGBoost 3.4.1 present here | Done; real run pending |
| Compare satellite-only vs satellite+weather | `compare_feature_sets` on identical folds | Test | Done (mechanism); real result pending |
| Prevent location/season leakage | field CV, season CV, spatio-temporal blocking with asserted disjointness | 10 modeling tests | Done (verified); neighbour-field autocorrelation not modelled |
| Report MAE, RMSE, R² with sample sizes | Holdout and CV tables, RUN_RECORD | Tests | Implemented; **values pending** |
| Pre-harvest forecasting | Optional `forecast_cutoff_date` | Test | Done (mechanism); cutoff choice pending |
| Reproducible, transparent pipeline | CLI + RUN_RECORD with hashes/versions, docs | CLI tests | Done |
| Historical analysis, no live system, no advice | Local app only | Inspection | Done |
| Deployment / GitHub | — | — | **Not done** |
