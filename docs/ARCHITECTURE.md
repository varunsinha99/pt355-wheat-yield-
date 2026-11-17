# Architecture

```text
Sentinel-2 L2A (optional GEE script, unverified)       Yield + seasonal weather table
                  |                                               |
        field/date red & NIR rows                                 |
                  |                                               |
   validate -> normalise keys -> per-observation NDVI             |
   (every exclusion counted)   -> merge same-day scenes           |
                  |             -> optional forecast cutoff       |
        field-season NDVI features                                |
                  +--------- join on field_id + season ----------+
                                    |
                     fused table + DataQualityReport
                                    |
            predictor selection (leakage-name warning/auto-exclude)
                                    |
      holdout | field CV | season CV | spatio-temporal blocked CV
                 /             |               \
        mean baseline    Random Forest    XGBoost (if installed)
        (imputer fitted inside each training fold)
                                    |
        MAE, RMSE, R2 + sample sizes + out-of-fold predictions
                                    |
         Streamlit UI  /  scripts/run_pipeline.py -> RUN_RECORD.md
```

## Components
- `src/processing.py` — column/key standardisation, NDVI and seasonal aggregation, `fuse_with_report()` returning the fused table and a `DataQualityReport` (counts, warnings, unmatched keys, yield summary), `candidate_predictors()`.
- `src/modeling.py` — `build_models`, `compare_models` (grouped holdout), `cross_validate_models` (`field`, `season`, `spatiotemporal`), `compare_feature_sets`, `available_schemes`/`recommend_scheme`. Leakage disjointness is asserted per fold.
- `app.py` — upload, preview, audit panel, predictor and scheme selection, results, downloads; results are cleared when the data change.
- `scripts/run_pipeline.py` — headless equivalent that writes tables plus `RUN_RECORD.md` with SHA-256 of inputs.
- `scripts/export_sentinel2_gee.py` — optional extraction; requires user credentials and polygons; not executed.
- `tests/` — software fixtures only.

## Data contracts
Satellite: one row per field/date scene. Context: one row per field-season with numeric `yield`; other numeric columns become predictors. Keys are matched after canonicalising text (1, 1.0, " 1 " → "1"); season labels must otherwise match exactly. NDVI = (NIR−red)/(NIR+red); rows with negative reflectance, zero denominator or NDVI outside [−1, 1] are excluded and counted.

## Validation design and its limits
Field CV prevents the same field in train and test but lets the same season (weather year) appear in both; season CV does the reverse. The spatio-temporal scheme removes both overlaps at the cost of less training data and some skipped cells (counted). None of them models spatial autocorrelation between neighbouring fields, so with few fields scores remain optimistic or unstable. The scheme and sample sizes must accompany every score.
