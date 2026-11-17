# Experimental results

## Status: PENDING — no real dataset was available

No Phase II data, code or outputs were supplied (the only data files received were header-only templates). Therefore **there are no dataset sizes, MAE, RMSE or R² values to report**, and none are invented here. Scores obtained from `tests/fixtures.py` are software-test artefacts and must never be reported.

## How to fill this in

1. Put your files in `data/private/` and run `python scripts/run_pipeline.py --satellite … --context … --out-dir outputs/run1`.
2. Copy from `outputs/run1/RUN_RECORD.md` (it already contains file SHA-256 hashes, software versions, counts, split design and metrics):

| Item | Value (to be filled) |
|---|---|
| Data source, access date, licence | |
| Study region and seasons | |
| Yield variable and **unit** | |
| Fields / seasons / fused field-seasons | |
| Rows excluded by reason (from data-quality report) | |
| Reflectance scale and cloud-mask choice | |
| Weather variables and aggregation | |
| Validation scheme (field / season / spatio-temporal) and folds | |
| Mean baseline: MAE / RMSE / R² | |
| Random Forest: MAE / RMSE / R² | |
| XGBoost: MAE / RMSE / R² (or "not installed") | |
| Satellite-only vs weather-only vs combined (same folds) | |
| Fold-to-fold spread and limitations | |

## Interpretation rules
MAE/RMSE share the yield unit; R² is unitless and may be negative. A model is only interesting if it beats the mean baseline under the strictest scheme the data allow. Differences smaller than the fold-to-fold spread are not evidence. Do not generalise beyond the represented fields and seasons, and do not claim NDVI "causes" yield.
