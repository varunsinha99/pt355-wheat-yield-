# PT355 — Sentinel-2 NDVI Time-Series Analysis for Wheat Crop Health Monitoring and Yield Forecasting

**Phase III implementation report (draft)** — B.Tech CSE, IILM University, Session 2026–27
**Team:** Varun Sinha (2410031273), Rishabh Singh (2410031257), Mohit Pandey (2410031275), Aman Jha (2410031330) · **Guide:** Assistant Prof. TS Arthi

> Draft describing the implementation in this repository. It makes no claim about Phase II work that was not supplied. Sections marked **PENDING** need your real data; institutional formatting must be added.

## Abstract
This project addresses wheat crop-health monitoring and pre-harvest yield estimation from multi-temporal Sentinel-2 observations and historical agricultural context. The implemented prototype ingests red/NIR reflectance observations and a field-season yield/weather table, computes NDVI, aggregates seasonal features, joins the sources, audits every exclusion, and compares a mean baseline, Random Forest and (when installed) XGBoost under field-, season- and spatio-temporally-grouped validation, reporting MAE, RMSE and R² with sample sizes. A Streamlit interface and a command-line runner provide the workflow. The software is verified by 30 automated tests on software fixtures. No real dataset was available, so empirical performance is **PENDING**.

## 1. Problem and motivation
Manual field surveys are slow and cover few locations. Repeated satellite observation can characterise crop vigour over a season; combined with yield and weather history it may support pre-harvest forecasts. The challenge is aligning heterogeneous data and validating without leakage.

## 2. Objectives and scope
Phase I objectives: (1) acquire historical wheat yield/weather data; (2) obtain Sentinel-2 observations via Google Earth Engine; (3) derive seasonal NDVI features; (4) fuse the sources; (5) train and compare Random Forest and XGBoost; (6) evaluate with regression metrics and document limitations. Scope: historical analysis and pre-harvest estimation with open data; no real-time deployment and no irrigation/fertilizer advice.

## 3. Existing and proposed system
Manual surveys remain valuable but are spatially limited. The pipeline complements, not replaces, agronomic expertise.

## 4. Requirements and architecture
See `ARCHITECTURE.md` and `REQUIREMENT_TRACEABILITY.md`. Inputs: satellite rows (`field_id`, `season`, `observation_date`, `red_reflectance`, `nir_reflectance`) and one yield/weather row per field-season. Outputs: fused feature table, data-quality report, metrics, predictions.

## 5. Technology
Python, Pandas, NumPy, scikit-learn, XGBoost (optional), Plotly, Streamlit; Earth Engine Python API for optional extraction (aligned with Phase I's Python + GEE plan).

## 6. Methodology and implementation
**NDVI.** (NIR − red)/(NIR + red) per observation. Excluded and counted: missing keys, unparseable dates, missing/negative reflectance, zero denominators, NDVI outside [−1, 1]. Scenes on the same date are averaged into one observation.
**Features.** Mean, max, min, standard deviation, amplitude, valid-observation count, peak day-of-year, and days from first observation to peak (robust to seasons spanning New Year). An optional per-field-season `forecast_cutoff_date` excludes later imagery for genuine pre-harvest forecasting.
**Fusion.** Identifiers are canonicalised (1, 1.0, " 1 " are equal); join on `field_id` + `season`; duplicate yield keys are rejected; unmatched keys on either side, missing yields and thin coverage are reported, never silently dropped.
**Models.** Mean baseline, Random Forest (300 trees), XGBoost if installed; median imputation fitted within each training fold. Columns that look like target outputs (yield/production/harvest) are flagged and excluded by default.
**Validation.** Grouped holdout; field-grouped CV; leave-one-season-out CV; spatio-temporal blocked CV (test = held-out season × held-out field fold; training = other seasons and other fields). Disjointness is asserted. A satellite-only vs weather-only vs combined comparison uses identical folds. Metrics: MAE, RMSE, R²; pooled out-of-fold values plus fold-level spread.

## 7. Dashboard
CSV/XLSX upload, previews, audit panel, predictor and scheme selection, results with sample sizes, charts, and CSV/JSON downloads. Old results are cleared when new data are loaded.

## 8. Testing
30 automated tests pass (`TESTING.md`): feature calculations, exclusion handling, key normalisation, report contents, leakage warnings, forecast cutoff, validation-scheme guarantees, CLI outputs. Five concrete defects of the original prototype were reproduced and fixed. Browser-driven checks of the upload → audit → CV → comparison flow succeeded on software fixtures. Fixtures are not study data.

## 9. Experimental setup and results — **PENDING**
Dataset, geography, seasons, sample size and all model metrics are pending. Complete `EXPERIMENTAL_RESULTS.md` and `DATA_PROVENANCE.md` after running `scripts/run_pipeline.py` on the real data. Do not report fixture outputs.

## 10. Challenges and limitations
Cloud contamination, sparse or biased ground truth, season/field matching, small sample size, neighbouring-field autocorrelation (not modelled), geographic shift. No confidence intervals or uncertainty modelling. The Earth Engine script is untested without credentials. Spatio-temporal blocking reduces training data and may skip cells (counted).

## 11. Security and ethics
Private data and credentials are git-ignored (`data/private/`, `outputs/`, key files) and no secrets are required by the code. Check dataset licences and attribute sources. Present outputs as research estimates, not recommendations.

## 12. Conclusion and future scope
A tested end-to-end prototype with leakage-aware validation is ready for the team's real data. Remaining: integrate and document the verified dataset, run the experiments, test the Earth Engine export, add uncertainty estimates and spatial diagnostics, and deploy if required.

## References
Use the Phase I bibliography (Copernicus Sentinel-2 documentation; Aslan et al.; Zhao et al.; Gorelick et al.; Rouse et al.; Breiman; Chen & Guestrin) and verify formatting against the university's style. No new sources are asserted.
