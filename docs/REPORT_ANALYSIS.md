# Phase I report analysis

## Source and identity

Source: *PT355 - Sentinel-2 NDVI Time-Series Analysis for Wheat Crop Health Monitoring and Yield Forecasting*, Phase I report, IILM University, School of Computer Science and Engineering, session 2026–27, submitted May 2026. Team listed: Varun Sinha, Rishabh Singh, Mohit Pandey, Aman Jha. Guide listed: Assistant Prof. TS Arthi.

## Problem and proposed response

The report identifies slow, costly, and geographically limited manual crop surveys as a barrier to early wheat stress detection and pre-harvest yield planning. It proposes an end-to-end historical pipeline combining Sentinel-2 surface reflectance and NDVI time series with wheat yield and weather records.

## Objectives extracted from Phase I

1. Acquire and preprocess historical wheat yield and weather records (planned Kaggle sources).
2. Obtain Sentinel-2 observations for corresponding places and periods through Google Earth Engine.
3. Calculate time-series NDVI features that represent crop vigor and seasonal development.
4. Align satellite, yield, weather, spatial, and temporal variables.
5. Train and compare Random Forest and XGBoost for crop-health/yield prediction.
6. Evaluate with suitable regression measures and document limitations.

## Methodology described

Data sourcing and wheat filtering; Sentinel-2 image filtering and NDVI; season-level summaries (mean, max, min, amplitude, peak timing, observation count); spatial/seasonal data fusion; baseline and tree-based regression; held-out or grouped validation intended to prevent location/season leakage; MAE, RMSE, R²; transparent logging of data and preprocessing assumptions.

## Scope and constraints

Historical analysis and pre-harvest yield estimation using open data; no live real-time deployment and no direct agronomic recommendations are in scope. Risks named include clouds, missing observations, alignment mismatches, limited or biased ground truth, overfitting, and geographic generalization.

## Evidence available for this implementation

Supplied: the Phase I report (PDF and DOCX), an earlier prototype ZIP, loose copies of its documentation, a style image describing the evaluation procedure, and two **header-only** CSV templates. A file `pasted_content.txt` is a prompt for a different project (ReviewSense AI) and was not used. No Phase II repository/code, Kaggle data, field polygons, Earth Engine exports, model artefacts, outputs or screenshots were supplied. The code here is therefore a runnable prototype aligned to the Phase I report, not a continuation of unseen Phase II work, and no experimental results are asserted.

## Decisions and gaps to resolve with the team

- Confirm the study geography, season definition, source dataset URLs/licenses, target yield column and unit, field identifiers, and sample size.
- Confirm reflectance scale and whether all uploaded yield/weather observations represent one row per field-season.
- Confirm what Phase II already implemented and incorporate that code/history before final submission.
- Agree on spatial or temporal holdout design with the supervisor; a random row split is not sufficient when related observations can leak across sets.
- Add measured results only after running this pipeline on the real project dataset.
