# Phase III presentation, demo and viva preparation

## Slide outline (9 slides)

| # | Content | Note |
|---|---|---|
| 1 | Title, team, guide | NDVI visual from your own scene |
| 2 | Problem and motivation | Manual survey limits |
| 3 | Objectives and boundaries | Six Phase I objectives; historical, pre-harvest, no advice |
| 4 | Data and workflow | Sentinel-2 → NDVI → seasonal features → fusion → models |
| 5 | Architecture and stack | Show `ARCHITECTURE.md` diagram |
| 6 | Prototype demo | Upload, audit panel, fused table |
| 7 | Validation and models | Four schemes, why spatio-temporal; **insert real metrics only after your run** |
| 8 | Challenges and limitations | Clouds, labels, leakage, sample size, autocorrelation |
| 9 | Conclusion and next work | Tested prototype; real-data results pending/complete as applicable |

## Demo script (≈6 min)

1. **Open (30 s):** "We test whether Sentinel-2 NDVI time series plus weather support pre-harvest wheat yield estimation. This is a research prototype, not field advice."
2. **Start (30 s):** `source .venv/bin/activate && streamlit run app.py`.
3. **Inputs (45 s):** explain the two files, units and study area from your real documentation (`DATA_PROVENANCE.md`).
4. **Upload + audit (90 s):** upload your real CSVs. Show the joined count, then open "Everything checked, excluded or unmatched" — point at exclusion counts, unmatched keys and any warnings. *If you have no real data on the day, show the templates and say empirical evaluation is pending; do not upload made-up records.*
5. **Features (45 s):** NDVI formula; mean/max/min/std/amplitude/count/peak timing; why days-to-peak exists.
6. **Predictors and validation (60 s):** show leakage banner logic; choose the scheme; explain what each prevents.
7. **Train (60–90 s):** Train and evaluate; read MAE/RMSE/R², folds, rows evaluated; compare to the mean baseline. Optionally run the feature-set comparison.
8. **Export + close (45 s):** download fused table / predictions / quality report; state limitations and pending items honestly.

Headless alternative: `python scripts/run_pipeline.py …` then show `RUN_RECORD.md`.

## Code flow
`app.py` (or `run_pipeline.py`) → `fuse_with_report()` in `src/processing.py` (validate, normalise keys, NDVI, aggregate, join, audit) → `cross_validate_models()` / `compare_models()` in `src/modeling.py` → metrics, predictions, charts, downloads.

## Viva questions and answers

1. **Problem?** Scalable wheat monitoring and pre-harvest yield estimation from satellite vegetation data and historical context.
2. **NDVI?** (NIR − red)/(NIR + red), a normalised vegetation-greenness contrast.
3. **Why Sentinel-2?** Free multispectral imagery at 10 m with Level-2A surface reflectance and a scene classification layer.
4. **Why a time series?** Crop development is seasonal; one image misses phenology.
5. **Why aggregate to field-season?** To make one row per field-season that joins with seasonal yield and weather.
6. **Features?** Mean, max, min, std, amplitude, valid-observation count, peak day-of-year, days from first observation to peak.
7. **Why two peak features?** Calendar day-of-year wraps at New Year; wheat seasons often span two calendar years.
8. **Cloud problem?** Clouds/shadows distort reflectance; SCL masking in GEE plus validity checks and observation counts locally.
9. **What does the audit report?** Every excluded row by reason, unmatched keys on both sides, missing yields, thin coverage, scale and leakage warnings.
10. **Why merge same-day scenes?** Overlapping tiles give two rows for one date; counting both would inflate observation counts and weight that date twice.
11. **Why reject negative reflectance?** It is non-physical and can yield a plausible-looking NDVI from two meaningless values.
12. **How are datasets joined?** On `field_id` + `season`, after canonicalising identifiers; one-to-one, duplicates rejected.
13. **Target?** The numeric `yield` column; its unit must be documented from the source and is not converted by the app.
14. **Why Random Forest / XGBoost?** Tree ensembles handle nonlinear structured data with small samples; XGBoost is boosted trees, used if installed.
15. **Why a mean baseline?** A model must beat "always predict the average" to be useful.
16. **MAE / RMSE / R²?** Mean absolute error and root mean squared error in yield units (RMSE penalises large errors more); R² is unitless and can be negative.
17. **Data leakage?** Information from test data or near-duplicates influencing training so scores are optimistic.
18. **How do you limit it?** Preprocessing is fitted inside training folds; field-grouped, season-grouped and spatio-temporal blocked splits; disjointness asserted in code.
19. **Why is field CV alone insufficient?** The same season/weather year can sit in train and test; season CV fixes that but reuses fields; the blocked scheme removes both.
20. **What leakage remains?** Spatial autocorrelation between neighbouring fields; with few fields scores are unstable.
21. **Column named `production`?** Likely derived from yield (area × yield) — flagged and excluded by default.
22. **How to test whether weather/NDVI help?** Satellite-only vs weather-only vs combined on identical folds; only differences larger than fold spread matter.
23. **What is `forecast_cutoff_date`?** Optional date per field-season after which imagery is ignored, so features use only information available before the forecast.
24. **Your results?** State your measured values with scheme and sample size — or, if the run is not finished, say plainly that empirical evaluation is pending. Never quote test-fixture numbers.
25. **Are the unit tests your data?** No — software fixtures only.
26. **What was tested and how?** 30 pytest tests; browser-driven app flow on fixtures; five original defects reproduced and fixed; tests shown to fail when defects are re-introduced.
27. **What did you not verify?** Real-data run, Earth Engine export, Excel upload, deployment.
28. **Architecture?** Streamlit UI, Pandas processing, scikit-learn/XGBoost modelling, optional GEE extraction, CLI that writes a run record.
29. **Security?** No credentials in code; private data and outputs git-ignored; check dataset licences.
30. **Can it recommend irrigation or fertilizer?** No; scope is monitoring and forecasting.
31. **Next steps?** Integrate and document real data, run the experiments, test GEE, add uncertainty and spatial diagnostics.
