# Testing

All results below were produced in this build environment: Python 3.13.18, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1, Streamlit 1.66.1, pytest 8.4.2. Other versions inside the pinned ranges were not tested.

## Automated suite

```bash
pytest -q
```

**Verified result: `30 passed`** (≈23 s), `python -m compileall -q app.py src scripts tests` OK.

| File | Tests | Covers |
|---|---|---|
| `tests/test_pipeline.py` | 6 | NDVI/seasonal features, invalid rows, schema errors, join, duplicate keys, metric structure |
| `tests/test_processing_quality.py` | 12 | int/str/float/whitespace IDs, header case, negative reflectance, same-day scenes, every exclusion reported, duplicate-key message, no-match message, year-boundary peak feature, forecast cutoff, min-observation exclusion, leakage and DN-scale warnings |
| `tests/test_modeling_validation.py` | 10 | field/season/spatio-temporal CV (each row scored once, no overlap), scheme availability, single-field fallback labelled weak, holdout sizes, feature sets, imputation counted, all-missing column dropped and recorded, exclusions |
| `tests/test_cli.py` | 2 | end-to-end `run_pipeline.py` outputs and RUN_RECORD, missing-input refusal |

### Defects found by auditing the original prototype and fixed
Reproduced on the original code before fixing: (1) integer vs string `field_id` crashed the join; (2) trailing whitespace silently prevented matches; (3) negative reflectance was accepted (−0.1/−0.3 → NDVI 0.5) although the docstring said otherwise; (4) same-day duplicate rows were double-counted (4 vs 3 observations); (5) unmatched/missing-yield rows disappeared without any report. Also corrected: the median imputer never ran (rows with missing predictors were pre-dropped); results from a previous upload stayed on screen after new files were uploaded; calendar-DOY peak day is misleading across New Year (days-since-first-observation added). A header BOM was suspected and **ruled out** (pandas handles it).

**Test sensitivity check:** three of these defects were re-introduced into a scratch copy; the matching regression tests failed (4 failures), confirming the tests detect them.

## App checks (browser-driven, Chromium via Playwright)

Server started with `streamlit run app.py`; `/` and `/_stcore/health` returned HTTP 200. Using software-fixture CSVs (not study data) the following were observed: upload → "Joined 18 field-season records (6 fields, 3 seasons)" → default scheme spatio-temporal → Train and evaluate completed (15 folds, 18 out-of-fold rows) → feature-set comparison displayed → uploading a different dataset removed the old results → a deliberately messy fixture produced the unmatched-imagery, missing-yield and target-leakage warnings. No JavaScript page errors. Scores from these fixtures are meaningless and are not recorded anywhere.

## Not verified
- Any run on real wheat data (none available).
- Earth Engine authentication and export (no credentials/asset).
- Excel (.xlsx) upload through the UI (CSV only was exercised; `openpyxl` is installed).
- Streamlit Community Cloud deployment, GitHub push and the GitHub Actions workflow.
- Python 3.10–3.12 and older pandas/scikit-learn versions.
