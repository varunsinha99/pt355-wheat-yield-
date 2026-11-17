# PT355 — Sentinel-2 NDVI Wheat Crop Health and Yield Forecasting

A Python prototype implementing the Phase I methodology: compute field-season NDVI summaries from Sentinel-2 red/NIR reflectance, join them with seasonal yield and weather data, and compare regression models under leakage-aware validation.

> **Data integrity.** No Phase II dataset, source code, or measured results were supplied; the only data files are header-only templates. The repository contains **no research data and no reported accuracy**. Tests use in-memory software fixtures that are not wheat data. Every result must come from your own documented data via the app or `scripts/run_pipeline.py`.

## What is implemented (verified by tests, see `docs/TESTING.md`)

- Per-observation NDVI; per-field-season mean, max, min, std, amplitude, valid-observation count, peak day-of-year and days-from-first-observation to peak.
- Data-quality audit: every excluded row is counted by reason (bad date, missing/negative reflectance, zero denominator, NDVI out of range, same-day duplicate scenes merged, missing yield, unmatched keys on either side). Nothing is dropped silently.
- Robust joining on `field_id` + `season` (int/str/float IDs and stray whitespace normalised; duplicate yield keys rejected with the offending keys named).
- Warnings for DN-scale reflectance, mismatched band scales, non-positive yield, thin coverage, and columns that look like target leakage (`production`, `harvest`, …).
- Optional `forecast_cutoff_date` column in the yield table: imagery after that date is excluded, for genuine pre-harvest forecasting.
- Models: mean baseline, Random Forest, XGBoost (when installed). Preprocessing (median imputation) is fitted inside each training fold.
- Validation: grouped holdout; field-grouped CV; leave-one-season-out CV; spatio-temporal blocked CV (test cell = held-out season × held-out field fold; training uses other seasons **and** other fields). Group disjointness is asserted in code.
- Feature-set comparison (satellite-only vs weather-only vs combined) on identical folds.
- Streamlit app, and a headless runner that writes an auditable `RUN_RECORD.md` (file hashes, versions, split design, metrics).

## Install and run (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pytest -q                            # expect 30 passed
streamlit run app.py                 # open the printed local URL
```

## Run on your real data

Place your files in `data/private/` (git-ignored), then either upload them in the app or run:

```bash
python scripts/run_pipeline.py \
  --satellite data/private/satellite_observations.csv \
  --context   data/private/yield_weather.csv \
  --out-dir   outputs/run1
```

Then copy the measured numbers from `outputs/run1/RUN_RECORD.md` into `docs/EXPERIMENTAL_RESULTS.md` and complete `docs/DATA_PROVENANCE.md`.

### Data format

**Satellite observations** — one row per field/date: `field_id`, `season`, `observation_date`, `red_reflectance` (B4), `nir_reflectance` (B8). Both bands on the same scale.
**Yield and weather** — one row per field-season: `field_id`, `season`, `yield` (single consistent unit), optional numeric seasonal weather columns, optional `forecast_cutoff_date`. Aggregate daily weather to the season first. Season labels must match exactly across files (`2022-23` ≠ `2022/23`; the audit tells you when keys do not match).

## Earth Engine extraction (optional, not executed)

```bash
python -m pip install -r requirements-gee.txt
earthengine authenticate             # credentials stay in your environment, never in the repo
python scripts/export_sentinel2_gee.py --project YOUR_GCP_PROJECT \
  --asset projects/YOUR_GCP_PROJECT/assets/wheat_fields \
  --start 2022-11-01 --end 2023-05-01 --season 2022-23
```

The script exports `field_id, season, observation_date, red_reflectance, nir_reflectance` (0–1 scale) to Drive with SCL cloud/shadow/snow masking. **It has not been run** (no credentials or field polygons available); test on one field and inspect the CSV first.

## Repository layout

```text
app.py                          Streamlit UI
src/processing.py               validation, NDVI, aggregation, fusion, quality report
src/modeling.py                 models, holdout, grouped CV, feature-set comparison
scripts/run_pipeline.py         headless run + RUN_RECORD.md
scripts/export_sentinel2_gee.py optional Earth Engine extraction (unverified)
data/templates/                 header-only templates;  data/private/ = your data (git-ignored)
tests/                          30 tests; fixtures.py = software fixtures only
docs/                           report, architecture, testing, results, traceability, demo/viva, provenance
```

## Limitations

- No empirical result exists yet; XGBoost is evaluated only if installed.
- Small samples give unstable scores; no confidence intervals or uncertainty modelling are implemented.
- Neighbouring-field spatial autocorrelation is not modelled; field/season grouping reduces but does not eliminate leakage.
- Earth Engine export, Streamlit Cloud deployment and GitHub upload have not been done.
- No irrigation/fertilizer advice, no real-time monitoring.

## Deployment (not performed)

Streamlit Community Cloud: push to GitHub, entry point `app.py`, dependencies from `requirements.txt`. Never commit `data/private/`, `outputs/`, or credentials (all git-ignored by default).

## Academic source

Phase I report *PT355 – Sentinel-2 NDVI Time-Series Analysis for Wheat Crop Health Monitoring and Yield Forecasting* (IILM University, May 2026).
