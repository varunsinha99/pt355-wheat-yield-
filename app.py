from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from src.modeling import (available_schemes, compare_feature_sets, compare_models, cross_validate_models,
                          recommend_scheme, xgboost_available)
from src.processing import (CONTEXT_REQUIRED, SATELLITE_REQUIRED, candidate_predictors, fuse_with_report,
                            suspected_leakage_columns)

SCHEME_LABELS = {
    "holdout": "Single grouped holdout",
    "field": "Field-grouped cross-validation",
    "season": "Leave-one-season-out cross-validation",
    "spatiotemporal": "Spatio-temporal blocked cross-validation (strictest)",
}

st.set_page_config(page_title="PT355 Wheat Yield Forecasting", page_icon="🌾", layout="wide")
st.title("Sentinel-2 NDVI Time-Series Analysis")
st.caption("PT355 · Wheat crop health monitoring and yield forecasting · research prototype, historical analysis only")
st.info("Upload your project's real data to run the analysis. The app bundles no observations and shows no score until you run an evaluation.")

with st.sidebar:
    st.header("Project workflow")
    st.markdown("1. Upload satellite observations\n2. Upload field-season yield and weather\n3. Review the data-quality audit\n4. Choose predictors and validation\n5. Compare models\n6. Download outputs")
    st.caption("Keep red and NIR reflectance on the same scale. Do not mix raw digital numbers and scaled reflectance.")
    st.caption(f"XGBoost: {'installed' if xgboost_available() else 'not installed - it will not be evaluated'}")

st.subheader("1 · Upload data")
left, right = st.columns(2)
with left:
    sat_file = st.file_uploader("Satellite observations (CSV or Excel)", type=["csv", "xlsx"], key="sat")
    st.caption("Required columns: " + ", ".join(sorted(SATELLITE_REQUIRED)))
with right:
    ctx_file = st.file_uploader("Yield and seasonal weather (CSV or Excel)", type=["csv", "xlsx"], key="ctx")
    st.caption("Required: " + ", ".join(sorted(CONTEXT_REQUIRED)) + ". Extra numeric columns are candidate predictors. Optional `forecast_cutoff_date` limits imagery to a pre-harvest date per field-season.")


def read_table(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None


def fingerprint(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


sat, ctx = read_table(sat_file), read_table(ctx_file)

if sat is not None and ctx is not None:
    st.subheader("2 · Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Satellite input: {len(sat):,} rows")
        st.dataframe(sat.head(8), width="stretch")
    with c2:
        st.write(f"Yield/weather input: {len(ctx):,} rows")
        st.dataframe(ctx.head(8), width="stretch")

    min_obs = st.number_input("Minimum valid observations per field-season (1 keeps everything; excluded rows are reported)", min_value=1, value=1, step=1)
    try:
        fused, report = fuse_with_report(sat, ctx, min_observations=int(min_obs))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    data_id = fingerprint(sat_file.name, len(sat), ctx_file.name, len(ctx), int(min_obs), pd.util.hash_pandas_object(fused, index=False).sum())
    if st.session_state.get("data_id") != data_id:  # new data -> discard results from the previous upload
        for key in ("holdout", "cv", "featsets"):
            st.session_state.pop(key, None)
        st.session_state["data_id"] = data_id

    st.subheader("3 · Data-quality audit")
    st.success(f"Joined {len(fused):,} field-season records ({fused.field_id.nunique()} fields, {fused.season.nunique()} seasons).")
    for warning in report.warnings:
        st.warning(warning)
    with st.expander("Everything checked, excluded or unmatched", expanded=bool(report.warnings)):
        st.dataframe(pd.DataFrame({"check": list(report.counts), "value": list(report.counts.values())}), hide_index=True, width="stretch")
        st.write("Yield summary (verify the unit against your source metadata):", report.yield_summary)
        if report.satellite_only_keys:
            st.write("Field-seasons with imagery but no yield row (first 20):", report.satellite_only_keys)
        if report.context_only_keys:
            st.write("Yield rows without usable imagery (first 20):", report.context_only_keys)
    st.dataframe(fused, width="stretch")
    with st.expander("Feature definitions"):
        st.markdown("NDVI = (NIR − red) / (NIR + red), computed per observation; several scenes on one date are averaged. Per field-season: mean, max, min, standard deviation, amplitude (max − min), valid observation count, peak day-of-year, and days from first observation to the peak (use this one if a season crosses New Year).")
    d1, d2 = st.columns(2)
    d1.download_button("Download fused modeling table (CSV)", fused.to_csv(index=False).encode("utf-8"), "pt355_fused_features.csv", "text/csv")
    d2.download_button("Download data-quality report (JSON)", pd.Series(report.to_dict()).to_json(indent=2).encode("utf-8"), "pt355_data_quality.json", "application/json")

    st.subheader("4 · Predictors and validation")
    cands = candidate_predictors(fused)
    suspects = suspected_leakage_columns(fused)
    if suspects:
        st.error(f"Columns that look like yield/production outputs were removed from the predictor list by default: {suspects}. Re-add them only if they are known before harvest.")
    chosen = st.multiselect("Predictors", cands, default=[c for c in cands if c not in suspects])
    exclude = [c for c in cands if c not in chosen]
    avail = available_schemes(fused)
    default_scheme = recommend_scheme(fused)
    options = [s for s in SCHEME_LABELS if avail[s] == "ok"]
    scheme = st.selectbox("Validation scheme", options, index=options.index(default_scheme), format_func=SCHEME_LABELS.get)
    for s, why in avail.items():
        if why != "ok":
            st.caption(f"Not available - {SCHEME_LABELS[s]}: {why}.")
    if scheme == "holdout" and fused.field_id.nunique() < 2:
        st.warning("Only one field: the holdout is a random row split and is weak evidence of generalization.")
    st.warning("Scores describe only the uploaded dataset and the chosen split. With few fields or seasons they are unstable; no score is a guarantee of performance elsewhere.")

    if not chosen:
        st.info("Select at least one predictor.")
    else:
        test_size = st.slider("Holdout fraction (single holdout only)", 0.15, 0.40, 0.25, 0.05)
        n_splits = st.slider("Folds (cross-validation)", 2, 10, 5)
        run, run_sets = st.columns(2)
        if run.button("Train and evaluate", type="primary"):
            try:
                with st.spinner("Training on the uploaded records…"):
                    if scheme == "holdout":
                        st.session_state["holdout"] = compare_models(fused, test_size=test_size, exclude=exclude) + (list(chosen),)
                        st.session_state.pop("cv", None)
                    else:
                        st.session_state["cv"] = cross_validate_models(fused, scheme, n_splits=n_splits, exclude=exclude)
                        st.session_state.pop("holdout", None)
            except Exception as exc:
                st.error(f"Evaluation could not be completed: {exc}")
        if run_sets.button("Compare satellite-only vs weather-only vs combined"):
            try:
                with st.spinner("Running the same folds for each feature set…"):
                    st.session_state["featsets"] = compare_feature_sets(fused, scheme if scheme != "holdout" else "field", n_splits=n_splits, exclude=exclude)
                    st.session_state["featsets_scheme"] = scheme if scheme != "holdout" else "field"
            except Exception as exc:
                st.error(f"Feature-set comparison could not be completed: {exc}")

        if "holdout" in st.session_state:
            metrics, predictions, _, features, _ = st.session_state["holdout"]
            st.markdown("#### Holdout results")
            st.caption(f"{metrics.split.iloc[0]} · train {metrics.train_rows.iloc[0]} rows / test {metrics.test_rows.iloc[0]} rows · {metrics.n_features.iloc[0]} predictors")
            st.dataframe(metrics, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(metrics, x="model", y="RMSE", title="Holdout RMSE by model", text_auto=".3f"), width="stretch")
            if len(predictions):
                chart = px.scatter(predictions, x="actual_yield", y="Random Forest_prediction", hover_data=[c for c in ["field_id", "season"] if c in predictions], title="Random Forest: holdout actual vs predicted")
                lo, hi = predictions.actual_yield.min(), predictions.actual_yield.max()
                chart.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
                st.plotly_chart(chart, width="stretch")
                st.download_button("Download holdout predictions (CSV)", predictions.to_csv(index=False).encode("utf-8"), "pt355_holdout_predictions.csv", "text/csv")
        if "cv" in st.session_state:
            cv = st.session_state["cv"]
            summary = cv["summary"]
            st.markdown("#### Cross-validation results")
            st.caption(f"{SCHEME_LABELS[summary.scheme.iloc[0]]} · {summary.folds.iloc[0]} folds · {summary.rows_evaluated.iloc[0]} out-of-fold rows · {summary.n_features.iloc[0]} predictors. Each row is scored exactly once by a model that never saw its field/season blocks.")
            st.dataframe(summary, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(summary, x="model", y="RMSE_pooled", title="Pooled out-of-fold RMSE by model", text_auto=".3f"), width="stretch")
            oof = cv["oof_predictions"]
            chart = px.scatter(oof, x="actual_yield", y="Random Forest_prediction", color="season", hover_data=["field_id", "fold"], title="Random Forest: out-of-fold actual vs predicted")
            lo, hi = oof.actual_yield.min(), oof.actual_yield.max()
            chart.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
            st.plotly_chart(chart, width="stretch")
            with st.expander("Per-fold metrics"):
                st.dataframe(cv["fold_metrics"], hide_index=True, width="stretch")
            st.download_button("Download out-of-fold predictions (CSV)", oof.to_csv(index=False).encode("utf-8"), "pt355_oof_predictions.csv", "text/csv")
        if "featsets" in st.session_state:
            st.markdown("#### Feature-set comparison")
            st.caption(f"Same folds ({SCHEME_LABELS[st.session_state['featsets_scheme']]}) for every feature set. A difference between sets is only informative relative to the fold-to-fold spread.")
            st.dataframe(st.session_state["featsets"], hide_index=True, width="stretch")
        st.caption("MAE and RMSE are in the unit of the yield column you supplied. R² is unitless and can be negative.")
elif sat is not None or ctx is not None:
    st.info("Upload both input tables to build the joined field-season analysis table.")
else:
    st.markdown("### Input format\n\n**Satellite CSV/XLSX:** one row per field observation (date), with `field_id`, `season`, `observation_date`, `red_reflectance`, `nir_reflectance`.\n\n**Yield/weather CSV/XLSX:** one row per field and season, with `field_id`, `season`, `yield`, plus optional numeric seasonal weather fields.")
    st.markdown("Header-only templates are in `data/templates/`. Put your real files in `data/private/` (git-ignored) and upload them here, or run `python scripts/run_pipeline.py` for a headless run with an audit record.")
