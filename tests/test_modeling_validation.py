"""Tests for validation schemes and feature-set comparison. Fixtures are software-only."""
import numpy as np
import pandas as pd
import pytest

from src.modeling import (available_schemes, compare_feature_sets, compare_models, cross_validate_models,
                          recommend_scheme, select_features)
from src.processing import fuse_seasonal_data
from tests.fixtures import make_fixture


@pytest.fixture(scope="module")
def fused():
    sat, ctx = make_fixture(n_fields=6, seasons=("S1", "S2", "S3"))
    return fuse_seasonal_data(sat, ctx)


def test_field_cv_every_row_scored_once_with_no_field_overlap(fused):
    res = cross_validate_models(fused, "field", n_splits=3)
    oof = res["oof_predictions"]
    assert len(oof) == len(fused)
    assert not oof.duplicated(["field_id", "season"]).any()
    # fields in different folds never share a fold with their own other-fold rows
    assert oof.groupby("field_id").fold.nunique().max() == 1
    assert {"MAE_pooled", "RMSE_pooled", "R2_pooled", "folds", "rows_evaluated"} <= set(res["summary"].columns)


def test_season_cv_holds_out_one_season_per_fold(fused):
    res = cross_validate_models(fused, "season")
    assert res["summary"].folds.iloc[0] == 3
    assert set(res["fold_metrics"].test_block) == {"season S1", "season S2", "season S3"}


def test_spatiotemporal_blocking_trains_on_other_fields_and_seasons(fused):
    res = cross_validate_models(fused, "spatiotemporal", n_splits=3)  # internal asserts check disjointness
    oof = res["oof_predictions"]
    assert len(oof) == len(fused) and not oof.duplicated(["field_id", "season"]).any()
    assert res["fold_metrics"].train_rows.max() < len(fused)


def test_scheme_requirements_are_enforced():
    sat, ctx = make_fixture(n_fields=6, seasons=("S1",))
    one_season = fuse_seasonal_data(sat, ctx)
    assert available_schemes(one_season)["season"] != "ok"
    assert recommend_scheme(one_season) == "field"
    with pytest.raises(ValueError, match="at least 2 distinct seasons"):
        cross_validate_models(one_season, "season")
    sat, ctx = make_fixture(n_fields=1, seasons=("S1", "S2"))
    one_field = fuse_seasonal_data(sat, ctx)
    assert recommend_scheme(one_field) == "season"
    with pytest.raises(ValueError, match="at least 2 distinct fields"):
        cross_validate_models(one_field, "field")


def test_single_field_holdout_is_labelled_weaker():
    sat, ctx = make_fixture(n_fields=1, seasons=tuple(f"S{i}" for i in range(8)))
    metrics, *_ = compare_models(fuse_seasonal_data(sat, ctx))
    assert "random holdout" in metrics.split.iloc[0] and "weaker" in metrics.split.iloc[0]


def test_holdout_has_no_field_overlap_and_reports_sizes(fused):
    metrics, preds, _, feats = compare_models(fused)
    test_fields = set(preds.field_id)
    assert metrics.test_fields.iloc[0] == len(test_fields)
    assert metrics.train_fields.iloc[0] + metrics.test_fields.iloc[0] == fused.field_id.nunique()
    assert metrics.train_rows.iloc[0] + metrics.test_rows.iloc[0] == len(fused)


def test_feature_sets_are_disjoint_and_comparison_runs(fused):
    sat_cols, weather_cols = select_features(fused, feature_set="satellite"), select_features(fused, feature_set="weather")
    assert "rain_mm" in weather_cols and "rain_mm" not in sat_cols and "ndvi_mean" in sat_cols
    assert not set(sat_cols) & set(weather_cols)
    table = compare_feature_sets(fused, "field", n_splits=3)
    assert set(table.feature_set) == {"satellite", "weather", "combined"}
    nf = table.drop_duplicates("feature_set").set_index("feature_set").n_features
    assert nf["satellite"] == len(sat_cols) and nf["weather"] == len(weather_cols)


def test_missing_predictors_are_imputed_in_training_fold_and_counted(fused):
    data = fused.copy()
    data.loc[0, "rain_mm"] = np.nan
    metrics, *_ = compare_models(data)
    assert metrics.rows_with_imputed_predictors.iloc[0] == 1  # kept, not silently dropped


def test_all_missing_predictor_is_dropped_and_recorded(fused):
    data = fused.copy()
    data["empty_weather"] = np.nan
    res = cross_validate_models(data, "field", n_splits=3)
    assert res["notes"]["all_missing_predictors_dropped"] == ["empty_weather"]
    assert "empty_weather" not in res["features"]


def test_exclude_removes_columns(fused):
    _, _, _, feats = compare_models(fused, exclude=["rain_mm"])
    assert "rain_mm" not in feats
