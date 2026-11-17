import pandas as pd
import pytest
from src.processing import build_seasonal_ndvi, fuse_seasonal_data
from src.modeling import compare_models


def sat_rows():
    return pd.DataFrame({
        "field_id": ["A", "A", "B", "B"], "season": ["S1"] * 4,
        "observation_date": ["2023-01-01", "2023-02-01", "2023-01-02", "2023-02-02"],
        "red_reflectance": [0.2, 0.1, 0.3, 0.15], "nir_reflectance": [0.6, 0.7, 0.5, 0.65],
    })


def test_ndvi_and_seasonal_features():
    out = build_seasonal_ndvi(sat_rows())
    assert len(out) == 2
    assert out.loc[out.field_id == "A", "ndvi_mean"].iloc[0] == pytest.approx((0.5 + 0.75) / 2)
    assert out.loc[out.field_id == "A", "ndvi_observation_count"].iloc[0] == 2
    assert out.loc[out.field_id == "A", "ndvi_amplitude"].iloc[0] == pytest.approx(0.25)


def test_zero_denominator_and_bad_date_are_excluded():
    data = pd.DataFrame({"field_id": ["A", "A"], "season": ["S1", "S1"], "observation_date": ["2023-01-01", "bad"], "red_reflectance": [0, 0.2], "nir_reflectance": [0, 0.5]})
    with pytest.raises(ValueError, match="No valid"):
        build_seasonal_ndvi(data)


def test_required_columns_are_checked():
    with pytest.raises(ValueError, match="missing required"):
        build_seasonal_ndvi(pd.DataFrame({"field_id": []}))


def test_fusion_and_duplicate_context_rejected():
    context = pd.DataFrame({"field_id": ["A", "B"], "season": ["S1", "S1"], "yield": [4.0, 5.0], "rain_mm": [120, 150]})
    joined = fuse_seasonal_data(sat_rows(), context)
    assert len(joined) == 2 and "rain_mm" in joined
    duplicate = pd.concat([context, context.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="at most one row"):
        fuse_seasonal_data(sat_rows(), duplicate)


def test_no_matching_keys_rejected():
    context = pd.DataFrame({"field_id": ["Z"], "season": ["S9"], "yield": [3.0]})
    with pytest.raises(ValueError, match="No matching"):
        fuse_seasonal_data(sat_rows(), context)


def test_model_evaluation_outputs_real_metric_fields():
    # Deterministic in-memory fixture verifies code behavior only; it is not a study dataset.
    rows = []
    for i in range(12):
        rows.append({"field_id": f"F{i}", "season": "S1", "yield": float(2 + i), "ndvi_mean": 0.2 + 0.03 * i, "rain_mm": 100 + i})
    metrics, preds, _, features = compare_models(pd.DataFrame(rows), random_state=3)
    assert {"MAE", "RMSE", "R2", "train_rows", "test_rows", "split"}.issubset(metrics.columns)
    assert "Random Forest" in set(metrics.model)
    assert len(preds) > 0 and "ndvi_mean" in features
    assert "field-group holdout" in set(metrics.split)
