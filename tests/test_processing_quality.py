"""Regression tests for data-handling defects found while auditing the prototype.

All data here are in-memory software fixtures, not study data.
"""
import pandas as pd
import pytest

from src.processing import build_seasonal_ndvi, fuse_with_report, fuse_seasonal_data, DataQualityReport


def sat(field="A", season="S1", n=3, red=0.2, nir=None):
    nir = nir or [0.6, 0.7, 0.8][:n]
    return pd.DataFrame({"field_id": [field] * n, "season": [season] * n,
                         "observation_date": pd.date_range("2023-01-01", periods=n, freq="30D"),
                         "red_reflectance": [red] * n, "nir_reflectance": nir})


def ctx(rows):
    return pd.DataFrame(rows, columns=["field_id", "season", "yield"])


def test_integer_and_string_field_ids_join():
    out = fuse_seasonal_data(sat(field=1), ctx([["1", "S1", 4.0]]))
    assert len(out) == 1


def test_whitespace_and_float_ids_are_normalized():
    satellite = sat(field=2.0)  # e.g. Excel turns 2 into 2.0
    out = fuse_seasonal_data(satellite, ctx([[" 2 ", " S1", 4.0]]))
    assert len(out) == 1 and out.field_id.iloc[0] == "2"


def test_header_case_and_spaces_are_standardized():
    satellite = sat().rename(columns={"field_id": " Field_ID ", "season": "SEASON"})
    assert len(fuse_seasonal_data(satellite, ctx([["A", "S1", 4.0]]))) == 1


def test_negative_reflectance_excluded_and_counted():
    data = pd.concat([sat(), pd.DataFrame({"field_id": ["A"], "season": ["S1"], "observation_date": ["2023-06-01"],
                                           "red_reflectance": [-0.1], "nir_reflectance": [-0.3]})])
    rep = DataQualityReport()
    out = build_seasonal_ndvi(data, report=rep)
    assert out.ndvi_observation_count.iloc[0] == 3
    assert rep.counts["satellite_rows_negative_reflectance"] == 1


def test_same_day_scenes_are_one_observation():
    data = pd.concat([sat(), sat().iloc[[0]]], ignore_index=True)
    rep = DataQualityReport()
    out = build_seasonal_ndvi(data, report=rep)
    assert out.ndvi_observation_count.iloc[0] == 3
    assert rep.counts["satellite_same_day_rows_merged"] == 1


def test_every_exclusion_is_reported():
    satellite = pd.concat([sat("A"), sat("B"), sat("C")], ignore_index=True)
    context = ctx([["A", "S1", 4.0], ["B", "S1", None], ["D", "S1", 5.0]])
    fused, rep = fuse_with_report(satellite, context)
    assert list(fused.field_id) == ["A"]
    c = rep.counts
    assert c["context_rows_in"] == 3
    assert c["context_rows_missing_or_nonnumeric_yield"] == 1
    assert c["field_seasons_with_imagery_but_no_yield_row"] == 2  # B (yield missing) and C
    assert c["yield_rows_without_usable_imagery"] == 1  # D
    assert ["C", "S1"] in rep.satellite_only_keys
    assert any("D" in key for key in rep.context_only_keys)
    assert rep.warnings


def test_duplicate_context_error_names_the_keys():
    with pytest.raises(ValueError, match=r"at most one row.*\['A', 'S1'\]"):
        fuse_seasonal_data(sat(), ctx([["A", "S1", 1.0], ["A", "S1", 2.0]]))


def test_no_match_error_shows_example_keys_from_both_sides():
    with pytest.raises(ValueError, match="2022/23"):
        fuse_seasonal_data(sat(season="2022-23"), ctx([["A", "2022/23", 3.0]]))


def test_peak_features_and_year_boundary():
    # Peak in January of the following year: calendar DOY is small, days-since-first is not.
    data = pd.DataFrame({"field_id": ["A"] * 3, "season": ["S"] * 3,
                         "observation_date": ["2022-11-15", "2023-01-10", "2023-03-01"],
                         "red_reflectance": [0.2, 0.05, 0.2], "nir_reflectance": [0.4, 0.8, 0.4]})
    row = build_seasonal_ndvi(data).iloc[0]
    assert row.peak_ndvi_day == 10
    assert row.peak_ndvi_days_since_first_obs == (pd.Timestamp("2023-01-10") - pd.Timestamp("2022-11-15")).days


def test_forecast_cutoff_excludes_later_observations():
    context = ctx([["A", "S1", 4.0]])
    context["forecast_cutoff_date"] = "2023-02-01"
    fused, rep = fuse_with_report(sat(), context)
    assert fused.ndvi_observation_count.iloc[0] == 2  # 2023-01-01 and 2023-01-31
    assert rep.counts["satellite_rows_after_forecast_cutoff"] == 1
    assert "forecast_cutoff_date" not in {c for c in fused.select_dtypes("number").columns}


def test_min_observations_exclusion_is_reported_not_silent():
    satellite = pd.concat([sat("A", n=3), sat("B", n=1, nir=[0.7])], ignore_index=True)
    context = ctx([["A", "S1", 4.0], ["B", "S1", 5.0]])
    fused, rep = fuse_with_report(satellite, context, min_observations=2)
    assert list(fused.field_id) == ["A"]
    assert rep.counts["field_seasons_below_min_observations"] == 1
    assert any("fewer than 2" in w for w in rep.warnings)


def test_leakage_like_column_and_dn_scale_warnings():
    satellite = sat(red=2000, nir=[6000, 7000, 8000])
    context = pd.DataFrame({"field_id": ["A"], "season": ["S1"], "yield": [4.0], "production_tonnes": [400.0]})
    _, rep = fuse_with_report(satellite, context)
    text = " ".join(rep.warnings)
    assert "target leakage" in text and "production_tonnes" in text
    assert "digital numbers" in text
