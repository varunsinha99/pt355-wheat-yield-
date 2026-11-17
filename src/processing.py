"""Input validation, NDVI feature engineering and data fusion for PT355.

Design rule: nothing is dropped without being counted. Every exclusion is
recorded in a :class:`DataQualityReport` so it can be shown in the app and
copied into the project report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SATELLITE_REQUIRED = {"field_id", "season", "observation_date", "red_reflectance", "nir_reflectance"}
CONTEXT_REQUIRED = {"field_id", "season", "yield"}
JOIN_KEYS = ["field_id", "season"]
CUTOFF_COLUMN = "forecast_cutoff_date"  # optional context column, see build_seasonal_ndvi

NDVI_FEATURES = [
    "ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_std", "ndvi_amplitude",
    "peak_ndvi_day", "peak_ndvi_days_since_first_obs", "ndvi_observation_count",
]
# Never used as predictors.
NON_PREDICTORS = {"field_id", "season", "first_observation_date", "last_observation_date", CUTOFF_COLUMN}
LOW_COVERAGE_WARN = 3  # field-seasons with fewer valid observations get a warning (not dropped)
# Column names that often encode the target (e.g. production = yield x area).
LEAKAGE_NAME_PATTERN = re.compile(r"yield|production|harvest|output", re.IGNORECASE)


@dataclass
class DataQualityReport:
    """Counts of everything checked, excluded, or left unmatched."""

    counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    satellite_only_keys: list = field(default_factory=list)  # field-seasons with imagery but no yield row
    context_only_keys: list = field(default_factory=list)  # yield rows with no usable imagery
    yield_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "warnings": self.warnings,
            "satellite_only_keys": self.satellite_only_keys,
            "context_only_keys": self.context_only_keys,
            "yield_summary": self.yield_summary,
        }


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def standardize_columns(frame: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    """Strip header whitespace and match required column names case-insensitively."""
    lookup = {name.lower(): name for name in required | {CUTOFF_COLUMN}}
    renamed = {}
    for col in frame.columns:
        stripped = str(col).strip()
        renamed[col] = lookup.get(stripped.lower(), stripped)
    return frame.rename(columns=renamed)


def _clean_key(value):
    """Canonical text form of an identifier: 1, 1.0 and ' 1 ' all become '1'."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return pd.NA
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else pd.NA


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for key in JOIN_KEYS:
        out[key] = pd.Series([_clean_key(v) for v in out[key]], index=out.index, dtype="object")
    return out


def _sample(keys, n=20):
    return [list(k) for k in keys[:n]]


def _prepare_observations(satellite: pd.DataFrame, cutoffs: pd.DataFrame | None, counts: dict) -> pd.DataFrame:
    """Row-level NDVI with every exclusion reason counted. Returns valid rows."""
    frame = standardize_columns(satellite, SATELLITE_REQUIRED)
    validate_columns(frame, SATELLITE_REQUIRED, "Satellite observations")
    frame = normalize_keys(frame)
    counts["satellite_rows_in"] = int(len(frame))

    missing_key = frame[JOIN_KEYS].isna().any(axis=1)
    counts["satellite_rows_missing_field_or_season"] = int(missing_key.sum())
    frame = frame.loc[~missing_key].copy()

    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    for band in ("red_reflectance", "nir_reflectance"):
        frame[band] = pd.to_numeric(frame[band], errors="coerce")
    bad_date = frame["observation_date"].isna()
    bad_band = frame[["red_reflectance", "nir_reflectance"]].isna().any(axis=1) & ~bad_date
    counts["satellite_rows_invalid_date"] = int(bad_date.sum())
    counts["satellite_rows_missing_or_nonnumeric_reflectance"] = int(bad_band.sum())
    frame = frame.loc[~bad_date & ~bad_band].copy()

    negative = (frame["red_reflectance"] < 0) | (frame["nir_reflectance"] < 0)
    counts["satellite_rows_negative_reflectance"] = int(negative.sum())
    frame = frame.loc[~negative].copy()

    denom = frame["nir_reflectance"] + frame["red_reflectance"]
    zero = denom == 0
    counts["satellite_rows_zero_denominator"] = int(zero.sum())
    frame = frame.loc[~zero].copy()
    frame["ndvi"] = (frame["nir_reflectance"] - frame["red_reflectance"]) / (frame["nir_reflectance"] + frame["red_reflectance"])
    out_of_range = ~frame["ndvi"].between(-1, 1, inclusive="both")
    counts["satellite_rows_ndvi_out_of_range"] = int(out_of_range.sum())
    frame = frame.loc[~out_of_range].copy()

    counts["satellite_rows_after_forecast_cutoff"] = 0
    if cutoffs is not None:
        frame = frame.merge(cutoffs, on=JOIN_KEYS, how="left")
        late = frame[CUTOFF_COLUMN].notna() & (frame["observation_date"] > frame[CUTOFF_COLUMN])
        counts["satellite_rows_after_forecast_cutoff"] = int(late.sum())
        frame = frame.loc[~late].drop(columns=[CUTOFF_COLUMN])

    # Several scenes on one date (overlapping tiles) are one observation, not two.
    before = len(frame)
    frame = (frame.groupby(JOIN_KEYS + ["observation_date"], as_index=False, dropna=False)
             .agg(ndvi=("ndvi", "mean"), red_reflectance=("red_reflectance", "mean"), nir_reflectance=("nir_reflectance", "mean"),
                  scenes_merged=("ndvi", "size")))
    counts["satellite_same_day_rows_merged"] = int(before - len(frame))
    counts["satellite_observations_used"] = int(len(frame))
    return frame


def _aggregate(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values(JOIN_KEYS + ["observation_date"]).reset_index(drop=True)
    obs["doy"] = obs["observation_date"].dt.day_of_year
    grouped = obs.groupby(JOIN_KEYS, dropna=False)
    feats = grouped.agg(
        ndvi_mean=("ndvi", "mean"), ndvi_max=("ndvi", "max"), ndvi_min=("ndvi", "min"),
        ndvi_std=("ndvi", "std"), ndvi_observation_count=("ndvi", "count"),
        first_observation_date=("observation_date", "min"), last_observation_date=("observation_date", "max"),
    ).reset_index()
    peak = obs.loc[grouped["ndvi"].idxmax(), JOIN_KEYS + ["doy", "observation_date"]].rename(
        columns={"doy": "peak_ndvi_day", "observation_date": "_peak_date"})
    feats = feats.merge(peak, on=JOIN_KEYS, how="left", validate="one_to_one")
    feats["peak_ndvi_days_since_first_obs"] = (feats["_peak_date"] - feats["first_observation_date"]).dt.days
    feats = feats.drop(columns="_peak_date")
    feats["ndvi_amplitude"] = feats["ndvi_max"] - feats["ndvi_min"]
    feats["ndvi_std"] = feats["ndvi_std"].fillna(0.0)  # single observation: no spread
    return feats


def build_seasonal_ndvi(satellite: pd.DataFrame, cutoffs: pd.DataFrame | None = None, report: DataQualityReport | None = None) -> pd.DataFrame:
    """Compute per-observation NDVI, then aggregate valid observations to field-season.

    Excluded (and counted in ``report.counts``): missing keys, unparseable dates,
    missing/non-numeric or negative reflectance, zero denominators, NDVI outside
    [-1, 1], and observations after an optional per-field-season forecast cutoff.
    Multiple scenes on one date are averaged into one observation.

    ``peak_ndvi_day`` is the calendar day-of-year of peak NDVI. It wraps at New
    Year, so for seasons that span two calendar years prefer
    ``peak_ndvi_days_since_first_obs``.
    """
    counts = report.counts if report is not None else {}
    obs = _prepare_observations(satellite, cutoffs, counts)
    if obs.empty:
        raise ValueError("No valid satellite observations remain after date, reflectance and NDVI checks.")
    return _aggregate(obs)


def fuse_with_report(satellite: pd.DataFrame, context: pd.DataFrame, min_observations: int = 1) -> tuple[pd.DataFrame, DataQualityReport]:
    """Join seasonal NDVI features to yield/weather rows and report every exclusion.

    ``min_observations`` (default 1 = keep all) excludes field-seasons with fewer
    valid observations; the exclusions are listed in the report, never silent.
    """
    report = DataQualityReport()
    ctx = standardize_columns(context, CONTEXT_REQUIRED)
    validate_columns(ctx, CONTEXT_REQUIRED, "Yield/weather table")
    ctx = normalize_keys(ctx)
    report.counts["context_rows_in"] = int(len(ctx))

    missing_key = ctx[JOIN_KEYS].isna().any(axis=1)
    report.counts["context_rows_missing_field_or_season"] = int(missing_key.sum())
    ctx = ctx.loc[~missing_key].copy()

    dup_mask = ctx.duplicated(JOIN_KEYS, keep=False)
    if dup_mask.any():
        dup_keys = ctx.loc[dup_mask, JOIN_KEYS].drop_duplicates().values.tolist()
        raise ValueError(
            "Yield/weather table must contain at most one row per field_id and season. "
            f"{len(dup_keys)} duplicated key(s), e.g. {dup_keys[:5]}")

    ctx["yield"] = pd.to_numeric(ctx["yield"], errors="coerce")
    bad_yield = ctx["yield"].isna()
    report.counts["context_rows_missing_or_nonnumeric_yield"] = int(bad_yield.sum())
    dropped_yield_keys = ctx.loc[bad_yield, JOIN_KEYS].values.tolist()
    ctx = ctx.loc[~bad_yield].copy()
    if ctx.empty:
        raise ValueError("The yield/weather table has no rows with a numeric yield.")

    cutoffs = None
    if CUTOFF_COLUMN in ctx.columns:
        ctx[CUTOFF_COLUMN] = pd.to_datetime(ctx[CUTOFF_COLUMN], errors="coerce")
        cutoffs = ctx[JOIN_KEYS + [CUTOFF_COLUMN]]
        report.counts["context_rows_missing_cutoff_date"] = int(ctx[CUTOFF_COLUMN].isna().sum())

    seasonal = build_seasonal_ndvi(satellite, cutoffs=cutoffs, report=report)
    report.counts["satellite_field_seasons_with_valid_data"] = int(len(seasonal))

    if min_observations > 1:
        thin = seasonal["ndvi_observation_count"] < min_observations
        report.counts["field_seasons_below_min_observations"] = int(thin.sum())
        report.warnings.append(f"{int(thin.sum())} field-season(s) excluded: fewer than {min_observations} valid observations.")
        thin_keys = seasonal.loc[thin, JOIN_KEYS].values.tolist()
        seasonal = seasonal.loc[~thin].copy()
    else:
        thin_keys = []

    sat_keys = set(map(tuple, seasonal[JOIN_KEYS].values.tolist()))
    ctx_keys = set(map(tuple, ctx[JOIN_KEYS].values.tolist()))
    sat_only, ctx_only = sorted(sat_keys - ctx_keys), sorted(ctx_keys - sat_keys)
    report.satellite_only_keys = _sample(sat_only)
    report.context_only_keys = _sample(sorted(ctx_only)) + [list(k) + ["(missing yield)"] for k in dropped_yield_keys[:10]]
    report.counts["field_seasons_with_imagery_but_no_yield_row"] = len(sat_only)
    report.counts["yield_rows_without_usable_imagery"] = len(ctx_only)
    report.counts["thin_coverage_keys_excluded"] = len(thin_keys)

    merged = ctx.merge(seasonal, on=JOIN_KEYS, how="inner", validate="one_to_one")
    report.counts["fused_field_seasons"] = int(len(merged))
    if merged.empty:
        raise ValueError(
            "No matching field_id/season keys were found across the two input files. "
            f"Example satellite keys: {_sample(sorted(sat_keys), 3)}; example yield keys: {_sample(sorted(ctx_keys), 3)}. "
            "Check identifier and season label formats (e.g. '2022-23' vs '2022/23').")

    low = merged["ndvi_observation_count"] < LOW_COVERAGE_WARN
    report.counts[f"fused_field_seasons_with_fewer_than_{LOW_COVERAGE_WARN}_observations"] = int(low.sum())
    if low.any():
        report.warnings.append(f"{int(low.sum())} fused field-season(s) have fewer than {LOW_COVERAGE_WARN} valid observations; their seasonal statistics are weak.")
    if sat_only:
        report.warnings.append(f"{len(sat_only)} field-season(s) have imagery but no yield row and are not in the fused table.")
    if ctx_only:
        report.warnings.append(f"{len(ctx_only)} yield row(s) have no usable imagery and are not in the fused table.")
    if report.counts["context_rows_missing_or_nonnumeric_yield"]:
        report.warnings.append(f"{report.counts['context_rows_missing_or_nonnumeric_yield']} yield row(s) excluded: missing or non-numeric yield.")

    y = merged["yield"]
    report.yield_summary = {"min": float(y.min()), "median": float(y.median()), "max": float(y.max()), "n": int(len(y)), "non_positive": int((y <= 0).sum())}
    if report.yield_summary["non_positive"]:
        report.warnings.append(f"{report.yield_summary['non_positive']} fused row(s) have yield <= 0; verify units and missing-value coding.")

    # Scale heuristics use the raw input, before any exclusion.
    raw = standardize_columns(satellite, SATELLITE_REQUIRED)
    red_max = pd.to_numeric(raw["red_reflectance"], errors="coerce").max()
    nir_max = pd.to_numeric(raw["nir_reflectance"], errors="coerce").max()
    if pd.notna(red_max) and pd.notna(nir_max):
        report.counts["max_red_reflectance"] = float(red_max)
        report.counts["max_nir_reflectance"] = float(nir_max)
        if max(red_max, nir_max) > 1.5:
            report.warnings.append("Reflectance values exceed 1.5, so they look like digital numbers (e.g. 0-10000), not 0-1 reflectance. NDVI is unaffected if both bands share one scale, but confirm the scale and any processing-baseline offset.")
        if min(red_max, nir_max) > 0 and max(red_max, nir_max) / min(red_max, nir_max) > 50:
            report.warnings.append("Red and NIR maxima differ by more than 50x: the bands may be on different scales.")

    leak_like = [c for c in merged.columns if c != "yield" and LEAKAGE_NAME_PATTERN.search(c)]
    if leak_like:
        report.warnings.append(f"Possible target leakage: column(s) {leak_like} look like yield/production/harvest outputs. Exclude them from the predictors unless they are known before harvest.")
    return merged, report


def fuse_seasonal_data(satellite: pd.DataFrame, context: pd.DataFrame, min_observations: int = 1) -> pd.DataFrame:
    """Return only the fused table (see :func:`fuse_with_report` for the audit trail)."""
    return fuse_with_report(satellite, context, min_observations)[0]


def candidate_predictors(frame: pd.DataFrame, target: str = "yield") -> list[str]:
    """Numeric columns that may be used as predictors (identifiers/dates/target excluded)."""
    return [c for c in frame.columns if c != target and c not in NON_PREDICTORS and pd.api.types.is_numeric_dtype(frame[c])]


def suspected_leakage_columns(frame: pd.DataFrame, target: str = "yield") -> list[str]:
    return [c for c in candidate_predictors(frame, target) if LEAKAGE_NAME_PATTERN.search(c)]
