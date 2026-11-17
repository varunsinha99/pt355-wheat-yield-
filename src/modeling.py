"""Leakage-conscious regression training and evaluation.

Validation schemes (all keep preprocessing inside the training fold):

* ``holdout``        - one field-grouped holdout (random row holdout only if <2 fields).
* ``field``          - GroupKFold by field_id: a field is never in train and test together.
* ``season``         - leave-one-season-out: a season is never in train and test together.
* ``spatiotemporal`` - blocked: a test cell is (held-out season x held-out field fold) and
                       training uses only rows from *other* seasons AND *other* fields.

No scheme can remove every dependency (e.g. neighbouring fields), so report the
scheme and sample sizes with every score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import make_pipeline

from .processing import NDVI_FEATURES, candidate_predictors

SCHEMES = ("holdout", "field", "season", "spatiotemporal")
MIN_TRAIN_ROWS = 3


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def build_models(random_state: int = 42, n_estimators: int = 300) -> dict:
    models = {
        "Mean baseline": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean")),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    }
    if xgboost_available():
        from xgboost import XGBRegressor
        models["XGBoost"] = make_pipeline(SimpleImputer(strategy="median"), XGBRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=random_state, n_jobs=2))
    return models


def select_features(frame: pd.DataFrame, target: str = "yield", feature_set: str = "combined", exclude=()) -> list[str]:
    """``satellite`` = NDVI features only, ``weather`` = other numeric context columns only."""
    cands = [c for c in candidate_predictors(frame, target) if c not in set(exclude)]
    ndvi = [c for c in cands if c in NDVI_FEATURES]
    other = [c for c in cands if c not in NDVI_FEATURES]
    if feature_set == "satellite":
        return ndvi
    if feature_set == "weather":
        return other
    if feature_set == "combined":
        return cands
    raise ValueError(f"Unknown feature_set '{feature_set}'.")


def _prepare(frame: pd.DataFrame, target: str, features: list[str] | None, exclude) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], dict]:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    cols = list(features) if features is not None else select_features(frame, target, "combined", exclude)
    if not cols:
        raise ValueError("No numeric predictor columns are available.")
    y_all = pd.to_numeric(frame[target], errors="coerce")
    X_all = frame[cols].apply(pd.to_numeric, errors="coerce")
    have_y = y_all.notna()
    note = {"rows_without_target_dropped": int((~have_y).sum())}
    X_all = X_all.loc[have_y]
    all_nan = [c for c in cols if X_all[c].isna().all()]
    if all_nan:  # an imputer cannot fill a column with no observed value
        X_all = X_all.drop(columns=all_nan)
        cols = [c for c in cols if c not in all_nan]
    note["all_missing_predictors_dropped"] = all_nan
    note["rows_with_imputed_predictors"] = int(X_all.isna().any(axis=1).sum())
    if not cols:
        raise ValueError("Every predictor column is entirely missing.")
    usable = frame.loc[have_y].reset_index(drop=True)
    return X_all.reset_index(drop=True), y_all.loc[have_y].reset_index(drop=True), usable, cols, note


def _metrics(y_true, pred) -> dict:
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else float("nan"),
    }


def _groups(usable: pd.DataFrame, column: str) -> pd.Series | None:
    if column in usable.columns and usable[column].nunique() >= 2:
        return usable[column]
    return None


def compare_models(frame: pd.DataFrame, target: str = "yield", group_column: str = "field_id", test_size: float = 0.25,
                   random_state: int = 42, feature_columns: list[str] | None = None, exclude=()):
    """Single holdout comparison. Returns (metrics, predictions, fitted models, feature names)."""
    X, y, usable, features, note = _prepare(frame, target, feature_columns, exclude)
    if len(X) < 4:
        raise ValueError("At least 4 rows with a numeric target are required for a holdout evaluation.")
    groups = _groups(usable, group_column)
    if groups is not None:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(X, y, groups))
        split = "field-group holdout"
        overlap = len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
        assert overlap == 0, "group leakage between train and test"
    else:
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size, random_state=random_state)
        split, overlap = "random holdout (fewer than 2 field groups; weaker evidence)", None
    models = build_models(random_state)
    keys = [c for c in ["field_id", "season"] if c in usable.columns]
    pred_frame = usable.loc[test_idx, keys].copy()
    pred_frame["actual_yield"] = y.iloc[test_idx].to_numpy()
    rows, fitted = [], {}
    for name, est in models.items():
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        rows.append({"model": name, **_metrics(y.iloc[test_idx], pred),
                     "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                     "train_fields": int(groups.iloc[train_idx].nunique()) if groups is not None else None,
                     "test_fields": int(groups.iloc[test_idx].nunique()) if groups is not None else None,
                     "n_features": len(features), "rows_with_imputed_predictors": note["rows_with_imputed_predictors"], "split": split})
        pred_frame[f"{name}_prediction"] = est.predict(X.iloc[test_idx])
        fitted[name] = est
    return pd.DataFrame(rows), pred_frame.reset_index(drop=True), fitted, features


def available_schemes(frame: pd.DataFrame) -> dict[str, str]:
    """Which validation schemes the data can support, with a reason when it cannot."""
    nf = frame["field_id"].nunique() if "field_id" in frame else 0
    ns = frame["season"].nunique() if "season" in frame else 0
    out = {"holdout": "ok"}
    out["field"] = "ok" if nf >= 3 else "needs at least 3 distinct fields"
    out["season"] = "ok" if ns >= 2 else "needs at least 2 distinct seasons"
    out["spatiotemporal"] = "ok" if (nf >= 3 and ns >= 2) else "needs at least 3 fields and 2 seasons"
    return out


def recommend_scheme(frame: pd.DataFrame) -> str:
    avail = available_schemes(frame)
    for scheme in ("spatiotemporal", "field", "season"):
        if avail[scheme] == "ok":
            return scheme
    return "holdout"


def _splits(usable: pd.DataFrame, scheme: str, n_splits: int):
    """Yield (fold_label, train_idx, test_idx, test_description)."""
    n = len(usable)
    if scheme == "field":
        g = usable["field_id"]
        k = min(n_splits, g.nunique())
        if k < 2:
            raise ValueError("Field-grouped CV needs at least 2 distinct fields.")
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(n), groups=g)):
            yield i + 1, tr, te, f"{g.iloc[te].nunique()} fields"
    elif scheme == "season":
        g = usable["season"]
        if g.nunique() < 2:
            raise ValueError("Season-held-out CV needs at least 2 distinct seasons.")
        for i, (tr, te) in enumerate(LeaveOneGroupOut().split(np.zeros(n), groups=g)):
            yield i + 1, tr, te, f"season {g.iloc[te].iloc[0]}"
    elif scheme == "spatiotemporal":
        fields, seasons = usable["field_id"], usable["season"]
        if fields.nunique() < 2 or seasons.nunique() < 2:
            raise ValueError("Spatio-temporal blocking needs at least 2 fields and 2 seasons.")
        k = min(n_splits, fields.nunique())
        fold_of = {}
        for i, (_, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(n), groups=fields)):
            for f in fields.iloc[te].unique():
                fold_of[f] = i
        field_fold = fields.map(fold_of)
        label = 0
        for s in sorted(seasons.unique()):
            for f in sorted(set(fold_of.values())):
                test = ((seasons == s) & (field_fold == f)).to_numpy()
                train = ((seasons != s) & (field_fold != f)).to_numpy()
                if test.sum() < 1 or train.sum() < MIN_TRAIN_ROWS:
                    continue  # counted by the caller as skipped
                label += 1
                yield label, np.flatnonzero(train), np.flatnonzero(test), f"season {s} x field-fold {f + 1}"
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from {SCHEMES[1:]}.")


def cross_validate_models(frame: pd.DataFrame, scheme: str = "field", target: str = "yield", n_splits: int = 5,
                          random_state: int = 42, feature_columns: list[str] | None = None, exclude=()) -> dict:
    """Grouped cross-validation. Returns fold metrics, pooled summary and out-of-fold predictions."""
    if scheme == "holdout":
        raise ValueError("Use compare_models() for the single holdout.")
    X, y, usable, features, note = _prepare(frame, target, feature_columns, exclude)
    fold_rows, oof_parts, n_folds = [], [], 0
    pooled = {}
    for label, tr, te, desc in _splits(usable, scheme, n_splits):
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        # The leakage guarantee, asserted rather than assumed.
        if scheme in ("field", "spatiotemporal"):
            assert not set(usable["field_id"].iloc[tr]) & set(usable["field_id"].iloc[te]), "field leakage"
        if scheme in ("season", "spatiotemporal"):
            assert not set(usable["season"].iloc[tr]) & set(usable["season"].iloc[te]), "season leakage"
        n_folds += 1
        part = usable.loc[te, [c for c in ["field_id", "season"] if c in usable.columns]].copy()
        part["fold"], part["actual_yield"] = label, y.iloc[te].to_numpy()
        for name, est in build_models(random_state).items():
            est.fit(X.iloc[tr], y.iloc[tr])
            pred = est.predict(X.iloc[te])
            fold_rows.append({"model": name, "fold": label, "test_block": desc, "train_rows": len(tr), "test_rows": len(te), **_metrics(y.iloc[te], pred)})
            part[f"{name}_prediction"] = pred
        oof_parts.append(part)
    if n_folds == 0:
        raise ValueError("No valid folds could be formed from this data for the chosen scheme.")
    if scheme == "spatiotemporal":
        planned = usable["season"].nunique() * min(n_splits, usable["field_id"].nunique())
        note["spatiotemporal_cells_skipped"] = int(planned - n_folds)  # empty test cell or too little training data
    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    summary = []
    for name in folds["model"].unique():
        pooled_m = _metrics(oof["actual_yield"], oof[f"{name}_prediction"])
        sub = folds[folds.model == name]
        summary.append({"model": name, **{f"{k}_pooled": v for k, v in pooled_m.items()},
                        "MAE_fold_mean": float(sub.MAE.mean()), "MAE_fold_sd": float(sub.MAE.std(ddof=1)) if len(sub) > 1 else float("nan"),
                        "folds": int(len(sub)), "rows_evaluated": int(len(oof)), "n_features": len(features), "scheme": scheme,
                        "rows_with_imputed_predictors": note["rows_with_imputed_predictors"]})
    return {"fold_metrics": folds, "summary": pd.DataFrame(summary), "oof_predictions": oof, "features": features, "notes": note}


def compare_feature_sets(frame: pd.DataFrame, scheme: str, target: str = "yield", n_splits: int = 5,
                         random_state: int = 42, exclude=()) -> pd.DataFrame:
    """Same scheme and folds for satellite-only, weather-only and combined predictors."""
    out = []
    for fs in ("satellite", "weather", "combined"):
        cols = select_features(frame, target, fs, exclude)
        if not cols:
            out.append({"feature_set": fs, "note": "no columns of this kind in the fused table"})
            continue
        res = cross_validate_models(frame, scheme, target, n_splits, random_state, feature_columns=cols)
        s = res["summary"].copy()
        s.insert(0, "feature_set", fs)
        out.append(s)
    frames = [o if isinstance(o, pd.DataFrame) else pd.DataFrame([o]) for o in out]
    return pd.concat(frames, ignore_index=True)
