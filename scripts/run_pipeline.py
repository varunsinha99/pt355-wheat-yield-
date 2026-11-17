"""Run the full PT355 pipeline on your real data and write an auditable run record.

    python scripts/run_pipeline.py --satellite data/private/satellite_observations.csv \
        --context data/private/yield_weather.csv --out-dir outputs/run1

Outputs (all under --out-dir, which is git-ignored by default):
  fused_features.csv            field-season NDVI features joined to yield/weather
  data_quality_report.json      every exclusion, unmatched key and warning
  holdout_metrics.csv           single grouped holdout (MAE, RMSE, R2, sample sizes)
  cv_summary.csv / cv_folds.csv grouped cross-validation (scheme chosen automatically
                                unless --scheme is given)
  feature_set_comparison.csv    satellite-only vs weather-only vs combined
  RUN_RECORD.md                 inputs (SHA-256), versions, split design, measured numbers

The script computes nothing without your input files and never generates data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling import (SCHEMES, available_schemes, compare_feature_sets, compare_models,  # noqa: E402
                          cross_validate_models, recommend_scheme, xgboost_available)
from src.processing import fuse_with_report, suspected_leakage_columns  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    return df.round(4).to_markdown(index=False) if hasattr(df, "to_markdown") and _has_tabulate() else "```\n" + df.round(4).to_string(index=False) + "\n```"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--satellite", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--scheme", choices=("auto",) + SCHEMES[1:], default="auto")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--min-observations", type=int, default=1, help="Exclude field-seasons with fewer valid observations (reported).")
    ap.add_argument("--exclude", nargs="*", default=[], help="Predictor columns to exclude (e.g. target-like or coordinate columns).")
    ap.add_argument("--keep-suspected-leakage", action="store_true", help="Do not auto-exclude columns named like yield/production/harvest.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sat, ctx = read_table(args.satellite), read_table(args.context)
    fused, report = fuse_with_report(sat, ctx, min_observations=args.min_observations)
    fused.to_csv(out / "fused_features.csv", index=False)
    (out / "data_quality_report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))

    exclude = list(args.exclude)
    auto_excluded = [] if args.keep_suspected_leakage else suspected_leakage_columns(fused)
    exclude += [c for c in auto_excluded if c not in exclude]

    scheme = recommend_scheme(fused) if args.scheme == "auto" else args.scheme
    avail = available_schemes(fused)
    if scheme != "holdout" and avail[scheme] != "ok":
        raise SystemExit(f"Scheme '{scheme}' is not possible with this data: {avail[scheme]}.")

    holdout, _, _, features = compare_models(fused, test_size=args.test_size, random_state=args.seed, exclude=exclude)
    holdout.to_csv(out / "holdout_metrics.csv", index=False)

    cv_summary = cv_folds = fs_table = None
    if scheme != "holdout":
        cv = cross_validate_models(fused, scheme, n_splits=args.n_splits, random_state=args.seed, exclude=exclude)
        cv_summary, cv_folds = cv["summary"], cv["fold_metrics"]
        cv_summary.to_csv(out / "cv_summary.csv", index=False)
        cv_folds.to_csv(out / "cv_folds.csv", index=False)
        cv["oof_predictions"].to_csv(out / "cv_out_of_fold_predictions.csv", index=False)
        fs_table = compare_feature_sets(fused, scheme, n_splits=args.n_splits, random_state=args.seed, exclude=exclude)
        fs_table.to_csv(out / "feature_set_comparison.csv", index=False)

    import sklearn
    lines = [
        "# RUN RECORD (generated; numbers below were measured on the supplied files)",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Python {platform.python_version()}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, XGBoost {'installed' if xgboost_available() else 'NOT installed (not evaluated)'}",
        f"- Satellite file: `{args.satellite.name}` SHA-256 `{sha256(args.satellite)}`",
        f"- Context file: `{args.context.name}` SHA-256 `{sha256(args.context)}`",
        f"- Fused field-seasons: {len(fused)}; distinct fields: {fused.field_id.nunique()}; distinct seasons: {fused.season.nunique()}",
        f"- Yield summary (units must be documented by the team): {report.yield_summary}",
        f"- Predictors used ({len(features)}): {', '.join(features)}",
        f"- Auto-excluded as suspected target leakage: {auto_excluded or 'none'}; user-excluded: {list(args.exclude) or 'none'}",
        f"- Validation scheme for cross-validation: {scheme} (available: {avail})",
        "", "## Data-quality counts", "```", json.dumps(report.counts, indent=2), "```", "## Warnings",
        *(f"- {w}" for w in (report.warnings or ["none"])),
        "", "## Single grouped holdout", md_table(holdout),
    ]
    if cv_summary is not None:
        lines += ["", f"## Cross-validation ({scheme})", md_table(cv_summary), "", "## Feature-set comparison (same folds)", md_table(fs_table)]
    else:
        lines += ["", "## Cross-validation", "Not run: the data do not support a grouped scheme (see availability above). Treat holdout scores as weak evidence."]
    lines += ["", "Interpretation limits: scores describe only these files and this split design. Small samples give unstable estimates; no confidence intervals are computed."]
    (out / "RUN_RECORD.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote results to {out}/ (see RUN_RECORD.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
