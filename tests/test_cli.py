"""End-to-end CLI test using software fixtures written to a pytest tmp dir (never to data/)."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from tests.fixtures import make_fixture

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    sys.path.insert(0, str(ROOT))
    ns = runpy.run_path(str(ROOT / "scripts" / "run_pipeline.py"))
    return ns["main"](args)


def test_cli_writes_run_record_and_outputs(tmp_path):
    sat, ctx = make_fixture(n_fields=6, seasons=("S1", "S2", "S3"))
    sat_path, ctx_path = tmp_path / "sat.csv", tmp_path / "ctx.csv"
    sat.to_csv(sat_path, index=False)
    ctx.assign(production_tonnes=100.0).to_csv(ctx_path, index=False)
    out = tmp_path / "out"
    assert run_cli(["--satellite", str(sat_path), "--context", str(ctx_path), "--out-dir", str(out), "--n-splits", "3"]) == 0
    for name in ["fused_features.csv", "data_quality_report.json", "holdout_metrics.csv", "cv_summary.csv", "cv_folds.csv",
                 "feature_set_comparison.csv", "RUN_RECORD.md"]:
        assert (out / name).exists(), name
    record = (out / "RUN_RECORD.md").read_text()
    assert "SHA-256" in record and "spatiotemporal" in record
    assert "production_tonnes" in record  # auto-excluded as suspected leakage and stated
    fused_cols = (out / "fused_features.csv").read_text().splitlines()[0]
    assert "production_tonnes" in fused_cols  # kept in the table, only excluded from predictors
    assert json.loads((out / "data_quality_report.json").read_text())["counts"]["fused_field_seasons"] == 18


def test_cli_refuses_missing_input(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        run_cli(["--satellite", str(tmp_path / "nope.csv"), "--context", str(tmp_path / "nope2.csv"), "--out-dir", str(tmp_path / "o")])
