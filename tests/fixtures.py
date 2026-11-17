"""SOFTWARE TEST FIXTURES ONLY.

These generators build small deterministic in-memory tables so the code paths can be
tested. They are NOT wheat observations, are never written into data/, and no score
computed from them may appear in the report as an experimental result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_fixture(n_fields: int = 6, seasons=("S1", "S2"), obs_per_season: int = 5, seed: int = 0):
    """Return (satellite_df, context_df) with consistent keys."""
    rng = np.random.default_rng(seed)
    sat_rows, ctx_rows = [], []
    for f in range(n_fields):
        for si, season in enumerate(seasons):
            level = 0.3 + 0.05 * f + 0.02 * si
            for k in range(obs_per_season):
                nir = 0.4 + level * np.sin(np.pi * (k + 1) / (obs_per_season + 1)) + rng.normal(0, 0.01)
                red = 0.15
                sat_rows.append({"field_id": f"F{f}", "season": season, "observation_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=15 * k),
                                 "red_reflectance": red, "nir_reflectance": nir})
            ctx_rows.append({"field_id": f"F{f}", "season": season, "yield": 2.0 + 4 * level + rng.normal(0, 0.1), "rain_mm": 100 + 10 * f + rng.normal(0, 5)})
    return pd.DataFrame(sat_rows), pd.DataFrame(ctx_rows)
