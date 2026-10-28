"""
diagnostics_week4.py
Why are single-day estimates from three thermocouples weak, and what would help?
Four experiments on the TRAIN and VAL splits only (test sets are never touched here).

  A. Would knowing material properties and the hot-side film resistance rescue single-day estimates?
     (analysis only: those columns are diagnostics and must never be real model inputs)
  B. Is the CHANGE since each furnace's own first days easier to estimate than the absolute value?
  C. Does a 30 to 90 day history of the readings help?

Run from the repo folder after make_dataset.py:  python simulator/diagnostics_week4.py
Saves figures/diagnostics_week4.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import baselines as bl  # noqa: E402

MODELS = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
          "gbm": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=0)}


def r2_table(tr, va, feature_sets, targets, experiment):
    rows = []
    for name, cols in feature_sets.items():
        mtr, mva = tr[cols].notna().all(axis=1), va[cols].notna().all(axis=1)
        for mname, make in MODELS.items():
            row = {"experiment": experiment, "inputs": name, "model": mname}
            for t in targets:
                m = make().fit(tr.loc[mtr, cols], tr.loc[mtr, t])
                row[t] = round(bl.metrics(va.loc[mva, t], m.predict(va.loc[mva, cols]))["R2"], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def load(name):
    d = bl.clean(pd.read_csv(os.path.join("data", f"{name}.csv")))
    d["d31"], d["d21"], d["d32"] = d.TC3 - d.TC1, d.TC2 - d.TC1, d.TC3 - d.TC2
    d["film_R"] = 1.0 / d.h_eff
    g = d.groupby("scenario")
    for c in bl.TC:
        d[c + "_ref"] = g[c].transform(lambda s: s.iloc[:10].median())
        d["d" + c] = d[c] - d[c + "_ref"]
    for t, n in [("remaining_m", "d_rem"), ("iso1150_m", "d_iso"), ("skull_m", "d_skull")]:
        d[n] = d[t] - g[t].transform("first")
    for w in (30, 90):
        for c in ("dTC1", "dTC3"):
            d[f"{c}_med{w}"] = d.groupby("scenario")[c].transform(lambda s: s.rolling(w, min_periods=w // 2).median())
            d[f"{c}_min{w}"] = d.groupby("scenario")[c].transform(lambda s: s.rolling(w, min_periods=w // 2).min())
    return d


if __name__ == "__main__":
    tr, va = load("train"), load("val")
    base = ["TC1", "TC2", "TC3", "d31", "d21", "d32"]
    abs_t = ["remaining_m", "iso1150_m", "skull_m"]
    chg_t = ["d_rem", "d_iso", "d_skull"]
    h30 = ["dTC1_med30", "dTC3_med30", "dTC1_min30", "dTC3_min30"]
    h90 = ["dTC1_med90", "dTC3_med90", "dTC1_min90", "dTC3_min90"]
    d3 = ["dTC1", "dTC2", "dTC3"]

    a = r2_table(tr, va, {
        "sensors only": base,
        "+ carbon conductivity": base + ["k_c_true"],
        "+ conductivities + backing resistance": base + ["k_c_true", "k_cer_true", "r_back_true"],
        "+ hot-side film resistance": base + ["film_R"],
        "+ everything above": base + ["film_R", "k_c_true", "k_cer_true", "r_back_true"],
    }, abs_t, "A absolute value, single day")
    b = r2_table(tr, va, {"readings (level)": ["TC1", "TC2", "TC3", "d31"], "change since start": d3 + ["d31"]}, chg_t, "B change since start, single day")
    c = r2_table(tr, va, {"today only": d3, "+ 30-day history": d3 + h30, "+ 30 and 90-day history": d3 + h30 + h90}, chg_t, "C change since start, with history")
    out = pd.concat([a, b, c], ignore_index=True)
    os.makedirs("figures", exist_ok=True)
    out.to_csv("figures/diagnostics_week4.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    for key, part in out.groupby("experiment", sort=True):
        print("\n" + key + "  (validation R2)")
        print(part.drop(columns="experiment").dropna(axis=1, how="all").to_string(index=False))
    v = va[va.d_rem.notna()]
    print("\nspread of the change targets in validation: wall sd %.2f m, isotherm sd %.2f m, skull sd %.3f m"
          % (v.d_rem.std(), v.d_iso.std(), v.d_skull.std()))
