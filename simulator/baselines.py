"""
baselines.py  (Week 4)
Three reference estimators that any neural network must beat.

  mean       : always predicts the training average (R2 about 0). The floor.
  classical  : the engineer's method. Extend the straight line through TC1 and TC3 to the 1150 C
               isotherm, then correct it with two fitted numbers (a straight-line calibration), and
               map isotherm distance to remaining wall with two more. Four numbers, no learning of patterns.
               The skull is predicted as the training average because three readings cannot show it.
  ridge, gbm : simple machine learning on the three readings and their differences (single day, no history).

Rules: fitted on train.csv only; scored on train and val. test.csv and test_ood.csv are scored only with --final,
which should be used once, at the end of the project.

Run from the repo folder (after python simulator/make_dataset.py):  python simulator/baselines.py
Saves figures/baseline_scores.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TC = ["TC1", "TC2", "TC3"]
TARGETS = ["remaining_m", "iso1150_m", "skull_m"]
TC_POS = (0.2, 0.4, 0.6)                 # m from the carbon cold face (TC1, TC2, TC3)
T_SOL = 1150.0


def clean(df, ffill_days=5):
    """Fill short sensor outages with the last good reading (within a furnace, at most ffill_days)."""
    df = df.sort_values(["scenario", "day"]).copy()
    df[TC] = df.groupby("scenario")[TC].ffill(limit=ffill_days)
    return df


def usable(df):
    return df[TC].notna().all(axis=1)


def features(df):
    X = df[TC].copy()
    X["d31"] = df.TC3 - df.TC1
    X["d21"] = df.TC2 - df.TC1
    X["d32"] = df.TC3 - df.TC2
    return X


def extrapolated_isotherm(df):
    """Distance to 1150 C if the line through TC1 and TC3 simply continued (m from the carbon cold face)."""
    gap = np.clip(df.TC3 - df.TC1, 10.0, None)
    return TC_POS[2] + (T_SOL - df.TC3) * (TC_POS[2] - TC_POS[0]) / gap


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(np.abs(err))),
                R2=float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"))


class MeanBaseline:
    def fit(self, df):
        self.m = {t: float(df[t].mean()) for t in TARGETS}
        return self

    def predict(self, df):
        return pd.DataFrame({t: np.full(len(df), self.m[t]) for t in TARGETS}, index=df.index)


class ClassicalBaseline:
    """Straight-line extrapolation, then four fitted numbers (two lines)."""
    def fit(self, df):
        raw = extrapolated_isotherm(df)
        self.iso_fit = np.polyfit(raw, df.iso1150_m, 1)                     # raw extrapolation -> true isotherm distance
        iso_hat = np.polyval(self.iso_fit, raw)
        self.rem_fit = np.polyfit(iso_hat, df.remaining_m, 1)               # isotherm distance -> remaining wall
        self.skull_mean = float(df.skull_m.mean())
        return self

    def predict(self, df):
        iso = np.polyval(self.iso_fit, extrapolated_isotherm(df))
        return pd.DataFrame({"remaining_m": np.polyval(self.rem_fit, iso), "iso1150_m": iso,
                             "skull_m": np.full(len(df), self.skull_mean)}, index=df.index)


class SklearnBaseline:
    def __init__(self, make):
        self.make = make

    def fit(self, df):
        X = features(df)
        self.models = {t: self.make().fit(X, df[t]) for t in TARGETS}
        return self

    def predict(self, df):
        X = features(df)
        return pd.DataFrame({t: self.models[t].predict(X) for t in TARGETS}, index=df.index)


def make_models():
    return {
        "mean": MeanBaseline(),
        "classical": ClassicalBaseline(),
        "ridge": SklearnBaseline(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "gbm": SklearnBaseline(lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)),
    }


def score(frames, final=False):
    """frames: dict split name -> DataFrame. Fit on train; score train and val (and test sets only if final)."""
    clean_frames = {k: clean(v) for k, v in frames.items()}
    train = clean_frames["train"][usable(clean_frames["train"])]
    names = ["train", "val"] + (["test", "test_ood"] if final else [])
    rows = []
    for mname, model in make_models().items():
        model.fit(train)
        for split in names:
            d = clean_frames[split]
            d = d[usable(d)]
            pred = model.predict(d)
            for t in TARGETS:
                rows.append(dict(model=mname, split=split, target=t, rows=len(d), **metrics(d[t], pred[t])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    final = "--final" in sys.argv
    frames = {n: pd.read_csv(os.path.join("data", f"{n}.csv")) for n in ["train", "val"] + (["test", "test_ood"] if final else [])}
    table = score(frames, final=final)
    os.makedirs("figures", exist_ok=True)
    table.round(4).to_csv("figures/baseline_scores.csv", index=False)
    show = table[table.split == ("val" if not final else table.split)]
    pd.set_option("display.width", 160)
    print(table[table.split == "val"].round(3).pivot(index="model", columns="target", values="RMSE").add_prefix("RMSE ").to_string())
    print()
    print(table[table.split == "val"].round(3).pivot(index="model", columns="target", values="R2").add_prefix("R2 ").to_string())
    print("\nScored on train and validation only. Saved figures/baseline_scores.csv" if not final else "\nFINAL scores including test sets. Saved figures/baseline_scores.csv")
