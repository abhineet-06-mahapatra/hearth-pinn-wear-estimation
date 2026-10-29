"""
windows.py  (Week 4)
Turns each furnace's daily readings into sliding windows for a sequence model.

For a window ending on day t the model sees days t-L+1 ... t (never later). Per day it gets 9 channels:
  change   (3): reading minus the furnace's own reference (median of its first REF_DAYS days), in C
  level    (3): the reading itself, in C
  missing  (3): 1 if that sensor was out that day, else 0 (the reading is then set to 0 after scaling).
            A sensor with no reading in the reference period is flagged missing for the whole furnace.
The targets are CHANGES since day 0 (label on day t minus label on day 0), because absolute values are poorly posed
(see docs/findings_week4.md): skull, 1150 C distance, remaining wall.

Windows never cross a furnace boundary. The scaler is fitted on the TRAIN split only.
"""
import numpy as np
import pandas as pd

TC = ["TC1", "TC2", "TC3"]
TARGETS = ["skull_m", "iso1150_m", "remaining_m"]
REF_DAYS = 10
CHANNELS = [f"d{c}" for c in TC] + TC + [f"miss_{c}" for c in TC]


def furnace_windows(f, length=60, stride=5):
    """Windows of one furnace (a DataFrame sorted by day). Returns X (n, length, 9), y (n, 3), y_abs (n, 3), days (n,)."""
    f = f.sort_values("day")
    tc = f[TC].to_numpy(float)
    with np.errstate(all="ignore"):
        ref = np.array([np.nanmedian(col) if np.isfinite(col).any() else np.nan for col in tc[:REF_DAYS].T])
    # a sensor with no reading in the first REF_DAYS days has no reference: treat it as missing for the whole furnace
    tc = tc.copy()
    tc[:, np.isnan(ref)] = np.nan
    change = tc - ref
    miss = np.isnan(tc).astype(float)
    raw = np.concatenate([change, tc, miss], axis=1)                      # (days, 9), NaN where a sensor was out
    lab = f[TARGETS].to_numpy(float)
    first = REF_DAYS - 1
    ends = [t for t in range(max(length - 1, first), len(f), stride)]
    X = np.stack([raw[t - length + 1:t + 1] for t in ends]) if ends else np.empty((0, length, 9))
    y_abs = lab[ends] if ends else np.empty((0, 3))
    y = y_abs - lab[0]
    return X, y, y_abs, f.day.to_numpy()[ends] if ends else np.empty(0)


def build_windows(df, length=60, stride=5):
    """All furnaces of a split. Returns dict with X, y, y_abs and meta (scenario, day of the window's last day)."""
    Xs, ys, yas, sc, dd = [], [], [], [], []
    for sid, f in df.groupby("scenario"):
        X, y, ya, days = furnace_windows(f, length, stride)
        Xs.append(X); ys.append(y); yas.append(ya); sc.append(np.full(len(X), sid)); dd.append(days)
    return {"X": np.concatenate(Xs), "y": np.concatenate(ys), "y_abs": np.concatenate(yas),
            "scenario": np.concatenate(sc), "day": np.concatenate(dd)}


def fit_scaler(X):
    """Mean and standard deviation per channel over observed (non-NaN) values. Use the train split only."""
    flat = X.reshape(-1, X.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    mean[6:], std[6:] = 0.0, 1.0                                          # the missing-flags are left as 0 / 1
    std[std == 0] = 1.0
    return {"mean": mean, "std": std}


def apply_scaler(X, scaler):
    Z = (X - scaler["mean"]) / scaler["std"]
    return np.nan_to_num(Z, nan=0.0)                                      # a missing reading becomes 0 (its flag tells the model)


def make_window_sets(frames, length=60, stride=5):
    """frames: dict split -> DataFrame (needs 'train'). Returns dict split -> windows, scaled with the train scaler."""
    sets = {name: build_windows(df, length, stride) for name, df in frames.items()}
    scaler = fit_scaler(sets["train"]["X"])
    for w in sets.values():
        w["X"] = apply_scaler(w["X"], scaler)
    y_std = sets["train"]["y"].std(axis=0)
    return sets, scaler, y_std
