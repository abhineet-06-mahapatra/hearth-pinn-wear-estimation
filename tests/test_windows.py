"""Tests for simulator/windows.py: shapes, alignment, no look-ahead, no leakage across furnaces or splits."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import windows as wd  # noqa: E402


def toy(scenario=1, days=100, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"scenario": scenario, "day": np.arange(days),
                      "TC1": 200 + offset + rng.normal(0, 1, days).cumsum(), "TC2": 300 + offset + rng.normal(0, 1, days).cumsum(),
                      "TC3": 400 + offset + rng.normal(0, 1, days).cumsum()})
    d["skull_m"] = np.linspace(0, 0.2, days)
    d["iso1150_m"] = np.linspace(1.5, 1.2, days)
    d["remaining_m"] = np.linspace(2.0, 1.8, days)
    return d


def test_number_of_windows_and_shapes():
    X, y, ya, days = wd.furnace_windows(toy(days=730), length=60, stride=5)
    assert X.shape == (135, 60, 9) and y.shape == (135, 3) and len(days) == 135


def test_target_is_the_change_since_day_zero_at_the_windows_last_day():
    f = toy()
    X, y, ya, days = wd.furnace_windows(f, length=30, stride=10)
    for i, t in enumerate(days):
        assert ya[i, 0] == pytest.approx(f.skull_m.iloc[t])
        assert y[i, 0] == pytest.approx(f.skull_m.iloc[t] - f.skull_m.iloc[0])


def test_last_row_of_a_window_is_the_last_days_reading():
    f = toy()
    X, y, ya, days = wd.furnace_windows(f, length=30, stride=10)
    ref = np.median(f[wd.TC].to_numpy()[:wd.REF_DAYS], axis=0)
    for i, t in enumerate(days):
        assert X[i, -1, 3:6] == pytest.approx(f[wd.TC].to_numpy()[t])
        assert X[i, -1, 0:3] == pytest.approx(f[wd.TC].to_numpy()[t] - ref)


def test_no_look_ahead_future_readings_do_not_change_a_window():
    f = toy(days=120)
    X1, _, _, days = wd.furnace_windows(f, length=30, stride=10)
    g = f.copy()
    g.loc[g.day > 60, wd.TC] += 500.0                                      # rewrite the future after day 60
    X2, _, _, _ = wd.furnace_windows(g, length=30, stride=10)
    keep = days <= 60
    assert np.allclose(X1[keep], X2[keep])


def test_missing_readings_are_flagged_and_zeroed_after_scaling():
    f = toy()
    f.loc[f.day == 50, "TC2"] = np.nan
    sets, scaler, _ = wd.make_window_sets({"train": f}, length=30, stride=1)
    X = sets["train"]["X"]
    days = sets["train"]["day"]
    i = np.flatnonzero(days == 50)[0]
    assert X[i, -1, 7] == 1.0                                              # missing flag for TC2
    assert X[i, -1, 1] == 0.0 and X[i, -1, 4] == 0.0                       # change and level channels zeroed
    assert not np.isnan(X).any()


def test_windows_do_not_cross_furnaces():
    df = pd.concat([toy(1, seed=1), toy(2, seed=2, offset=100.0)], ignore_index=True)
    w = wd.build_windows(df, length=30, stride=10)
    for sid in (1, 2):
        sub = w["X"][w["scenario"] == sid]
        f = df[df.scenario == sid]
        assert np.allclose(sub[0][:, 3:6], f[wd.TC].to_numpy()[w["day"][w["scenario"] == sid][0] - 29:w["day"][w["scenario"] == sid][0] + 1])


def test_scaler_is_fitted_on_train_only():
    tr, va = toy(1, seed=1), toy(2, seed=2, offset=500.0)
    s1, _, _ = wd.make_window_sets({"train": tr, "val": va}, length=30, stride=10)
    va2 = va.copy()
    va2[wd.TC] = va2[wd.TC] * 3.0                                          # change the validation data completely
    s2, _, _ = wd.make_window_sets({"train": tr, "val": va2}, length=30, stride=10)
    assert np.allclose(s1["train"]["X"], s2["train"]["X"])                 # train windows unaffected by validation data


def test_scaled_training_channels_have_unit_scale():
    s, scaler, _ = wd.make_window_sets({"train": toy(days=400)}, length=30, stride=5)
    X = s["train"]["X"][:, :, :6]
    assert np.abs(X.mean()) < 0.3 and 0.7 < X.std() < 1.3


def test_sensor_with_no_reference_is_flagged_missing_for_the_whole_furnace():
    f = toy(days=100)
    f.loc[f.day < wd.REF_DAYS, "TC3"] = np.nan                            # no reading at all in the reference period
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                     # no silent numpy warnings either
        X, y, ya, days = wd.furnace_windows(f, length=30, stride=10)
    assert (X[:, :, 8] == 1.0).all()                                      # TC3 flagged missing in every window
    assert np.isnan(X[:, :, [2, 5]]).all()                                # its change and level channels are unusable
    assert np.isfinite(X[:, :, [0, 1, 3, 4]]).all()                       # the other sensors are untouched
