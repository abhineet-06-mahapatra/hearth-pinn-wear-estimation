"""Tests for simulator/baselines.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import baselines as bl  # noqa: E402
import make_dataset as md  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    _, f = md.build_all(n_in=40, n_ood=2)
    return f


def test_metrics_known_values():
    m = bl.metrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert m["RMSE"] == pytest.approx(1.0)
    assert m["MAE"] == pytest.approx(0.5)
    assert m["R2"] == pytest.approx(1 - 4 / 5)


def test_mean_baseline_has_zero_r2_on_training_data(frames):
    train = bl.clean(frames["train"])
    train = train[bl.usable(train)]
    p = bl.MeanBaseline().fit(train).predict(train)
    assert bl.metrics(train.remaining_m, p.remaining_m)["R2"] == pytest.approx(0.0, abs=1e-9)


def test_clean_fills_only_short_gaps_within_a_furnace():
    d = pd.DataFrame({"scenario": [1] * 9 + [2] * 2, "day": list(range(9)) + [0, 1],
                      "TC1": [10.0] + [np.nan] * 7 + [20.0] + [30.0, np.nan],
                      "TC2": 1.0, "TC3": 2.0})
    out = bl.clean(d, ffill_days=5)
    assert out.TC1.iloc[1:6].eq(10.0).all()                  # first five missing days filled
    assert np.isnan(out.TC1.iloc[6])                         # longer gap stays missing
    assert out.TC1.iloc[10] == 30.0                          # furnace 2 filled from its own history only


def test_extrapolation_recovers_a_straight_profile_exactly():
    # readings on a straight line reaching 1150 C at 1.0 m: T(x) = 1150 - 500*(1.0 - x)
    x = np.array(bl.TC_POS)
    t = 1150 - 500 * (1.0 - x)
    d = pd.DataFrame({"TC1": [t[0]], "TC2": [t[1]], "TC3": [t[2]]})
    assert bl.extrapolated_isotherm(d).iloc[0] == pytest.approx(1.0)


def test_only_train_and_val_are_scored_by_default(frames):
    table = bl.score({k: frames[k] for k in ("train", "val")}, final=False)
    assert set(table.split) == {"train", "val"}
    assert set(table.model) == {"mean", "classical", "ridge", "gbm"}


def test_test_sets_are_scored_only_when_final(frames):
    table = bl.score(frames, final=True)
    assert {"test", "test_ood"} <= set(table.split)


def test_fitted_models_beat_the_mean_on_the_data_they_were_fitted_to(frames):
    """A sanity check that fitting works. Beating the mean on unseen furnaces is what the experiments measure."""
    table = bl.score({k: frames[k] for k in ("train", "val")}, final=False)
    v = table[(table.split == "train") & (table.target == "iso1150_m")].set_index("model").RMSE
    assert v["ridge"] < v["mean"] and v["gbm"] < v["mean"] and v["classical"] < v["mean"]
