"""Tests for the sensor-fault model in simulator/hearth_simulator_v2.py (Week 3)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import hearth_simulator_v2 as sim  # noqa: E402


def run_lengths(mask):
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return np.array(out)


def test_outage_share_is_small():
    mask = sim.dropout_mask(300_000, 0.003, 0.02, np.random.default_rng(0))
    assert 0.003 < mask.mean() < 0.008


def test_most_outages_last_about_a_day():
    runs = run_lengths(sim.dropout_mask(300_000, 0.003, 0.02, np.random.default_rng(1)))
    assert np.median(runs) <= 2
    assert np.percentile(runs, 90) <= 4


def test_rare_long_outages_exist():
    runs = run_lengths(sim.dropout_mask(300_000, 0.003, 0.02, np.random.default_rng(2)))
    assert runs.max() >= 8


def test_no_outages_when_probability_is_zero():
    assert not sim.dropout_mask(10_000, 0.0, 0.5, np.random.default_rng(3)).any()


def test_common_cause_outages_hit_all_sensors_together():
    clean = sim.simulate_scenario(4)
    noisy = sim.add_sensor_effects(clean, 4, p_dropout=0.0, p_common=0.05)
    all_out = noisy[["TC1", "TC2", "TC3"]].isna().all(axis=1)
    any_out = noisy[["TC1", "TC2", "TC3"]].isna().any(axis=1)
    assert all_out.any()
    assert (all_out == any_out).all()                      # with no single-sensor outages, sensors fail only together


def test_true_columns_are_never_damaged():
    noisy = sim.add_sensor_effects(sim.simulate_scenario(5), 5, p_dropout=0.05, p_common=0.05)
    assert noisy[["TC1_true", "TC2_true", "TC3_true"]].notna().all().all()


def test_noise_level_matches_setting():
    clean = sim.simulate_scenario(6)
    noisy = sim.add_sensor_effects(clean, 6, sigma=1.0, p_dropout=0.0, p_common=0.0)
    assert (noisy.TC2 - noisy.TC2_true).std() == pytest.approx(1.0, rel=0.1)
