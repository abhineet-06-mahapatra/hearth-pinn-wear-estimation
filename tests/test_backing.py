"""Tests for the per-furnace backing resistance (Week 3)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import hearth_simulator_v2 as sim  # noqa: E402
import hearth_transient as tr  # noqa: E402


def test_default_behaviour_is_unchanged():
    q = 7384.6
    assert sim.tc_temps(40.0, q, 20.0) == pytest.approx(sim.tc_temps(40.0, q, 20.0, sim.R_BACK))


def test_higher_backing_resistance_raises_every_reading():
    q = 8000.0
    low = sim.tc_temps(40.0, q, 20.0, 0.0083)
    high = sim.tc_temps(40.0, q, 20.0, 0.020)
    assert (high > low).all()
    assert np.allclose(high - low, q * (0.020 - 0.0083))       # same shift for all three sensors


def test_gaps_between_sensors_do_not_depend_on_backing_resistance():
    q = 8000.0
    a = np.diff(sim.tc_temps(40.0, q, 20.0, 0.0083))
    b = np.diff(sim.tc_temps(40.0, q, 20.0, 0.020))
    assert a == pytest.approx(b)


def test_each_furnace_draws_a_value_inside_the_range():
    values = [sim.simulate_scenario(s).r_back_true.iloc[0] for s in range(15)]
    assert min(values) >= sim.R_BACK_RANGE[0] and max(values) <= sim.R_BACK_RANGE[1]
    assert len(set(np.round(values, 6))) > 10                  # genuinely different furnace to furnace


def test_value_is_constant_within_a_furnace():
    assert sim.simulate_scenario(2).r_back_true.nunique() == 1


def test_transient_model_uses_the_same_value_as_the_steady_model():
    assert tr.simulate_scenario_transient(8).r_back_true.iloc[0] == pytest.approx(sim.simulate_scenario(8).r_back_true.iloc[0])


def test_nominal_value_sits_inside_the_range():
    assert sim.R_BACK_RANGE[0] < sim.R_BACK_NOMINAL < sim.R_BACK_RANGE[1]
