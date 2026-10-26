"""
Tests for simulator/hearth_transient.py (time-dependent model)
Run from the repo folder:  pytest -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import hearth_simulator_v2 as sim  # noqa: E402
import hearth_transient as tr  # noqa: E402

# worked example from docs/hand_calculation.md
T_HM, T_W, H = 1480.0, 40.0, 30.0
S, K_S, CER, K_CER, CARB, K_C = 0.20, 30.0, 0.30, 5.0, 1.50, 20.0


def example_grid():
    return tr.build_grid(CARB, CER, S, K_C, K_CER, K_S)


def test_steady_state_matches_series_resistance():
    dx, k, rcp = example_grid()
    T = tr.steady_state(dx, k, rcp, T_W, T_HM, H)
    r_rest = CER / K_CER + CARB / K_C + sim.R_BACK
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, r_rest)
    assert tr.thermocouples(T, dx) == pytest.approx(sim.tc_temps(T_W, q, K_C), rel=1e-6)


def test_flux_balance_at_steady_state():
    dx, k, rcp = example_grid()
    T = tr.steady_state(dx, k, rcp, T_W, T_HM, H)
    _, q_out, q_in = tr.implicit_step(T, dx, k, rcp, T_W, T_HM, H, tr.DT)
    r_rest = CER / K_CER + CARB / K_C + sim.R_BACK
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, r_rest)
    assert q_out == pytest.approx(q, rel=1e-6)
    assert q_in == pytest.approx(q, rel=1e-6)


def test_heat_is_conserved_in_a_step():
    dx, k, rcp = example_grid()
    rng = np.random.default_rng(1)
    T0 = rng.uniform(100, 900, len(dx))                       # arbitrary starting temperatures
    T1, q_out, q_in = tr.implicit_step(T0, dx, k, rcp, T_W, T_HM, H, tr.DT)
    stored = np.sum(rcp * dx * (T1 - T0))
    assert stored == pytest.approx((q_in - q_out) * tr.DT, rel=1e-9)


def test_nothing_changes_when_conditions_are_constant():
    dx, k, rcp = example_grid()
    T = tr.steady_state(dx, k, rcp, T_W, T_HM, H)
    T2 = T.copy()
    for _ in range(10):
        T2, _, _ = tr.implicit_step(T2, dx, k, rcp, T_W, T_HM, H, tr.DT)
    assert T2 == pytest.approx(T, abs=1e-6)


def test_step_response_is_gradual_and_slower_in_thicker_walls():
    dx, k, rcp = tr.build_grid(1.5, 0.30, 0.10, 20.0, 5.0, 2.0)
    T = tr.steady_state(dx, k, rcp, 40.0, 1480.0, 25.0)
    start = tr.thermocouples(T, dx)[2]
    final = tr.thermocouples(tr.steady_state(dx, k, rcp, 40.0, 1580.0, 25.0), dx)[2]
    path = []
    for _ in range(40 * tr.SUBSTEPS):
        T, _, _ = tr.implicit_step(T, dx, k, rcp, 40.0, 1580.0, 25.0, tr.DT)
        path.append(tr.thermocouples(T, dx)[2])
    path = np.array(path)
    assert (np.diff(path) >= -1e-9).all()                       # rises monotonically
    assert path[0] - start < 0.5 * (final - start)              # first 6 hours: well under half the change
    assert path[-1] == pytest.approx(final, rel=1e-3)           # settles to the new steady state
    assert tr.step_response_days(1.9) > tr.step_response_days(0.7)


def test_isotherm_from_profile_matches_analytic():
    dx, k, rcp = example_grid()
    T = tr.steady_state(dx, k, rcp, T_W, T_HM, H)
    _, q_out, q_in = tr.implicit_step(T, dx, k, rcp, T_W, T_HM, H, tr.DT)
    x = tr.isotherm_from_profile(T, dx, T_W, T_HM, H, q_out, q_in)
    assert x == pytest.approx(1.7766, abs=5e-3)


@pytest.fixture(scope="module")
def furnace():
    return tr.simulate_scenario_transient(3)


def test_transient_furnace_has_expected_columns_and_length(furnace):
    for c in ["TC1_true", "TC2_true", "TC3_true", "TC1_ss", "iso1150_m", "flow_regime", "skull_m"]:
        assert c in furnace.columns
    assert len(furnace) == sim.N_DAYS


def test_transient_readings_stay_between_water_and_hot_metal(furnace):
    for c in ["TC1_true", "TC2_true", "TC3_true"]:
        assert (furnace[c] > furnace.T_water).all()
        assert (furnace[c] < furnace.T_hotmetal).all()
    assert (furnace.TC1_true < furnace.TC2_true).all() and (furnace.TC2_true < furnace.TC3_true).all()


def test_transient_follows_same_history_as_steady_model():
    a = sim.simulate_scenario(12)
    b = tr.simulate_scenario_transient(12)
    for c in ["remaining_m", "ceramic_m", "carbon_m", "skull_m", "T_hotmetal", "h_eff", "T_water"]:
        assert np.allclose(a[c], b[c], rtol=0, atol=1e-12)


def test_transient_differs_from_steady_only_modestly(furnace):
    diff = (furnace.TC3_true - furnace.TC3_ss).abs()
    assert diff.median() < 5.0                                  # typical lag effect is a few degrees
    assert diff.max() < 150.0                                   # larger only during upsets


def test_isotherm_label_matches_steady_model_closely():
    a = sim.simulate_scenario(0)
    b = tr.simulate_scenario_transient(0)
    assert (b.iso1150_m - a.iso1150_m).abs().median() < 0.02
