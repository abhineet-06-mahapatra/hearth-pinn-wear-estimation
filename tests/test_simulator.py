"""
Tests for simulator/hearth_simulator_v2.py
Run from the repo folder:  pytest -v

Group 1 - hand calculations (one worked example you can reproduce on paper)
Group 2 - physics checks (energy balance, the 1150 C skull condition, monotonic behaviour)
Group 3 - behaviour of full simulated furnaces (limits, wear direction, reproducibility)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import hearth_simulator_v2 as sim  # noqa: E402

# ---- the worked example (same numbers as in docs/hand_calculation.md) ----
T_HM, T_W, H = 1480.0, 40.0, 30.0
S, K_S = 0.20, 30.0
CER, K_CER = 0.30, 5.0
CARB, K_C = 1.50, 20.0
R_REST = CER / K_CER + CARB / K_C + sim.R_BACK     # 0.155 m2K/W


# ---------------- Group 1: hand calculations ----------------
def test_hand_calc_total_resistance():
    layers = [1 / H, S / K_S, CER / K_CER, CARB / K_C, sim.R_BACK]
    assert sum(layers) == pytest.approx(0.195, abs=1e-6)


def test_hand_calc_heat_flux():
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, R_REST)
    assert q == pytest.approx(1440.0 / 0.195, rel=1e-9)      # 7384.6 W/m2


def test_hand_calc_thermocouples():
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, R_REST)
    tc = sim.tc_temps(T_W, q, K_C)
    expected = [T_W + q * (sim.R_BACK + x / K_C) for x in sim.TC_POS]
    assert tc == pytest.approx(expected, rel=1e-12)
    assert tc == pytest.approx([261.5, 335.4, 409.2], abs=0.05)


# ---------------- Group 2: physics checks ----------------
def test_temperature_drops_add_up_to_total():
    """Energy balance: the same flux crosses every layer, so the drops must sum to T_hm - T_water."""
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, R_REST)
    drops = [q / H, q * S / K_S, q * CER / K_CER, q * CARB / K_C, q * sim.R_BACK]
    assert sum(drops) == pytest.approx(T_HM - T_W, rel=1e-12)


def test_skull_hot_face_sits_at_1150():
    """When the skull is at its target thickness, its hot face must be at T_SOL."""
    h = 20.0
    cer, carb = 0.30, 1.60
    r_rest = cer / K_CER + carb / K_C + sim.R_BACK            # 0.16
    s = sim.skull_target(T_HM, T_W, h, r_rest, K_S)
    assert 0.0 < s < sim.S_MAX                                # not clipped, so the test is meaningful
    q = sim.heat_flux(T_HM, T_W, h, s, K_S, r_rest)
    hot_face_of_skull = T_HM - q / h
    assert hot_face_of_skull == pytest.approx(sim.T_SOL, abs=1e-6)


def test_no_skull_when_wall_is_thick():
    assert sim.skull_target(T_HM, T_W, H, R_REST, K_S) == 0.0


def test_skull_target_never_exceeds_cap():
    assert sim.skull_target(T_HM, T_W, 20.0, 0.02, K_S) == sim.S_MAX


def test_thermocouples_increase_toward_hot_side():
    q = sim.heat_flux(T_HM, T_W, H, S, K_S, R_REST)
    tc = sim.tc_temps(T_W, q, K_C)
    assert np.all(np.diff(tc) > 0)                            # TC1 < TC2 < TC3
    assert tc[0] > T_W and tc[-1] < sim.T_SOL + 1000          # sanity range


def test_thinner_wall_gives_more_flux_and_hotter_thermocouples():
    thick = sim.heat_flux(T_HM, T_W, H, S, K_S, CER / K_CER + 1.6 / K_C + sim.R_BACK)
    thin = sim.heat_flux(T_HM, T_W, H, S, K_S, CER / K_CER + 0.8 / K_C + sim.R_BACK)
    assert thin > thick
    assert np.all(sim.tc_temps(T_W, thin, K_C) > sim.tc_temps(T_W, thick, K_C))


def test_skull_lowers_flux():
    without = sim.heat_flux(T_HM, T_W, H, 0.0, K_S, R_REST)
    with_skull = sim.heat_flux(T_HM, T_W, H, 0.3, K_S, R_REST)
    assert with_skull < without


# ---------------- Group 3: whole simulated furnaces ----------------
@pytest.fixture(scope="module")
def furnaces():
    return [sim.simulate_scenario(seed) for seed in range(30)]


def test_same_seed_gives_same_data():
    a = sim.simulate_scenario(7)
    b = sim.simulate_scenario(7)
    assert a.equals(b)


def test_different_seeds_differ():
    assert not sim.simulate_scenario(1).equals(sim.simulate_scenario(2))


def test_wear_never_reverses(furnaces):
    for d in furnaces:
        assert (d.ceramic_m.diff().dropna() <= 1e-12).all()
        assert (d.carbon_m.diff().dropna() <= 1e-12).all()


def test_layers_stay_within_limits(furnaces):
    for d in furnaces:
        assert (d.ceramic_m >= 0).all()
        assert (d.carbon_m >= sim.CARBON_MIN - 1e-9).all()
        assert (d.skull_m >= 0).all() and (d.skull_m <= sim.S_MAX + 1e-12).all()


def test_skull_changes_slowly(furnaces):
    for d in furnaces:
        step = d.skull_m.diff().dropna()
        assert step.max() <= sim.S_GROW + 1e-12
        assert step.min() >= -sim.S_MELT - 1e-12


def test_normal_wear_rate_matches_field_range(furnaces):
    """Furnaces that keep their ceramic wall should lose at most about 38 mm/year (plus noise margin)."""
    checked = 0
    for d in furnaces:
        if d.ceramic_m.iloc[-1] > 0:                          # ceramic never ran out
            loss = d.ceramic_m.iloc[0] - d.ceramic_m.iloc[-1]
            assert loss <= 0.038 * 2 * 1.05
            checked += 1
    assert checked > 5


def test_thermocouples_in_plausible_range(furnaces):
    for d in furnaces:
        for c in ["TC1_true", "TC2_true", "TC3_true"]:
            assert d[c].between(0, 1500).all()
        assert (d.TC1_true < d.TC2_true).all() and (d.TC2_true < d.TC3_true).all()


def test_sensor_effects_noise_and_dropouts():
    clean = sim.simulate_scenario(3)
    noisy = sim.add_sensor_effects(clean, 3, sigma=1.5, p_dropout=0.01)
    err = (noisy.TC2 - noisy.TC2_true).dropna()
    assert err.std() == pytest.approx(1.5, rel=0.1)
    assert 0.002 < noisy.TC2.isna().mean() < 0.08
    assert noisy.TC2_true.notna().all()                       # truth is never damaged


# ---------------- Group 4: flow regimes (added in Week 2) ----------------
def test_regimes_are_zero_or_one_and_both_occur():
    seen = set()
    for seed in range(20):
        r = sim.make_regimes(sim.N_DAYS, np.random.default_rng(seed))
        assert len(r) == sim.N_DAYS
        assert set(np.unique(r)) <= {0, 1}
        seen |= set(np.unique(r))
    assert seen == {0, 1}


def test_regime_spells_are_not_shorter_than_minimum():
    for seed in range(20):
        r = sim.make_regimes(sim.N_DAYS, np.random.default_rng(seed))
        change = np.flatnonzero(np.diff(r)) + 1            # first day of each new spell
        starts = np.concatenate([[0], change, [len(r)]])
        lengths = np.diff(starts)
        assert (lengths[:-1] >= sim.REGIME_MIN_DAYS).all()  # last spell may be cut by the window


def test_h_values_follow_regime_ranges(furnaces):
    for d in furnaces:
        stag = d.h_eff[d.flow_regime == 0]
        act = d.h_eff[d.flow_regime == 1]
        if len(stag) > 30:
            assert sim.H_STAG_RANGE[0] * 0.85 <= stag.median() <= sim.H_STAG_RANGE[1] * 1.15
        if len(act) > 30:
            assert sim.H_ACT_RANGE[0] * 0.85 <= act.median() <= sim.H_ACT_RANGE[1] * 1.15


def test_skull_is_thinner_in_active_flow(furnaces):
    import pandas as pd
    d = pd.concat(furnaces)
    assert d.skull_m[d.flow_regime == 1].mean() < d.skull_m[d.flow_regime == 0].mean()
