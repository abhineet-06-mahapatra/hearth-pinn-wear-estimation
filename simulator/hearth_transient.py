"""
hearth_transient.py
Time-dependent version of the hearth wall model (Week 3).

The wall is split into cells: carbon (N_C), ceramic (N_CER) and skull (N_S). Each day the heat
equation is advanced in SUBSTEPS implicit (backward Euler) steps, so the thermocouples respond to
changes with a lag that depends on the wall: thick walls respond slowly, thin walls quickly.
It starts from the steady state of day 0 and uses the same operating conditions, skull rule and
wear rules as hearth_simulator_v2.py, so the two can be compared day by day.

Layer heat capacities are ASSUMPTIONS (see HEAT_CAPACITY below), not sourced values.
When a layer changes thickness (wear, skull growth) its cells stretch and keep their temperatures;
this is an approximation that does not conserve heat exactly across that change.

Run:  python simulator/hearth_transient.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

sys.path.insert(0, os.path.dirname(__file__))
import hearth_simulator_v2 as sim  # noqa: E402

N_C, N_CER, N_S = 15, 6, 3                # cells per layer
SUBSTEPS = 4                              # implicit steps per day
DT = 86400.0 / SUBSTEPS                   # seconds
MIN_LAYER = 1e-4                          # m, keeps zero-thickness layers numerically present
# volumetric heat capacity rho*cp (J/m3.K): ASSUMPTIONS
HEAT_CAPACITY = dict(carbon=2.0e6, ceramic=3.0e6, skull=3.5e6)


def build_grid(carb, cer, s, k_c, k_cer, k_s):
    """Cell thicknesses, conductivities and heat capacities from the cold face (carbon) to the hot face (skull)."""
    cer_e, s_e = max(cer, MIN_LAYER), max(s, MIN_LAYER)
    dx = np.concatenate([np.full(N_C, carb / N_C), np.full(N_CER, cer_e / N_CER), np.full(N_S, s_e / N_S)])
    k = np.concatenate([np.full(N_C, k_c), np.full(N_CER, k_cer), np.full(N_S, k_s)])
    rcp = np.concatenate([np.full(N_C, HEAT_CAPACITY["carbon"]), np.full(N_CER, HEAT_CAPACITY["ceramic"]),
                          np.full(N_S, HEAT_CAPACITY["skull"])])
    return dx, k, rcp


def implicit_step(T, dx, k, rcp, T_w, T_hm, h, dt):
    """Advance cell temperatures by dt. Returns (new T, heat flux to the water, heat flux from the hot metal)."""
    n = len(T)
    G = 1.0 / (dx[:-1] / (2 * k[:-1]) + dx[1:] / (2 * k[1:]))          # between cells
    Gc = 1.0 / (sim.R_BACK + dx[0] / (2 * k[0]))                         # cold face to water
    Gh = 1.0 / (1.0 / h + dx[-1] / (2 * k[-1]))                          # hot metal to hot face
    C = rcp * dx / dt
    diag = C.copy()
    diag[:-1] += G
    diag[1:] += G
    diag[0] += Gc
    diag[-1] += Gh
    ab = np.zeros((3, n))
    ab[0, 1:] = -G
    ab[1] = diag
    ab[2, :-1] = -G
    rhs = C * T
    rhs[0] += Gc * T_w
    rhs[-1] += Gh * T_hm
    T_new = solve_banded((1, 1), ab, rhs)
    return T_new, Gc * (T_new[0] - T_w), Gh * (T_hm - T_new[-1])


def steady_state(dx, k, rcp, T_w, T_hm, h):
    """Steady temperatures for the grid (an implicit step with an infinite time step, so heat capacity drops out)."""
    T0 = np.full(len(dx), 0.5 * (T_w + T_hm))
    T, _, _ = implicit_step(T0, dx, k, rcp, T_w, T_hm, h, np.inf)
    return T


def thermocouples(T, dx):
    """Readings at TC_POS (m from the carbon cold face), interpolated between cell centres."""
    centres = np.cumsum(dx) - dx / 2
    return np.interp(sim.TC_POS, centres, T)


def isotherm_from_profile(T, dx, T_w, T_hm, h, q_out, q_in):
    """Distance from the carbon cold face to the 1150 C isotherm from the cell temperatures."""
    centres = np.cumsum(dx) - dx / 2
    xs = np.concatenate([[0.0], centres, [dx.sum()]])
    Ts = np.concatenate([[T_w + q_out * sim.R_BACK], T, [T_hm - q_in / h]])
    above = np.flatnonzero(Ts >= sim.T_SOL)
    if len(above) == 0:
        return float(dx.sum())
    i = above[0]
    if i == 0:
        return 0.0
    return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (sim.T_SOL - Ts[i - 1]) / (Ts[i] - Ts[i - 1]))


def step_response_days(carb, cer=0.30, s=0.10, h=25.0, jump=100.0, days=40, tc_index=2):
    """Days for thermocouple `tc_index` to cover 63% of its change after a step of `jump` C in hot metal temperature."""
    k_c, k_cer, k_s = 20.0, 5.0, 2.0
    dx, k, rcp = build_grid(carb, cer, s, k_c, k_cer, k_s)
    T = steady_state(dx, k, rcp, 40.0, 1480.0, h)
    t_start = thermocouples(T, dx)[tc_index]
    t_end = thermocouples(steady_state(dx, k, rcp, 40.0, 1480.0 + jump, h), dx)[tc_index]
    for day in range(1, days * SUBSTEPS + 1):
        T, _, _ = implicit_step(T, dx, k, rcp, 40.0, 1480.0 + jump, h, DT)
        if (thermocouples(T, dx)[tc_index] - t_start) >= 0.632 * (t_end - t_start):
            return day / SUBSTEPS
    return float("nan")


def simulate_scenario_transient(seed):
    """Same furnace as sim.simulate_scenario(seed), but thermocouples come from the time-dependent model."""
    rng = np.random.default_rng(seed)
    n = sim.N_DAYS
    k_c = rng.uniform(*sim.K_C_RANGE)
    k_cer = sim.K_CER_NOM * rng.uniform(0.8, 1.2)
    k_s = sim.K_S_NOM * rng.uniform(0.85, 1.15)
    grw0 = rng.uniform(1.58, 2.42)
    cer0 = rng.uniform(0.40, 0.50)
    carb0 = grw0 - cer0
    cer = 0.0 if rng.random() < sim.P_CUP_GONE else cer0 * rng.uniform(0.1, 1.0)
    carb = carb0 * (rng.uniform(0.80, 1.0) if cer < 0.05 else 1.0)
    cer_rate = rng.uniform(*sim.CER_RATE_MM_YR) / 1000.0
    carb_rate = rng.uniform(*sim.CARBON_RATE_M_YR)
    T_hm = sim.ar1(n, sim.T_HM_NOM, 12.0, 0.95, rng)
    flow = sim.make_regimes(n, rng)
    h_stag, h_act = rng.uniform(*sim.H_STAG_RANGE), rng.uniform(*sim.H_ACT_RANGE)
    h = np.where(flow == 0, h_stag, h_act) * (1.0 + 0.05 * sim.ar1(n, 0.0, 1.0, 0.9, rng))
    T_w = np.clip(sim.ar1(n, sim.T_WATER_NOM, 3.0, 0.9, rng), 20, None)
    for _ in range(rng.integers(1, 5)):
        i0 = rng.integers(30, n - 30)
        d = rng.integers(3, 15)
        h[i0:i0 + d] *= rng.uniform(1.8, 3.0)
        T_hm[i0:i0 + d] += rng.uniform(15, 40)

    s = 0.0
    T = None
    rows = []
    for t in range(n):
        R_rest = cer / k_cer + carb / k_c + sim.R_BACK
        target = sim.skull_target(T_hm[t], T_w[t], h[t], R_rest, k_s)
        s += float(np.clip(target - s, -sim.S_MELT, sim.S_GROW))
        dx, k, rcp = build_grid(carb, cer, s, k_c, k_cer, k_s)
        if T is None:
            T = steady_state(dx, k, rcp, T_w[t], T_hm[t], h[t])
        for _ in range(SUBSTEPS):
            T, q_out, q_in = implicit_step(T, dx, k, rcp, T_w[t], T_hm[t], h[t], DT)
        tc = thermocouples(T, dx)
        q_ss = sim.heat_flux(T_hm[t], T_w[t], h[t], s, k_s, R_rest)
        tc_ss = sim.tc_temps(T_w[t], q_ss, k_c)
        iso = isotherm_from_profile(T, dx, T_w[t], T_hm[t], h[t], q_out, q_in)
        cup_gone = 0 if cer > 0.0 else 1
        rows.append([seed, t, *tc, *tc_ss, cer + carb, cer, carb, s, q_out, T_hm[t], h[t], T_w[t], k_c, k_cer, cup_gone, flow[t], iso])
        protected = s > sim.SKULL_THRESHOLD
        noise = rng.uniform(0.5, 1.5)
        if cer > 0.0:
            cer = max(0.0, cer - cer_rate * (sim.SKULL_WEAR_FACTOR[0] if protected else 1.0) / 365.0 * noise)
        else:
            carb = max(sim.CARBON_MIN, carb - carb_rate * (sim.SKULL_WEAR_FACTOR[1] if protected else 1.0) / 365.0 * noise)

    cols = ["scenario", "day"] + [f"TC{i + 1}_true" for i in range(3)] + [f"TC{i + 1}_ss" for i in range(3)] + \
           ["remaining_m", "ceramic_m", "carbon_m", "skull_m", "heat_flux_Wm2", "T_hotmetal", "h_eff", "T_water",
            "k_c_true", "k_cer_true", "cup_gone", "flow_regime", "iso1150_m"]
    return pd.DataFrame(rows, columns=cols)


def build_dataset(n_scenarios=200, out_csv="hearth_synthetic_transient.csv"):
    frames = [sim.add_sensor_effects(simulate_scenario_transient(s), s) for s in range(n_scenarios)]
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out_csv, index=False)
    return data


if __name__ == "__main__":
    print("Time for TC3 to cover 63% of a +100 C hot-metal step (days), ceramic 0.30 m, skull 0.10 m:")
    for carb in (0.7, 1.0, 1.5, 1.9):
        print(f"  carbon {carb:.1f} m: {step_response_days(carb):5.2f} days")
    d = pd.concat([simulate_scenario_transient(s) for s in range(20)])
    lag = (d.TC3_true - d.TC3_ss)
    print("\n20 furnaces: TC3 transient minus steady-state reading (C): median %.1f, 5th %.1f, 95th %.1f, largest abs %.0f"
          % (lag.median(), lag.quantile(.05), lag.quantile(.95), lag.abs().max()))
