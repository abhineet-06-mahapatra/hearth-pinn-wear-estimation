"""
hearth_simulator_v2.py
Full-wall version of the 1D hearth sidewall simulator.

Layers, hot side to cold side:
  hot metal -> film (h_eff) -> skull (k_s) -> ceramic wall (k_cer) -> carbon (k_c) -> backing R_back -> water

Real-data anchors (see parameters.csv for sources):
  * initial global refractory wall 1.58-2.42 m, ceramic wall 0.40-0.50 m  (Boumahdi et al. 2017)
  * normal whole-wall wear about 16-38 mm/year (derived by verify_wear_rates.py)
  * over 400 mm of carbon lost in a few months once the ceramic cup is gone (same paper)
  * carbon block conductivity 12-37 W/m.K (Nippon Steel Tech. Report 415; Ternium post-mortem)
  * hot metal 1400-1600 C; wear surface at the 1150 C isotherm (Kaymak et al. 2018)

ASSUMPTIONS / TUNED (not from sources): k_cer, R_back, T_water, h_eff, skull
growth and melt rates, skull cap, the protective effect of a skull on wear, thermocouple depths.

Skull: its target thickness comes from the heat balance with h_eff, but the actual
skull moves toward the target at limited speeds (grows slowly, melts faster).
h_eff switches between a stagnant-flow value (skull can form) and an active-flow value
(skull melts); spells last weeks to months. These two values and the spell lengths are assumptions.

Run:  python simulator/hearth_simulator_v2.py
"""
import numpy as np
import pandas as pd

# ---------------- nominal parameters (edit these) ----------------
N_DAYS = 730
T_SOL = 1150.0                 # C, wear-surface isotherm (Kaymak 2018)
T_HM_NOM = 1480.0              # C, inside 1400-1600 (Boumahdi 2017)
T_WATER_NOM = 40.0             # C, ASSUMPTION
R_BACK = 0.02                  # m2K/W ramming + stave + water, ASSUMPTION
K_C_RANGE = (13.0, 35.0)       # W/mK carbon, inside the real 12-37 range
K_CER_NOM = 5.0                # W/mK ceramic wall, ASSUMPTION (low-k ceramic, patent says <7)
K_S_NOM = 2.0                  # W/mK skull; value used for the skull layer by Helle et al. 2022 (MMTB 53(1))
H_STAG_RANGE = (15.0, 30.0)    # W/m2K effective hot-side coefficient, STAGNANT flow (skull can form), ASSUMPTION/TUNED
H_ACT_RANGE = (45.0, 80.0)     # W/m2K effective hot-side coefficient, ACTIVE flow (skull melts), ASSUMPTION/TUNED
REGIME_MEAN_DAYS = (90.0, 45.0)  # mean length of stagnant, active spells (days), ASSUMPTION; skull cycles last weeks to months (Helle et al. 2022)
REGIME_MIN_DAYS = 7            # shortest spell, days
S_MAX = 0.30                   # m skull cap (observed build-up 0.07-0.26 m), ASSUMPTION
S_GROW = 0.003                 # m/day, ASSUMPTION
S_MELT = 0.015                 # m/day, ASSUMPTION
TC_POS = np.array([0.2, 0.4, 0.6])   # m from cold face, ASSUMPTION
CARBON_MIN = 0.70              # m, wear stops here so thermocouples stay inside carbon
CER_RATE_MM_YR = (16.0, 38.0)  # normal whole-wall wear (derived from field tables)
CARBON_RATE_M_YR = (0.3, 1.2)  # carbon wear once the cup is gone, ASSUMPTION from ">400 mm in a few months"
SKULL_WEAR_FACTOR = (0.5, 0.3) # wear multiplier with skull (ceramic, carbon), ASSUMPTION
SKULL_THRESHOLD = 0.05         # m, skull counted as protective above this
P_CUP_GONE = 0.30              # share of furnaces that start with the ceramic wall already gone - a SCENARIO CHOICE, not a real frequency


def ar1(n, mean, sd, phi, rng):
    x = np.zeros(n); x[0] = mean
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + e[i]
    return x


def make_regimes(n, rng):
    """Hot-metal flow regime for each day: 0 = stagnant (skull can form), 1 = active (skull melts).
    Spell lengths are exponential with the means in REGIME_MEAN_DAYS (an assumption)."""
    out = np.empty(n, dtype=int)
    state = int(rng.integers(0, 2))
    i = 0
    while i < n:
        d = max(REGIME_MIN_DAYS, int(rng.exponential(REGIME_MEAN_DAYS[state])))
        out[i:i + d] = state
        i += d
        state = 1 - state
    return out


def skull_target(T_hm, T_w, h, R_rest, k_s):
    """Skull thickness (m) at which the skull hot face sits at T_SOL, clipped to [0, S_MAX]."""
    raw = k_s * ((T_SOL - T_w) / (h * (T_hm - T_SOL)) - R_rest)
    return float(np.clip(raw, 0.0, S_MAX))


def heat_flux(T_hm, T_w, h, s, k_s, R_rest):
    """Steady heat flux (W/m2) through film, skull and the rest of the wall."""
    return (T_hm - T_w) / (1.0 / h + s / k_s + R_rest)


def tc_temps(T_w, q, k_c):
    """Thermocouple temperatures (C) at TC_POS metres from the cold face of the carbon."""
    return T_w + q * (R_BACK + TC_POS / k_c)


def simulate_scenario(seed):
    rng = np.random.default_rng(seed)
    n = N_DAYS
    # per-furnace true properties
    k_c = rng.uniform(*K_C_RANGE)
    k_cer = K_CER_NOM * rng.uniform(0.8, 1.2)
    k_s = K_S_NOM * rng.uniform(0.85, 1.15)
    # starting wall (real ranges)
    grw0 = rng.uniform(1.58, 2.42)
    cer0 = rng.uniform(0.40, 0.50)
    carb0 = grw0 - cer0
    cer = 0.0 if rng.random() < P_CUP_GONE else cer0 * rng.uniform(0.1, 1.0)   # lining age
    carb = carb0 * (rng.uniform(0.80, 1.0) if cer < 0.05 else 1.0)
    cer_rate = rng.uniform(*CER_RATE_MM_YR) / 1000.0     # m/year
    carb_rate = rng.uniform(*CARBON_RATE_M_YR)           # m/year (used once ceramic is gone)
    # operating conditions
    T_hm = ar1(n, T_HM_NOM, 12.0, 0.95, rng)
    flow = make_regimes(n, rng)
    h_stag, h_act = rng.uniform(*H_STAG_RANGE), rng.uniform(*H_ACT_RANGE)
    h = np.where(flow == 0, h_stag, h_act) * (1.0 + 0.05 * ar1(n, 0.0, 1.0, 0.9, rng))
    T_w = np.clip(ar1(n, T_WATER_NOM, 3.0, 0.9, rng), 20, None)
    for _ in range(rng.integers(1, 5)):                  # upsets
        i0 = rng.integers(30, n - 30); d = rng.integers(3, 15)
        h[i0:i0 + d] *= rng.uniform(1.8, 3.0)
        T_hm[i0:i0 + d] += rng.uniform(15, 40)

    s = 0.0
    rows = []
    for t in range(n):
        R_rest = cer / k_cer + carb / k_c + R_BACK
        # target skull from the heat balance at the solidification isotherm
        target = skull_target(T_hm[t], T_w[t], h[t], R_rest, k_s)
        s += float(np.clip(target - s, -S_MELT, S_GROW))   # rate-limited skull
        q = heat_flux(T_hm[t], T_w[t], h[t], s, k_s, R_rest)
        tc = tc_temps(T_w[t], q, k_c)
        cup_gone = 0 if cer > 0.0 else 1
        rows.append([seed, t, *tc, cer + carb, cer, carb, s, q, T_hm[t], h[t], T_w[t], k_c, k_cer, cup_gone, flow[t]])
        protected = s > SKULL_THRESHOLD
        noise = rng.uniform(0.5, 1.5)
        if cer > 0.0:
            rate = cer_rate * (SKULL_WEAR_FACTOR[0] if protected else 1.0)
            cer = max(0.0, cer - rate / 365.0 * noise)
        else:
            rate = carb_rate * (SKULL_WEAR_FACTOR[1] if protected else 1.0)
            carb = max(CARBON_MIN, carb - rate / 365.0 * noise)

    cols = ["scenario", "day"] + [f"TC{i+1}_true" for i in range(len(TC_POS))] + \
           ["remaining_m", "ceramic_m", "carbon_m", "skull_m", "heat_flux_Wm2", "T_hotmetal",
            "h_eff", "T_water", "k_c_true", "k_cer_true", "cup_gone", "flow_regime"]
    return pd.DataFrame(rows, columns=cols)


def add_sensor_effects(df, seed, sigma=1.5, p_dropout=0.01):
    rng = np.random.default_rng(seed + 10_000)
    out = df.copy()
    for c in [c for c in df.columns if c.endswith("_true") and c.startswith("TC")]:
        x = df[c].to_numpy() + rng.normal(0, sigma, len(df))
        for i in np.where(rng.random(len(df)) < p_dropout)[0]:
            x[i:i + rng.integers(1, 6)] = np.nan
        out[c.replace("_true", "")] = x
    return out


def build_dataset(n_scenarios=200, out_csv="hearth_synthetic_v2.csv"):
    data = pd.concat([add_sensor_effects(simulate_scenario(s), s) for s in range(n_scenarios)], ignore_index=True)
    data.to_csv(out_csv, index=False)
    return data


if __name__ == "__main__":
    d = build_dataset(200)
    print(d.shape)
    print(d[["TC1", "TC2", "TC3", "remaining_m", "ceramic_m", "carbon_m", "skull_m", "heat_flux_Wm2"]].describe().round(2))
    print("days with skull > 0.05 m:", round((d.skull_m > 0.05).mean(), 2))
    print("days with ceramic gone  :", round((d.cup_gone == 1).mean(), 2))
    g = d.groupby("scenario")
    print("max one-day skull change (m): %.3f" % d.groupby("scenario").skull_m.diff().abs().max())
