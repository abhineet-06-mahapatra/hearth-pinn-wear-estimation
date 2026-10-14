"""
hearth_simulator.py
Step 1 of the project: generate a labelled synthetic dataset of blast furnace
hearth sidewall thermocouple readings with KNOWN refractory thickness and skull thickness.

Model (quasi-steady 1D conduction, planar wall; upgrade to cylindrical/2D later):

    hot metal --(film h)--> [skull, k_s] --> [carbon block, k_c] --> cooling (R_cool) --> water

Skull thickness is NOT picked at random. It is the thickness at which the skull hot
face sits at the iron solidification temperature T_SOL (~1150 C). It therefore
depends on remaining refractory thickness, hot-side heat transfer and cooling.
Wear then accelerates when no skull protects the lining (positive feedback).

All parameter values are typical literature-range numbers. Check and cite your own.
Run:  python hearth_simulator.py
"""
import numpy as np
import pandas as pd

# ---------------- nominal parameters (edit these) ----------------
L0 = 1.00            # initial carbon block thickness, m
L_MIN = 0.40         # stop wear here so thermocouples stay inside refractory, m
TC_POS = np.array([0.10, 0.20, 0.30])   # thermocouple depth from COLD face, m
K_C_NOM = 15.0       # carbon block conductivity, W/m.K
K_S_NOM = 30.0       # skull (solidified iron) conductivity, W/m.K
R_COOL = 0.02        # stave + water film resistance, m2.K/W
T_SOL = 1150.0       # solidification isotherm of hot metal, C
T_WATER = 40.0       # cooling water temperature, C
H_NOM = 60.0         # EFFECTIVE hot-face heat transfer coeff., W/m2.K (tuned so skull forms as lining thins)
T_HM_NOM = 1480.0    # hot metal temperature, C
S_MAX = 0.60         # maximum skull thickness, m
N_DAYS = 730         # 2 years, daily steps
BASE_WEAR = 0.05 / 365   # m/day with a protective skull (~0.05 m/yr)
EXPOSED_FACTOR = 4.0     # wear multiplier when there is no skull


def ar1(n, mean, sd, phi, rng):
    """Slowly varying operating signal (stand-in for real plant data)."""
    x = np.zeros(n)
    x[0] = mean
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    for i in range(1, n):
        x[i] = mean + phi * (x[i - 1] - mean) + e[i]
    return x


def steady_state(L, T_hm, h, T_w, k_c, k_s):
    """Return (skull thickness s, heat flux q, cold-face temperature of carbon)."""
    R_rest = L / k_c + R_COOL                    # carbon + cooling
    # Skull present if solidification isotherm sits inside the hot metal.
    # Balance: h*(T_hm - T_SOL) = (T_SOL - T_w)/(s/k_s + R_rest)
    s = k_s * ((T_SOL - T_w) / (h * (T_hm - T_SOL)) - R_rest)
    if s > S_MAX:                                # geometric cap on skull growth
        s = S_MAX
        q = (T_hm - T_w) / (1.0 / h + s / k_s + R_rest)
    elif s > 0:
        q = h * (T_hm - T_SOL)
    else:
        s = 0.0
        q = (T_hm - T_w) / (1.0 / h + R_rest)    # hot metal touches carbon
    T_cold = T_w + q * R_COOL                    # carbon cold-face temperature
    return s, q, T_cold


def simulate_scenario(seed, drivers=None):
    rng = np.random.default_rng(seed)
    n = N_DAYS
    # per-scenario "true" properties differ from the nominal ones the physics
    # loss will use -> deliberate model mismatch so the network cannot just
    # memorise your formula
    k_c = K_C_NOM * rng.uniform(0.85, 1.15)
    k_s = K_S_NOM * rng.uniform(0.85, 1.15)
    wear_scale = rng.uniform(0.6, 2.0)

    if drivers is None:   # replace with columns from the Mendeley data if you like
        T_hm = ar1(n, T_HM_NOM, 12.0, 0.95, rng)
        h = np.clip(ar1(n, H_NOM * rng.uniform(0.8, 1.2), 8.0, 0.97, rng), 25, None)
        T_w = np.clip(ar1(n, T_WATER, 3.0, 0.9, rng), 20, None)
    else:
        T_hm, h, T_w = drivers

    # occasional hot-side upsets (e.g. unstable hearth flow) that melt the skull
    for _ in range(rng.integers(1, 5)):
        i0 = rng.integers(30, n - 30)
        d = rng.integers(3, 15)
        h[i0:i0 + d] *= rng.uniform(1.8, 3.0)
        T_hm[i0:i0 + d] += rng.uniform(15, 40)

    rows = []
    L = rng.uniform(0.60, L0)      # scenarios start at different lining ages
    for t in range(n):
        s, q, T_cold = steady_state(L, T_hm[t], h[t], T_w[t], k_c, k_s)
        tc_true = T_cold + q * TC_POS / k_c      # temperature inside the carbon
        rows.append([seed, t, *tc_true, L, s, q, T_hm[t], h[t], T_w[t]])
        rate = BASE_WEAR * wear_scale * (EXPOSED_FACTOR if s == 0.0 else 1.0)
        L = max(L_MIN, L - rate * rng.uniform(0.5, 1.5))

    cols = ["scenario", "day"] + [f"TC{i+1}_true" for i in range(len(TC_POS))] + \
           ["remaining_m", "skull_m", "heat_flux_Wm2", "T_hotmetal", "h_film", "T_water"]
    return pd.DataFrame(rows, columns=cols)


def add_sensor_effects(df, seed, sigma=1.5, p_dropout=0.01, drift_per_day=0.0):
    """Noise, drift and dropout/stuck faults on the thermocouple columns."""
    rng = np.random.default_rng(seed + 10_000)
    out = df.copy()
    for c in [c for c in df.columns if c.endswith("_true")]:
        name = c.replace("_true", "")
        x = df[c].to_numpy() + rng.normal(0, sigma, len(df)) + drift_per_day * df["day"]
        dead = rng.random(len(df)) < p_dropout
        for i in np.where(dead)[0]:          # a dropout lasts a few days
            x[i:i + rng.integers(1, 6)] = np.nan
        out[name] = x
    return out


def build_dataset(n_scenarios=200, out_csv="hearth_synthetic.csv"):
    frames = []
    for s in range(n_scenarios):
        frames.append(add_sensor_effects(simulate_scenario(s), s))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(out_csv, index=False)
    return data


if __name__ == "__main__":
    data = build_dataset(200)
    print(data.shape)
    print(data[["TC1", "TC2", "TC3", "remaining_m", "skull_m"]].describe().round(2))
    print("fraction of days with skull:", (data.skull_m > 0).mean().round(2))
