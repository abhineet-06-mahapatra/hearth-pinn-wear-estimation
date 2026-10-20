"""
sensitivity_plots.py
How much do the three thermocouple readings move when the wall, skull or an uncertain
parameter changes? Uses the tested functions from hearth_simulator_v2.py.

Run from the repo folder:  python simulator/sensitivity_plots.py
Saves figures/sensitivity.png and figures/sensitivity_summary.csv. No plot window opens
unless you add --show.

Reference state (an illustration, not a typical furnace): hot metal 1480 C, water 40 C,
ceramic 0.30 m (k 5), carbon 1.50 m (k 20), skull 0.10 m (k 2), h = 25 W/m2K, backing 0.02.
"""
import os
import sys

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import hearth_simulator_v2 as sim  # noqa: E402

T_HM, T_W = 1480.0, 40.0
BASE = dict(h=25.0, s=0.10, k_s=2.0, cer=0.30, k_cer=5.0, carb=1.50, k_c=20.0, r_back=0.02)


def readings(**kw):
    """TC1, TC2, TC3 (C) and heat flux (W/m2) for the reference state with some values changed."""
    p = {**BASE, **kw}
    old = sim.R_BACK
    sim.R_BACK = p["r_back"]
    try:
        r_rest = p["cer"] / p["k_cer"] + p["carb"] / p["k_c"] + p["r_back"]
        q = sim.heat_flux(T_HM, T_W, p["h"], p["s"], p["k_s"], r_rest)
        return sim.tc_temps(T_W, q, p["k_c"]), q
    finally:
        sim.R_BACK = old


def tc3(**kw):
    return float(readings(**kw)[0][2])


# ---------- numbers for the summary ----------
base3 = tc3()
per_100mm = tc3(carb=BASE["carb"] - 0.10) - base3                    # 100 mm of carbon worn away
kc_effect = tc3(k_c=35.0) - tc3(k_c=13.0)                             # whole real conductivity range
skull_effect = tc3(s=0.0) - tc3(s=0.20)                               # skull of 0.20 m melts completely
rb_effect = tc3(r_back=0.0083) - tc3(r_back=0.02)                     # published equivalent vs assumed backing
cer_100 = tc3(cer=BASE["cer"] - 0.10) - base3                         # 100 mm of ceramic worn away
# skull melting (stagnant -> active flow) in a thick and in a thin wall
melt_thick = tc3(s=0.0, h=60.0) - tc3(s=0.20, h=25.0)
melt_thin = tc3(s=0.0, h=60.0, cer=0.0, carb=0.80) - tc3(s=0.20, h=25.0, cer=0.0, carb=0.80)

summary = pd.DataFrame([
    ["Reference TC3 reading", base3, ""],
    ["100 mm of carbon worn away", per_100mm, "wear signal per 100 mm (carbon)"],
    ["100 mm of ceramic worn away", cer_100, "wear signal per 100 mm (ceramic)"],
    ["Carbon conductivity 13 -> 35 W/m.K", kc_effect, f"equal to {abs(kc_effect / per_100mm) * 100:.0f} mm of carbon wear"],
    ["Skull 0.20 m melts away (same flow)", skull_effect, f"equal to {abs(skull_effect / per_100mm) * 100:.0f} mm of carbon wear"],
    ["Backing resistance 0.02 -> 0.0083", rb_effect, "shifts absolute readings, not the wear trend"],
    ["Skull melts with flow change, thick wall", melt_thick, "ceramic 0.30 m + carbon 1.50 m"],
    ["Skull melts with flow change, thin wall", melt_thin, "carbon 0.80 m, no ceramic"],
], columns=["case", "change_in_TC3_C", "note"])
summary["change_in_TC3_C"] = summary.change_in_TC3_C.round(1)
os.makedirs("figures", exist_ok=True)
summary.to_csv("figures/sensitivity_summary.csv", index=False)
print(summary.to_string(index=False))

# ---------- plots ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
cols = ["tab:blue", "tab:orange", "tab:green"]


def lines(a, xs, key, xlabel, title, scale=1.0):
    vals = np.array([readings(**{key: x})[0] for x in xs])
    for k in range(3):
        a.plot(np.array(xs) * scale, vals[:, k], color=cols[k], label=f"TC{k + 1}")
    a.set_xlabel(xlabel); a.set_ylabel("reading (C)"); a.set_title(title, fontsize=10); a.grid(alpha=0.3)


lines(ax[0, 0], np.linspace(0.7, 1.9, 25), "carb", "remaining carbon (mm)", "A. Wear: readings rise as carbon thins", 1000)
ax[0, 0].invert_xaxis(); ax[0, 0].legend(fontsize=8)
lines(ax[0, 1], np.linspace(12, 37, 25), "k_c", "carbon conductivity (W/m.K)", "B. Uncertain conductivity moves readings too")
lines(ax[0, 2], np.linspace(0, 0.3, 25), "s", "skull thickness (mm)", "C. Skull: readings fall as it grows", 1000)
lines(ax[1, 0], np.linspace(0.005, 0.03, 25), "r_back", "backing resistance (m2K/W)", "D. Backing resistance shifts all readings")
ax[1, 0].axvline(0.0083, color="gray", ls=":"); ax[1, 0].axvline(0.02, color="gray", ls="--")
ax[1, 0].text(0.0085, ax[1, 0].get_ylim()[0] + 5, "published\nequivalent", fontsize=7)
ax[1, 0].text(0.0205, ax[1, 0].get_ylim()[0] + 5, "simulator", fontsize=7)

labels = ["thick wall", "thin wall"]
ax[1, 1].bar(labels, [melt_thick, melt_thin], color=["tab:gray", "tab:red"])
for i, v in enumerate([melt_thick, melt_thin]):
    ax[1, 1].text(i, v, f"+{v:.0f} C", ha="center", va="bottom", fontsize=9)
ax[1, 1].set_ylabel("rise in TC3 (C)"); ax[1, 1].set_title("E. Skull melts (flow changes): TC3 jump", fontsize=10)
ax[1, 1].grid(alpha=0.3, axis="y")

names = ["100 mm carbon", "k_c 13 to 35", "skull 0.2 m", "backing", "skull melt\n(thin wall)"]
vals = [abs(per_100mm), abs(kc_effect), abs(skull_effect), abs(rb_effect), abs(melt_thin)]
ax[1, 2].barh(names, vals, color=["tab:brown", "tab:purple", "tab:teal" if False else "tab:cyan", "tab:olive", "tab:red"])
ax[1, 2].set_xlabel("size of change in TC3 (C)"); ax[1, 2].set_title("F. Wear signal against the other effects", fontsize=10)
ax[1, 2].invert_yaxis(); ax[1, 2].grid(alpha=0.3, axis="x")
fig.suptitle("Sensitivity of the thermocouple readings (reference state, not a typical furnace)", fontsize=12)
fig.tight_layout()
fig.savefig("figures/sensitivity.png", dpi=150)
print("\nSaved figures/sensitivity.png and figures/sensitivity_summary.csv")
if "--show" in sys.argv:
    plt.show()
