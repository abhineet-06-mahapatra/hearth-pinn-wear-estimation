"""
compare_to_published.py
Compares the v2 simulator with values reported in published hearth studies.
Run from the repo folder:  python simulator/compare_to_published.py
Outputs: figures/published_comparison.csv and figures/published_comparison.png

Every published number below comes from a source listed in parameters.csv.
Several are alarm or threshold values, not typical operating values, so a match is
only a weak test. Rows marked NOT VERIFIED have no published value yet.
To add real thermocouple data, fill TABLE4_TC_C (degrees C) from Table 4 of
Wang et al., Processes 11(2), 531 (2023) (the table lists kelvin; subtract 273).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import hearth_simulator_v2 as sim  # noqa: E402

N_FURNACES = 200
TABLE4_TC_C = []        # <- paste measured thermocouple temperatures in C here (empty = not verified)

d = pd.concat([sim.simulate_scenario(s) for s in range(N_FURNACES)], ignore_index=True)
flux = d.heat_flux_Wm2 / 1000.0
skull_present = d.skull_m[d.skull_m > 0.05]

# per-furnace ceramic wear rate (mm/year) for furnaces that keep their ceramic wall for the whole window
wear = []
for _, g in d.groupby("scenario"):
    if g.ceramic_m.iloc[-1] > 0:
        wear.append(1000.0 * (g.ceramic_m.iloc[0] - g.ceramic_m.iloc[-1]) / (len(g) / 365.0))
wear = np.array(wear)


def pct(x, qs=(5, 50, 95)):
    return [float(np.percentile(x, q)) for q in qs]


rows = []


def add(quantity, unit, pub_lo, pub_hi, pub_note, sim_vals, median_check=True, verdict=None):
    p5, p50, p95 = sim_vals
    if verdict is None:
        verdict = "median inside published range" if pub_lo <= p50 <= pub_hi else "median OUTSIDE published range"
    rows.append(dict(quantity=quantity, unit=unit, published_low=pub_lo, published_high=pub_hi,
                     published_note=pub_note, sim_p5=round(p5, 3), sim_p50=round(p50, 3), sim_p95=round(p95, 3),
                     verdict=verdict))


add("Sidewall heat flux", "kW/m2", 8.1, 17.4,
    "risk-band boundaries at one furnace (AIP Adv. 10, 025308); critical values 9.4-57 across 13 furnaces",
    pct(flux))
add("Skull thickness when present (above 0.05 m)", "m", 0.07, 0.26,
    "build-up above original thickness at two levels (Boumahdi et al., AISTech 2017); only two data points",
    pct(skull_present))
add("Ceramic-wall wear rate (furnaces keeping ceramic)", "mm/year", 16, 38,
    "derived from field tables (verify_wear_rates.py); thermocouple-model estimates",
    pct(wear))
add("Hot metal temperature", "C", 1400, 1600,
    "range quoted by Boumahdi et al.; Processes 11(2), 531 studied about 1442-1559 C",
    pct(d.T_hotmetal))
add("Thinnest carbon reached", "m", 0.248, 0.475,
    "end-of-life residual carbon (Metals 13(8), 1371; ISIJ sidewall erosion study)",
    [float(d.carbon_m.min())] * 3,
    verdict="simulator does NOT reach end-of-life thickness (floor %.2f m)" % sim.CARBON_MIN)
melt_days = sim.S_MAX / sim.S_MELT
rows.append(dict(quantity="Time to melt a full skull", unit="days", published_low=np.nan, published_high=30,
                 published_note="a majority of the skull melted within about one month (Helle et al. 2022)",
                 sim_p5=np.nan, sim_p50=round(melt_days, 1), sim_p95=np.nan,
                 verdict="consistent in order of magnitude (set by S_MAX / S_MELT)"))

if TABLE4_TC_C:
    tc_pub = np.array(TABLE4_TC_C, dtype=float)
    tc_all = np.concatenate([d.TC1_true, d.TC2_true, d.TC3_true])
    add("Thermocouple temperature (all three sensors)", "C", float(tc_pub.min()), float(tc_pub.max()),
        "Table 4 of Processes 11(2), 531 (2023)", pct(tc_all))
else:
    rows.append(dict(quantity="Thermocouple temperature (TC1 / TC2 / TC3 medians)", unit="C", published_low=np.nan,
                     published_high=np.nan, published_note="no verified published values yet (fill TABLE4_TC_C)",
                     sim_p5=round(float(d.TC1_true.median()), 0), sim_p50=round(float(d.TC2_true.median()), 0),
                     sim_p95=round(float(d.TC3_true.median()), 0), verdict="NOT VERIFIED"))

table = pd.DataFrame(rows)
os.makedirs("figures", exist_ok=True)
table.to_csv("figures/published_comparison.csv", index=False)

pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(table[["quantity", "unit", "published_low", "published_high", "sim_p5", "sim_p50", "sim_p95", "verdict"]].to_string(index=False))

# ---- plot: published band vs simulated 5th-95th percentile, one panel per quantity ----
panels = table[table.published_low.notna() & table.sim_p95.notna() & (table.quantity != "Thinnest carbon reached")]
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
axes = np.atleast_1d(axes)
for ax, (_, r) in zip(axes, panels.iterrows()):
    ax.axhspan(r.published_low, r.published_high, color="tab:green", alpha=0.25, label="published range")
    ax.vlines(0, r.sim_p5, r.sim_p95, color="tab:blue", lw=6, label="simulated 5-95%")
    ax.plot([0], [r.sim_p50], "wo", mec="k", label="simulated median")
    ax.set_xlim(-1, 1); ax.set_xticks([])
    ax.set_title(r.quantity.split(" (")[0], fontsize=9)
    ax.set_ylabel(r.unit)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
fig.suptitle("Simulator versus published values", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/published_comparison.png", dpi=150)
print("\nSaved figures/published_comparison.csv and figures/published_comparison.png")
plt.show()
