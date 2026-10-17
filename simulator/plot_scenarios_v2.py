"""
plot_scenarios_v2.py
Plots furnaces from the v2 (full-wall) dataset.
Needs hearth_synthetic_v2.csv in the repo folder (made by hearth_simulator_v2.py).
Run from the repo folder:  python simulator/plot_scenarios_v2.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("hearth_synthetic_v2.csv")
os.makedirs("figures", exist_ok=True)

first = d[d.day == 0]
normal = first[first.cup_gone == 0].scenario.tolist()[:2]    # ceramic wall still present
gone = first[first.cup_gone == 1].scenario.tolist()[:2]      # ceramic wall already gone
chosen = normal + gone

fig, ax = plt.subplots(len(chosen), 2, figsize=(12, 3 * len(chosen)), sharex=True)
for r, s in enumerate(chosen):
    x = d[d.scenario == s]
    kind = "ceramic present" if s in normal else "ceramic gone"
    for tc in ["TC1", "TC2", "TC3"]:
        ax[r, 0].plot(x.day, x[tc], lw=0.8, label=tc)
    ax[r, 0].set_ylabel(f"Furnace {s}\n({kind})\nTemp (C)")
    ax[r, 1].plot(x.day, x.ceramic_m * 1000, label="ceramic (mm)")
    ax[r, 1].plot(x.day, x.carbon_m * 1000, label="carbon (mm)")
    ax[r, 1].plot(x.day, x.skull_m * 1000, label="skull (mm)")
    ax[r, 1].set_ylabel("Thickness (mm)")
ax[0, 0].set_title("Thermocouples (noisy, with dropouts)")
ax[0, 1].set_title("True layer thicknesses")
ax[0, 0].legend(ncol=3, fontsize=8)
ax[0, 1].legend(ncol=3, fontsize=8)
for a in ax[-1]:
    a.set_xlabel("Day")
fig.tight_layout()
fig.savefig("figures/scenarios_v2_check.png", dpi=150)
print("Saved figures/scenarios_v2_check.png")
plt.show()
