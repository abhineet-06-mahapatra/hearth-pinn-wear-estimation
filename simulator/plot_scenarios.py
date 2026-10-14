"""Plot a few simulated scenarios so you can check they look physically sensible.
Run from the repo folder:  python simulator/plot_scenarios.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

SCENARIOS = [0, 1, 2]
data = pd.read_csv("hearth_synthetic.csv")
os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(12, 3.2 * len(SCENARIOS)), sharex=True)
for row, s in enumerate(SCENARIOS):
    d = data[data.scenario == s]
    ax = axes[row, 0]
    for tc in ["TC1", "TC2", "TC3"]:
        ax.plot(d.day, d[tc], lw=0.8, label=tc)
    ax.set_ylabel(f"Scenario {s}\nTemperature (C)")
    if row == 0:
        ax.set_title("Thermocouple readings (noisy, with dropouts)")
        ax.legend(loc="upper left", ncol=3, fontsize=8)

    ax = axes[row, 1]
    ax.plot(d.day, d.remaining_m * 1000, color="tab:brown", label="Remaining carbon (mm)")
    ax.plot(d.day, d.skull_m * 1000, color="tab:blue", label="Skull (mm)")
    ax.set_ylabel("Thickness (mm)")
    if row == 0:
        ax.set_title("True labels")
        ax.legend(loc="upper right", fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("Day")
fig.tight_layout()
fig.savefig("figures/scenarios_check.png", dpi=150)
print("Saved figures/scenarios_check.png")
plt.show()
