"""
extrapolation_check.py
Tests the idea that the 1150 C isotherm can be found by extending a straight line through two
thermocouple readings (no conductivity needed). Result in Week 2: it overshoots by about a
metre, because the 1150 C line usually lies in the ceramic or skull, where the heat flows
through a different conductivity than in the carbon where the sensors sit.

Run from the repo folder:  python simulator/extrapolation_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import hearth_simulator_v2 as sim  # noqa: E402

rows = []
for sid in range(120):
    d = sim.add_sensor_effects(sim.simulate_scenario(sid), sid).iloc[::10]
    d = d.dropna(subset=["TC1", "TC3"])
    # straight line through TC1 (0.2 m) and TC3 (0.6 m from the cold face), extended to 1150 C
    est = sim.TC_POS[2] + (sim.T_SOL - d.TC3) * (sim.TC_POS[2] - sim.TC_POS[0]) / (d.TC3 - d.TC1)
    rows.append(pd.DataFrame(dict(cup=d.cup_gone, true=d.iso1150_m, est=est, rem=d.remaining_m)))
e = pd.concat(rows, ignore_index=True)
e["err"] = e.est - e.true

print("rows:", len(e))
print("true isotherm distance (m): min %.2f  median %.2f  max %.2f" % (e.true.min(), e.true.median(), e.true.max()))
for name, g in [("all", e), ("ceramic present", e[e.cup == 0]), ("ceramic gone", e[e.cup == 1])]:
    print(f"{name:16s} median abs error {g.err.abs().median():.2f} m | 90th percentile {g.err.abs().quantile(.9):.2f} m | bias {g.err.median():+.2f} m | corr(true, remaining wall) {np.corrcoef(g.true, g.rem)[0, 1]:.2f}")
