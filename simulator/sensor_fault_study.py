"""
sensor_fault_study.py
How do real plant temperature sensors fail? Reads the Mendeley blast furnace file and measures
out-of-range readings, how long they last, whether several sensors fail together, and a
noise ceiling. This is a PROXY: the sensors are upper-furnace gas and air temperatures, not
thermocouples buried in the hearth lining.

Run from the repo folder:  python simulator/sensor_fault_study.py [path to the Mendeley zip]
If no path is given it looks in your Downloads folder.
Saves figures/sensor_fault_study.csv.
"""
import glob
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

COLS = ["Tt1", "Tt2", "Tt3", "Tt4"] + [f"Tp{i}" for i in range(1, 11)] + ["Tc"]
LOW, HIGH = 0.0, 1000.0            # readings outside this range (e.g. 3311, negatives) count as faults


def find_zip():
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "Process*data*.zip"))
    if not hits:
        sys.exit("Mendeley zip not found. Pass its path: python simulator/sensor_fault_study.py <path>")
    return sorted(hits)[-1]


def run_lengths(mask):
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return np.array(out)


with zipfile.ZipFile(find_zip()) as z:
    name = [n for n in z.namelist() if n.endswith("1_blast_furnace_data_first_dataset.xlsx")][0]
    sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None)
d = [v for v in sheets.values() if v.shape[0] > 1000][0]
d["dt"] = pd.to_datetime(d["dt"])
days = len(d) / 24.0
print(f"{len(d)} hourly rows, {d.dt.min().date()} to {d.dt.max().date()} ({days:.0f} days); "
      f"timestamp gaps: {int((d.dt.diff().dt.total_seconds().div(3600).dropna() != 1).sum())}")

bad = pd.DataFrame({c: ~((d[c] > LOW) & (d[c] < HIGH)) for c in COLS})
rows = []
for c in COLS:
    r = run_lengths(bad[c])
    rows.append(dict(sensor=c, missing_cells=int(d[c].isna().sum()), bad_hours=int(bad[c].sum()), events=len(r),
                     mean_event_hours=round(float(r.mean()), 1) if len(r) else 0.0,
                     longest_event_hours=int(r.max()) if len(r) else 0))
table = pd.DataFrame(rows)
os.makedirs("figures", exist_ok=True)
table.to_csv("figures/sensor_fault_study.csv", index=False)
print(table.to_string(index=False))

events = int(table.events.sum())
mean_h = float((table.events * table.mean_event_hours).sum() / events)
print(f"\nFault events: {events} in total = {100 * events / len(COLS) / days:.2f} per sensor per 100 days; mean length {mean_h:.0f} hours ({mean_h / 24:.1f} days)")
n_bad = bad.sum(axis=1)
print(f"Hours with at least one faulty sensor: {int((n_bad > 0).sum())} ({100 * (n_bad > 0).mean():.1f}%); "
      f"exactly one sensor: {int((n_bad == 1).sum())}; all {len(COLS)} at once: {int((n_bad == len(COLS)).sum())}")
vals = pd.Series(d[COLS].values.ravel())
print("Most common out-of-range values:", vals[(vals <= LOW) | (vals >= HIGH)].round(0).value_counts().head(4).to_dict())
g = d["Tc"].where((d["Tc"] > LOW) & (d["Tc"] < HIGH)).diff().dropna()
print(f"Noise ceiling from the smoothest sensor (Tc): hour-to-hour robust SD {1.4826 * (g - g.median()).abs().median() / np.sqrt(2):.2f} C "
      "(includes real process change, so true sensor noise is at most this)")
