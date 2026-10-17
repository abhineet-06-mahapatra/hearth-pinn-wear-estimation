"""
verify_wear_rates.py
Re-derives whole-wall wear rates from the tables printed in
Boumahdi et al., "Saint-Gobain's Ceramic Cup: an update", AISTech 2017
(Figures 6, 7, 9). Numbers below are copied from those tables.

IMPORTANT: the paper's "in-situ measure" is the output of a thermal wear model
driven by thermocouples (e.g. Mothus), not a dissection measurement, so these
rates are model-based field estimates.
Run:  python simulator/verify_wear_rates.py
"""
import pandas as pd

# (furnace, years_after_blow_in, level, elevation_m, initial_GRW_m, measured_GRW_m, paper_GRW_pct, paper_CW_pct)
rows = [
    ("Dunkerque BF4", 8.0, "01", 2.40, 2.42, 2.68, 0, 0),
    ("Dunkerque BF4", 8.0, "02", 3.10, 2.40, 2.47, 0, 0),
    ("Dunkerque BF4", 8.0, "03", 3.80, 2.38, 2.16, 9, 54),
    ("Dunkerque BF4", 8.0, "04", 4.38, 2.35, 2.06, 13, 74),
    ("Fos BF1", 8.0, "02", 5.95, 1.83, 1.80, 1, 5),
    ("Fos BF1", 8.0, "03", 7.15, 1.73, 1.58, 9, 37),
    ("Fos BF1", 8.0, "04", 8.40, 1.73, 1.60, 7, 31),
    ("Fos BF1", 8.0, "05", 9.80, 1.73, 1.59, 8, 35),
    ("Fos BF1", 8.0, "06", 11.15, 1.58, 1.39, 12, 48),
    ("Fos BF2", 4.5, "02", 5.95, 1.83, 1.80, 1, 5),
    ("Fos BF2", 4.5, "03", 7.15, 1.73, 1.56, 10, 41),
    ("Fos BF2", 4.5, "04", 8.40, 1.73, 1.59, 8, 33),
    ("Fos BF2", 4.5, "05", 9.80, 1.73, 1.60, 7, 31),
    ("Fos BF2", 4.5, "06", 11.15, 1.58, 1.43, 10, 38),
]
d = pd.DataFrame(rows, columns=["furnace", "years", "level", "elev_m", "init_m", "meas_m", "paper_GRW_pct", "paper_CW_pct"])
d["loss_m"] = (d.init_m - d.meas_m).round(3)
d["my_GRW_pct"] = (100 * d.loss_m / d.init_m).round(1)
d["rate_mm_per_yr"] = (1000 * d.loss_m / d.years).round(1)
d["rate_if_7.5yr"] = (1000 * d.loss_m / 7.5).round(1)
d["CW_pct_if_0.40m"] = (100 * d.loss_m / 0.40).round(0)

print(d.drop(columns=["init_m"]).to_string(index=False))

worn = d[d.loss_m > 0.1]
print("\nZones with more than 0.1 m loss (n=%d):" % len(worn))
print("  rate range  : %.0f to %.0f mm/year (using the stated years)" % (worn.rate_mm_per_yr.min(), worn.rate_mm_per_yr.max()))
print("  median rate : %.0f mm/year" % worn.rate_mm_per_yr.median())
print("Dunkerque BF4 at 7.5 years instead of 8: %.0f to %.0f mm/year"
      % (d[(d.furnace == "Dunkerque BF4") & (d.loss_m > 0.1)]["rate_if_7.5yr"].min(),
         d[(d.furnace == "Dunkerque BF4") & (d.loss_m > 0.1)]["rate_if_7.5yr"].max()))
print("\nNegative loss means thickness above the initial value (iron build-up, per the paper):")
print(d[d.loss_m < 0][["furnace", "level", "loss_m"]].to_string(index=False))
print("\nDifferences between the paper's printed % and recomputed % (rounding or averaging):")
d["diff_GRW_pct"] = (d.paper_GRW_pct - d.my_GRW_pct).round(1)
print(d[d.diff_GRW_pct.abs() >= 1][["furnace", "level", "paper_GRW_pct", "my_GRW_pct"]].to_string(index=False))
