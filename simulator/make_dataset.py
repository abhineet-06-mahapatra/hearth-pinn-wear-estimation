"""
make_dataset.py
Generates the training, validation and test data with the time-dependent model, splits it BY FURNACE
(no furnace appears in two splits), and writes a manifest with checksums so the data can be frozen.

Run from the repo folder:  python simulator/make_dataset.py
Writes data/train.csv, val.csv, test.csv, test_ood.csv and data/dataset_manifest.json.
The CSV files are large and are not committed to Git; commit only the manifest.

Splits
  train, val, test : furnaces drawn from the same ranges (70% / 15% / 15% of 200 furnaces)
  test_ood         : 20 furnaces whose carbon conductivity is ABOVE the real range (38 to 45 W/m.K
                     against 13 to 35 in training). It tests whether a model copes with conditions it
                     has never seen. 38 to 45 is a stress test, not a real carbon grade.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import hearth_simulator_v2 as sim  # noqa: E402
import hearth_transient as tr  # noqa: E402

SPLIT_SEED = 2026
OOD_SEED_START = 1000
OOD_K_C_RANGE = (38.0, 45.0)

# what a model may see
INPUT_COLUMNS = ["TC1", "TC2", "TC3"]
# what a model must estimate
LABEL_COLUMNS = ["remaining_m", "ceramic_m", "carbon_m", "skull_m", "iso1150_m"]
# kept for analysis only: using any of these as model input would leak the answer
DIAGNOSTIC_COLUMNS = ["TC1_true", "TC2_true", "TC3_true", "TC1_ss", "TC2_ss", "TC3_ss", "heat_flux_Wm2",
                      "T_hotmetal", "h_eff", "T_water", "k_c_true", "k_cer_true", "cup_gone", "flow_regime"]
ID_COLUMNS = ["scenario", "day"]


def make_furnace(seed):
    return sim.add_sensor_effects(tr.simulate_scenario_transient(seed), seed)


def assign_splits(n_in=200, split_seed=SPLIT_SEED, fractions=(0.70, 0.15)):
    perm = np.random.default_rng(split_seed).permutation(n_in)
    n_train, n_val = int(round(fractions[0] * n_in)), int(round(fractions[1] * n_in))
    return {"train": sorted(int(s) for s in perm[:n_train]),
            "val": sorted(int(s) for s in perm[n_train:n_train + n_val]),
            "test": sorted(int(s) for s in perm[n_train + n_val:])}


def build_all(n_in=200, n_ood=20, verbose=False):
    splits = assign_splits(n_in)
    frames = {}
    for name, seeds in splits.items():
        frames[name] = pd.concat([make_furnace(s) for s in seeds], ignore_index=True)
        if verbose:
            print(f"  {name}: {len(seeds)} furnaces")
    old = sim.K_C_RANGE
    sim.K_C_RANGE = OOD_K_C_RANGE
    try:
        ood_seeds = [OOD_SEED_START + i for i in range(n_ood)]
        frames["test_ood"] = pd.concat([make_furnace(s) for s in ood_seeds], ignore_index=True)
    finally:
        sim.K_C_RANGE = old
    splits["test_ood"] = ood_seeds
    return splits, frames


def checksum(df):
    """SHA-256 of the data rounded to 4 decimals, so tiny floating-point differences do not matter."""
    text = df[sorted(df.columns)].round(4).to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest(splits, frames):
    return {
        "split_seed": SPLIT_SEED,
        "inputs": INPUT_COLUMNS, "labels": LABEL_COLUMNS, "diagnostic_columns_do_not_use_as_inputs": DIAGNOSTIC_COLUMNS,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "splits": {name: {"furnace_seeds": splits[name], "rows": int(len(frames[name])), "sha256": checksum(frames[name])}
                   for name in frames},
    }


def summary(frames):
    rows = []
    for name, d in frames.items():
        rows.append(dict(split=name, furnaces=d.scenario.nunique(), rows=len(d),
                         missing_TC_share=round(float(d[INPUT_COLUMNS].isna().mean().mean()), 4),
                         k_c_min=round(float(d.k_c_true.min()), 1), k_c_max=round(float(d.k_c_true.max()), 1),
                         skull_present=round(float((d.skull_m > 0.05).mean()), 2),
                         remaining_median=round(float(d.remaining_m.median()), 2),
                         iso_median=round(float(d.iso1150_m.median()), 2)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = "data"
    os.makedirs(out, exist_ok=True)
    print("Generating furnaces with the time-dependent model ...")
    splits, frames = build_all(verbose=True)
    keep = ID_COLUMNS + INPUT_COLUMNS + LABEL_COLUMNS + DIAGNOSTIC_COLUMNS
    for name, d in frames.items():
        d[keep].to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, "dataset_manifest.json"), "w") as f:
        json.dump(manifest(splits, frames), f, indent=1)
    print(summary(frames).to_string(index=False))
    print("\nSaved data/*.csv and data/dataset_manifest.json (commit only the manifest).")
    print("Checksums:")
    for name in frames:
        print(f"  {name:9s} {checksum(frames[name])[:16]}...")
