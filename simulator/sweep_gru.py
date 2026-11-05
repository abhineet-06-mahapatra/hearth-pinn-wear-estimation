"""
sweep_gru.py  (Week 5)
Chooses the GRU's window length and size on VALIDATION furnaces only, BEFORE the physics-informed network exists,
so the later comparison cannot be tuned in favour of the physics.

Selection rule (fixed in advance): best mean validation R2 over the three targets, averaged over seeds.
If several settings are within TOLERANCE (0.01) of the best, the simplest wins (smallest length x hidden size).

All settings are scored on the SAME validation days (those a 120-day window can reach), so lengths are comparable.
Test sets are never loaded.

Run from the repo folder:  python simulator/sweep_gru.py   (about 5 to 15 minutes)   Saves figures/gru_sweep.csv.
Every finished run is saved to figures/gru_sweep_runs.csv, so if it is interrupted, running it again resumes.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import train_gru as tg  # noqa: E402
import windows as wd  # noqa: E402

LENGTHS = (30, 60, 120)
HIDDEN = (16, 32, 64)
SEEDS = 2
TOLERANCE = 0.01


RUNS_FILE = os.path.join("figures", "gru_sweep_runs.csv")


def completed(runs):
    """Set of (length, hidden, seed) already finished, so an interrupted sweep can resume instead of starting again."""
    return {(int(r.length), int(r.hidden), int(r.seed)) for r in runs.itertuples()}


def pick_best(table, tol=TOLERANCE):
    """table needs columns length, hidden, mean_R2. Best score wins; within tol of it, the simplest (length x hidden) wins."""
    best = table.mean_R2.max()
    close = table[table.mean_R2 >= best - tol].copy()
    close["size"] = close.length * close.hidden
    return close.sort_values(["size", "mean_R2"], ascending=[True, False]).iloc[0]


def main():
    frames = {n: pd.read_csv(os.path.join("data", f"{n}.csv")) for n in ("train", "val")}      # test sets are never loaded
    first_day = max(LENGTHS) - 1
    os.makedirs("figures", exist_ok=True)
    runs = pd.read_csv(RUNS_FILE) if os.path.exists(RUNS_FILE) else pd.DataFrame(
        columns=["length", "hidden", "seed", "skull", "iso", "wall", "mean_R2", "best_epoch"])
    done = completed(runs)
    if done:
        print(f"Resuming: {len(done)} runs already finished ({RUNS_FILE})")
    for length in LENGTHS:
        if all((length, h, sd) in done for h in HIDDEN for sd in range(SEEDS)):
            continue
        sets, _, y_std = wd.make_window_sets(frames, length=length, stride=5)
        keep = sets["val"]["day"] >= first_day                                                   # same validation days for every length
        Xva, yva_raw = sets["val"]["X"][keep], sets["val"]["y"][keep]
        ytr, yva = sets["train"]["y"] / y_std, yva_raw / y_std
        for hidden in HIDDEN:
            for seed in range(SEEDS):
                if (length, hidden, seed) in done:
                    continue
                t = time.time()
                model, hist = tg.fit_gru(sets["train"]["X"], ytr, Xva, yva, seed=seed, hidden=hidden, epochs=25, patience=4)
                r2 = tg.r2_per_target(yva_raw, tg.predict(model, Xva) * y_std)
                row = dict(length=length, hidden=hidden, seed=seed, skull=r2[0], iso=r2[1], wall=r2[2], mean_R2=float(np.mean(r2)),
                           best_epoch=int(np.argmin(hist)) + 1)
                runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
                runs.to_csv(RUNS_FILE, index=False)                                              # saved after every run
                print(f"  length {length:3d} hidden {hidden:2d} seed {seed}: R2 {np.round(r2, 3)}  ({time.time() - t:.0f} s)", flush=True)
    if len(runs) < len(LENGTHS) * len(HIDDEN) * SEEDS:
        print(f"{len(runs)} of {len(LENGTHS) * len(HIDDEN) * SEEDS} runs finished. Run the script again to continue.")
        return
    table = runs.groupby(["length", "hidden"], as_index=False)[["skull", "iso", "wall", "mean_R2", "best_epoch"]].mean()
    os.makedirs("figures", exist_ok=True)
    table.round(4).to_csv("figures/gru_sweep.csv", index=False)
    pd.set_option("display.width", 160)
    print("\nMean validation R2 over seeds (same validation days for every setting)")
    print(table.round(3).to_string(index=False))
    best = pick_best(table)
    print(f"\nChosen by the fixed rule: window length {int(best.length)}, hidden size {int(best.hidden)} (mean R2 {best.mean_R2:.3f})")
    print("Saved figures/gru_sweep.csv. Test sets were not used.")


if __name__ == "__main__":
    main()
