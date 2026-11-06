"""
train_pinn.py  (Week 6)
Trains the same GRU as train_gru.py with four loss settings and compares them on VALIDATION furnaces only:
    (0, 0)  data only                 (lam_bounds, lam_mono) = the weights of the two physics terms in physics_loss.py (which you wrote)
    (1, 0)  + impossible-value penalty
    (0, 1)  + no-regrowth penalty
    (1, 1)  both
Everything else is identical: window 60, hidden 64, same windows (those with a next window), same seeds, same optimiser, and the same
early-stopping rule, which uses the ORDINARY data loss on validation furnaces for every setting. So any difference comes from the physics terms.

Besides R2 it reports how often predictions break the two physical facts, measured by a separate numpy check (violation_rates), not by your loss:
    bounds_violation: share of windows with an impossible absolute value (more than 1 cm past a limit)
    regrowth:         share of consecutive window pairs where the predicted wall change rises by more than 2 cm

Test sets are never loaded. Every finished run is saved to figures/pinn_runs.csv, so an interrupted run resumes.
Run from the repo folder after physics_loss.py is written:  python simulator/train_pinn.py     (roughly 5 to 15 minutes)
Options: --seeds 3 --epochs 40 --configs 0:0,1:0,0:1,1:1 --quick
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import physics_loss as pl  # noqa: E402
import train_gru as tg  # noqa: E402
import windows as wd  # noqa: E402

LENGTH, HIDDEN, STRIDE = 60, 64, 5
DEFAULT_CONFIGS = "0:0,1:0,0:1,1:1"
RUNS_FILE = os.path.join("figures", "pinn_runs.csv")
TARGET_NAMES = ["skull", "iso", "wall"]


def violation_rates(pred_abs, wall_change, nxt, tol_bounds=0.01, tol_growth=0.02):
    """Independent numpy check of the two physical facts.
    pred_abs: (N, 3) absolute predictions in metres [skull, isotherm, wall]. wall_change: (N,) predicted wall change in metres.
    nxt: index of the next window of the same furnace, or -1."""
    skull, iso, wall = pred_abs[:, 0], pred_abs[:, 1], pred_abs[:, 2]
    bad = (skull < -tol_bounds) | (iso < -tol_bounds) | (wall < -tol_bounds) | (iso - wall - skull > tol_bounds)
    has = nxt >= 0
    growth = wall_change[nxt[has]] - wall_change[has]
    return float(bad.mean()), float((growth > tol_growth).mean()) if has.any() else float("nan")


def fit_pinn(sets, y_std, seed, lam_b, lam_m, hidden=HIDDEN, epochs=40, patience=6, lr=2e-3, batch=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = sets["train"], sets["val"]
    X = torch.as_tensor(tr["X"], dtype=torch.float32)
    Y = torch.as_tensor(tr["y"] / y_std, dtype=torch.float32)
    Y0 = torch.as_tensor(tr["y0"], dtype=torch.float32)
    ystd_t = torch.as_tensor(y_std, dtype=torch.float32)
    nxt = wd.pair_index(tr["scenario"], tr["day"], STRIDE)
    idx_all = np.flatnonzero(nxt >= 0)                                     # same training windows for every setting
    yva = va["y"] / y_std
    model = tg.GRUNet(X.shape[-1], hidden, 1, 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, since, hist = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(idx_all)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            pred = model(X[idx])
            later = model(X[nxt[idx]]) if lam_m > 0 else None
            out = pl.combined_loss(pred, Y[idx], Y0[idx], ystd_t, pred_later=later, lam_bounds=lam_b, lam_mono=lam_m)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = float(np.mean((tg.predict(model, va["X"]) - yva) ** 2))   # ordinary data loss for every setting
        hist.append(val_loss)
        if val_loss < best - 1e-5:
            best, since = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def evaluate(model, sets, y_std):
    va = sets["val"]
    pred = tg.predict(model, va["X"]) * y_std                              # metres, change since day 0
    r2 = tg.r2_per_target(va["y"], pred)
    nxt = wd.pair_index(va["scenario"], va["day"], STRIDE)
    bounds_rate, regrowth_rate = violation_rates(va["y0"] + pred, pred[:, 2], nxt)
    return r2, bounds_rate, regrowth_rate


def parse_configs(text):
    return [tuple(float(x) for x in item.split(":")) for item in text.split(",")]


def main(argv):
    arg = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
    quick = "--quick" in argv
    n_seeds = arg("--seeds", 1 if quick else 3)
    epochs = arg("--epochs", 3 if quick else 40)
    configs = parse_configs(arg("--configs", "0:0,1:1" if quick else DEFAULT_CONFIGS))
    hidden = 16 if quick else HIDDEN
    frames = {n: pd.read_csv(os.path.join("data", f"{n}.csv")) for n in ("train", "val")}      # test sets are never loaded
    if quick:
        keep = sorted(frames["train"].scenario.unique())[:40]
        frames["train"] = frames["train"][frames["train"].scenario.isin(keep)]
    sets, scaler, y_std = wd.make_window_sets(frames, length=LENGTH, stride=STRIDE)
    os.makedirs("figures", exist_ok=True)
    cols = ["lam_bounds", "lam_mono", "seed", "skull", "iso", "wall", "bounds_violation", "regrowth", "best_epoch"]
    runs = pd.read_csv(RUNS_FILE) if os.path.exists(RUNS_FILE) else pd.DataFrame(columns=cols)
    done = {(r.lam_bounds, r.lam_mono, int(r.seed)) for r in runs.itertuples()}
    for lam_b, lam_m in configs:
        for seed in range(n_seeds):
            if (lam_b, lam_m, seed) in done:
                continue
            t = time.time()
            model, hist = fit_pinn(sets, y_std, seed, lam_b, lam_m, hidden=hidden, epochs=epochs)
            r2, bounds_rate, regrowth_rate = evaluate(model, sets, y_std)
            row = dict(lam_bounds=lam_b, lam_mono=lam_m, seed=seed, skull=r2[0], iso=r2[1], wall=r2[2], bounds_violation=bounds_rate,
                       regrowth=regrowth_rate, best_epoch=int(np.argmin(hist)) + 1)
            runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
            runs.to_csv(RUNS_FILE, index=False)
            print(f"  lam_bounds {lam_b:g} lam_mono {lam_m:g} seed {seed}: R2 {np.round(r2, 3)}  bounds {bounds_rate:.3f}  regrowth {regrowth_rate:.3f}  ({time.time() - t:.0f} s)", flush=True)
    wanted = {(b, m) for b, m in configs}
    sel = runs[[(r.lam_bounds, r.lam_mono) in wanted and int(r.seed) < n_seeds for r in runs.itertuples()]]
    table = sel.groupby(["lam_bounds", "lam_mono"])[["skull", "iso", "wall", "bounds_violation", "regrowth"]].agg(["mean", "std"]).round(3)
    table.to_csv("figures/pinn_scores.csv")
    pd.set_option("display.width", 200)
    print("\nValidation results, mean and spread over seeds (R2: higher is better; violation rates: lower is better)")
    print(table.to_string())
    print("\n(0, 0) is the data-only network. Saved figures/pinn_scores.csv. Test sets were not used.")


if __name__ == "__main__":
    main(sys.argv[1:])
