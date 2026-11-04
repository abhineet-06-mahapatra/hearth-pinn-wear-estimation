"""
train_gru.py  (Week 5)
A data-only GRU that estimates the CHANGE since day 0 of skull, 1150 C distance and remaining wall from 60-day windows
of the three thermocouples. No physics in it: this is the network the physics-informed one (Week 6) has to beat.

Rules: trains on train.csv, picks the best epoch on val.csv, reports val.csv. test.csv and test_ood.csv are never loaded here.

Run from the repo folder:  python simulator/train_gru.py            (3 seeds, about 5 to 15 minutes on a laptop CPU)
Options: --seeds 3 --length 60 --hidden 32 --epochs 40 --quick (small and fast, for checking the setup)
Saves figures/gru_scores.csv. Model weights are not saved to Git.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import baselines as bl  # noqa: E402
import windows as wd  # noqa: E402

TARGET_NAMES = ["skull change", "1150 C distance change", "wall change"]


class GRUNet(nn.Module):
    def __init__(self, n_in=9, hidden=32, layers=1, n_out=3):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def predict(model, X, batch=1024):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(X[i:i + batch], dtype=torch.float32)).numpy() for i in range(0, len(X), batch)])


def fit_gru(Xtr, ytr, Xva, yva, seed=0, hidden=32, epochs=40, patience=6, lr=2e-3, batch=128, verbose=False):
    """Train with Adam on mean squared error of standardised targets. Returns the model with the best validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GRUNet(Xtr.shape[-1], hidden, 1, ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best, best_state, since, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = float(np.mean((predict(model, Xva) - yva) ** 2))
        history.append(val_loss)
        if verbose:
            print(f"    epoch {epoch + 1:2d}  val loss {val_loss:.4f}")
        if val_loss < best - 1e-5:
            best, since = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def summary_features(X):
    """Hand-made summary of a window for ridge / trees: last day (change and level) and the last 30 days' mean and minimum of the changes."""
    last = X[:, -1, :6]
    recent = X[:, -30:, :3]
    return np.concatenate([last, recent.mean(axis=1), recent.min(axis=1), X[:, :, 6:].mean(axis=1)], axis=1)


def window_baselines(sets):
    """Ridge and boosted trees on window summaries: same windows, same targets as the GRU."""
    Ftr, Fva = summary_features(sets["train"]["X"]), summary_features(sets["val"]["X"])
    out = {}
    for name, make in [("ridge on window summary", lambda: make_pipeline(StandardScaler(), Ridge(1.0))),
                       ("boosted trees on window summary", lambda: HistGradientBoostingRegressor(max_iter=200, random_state=0))]:
        r2 = []
        for j in range(3):
            m = make().fit(Ftr, sets["train"]["y"][:, j])
            r2.append(bl.metrics(sets["val"]["y"][:, j], m.predict(Fva))["R2"])
        out[name] = r2
    return out


def r2_per_target(y, p):
    return [bl.metrics(y[:, j], p[:, j])["R2"] for j in range(3)]


def main(argv):
    arg = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
    quick = "--quick" in argv
    n_seeds, length, hidden = arg("--seeds", 1 if quick else 3), arg("--length", 60), arg("--hidden", 16 if quick else 32)
    epochs = arg("--epochs", 5 if quick else 40)
    frames = {n: pd.read_csv(os.path.join("data", f"{n}.csv")) for n in ("train", "val")}      # test sets are never loaded
    if quick:
        keep = sorted(frames["train"].scenario.unique())[:40]
        frames["train"] = frames["train"][frames["train"].scenario.isin(keep)]
    sets, scaler, y_std = wd.make_window_sets(frames, length=length, stride=5)
    ytr, yva = sets["train"]["y"] / y_std, sets["val"]["y"] / y_std
    print(f"windows: train {sets['train']['X'].shape}, val {sets['val']['X'].shape}")
    rows = []
    for name, r2 in window_baselines(sets).items():
        rows.append(dict(model=name, seed="-", **dict(zip(TARGET_NAMES, r2))))
    for seed in range(n_seeds):
        t = time.time()
        model, hist = fit_gru(sets["train"]["X"], ytr, sets["val"]["X"], yva, seed=seed, hidden=hidden, epochs=epochs)
        r2 = r2_per_target(sets["val"]["y"], predict(model, sets["val"]["X"]) * y_std)
        rows.append(dict(model="GRU (data only)", seed=seed, **dict(zip(TARGET_NAMES, r2))))
        print(f"  GRU seed {seed}: best epoch {int(np.argmin(hist)) + 1}/{len(hist)}, {time.time() - t:.0f} s, val R2 {np.round(r2, 3)}")
    table = pd.DataFrame(rows)
    os.makedirs("figures", exist_ok=True)
    table.round(4).to_csv("figures/gru_scores.csv", index=False)
    grp = table.groupby("model")[TARGET_NAMES].agg(["mean", "std"]).round(2)
    print("\nValidation R2 (mean and standard deviation over seeds; higher is better)")
    for model, row in grp.iterrows():
        cells = [f"{row[(t, 'mean')]:5.2f} +/- {0 if np.isnan(row[(t, 'std')]) else row[(t, 'std')]:.2f}" for t in TARGET_NAMES]
        print(f"  {model:34s} " + "   ".join(f"{t}: {c}" for t, c in zip(TARGET_NAMES, cells)))
    print("\nSaved figures/gru_scores.csv. Test sets were not used.")


if __name__ == "__main__":
    main(sys.argv[1:])
