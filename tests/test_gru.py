"""Tests for simulator/train_gru.py (skipped if PyTorch is not installed)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import train_gru as tg  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))


def toy_task(n=400, length=20, seed=0):
    """Target = scaled change of TC3 on the window's last day: a network that can read a sequence should learn it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, length, 9)).astype("float32")
    y = np.stack([X[:, -1, 2], 0.5 * X[:, -5:, 1].mean(axis=1), np.zeros(n)], axis=1).astype("float32")
    return X, y


def test_output_shape():
    m = tg.GRUNet(9, 8, 1, 3)
    assert m(torch.zeros(5, 20, 9)).shape == (5, 3)


def test_network_learns_a_simple_sequence_task():
    X, y = toy_task()
    model, hist = tg.fit_gru(X[:300], y[:300], X[300:], y[300:], seed=0, hidden=16, epochs=40, patience=40, lr=5e-3)
    p = tg.predict(model, X[300:])
    assert tg.bl.metrics(y[300:, 0], p[:, 0])["R2"] > 0.7
    assert min(hist) < 0.5 * hist[0]


def test_training_is_reproducible_for_a_given_seed():
    X, y = toy_task(n=120)
    a, _ = tg.fit_gru(X[:90], y[:90], X[90:], y[90:], seed=3, hidden=8, epochs=3)
    b, _ = tg.fit_gru(X[:90], y[:90], X[90:], y[90:], seed=3, hidden=8, epochs=3)
    assert np.allclose(tg.predict(a, X[90:]), tg.predict(b, X[90:]), atol=1e-6)


def test_returned_model_has_the_best_validation_loss():
    X, y = toy_task(n=150)
    model, hist = tg.fit_gru(X[:100], y[:100], X[100:], y[100:], seed=1, hidden=8, epochs=12, patience=12, lr=1e-2)
    final = float(np.mean((tg.predict(model, X[100:]) - y[100:]) ** 2))
    assert final == pytest.approx(min(hist), rel=1e-4)


def test_early_stopping_stops_before_the_epoch_limit_when_nothing_improves():
    X, y = toy_task(n=100)
    y_noise = np.random.default_rng(5).normal(size=y.shape).astype("float32")      # unlearnable target
    _, hist = tg.fit_gru(X[:70], y_noise[:70], X[70:], y_noise[70:], seed=0, hidden=8, epochs=60, patience=3, lr=1e-2)
    assert len(hist) < 60


def test_summary_features_shape():
    X = np.zeros((10, 60, 9), dtype="float32")
    assert tg.summary_features(X).shape == (10, 6 + 3 + 3 + 3)


def test_whole_script_runs_with_only_train_and_val_files_present(tmp_path, monkeypatch):
    """End-to-end smoke test in a folder that has no test files: the script must not need them."""
    import make_dataset as md
    _, frames = md.build_all(n_in=40, n_ood=1)
    (tmp_path / "data").mkdir()
    for name in ("train", "val"):
        frames[name].to_csv(tmp_path / "data" / f"{name}.csv", index=False)
    monkeypatch.chdir(tmp_path)
    tg.main(["--quick", "--seeds", "1", "--epochs", "2"])
    assert (tmp_path / "figures" / "gru_scores.csv").exists()
    scores = __import__("pandas").read_csv(tmp_path / "figures" / "gru_scores.csv")
    assert set(scores.model) == {"ridge on window summary", "boosted trees on window summary", "GRU (data only)"}
