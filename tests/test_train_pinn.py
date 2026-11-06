"""Tests for simulator/train_pinn.py. The end-to-end test is skipped until physics_loss.py has been written."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import train_pinn as tp  # noqa: E402


def test_no_violations_when_predictions_are_physically_possible():
    pred_abs = np.array([[0.1, 1.5, 1.8], [0.0, 1.0, 1.2]])
    nxt = np.array([1, -1])
    bounds, regrowth = tp.violation_rates(pred_abs, np.array([-0.10, -0.20]), nxt)
    assert bounds == 0.0 and regrowth == 0.0


def test_bounds_violation_rate_counts_windows_more_than_a_centimetre_past_a_limit():
    pred_abs = np.array([[-0.05, 1.5, 1.8],      # skull below 0 by 5 cm: violation
                         [-0.005, 1.5, 1.8],     # skull below 0 by 5 mm: within tolerance
                         [0.0, 1.95, 1.8],       # isotherm 15 cm beyond the stack: violation
                         [0.1, 1.5, 1.8]])
    bounds, _ = tp.violation_rates(pred_abs, np.zeros(4), -np.ones(4, dtype=int))
    assert bounds == pytest.approx(0.5)


def test_regrowth_rate_counts_pairs_where_the_wall_rises_by_more_than_two_centimetres():
    pred_abs = np.tile([0.1, 1.5, 1.8], (4, 1))
    wall_change = np.array([-0.20, -0.10, -0.11, -0.30])       # pairs: 0->1 grows 0.10 (violation), 2->3 shrinks
    nxt = np.array([1, -1, 3, -1])
    _, regrowth = tp.violation_rates(pred_abs, wall_change, nxt)
    assert regrowth == pytest.approx(0.5)


def test_config_parsing():
    assert tp.parse_configs("0:0,1:0.5") == [(0.0, 0.0), (1.0, 0.5)]


def test_end_to_end_run_with_only_train_and_val_files(tmp_path, monkeypatch):
    pl = tp.pl
    try:
        pl.combined_loss(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(3))
    except NotImplementedError:
        pytest.skip("physics_loss.py has not been written yet")
    import make_dataset as md
    _, frames = md.build_all(n_in=40, n_ood=1)
    (tmp_path / "data").mkdir()
    for name in ("train", "val"):
        frames[name].to_csv(tmp_path / "data" / f"{name}.csv", index=False)
    monkeypatch.chdir(tmp_path)
    tp.main(["--quick", "--seeds", "1", "--epochs", "2"])
    import pandas as pd
    runs = pd.read_csv(tmp_path / "figures" / "pinn_runs.csv")
    assert set(zip(runs.lam_bounds, runs.lam_mono)) == {(0.0, 0.0), (1.0, 1.0)}
    assert (tmp_path / "figures" / "pinn_scores.csv").exists()
