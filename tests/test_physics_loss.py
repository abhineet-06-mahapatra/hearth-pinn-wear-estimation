"""
The specification of physics_loss.py, written as tests with hand-calculated answers.
Run:  python -m pytest tests/test_physics_loss.py -v
Work through them from the top. Each one tells you what the function must do.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import physics_loss as pl  # noqa: E402

Y_STD = torch.tensor([0.05, 0.20, 0.20])           # metres: skull, isotherm, wall
ZERO = torch.zeros(1, 3)


def row(*v):
    return torch.tensor([list(v)], dtype=torch.float32)


# ---------------------------------------------------------------- to_absolute
def test_to_absolute_adds_the_scaled_change_to_the_day_zero_label():
    """y0 = [0.10, 1.50, 1.80]; outputs [1, -1, 0.5] times y_std [0.05, 0.20, 0.20] are changes [+0.05, -0.20, +0.10] m."""
    out = pl.to_absolute(row(1.0, -1.0, 0.5), row(0.10, 1.50, 1.80), Y_STD)
    assert out.shape == (1, 3)
    assert out[0].tolist() == pytest.approx([0.15, 1.30, 1.90], abs=1e-6)


# ---------------------------------------------------------------- bounds_penalty
def test_bounds_penalty_is_zero_when_everything_is_physically_possible():
    """Absolute values [0.10, 1.50, 1.80]: skull, isotherm, wall are all >= 0 and 1.50 <= 1.80 + 0.10."""
    assert pl.bounds_penalty(ZERO, row(0.10, 1.50, 1.80), Y_STD).item() == pytest.approx(0.0, abs=1e-9)


def test_bounds_penalty_returns_a_single_number():
    assert pl.bounds_penalty(ZERO, row(0.10, 1.50, 1.80), Y_STD).dim() == 0


def test_negative_skull_is_penalised():
    """y0 skull 0.02, output -1.0 means -0.05 m, so skull = -0.03 m. Violation 0.03 m -> (0.03 / 0.10)^2 = 0.09."""
    p = pl.bounds_penalty(row(-1.0, 0.0, 0.0), row(0.02, 1.50, 1.80), Y_STD)
    assert p.item() == pytest.approx(0.09, abs=1e-6)


def test_isotherm_beyond_the_solid_stack_is_penalised():
    """Absolute [0.0, 1.9, 1.8]: the isotherm is 0.1 m beyond wall + skull = 1.8. (0.1 / 0.1)^2 = 1.0."""
    p = pl.bounds_penalty(ZERO, row(0.0, 1.9, 1.8), Y_STD)
    assert p.item() == pytest.approx(1.0, abs=1e-6)


def test_several_violations_in_one_row_add_up():
    """y0 [0.0, 0.5, 0.1] and wall output -1.0 (-0.20 m): absolute [0.0, 0.5, -0.1].
    Wall below 0 by 0.1 -> 1.0.  Isotherm exceeds wall + skull = -0.1 by 0.6 -> 36.0.  Total 37.0."""
    p = pl.bounds_penalty(row(0.0, 0.0, -1.0), row(0.0, 0.5, 0.1), Y_STD)
    assert p.item() == pytest.approx(37.0, abs=1e-4)


def test_bounds_penalty_averages_over_the_rows():
    """One row with penalty 0.09 (negative skull) and one with 0: the mean is 0.045."""
    pred = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    y0 = torch.tensor([[0.02, 1.50, 1.80], [0.10, 1.50, 1.80]])
    assert pl.bounds_penalty(pred, y0, Y_STD).item() == pytest.approx(0.045, abs=1e-6)


def test_bounds_penalty_pushes_the_prediction_in_the_right_direction():
    """With a negative skull, raising the skull output must reduce the penalty, so the gradient on that output is negative.
    The other outputs are feasible, so their gradients are zero."""
    pred = row(-1.0, 0.0, 0.0).clone().requires_grad_(True)
    pl.bounds_penalty(pred, row(0.02, 1.50, 1.80), Y_STD).backward()
    g = pred.grad[0].tolist()
    assert g[0] < 0
    assert g[1] == pytest.approx(0.0, abs=1e-9) and g[2] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- no_regrowth_penalty
def test_no_penalty_when_the_wall_keeps_shrinking():
    """Wall outputs -0.5 then -1.0 are changes of -0.10 m then -0.20 m: the wall shrinks, as it should."""
    assert pl.no_regrowth_penalty(row(0, 0, -0.5), row(0, 0, -1.0), Y_STD).item() == pytest.approx(0.0, abs=1e-9)


def test_penalty_when_the_wall_grows_back():
    """Wall outputs -1.0 then -0.5: -0.20 m then -0.10 m, so the wall appears to grow by 0.10 m. (0.10 / 0.10)^2 = 1.0."""
    assert pl.no_regrowth_penalty(row(0, 0, -1.0), row(0, 0, -0.5), Y_STD).item() == pytest.approx(1.0, abs=1e-6)


def test_tolerance_forgives_small_growth():
    """With tol = 0.05 m, the same 0.10 m of growth counts as 0.05 m: (0.05 / 0.10)^2 = 0.25."""
    p = pl.no_regrowth_penalty(row(0, 0, -1.0), row(0, 0, -0.5), Y_STD, tol=0.05)
    assert p.item() == pytest.approx(0.25, abs=1e-6)


def test_only_the_wall_column_counts():
    """The skull and the isotherm may rise freely; only the wall must not grow back."""
    assert pl.no_regrowth_penalty(row(-2.0, -2.0, -1.0), row(3.0, 3.0, -1.0), Y_STD).item() == pytest.approx(0.0, abs=1e-9)


def test_no_regrowth_penalty_averages_over_the_rows():
    now = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    later = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, -1.5]])      # row 1 grows back by 0.10 m (penalty 1.0), row 2 shrinks (0)
    assert pl.no_regrowth_penalty(now, later, Y_STD).item() == pytest.approx(0.5, abs=1e-6)


def test_regrowth_penalty_pushes_the_later_prediction_down():
    now, later = row(0, 0, -1.0), row(0, 0, -0.5).clone().requires_grad_(True)
    pl.no_regrowth_penalty(now, later, Y_STD).backward()
    assert later.grad[0, 2] > 0                                      # lowering the later wall output reduces the penalty


# ---------------------------------------------------------------- combined_loss
def test_combined_loss_returns_the_four_parts():
    out = pl.combined_loss(ZERO, ZERO, row(0.10, 1.50, 1.80), Y_STD)
    assert set(out) == {"data", "bounds", "mono", "total"}
    assert all(v.dim() == 0 for v in out.values())


def test_data_term_is_the_mean_squared_error():
    """pred [1, 0, 0], target [0, 0, 0]: mse = (1 + 0 + 0) / 3."""
    out = pl.combined_loss(row(1.0, 0.0, 0.0), ZERO, row(0.10, 1.50, 1.80), Y_STD)
    assert out["data"].item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_total_is_data_plus_weighted_penalties():
    pred, later = row(-1.0, 0.0, -1.0), row(0.0, 0.0, -0.5)
    out = pl.combined_loss(pred, ZERO, row(0.02, 1.50, 1.80), Y_STD, pred_later=later, lam_bounds=2.0, lam_mono=3.0)
    expected = out["data"].item() + 2.0 * out["bounds"].item() + 3.0 * out["mono"].item()
    assert out["total"].item() == pytest.approx(expected, rel=1e-6)
    assert out["bounds"].item() > 0 and out["mono"].item() > 0       # this example violates both constraints


def test_without_a_later_window_the_monotone_term_is_zero():
    out = pl.combined_loss(ZERO, ZERO, row(0.10, 1.50, 1.80), Y_STD, pred_later=None)
    assert out["mono"].item() == 0.0


def test_zero_weights_leave_only_the_data_loss():
    pred, later = row(-1.0, 0.0, -1.0), row(0.0, 0.0, -0.5)
    out = pl.combined_loss(pred, ZERO, row(0.02, 1.50, 1.80), Y_STD, pred_later=later, lam_bounds=0.0, lam_mono=0.0)
    assert out["total"].item() == pytest.approx(out["data"].item(), rel=1e-6)
