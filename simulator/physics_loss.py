"""
physics_loss.py  (Week 6)  --  THIS FILE IS YOURS TO WRITE.

The network outputs, for each 60-day window, three numbers in STANDARDISED units: the estimated change since day 0 of
  column 0: skull thickness,  column 1: 1150 C isotherm distance,  column 2: remaining wall (ceramic + carbon).
To turn an output into metres:  change_in_metres = output * y_std   (y_std holds one standard deviation per column, in metres).
Adding the furnace's day-0 label y0 (metres) gives the ABSOLUTE value.

Two facts are true of every real hearth, and the data-only network knows neither:
  1. Impossible values.  Absolute skull >= 0, isotherm distance >= 0, wall >= 0, and the isotherm cannot lie beyond the solid stack:
        isotherm  <=  wall + skull.
  2. Wear does not undo itself.  The remaining wall never grows back, so for two windows of the SAME furnace, the later one
     cannot have a larger wall change than the earlier one.

Your job: write the four functions below so that tests/test_physics_loss.py passes. Run it with
        python -m pytest tests/test_physics_loss.py -v
Each test says in its name and docstring what it expects, with hand-calculated numbers. Work through them in order.

A violation is measured in units of VIOLATION_SCALE = 0.10 m, so a 5 cm violation costs (0.05 / 0.10)^2 = 0.25.
Hints (no solutions):
  - torch.relu(x) is x where x > 0 and 0 elsewhere; it turns "how far past the limit" into a penalty that is 0 when the limit is respected.
  - Square the violation so small errors cost little and large ones cost a lot.
  - Work on whole batches at once (tensors of shape (N, 3)); do not write a Python loop over rows.
  - Several constraints on one row ADD UP; across rows, take the MEAN.
"""
import torch
import torch.nn.functional as F  # noqa: F401  (you will need F.mse_loss)

VIOLATION_SCALE = 0.10      # metres


def to_absolute(pred_scaled, y0, y_std):
    """Absolute values in metres from standardised outputs: y0 + pred_scaled * y_std. Shapes (N, 3), (N, 3), (3,) -> (N, 3)."""
    raise NotImplementedError("your turn")


def bounds_penalty(pred_scaled, y0, y_std):
    """Penalty for impossible values. Returns a single number (a 0-d tensor).

    For each row compute four violations in metres:
        how far skull is below 0, how far isotherm is below 0, how far wall is below 0,
        and how far isotherm exceeds (wall + skull).
    Divide each by VIOLATION_SCALE, square it, add the four up for the row, then average over the rows.
    """
    raise NotImplementedError("your turn")


def no_regrowth_penalty(pred_now, pred_later, y_std, tol=0.0):
    """Penalty for a wall that seems to grow back. pred_now and pred_later are (N, 3) outputs for windows of the same furnace, a few days apart.

    Only column 2 (the wall) matters. The wall change in metres is pred * y_std[2]. Growth is (later - now - tol); it is a violation
    only when positive. Divide by VIOLATION_SCALE, square, and average over the rows.
    """
    raise NotImplementedError("your turn")


def combined_loss(pred, y_scaled, y0, y_std, pred_later=None, lam_bounds=1.0, lam_mono=1.0):
    """The training loss. Returns a dict with four entries, each a 0-d tensor:
        'data'   : ordinary mean squared error between pred and y_scaled (use F.mse_loss)
        'bounds' : bounds_penalty(pred, y0, y_std)
        'mono'   : no_regrowth_penalty(pred, pred_later, y_std)   -- or 0 if pred_later is None
        'total'  : data + lam_bounds * bounds + lam_mono * mono
    """
    raise NotImplementedError("your turn")
