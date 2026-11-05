"""Tests for the selection rule in simulator/sweep_gru.py (no training is run)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
pytest = __import__("pytest")
pytest.importorskip("torch")
import sweep_gru as sw  # noqa: E402


def table(rows):
    return pd.DataFrame(rows, columns=["length", "hidden", "mean_R2"])


def test_clear_winner_is_chosen():
    best = sw.pick_best(table([(30, 16, 0.30), (60, 32, 0.50), (120, 64, 0.35)]))
    assert (best.length, best.hidden) == (60, 32)


def test_simplest_setting_wins_a_near_tie():
    best = sw.pick_best(table([(60, 32, 0.500), (30, 16, 0.495), (120, 64, 0.505)]))
    assert (best.length, best.hidden) == (30, 16)


def test_a_setting_outside_the_tolerance_does_not_win_by_being_simple():
    best = sw.pick_best(table([(30, 16, 0.40), (60, 32, 0.50)]))
    assert (best.length, best.hidden) == (60, 32)


def test_settings_cover_the_planned_grid():
    assert sw.LENGTHS == (30, 60, 120) and sw.HIDDEN == (16, 32, 64) and sw.TOLERANCE == 0.01


def test_completed_runs_are_recognised_so_a_sweep_can_resume():
    runs = pd.DataFrame({"length": [30, 30], "hidden": [16, 32], "seed": [0, 1]})
    assert sw.completed(runs) == {(30, 16, 0), (30, 32, 1)}
    assert sw.completed(runs.iloc[0:0]) == set()
