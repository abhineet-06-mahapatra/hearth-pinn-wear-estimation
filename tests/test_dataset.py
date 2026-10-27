"""Tests for simulator/make_dataset.py: splits by furnace, no leakage, out-of-range set, determinism."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))
import make_dataset as md  # noqa: E402


@pytest.fixture(scope="module")
def small():
    return md.build_all(n_in=10, n_ood=2)


def test_splits_are_disjoint_and_complete():
    s = md.assign_splits(200)
    seeds = [x for k in ("train", "val", "test") for x in s[k]]
    assert len(seeds) == 200 and len(set(seeds)) == 200
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (140, 30, 30)


def test_no_furnace_appears_in_two_splits(small):
    splits, frames = small
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (set(frames[a].scenario) & set(frames[b].scenario))


def test_frames_contain_exactly_their_assigned_furnaces(small):
    splits, frames = small
    for name in frames:
        assert sorted(frames[name].scenario.unique()) == sorted(splits[name])


def test_inputs_never_include_diagnostic_columns():
    assert not (set(md.INPUT_COLUMNS) & set(md.DIAGNOSTIC_COLUMNS))
    assert not (set(md.INPUT_COLUMNS) & set(md.LABEL_COLUMNS))
    assert md.INPUT_COLUMNS == ["TC1", "TC2", "TC3"]


def test_all_listed_columns_exist(small):
    _, frames = small
    d = frames["train"]
    for c in md.ID_COLUMNS + md.INPUT_COLUMNS + md.LABEL_COLUMNS + md.DIAGNOSTIC_COLUMNS:
        assert c in d.columns


def test_ood_furnaces_have_conductivity_above_the_training_range(small):
    _, frames = small
    assert frames["test_ood"].k_c_true.min() >= md.OOD_K_C_RANGE[0] - 1e-9
    assert frames["train"].k_c_true.max() <= 35.0 + 1e-9


def test_ood_override_does_not_leak_into_later_runs(small):
    import hearth_simulator_v2 as sim
    assert sim.K_C_RANGE == (13.0, 35.0)


def test_generation_is_deterministic():
    a = md.make_furnace(5)
    b = md.make_furnace(5)
    assert md.checksum(a) == md.checksum(b)


def test_checksum_detects_a_change(small):
    _, frames = small
    d = frames["val"].copy()
    before = md.checksum(d)
    d.loc[d.index[0], "remaining_m"] += 0.01
    assert md.checksum(d) != before


def test_manifest_lists_every_split_with_checksum(small):
    splits, frames = small
    m = md.manifest(splits, frames)
    assert set(m["splits"]) == {"train", "val", "test", "test_ood"}
    assert all(len(v["sha256"]) == 64 for v in m["splits"].values())
