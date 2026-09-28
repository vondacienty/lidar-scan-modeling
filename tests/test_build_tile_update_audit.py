"""Tests for :func:`lidar_scan.build_tile_update_audit`."""

from __future__ import annotations

import json
import os

import pytest

from lidar_scan import (build_recovery_history, build_tile_pyramid,
                        build_tile_update_audit, merge_checkout_logs,
                        plan_checkout_recovery, publish_updates,
                        update_recovery_index)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_build_recovery_history as recovery_fixtures


def _completed_history(plan, units):
    states = recovery_fixtures._states(plan, units)
    return build_recovery_history(
        plan, tuple(recovery_fixtures._summary(plan, states[:i + 1])
                    for i in range(units)))


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    plan_zero = plan_checkout_recovery(log, None, ledgers[:1])
    plan_twelve = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    history_zero = _completed_history(plan_zero, 1)
    history_twelve = _completed_history(plan_twelve, 2)

    empty = update_recovery_index(None, ())
    after_zero = update_recovery_index(empty, ((plan_zero, history_zero),))
    after_all = update_recovery_index(
        after_zero, ((plan_twelve, history_twelve),))

    return {
        "empty": empty,
        "after_zero": after_zero,
        "after_all": after_all,
        "two_batches": (("b0", ((plan_zero, history_zero),)),
                        ("b1", ((plan_twelve, history_twelve),))),
        "one_multi_batch": (
            ("b0", ((plan_zero, history_zero),
                    (plan_twelve, history_twelve))),),
        "with_empty_batches": (
            ("leading", ()), ("b0", ((plan_zero, history_zero),)),
            ("trailing", ())),
    }


@pytest.fixture
def paths(tmp_path):
    return {
        "state": str(tmp_path / "state.json"),
        "index": str(tmp_path / "index.json"),
    }


def _write(path, text):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text)


def _publish(paths, scene, batches, **kwargs):
    _write(paths["index"], scene["empty"])
    return publish_updates(paths["state"], paths["index"], batches, **kwargs)


POINTS_A = ((0.0, 0.0, 1.0, 5, 0.1),
            (300.0, 300.0, 2.0, 5, 0.1),
            (300.5, 300.2, -1.5, 4, 0.2))
POINTS_B = ((-5.0, 700.0, 9.0, 1, 1.0),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.build_tile_update_audit is build_tile_update_audit
    assert "build_tile_update_audit" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_completed_state_audit_rows(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    audit = build_tile_update_audit(
        state, (("b0", POINTS_A), ("b1", POINTS_B)))
    assert len(audit) == 2
    row_zero, row_twelve = audit
    assert row_zero[:5] == ("b0", 1, 1, True, (0, 1, 1))
    assert row_twelve[0] == "b1"
    assert row_twelve[1] == row_twelve[2] == 1
    assert row_twelve[3] is True
    assert isinstance(row_twelve[4], tuple) and len(row_twelve[4]) == 3
    for row in audit:
        assert len(row) == 6
        assert isinstance(row[5], tuple) and len(row[5]) == 3


def test_state_values_are_taken_from_the_state(scene, paths):
    state = _publish(paths, scene, scene["one_multi_batch"], limit=1)
    audit = build_tile_update_audit(state, (("b0", POINTS_A),))
    (row,) = audit
    assert row[0] == "b0"
    assert row[1] == 1
    assert row[2] == 2
    assert row[3] is False
    assert row[4] is None


def test_pyramids_match_build_tile_pyramid_in_state_order(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    audit = build_tile_update_audit(
        state, (("b0", POINTS_A), ("b1", POINTS_B)))
    assert audit[0][5] == build_tile_pyramid(POINTS_A)
    assert audit[1][5] == build_tile_pyramid(POINTS_B)
    for row in audit:
        for level in row[5]:
            keys = [(tile[0], tile[1]) for tile in level]
            assert keys == sorted(keys)


def test_pyramid_parameters_are_forwarded(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    audit = build_tile_update_audit(
        state, (("b0", POINTS_A), ("b1", POINTS_B)),
        cell_size=2.0, tile_cells=4, levels=2)
    assert audit[0][5] == build_tile_pyramid(
        POINTS_A, cell_size=2.0, tile_cells=4, levels=2)
    assert audit[1][5] == build_tile_pyramid(
        POINTS_B, cell_size=2.0, tile_cells=4, levels=2)
    assert all(len(row[5]) == 2 for row in audit)


def test_empty_point_batches_keep_levels_of_empty_tiles(scene, paths):
    state = _publish(paths, scene, scene["with_empty_batches"])
    audit = build_tile_update_audit(
        state, (("leading", ()), ("b0", POINTS_A), ("trailing", ())))
    assert [row[0] for row in audit] == ["leading", "b0", "trailing"]
    assert audit[0][5] == ((), (), ())
    assert audit[2][5] == ((), (), ())
    assert audit[1][5] == build_tile_pyramid(POINTS_A)
    # The zero-item published batches carry a zero-step receipt from state.
    assert audit[0][1:4] == (0, 0, True)
    assert audit[2][1:4] == (0, 0, True)


def test_empty_state_with_empty_batches_returns_empty_tuple(scene, paths):
    _write(paths["index"], scene["empty"])
    state = publish_updates(paths["state"], paths["index"], ())
    assert build_tile_update_audit(state, ()) == ()


def test_repeated_results_are_equal(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    batches = (("b0", POINTS_A), ("b1", POINTS_B))
    first = build_tile_update_audit(state, batches)
    second = build_tile_update_audit(state, batches)
    assert first == second


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        build_tile_update_audit(bad, ())


@pytest.mark.parametrize("bad", [[], None, 1, "x"])
def test_batches_wrong_type(scene, paths, bad):
    state = _publish(paths, scene, ())
    with pytest.raises(TypeError):
        build_tile_update_audit(state, bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), ("b0", []),
    ("b0", [POINTS_A[0]]), ("b0", None),
])
def test_batch_or_points_wrong_type(scene, paths, bad):
    state = _publish(paths, scene, ())
    with pytest.raises(TypeError):
        build_tile_update_audit(state, (bad,))


@pytest.mark.parametrize("bad", [True, False, "x", 1j])
def test_cell_size_wrong_type(scene, paths, bad):
    state = _publish(paths, scene, ())
    with pytest.raises(TypeError):
        build_tile_update_audit(state, (), cell_size=bad)


@pytest.mark.parametrize("bad", [1.0, True, False, "x"])
def test_tile_cells_wrong_type(scene, paths, bad):
    state = _publish(paths, scene, ())
    with pytest.raises(TypeError):
        build_tile_update_audit(state, (), tile_cells=bad)


@pytest.mark.parametrize("bad", [1.0, True, False, "x"])
def test_levels_wrong_type(scene, paths, bad):
    state = _publish(paths, scene, ())
    with pytest.raises(TypeError):
        build_tile_update_audit(state, (), levels=bad)


def test_point_type_errors_propagate(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    bad_batches = (("b0", ((0.0, 0.0, 1.0, 5),),), ("b1", POINTS_B))
    with pytest.raises(TypeError):
        build_tile_update_audit(state, bad_batches)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tampered", [
    "{not json",
    '{"batches":[],"index":{},"complete":false}',
    "not a state",
])
def test_non_canonical_state_rejected(tampered):
    with pytest.raises(ValueError):
        build_tile_update_audit(tampered, ())


def test_state_with_trailing_data_rejected(scene, paths):
    state = _publish(paths, scene, ())
    with pytest.raises(ValueError):
        build_tile_update_audit(state + " ", ())


def test_batch_ids_must_match_state_in_order(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    with pytest.raises(ValueError):
        build_tile_update_audit(
            state, (("other", POINTS_A), ("b1", POINTS_B)))
    with pytest.raises(ValueError):
        build_tile_update_audit(
            state, (("b1", POINTS_B), ("b0", POINTS_A)))


def test_batch_count_must_match_state(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    with pytest.raises(ValueError):
        build_tile_update_audit(state, (("b0", POINTS_A),))
    with pytest.raises(ValueError):
        build_tile_update_audit(state, ())


def test_empty_state_rejects_any_batches(scene, paths):
    _write(paths["index"], scene["empty"])
    state = publish_updates(paths["state"], paths["index"], ())
    with pytest.raises(ValueError):
        build_tile_update_audit(state, (("b0", POINTS_A),))


def test_point_value_errors_propagate(scene, paths):
    state = _publish(paths, scene, scene["two_batches"])
    bad_batches = (("b0", ((0.0, 0.0, 1.0, 5, -1.0),),),
                   ("b1", POINTS_B))
    with pytest.raises(ValueError):
        build_tile_update_audit(state, bad_batches)


@pytest.mark.parametrize("kwargs", [
    {"cell_size": 0}, {"cell_size": -1.0}, {"tile_cells": 0},
    {"tile_cells": -2}, {"levels": 0}, {"levels": -1},
])
def test_bad_pyramid_parameter_values_rejected(scene, paths, kwargs):
    state = _publish(paths, scene, ())
    with pytest.raises(ValueError):
        build_tile_update_audit(state, (), **kwargs)


def test_non_finite_cell_size_rejected(scene, paths):
    state = _publish(paths, scene, ())
    with pytest.raises(ValueError):
        build_tile_update_audit(state, (), cell_size=float("nan"))
