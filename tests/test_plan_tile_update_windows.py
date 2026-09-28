"""Tests for :func:`lidar_scan.plan_tile_update_windows`."""

from __future__ import annotations

import pytest

from lidar_scan import (build_tile_pyramid, build_tile_update_audit,
                        plan_tile_update_windows, publish_updates)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_publish_updates as publish_fixtures

# Reuse the prepared publication scene and path layout.
scene = publish_fixtures.scene
paths = publish_fixtures.paths

_POINTS_B0 = (
    (0.0, 0.0, 5.0, 1, 1.0),
    (255.0, 255.0, 9.0, 2, 1.0),
    (256.0, 256.0, 7.0, 1, 2.0),
    (-1.0, 300.0, 4.0, 3, 1.5),
)
_POINTS_B1 = (
    (600.0, 600.0, -2.0, 0, 1.0),
    (512.0, 512.0, 11.0, 4, 1.0),
)


def _audit_batches():
    return (("b0", _POINTS_B0), ("b1", _POINTS_B1))


def _completed_state(scene_fixture, path_fixture):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    return publish_updates(path_fixture["state"], path_fixture["index"],
                           scene_fixture["two_batches"])


def _empty_state(scene_fixture, path_fixture):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    return publish_updates(path_fixture["state"], path_fixture["index"], ())


def _expected_windows(pyramid):
    return tuple((level, tile[2], tile[3], tile[4], tile[5])
                 for level, level_tiles in enumerate(pyramid)
                 for tile in level_tiles)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.plan_tile_update_windows is plan_tile_update_windows
    assert "plan_tile_update_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# planning results
# ---------------------------------------------------------------------------

def test_rows_extend_audit_rows_with_windows(scene, paths):
    state_text = _completed_state(scene, paths)
    batches = _audit_batches()
    result = plan_tile_update_windows(state_text, batches)
    audit = build_tile_update_audit(state_text, batches)

    assert isinstance(result, tuple) and len(result) == 2
    for row in result:
        assert isinstance(row, tuple) and len(row) == 7
    for row, audit_row in zip(result, audit):
        assert row[:6] == audit_row
        assert row[6] == _expected_windows(row[5])


def test_windows_sorted_by_level_and_tile_coordinates(scene, paths):
    state_text = _completed_state(scene, paths)
    result = plan_tile_update_windows(state_text, _audit_batches())
    windows = result[0][6]
    pyramid = result[0][5]
    level_of = {}
    for level, level_tiles in enumerate(pyramid):
        for tile in level_tiles:
            level_of[(tile[2], tile[3], tile[4], tile[5])] = (
                level, tile[0], tile[1])
    ordered = tuple(level_of[w[1:]] for w in windows)
    assert ordered == tuple(sorted(ordered))
    assert (0, 0, 0, 255, 255) in windows


def test_empty_points_get_empty_windows(scene, paths):
    state_text = _completed_state(scene, paths)
    result = plan_tile_update_windows(
        state_text, (("b0", ()), ("b1", _POINTS_B1)))
    assert result[0][5] == build_tile_pyramid(())
    assert result[0][6] == ()
    assert result[1][6] == _expected_windows(build_tile_pyramid(_POINTS_B1))


def test_empty_state_with_empty_batches_returns_empty_tuple(scene, paths):
    state_text = _empty_state(scene, paths)
    assert plan_tile_update_windows(state_text, ()) == ()


def test_pyramid_parameters_are_forwarded(scene, paths):
    state_text = _completed_state(scene, paths)
    result = plan_tile_update_windows(
        state_text, _audit_batches(), cell_size=2.0,
        tile_cells=128, levels=2)
    assert len(result[0][5]) == 2
    assert result[0][5] == build_tile_pyramid(
        _POINTS_B0, cell_size=2.0, tile_cells=128, levels=2)
    assert result[0][6] == _expected_windows(result[0][5])


def test_points_iterable_is_walked_exactly_once(scene, paths):
    state_text = _completed_state(scene, paths)

    class Once:
        def __init__(self, values):
            self.values = values
            self.starts = 0

        def __iter__(self):
            self.starts += 1
            yield from self.values

    points = Once(_POINTS_B0)
    result = plan_tile_update_windows(
        state_text, (("b0", points), ("b1", ())))
    assert points.starts == 1
    assert result[0][5] == build_tile_pyramid(_POINTS_B0)


def test_repeated_calls_are_equal(scene, paths):
    state_text = _completed_state(scene, paths)
    batches = _audit_batches()
    first = plan_tile_update_windows(state_text, batches)
    second = plan_tile_update_windows(state_text, batches)
    assert first == second


def test_unfinished_state_row_matches_audit(scene, paths):
    with open(paths["index"], "w", encoding="utf-8") as stream:
        stream.write(scene["empty"])
    state_text = publish_updates(paths["state"], paths["index"],
                                 scene["one_multi_batch"], limit=1)
    result = plan_tile_update_windows(state_text, (("b0", _POINTS_B0),))
    audit = build_tile_update_audit(state_text, (("b0", _POINTS_B0),))
    assert len(result) == 1
    assert result[0][:6] == audit[0]
    assert result[0][3] is False
    assert result[0][4] is None
    assert result[0][6] == _expected_windows(result[0][5])


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        plan_tile_update_windows(bad, ())


@pytest.mark.parametrize("bad", [[], None, 1, "x", {}])
def test_batches_wrong_type(bad):
    with pytest.raises(TypeError):
        plan_tile_update_windows('{"batches":[],"index":{},"complete":true}',
                                 bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), {},
])
def test_batch_shape_wrong_type(bad, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(TypeError):
        plan_tile_update_windows(state_text, (bad, ("b1", ())))


@pytest.mark.parametrize("bad", [
    "x",
    (1, 2, 3),
    (1, 2, 3, "z", 1),
    [1, 2, 3, 4],
])
def test_bad_points_raise_type_error(bad, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(TypeError):
        plan_tile_update_windows(
            state_text, (("b0", (bad,)), ("b1", ())))


@pytest.mark.parametrize("value,exc", [
    (True, TypeError),
    ("x", TypeError),
    (float("nan"), ValueError),
    (0.0, ValueError),
    (-1.0, ValueError),
])
def test_cell_size_validation_matches_pyramid(value, exc, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(exc):
        plan_tile_update_windows(state_text, _audit_batches(),
                                 cell_size=value)


@pytest.mark.parametrize("kwargs,exc", [
    ({"tile_cells": 1.0}, TypeError),
    ({"tile_cells": True}, TypeError),
    ({"tile_cells": 0}, ValueError),
    ({"levels": 1.0}, TypeError),
    ({"levels": True}, TypeError),
    ({"levels": 0}, ValueError),
])
def test_pyramid_shape_parameter_validation(kwargs, exc, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(exc):
        plan_tile_update_windows(state_text, _audit_batches(), **kwargs)


@pytest.mark.parametrize("value,exc", [
    (True, TypeError),
    ("x", TypeError),
    (float("nan"), ValueError),
    (0.0, ValueError),
])
def test_empty_state_still_validates_cell_size(value, exc, scene, paths):
    state_text = _empty_state(scene, paths)
    with pytest.raises(exc):
        plan_tile_update_windows(state_text, (), cell_size=value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"batches":[],"index":{},"complete":true}',
    ' {"batches":[],"index":{},"complete":true}',
])
def test_non_canonical_state_rejected(tampered):
    with pytest.raises(ValueError):
        plan_tile_update_windows(tampered, ())


def test_empty_state_rejects_nonempty_batches(scene, paths):
    state_text = _empty_state(scene, paths)
    with pytest.raises(ValueError):
        plan_tile_update_windows(state_text, (("b0", ()),))


@pytest.mark.parametrize("wrong", [
    (("x", ()), ("b1", ())),
    (("b1", ()), ("b0", ())),
])
def test_batch_ids_must_match_state_in_order(wrong, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(ValueError):
        plan_tile_update_windows(state_text, wrong)


def test_invalid_point_value_propagates_value_error(scene, paths):
    state_text = _completed_state(scene, paths)
    bad_points = ((0.0, 0.0, 5.0, 1, 0.0),)
    with pytest.raises(ValueError):
        plan_tile_update_windows(
            state_text, (("b0", bad_points), ("b1", ())))
