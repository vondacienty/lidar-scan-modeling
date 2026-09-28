"""Tests for :func:`lidar_scan.build_tile_update_plan`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_tile_update_plan, plan_tile_update_windows,
                        publish_updates)
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


def _unfinished_state(scene_fixture, path_fixture, points_batch):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    return publish_updates(path_fixture["state"], path_fixture["index"],
                           scene_fixture[points_batch], limit=1)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.build_tile_update_plan is build_tile_update_plan
    assert "build_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_completed_state_has_empty_tasks(scene, paths):
    state_text = _completed_state(scene, paths)
    text = build_tile_update_plan(state_text, _audit_batches())
    assert text == '{"tasks":[],"complete":true}'
    document = json.loads(text)
    assert list(document) == ["tasks", "complete"]
    assert document["tasks"] == []
    assert document["complete"] is True
    assert not text.endswith("\n")


def test_empty_state_with_empty_batches(scene, paths):
    state_text = _empty_state(scene, paths)
    assert build_tile_update_plan(state_text, ()) \
        == '{"tasks":[],"complete":true}'


def test_repeated_calls_are_byte_for_byte_equal(scene, paths):
    state_text = _completed_state(scene, paths)
    first = build_tile_update_plan(state_text, _audit_batches())
    second = build_tile_update_plan(state_text, _audit_batches())
    assert first == second


def test_unfinished_state_plans_unpublished_batch(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    text = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    document = json.loads(text)
    assert list(document) == ["tasks", "complete"]
    assert document["complete"] is False

    rows = plan_tile_update_windows(state_text, (("b0", _POINTS_B0),))
    confirmed, total = rows[0][1], rows[0][2]

    keys = [tuple(task[:5]) for task in document["tasks"]]
    assert keys == sorted(keys)
    for task in document["tasks"]:
        assert len(task) == 6
        assert task[5] == [["b0", confirmed, total, None]]

    # Level-0 windows touch three distinct tiles; the (-1,1) and (0,0)
    # windows only meet at a corner and must not connect.
    level_zero = [task for task in document["tasks"] if task[0] == 0]
    assert len(level_zero) == 3
    assert level_zero[0][:5] == [0, -256, 256, -1, 511]
    assert level_zero[1][:5] == [0, 0, 0, 255, 255]
    assert level_zero[2][:5] == [0, 256, 256, 511, 511]

    # At level 1 the two touched tiles are x-edge adjacent with their y
    # intervals intersecting, so they merge into one component.
    level_one = [task for task in document["tasks"] if task[0] == 1]
    assert level_one == [[1, -512, 0, 511, 511,
                          [["b0", confirmed, total, None]]]]


def test_edge_adjacency_requires_interval_overlap_on_other_axis(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    # Tiles (0,0) and (1,1) touch only at the corner (255,255)/(256,256).
    points = (
        (0.0, 0.0, 5.0, 1, 1.0),
        (256.0, 256.0, 7.0, 1, 1.0),
    )
    document = json.loads(build_tile_update_plan(
        state_text, (("b0", points),), levels=1))
    level_zero = document["tasks"]
    assert [task[:5] for task in level_zero] == [
        [0, 0, 0, 255, 255],
        [0, 256, 256, 511, 511],
    ]


def test_edge_adjacency_merges_across_one_cell_gap(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    # (0,0) tile and the tile starting at ix=256: x edges one cell apart,
    # and the y intervals intersect (point at y=100).
    points = (
        (0.0, 0.0, 5.0, 1, 1.0),
        (256.0, 100.0, 7.0, 1, 1.0),
    )
    document = json.loads(build_tile_update_plan(
        state_text, (("b0", points),), levels=1))
    assert document["tasks"] == [
        [0, 0, 0, 511, 255, [["b0", 1, 2, None]]],
    ]


def test_transitive_closure_chains_components(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    # Three level-0 tiles in an L-chain: A adjacent to B on x, B adjacent
    # to C on y; A and C are disjoint but must join transitively.
    points = (
        (10.0, 10.0, 5.0, 1, 1.0),      # tile (0,0)
        (300.0, 10.0, 5.0, 1, 1.0),     # tile (1,0): x-adjacent to A
        (300.0, 300.0, 7.0, 1, 1.0),    # tile (1,1): y-adjacent to B
    )
    document = json.loads(build_tile_update_plan(
        state_text, (("b0", points),), levels=1))
    assert document["tasks"] == [
        [0, 0, 0, 511, 511, [["b0", 1, 2, None]]],
    ]


def test_component_bounds_take_endpoint_extremes(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    points = (
        (250.0, 250.0, 5.0, 1, 1.0),    # tile (0,0)
        (256.0, 256.0, 7.0, 1, 1.0),    # tile (1,1) corner only
        (256.0, 250.0, 7.0, 1, 1.0),    # tile (1,0) bridges x-edge
    )
    document = json.loads(build_tile_update_plan(
        state_text, (("b0", points),), levels=1))
    # (0,0) joins (1,0) by x-edge overlap; (1,0) joins (1,1) by y-edge
    # overlap, so all three close transitively.
    assert document["tasks"] == [
        [0, 0, 0, 511, 511, [["b0", 1, 2, None]]],
    ]


def test_tasks_sorted_across_levels_and_bounds(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    points = (
        (0.0, 0.0, 5.0, 1, 1.0),
        (100000.0, 100000.0, 7.0, 1, 1.0),
    )
    document = json.loads(build_tile_update_plan(
        state_text, (("b0", points),), levels=3))
    keys = [tuple(task[:5]) for task in document["tasks"]]
    assert keys == sorted(keys)
    # Levels group first: two tasks at each of three levels.
    assert [key[0] for key in keys] == [0, 0, 1, 1, 2, 2]


def test_pyramid_parameters_are_forwarded(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    text = build_tile_update_plan(
        state_text, (("b0", _POINTS_B0),),
        cell_size=2.0, tile_cells=128, levels=2)
    document = json.loads(text)
    assert max(task[0] for task in document["tasks"]) == 1


def test_only_unpublished_batches_contribute(scene, paths):
    state_text = _completed_state(scene, paths)
    document = json.loads(build_tile_update_plan(state_text, _audit_batches()))
    assert document["complete"] is True
    assert document["tasks"] == []


# ---------------------------------------------------------------------------
# validation (mirrors plan_tile_update_windows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        build_tile_update_plan(bad, ())


@pytest.mark.parametrize("bad", [[], None, 1, "x", {}])
def test_batches_wrong_type(bad):
    with pytest.raises(TypeError):
        build_tile_update_plan(
            '{"batches":[],"index":{},"complete":true}', bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), {},
])
def test_batch_shape_wrong_type(bad, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(TypeError):
        build_tile_update_plan(state_text, (bad, ("b1", ())))


@pytest.mark.parametrize("bad", [
    "x",
    (1, 2, 3),
    (1, 2, 3, "z", 1),
    [1, 2, 3, 4],
])
def test_bad_points_raise_type_error(bad, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(TypeError):
        build_tile_update_plan(
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
        build_tile_update_plan(state_text, _audit_batches(),
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
        build_tile_update_plan(state_text, _audit_batches(), **kwargs)


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
        build_tile_update_plan(tampered, ())


@pytest.mark.parametrize("wrong", [
    (("x", ()), ("b1", ())),
    (("b1", ()), ("b0", ())),
])
def test_batch_ids_must_match_state_in_order(wrong, scene, paths):
    state_text = _completed_state(scene, paths)
    with pytest.raises(ValueError):
        build_tile_update_plan(state_text, wrong)


def test_invalid_point_value_propagates_value_error(scene, paths):
    state_text = _completed_state(scene, paths)
    bad_points = ((0.0, 0.0, 5.0, 1, 0.0),)
    with pytest.raises(ValueError):
        build_tile_update_plan(
            state_text, (("b0", bad_points), ("b1", ())))


def test_points_iterable_is_walked_exactly_once(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")

    class Once:
        def __init__(self, values):
            self.values = values
            self.starts = 0

        def __iter__(self):
            self.starts += 1
            yield from self.values

    points = Once(_POINTS_B0)
    build_tile_update_plan(state_text, (("b0", points),))
    assert points.starts == 1
