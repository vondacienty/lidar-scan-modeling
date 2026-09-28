"""Tests for :func:`lidar_scan.execute_tile_update_plan`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_tile_pyramid, build_tile_update_plan,
                        execute_tile_update_plan, query_tile_pyramid_windows)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_build_tile_update_plan as plan_fixtures

# Reuse the prepared publication scene and path layout.
scene = plan_fixtures.scene
paths = plan_fixtures.paths

_POINTS_B0 = plan_fixtures._POINTS_B0
_POINTS_B1 = plan_fixtures._POINTS_B1

_completed_state = plan_fixtures._completed_state
_unfinished_state = plan_fixtures._unfinished_state


def _audit_batches():
    return (("b0", _POINTS_B0), ("b1", _POINTS_B1))


def _updates():
    return (("b0", build_tile_pyramid(_POINTS_B0)),
            ("b1", build_tile_pyramid(_POINTS_B1)))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.execute_tile_update_plan is execute_tile_update_plan
    assert "execute_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_document_shape_with_real_plan(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    text = execute_tile_update_plan(plan, (("b0", build_tile_pyramid(_POINTS_B0)),))
    assert text.startswith('{"plan":' + plan)
    document = json.loads(text)
    assert list(document) == ["plan", "confirmed", "results", "complete"]
    assert document["complete"] is True
    assert not text.endswith("\n")
    assert " " not in text


def test_plan_is_embedded_byte_for_byte(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    text = execute_tile_update_plan(plan, (("b0", build_tile_pyramid(_POINTS_B0)),))
    assert text[8:8 + len(plan)] == plan
    assert json.loads(text)["plan"] == json.loads(plan)


def test_results_match_window_intersection_and_merge(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    pyramid = build_tile_pyramid(_POINTS_B0)
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    plan_document = json.loads(plan)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    results = json.loads(text)["results"]

    expected = []
    for task in plan_document["tasks"]:
        level, ix_min, iy_min, ix_max, iy_max = task[:5]
        window_result = query_tile_pyramid_windows(
            pyramid, ((level, ix_min, iy_min, ix_max, iy_max),))[0]
        expected.append([level, ix_min, iy_min, ix_max, iy_max,
                         ["b0"], [list(tile) for tile in window_result[5]]])
    assert results == expected

    # Spot-check the five known tasks of the b0 scene.
    by_key = {tuple(result[:5]): result for result in results}
    assert by_key[(0, -256, 256, -1, 511)][6] == [
        [-1, 1, -256, 256, -1, 511, 4.0, 4.0, 1]]
    assert by_key[(0, 0, 0, 255, 255)][6] == [
        [0, 0, 0, 0, 255, 255, 5.0, 9.0, 2]]
    assert by_key[(0, 256, 256, 511, 511)][6] == [
        [1, 1, 256, 256, 511, 511, 7.0, 7.0, 1]]
    assert by_key[(1, -512, 0, 511, 511)][6] == [
        [-1, 0, -512, 0, -1, 511, 4.0, 4.0, 1],
        [0, 0, 0, 0, 511, 511, 5.0, 9.0, 3]]
    assert by_key[(2, -1024, 0, 1023, 1023)][6] == [
        [-1, 0, -1024, 0, -1, 1023, 4.0, 4.0, 1],
        [0, 0, 0, 0, 1023, 1023, 5.0, 9.0, 3]]


def test_completed_state_plan_has_no_tasks(scene, paths):
    state_text = _completed_state(scene, paths)
    plan = build_tile_update_plan(state_text, _audit_batches())
    assert plan == '{"tasks":[],"complete":true}'
    text = execute_tile_update_plan(plan, _updates())
    assert text == ('{"plan":{"tasks":[],"complete":true},'
                    '"confirmed":0,"results":[],"complete":true}')


def test_sources_keep_plan_order_and_tiles_sorted():
    plan = ('{"tasks":[[0,0,0,511,255,'
            '[["b0",1,2,null],["b1",0,1,null]]]],"complete":false}')
    pyramid_b0 = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    # b1 shares the (0, 0) tile with equal bounds and also contributes a
    # second tile that sorts after it.
    pyramid_b1 = (
        ((0, 0, 0, 0, 255, 255, 2.0, 7.0, 1),
         (1, 0, 256, 0, 511, 255, 1.0, 3.0, 4),),)
    text = execute_tile_update_plan(
        plan, (("b0", pyramid_b0), ("b1", pyramid_b1)))
    result = json.loads(text)["results"][0]
    assert result[5] == ["b0", "b1"]
    assert [tile[:2] for tile in result[6]] == [[0, 0], [1, 0]]
    # The shared (0,0) tile merges: min zmin, max zmax, summed count; the
    # b1-only tile passes through unchanged.
    assert result[6] == [
        [0, 0, 0, 0, 255, 255, 2.0, 9.0, 3],
        [1, 0, 256, 0, 511, 255, 1.0, 3.0, 4]]


def test_only_intersecting_tiles_are_selected():
    plan = '{"tasks":[[0,10,10,20,20,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),
                (1, 1, 256, 256, 511, 511, 1.0, 2.0, 1)),)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    assert json.loads(text)["results"][0][6] == [
        [0, 0, 0, 0, 255, 255, 5.0, 9.0, 2]]


def test_window_without_intersecting_tiles_gets_empty_tiles():
    plan = '{"tasks":[[0,1000,1000,1010,1010,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    document = json.loads(text)
    assert document["results"][0] == [
        0, 1000, 1000, 1010, 1010, ["b0"], []]
    assert document["complete"] is True
    assert text.endswith(
        '[0,1000,1000,1010,1010,["b0"],[]]],"complete":true}')


def test_z_values_use_six_decimals_and_negative_zero_normalized():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, -0.0, 7.5, 2),),)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    assert "-0.000000" not in text
    assert "0.000000" in text
    assert "7.500000" in text


# ---------------------------------------------------------------------------
# confirmation, paging and re-entry
# ---------------------------------------------------------------------------

def test_max_tasks_bounds_newly_confirmed(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    updates = (("b0", build_tile_pyramid(_POINTS_B0)),)
    n_tasks = len(json.loads(plan)["tasks"])

    first = execute_tile_update_plan(plan, updates, max_tasks=2)
    assert json.loads(first)["confirmed"] == 2
    assert json.loads(first)["complete"] is False

    second = execute_tile_update_plan(plan, updates, state=first, max_tasks=2)
    assert json.loads(second)["confirmed"] == min(4, n_tasks)

    full = execute_tile_update_plan(plan, updates)
    assert json.loads(full)["confirmed"] == n_tasks
    assert json.loads(full)["complete"] is True
    assert execute_tile_update_plan(plan, updates, max_tasks=n_tasks) == full


def test_state_resume_is_byte_for_byte_equal(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    updates = (("b0", build_tile_pyramid(_POINTS_B0)),)
    first = execute_tile_update_plan(plan, updates, max_tasks=1)
    assert execute_tile_update_plan(
        plan, updates, state=first, max_tasks=100) == \
        execute_tile_update_plan(plan, updates)


def test_zero_step_returns_state_byte_for_byte(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    updates = (("b0", build_tile_pyramid(_POINTS_B0)),)
    first = execute_tile_update_plan(plan, updates, max_tasks=2)
    assert execute_tile_update_plan(
        plan, updates, state=first, max_tasks=0) == first
    assert execute_tile_update_plan(
        plan, updates, state=first, max_tasks=0) is first


def test_zero_step_without_state_confirms_nothing():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    text = execute_tile_update_plan(
        plan, (("b0", pyramid),), max_tasks=0)
    document = json.loads(text)
    assert document["confirmed"] == 0
    assert document["results"] == []
    assert document["complete"] is False


def test_completed_state_reenters_byte_for_byte(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    updates = (("b0", build_tile_pyramid(_POINTS_B0)),)
    full = execute_tile_update_plan(plan, updates)
    assert execute_tile_update_plan(plan, updates, state=full) == full
    assert execute_tile_update_plan(
        plan, updates, state=full, max_tasks=0) == full
    assert execute_tile_update_plan(
        plan, updates, state=full, max_tasks=5) == full


def test_repeated_calls_are_byte_for_byte_equal(scene, paths):
    state_text = _unfinished_state(scene, paths, "one_multi_batch")
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    updates = (("b0", build_tile_pyramid(_POINTS_B0)),)
    first = execute_tile_update_plan(plan, updates)
    second = execute_tile_update_plan(plan, updates)
    assert first == second


def test_extra_unreferenced_updates_are_allowed():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    other = (((9, 9, 2304, 2304, 2559, 2559, 0.0, 0.0, 1),),)
    text = execute_tile_update_plan(
        plan, (("b0", pyramid), ("zz", other)))
    assert json.loads(text)["complete"] is True


# ---------------------------------------------------------------------------
# validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_plan_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(bad, ())


@pytest.mark.parametrize("bad", [[], None, 1, "x", {}])
def test_updates_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan('{"tasks":[],"complete":true}', bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), {},
])
def test_update_shape_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (bad,))


def test_pyramid_wrong_type():
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (("b0", []),))


@pytest.mark.parametrize("bad", [1, [], b"x", {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (), state=bad)


@pytest.mark.parametrize("bad", [True, 1.0, "1", 0.0])
def test_max_tasks_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (), max_tasks=bad)


# ---------------------------------------------------------------------------
# validation: values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "{not json",
    "",
    "null",
    "[]",
    "1",
    '{"tasks":[]}',
    '{"complete":true}',
    '{"tasks":[],"complete":true} ',
    ' {"tasks":[],"complete":true}',
    '{"tasks":[],"complete":1}',
    '{"tasks":{},"complete":true}',
    '{"complete":true,"tasks":[]}',
    '{"tasks":[[]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,[1,0,1]]]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[["b0",2,1,null]]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[["b0",-1,0,null]]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[["b~x",1,2,null]]]],"complete":false}',
    '{"tasks":[[0,256,0,255,255,[["b0",1,2,null]]]],"complete":false}',
    '{"tasks":[[-1,0,0,255,255,[["b0",1,2,null]]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":true}',
    '{"tasks":[[0,0,0,255,255,[["b0",true,2,null]]]],"complete":false}',
    '{"tasks":[[true,0,0,255,255,[["b0",1,2,null]]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,"b0"]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[[1,1,2,null]]]],"complete":false}',
])
def test_non_canonical_plan_rejected(bad):
    with pytest.raises(ValueError):
        execute_tile_update_plan(bad, ())


def test_well_formed_but_spaced_plan_rejected():
    plan = '{"tasks": [], "complete": true}'
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, ())


def test_duplicate_tasks_rejected():
    plan = ('{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
            '[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}')
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", ()),))


def test_unsorted_tasks_rejected():
    plan = ('{"tasks":[[0,256,256,511,511,[["b0",1,2,null]]],'
            '[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}')
    pyramid = (
        ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),
         (1, 1, 256, 256, 511, 511, 1.0, 2.0, 1)),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


def test_duplicate_source_in_task_rejected():
    plan = ('{"tasks":[[0,0,0,255,255,'
            '[["b0",1,2,null],["b0",1,2,null]]]],"complete":false}')
    pyramid = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


@pytest.mark.parametrize("tasks", [
    # closed windows intersect
    '[[0,0,0,255,255,[["b0",1,2,null]]],[0,255,0,511,255,[["b0",1,2,null]]]]',
    # one-cell x gap with y intervals intersecting
    '[[0,0,0,255,255,[["b0",1,2,null]]],[0,256,0,511,255,[["b0",1,2,null]]]]',
    # one-cell y gap with x intervals intersecting
    '[[0,0,0,255,255,[["b0",1,2,null]]],[0,0,256,255,511,[["b0",1,2,null]]]]',
])
def test_connected_tasks_rejected(tasks):
    plan = '{"tasks":' + tasks + ',"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


def test_tasks_at_different_levels_never_connect():
    plan = ('{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
            '[1,0,0,511,511,[["b0",1,2,null]]]],"complete":false}')
    pyramid = (
        ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),
        ((0, 0, 0, 0, 511, 511, 1.0, 2.0, 1),))
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    assert json.loads(text)["confirmed"] == 2


def test_missing_source_rejected():
    plan = ('{"tasks":[[0,0,0,255,255,'
            '[["b0",1,2,null],["b1",0,1,null]]]],"complete":false}')
    pyramid = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


def test_duplicate_update_id_rejected():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan, (("b0", pyramid), ("b0", pyramid)))


def test_negative_max_tasks_rejected():
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (), max_tasks=-1)


def test_pyramid_wrong_level_count_rejected():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (
        ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),
        ((0, 0, 0, 0, 511, 511, 1.0, 2.0, 1),))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


@pytest.mark.parametrize("tile", [
    # width 257: not a power of two
    (0, 0, 0, 0, 256, 256, 1.0, 2.0, 1),
    # width 256 but ix0 does not equal tx * width
    (1, 0, 200, 0, 455, 255, 1.0, 2.0, 1),
    # non-square tile
    (0, 0, 0, 0, 255, 127, 1.0, 2.0, 1),
])
def test_pyramid_misaligned_tiles_rejected(tile):
    plan = '{"tasks":[[0,0,0,511,511,[["b0",1,2,null]]]],"complete":false}'
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", ((tile,),)),))


def test_pyramid_level_widths_must_double():
    plan = ('{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
            '[1,0,0,511,511,[["b0",1,2,null]]]],"complete":false}')
    pyramid = (
        ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),
        # Level 1 must double the level-0 width 256 to 512.
        ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


def test_same_coordinate_tiles_with_different_bounds_rejected():
    plan = ('{"tasks":[[0,0,0,255,255,'
            '[["b0",1,2,null],["b1",0,1,null]]]],"complete":false}')
    pyramid_b0 = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    pyramid_b1 = (((0, 0, 0, 0, 127, 127, 2.0, 7.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan, (("b0", pyramid_b0), ("b1", pyramid_b1)))


def test_pyramid_structure_value_error():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    # Unsorted tiles within the level.
    pyramid = (
        ((1, 1, 256, 256, 511, 511, 1.0, 2.0, 1),
         (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1)),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))


# ---------------------------------------------------------------------------
# validation: state
# ---------------------------------------------------------------------------

def _one_task_plan_and_updates():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]]],"complete":false}'
    pyramid = (((0, 0, 0, 0, 255, 255, 5.0, 9.0, 2),),)
    return plan, (("b0", pyramid),)


@pytest.mark.parametrize("bad", [
    "",
    "{",
    "null",
    "[]",
    '{"confirmed":0,"results":[],"complete":false}',
    '{"plan":{},"confirmed":0,"results":[],"complete":false}',
])
def test_malformed_state_rejected(bad):
    plan, updates = _one_task_plan_and_updates()
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, updates, state=bad)


def test_state_embedding_other_plan_rejected():
    plan, updates = _one_task_plan_and_updates()
    other = '{"tasks":[[0,0,0,255,255,[["b1",1,2,null]]]],"complete":false}'
    other_pyramid = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)
    state = execute_tile_update_plan(other, (("b1", other_pyramid),))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, updates, state=state)


def test_state_with_stale_results_rejected():
    plan, updates = _one_task_plan_and_updates()
    first = execute_tile_update_plan(plan, updates, max_tasks=1)
    changed_pyramid = (((0, 0, 0, 0, 255, 255, 50.0, 90.0, 20),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan, (("b0", changed_pyramid),), state=first)


def test_state_confirmed_out_of_range_rejected():
    plan, updates = _one_task_plan_and_updates()
    state = ('{"plan":' + plan + ',"confirmed":2,"results":[],'
             '"complete":false}')
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, updates, state=state)


def test_state_complete_flag_must_agree():
    plan, updates = _one_task_plan_and_updates()
    good = execute_tile_update_plan(plan, updates, max_tasks=1)
    assert good.endswith('"complete":true}')
    tampered = good[:-len('"complete":true}')] + '"complete":false}'
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, updates, state=tampered)


def test_non_canonical_state_spacing_rejected():
    plan, updates = _one_task_plan_and_updates()
    good = execute_tile_update_plan(plan, updates, max_tasks=1)
    spaced = good.replace(',"results"', ', "results"')
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, updates, state=spaced)
