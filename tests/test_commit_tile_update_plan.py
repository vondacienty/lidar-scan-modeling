"""Tests for :func:`lidar_scan.commit_tile_update_plan`."""

from __future__ import annotations

import json
import re

import pytest

from lidar_scan import (build_tile_pyramid, build_tile_update_plan,
                        commit_tile_update_plan, execute_tile_update_plan,
                        publish_updates)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_publish_updates as publish_fixtures

# Reuse the prepared publication scene and path layout.
scene = publish_fixtures.scene
paths = publish_fixtures.paths

_PLAN_TWO = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
    '[0,512,512,767,767,[["b0",1,2,null]]]],"complete":false}')

_PLAN_TWO_LEVELS = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
    '[1,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')

_PLAN_EMPTY = '{"tasks":[],"complete":true}'


def _new_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (600.0, 600.0, 9.0, 1, 1.0)), levels=1)


def _base_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 1.0, 1, 1.0),
         (600.0, 600.0, 2.0, 1, 1.0),
         (900.0, 900.0, 3.0, 1, 1.0)), levels=1)


def _execution(plan=_PLAN_TWO, pyramid=None, **kwargs):
    if pyramid is None:
        pyramid = _new_pyramid()
    sources = sorted({source[0]
                      for task in json.loads(plan)["tasks"]
                      for source in task[5]})
    updates = tuple((source_id, pyramid) for source_id in sources)
    return execute_tile_update_plan(plan, updates, **kwargs)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.commit_tile_update_plan is commit_tile_update_plan
    assert "commit_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_document_shape_and_embedded_documents():
    base = _base_pyramid()
    text = commit_tile_update_plan(base, _execution())
    assert text.startswith('{"plan":' + _PLAN_TWO + ",")
    assert not text.endswith("\n")
    document = json.loads(text)
    assert list(document) == [
        "plan", "base", "committed", "receipts", "pyramid", "complete"]
    assert document["plan"] == json.loads(_PLAN_TWO)
    assert document["base"] == [[list(tile) for tile in level]
                                for level in base]
    assert not re.search(r"[ \t\n\r]", text)
    for raw in re.findall(r"-?\d+\.\d+", text):
        assert re.fullmatch(r"-?\d+\.\d{6}", raw)


def test_replace_intersecting_keep_disjoint_and_sort():
    text = commit_tile_update_plan(_base_pyramid(), _execution())
    pyramid = json.loads(text)["pyramid"]
    assert pyramid == [[
        [0, 0, 0, 0, 255, 255, 5.0, 5.0, 1],      # old (0,0) replaced
        [2, 2, 512, 512, 767, 767, 9.0, 9.0, 1],  # old (2,2) replaced
        [3, 3, 768, 768, 1023, 1023, 3.0, 3.0, 1],  # outside windows kept
    ]]
    keys = [(tile[0], tile[1]) for tile in pyramid[0]]
    assert keys == sorted(keys)


def test_untouched_levels_are_preserved():
    new_pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=2)
    base = build_tile_pyramid(
        ((10.0, 10.0, 1.0, 1, 1.0), (100000.0, 100000.0, 7.0, 1, 1.0)),
        levels=2)
    execution = _execution(_PLAN_TWO_LEVELS, new_pyramid)
    document = json.loads(commit_tile_update_plan(base, execution))
    # Level 0 task and level 1 task each replace the (0,0) tile; every
    # other tile keeps the base values.
    assert document["pyramid"][0] == [
        [0, 0, 0, 0, 255, 255, 5.0, 5.0, 1],
        [390, 390, 99840, 99840, 100095, 100095, 7.0, 7.0, 1]]
    assert document["pyramid"][1] == [
        [0, 0, 0, 0, 511, 511, 5.0, 5.0, 1],
        [195, 195, 99840, 99840, 100351, 100351, 7.0, 7.0, 1]]
    # A plan with no level-1 task leaves level 1 byte-for-byte intact.
    plan_one = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    execution_one = _execution(plan_one, new_pyramid)
    document_one = json.loads(commit_tile_update_plan(base, execution_one))
    assert document_one["pyramid"][1] == [[list(tile) for tile in base[1]]][0]


def test_empty_result_tiles_delete_the_old_tiles():
    plan = '{"tasks":[[0,512,512,767,767,[["b0",1,1,null]]]],"complete":false}'
    new_pyramid = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0),), levels=1)  # tile (0,0), outside window
    base = build_tile_pyramid(
        ((600.0, 600.0, 2.0, 0, 1.0),
         (10.0, 10.0, 1.0, 0, 1.0)), levels=1)
    execution = execute_tile_update_plan(
        plan, (("b0", new_pyramid),))
    assert json.loads(execution)["results"][0][6] == []
    document = json.loads(commit_tile_update_plan(base, execution))
    assert document["pyramid"] == [[[0, 0, 0, 0, 255, 255, 1.0, 1.0, 1]]]


def test_receipts_are_the_six_item_plan_prefix():
    document = json.loads(
        commit_tile_update_plan(_base_pyramid(), _execution()))
    assert document["receipts"] == [
        [0, 0, 0, 255, 255, [["b0", 1, 2, None]]],
        [0, 512, 512, 767, 767, [["b0", 1, 2, None]]],
    ]


def test_complete_requires_every_task_committed():
    # The execution only confirms one of the two tasks: the commit can
    # never be complete, even with max_tasks=None.
    partial = _execution(max_tasks=1)
    document = json.loads(
        commit_tile_update_plan(_base_pyramid(), partial))
    assert document["committed"] == 1
    assert document["complete"] is False
    assert len(document["receipts"]) == 1

    full = _execution()
    document = json.loads(
        commit_tile_update_plan(_base_pyramid(), full))
    assert document["committed"] == 2
    assert document["complete"] is True


def test_empty_completed_plan_keeps_the_base_pyramid():
    execution = execute_tile_update_plan(_PLAN_EMPTY, ())
    base = _base_pyramid()
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert document["committed"] == 0
    assert document["receipts"] == []
    assert document["complete"] is True
    assert document["pyramid"] == document["base"]
    assert document["pyramid"] == [[list(tile) for tile in level]
                                   for level in base]


def test_z_uses_six_decimals_and_negative_zero():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    new_pyramid = build_tile_pyramid(
        ((10.0, 10.0, -1e-9, 0, 1.0),), levels=1)
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 0, 1.0),), levels=1)
    execution = execute_tile_update_plan(plan, (("b0", new_pyramid),))
    text = commit_tile_update_plan(base, execution)
    assert "[0,0,0,0,255,255,0.000000,0.000000,1]" in text
    assert "-0.000000" not in text


# ---------------------------------------------------------------------------
# max_tasks and re-entry
# ---------------------------------------------------------------------------

def test_max_tasks_limits_newly_committed_results():
    document = json.loads(
        commit_tile_update_plan(_base_pyramid(), _execution(), max_tasks=1))
    assert document["committed"] == 1
    assert len(document["receipts"]) == 1
    assert document["complete"] is False
    # Only the first window was replaced; the old (2,2) tile survives.
    assert document["pyramid"][0][1] == [2, 2, 512, 512, 767, 767,
                                         2.0, 2.0, 1]


def test_max_tasks_is_clamped_to_confirmed_results():
    partial = _execution(max_tasks=1)
    document = json.loads(
        commit_tile_update_plan(_base_pyramid(), partial, max_tasks=99))
    assert document["committed"] == 1
    assert document["complete"] is False


def test_zero_step_fresh_state_is_canonical():
    base = _base_pyramid()
    # encode_tile_pyramid wraps the bare array as {"levels": ...}; strip
    # that wrapper to obtain the canonical base/pyramid array spelling.
    array_text = tiles_module.encode_tile_pyramid(base)
    array_text = array_text[len('{"levels":'):-1]
    text = commit_tile_update_plan(base, _execution(), max_tasks=0)
    document = json.loads(text)
    assert document["committed"] == 0
    assert document["receipts"] == []
    assert document["complete"] is False
    assert document["pyramid"] == [[list(tile) for tile in level]
                                   for level in base]
    assert text == (
        '{"plan":' + _PLAN_TWO + ',"base":' + array_text
        + ',"committed":0,"receipts":[],"pyramid":' + array_text
        + ',"complete":false}')


def test_re_entering_a_state_is_byte_identical():
    base = _base_pyramid()
    first = commit_tile_update_plan(base, _execution(), max_tasks=1)
    assert commit_tile_update_plan(
        base, _execution(), state=first, max_tasks=0) == first
    completed = commit_tile_update_plan(base, _execution(), state=first)
    assert commit_tile_update_plan(
        base, _execution(), state=completed) == completed
    assert commit_tile_update_plan(
        base, _execution(), state=completed, max_tasks=0) == completed


def test_chunked_commit_matches_one_shot():
    base = _base_pyramid()
    pyramid = _new_pyramid()
    execution = execute_tile_update_plan(_PLAN_TWO, (("b0", pyramid),),
                                         max_tasks=0)
    one_shot = commit_tile_update_plan(
        base, execute_tile_update_plan(_PLAN_TWO, (("b0", pyramid),)))
    state = None
    for _ in range(2):
        execution = execute_tile_update_plan(
            _PLAN_TWO, (("b0", pyramid),), state=execution, max_tasks=1)
        state = commit_tile_update_plan(
            base, execution, state=state, max_tasks=1)
    assert state == one_shot
    assert json.loads(state)["complete"] is True


def test_repeated_calls_are_byte_for_byte_equal():
    base = _base_pyramid()
    execution = _execution()
    first = commit_tile_update_plan(base, execution)
    second = commit_tile_update_plan(base, execution)
    assert first == second


def test_state_pyramid_must_reproduce_from_base_and_results():
    base = _base_pyramid()
    first = commit_tile_update_plan(base, _execution(), max_tasks=1)
    # An execution recomputed with a different z value inside the already
    # committed window must not continue the existing state.
    changed = build_tile_pyramid(
        ((10.0, 10.0, 99.0, 1, 1.0),
         (600.0, 600.0, 9.0, 1, 1.0)), levels=1)
    changed_execution = execute_tile_update_plan(
        _PLAN_TWO, (("b0", changed),), max_tasks=2)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, changed_execution, state=first)

    # A change that only touches a result beyond the committed prefix
    # leaves the prefix reproducible and is therefore accepted.
    future = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (600.0, 600.0, 99.0, 1, 1.0)), levels=1)
    future_execution = execute_tile_update_plan(
        _PLAN_TWO, (("b0", future),), max_tasks=2)
    continued = commit_tile_update_plan(
        base, future_execution, state=first)
    assert json.loads(continued)["committed"] == 2


def test_state_must_match_the_execution_plan_and_base():
    base = _base_pyramid()
    state = commit_tile_update_plan(base, _execution())
    other_base = build_tile_pyramid(
        ((999.0, 999.0, 1.0, 1, 1.0),), levels=1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(other_base, _execution(), state=state)
    other_plan = (
        '{"tasks":[[0,0,0,511,511,[["b0",1,2,null]]]],"complete":false}')
    other_execution = execute_tile_update_plan(
        other_plan, (("b0", _new_pyramid()),))
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, other_execution, state=state)


def test_state_may_not_commit_more_than_the_execution_confirms():
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 1, 1.0),), levels=1)
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    full = execute_tile_update_plan(plan, (("b0", pyramid),))
    state = commit_tile_update_plan(base, full)
    unconfirmed = execute_tile_update_plan(
        plan, (("b0", pyramid),), max_tasks=0)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, unconfirmed, state=state)


def test_same_coordinate_insert_collision_is_rejected():
    # Two sources contribute two disjoint tasks but each carries a tile at
    # coordinate (0,0) with different cell bounds. After the first result
    # inserts (0,0), the second result's insert collides with the survivor.
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
        '[0,512,512,767,767,[["b1",1,1,null]]]],"complete":false}')
    pyramid_b0 = (((0, 0, 0, 0, 255, 255, 5.0, 5.0, 1),),)
    pyramid_b1 = (((0, 0, 512, 512, 767, 767, 9.0, 9.0, 1),),)
    execution = execute_tile_update_plan(
        plan, (("b0", pyramid_b0), ("b1", pyramid_b1)))
    base = build_tile_pyramid(((900.0, 900.0, 3.0, 0, 1.0),), levels=1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution)


# ---------------------------------------------------------------------------
# end to end through publish_updates
# ---------------------------------------------------------------------------

def test_commit_matches_window_delete_insert_oracle(scene, paths):
    with open(paths["index"], "w", encoding="utf-8") as stream:
        stream.write(scene["empty"])
    state_text = publish_updates(
        paths["state"], paths["index"], scene["one_multi_batch"], limit=1)
    points_b0 = (
        (0.0, 0.0, 5.0, 1, 1.0),
        (255.0, 255.0, 9.0, 2, 1.0),
        (256.0, 256.0, 7.0, 1, 2.0),
        (-1.0, 300.0, 4.0, 3, 1.5),
    )
    plan = build_tile_update_plan(state_text, (("b0", points_b0),))
    pyramid = build_tile_pyramid(points_b0)
    execution = execute_tile_update_plan(plan, (("b0", pyramid),))
    base = build_tile_pyramid((
        (-1.0, 300.0, 1.0, 1, 1.0),
        (10.0, 10.0, 2.0, 1, 1.0),
        (70000.0, 70000.0, 8.0, 1, 1.0),
    ))
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)

    windows_by_level = {}
    inserted_by_level = {}
    for result in json.loads(execution)["results"]:
        level = result[0]
        windows_by_level.setdefault(level, []).append(tuple(result[:5]))
        inserted_by_level.setdefault(level, []).extend(
            tuple(tile) for tile in result[6])

    expected = []
    for level, level_tiles in enumerate(base):
        windows = windows_by_level.get(level, [])

        def intersects(tile):
            return any(tile[4] >= w[1] and tile[2] <= w[3]
                       and tile[5] >= w[2] and tile[3] <= w[4]
                       for w in windows)

        # Independent oracle: surviving base tiles keyed by (tx, ty),
        # then result tiles overwrite their coordinate in result order
        # (each commit deletes intersecting tiles before inserting).
        merged = {(tile[0], tile[1]): list(tile)
                  for tile in level_tiles if not intersects(tile)}
        for tile in inserted_by_level.get(level, []):
            merged[(tile[0], tile[1])] = list(tile)
        expected.append([merged[key] for key in sorted(merged)])
    assert document["pyramid"] == expected
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# TypeError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(bad, _execution())


def test_base_container_and_field_types():
    execution = _execution()
    with pytest.raises(TypeError):
        commit_tile_update_plan([()], execution)
    with pytest.raises(TypeError):
        commit_tile_update_plan(([()],), execution)
    with pytest.raises(TypeError):
        commit_tile_update_plan(
            (((0, 0, 0, 0, 255, 255, 1, 2.0, 1),),), execution)
    with pytest.raises(TypeError):
        commit_tile_update_plan(
            (((0, 0, 0, 0, 255, 255, 1.0, 2, 1),),), execution)
    with pytest.raises(TypeError):
        commit_tile_update_plan(
            (((True, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),), execution)


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_execution_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_base_pyramid(), bad)


@pytest.mark.parametrize("bad", [1, 1.0, (), [], {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_base_pyramid(), _execution(), state=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_base_pyramid(), _execution(), max_tasks=bad)


# ---------------------------------------------------------------------------
# ValueError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_base", [
    (),
    (((0, 0, 0, 0, 255, 255, float("nan"), 2.0, 1),),),
    (((1, 0, 256, 0, 511, 255, 1.0, 2.0, 1),
      (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1)),),
    (((0, 0, 256, 0, 0, 255, 1.0, 2.0, 1),),),
])
def test_base_value_errors(bad_base):
    with pytest.raises(ValueError):
        commit_tile_update_plan(bad_base, _execution())


def test_max_tasks_negative_rejected():
    with pytest.raises(ValueError):
        commit_tile_update_plan(_base_pyramid(), _execution(), max_tasks=-1)


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"confirmed":0,"results":[],"complete":false}',
])
def test_malformed_execution_rejected(tampered):
    with pytest.raises(ValueError):
        commit_tile_update_plan(_base_pyramid(), tampered)


@pytest.mark.parametrize("tampered", [
    # tasks listed but the plan claims completion
    '{"plan":{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
    '"complete":true},"confirmed":0,"results":[],"complete":false}',
    # empty task list but the plan claims unfinished work
    '{"plan":{"tasks":[],"complete":false},"confirmed":0,'
    '"results":[],"complete":true}',
    # execution complete flag disagrees with the confirmed prefix
    '{"plan":{"tasks":[],"complete":true},"confirmed":0,'
    '"results":[],"complete":false}',
])
def test_contradictory_plan_or_execution_flags_rejected(tampered):
    with pytest.raises(ValueError):
        commit_tile_update_plan(_base_pyramid(), tampered)


def test_execution_result_must_restate_its_plan_task():
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 1, 1.0),), levels=1)
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    execution = execute_tile_update_plan(plan, (("b0", pyramid),))
    tampered = json.loads(execution)
    tampered["results"][0][1] = 1
    with pytest.raises(ValueError):
        commit_tile_update_plan(
            base, json.dumps(tampered, separators=(",", ":")))


def test_result_level_not_covered_by_base():
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
        '[1,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=2)
    execution = execute_tile_update_plan(plan, (("b0", pyramid),))
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 1, 1.0),), levels=1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution)


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"base":[],"committed":0,"receipts":[],"pyramid":[],'
    '"complete":false}',
])
def test_malformed_state_rejected(tampered):
    with pytest.raises(ValueError):
        commit_tile_update_plan(
            _base_pyramid(), _execution(), state=tampered)


def test_non_canonical_or_tampered_state_rejected():
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 1, 1.0),), levels=1)
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    execution = execute_tile_update_plan(plan, (("b0", pyramid),))
    state = commit_tile_update_plan(base, execution)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution, state=state + " ")
    tampered = json.loads(state)
    tampered["pyramid"][0][0][6] = 99.0
    with pytest.raises(ValueError):
        commit_tile_update_plan(
            base, execution,
            state=json.dumps(tampered, separators=(",", ":")))
    tampered = json.loads(state)
    tampered["receipts"][0][1] = 5
    with pytest.raises(ValueError):
        commit_tile_update_plan(
            base, execution,
            state=json.dumps(tampered, separators=(",", ":")))
