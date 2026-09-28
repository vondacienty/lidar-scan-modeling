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

scene = publish_fixtures.scene
paths = publish_fixtures.paths

_PLAN_ONE = ('{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],'
             '"complete":false}')
_PLAN_TWO = ('{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
             '[0,512,512,767,767,[["b0",1,1,null]]]],'
             '"complete":false}')
_PLAN_TWO_LEVELS = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
    '[1,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')


def _pyramid(points, levels=1):
    return build_tile_pyramid(points, levels=levels)


def _execution(plan=_PLAN_ONE, pyramid=None, **kwargs):
    if pyramid is None:
        pyramid = _pyramid(((10.0, 10.0, 5.0, 0, 1.0),))
    return execute_tile_update_plan(plan, (("b0", pyramid),), **kwargs)


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
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution()
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert list(document) == [
        "plan", "base", "committed", "receipts", "pyramid", "complete"]
    assert document["plan"] == json.loads(_PLAN_ONE)
    assert document["base"] == [[[0, 0, 0, 0, 255, 255, 3.0, 3.0, 1]]]
    assert not re.search(r"[ \t\n\r]", text)
    assert not text.endswith("\n")
    for raw in re.findall(r"-?\d+\.\d+", text):
        assert re.fullmatch(r"-?\d+\.\d{6}", raw)
    assert "NaN" not in text and "Infinity" not in text


def test_plan_is_embedded_byte_for_byte():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution()
    text = commit_tile_update_plan(base, execution)
    assert '"plan":' + _PLAN_ONE + "," in text


def test_receipts_are_the_six_item_result_prefix():
    base = _pyramid(())
    text = commit_tile_update_plan(base, _execution())
    assert json.loads(text)["receipts"] == [
        [0, 0, 0, 255, 255, [["b0", 1, 1, None]]]]


# ---------------------------------------------------------------------------
# commit semantics
# ---------------------------------------------------------------------------

def test_old_intersecting_tiles_are_replaced_and_others_kept():
    base = (
        (
            (0, 0, 0, 0, 255, 255, 3.0, 3.0, 1),
            (1, 1, 256, 256, 511, 511, 8.0, 8.0, 1),
            (2, 2, 512, 512, 767, 767, 4.0, 4.0, 1),
        ),
    )
    new_pyramid = _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    ))
    text = commit_tile_update_plan(base, _execution(_PLAN_TWO, new_pyramid))
    document = json.loads(text)
    assert document["committed"] == 2
    assert document["complete"] is True
    assert document["pyramid"] == [[
        [0, 0, 0, 0, 255, 255, 5.0, 5.0, 1],
        [1, 1, 256, 256, 511, 511, 8.0, 8.0, 1],
        [2, 2, 512, 512, 767, 767, 6.0, 6.0, 1],
    ]]


def test_levels_not_named_by_a_result_are_untouched():
    base = build_tile_pyramid(((1.0, 1.0, 3.0, 0, 1.0),), levels=3)
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=3)
    text = commit_tile_update_plan(base, _execution(_PLAN_ONE, pyramid))
    document = json.loads(text)
    assert document["pyramid"][0] == [[0, 0, 0, 0, 255, 255, 5.0, 5.0, 1]]
    assert document["pyramid"][1] == document["base"][1]
    assert document["pyramid"][2] == document["base"][2]


def test_right_level_is_modified_per_result():
    base = build_tile_pyramid(((1.0, 1.0, 3.0, 0, 1.0),), levels=3)
    pyramid = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0), (300.0, 300.0, 7.0, 0, 1.0)),
        levels=3)
    text = commit_tile_update_plan(
        base, _execution(_PLAN_TWO_LEVELS, pyramid))
    document = json.loads(text)
    assert document["pyramid"][0] == [[0, 0, 0, 0, 255, 255, 5.0, 5.0, 1]]
    assert document["pyramid"][1] == [[0, 0, 0, 0, 511, 511, 5.0, 7.0, 2]]
    assert document["pyramid"][2] == document["base"][2]


def test_result_with_no_tiles_deletes_intersecting_old_tiles():
    base = (((2, 2, 512, 512, 767, 767, 3.0, 3.0, 1),),)
    pyramid = _pyramid(((10.0, 10.0, 5.0, 0, 1.0),))
    plan = ('{"tasks":[[0,512,512,767,767,[["b0",1,1,null]]]],'
            '"complete":false}')
    execution = _execution(plan, pyramid)
    assert json.loads(execution)["results"][0][6] == []
    text = commit_tile_update_plan(base, execution)
    assert json.loads(text)["pyramid"] == [[]]


def test_inserted_tiles_are_sorted_inside_the_level():
    base = (((0, 0, 0, 0, 255, 255, 3.0, 3.0, 1),
             (4, 4, 1024, 1024, 1279, 1279, 9.0, 9.0, 1)),)
    plan = ('{"tasks":[[0,0,0,1023,1023,[["b0",1,1,null]]]],'
            '"complete":false}')
    execution = _execution(plan, _pyramid((
        (600.0, 600.0, 6.0, 0, 1.0),
        (300.0, 300.0, 7.0, 0, 1.0),
    )))
    text = commit_tile_update_plan(base, execution)
    tiles = json.loads(text)["pyramid"][0]
    assert [(tile[0], tile[1]) for tile in tiles] == [(1, 1), (2, 2), (4, 4)]


def test_closed_intervals_intersect_at_endpoints():
    # Base tile covers cells [0,255]; window starts at 255 -> intersects;
    # a window starting at 256 does not.
    overlapping = (((0, 0, 0, 0, 255, 255, 3.0, 3.0, 1),),)
    pyramid = _pyramid(((600.0, 600.0, 6.0, 0, 1.0),))
    plan_touch = ('{"tasks":[[0,255,255,255,255,[["b0",1,1,null]]]],'
                  '"complete":false}')
    text = commit_tile_update_plan(
        overlapping, _execution(plan_touch, pyramid))
    assert json.loads(text)["pyramid"] == [[]]

    plan_gap = ('{"tasks":[[0,256,256,256,256,[["b0",1,1,null]]]],'
                '"complete":false}')
    text = commit_tile_update_plan(
        overlapping, _execution(plan_gap, pyramid))
    assert json.loads(text)["pyramid"] == [
        [[0, 0, 0, 0, 255, 255, 3.0, 3.0, 1]]]


def test_negative_zero_normalized():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    pyramid = _pyramid(((10.0, 10.0, -1e-9, 0, 1.0),))
    text = commit_tile_update_plan(base, _execution(_PLAN_ONE, pyramid))
    assert "0.000000" in text
    assert "-0.000000" not in text


def test_repeated_calls_are_byte_for_byte_equal():
    base = build_tile_pyramid(((1.0, 1.0, 3.0, 0, 1.0),), levels=3)
    execution = _execution(
        _PLAN_TWO_LEVELS,
        build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=3))
    first = commit_tile_update_plan(base, execution)
    second = commit_tile_update_plan(base, execution)
    assert first == second


# ---------------------------------------------------------------------------
# partial executions, max_tasks and state re-entry
# ---------------------------------------------------------------------------

def test_complete_is_true_when_no_result_is_confirmed():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution(max_tasks=0)
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert document["committed"] == 0
    assert document["receipts"] == []
    assert document["complete"] is True
    assert document["pyramid"] == document["base"]


def test_empty_completed_plan_commits_nothing():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = execute_tile_update_plan(
        '{"tasks":[],"complete":true}', ())
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert document["committed"] == 0
    assert document["receipts"] == []
    assert document["pyramid"] == document["base"]
    assert document["complete"] is True
    assert commit_tile_update_plan(base, execution, state=text) == text


def test_committing_the_confirmed_prefix_completes_even_with_tasks_left():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution(_PLAN_TWO, _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    )), max_tasks=1)
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert document["committed"] == 1
    assert document["complete"] is True
    assert [receipt[:5] for receipt in document["receipts"]] == [
        [0, 0, 0, 255, 255]]


def test_max_tasks_limits_newly_committed_results():
    base = _pyramid(())
    execution = _execution(_PLAN_TWO, _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    )))
    text = commit_tile_update_plan(base, execution, max_tasks=1)
    document = json.loads(text)
    assert document["committed"] == 1
    assert document["complete"] is False
    assert [receipt[:5] for receipt in document["receipts"]] == [
        [0, 0, 0, 255, 255]]


def test_zero_step_without_state_returns_unchanged_base_document():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution()
    text = commit_tile_update_plan(base, execution, max_tasks=0)
    assert text == (
        '{"plan":' + _PLAN_ONE + ',"base":[[[0,0,0,0,255,255,3.000000,'
        '3.000000,1]]],"committed":0,"receipts":[],"pyramid":[[[0,0,0,0,'
        '255,255,3.000000,3.000000,1]]],"complete":false}')


def test_re_entering_a_state_is_byte_identical():
    base = _pyramid(())
    execution = _execution(_PLAN_TWO, _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    )))
    first = commit_tile_update_plan(base, execution, max_tasks=1)
    again = commit_tile_update_plan(base, execution, state=first, max_tasks=0)
    assert again == first
    completed = commit_tile_update_plan(base, execution, state=first)
    assert commit_tile_update_plan(base, execution, state=completed) \
        == completed
    assert commit_tile_update_plan(
        base, execution, state=completed, max_tasks=0) == completed
    assert commit_tile_update_plan(
        base, execution, state=completed, max_tasks=5) == completed


def test_chunked_commit_matches_one_shot():
    base = build_tile_pyramid(((1.0, 1.0, 3.0, 0, 1.0),), levels=3)
    execution = _execution(
        _PLAN_TWO_LEVELS,
        build_tile_pyramid(
            ((10.0, 10.0, 5.0, 0, 1.0), (300.0, 300.0, 7.0, 0, 1.0)),
            levels=3))
    one_shot = commit_tile_update_plan(base, execution)
    state = commit_tile_update_plan(base, execution, max_tasks=0)
    state = commit_tile_update_plan(
        base, execution, state=state, max_tasks=1)
    state = commit_tile_update_plan(
        base, execution, state=state, max_tasks=1)
    assert state == one_shot
    assert json.loads(state)["complete"] is True


# ---------------------------------------------------------------------------
# TypeError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(bad, _execution())


@pytest.mark.parametrize("bad", [None, 1, (), b"x", {}])
def test_execution_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_pyramid(()), bad)


@pytest.mark.parametrize("bad", [1, 1.0, (), [], {}])
def test_state_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_pyramid(()), _execution(), state=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(bad):
    with pytest.raises(TypeError):
        commit_tile_update_plan(_pyramid(()), _execution(), max_tasks=bad)


# ---------------------------------------------------------------------------
# ValueError validation
# ---------------------------------------------------------------------------

def test_base_must_have_at_least_one_level():
    with pytest.raises(ValueError):
        commit_tile_update_plan((), _execution())


def test_base_structure_and_value_errors():
    good_pyramid = _pyramid(())
    execution = _execution()
    cases = [
        (([],),),
        ((([],),)),
        ((([0, 0, 0, 0, 255, 255, 1.0, 1.0, 1],),),),
        (((0, 0, 0, 0, 255, 255, 1.0, float("nan"), 1),),),
        (((1, 0, 256, 0, 511, 255, 1.0, 2.0, 1),
          (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1)),),
        (((0, 0, 256, 0, 0, 255, 1.0, 2.0, 1),),),
        (((0, 0, 0, 0, 255, 255, 2.0, 1.0, 1),),),
        (((0, 0, 0, 0, 255, 255, 1.0, 2.0, -1),),),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            commit_tile_update_plan(case, execution)
    # a merely valid shape is accepted
    commit_tile_update_plan(good_pyramid, execution)


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"confirmed":0,"results":[],"complete":false}',
])
def test_malformed_execution_rejected(tampered):
    with pytest.raises(ValueError):
        commit_tile_update_plan(_pyramid(()), tampered)


def test_execution_plan_with_tasks_and_complete_true_rejected():
    plan = ('{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],'
            '"complete":true}')
    execution = (
        '{"plan":' + plan + ',"confirmed":0,"results":[],"complete":false}')
    with pytest.raises(ValueError):
        commit_tile_update_plan(_pyramid(()), execution)


def test_confirmed_result_level_not_covered_by_base():
    execution = _execution(_PLAN_TWO_LEVELS, build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0),), levels=3))
    with pytest.raises(ValueError):
        commit_tile_update_plan(build_tile_pyramid((), levels=1), execution)


def test_max_tasks_negative_rejected():
    with pytest.raises(ValueError):
        commit_tile_update_plan(_pyramid(()), _execution(), max_tasks=-1)


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"base":[],"committed":0,"receipts":[],"pyramid":[],"complete":false}',
])
def test_malformed_state_rejected(tampered):
    with pytest.raises(ValueError):
        commit_tile_update_plan(_pyramid(()), _execution(), state=tampered)


def _zero_step_state(base, execution):
    return commit_tile_update_plan(base, execution, max_tasks=0)


def test_state_must_embed_the_same_plan():
    base = build_tile_pyramid(((1.0, 1.0, 3.0, 0, 1.0),), levels=3)
    state = _zero_step_state(base, _execution(
        _PLAN_TWO_LEVELS,
        build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=3)))
    other = _execution(
        _PLAN_ONE,
        build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=3))
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, other, state=state)


def test_state_must_embed_the_same_base():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    state = _zero_step_state(base, _execution())
    other_base = _pyramid(((2.0, 2.0, 4.0, 0, 1.0),))
    with pytest.raises(ValueError):
        commit_tile_update_plan(other_base, _execution(), state=state)


def test_state_committed_must_not_exceed_confirmed_prefix():
    base = _pyramid(())
    execution = _execution()
    state = _zero_step_state(base, execution)
    tampered = state.replace('"committed":0', '"committed":1', 1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution, state=tampered)


def test_state_receipts_must_match_committed_prefix():
    base = _pyramid(())
    execution = _execution(_PLAN_TWO, _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    )))
    state = commit_tile_update_plan(base, execution, max_tasks=1)
    tampered = state.replace("[0,0,0,255,255,", "[0,0,0,512,512,", 1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution, state=tampered)


def test_state_pyramid_must_match_prefix_applied_to_base():
    base = _pyramid(((1.0, 1.0, 3.0, 0, 1.0),))
    execution = _execution()
    state = _zero_step_state(base, execution)
    tampered = state.replace("3.000000,3.000000", "4.000000,4.000000", 1)
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution, state=tampered)


def test_state_complete_flag_must_agree_with_prefix():
    base = _pyramid(())
    execution = _execution(_PLAN_TWO, _pyramid((
        (10.0, 10.0, 5.0, 0, 1.0),
        (600.0, 600.0, 6.0, 0, 1.0),
    )))
    state = commit_tile_update_plan(base, execution, max_tasks=1)
    tampered = state[:-len('"complete":false}')] + '"complete":true}'
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, execution, state=tampered)


def test_state_trailing_whitespace_rejected():
    base = _pyramid(())
    state = _zero_step_state(base, _execution())
    with pytest.raises(ValueError):
        commit_tile_update_plan(base, _execution(), state=state + " ")


# ---------------------------------------------------------------------------
# integration with build_tile_update_plan
# ---------------------------------------------------------------------------

_POINTS_B0 = (
    (0.0, 0.0, 5.0, 1, 1.0),
    (255.0, 255.0, 9.0, 2, 1.0),
    (256.0, 256.0, 7.0, 1, 2.0),
    (-1.0, 300.0, 4.0, 3, 1.5),
)


def _unfinished_state(scene_fixture, path_fixture):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    return publish_updates(
        path_fixture["state"], path_fixture["index"],
        scene_fixture["one_multi_batch"], limit=1)


def test_end_to_end_against_a_built_plan(scene, paths):
    state_text = _unfinished_state(scene, paths)
    plan = build_tile_update_plan(state_text, (("b0", _POINTS_B0),))
    pyramid = build_tile_pyramid(_POINTS_B0)
    execution = execute_tile_update_plan(plan, (("b0", pyramid),))
    base = build_tile_pyramid(((12.0, 12.0, 2.0, 0, 1.0),))
    text = commit_tile_update_plan(base, execution)
    document = json.loads(text)
    assert document["plan"] == json.loads(plan)
    assert document["committed"] == len(document["plan"]["tasks"])
    assert document["complete"] is True
    assert len(document["receipts"]) == document["committed"]
    # Every committed receipt matches the corresponding execution result.
    for receipt, result in zip(document["receipts"],
                               json.loads(execution)["results"]):
        assert receipt == result[:6]
    # Re-entering the completed document is byte-identical.
    assert commit_tile_update_plan(base, execution, state=text) == text
