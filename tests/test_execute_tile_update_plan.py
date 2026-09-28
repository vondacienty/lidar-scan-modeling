"""Tests for :func:`lidar_scan.execute_tile_update_plan`."""

from __future__ import annotations

import json
import re

import pytest

from lidar_scan import (build_tile_pyramid, build_tile_update_plan,
                        execute_tile_update_plan, merge_tile_pyramid_windows,
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


def _unfinished_state(scene_fixture, path_fixture):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    return publish_updates(
        path_fixture["state"], path_fixture["index"],
        scene_fixture["one_multi_batch"], limit=1)


def _completed_plan_state(scene_fixture, path_fixture):
    with open(path_fixture["index"], "w", encoding="utf-8") as stream:
        stream.write(scene_fixture["empty"])
    state_text = publish_updates(
        path_fixture["state"], path_fixture["index"],
        scene_fixture["two_batches"])
    return build_tile_update_plan(
        state_text, (("b0", _POINTS_B0), ("b1", ())))


def _plan(scene_fixture, path_fixture, **kwargs):
    state_text = _unfinished_state(scene_fixture, path_fixture)
    return state_text, build_tile_update_plan(
        state_text, (("b0", _POINTS_B0),), **kwargs)


def _pyramid(**kwargs):
    return build_tile_pyramid(_POINTS_B0, **kwargs)


def _updates(pyramid=None):
    return (("b0", _pyramid() if pyramid is None else pyramid),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.execute_tile_update_plan is execute_tile_update_plan
    assert "execute_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_document_shape_and_embedded_plan(scene, paths):
    _state_text, plan = _plan(scene, paths)
    text = execute_tile_update_plan(plan, _updates())
    assert text.startswith('{"plan":' + plan + ",")
    assert not text.endswith("\n")
    document = json.loads(text)
    assert list(document) == ["plan", "confirmed", "results", "complete"]
    assert document["plan"] == json.loads(plan)
    # Compact spelling: no JSON whitespace, and every decimal number is a
    # z value formatted with exactly six decimal places.
    assert not re.search(r"[ \t\n\r]", text)
    for raw in re.findall(r"-?\d+\.\d+", text):
        assert re.fullmatch(r"-?\d+\.\d{6}", raw)


def test_every_task_confirmed_when_max_tasks_is_none(scene, paths):
    _state_text, plan = _plan(scene, paths)
    plan_document = json.loads(plan)
    text = execute_tile_update_plan(plan, _updates())
    document = json.loads(text)
    assert document["confirmed"] == len(plan_document["tasks"])
    assert document["complete"] is True
    assert len(document["results"]) == len(plan_document["tasks"])
    for result, task in zip(document["results"], plan_document["tasks"]):
        assert len(result) == 7
        assert result[:5] == task[:5]
        assert result[5] == task[5]
        for tile in result[6]:
            assert len(tile) == 9


def test_tiles_match_window_merge_oracle(scene, paths):
    _state_text, plan = _plan(scene, paths)
    pyramid = _pyramid()
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    for result, task in zip(json.loads(text)["results"],
                            json.loads(plan)["tasks"]):
        window = tuple(task[:5])
        expected = merge_tile_pyramid_windows(
            (pyramid,), (window,))[0][5]
        assert [list(tile) for tile in expected] == result[6]
        keys = [(tile[0], tile[1]) for tile in result[6]]
        assert keys == sorted(keys)


def test_sources_keep_plan_order_and_updates_may_be_reordered():
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,2,null],'
        '["b1",0,1,[0,1,1]]]]],"complete":false}')
    pyramid_b0 = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    pyramid_b1 = build_tile_pyramid(
        ((20.0, 20.0, 8.0, 0, 1.0),), levels=1)
    # updates in the reverse order of the task's sources
    text = execute_tile_update_plan(
        plan, (("b1", pyramid_b1), ("b0", pyramid_b0)))
    result = json.loads(text)["results"][0]
    assert result[5] == [["b0", 1, 2, None], ["b1", 0, 1, [0, 1, 1]]]
    assert result[6] == [[0, 0, 0, 0, 255, 255, 5.0, 8.0, 2]]


def test_tiles_merge_min_zmax_max_count_sum_and_sort():
    plan = '{"tasks":[[0,0,0,511,511,[["b0",1,1,null],' \
           '["b1",1,1,null]]]],"complete":false}'
    points_b0 = (
        (10.0, 10.0, 5.0, 0, 1.0),     # tile (0,0)
        (300.0, 300.0, 2.0, 0, 1.0),   # tile (1,1)
    )
    points_b1 = (
        (20.0, 20.0, 1.0, 0, 1.0),     # tile (0,0) same coordinate
        (20.0, 20.0, 9.0, 0, 1.0),     # tile (0,0)
        (300.0, 300.0, 6.0, 0, 1.0),   # tile (1,1)
    )
    text = execute_tile_update_plan(plan, (
        ("b0", build_tile_pyramid(points_b0, levels=1)),
        ("b1", build_tile_pyramid(points_b1, levels=1))))
    tiles = json.loads(text)["results"][0][6]
    assert tiles == [
        [0, 0, 0, 0, 255, 255, 1.0, 9.0, 3],
        [1, 1, 256, 256, 511, 511, 2.0, 6.0, 2],
    ]


def test_window_with_no_intersecting_tile_gets_empty_tiles():
    plan = '{"tasks":[[0,512,512,767,767,[["b0",1,1,null]]]],' \
           '"complete":false}'
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=1)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    result = json.loads(text)["results"][0]
    assert result[:5] == [0, 512, 512, 767, 767]
    assert result[6] == []


def test_z_uses_six_decimals_and_negative_zero():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],' \
           '"complete":false}'
    pyramid = build_tile_pyramid(
        ((10.0, 10.0, -1e-9, 0, 1.0),), levels=1)
    text = execute_tile_update_plan(plan, (("b0", pyramid),))
    assert "[0,0,0,0,255,255,0.000000,0.000000,1]" in text
    assert "-0.000000" not in text


def test_completed_plan_with_empty_tasks(scene, paths):
    plan = _completed_plan_state(scene, paths)
    assert plan == '{"tasks":[],"complete":true}'
    text = execute_tile_update_plan(plan, ())
    assert text == (
        '{"plan":{"tasks":[],"complete":true},"confirmed":0,'
        '"results":[],"complete":true}')
    document = json.loads(text)
    assert document["confirmed"] == 0
    assert document["results"] == []
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# max_tasks and re-entry
# ---------------------------------------------------------------------------

def test_max_tasks_limits_newly_confirmed_tasks(scene, paths):
    _state_text, plan = _plan(scene, paths)
    text = execute_tile_update_plan(plan, _updates(), max_tasks=2)
    document = json.loads(text)
    assert document["confirmed"] == 2
    assert document["complete"] is False
    tasks = json.loads(plan)["tasks"]
    assert [r[:5] for r in document["results"]] == [
        task[:5] for task in tasks[:2]]


def test_zero_step_confirms_nothing(scene, paths):
    _state_text, plan = _plan(scene, paths)
    text = execute_tile_update_plan(plan, _updates(), max_tasks=0)
    document = json.loads(text)
    assert document["confirmed"] == 0
    assert document["results"] == []
    assert document["complete"] is False
    assert text == (
        '{"plan":' + plan + ',"confirmed":0,"results":[],"complete":false}')


def test_re_entering_a_state_is_byte_identical(scene, paths):
    _state_text, plan = _plan(scene, paths)
    first = execute_tile_update_plan(plan, _updates(), max_tasks=2)
    again = execute_tile_update_plan(
        plan, _updates(), state=first, max_tasks=0)
    assert again == first
    completed = execute_tile_update_plan(plan, _updates(), state=first)
    assert execute_tile_update_plan(
        plan, _updates(), state=completed) == completed
    assert execute_tile_update_plan(
        plan, _updates(), state=completed, max_tasks=0) == completed


def test_chunked_confirmation_matches_one_shot(scene, paths):
    _state_text, plan = _plan(scene, paths)
    one_shot = execute_tile_update_plan(plan, _updates())
    state = execute_tile_update_plan(plan, _updates(), max_tasks=0)
    task_count = len(json.loads(plan)["tasks"])
    for _ in range(task_count):
        state = execute_tile_update_plan(
            plan, _updates(), state=state, max_tasks=1)
    assert state == one_shot
    assert json.loads(state)["complete"] is True


def test_state_prefix_must_match_recomputed_tiles(scene, paths):
    _state_text, plan = _plan(scene, paths)
    pyramid_a = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (255.0, 255.0, 9.0, 2, 1.0),
         (256.0, 256.0, 7.0, 1, 2.0),
         (-1.0, 300.0, 4.0, 3, 1.5)),)
    pyramid_b = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (255.0, 255.0, 9.0, 2, 1.0),
         (256.0, 256.0, 7.0, 1, 2.0),
         (-1.0, 300.0, 99.0, 3, 1.5)),)
    first = execute_tile_update_plan(
        plan, (("b0", pyramid_a),), max_tasks=1)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan, (("b0", pyramid_b),), state=first, max_tasks=1)


def test_repeated_calls_are_byte_for_byte_equal(scene, paths):
    _state_text, plan = _plan(scene, paths)
    first = execute_tile_update_plan(plan, _updates())
    second = execute_tile_update_plan(plan, _updates())
    assert first == second


# ---------------------------------------------------------------------------
# TypeError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_plan_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(bad, ())


@pytest.mark.parametrize("bad", [[], None, 1, "x", {}])
def test_updates_wrong_type(bad):
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), {},
])
def test_update_shape_wrong_type(bad):
    plan = '{"tasks":[],"complete":true}'
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (bad,))


def test_pyramid_and_tile_container_types():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],' \
           '"complete":false}'
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (("b0", []),))
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (("b0", [[]]),))
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (("b0", ([()],)),))
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            plan, (("b0", (((0, 0, 0, 0, 255, 255, 1, 2.0, 1),),)),))
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            plan, (("b0", (((0, 0, 0, 0, 255, 255, 1.0, 2, 1),),)),))
    with pytest.raises(TypeError):
        execute_tile_update_plan(
            plan, (("b0", (((True, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),)),))


@pytest.mark.parametrize("bad", [1, 1.0, (), [], {}])
def test_state_wrong_type(bad):
    plan = '{"tasks":[],"complete":true}'
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (), state=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(bad):
    plan = '{"tasks":[],"complete":true}'
    with pytest.raises(TypeError):
        execute_tile_update_plan(plan, (), max_tasks=bad)


# ---------------------------------------------------------------------------
# ValueError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"tasks":[],"complete":true} ',
    ' {"tasks":[],"complete":true}',
    '{"complete":true,"tasks":[]}',
    '{"tasks":[],"complete":1}',
    '{"tasks":{},"complete":true}',
    '{"tasks":[[]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,[]]],"complete":false}',
    '{"tasks":[[0,0,0,255,255,'
    '[["b0",1,1,null]],"x":1]],"complete":false}',
])
def test_non_canonical_plan_rejected(tampered):
    with pytest.raises(ValueError):
        execute_tile_update_plan(tampered, ())


def test_max_tasks_negative_rejected():
    plan = '{"tasks":[],"complete":true}'
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (), max_tasks=-1)


def test_update_ids_must_cover_sources_exactly():
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null],'
        '["b1",1,1,null]]]],"complete":false}')
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=1)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", pyramid),))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (
            ("b0", pyramid), ("b1", pyramid), ("b2", pyramid)))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (
            ("b0", pyramid), ("b0", pyramid)))
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            '{"tasks":[],"complete":true}', (("b0", pyramid),))


def test_pyramid_value_errors_propagate():
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],' \
           '"complete":false}'
    nan_pyramid = (
        ((0, 0, 0, 0, 255, 255, float("nan"), 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", nan_pyramid),))
    unsorted = (
        ((1, 0, 256, 0, 511, 255, 1.0, 2.0, 1),
         (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1)),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", unsorted),))
    inverted = (((0, 0, 256, 0, 0, 255, 1.0, 2.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", inverted),))
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (("b0", ()),))


def test_pyramid_level_counts_must_agree_and_cover_tasks():
    pyramid_one = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0),), levels=1)
    pyramid_three = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0),), levels=3)
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null],'
        '["b1",1,1,null]]]],"complete":false}')
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (
            ("b0", pyramid_one), ("b1", pyramid_three)))

    plan_two_levels = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]],'
        '[1,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan_two_levels, (("b0", pyramid_one),))
    # three-level pyramids cover both tasks
    text = execute_tile_update_plan(
        plan_two_levels, (("b0", pyramid_three),))
    assert json.loads(text)["complete"] is True


def test_same_coordinate_tiles_with_unequal_bounds_rejected():
    plan = (
        '{"tasks":[[0,0,0,255,255,[["b0",1,1,null],'
        '["b1",1,1,null]]]],"complete":false}')
    pyramid_b0 = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 0, 1.0),), tile_cells=256, levels=1)
    pyramid_b1 = (((0, 0, 0, 0, 127, 127, 3.0, 4.0, 1),),)
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (
            ("b0", pyramid_b0), ("b1", pyramid_b1)))


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"confirmed":0,"results":[],"complete":false}',
])
def test_non_canonical_state_rejected(scene, paths, tampered):
    _state_text, plan = _plan(scene, paths)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan, _updates(), state=tampered)


def test_state_must_embed_the_same_plan():
    plan_a = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],' \
             '"complete":false}'
    plan_b = '{"tasks":[[0,0,0,511,511,[["b0",1,1,null]]]],' \
             '"complete":false}'
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 0, 1.0),), levels=1)
    state = execute_tile_update_plan(
        plan_a, (("b0", pyramid),), max_tasks=0)
    with pytest.raises(ValueError):
        execute_tile_update_plan(
            plan_b, (("b0", pyramid),), state=state)


def test_state_complete_flag_must_agree_with_prefix():
    plan = '{"tasks":[],"complete":true}'
    bad = ('{"plan":' + plan + ',"confirmed":0,"results":[],'
           '"complete":false}')
    with pytest.raises(ValueError):
        execute_tile_update_plan(plan, (), state=bad)
