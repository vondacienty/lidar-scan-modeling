"""Tests for :func:`lidar_scan.coordinate_tile_update`."""

from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import os

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        coordinate_tile_update,
                        execute_tile_update_plan)
from lidar_scan import tiles as tiles_module
import lidar_scan

_PLAN_TWO = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
    '[0,512,512,767,767,[["b0",1,2,null]]]],"complete":false}')

_PLAN_THREE = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,3,null]]],'
    '[0,512,512,767,767,[["b0",1,3,null]]],'
    '[0,1024,1024,1279,1279,[["b0",1,3,null]]]],"complete":false}')


def _new_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (600.0, 600.0, 9.0, 1, 1.0),
         (1100.0, 1100.0, 4.0, 1, 1.0)), levels=1)


def _base_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 1.0, 1, 1.0),
         (600.0, 600.0, 2.0, 1, 1.0),
         (1100.0, 1100.0, 3.0, 1, 1.0)), levels=1)


def _execution(plan=_PLAN_TWO, pyramid=None, **kwargs):
    if pyramid is None:
        pyramid = _new_pyramid()
    sources = sorted({source[0]
                      for task in json.loads(plan)["tasks"]
                      for source in task[5]})
    updates = tuple((source_id, pyramid) for source_id in sources)
    return execute_tile_update_plan(plan, updates, **kwargs)


def _state_node_text(state_text: str, key: str) -> str:
    node = tiles_module._parse_json_node(
        state_text, tiles_module._skip_json_ws(state_text, 0))
    start, end = node[0][key][1], node[0][key][2]
    return state_text[start:end]


def _write(path, text):
    with open(path, "wb") as stream:
        stream.write(text.encode("utf-8"))


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


def _paths(tmp_path):
    return (str(tmp_path / "lock"), str(tmp_path / "records.json"),
            str(tmp_path / "state.json"), str(tmp_path / "pyramid.json"))


def _seed_pyramid(paths, base, execution):
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(paths[3], _state_node_text(zero_state, "base"))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.coordinate_tile_update is coordinate_tile_update
    assert "coordinate_tile_update" in lidar_scan.__all__


def test_atomic_write_helper_renamed():
    assert hasattr(tiles_module, "_atomic_write_text")
    assert not hasattr(tiles_module, "_atomic_write_json")


# ---------------------------------------------------------------------------
# publication mechanics
# ---------------------------------------------------------------------------

def test_fresh_coordination_registers_state_pyramid_and_records(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)

    result = coordinate_tile_update(paths, base, execution, limit=1)

    state_text = _read(paths[2]).decode("utf-8")
    pyramid_text = _read(paths[3]).decode("utf-8")
    records_text = _read(paths[1]).decode("utf-8")
    assert state_text == result
    assert json.loads(result)["committed"] == 1
    assert pyramid_text == _state_node_text(result, "pyramid")
    records_doc = json.loads(records_text)
    assert list(records_doc) == ["records"]
    assert records_doc["records"] == [[0, json.loads(result)]]
    # Canonical compact UTF-8, no BOM/whitespace/newline.
    records_bytes = _read(paths[1])
    assert not records_bytes.startswith(b"\xef\xbb\xbf")
    assert not records_bytes.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in records_bytes)


def test_one_shot_completion_appends_one_record(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)

    result = coordinate_tile_update(paths, base, execution)

    assert json.loads(result)["complete"] is True
    assert result == commit_tile_update_plan(base, execution)
    records_doc = json.loads(_read(paths[1]).decode("utf-8"))
    assert [row[0] for row in records_doc["records"]] == [0]
    assert records_doc["records"][0][1] == json.loads(result)
    assert _read(paths[3]).decode("utf-8") == \
        _state_node_text(result, "pyramid")


def test_chunked_coordination_appends_numbered_records(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)

    first = coordinate_tile_update(paths, base, execution, limit=1)
    assert json.loads(first)["committed"] == 1
    second = coordinate_tile_update(paths, base, execution, limit=1)
    assert json.loads(second)["committed"] == 2
    third = coordinate_tile_update(paths, base, execution)
    assert json.loads(third)["committed"] == 3
    assert json.loads(third)["complete"] is True

    records_doc = json.loads(_read(paths[1]).decode("utf-8"))
    assert [row[0] for row in records_doc["records"]] == [0, 1, 2]
    assert [row[1] for row in records_doc["records"]] == [
        json.loads(first), json.loads(second), json.loads(third)]


def test_zero_step_fresh_call_materializes_empty_records(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _seed_pyramid(paths, base, execution)

    result = coordinate_tile_update(paths, base, execution, limit=0)

    assert result == zero_state
    assert _read(paths[2]).decode("utf-8") == zero_state
    assert _read(paths[1]).decode("utf-8") == '{"records":[]}'
    assert _read(paths[3]).decode("utf-8") == \
        _state_node_text(zero_state, "base")


def test_re_entering_completed_coordination_is_byte_identical(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    completed = coordinate_tile_update(paths, base, execution)

    before = {path: _read(path) for path in paths[1:]}
    again = coordinate_tile_update(paths, base, execution, limit=0)
    assert again == completed
    for path in paths[1:]:
        assert _read(path) == before[path]
    again = coordinate_tile_update(paths, base, execution)
    assert again == completed
    for path in paths[1:]:
        assert _read(path) == before[path]


def test_records_replay_through_commit_states(tmp_path):
    # Each recorded state is exactly what commit_tile_update_plan
    # reproduces from the previous record with the step budget.
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    records_text = _read(paths[1]).decode("utf-8")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    state0 = records_text  # decoded below
    node = tiles_module._parse_json_node(
        state0, tiles_module._skip_json_ws(state0, 0))
    state_node = node[0]["records"][0][0][0][1]
    recorded = state0[state_node[1]:state_node[2]]
    assert commit_tile_update_plan(
        base, execution, state=zero_state, max_tasks=1) == recorded


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------

def test_records_crash_behind_state_is_backfilled(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = coordinate_tile_update(paths, base, execution, limit=1)
    # Crash between the S update and the A update: A is left empty.
    _write(paths[1], '{"records":[]}')

    result = coordinate_tile_update(paths, base, execution, limit=0)

    assert result == state_one
    records_doc = json.loads(_read(paths[1]).decode("utf-8"))
    assert records_doc["records"] == [[0, json.loads(state_one)]]
    assert _read(paths[3]).decode("utf-8") == \
        _state_node_text(state_one, "pyramid")


def test_backfill_then_advance_within_one_call(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = coordinate_tile_update(paths, base, execution, limit=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(paths[1], '{"records":[]}')

    result = coordinate_tile_update(paths, base, execution, limit=1)

    assert result == state_two
    records_doc = json.loads(_read(paths[1]).decode("utf-8"))
    assert [row[0] for row in records_doc["records"]] == [0, 1]
    assert [row[1] for row in records_doc["records"]] == [
        json.loads(state_one), json.loads(state_two)]


def test_lagging_pyramid_is_caught_up(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = coordinate_tile_update(paths, base, execution, limit=1)
    # Crash between replacing S and replacing P: P lags at the base.
    _write(paths[3], _state_node_text(
        commit_tile_update_plan(base, execution, max_tasks=0), "base"))

    result = coordinate_tile_update(paths, base, execution, limit=0)
    assert result == state_one
    assert _read(paths[3]).decode("utf-8") == \
        _state_node_text(state_one, "pyramid")


def test_missing_state_with_empty_records_is_fresh_publication(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    _write(paths[1], '{"records":[]}')

    result = coordinate_tile_update(paths, base, execution, limit=1)
    assert json.loads(result)["committed"] == 1
    records_doc = json.loads(_read(paths[1]).decode("utf-8"))
    assert [row[0] for row in records_doc["records"]] == [0]


# ---------------------------------------------------------------------------
# verify mode
# ---------------------------------------------------------------------------

def test_verify_replays_chain_and_returns_tuple(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    coordinate_tile_update(paths, base, execution, limit=1)

    before = {path: _read(path) for path in paths[1:]}
    result = coordinate_tile_update(paths, base, execution, verify=True)
    assert result == (2, False)
    for path in paths[1:]:
        assert _read(path) == before[path]

    coordinate_tile_update(paths, base, execution)
    assert coordinate_tile_update(paths, base, execution, verify=True) \
        == (3, True)


def test_verify_empty_chain(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(paths[3], _state_node_text(zero_state, "base"))
    # Neither A nor S exists: the zero-step state is the chain head.
    assert coordinate_tile_update(paths, base, execution, verify=True) \
        == (0, False)


def test_verify_requires_limit_none(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0, verify=True)


def test_verify_rejects_state_ahead_of_records(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = coordinate_tile_update(paths, base, execution, limit=1)
    # A crashes behind S; verify must not repair the chain.
    _write(paths[1], '{"records":[]}')
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)
    # Files untouched.
    assert _read(paths[1]).decode("utf-8") == '{"records":[]}'
    assert _read(paths[2]).decode("utf-8") == state_one


def test_verify_rejects_lagging_pyramid(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    _write(paths[3], _state_node_text(
        commit_tile_update_plan(base, execution, max_tasks=0), "base"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


# ---------------------------------------------------------------------------
# cross-process lock
# ---------------------------------------------------------------------------

def _hold_lock(path, ready, proceed):
    stream = open(path, "a+b")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    ready.set()
    proceed.wait(timeout=10)
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def test_contended_lock_raises_blocking_io_error(tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    ctx = mp.get_context("fork")
    ready, proceed = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_lock, args=(paths[0], ready, proceed))
    child.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(BlockingIOError):
            coordinate_tile_update(paths, _base_pyramid(), _execution())
        # Data files are untouched by the contended call.
        assert not os.path.exists(paths[1])
        assert not os.path.exists(paths[2])
    finally:
        proceed.set()
        child.join(timeout=10)
        assert child.exitcode == 0


# ---------------------------------------------------------------------------
# TypeError / ValueError argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], [()], set(), "x",
                                 (1, 2, 3), (1, 2, 3, 4),
                                 ("a", "b", "c")])
def test_paths_wrong_shape(bad, tmp_path):
    with pytest.raises(TypeError):
        coordinate_tile_update(bad, _base_pyramid(), _execution())


@pytest.mark.parametrize("member", [None, 1, b"x", (), []])
def test_path_member_wrong_type(member, tmp_path):
    paths = [str(tmp_path / p) for p in ("l", "a", "s", "p")]
    paths[0] = member
    with pytest.raises(TypeError):
        coordinate_tile_update(tuple(paths), _base_pyramid(), _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(bad, tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update(paths, bad, _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_execution_wrong_type(bad, tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update(paths, _base_pyramid(), bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_limit_wrong_type(bad, tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update(paths, _base_pyramid(), _execution(),
                               limit=bad)


@pytest.mark.parametrize("bad", [None, 1, "true", (), []])
def test_verify_wrong_type(bad, tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update(paths, _base_pyramid(), _execution(),
                               verify=bad)


def test_empty_and_equal_paths_are_value_errors(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    good = _paths(tmp_path)
    _seed_pyramid(good, base, execution)
    with pytest.raises(ValueError):
        coordinate_tile_update(("", good[1], good[2], good[3]),
                               base, execution)
    with pytest.raises(ValueError):
        coordinate_tile_update((good[0], "", good[2], good[3]),
                               base, execution)
    duplicated = (good[0], good[1], good[2], good[2])
    with pytest.raises(ValueError):
        coordinate_tile_update(duplicated, base, execution)


def test_negative_limit_is_value_error(tmp_path):
    paths = _paths(tmp_path)
    _seed_pyramid(paths, _base_pyramid(), _execution())
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, _base_pyramid(), _execution(),
                               limit=-1)


# ---------------------------------------------------------------------------
# file and chain validation
# ---------------------------------------------------------------------------

def test_missing_pyramid_is_oserror(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(OSError):
        coordinate_tile_update(paths, _base_pyramid(), _execution())


def test_pyramid_not_utf8_is_value_error(tmp_path):
    paths = _paths(tmp_path)
    with open(paths[3], "wb") as stream:
        stream.write(b"\xff\xfe[")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, _base_pyramid(), _execution())


def test_records_not_utf8_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(paths[2], zero_state)
    with open(paths[1], "wb") as stream:
        stream.write(b"\xff\xfe{}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


@pytest.mark.parametrize("bad", [
    "",
    "{not json",
    "null",
    "[]",
    "{}",
    '{"records":1}',
    '{"records":null}',
    '{"records":[1]}',
    '{"records":[[0]]}',
    '{"records":[["0",{}]]}',
    '{"records":[[0,{}]]}',
    '{"records":[[1,{}]]}',
    '{"records":[[0,null]]}',
    '{"other":[]}',
    '{"records":[],"x":1}',
    ' {"records":[]}',
    '{"records":[]} ',
    '{"records":[]}\n',
    '{"records": []}',
])
def test_malformed_records_document_is_value_error(bad, tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(paths[2], zero_state)
    with open(paths[1], "wb") as stream:
        stream.write(bad.encode("utf-8"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


def test_records_document_with_bom_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(paths[2], zero_state)
    with open(paths[1], "wb") as stream:
        stream.write(b"\xef\xbb\xbf" + b'{"records":[]}')
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


def test_missing_records_requires_zero_step_state(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths[2], state_one)
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


def test_records_without_state_is_broken_chain_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths[1], '{"records":[[0,' + state_one + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


def test_record_numbers_must_start_at_zero(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(paths[2], state_two)
    _write(paths[3], _state_node_text(state_two, "pyramid"))
    _write(paths[1], '{"records":[[-1,' + state_one + "],[1," + state_two
                     + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_tampered_record_state_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(paths[2], state_two)
    _write(paths[3], _state_node_text(state_two, "pyramid"))
    tampered = json.loads(state_one)
    tampered["committed"] = 1
    tampered["pyramid"][0][0][6] = 42.0
    bad_one = json.dumps(tampered, separators=(",", ":"))
    _write(paths[1], '{"records":[[0,' + bad_one + "],[1," + state_two
                     + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_record_bound_to_another_execution_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths[2], state_one)
    _write(paths[3], _state_node_text(state_one, "pyramid"))
    other_plan = (
        '{"tasks":[[0,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')
    other_execution = _execution(other_plan)
    other_state = commit_tile_update_plan(base, other_execution, max_tasks=1)
    _write(paths[1], '{"records":[[0,' + other_state + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_state_lagging_records_is_broken_chain(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    # A records two states but S only holds the first.
    _write(paths[1], '{"records":[[0,' + state_one + "],[1," + state_two
                     + "]]}")
    _write(paths[2], state_one)
    _write(paths[3], _state_node_text(state_two, "pyramid"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_pyramid_ahead_of_state_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(paths[1], '{"records":[[0,' + state_one + "]]}")
    _write(paths[2], state_one)
    _write(paths[3], _state_node_text(state_two, "pyramid"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)


def test_unrelated_pyramid_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    paths = _paths(tmp_path)
    _seed_pyramid(paths, base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths[1], '{"records":[[0,' + state_one + "]]}")
    _write(paths[2], state_one)
    unrelated = tiles_module._format_tile_update_pyramid_array(
        build_tile_pyramid(((30000.0, 30000.0, 8.0, 1, 1.0),),
                           levels=1))
    _write(paths[3], unrelated)
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, limit=0)
