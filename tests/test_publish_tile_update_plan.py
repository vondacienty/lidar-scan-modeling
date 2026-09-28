"""Tests for :func:`lidar_scan.publish_tile_update_plan`."""

from __future__ import annotations

import json
import os

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        execute_tile_update_plan, publish_tile_update_plan)
from lidar_scan import tiles as tiles_module
import lidar_scan

_PLAN_TWO = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
    '[0,512,512,767,767,[["b0",1,2,null]]]],"complete":false}')

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


def _write(path, text):
    if isinstance(text, bytes):
        with open(path, "wb") as stream:
            stream.write(text)
    else:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)


def _read_bytes(path):
    with open(path, "rb") as stream:
        return stream.read()


def _read(path):
    return _read_bytes(path).decode("utf-8")


def _raw_pyramid(state_text):
    """Slice the canonical ``pyramid`` array text out of a state document."""
    node = tiles_module._parse_json_node(state_text, 0)
    pyramid_node = node[0]["pyramid"]
    return state_text[pyramid_node[1]:pyramid_node[2]]


@pytest.fixture
def paths(tmp_path):
    return {
        "state": str(tmp_path / "state.json"),
        "pyramid": str(tmp_path / "pyramid.json"),
    }


def _start(paths, base=None):
    """Create P holding the base pyramid array; leave S missing."""
    if base is None:
        base = _base_pyramid()
    array_text = tiles_module.encode_tile_pyramid(base)
    array_text = array_text[len('{"levels":'):-1]
    _write(paths["pyramid"], array_text)
    return base, array_text


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.publish_tile_update_plan is publish_tile_update_plan
    assert "publish_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic publication
# ---------------------------------------------------------------------------

def test_fresh_run_publishes_state_and_pyramid(paths):
    base, _ = _start(paths)
    execution = _execution()
    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution)

    assert result == _read(paths["state"])
    one_shot = commit_tile_update_plan(base, execution)
    assert result == one_shot
    document = json.loads(result)
    assert list(document) == [
        "plan", "base", "committed", "receipts", "pyramid", "complete"]
    assert document["committed"] == 2
    assert document["complete"] is True

    data = _read_bytes(paths["pyramid"])
    assert not data.startswith(b"\xef\xbb\xbf")
    assert not data.endswith(b"\n")
    assert b" " not in data and b"\t" not in data and b"\n" not in data
    assert json.loads(data) == document["pyramid"]


def test_pyramid_file_is_the_canonical_nine_tile_array(paths):
    base, _ = _start(paths)
    publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                             _execution())
    data = _read_bytes(paths["pyramid"]).decode("utf-8")
    assert data.startswith("[[[")
    for tile in json.loads(data)[0]:
        assert len(tile) == 9
    import re
    for raw in re.findall(r"-?\d+\.\d+", data):
        assert re.fullmatch(r"-?\d+\.\d{6}", raw)


def test_max_tasks_chunks_match_the_one_shot_commit(paths):
    base, _ = _start(paths)
    execution = _execution()
    first = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=1)
    assert json.loads(first)["committed"] == 1
    assert json.loads(_read(paths["pyramid"])) == \
        json.loads(first)["pyramid"]
    second = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=1)
    assert json.loads(second)["committed"] == 2
    assert second == commit_tile_update_plan(base, execution)
    assert json.loads(_read(paths["pyramid"])) == \
        json.loads(second)["pyramid"]


def test_completed_reentry_is_byte_identical_and_does_not_touch_p(paths):
    base, _ = _start(paths)
    execution = _execution()
    first = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution)
    pyramid_bytes = _read_bytes(paths["pyramid"])
    state_mtime = os.stat(paths["state"]).st_mtime_ns
    pyramid_mtime = os.stat(paths["pyramid"]).st_mtime_ns
    again = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution)
    assert again == first
    assert publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=0) \
        == first
    assert os.stat(paths["state"]).st_mtime_ns == state_mtime
    assert os.stat(paths["pyramid"]).st_mtime_ns == pyramid_mtime
    assert _read_bytes(paths["pyramid"]) == pyramid_bytes


def test_fresh_zero_step_registers_state_without_touching_pyramid(paths):
    base, base_array = _start(paths)
    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, _execution(), max_tasks=0)
    document = json.loads(result)
    assert document["committed"] == 0
    assert document["complete"] is False
    assert document["pyramid"] == json.loads(base_array)
    assert _read(paths["pyramid"]) == base_array


def test_empty_plan_completes_with_state_ahead_of_base_pyramid(paths):
    base, base_array = _start(paths)
    execution = execute_tile_update_plan(_PLAN_EMPTY, ())
    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution)
    document = json.loads(result)
    assert document["committed"] == 0
    assert document["complete"] is True
    assert _read(paths["pyramid"]) == base_array
    assert publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution) == result


def test_state_is_written_before_the_pyramid_for_each_step(
        paths, monkeypatch):
    base, _ = _start(paths)
    execution = _execution()
    real_replace = os.replace
    replaced = []

    def spy(src, dst):
        replaced.append(dst)
        return real_replace(src, dst)

    monkeypatch.setattr(tiles_module.os, "replace", spy)
    publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=1)
    # Fresh publication: the zero state, then state step 1, then P step 1;
    # each advancement's state replacement precedes its pyramid one.
    assert replaced[0] == paths["state"]
    assert replaced[1] == paths["state"]
    assert replaced[2] == paths["pyramid"]


# ---------------------------------------------------------------------------
# crash recovery: S may lead P
# ---------------------------------------------------------------------------

def test_pyramid_lagging_state_is_backfilled_without_budget(paths):
    base, _ = _start(paths)
    execution = _execution()
    state_two = commit_tile_update_plan(base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths["state"], state_two)
    _write(paths["pyramid"], _raw_pyramid(state_one))

    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=0)
    assert result == state_two
    assert json.loads(_read(paths["pyramid"])) == \
        json.loads(state_two)["pyramid"]


def test_backfill_then_remaining_budget_commits_new_tasks(paths):
    base, _ = _start(paths)
    execution = _execution()
    state_two = commit_tile_update_plan(base, execution)
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(paths["state"], state_two)
    _write(paths["pyramid"], _raw_pyramid(state_one))
    # Execution only confirms two tasks, so backfill reaches the end; the
    # zero extra budget changes nothing beyond the backfill itself.
    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=0)
    assert json.loads(result)["committed"] == 2


def test_crash_between_state_and_pyramid_is_recovered(paths, monkeypatch):
    base, _ = _start(paths)
    execution = _execution()
    publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=1)
    pyramid_after_one = _read_bytes(paths["pyramid"])

    real_replace = os.replace

    def fail_pyramid(src, dst):
        if dst == paths["pyramid"]:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(tiles_module.os, "replace", fail_pyramid)
    with pytest.raises(OSError):
        publish_tile_update_plan(
            paths["state"], paths["pyramid"], base, execution, max_tasks=1)
    monkeypatch.undo()

    # S advanced to committed 2; P stayed at the first prefix.
    assert json.loads(_read(paths["state"]))["committed"] == 2
    assert _read_bytes(paths["pyramid"]) == pyramid_after_one
    # No temporary file left behind.
    assert not [name for name in os.listdir(os.path.dirname(paths["state"]))
                if name.startswith(".publish-tile-")]

    result = publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution, max_tasks=0)
    assert json.loads(result)["complete"] is True
    assert json.loads(_read(paths["pyramid"])) == \
        json.loads(result)["pyramid"]


def test_state_write_failure_leaves_both_files_unstarted(paths, monkeypatch):
    base, base_array = _start(paths)

    def fail_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(tiles_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError):
        publish_tile_update_plan(
            paths["state"], paths["pyramid"], base, _execution())
    monkeypatch.undo()

    assert not os.path.exists(paths["state"])
    assert _read(paths["pyramid"]) == base_array
    assert not [name for name in os.listdir(os.path.dirname(paths["state"]))
                if name.startswith(".publish-tile-")]


# ---------------------------------------------------------------------------
# TypeError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_path_wrong_type(paths, bad):
    base, _ = _start(paths)
    with pytest.raises(TypeError):
        publish_tile_update_plan(bad, paths["pyramid"], base, _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_pyramid_path_wrong_type(paths, bad):
    base, _ = _start(paths)
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], bad, base, _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(paths, bad):
    _start(paths)
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], bad,
                                 _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_execution_wrong_type(paths, bad):
    base, _ = _start(paths)
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base, bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(paths, bad):
    base, _ = _start(paths)
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution(), max_tasks=bad)


# ---------------------------------------------------------------------------
# ValueError validation
# ---------------------------------------------------------------------------

def test_empty_and_equal_paths_rejected(paths):
    base, _ = _start(paths)
    with pytest.raises(ValueError):
        publish_tile_update_plan("", paths["pyramid"], base, _execution())
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], "", base, _execution())
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["state"], base,
                                 _execution())


def test_negative_max_tasks_rejected(paths):
    base, _ = _start(paths)
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution(), max_tasks=-1)


def test_internal_base_structure_errors_are_value_errors(paths):
    _start(paths)
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            paths["state"], paths["pyramid"],
            (((0, 0, 0, 0, 255, 255, 1.0, 2, 1),),), _execution())


@pytest.mark.parametrize("bad", [b"\xff\xfe not json",
                                 b"\xef\xbb\xbf[]",
                                 b"[] ",
                                 b"{}",
                                 b"null",
                                 b"[[]] "[:-1] + b",",
                                 b"[[[0,0,0,0,255,255,1,2.0,1]]]",
                                 b"[[[0,0,0,0,255,255,1.0,2.0,1.0]]]"])
def test_malformed_pyramid_file_rejected(paths, bad):
    _write(paths["pyramid"], bad)
    base = _base_pyramid()
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution())


def test_pyramid_not_utf8_rejected(paths):
    base, _ = _start(paths)
    _write(paths["pyramid"], b"[\xff]")
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution())


def test_pyramid_file_ahead_of_state_rejected(paths):
    base, _ = _start(paths)
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(base, execution)
    _write(paths["state"], state_one)
    _write(paths["pyramid"], _raw_pyramid(state_two))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 execution)


def test_pyramid_file_unrelated_to_the_prefix_chain_rejected(paths):
    base, _ = _start(paths)
    execution = _execution()
    # A valid canonical pyramid, but one never produced by this run.
    other = build_tile_pyramid(
        ((12345.0, 12345.0, 8.0, 1, 1.0),), levels=1)
    _write(paths["pyramid"],
           tiles_module.encode_tile_pyramid(other)[len('{"levels":'):-1])
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 execution)


def test_state_bound_to_another_base_rejected(paths):
    base, _ = _start(paths)
    execution = _execution()
    state = commit_tile_update_plan(base, execution)
    _write(paths["state"], state)
    other_base = build_tile_pyramid(
        ((999.0, 999.0, 1.0, 1, 1.0),), levels=1)
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 other_base, execution)


def test_state_bound_to_another_execution_rejected(paths):
    base, _ = _start(paths)
    _write(paths["state"], commit_tile_update_plan(base, _execution()))
    other_plan = (
        '{"tasks":[[0,0,0,511,511,[["b0",1,2,null]]]],"complete":false}')
    other_execution = execute_tile_update_plan(
        other_plan, (("b0", _new_pyramid()),))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 other_execution)


def test_non_canonical_state_rejected(paths):
    base, _ = _start(paths)
    execution = _execution()
    state = commit_tile_update_plan(base, execution)
    _write(paths["state"], state + "\n")
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 execution)


def test_state_not_utf8_rejected(paths):
    base, _ = _start(paths)
    _write(paths["state"], b"\xff")
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution())


# ---------------------------------------------------------------------------
# OSError validation
# ---------------------------------------------------------------------------

def test_missing_pyramid_file_raises_oserror(paths):
    base = _base_pyramid()
    with pytest.raises(OSError):
        publish_tile_update_plan(paths["state"], paths["pyramid"], base,
                                 _execution())


def test_pyramid_directory_missing_raises_oserror(paths):
    base = _base_pyramid()
    missing_pyramid = os.path.join(paths["pyramid"], "deep", "gone.json")
    with pytest.raises(OSError):
        publish_tile_update_plan(paths["state"], missing_pyramid, base,
                                 _execution())
