"""Tests for :func:`lidar_scan.publish_tile_update_plan`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        execute_tile_update_plan, publish_tile_update_plan)
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


def _state_pyramid_text(state_text: str) -> str:
    node = tiles_module._parse_json_node(
        state_text, tiles_module._skip_json_ws(state_text, 0))
    start, end = node[0]["pyramid"][1], node[0]["pyramid"][2]
    return state_text[start:end]


def _state_base_text(state_text: str) -> str:
    node = tiles_module._parse_json_node(
        state_text, tiles_module._skip_json_ws(state_text, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    return state_text[start:end]


def _write(path, text):
    with open(path, "wb") as stream:
        stream.write(text.encode("utf-8"))


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.publish_tile_update_plan
            is publish_tile_update_plan)
    assert "publish_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# publication mechanics
# ---------------------------------------------------------------------------

def test_fresh_publication_registers_state_and_publishes_one(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=1)

    state_bytes = _read(state_path)
    pyramid_bytes = _read(pyramid_path)
    assert state_bytes.decode("utf-8") == result
    assert pyramid_bytes.decode("utf-8") == _state_pyramid_text(result)
    document = json.loads(result)
    assert list(document) == [
        "plan", "base", "committed", "receipts", "pyramid", "complete"]
    assert document["committed"] == 1
    assert document["complete"] is False
    # Canonical UTF-8 bytes: no BOM, whitespace or trailing newline.
    assert not state_bytes.startswith(b"\xef\xbb\xbf")
    assert not pyramid_bytes.startswith(b"\xef\xbb\xbf")
    assert not state_bytes.endswith(b"\n")
    assert not pyramid_bytes.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in pyramid_bytes)


def test_one_shot_publication_completes(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution)

    document = json.loads(result)
    assert document["committed"] == 2
    assert document["complete"] is True
    one_shot = commit_tile_update_plan(base, execution)
    assert result == one_shot
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(one_shot)


def test_chunked_publication_matches_one_shot(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))

    first = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=1)
    assert json.loads(first)["committed"] == 1
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(first)
    second = publish_tile_update_plan(
        state_path, pyramid_path, base, execution)
    one_shot = commit_tile_update_plan(base, execution)
    assert second == one_shot
    assert _read(state_path).decode("utf-8") == one_shot
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(one_shot)


def test_zero_step_on_fresh_state_registers_zero_committed_state(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    base_text = _state_base_text(zero_state)
    _write(pyramid_path, base_text)

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=0)
    assert result == zero_state
    assert _read(state_path).decode("utf-8") == zero_state
    # P is not touched.
    assert _read(pyramid_path).decode("utf-8") == base_text

    # Re-entering the zero-step state is byte-for-byte idempotent.
    again = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=0)
    assert again == zero_state
    assert _read(state_path).decode("utf-8") == zero_state
    assert _read(pyramid_path).decode("utf-8") == base_text


def test_re_entering_completed_publication_is_byte_identical(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))
    completed = publish_tile_update_plan(
        state_path, pyramid_path, base, execution)

    state_before = _read(state_path)
    pyramid_before = _read(pyramid_path)
    again = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=0)
    assert again == completed
    assert _read(state_path) == state_before
    assert _read(pyramid_path) == pyramid_before
    again = publish_tile_update_plan(
        state_path, pyramid_path, base, execution)
    assert again == completed
    assert _read(state_path) == state_before
    assert _read(pyramid_path) == pyramid_before


def test_lagging_pyramid_is_caught_up_without_spending_budget(tmp_path):
    # S has committed one result but P still holds the base: a zero-budget
    # re-entry backfills P to S and adds nothing.
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    base_text = _state_base_text(state_one)
    _write(state_path, state_one)
    _write(pyramid_path, base_text)

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=0)
    assert result == state_one
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(state_one)
    assert _read(state_path).decode("utf-8") == state_one


def test_catch_up_then_advance_within_one_call(tmp_path):
    # S at committed 1, P lagging at base; the call first catches P up to
    # committed 1 for free and then spends its budget of one to reach 2.
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(state_path, state_one)
    _write(pyramid_path, _state_base_text(state_one))

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=1)
    assert result == state_two
    assert _read(state_path).decode("utf-8") == state_two
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(state_two)


def test_intermediate_prefix_pyramid_is_accepted(tmp_path):
    # P holding an earlier committed prefix than S (not merely base) is a
    # legal lag and gets backfilled.
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(state_path, state_two)
    _write(pyramid_path, _state_pyramid_text(state_one))

    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution, max_tasks=0)
    assert result == state_two
    assert _read(pyramid_path).decode("utf-8") == \
        _state_pyramid_text(state_two)


def test_returned_text_is_the_state_file_text(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))
    result = publish_tile_update_plan(
        state_path, pyramid_path, base, execution)
    assert result == _read(state_path).decode("utf-8")


# ---------------------------------------------------------------------------
# TypeError / ValueError argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_path_wrong_type(bad, tmp_path):
    with pytest.raises(TypeError):
        publish_tile_update_plan(bad, str(tmp_path / "p"),
                                 _base_pyramid(), _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_pyramid_path_wrong_type(bad, tmp_path):
    with pytest.raises(TypeError):
        publish_tile_update_plan(str(tmp_path / "s"), bad,
                                 _base_pyramid(), _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(bad, tmp_path):
    p = str(tmp_path / "p")
    _write(p, "[]")
    with pytest.raises(TypeError):
        publish_tile_update_plan(str(tmp_path / "s"), p, bad, _execution())


def test_base_internal_structure_is_value_error(tmp_path):
    p = str(tmp_path / "p")
    _write(p, "[]")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "s"), p, (), _execution())
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "s"), p, ([()],), _execution())


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_execution_wrong_type(bad, tmp_path):
    p = str(tmp_path / "p")
    _write(p, "[]")
    with pytest.raises(TypeError):
        publish_tile_update_plan(
            str(tmp_path / "s"), p, _base_pyramid(), bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(bad, tmp_path):
    p = str(tmp_path / "p")
    _write(p, "[]")
    with pytest.raises(TypeError):
        publish_tile_update_plan(
            str(tmp_path / "s"), p, _base_pyramid(), _execution(),
            max_tasks=bad)


def test_empty_and_equal_paths_are_value_errors(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    p = str(tmp_path / "p")
    _write(p, "[]")
    with pytest.raises(ValueError):
        publish_tile_update_plan("", p, base, execution)
    with pytest.raises(ValueError):
        publish_tile_update_plan(str(tmp_path / "s"), "", base, execution)
    with pytest.raises(ValueError):
        publish_tile_update_plan(p, p, base, execution)


def test_negative_max_tasks_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_path = str(tmp_path / "state.json")
    pyramid_path = str(tmp_path / "pyramid.json")
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_base_text(zero_state))
    with pytest.raises(ValueError):
        publish_tile_update_plan(state_path, pyramid_path, base, execution,
                                 max_tasks=-1)


# ---------------------------------------------------------------------------
# file content validation
# ---------------------------------------------------------------------------

def test_missing_pyramid_file_is_oserror(tmp_path):
    with pytest.raises(OSError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "missing.json"),
            _base_pyramid(), _execution())


def test_pyramid_file_not_utf8_is_value_error(tmp_path):
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_bytes(b"\xff\xfe[")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            _base_pyramid(), _execution())


@pytest.mark.parametrize("bad", [
    "",
    "{not json",
    "null",
    "{}",
    '{"levels":[]}',
    "[[]] ",
    "[]\n",
    " [[]]",
    "[ [ ] ]",
])
def test_malformed_pyramid_file_is_value_error(bad, tmp_path):
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_bytes(bad.encode("utf-8"))
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            _base_pyramid(), _execution())


def test_pyramid_file_with_bom_is_value_error(tmp_path):
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_bytes(b"\xef\xbb\xbf[]")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            _base_pyramid(), _execution())


def test_pyramid_file_wrong_number_format_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    base_text = _state_base_text(zero_state)
    tampered = base_text.replace("1.000000", "1.0", 1)
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            base, execution)


def test_fresh_state_requires_pyramid_to_equal_base(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_text(_state_pyramid_text(state_one), encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            base, execution)


def test_pyramid_bound_to_another_base_rejected(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(tmp_path / "state.json", state_one)
    other_base = build_tile_pyramid(
        ((999.0, 999.0, 1.0, 1, 1.0),
         (600.0, 600.0, 2.0, 1, 1.0)), levels=1)
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_text(_state_base_text(state_one), encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            other_base, execution)


def test_pyramid_ahead_of_state_rejected(tmp_path):
    # S is at committed 1 but P already holds the committed-2 pyramid: P
    # must never be ahead of S.
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    state_two = commit_tile_update_plan(
        base, execution, state=state_one, max_tasks=1)
    _write(tmp_path / "state.json", state_one)
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_text(_state_pyramid_text(state_two), encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            base, execution, max_tasks=0)


def test_unrelated_pyramid_rejected(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(tmp_path / "state.json", state_one)
    unrelated = _format_array(build_tile_pyramid(
        ((30000.0, 30000.0, 8.0, 1, 1.0),), levels=1))
    pyramid_path = tmp_path / "pyramid.json"
    pyramid_path.write_text(unrelated, encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"), str(pyramid_path),
            base, execution)


def test_state_file_not_utf8_is_value_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(tmp_path / "pyramid.json", _state_base_text(zero_state))
    (tmp_path / "state.json").write_bytes(b"\xff\xfe{")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), base, execution)


def test_state_bound_to_another_base_rejected(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(tmp_path / "state.json", state_one)
    _write(tmp_path / "pyramid.json", _state_pyramid_text(state_one))
    other_base = build_tile_pyramid(
        ((999.0, 999.0, 1.0, 1, 1.0),
         (600.0, 600.0, 2.0, 1, 1.0)), levels=1)
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), other_base, execution)


def test_state_bound_to_another_execution_rejected(tmp_path):
    base = _base_pyramid()
    state_one = commit_tile_update_plan(base, _execution(), max_tasks=1)
    _write(tmp_path / "state.json", state_one)
    _write(tmp_path / "pyramid.json", _state_pyramid_text(state_one))
    other_plan = (
        '{"tasks":[[0,0,0,511,511,[["b0",1,1,null]]]],"complete":false}')
    other_execution = _execution(other_plan)
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), base, other_execution)


@pytest.mark.parametrize("tampered", [
    "{not json",
    "",
    "null",
    "[]",
    '{"plan":{},"base":[],"committed":0,"receipts":[],"pyramid":[],'
    '"complete":false}',
])
def test_malformed_state_rejected(tampered, tmp_path):
    base = _base_pyramid()
    execution = _execution()
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(tmp_path / "pyramid.json", _state_base_text(zero_state))
    (tmp_path / "state.json").write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), base, execution)


def test_tampered_state_rejected(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    state_one = commit_tile_update_plan(base, execution, max_tasks=1)
    _write(tmp_path / "pyramid.json", _state_pyramid_text(state_one))
    tampered = json.loads(state_one)
    tampered["pyramid"][0][0][6] = 99.0
    _write(tmp_path / "state.json",
           json.dumps(tampered, separators=(",", ":")))
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), base, execution)


def test_state_committing_more_than_execution_confirms_rejected(tmp_path):
    base = build_tile_pyramid(((10.0, 10.0, 1.0, 1, 1.0),), levels=1)
    pyramid = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    plan = '{"tasks":[[0,0,0,255,255,[["b0",1,1,null]]]],"complete":false}'
    full = execute_tile_update_plan(plan, (("b0", pyramid),))
    state = commit_tile_update_plan(base, full)
    unconfirmed = execute_tile_update_plan(
        plan, (("b0", pyramid),), max_tasks=0)
    _write(tmp_path / "state.json", state)
    _write(tmp_path / "pyramid.json", _state_pyramid_text(state))
    with pytest.raises(ValueError):
        publish_tile_update_plan(
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"), base, unconfirmed)


def _format_array(pyramid) -> str:
    return tiles_module._format_tile_update_pyramid_array(pyramid)
