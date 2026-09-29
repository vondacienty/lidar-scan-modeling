"""Tests for :func:`lidar_scan.coordinate_tile_update`."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        coordinate_tile_update, execute_tile_update_plan)
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


def _state_node_text(state_text, key):
    node = tiles_module._parse_json_node(
        state_text, tiles_module._skip_json_ws(state_text, 0))
    start, end = node[0][key][1], node[0][key][2]
    return state_text[start:end]


def _journal_records_raw(text):
    """Return the embedded raw state texts of a journal document."""
    node = tiles_module._parse_json_node(text, 0)
    return [(record[0][0][0], text[record[0][1][1]:record[0][1][2]])
            for record in node[0]["records"][0]]


def _write(path, text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(text)


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


def _paths(tmp_path):
    return (str(tmp_path / "lock"),
            str(tmp_path / "journal.json"),
            str(tmp_path / "state.json"),
            str(tmp_path / "pyramid.json"))


def _seed_base_pyramid(paths, base, execution):
    _L, _A, _S, pyramid_path = paths
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(pyramid_path, _state_node_text(zero_state, "base"))
    return zero_state


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.coordinate_tile_update is coordinate_tile_update
    assert "coordinate_tile_update" in lidar_scan.__all__


def test_atomic_write_helper_was_renamed():
    assert hasattr(tiles_module, "_atomic_write_text")
    assert not hasattr(tiles_module, "_atomic_write_json")


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_paths_must_be_a_tuple(tmp_path):
    L, A, S, P = _paths(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update([L, A, S, P], _base_pyramid(), _execution())


def test_paths_must_have_four_items(tmp_path):
    L, A, S, P = _paths(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S), _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P, L), _base_pyramid(),
                               _execution())


def test_path_members_must_be_non_empty_strs(tmp_path):
    L, A, S, P = _paths(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, 1), _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, b"S", P), _base_pyramid(),
                               _execution())
    with pytest.raises(ValueError):
        coordinate_tile_update(("", A, S, P), _base_pyramid(), _execution())


def test_paths_must_be_pairwise_distinct(tmp_path):
    L, A, S, P = _paths(tmp_path)
    with pytest.raises(ValueError):
        coordinate_tile_update((L, A, S, S), _base_pyramid(), _execution())
    with pytest.raises(ValueError):
        coordinate_tile_update((L, L, S, P), _base_pyramid(), _execution())


def test_base_and_execution_types(tmp_path):
    L, A, S, P = _paths(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P), [], _execution())
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P), _base_pyramid(), 1)


def test_limit_type_and_value_contract(tmp_path):
    L, A, S, P = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P), base, execution, limit=1.0)
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P), base, execution, limit=True)
    with pytest.raises(ValueError):
        coordinate_tile_update((L, A, S, P), base, execution, limit=-1)


def test_verify_must_be_bool_and_limit_none_when_verifying(tmp_path):
    L, A, S, P = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    with pytest.raises(TypeError):
        coordinate_tile_update((L, A, S, P), base, execution, verify=0)
    with pytest.raises(ValueError):
        coordinate_tile_update((L, A, S, P), base, execution, limit=1,
                               verify=True)


# ---------------------------------------------------------------------------
# publish flow
# ---------------------------------------------------------------------------

def test_fresh_publication_seeds_journal_and_publishes_one(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)

    result = coordinate_tile_update(paths, base, execution, limit=1)

    journal_bytes = _read(A)
    state_bytes = _read(S)
    pyramid_bytes = _read(P)
    assert state_bytes.decode("utf-8") == result
    assert json.loads(result)["committed"] == 1
    records = _journal_records_raw(journal_bytes.decode("utf-8"))
    assert [n for n, _state in records] == [0, 1]
    assert json.loads(records[0][1])["committed"] == 0
    assert records[1][1] == result
    assert pyramid_bytes.decode("utf-8") == \
        _state_node_text(result, "pyramid")
    # Compact canonical UTF-8: no BOM, whitespace or trailing newline.
    assert not journal_bytes.startswith(b"\xef\xbb\xbf")
    assert not journal_bytes.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in journal_bytes)
    # The lock file exists and has been released for re-entry.
    assert open(L, "rb").close() is None


def test_limit_zero_on_fresh_state_registers_only_genesis(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    zero_state = _seed_base_pyramid(paths, base, execution)

    result = coordinate_tile_update(paths, base, execution, limit=0)

    assert result == zero_state
    assert _read(S).decode("utf-8") == zero_state
    records = _journal_records_raw(_read(A).decode("utf-8"))
    assert [n for n, _state in records] == [0]
    assert records[0][1] == zero_state


def test_chunked_coordination_completes_the_chain(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    _seed_base_pyramid(paths, base, execution)

    first = coordinate_tile_update(paths, base, execution, limit=1)
    second = coordinate_tile_update(paths, base, execution, limit=1)
    third = coordinate_tile_update(paths, base, execution)

    one_shot = commit_tile_update_plan(base, execution)
    assert first != second != third
    assert third == one_shot
    assert _read(S).decode("utf-8") == one_shot
    assert _read(P).decode("utf-8") == \
        _state_node_text(one_shot, "pyramid")
    records = _journal_records_raw(_read(A).decode("utf-8"))
    assert [n for n, _state in records] == [0, 1, 2, 3]
    assert [json.loads(state)["committed"] for _n, state in records] == \
        [0, 1, 2, 3]
    assert records[-1][1] == one_shot


def test_re_entered_completed_coordination_is_byte_identical(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    completed = coordinate_tile_update(paths, base, execution)

    journal_before = _read(paths[1])
    state_before = _read(paths[2])
    pyramid_before = _read(paths[3])
    again = coordinate_tile_update(paths, base, execution)
    assert again == completed
    assert _read(paths[1]) == journal_before
    assert _read(paths[2]) == state_before
    assert _read(paths[3]) == pyramid_before


def test_genesis_accepts_preexisting_zero_step_state(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    zero_state = _seed_base_pyramid(paths, base, execution)
    _write(S, zero_state)

    result = coordinate_tile_update(paths, base, execution, limit=1)

    records = _journal_records_raw(_read(A).decode("utf-8"))
    assert [n for n, _state in records] == [0, 1]
    assert records[0][1] == zero_state
    assert _read(S).decode("utf-8") == result


def test_genesis_rejects_advanced_state_without_journal(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    _write(S, commit_tile_update_plan(base, execution, max_tasks=1))

    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution)
    # The journal must not have been created on failure.
    with pytest.raises(FileNotFoundError):
        _read(A)


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------

def test_lagging_pyramid_is_backfilled_without_journal_change(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    first = coordinate_tile_update(paths, base, execution, limit=1)
    # Crash between the S and P replacements: P is back at the base.
    _write(P, _state_node_text(first, "base"))
    journal_before = _read(A)

    result = coordinate_tile_update(paths, base, execution, limit=0)

    assert result == first
    assert _read(P).decode("utf-8") == \
        _state_node_text(first, "pyramid")
    assert _read(A) == journal_before


def test_state_leading_the_journal_is_backfilled_then_publication_resumes(
        tmp_path):
    # Crash window: publish_tile_update_plan replaced S and P but the
    # journal replacement never happened. The replayable S is the sole
    # record appended, after which the publication resumes normally.
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)

    journal = _read(A).decode("utf-8")
    node = tiles_module._parse_json_node(journal, 0)
    genesis_node = node[0]["records"][0][0]
    _write(A, '{"records":['
              + journal[genesis_node[1]:genesis_node[2]] + "]}")

    result = coordinate_tile_update(paths, base, execution)

    one_shot = commit_tile_update_plan(base, execution)
    assert result == one_shot
    records = _journal_records_raw(_read(A).decode("utf-8"))
    assert [n for n, _state in records] == [0, 1, 2]
    assert records[1][1] == commit_tile_update_plan(
        base, execution, max_tasks=1)
    assert records[-1][1] == one_shot
    assert _read(S).decode("utf-8") == one_shot
    assert _read(P).decode("utf-8") == \
        _state_node_text(one_shot, "pyramid")


def test_state_lagging_the_journal_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution)

    # Tamper S back to the zero-step state while A records completion.
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    _write(S, zero_state)
    _write(P, _state_node_text(zero_state, "base"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution)


def test_missing_state_with_journal_is_value_error(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    os.unlink(S)
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution)


# ---------------------------------------------------------------------------
# verify flow
# ---------------------------------------------------------------------------

def test_verify_returns_terminal_n_and_completion(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)

    assert coordinate_tile_update(paths, base, execution, verify=True) == \
        (1, False)

    coordinate_tile_update(paths, base, execution)
    assert coordinate_tile_update(paths, base, execution, verify=True) == \
        (2, True)


def test_verify_is_strictly_read_only(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution)

    before = tuple(_read(path) for path in paths)
    result = coordinate_tile_update(paths, base, execution, verify=True)
    assert result == (1, True)
    after = tuple(_read(path) for path in paths)
    assert before == after


def test_verify_requires_every_file(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)

    # No journal, state or pyramid written yet: every absence is OSError.
    with pytest.raises(OSError):
        coordinate_tile_update(paths, base, execution, verify=True)

    coordinate_tile_update(paths, base, execution, limit=1)
    os.unlink(S)
    with pytest.raises(OSError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_verify_replays_every_journal_state(tmp_path):
    # A hand-built journal whose intermediate state embeds a pyramid that
    # does not reproduce from base and execution is rejected even though
    # its terminal state alone would be consistent.
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)

    other_base = build_tile_pyramid(
        ((10.0, 10.0, 9.0, 1, 1.0),
         (600.0, 600.0, 8.0, 1, 1.0),
         (1100.0, 1100.0, 7.0, 1, 1.0)), levels=1)
    foreign_one = commit_tile_update_plan(other_base, execution, max_tasks=1)
    records = _journal_records_raw(_read(A).decode("utf-8"))
    zero_text = records[0][1]
    _write(A, '{"records":[[0,' + zero_text + "],[1,"
              + foreign_one + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_verify_rejects_state_and_pyramid_tampering(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    completed = coordinate_tile_update(paths, base, execution)

    # S must equal the journal's terminal state byte for byte.
    state_before = _read(S)
    _write(S, commit_tile_update_plan(base, execution, max_tasks=0))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)
    _write(S, state_before)

    # P must equal the terminal state's embedded pyramid byte for byte.
    _write(P, _state_node_text(completed, "base"))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_verify_rejects_binding_to_another_execution(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution)

    other_execution = _execution(_PLAN_THREE)
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, other_execution, verify=True)


# ---------------------------------------------------------------------------
# journal encoding validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate", [
    # JSON pretty printing: whitespace present.
    lambda text: json.dumps(json.loads(text)),
    # Leading byte order mark.
    lambda text: "\ufeff" + text,
    # Trailing newline.
    lambda text: text + "\n",
    # Trailing data after the document.
    lambda text: text + " ",
    # Wrong top-level key.
    lambda text: '{"entries":' + text[len('{"records":'):],
    # records missing entirely.
    lambda text: "{}",
    # n does not start at zero.
    lambda text: _replace_first_n(text, 1),
    # n values not strictly increasing.
    lambda text: _replace_last_n(text, 0),
    # n is a boolean.
    lambda text: _replace_first_n(text, "true"),
])
def test_malformed_journal_is_value_error(tmp_path, mutate):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    _write(paths[1], mutate(_read(paths[1]).decode("utf-8")))
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def test_empty_records_array_is_value_error(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    _write(paths[1], '{"records":[]}')
    # Even the publish flow rejects an existing but empty journal.
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution)


def test_journal_with_non_advancing_state_is_value_error(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    records = _journal_records_raw(_read(A).decode("utf-8"))
    # Append a second record that repeats the zero-step state.
    _write(A, '{"records":[[0,' + records[0][1] + "],[1,"
              + records[0][1] + "]]}")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


def _replace_first_n(text, value):
    node = tiles_module._parse_json_node(text, 0)
    n_node = node[0]["records"][0][0][0][0]
    return text[:n_node[1]] + str(value) + text[n_node[2]:]


def _replace_last_n(text, value):
    node = tiles_module._parse_json_node(text, 0)
    n_node = node[0]["records"][0][-1][0][0]
    return text[:n_node[1]] + str(value) + text[n_node[2]:]


@pytest.mark.parametrize("which", ["journal", "state", "pyramid"])
def test_non_utf8_files_are_value_errors(tmp_path, which):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)
    index = {"journal": 1, "state": 2, "pyramid": 3}[which]
    _write(paths[index], b"\xff\xfe\x00bad")
    with pytest.raises(ValueError):
        coordinate_tile_update(paths, base, execution, verify=True)


# ---------------------------------------------------------------------------
# missing files and I/O failures
# ---------------------------------------------------------------------------

def test_missing_pyramid_is_os_error(tmp_path):
    paths = _paths(tmp_path)
    base = _base_pyramid()
    execution = _execution()
    with pytest.raises(OSError):
        coordinate_tile_update(paths, base, execution)


def test_missing_journal_but_present_pyramid_creates_state(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    result = coordinate_tile_update(paths, base, execution, limit=0)
    assert json.loads(result)["committed"] == 0
    assert _read(A)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_in_process_raises_blocking_io_error(tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)

    holder = open(L, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_update(paths, base, execution, verify=True)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    # The lock is released by the call itself on success.
    assert coordinate_tile_update(paths, base, execution, verify=True) == \
        (1, False)


def test_lock_contention_across_processes_raises_blocking_io_error(
        tmp_path):
    paths = _paths(tmp_path)
    L, A, S, P = paths
    base = _base_pyramid()
    execution = _execution()
    _seed_base_pyramid(paths, base, execution)
    coordinate_tile_update(paths, base, execution, limit=1)

    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,sys,time\n"
         "f=open(sys.argv[1],'rb')\n"
         "fcntl.flock(f, fcntl.LOCK_EX)\n"
         "time.sleep(5)\n", L],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        started = time.monotonic()
        with pytest.raises(BlockingIOError):
            coordinate_tile_update(paths, base, execution, verify=True)
        # Non-blocking: it must fail immediately rather than wait.
        assert time.monotonic() - started < 2.0
    finally:
        holder.terminate()
        holder.wait()
