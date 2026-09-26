"""Tests for :func:`lidar_scan.summarize_recovery_index`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        merge_recovery_indexes, plan_checkout_recovery,
                        summarize_recovery_index, update_recovery_index)
from lidar_scan import tiles as tiles_module
import test_build_recovery_history as recovery_fixtures


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    plan_one = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    plan_two = plan_checkout_recovery(log, checkpoint, ledgers[:3])

    states_one = recovery_fixtures._states(plan_one, 2)
    completed_one = build_recovery_history(
        plan_one,
        (recovery_fixtures._summary(plan_one, states_one[:1]),
         recovery_fixtures._summary(plan_one, states_one)))
    states_two = recovery_fixtures._states(plan_two, 1)
    completed_two = build_recovery_history(
        plan_two,
        (recovery_fixtures._summary(plan_two, states_two),))
    started_two = build_recovery_history(plan_two, ())
    return {
        "snaps": snaps, "plan_one": plan_one, "plan_two": plan_two,
        "completed_one": completed_one, "completed_two": completed_two,
        "started_two": started_two}


def _completed_index(scene):
    return merge_recovery_indexes((
        update_recovery_index(
            None, ((scene["plan_one"], scene["completed_one"]),)),
        update_recovery_index(
            None, ((scene["plan_two"], scene["completed_two"]),)),))


def _unfinished_index(scene):
    return merge_recovery_indexes((
        update_recovery_index(
            None, ((scene["plan_one"], scene["completed_one"]),)),
        update_recovery_index(
            None, ((scene["plan_two"], scene["started_two"]),)),))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.summarize_recovery_index \
        is summarize_recovery_index
    assert "summarize_recovery_index" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# full and partial ranges
# ---------------------------------------------------------------------------

def test_full_range_of_completed_index(scene):
    index = _completed_index(scene)
    document = json.loads(index)
    result = summarize_recovery_index(index, 0, 2)
    start, stop, audit, before, after, resume, complete = result
    assert result == (0, 2, audit, before, after, resume, complete)
    assert audit == tuple(tuple(row) for row in document["audit"])
    assert json.loads(before) == document["entries"][0][1]["start"]
    assert json.loads(after) == document["snapshot"]
    assert resume is None
    assert complete is True


def test_single_entry_range(scene):
    index = _completed_index(scene)
    document = json.loads(index)
    result = summarize_recovery_index(index, 0, 1)
    assert result[0] == 0 and result[1] == 1
    assert result[2] == tuple(
        (0, batch_id, status)
        for batch_id, status in document["entries"][0][1]["audit"])
    assert json.loads(result[3]) == document["entries"][0][1]["start"]
    assert json.loads(result[4]) == document["entries"][1][1]["start"]
    assert result[5] is None
    assert result[6] is True
    tail = summarize_recovery_index(index, 1, 2)
    assert tail[2] == tuple(
        (1, batch_id, status)
        for batch_id, status in document["entries"][1][1]["audit"])


def test_empty_range_inside_index(scene):
    index = _completed_index(scene)
    document = json.loads(index)
    result = summarize_recovery_index(index, 1, 1)
    assert result[:3] == (1, 1, ())
    assert result[3] == result[4]
    assert json.loads(result[3]) == document["entries"][1][1]["start"]
    assert result[5] is None
    assert result[6] is True


def test_empty_range_at_end_uses_terminal_snapshot(scene):
    index = _completed_index(scene)
    document = json.loads(index)
    result = summarize_recovery_index(index, 2, 2)
    assert result[2] == ()
    assert result[3] == result[4]
    assert json.loads(result[3]) == document["snapshot"]


def test_empty_index():
    index = merge_recovery_indexes(())
    assert summarize_recovery_index(index, 0, 0) == \
        (0, 0, (), None, None, None, True)


def test_range_ending_at_unfinished_last_entry(scene):
    index = _unfinished_index(scene)
    document = json.loads(index)
    assert document["complete"] is False
    result = summarize_recovery_index(index, 0, 2)
    assert json.loads(result[4]) == document["snapshot"]
    assert json.loads(result[5]) == document["resume"][1]
    assert result[6] is False
    partial = summarize_recovery_index(index, 1, 2)
    assert json.loads(partial[3]) == document["entries"][1][1]["start"]
    assert json.loads(partial[5]) == document["resume"][1]
    assert partial[6] is False


def test_range_excluding_unfinished_last_entry_is_complete(scene):
    index = _unfinished_index(scene)
    result = summarize_recovery_index(index, 0, 1)
    assert result[5] is None
    assert result[6] is True
    empty = summarize_recovery_index(index, 1, 1)
    assert empty[5] is None and empty[6] is True


def test_range_audit_preserves_order_and_entry_indexes(scene):
    index = _completed_index(scene)
    result = summarize_recovery_index(index, 0, 2)
    assert [row[0] for row in result[2]] == sorted(row[0] for row in result[2])
    assert all(isinstance(row, tuple) and len(row) == 3 for row in result[2])


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_index_type_error(scene):
    index = _completed_index(scene)
    for value in (None, 1, [], b"x", {}):
        with pytest.raises(TypeError):
            summarize_recovery_index(value, 0, 1)


def test_bound_types_must_be_non_boolean_integers(scene):
    index = _completed_index(scene)
    for value in (True, False, 1.0, "0", None, (0,)):
        with pytest.raises(TypeError):
            summarize_recovery_index(index, value, 1)
        with pytest.raises(TypeError):
            summarize_recovery_index(index, 0, value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_out_of_bounds_or_inverted_ranges_rejected(scene):
    index = _completed_index(scene)
    for start, stop in ((-1, 0), (0, 3), (2, 1), (1, 0), (3, 3)):
        with pytest.raises(ValueError):
            summarize_recovery_index(index, start, stop)


def test_malformed_or_non_merge_document_rejected(scene):
    with pytest.raises(ValueError):
        summarize_recovery_index("not json", 0, 0)
    with pytest.raises(ValueError):
        summarize_recovery_index("null", 0, 0)
    # the four-key update_recovery_index format is not a merge document
    with pytest.raises(ValueError):
        summarize_recovery_index(update_recovery_index(None, ()), 0, 0)


def test_duplicate_batch_id_rejected(scene):
    index = _completed_index(scene)
    document = json.loads(index)
    document["entries"].append(document["entries"][0])
    with pytest.raises(ValueError):
        summarize_recovery_index(_canonical(document), 0, 3)


def test_repeatable_and_input_unchanged(scene):
    index = _completed_index(scene)
    saved = index
    first = summarize_recovery_index(index, 0, 2)
    second = summarize_recovery_index(index, 0, 2)
    assert first == second
    assert index == saved
