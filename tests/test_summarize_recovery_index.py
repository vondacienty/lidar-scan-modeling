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
        plan_one, (recovery_fixtures._summary(plan_one, states_one[:1]),
                   recovery_fixtures._summary(plan_one, states_one)))
    started_two = build_recovery_history(plan_two, ())
    states_two = recovery_fixtures._states(plan_two, 1)
    completed_two = build_recovery_history(
        plan_two, (recovery_fixtures._summary(plan_two, states_two),))

    incomplete = merge_recovery_indexes((update_recovery_index(
        None, ((plan_one, completed_one), (plan_two, started_two))),))
    complete = merge_recovery_indexes((update_recovery_index(
        None, ((plan_one, completed_one), (plan_two, completed_two))),))
    return {"incomplete": incomplete, "complete": complete,
            "completed_one": json.loads(completed_one),
            "started_two": json.loads(started_two),
            "completed_two": json.loads(completed_two)}


def _expected_audit(*entries):
    rows = []
    for entry_index, history in enumerate(entries):
        for batch_id, status in history["audit"]:
            rows.append((entry_index, batch_id, status))
    return tuple(rows)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.summarize_recovery_index is summarize_recovery_index
    assert "summarize_recovery_index" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty index
# ---------------------------------------------------------------------------

def test_empty_index_empty_range():
    index = merge_recovery_indexes(())
    assert index == '{"entries":[],"audit":[],"snapshot":null,' \
                    '"resume":null,"complete":true}'
    assert summarize_recovery_index(index, 0, 0) == \
        (0, 0, (), None, None, None, True)


def test_empty_index_bounds():
    index = merge_recovery_indexes(())
    with pytest.raises(ValueError):
        summarize_recovery_index(index, 0, 1)
    with pytest.raises(ValueError):
        summarize_recovery_index(index, 1, 1)


# ---------------------------------------------------------------------------
# ranges of a completed index
# ---------------------------------------------------------------------------

def test_full_range_of_completed_index(scene):
    index = scene["complete"]
    one = scene["completed_one"]
    two = scene["completed_two"]
    result = summarize_recovery_index(index, 0, 2)
    assert result == (0, 2, _expected_audit(one, two),
                      _canonical(one["start"]), _canonical(two["end"]),
                      None, True)


def test_prefix_range_of_completed_index(scene):
    index = scene["complete"]
    one = scene["completed_one"]
    result = summarize_recovery_index(index, 0, 1)
    assert result == (0, 1, _expected_audit(one),
                      _canonical(one["start"]), _canonical(one["end"]),
                      None, True)


def test_suffix_range_of_completed_index(scene):
    index = scene["complete"]
    two = scene["completed_two"]
    audit = tuple((1, batch_id, status) for batch_id, status in two["audit"])
    result = summarize_recovery_index(index, 1, 2)
    assert result == (1, 2, audit, _canonical(two["start"]),
                      _canonical(two["end"]), None, True)


def test_empty_range_inside_completed_index(scene):
    index = scene["complete"]
    one = scene["completed_one"]
    two = scene["completed_two"]
    result = summarize_recovery_index(index, 1, 1)
    assert result == (1, 1, (), _canonical(two["start"]),
                      _canonical(two["start"]), None, True)
    result = summarize_recovery_index(index, 0, 0)
    assert result == (0, 0, (), _canonical(one["start"]),
                      _canonical(one["start"]), None, True)
    result = summarize_recovery_index(index, 2, 2)
    assert result == (2, 2, (), _canonical(two["end"]),
                      _canonical(two["end"]), None, True)


# ---------------------------------------------------------------------------
# ranges of an in-progress index
# ---------------------------------------------------------------------------

def test_full_range_of_incomplete_index(scene):
    index = scene["incomplete"]
    one = scene["completed_one"]
    two = scene["started_two"]
    result = summarize_recovery_index(index, 0, 2)
    assert result == (0, 2, _expected_audit(one, two),
                      _canonical(one["start"]), _canonical(two["end"]),
                      _canonical(two["resume"]), False)


def test_completed_prefix_of_incomplete_index(scene):
    index = scene["incomplete"]
    one = scene["completed_one"]
    result = summarize_recovery_index(index, 0, 1)
    assert result == (0, 1, _expected_audit(one),
                      _canonical(one["start"]), _canonical(one["end"]),
                      None, True)


def test_unfinished_suffix_of_incomplete_index(scene):
    index = scene["incomplete"]
    two = scene["started_two"]
    result = summarize_recovery_index(index, 1, 2)
    assert result == (1, 2, (), _canonical(two["start"]),
                      _canonical(two["end"]), _canonical(two["resume"]),
                      False)


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    index = scene["complete"]
    for value in (None, 1, [], b"x", {}):
        with pytest.raises(TypeError):
            summarize_recovery_index(value, 0, 1)
    for value in (True, False, 0.5, "0", None, (0,)):
        with pytest.raises(TypeError):
            summarize_recovery_index(index, value, 1)
        with pytest.raises(TypeError):
            summarize_recovery_index(index, 0, value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_and_non_canonical_indexes_rejected(scene):
    index = scene["complete"]
    with pytest.raises(ValueError):
        summarize_recovery_index("{", 0, 0)
    with pytest.raises(ValueError):
        summarize_recovery_index(index + " ", 0, 1)
    pretty = json.dumps(json.loads(index), indent=2)
    with pytest.raises(ValueError):
        summarize_recovery_index(pretty, 0, 1)


def test_tampered_index_fields_rejected(scene):
    index = scene["complete"]
    tampered = json.loads(index)
    tampered["audit"] = tampered["audit"][1:]
    with pytest.raises(ValueError):
        summarize_recovery_index(_canonical(tampered), 0, 2)
    tampered = json.loads(index)
    tampered["snapshot"] = None
    with pytest.raises(ValueError):
        summarize_recovery_index(_canonical(tampered), 0, 2)
    tampered = json.loads(index)
    tampered["complete"] = False
    with pytest.raises(ValueError):
        summarize_recovery_index(_canonical(tampered), 0, 2)


def test_duplicate_batch_ids_rejected(scene):
    index = scene["complete"]
    document = json.loads(index)
    document["entries"].append(document["entries"][1])
    with pytest.raises(ValueError):
        summarize_recovery_index(_canonical(document), 0, 2)


def test_range_bounds_rejected(scene):
    index = scene["complete"]
    for start, stop in ((-1, 1), (0, 3), (1, 3), (2, 1), (1, 0), (0, -1),
                        (-2, -1)):
        with pytest.raises(ValueError):
            summarize_recovery_index(index, start, stop)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeatable_and_input_unchanged(scene):
    index = scene["incomplete"]
    saved = index
    one = summarize_recovery_index(index, 0, 2)
    two = summarize_recovery_index(index, 0, 2)
    assert one == two
    assert index == saved
