"""Tests for :func:`lidar_scan.merge_recovery_indexes`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        merge_recovery_indexes, plan_checkout_recovery,
                        update_recovery_index)
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
    plan_gap = plan_checkout_recovery(log, None, ledgers[:3])
    plan_zero = plan_checkout_recovery(log, None, ())

    def histories(plan, count):
        states = recovery_fixtures._states(plan, count)
        partial = build_recovery_history(
            plan, (recovery_fixtures._summary(plan, states[:1]),))
        if count == 1:
            completed = partial
        else:
            completed = build_recovery_history(
                plan,
                (recovery_fixtures._summary(plan, states[:1]),
                 recovery_fixtures._summary(plan, states)))
        return states, partial, completed

    states_one, partial_one, completed_one = histories(plan_one, 2)
    states_two, partial_two, completed_two = histories(plan_two, 1)
    started_two = build_recovery_history(plan_two, ())
    history_zero = build_recovery_history(plan_zero, ())
    empty = update_recovery_index(None, ())
    return {
        "log": log, "snaps": snaps, "plan_one": plan_one,
        "plan_two": plan_two, "plan_gap": plan_gap, "plan_zero": plan_zero,
        "state_one": states_one[0], "partial_one": partial_one,
        "completed_one": completed_one, "state_two": states_two[0],
        "partial_two": partial_two, "completed_two": completed_two,
        "started_two": started_two, "history_zero": history_zero,
        "empty": empty}


def _index(*items):
    return update_recovery_index(None, tuple(items))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.merge_recovery_indexes is merge_recovery_indexes
    assert "merge_recovery_indexes" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty inputs
# ---------------------------------------------------------------------------

def test_empty_tuple_document_shape():
    text = merge_recovery_indexes(())
    assert text == '{"entries":[],"audit":[],"snapshot":null,' \
                   '"resume":null,"complete":true}'
    assert json.loads(text) == {
        "entries": [], "audit": [], "snapshot": None, "resume": None,
        "complete": True}


def test_empty_indexes_are_skipped(scene):
    empty = scene["empty"]
    assert merge_recovery_indexes((empty,)) == \
        '{"entries":[],"audit":[],"snapshot":null,"resume":null,' \
        '"complete":true}'
    index = _index((scene["plan_one"], scene["partial_one"]))
    merged = merge_recovery_indexes((empty, index, empty))
    assert json.loads(merged)["entries"] == json.loads(index)["entries"]


# ---------------------------------------------------------------------------
# single index
# ---------------------------------------------------------------------------

def test_single_index_adds_audit_array(scene):
    index = _index((scene["plan_one"], scene["partial_one"]))
    text = merge_recovery_indexes((index,))
    document = json.loads(text)
    assert list(document) == ["entries", "audit", "snapshot", "resume",
                              "complete"]
    assert document["entries"] == json.loads(index)["entries"]
    assert document["audit"] == [[0, "b0", "applied"]]
    assert document["snapshot"] == json.loads(scene["snaps"][2])
    assert document["resume"] == [0, json.loads(scene["state_one"])]
    assert document["complete"] is False
    assert text == _canonical(document)
    assert " " not in text and not text.endswith("\n")


def test_single_completed_index(scene):
    index = _index((scene["plan_one"], scene["completed_one"]),
                   (scene["plan_two"], scene["completed_two"]))
    document = json.loads(merge_recovery_indexes((index,)))
    assert document["audit"] == [
        [0, "b0", "applied"], [0, "b1", "applied"], [1, "b2", "applied"]]
    assert document["resume"] is None
    assert document["complete"] is True


def test_entries_embed_documents_byte_for_byte(scene):
    index = _index((scene["plan_one"], scene["partial_one"]))
    text = merge_recovery_indexes((index,))
    assert ("[" + scene["plan_one"] + "," + scene["partial_one"] + "]") in text


# ---------------------------------------------------------------------------
# joining indexes
# ---------------------------------------------------------------------------

def test_completed_then_chained_completed(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    second = _index((scene["plan_two"], scene["completed_two"]))
    document = json.loads(merge_recovery_indexes((first, second)))
    assert len(document["entries"]) == 2
    assert document["complete"] is True
    assert document["resume"] is None
    assert document["snapshot"] == json.loads(scene["snaps"][4])
    assert document["audit"] == [
        [0, "b0", "applied"], [0, "b1", "applied"], [1, "b2", "applied"]]


def test_unfinished_last_entry_kept(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    second = _index((scene["plan_two"], scene["started_two"]))
    document = json.loads(merge_recovery_indexes((first, second)))
    assert len(document["entries"]) == 2
    assert document["complete"] is False
    assert document["resume"][0] == 1
    assert document["audit"] == [
        [0, "b0", "applied"], [0, "b1", "applied"]]


def test_replacement_across_indexes(scene):
    first = _index((scene["plan_one"], scene["completed_one"]),
                   (scene["plan_two"], scene["started_two"]))
    second = _index((scene["plan_two"], scene["completed_two"]))
    document = json.loads(merge_recovery_indexes((first, second)))
    assert len(document["entries"]) == 2
    assert document["complete"] is True
    assert document["snapshot"] == json.loads(scene["snaps"][4])
    assert document["audit"] == [
        [0, "b0", "applied"], [0, "b1", "applied"], [1, "b2", "applied"]]


def test_three_index_merge(scene):
    first = _index((scene["plan_one"], scene["partial_one"]))
    middle = _index((scene["plan_one"], scene["completed_one"]),
                    (scene["plan_two"], scene["started_two"]))
    last = _index((scene["plan_two"], scene["completed_two"]))
    document = json.loads(merge_recovery_indexes((first, middle, last)))
    assert len(document["entries"]) == 2
    assert document["complete"] is True
    assert document["audit"] == [
        [0, "b0", "applied"], [0, "b1", "applied"], [1, "b2", "applied"]]


def test_zero_todo_completed_plan_chains_at_null(scene):
    zero = _index((scene["plan_zero"], scene["history_zero"]))
    first = _index((scene["plan_one"], scene["completed_one"]))
    document = json.loads(merge_recovery_indexes((zero, first)))
    assert [len(e) for e in document["entries"]] == [2, 2]
    assert document["snapshot"] == json.loads(scene["snaps"][3])
    assert document["complete"] is True
    assert document["audit"] == [
        [1, "b0", "applied"], [1, "b1", "applied"]]


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    index = _index((scene["plan_one"], scene["partial_one"]))
    for value in ([], None, 1, "x", {}):
        with pytest.raises(TypeError):
            merge_recovery_indexes(value)
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            merge_recovery_indexes((index, value))


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_member_rejected():
    for value in ("", "not json", "{", "null", "[]", "1"):
        with pytest.raises(ValueError):
            merge_recovery_indexes((value,))


def test_non_canonical_member_rejected(scene):
    index = _index((scene["plan_one"], scene["partial_one"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((index + " ",))
    with pytest.raises(ValueError):
        merge_recovery_indexes((" " + index,))
    pretty = json.dumps(json.loads(index), indent=2)
    with pytest.raises(ValueError):
        merge_recovery_indexes((pretty,))


def test_join_requires_chained_boundary(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    gap_states = recovery_fixtures._states(scene["plan_gap"], 3)
    gap_history = build_recovery_history(
        scene["plan_gap"],
        (recovery_fixtures._summary(scene["plan_gap"], gap_states),))
    gap = _index((scene["plan_gap"], gap_history))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, gap))


def test_unfinished_join_requires_same_plan(scene):
    first = _index((scene["plan_one"], scene["partial_one"]))
    second = _index((scene["plan_two"], scene["completed_two"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, second))


def test_unfinished_join_requires_strict_extension(scene):
    first = _index((scene["plan_one"], scene["partial_one"]))
    same = _index((scene["plan_one"], scene["partial_one"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, same))
    trimmed = json.loads(scene["completed_one"])
    trimmed["runs"] = trimmed["runs"][1:]
    rewritten = _index((scene["plan_one"], _canonical(trimmed)))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, rewritten))


def test_completed_plan_may_not_be_repeated(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, first))


def test_completed_zero_todo_plan_may_not_be_repeated(scene):
    zero = _index((scene["plan_zero"], scene["history_zero"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((zero, zero))


def test_completed_zero_todo_plan_repeated_inside_member_rejected(scene):
    # update_recovery_index permits recording the same completed
    # zero-todo plan twice, but merge must reject the global repeat
    zero = _index((scene["plan_zero"], scene["history_zero"]))
    repeated = update_recovery_index(
        zero, ((scene["plan_zero"], scene["history_zero"]),))
    assert len(json.loads(repeated)["entries"]) == 2
    with pytest.raises(ValueError):
        merge_recovery_indexes((repeated,))


def test_completed_entry_may_not_be_duplicated_across_join(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    second = _index((scene["plan_one"], scene["completed_one"]),
                    (scene["plan_two"], scene["started_two"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((first, second))


def test_duplicate_batch_id_across_indexes_rejected(scene):
    second = _index((scene["plan_two"], scene["completed_two"]))
    with pytest.raises(ValueError):
        merge_recovery_indexes((second, second))


def test_tampered_member_rejected(scene):
    index = _index((scene["plan_one"], scene["partial_one"]))
    tampered = json.loads(index)
    tampered["snapshot"] = None
    with pytest.raises(ValueError):
        merge_recovery_indexes((_canonical(tampered),))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    first = _index((scene["plan_one"], scene["completed_one"]))
    second = _index((scene["plan_two"], scene["completed_two"]))
    indexes = (first, second)
    saved = tuple(indexes)
    one = merge_recovery_indexes(indexes)
    two = merge_recovery_indexes(indexes)
    assert one == two
    assert indexes == saved
    # members must be update_recovery_index documents: the merged
    # five-key shape is not itself a valid member
    with pytest.raises(ValueError):
        merge_recovery_indexes((one, scene["empty"]))
