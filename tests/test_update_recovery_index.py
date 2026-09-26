"""Tests for :func:`lidar_scan.update_recovery_index`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        plan_checkout_recovery, update_recovery_index)
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
    return {
        "snaps": snaps, "plan_one": plan_one, "plan_two": plan_two,
        "plan_gap": plan_gap, "state_one": states_one[0],
        "partial_one": partial_one, "completed_one": completed_one,
        "state_two": states_two[0], "partial_two": partial_two,
        "completed_two": completed_two, "started_two": started_two}


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.update_recovery_index is update_recovery_index
    assert "update_recovery_index" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty inputs
# ---------------------------------------------------------------------------

def test_empty_index_document_shape():
    text = update_recovery_index(None, ())
    assert text == '{"entries":[],"snapshot":null,"resume":null,' \
                   '"complete":true}'
    assert json.loads(text) == {
        "entries": [], "snapshot": None, "resume": None, "complete": True}


def test_empty_items_leave_index_byte_for_byte_unchanged(scene):
    index = update_recovery_index(None, ((scene["plan_one"],
                                          scene["partial_one"]),))
    assert update_recovery_index(index, ()) == index
    empty = update_recovery_index(None, ())
    assert update_recovery_index(empty, ()) == empty


# ---------------------------------------------------------------------------
# appending and replacing
# ---------------------------------------------------------------------------

def test_first_in_progress_entry_fields(scene):
    snaps = scene["snaps"]
    text = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    document = json.loads(text)
    assert list(document) == ["entries", "snapshot", "resume", "complete"]
    assert document["entries"] == [
        [json.loads(scene["plan_one"]),
         json.loads(scene["partial_one"])]]
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["resume"] == [0, json.loads(scene["state_one"])]
    assert document["complete"] is False
    assert text == _canonical(document)
    assert not text.endswith("\n") and " " not in text


def test_entries_embed_documents_byte_for_byte(scene):
    text = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    assert ("[" + scene["plan_one"] + "," + scene["partial_one"] + "]") in text


def test_replace_unfinished_then_chain_second_entry(scene):
    snaps = scene["snaps"]
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    text = update_recovery_index(
        index, ((scene["plan_one"], scene["completed_one"]),
                (scene["plan_two"], scene["completed_two"])))
    document = json.loads(text)
    assert len(document["entries"]) == 2
    assert document["entries"][0] == [
        json.loads(scene["plan_one"]),
        json.loads(scene["completed_one"])]
    assert document["entries"][1] == [
        json.loads(scene["plan_two"]),
        json.loads(scene["completed_two"])]
    assert document["snapshot"] == json.loads(snaps[4])
    assert document["resume"] is None
    assert document["complete"] is True
    assert text == _canonical(document)


def test_replace_empty_runs_unfinished_history(scene):
    snaps = scene["snaps"]
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),
               (scene["plan_two"], scene["started_two"])))
    document = json.loads(index)
    assert document["complete"] is False
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"][0] == 1
    text = update_recovery_index(
        index, ((scene["plan_two"], scene["completed_two"]),))
    completed = json.loads(text)
    assert completed["complete"] is True
    assert completed["resume"] is None
    assert completed["snapshot"] == json.loads(snaps[4])
    assert len(completed["entries"]) == 2


def test_completed_entries_are_kept_not_replaced(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),
               (scene["plan_two"], scene["completed_two"])))
    assert update_recovery_index(index, ()) == index


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    for value in (1, [], b"x", {}):
        with pytest.raises(TypeError):
            update_recovery_index(value, ())
    for value in ([], None, "x", 1):
        with pytest.raises(TypeError):
            update_recovery_index(None, value)
    pair = (scene["plan_one"], scene["partial_one"])
    for value in (pair, None, 1, "x", [pair], (scene["plan_one"],),
                  (scene["plan_one"], scene["partial_one"], 1)):
        with pytest.raises(TypeError):
            update_recovery_index(None, (value,) if not isinstance(value, tuple)
                                  or len(value) != 2 else value)
    with pytest.raises(TypeError):
        update_recovery_index(None, ((1, scene["partial_one"]),))
    with pytest.raises(TypeError):
        update_recovery_index(None, ((scene["plan_one"], 1),))
    with pytest.raises(TypeError):
        update_recovery_index(None, ((scene["plan_one"], None),))


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_documents_rejected(scene):
    with pytest.raises(ValueError):
        update_recovery_index(None, (("{", scene["partial_one"]),))
    with pytest.raises(ValueError):
        update_recovery_index(None,
                              ((scene["plan_one"], "not json"),))
    with pytest.raises(ValueError):
        update_recovery_index(None, ((scene["plan_one"], "null"),))


def test_non_canonical_documents_rejected(scene):
    pretty = json.dumps(json.loads(scene["partial_one"]), indent=2)
    with pytest.raises(ValueError):
        update_recovery_index(None, ((scene["plan_one"], pretty),))
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    with pytest.raises(ValueError):
        update_recovery_index(index + " ", ())


def test_history_must_belong_to_its_plan(scene):
    with pytest.raises(ValueError):
        update_recovery_index(
            None, ((scene["plan_two"], scene["partial_one"]),))


def test_non_last_item_must_be_complete(scene):
    with pytest.raises(ValueError):
        update_recovery_index(
            None, ((scene["plan_one"], scene["partial_one"]),
                   (scene["plan_two"], scene["completed_two"])))


def test_replacement_requires_same_plan(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    with pytest.raises(ValueError):
        update_recovery_index(
            index, ((scene["plan_two"], scene["completed_two"]),))


def test_replacement_must_strictly_extend(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    with pytest.raises(ValueError):
        update_recovery_index(
            index, ((scene["plan_one"], scene["partial_one"]),))
    trimmed = json.loads(scene["completed_one"])
    trimmed["runs"] = trimmed["runs"][1:]
    with pytest.raises(ValueError):
        update_recovery_index(
            index, ((scene["plan_one"], _canonical(trimmed)),))


def test_chained_items_must_share_boundary(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    gap_states = recovery_fixtures._states(scene["plan_gap"], 3)
    gap_history = build_recovery_history(
        scene["plan_gap"],
        (recovery_fixtures._summary(scene["plan_gap"], gap_states),))
    with pytest.raises(ValueError):
        update_recovery_index(
            index, ((scene["plan_one"], scene["completed_one"]),
                    (scene["plan_gap"], gap_history)))
    # the same content is valid once plan one reaches snaps[3]
    completed_index = update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),))
    text = update_recovery_index(
        completed_index,
        ((scene["plan_two"], scene["completed_two"]),))
    assert json.loads(text)["complete"] is True


def test_duplicate_batch_ids_across_entries_rejected(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),
               (scene["plan_two"], scene["completed_two"])))
    document = json.loads(index)
    document["entries"].append(document["entries"][1])
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(document), ())


def test_tampered_index_fields_rejected(scene):
    index = update_recovery_index(
        None, ((scene["plan_one"], scene["partial_one"]),))
    tampered = json.loads(index)
    tampered["snapshot"] = None
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(tampered), ())
    tampered = json.loads(index)
    tampered["resume"][0] = 1
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(tampered), ())
    with pytest.raises(ValueError):
        update_recovery_index(
            '{"entries":[],"snapshot":null,"resume":null,"complete":false}',
            ())


def test_repeatable_and_inputs_unchanged(scene):
    items = ((scene["plan_one"], scene["partial_one"]),)
    saved = items
    one = update_recovery_index(None, items)
    two = update_recovery_index(None, items)
    assert one == two
    assert items == saved
    completed = ((scene["plan_one"], scene["completed_one"]),
                 (scene["plan_two"], scene["completed_two"]))
    again = update_recovery_index(one, completed)
    assert update_recovery_index(one, completed) == again
