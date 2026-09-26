"""Tests for :func:`lidar_scan.plan_recovery_index_migration`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        merge_recovery_indexes, plan_checkout_recovery,
                        plan_recovery_index_migration, update_recovery_index)
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
    prefix = merge_recovery_indexes((update_recovery_index(
        None, ((plan_one, completed_one),)),))
    empty = merge_recovery_indexes(())
    return {"log": log, "snaps": snaps, "ledgers": ledgers,
            "plan_one": plan_one, "plan_two": plan_two,
            "completed_one": completed_one, "started_two": started_two,
            "completed_two": completed_two, "incomplete": incomplete,
            "complete": complete, "prefix": prefix, "empty": empty}


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.plan_recovery_index_migration \
        is plan_recovery_index_migration
    assert "plan_recovery_index_migration" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape and encoding
# ---------------------------------------------------------------------------

def test_document_shape_and_encoding(scene):
    text = plan_recovery_index_migration(scene["empty"], scene["empty"])
    assert text == (
        '{"common":[0,0,null],"rollback":[],"pending":[],'
        '"snapshot":null,"resume":null,"complete":true}')
    assert not text.endswith("\n")
    assert list(json.loads(text)) == ["common", "rollback", "pending",
                                      "snapshot", "resume", "complete"]


def test_empty_indexes(scene):
    empty = scene["empty"]
    assert json.loads(plan_recovery_index_migration(empty, empty)) == {
        "common": [0, 0, None], "rollback": [], "pending": [],
        "snapshot": None, "resume": None, "complete": True}


def test_empty_before_full_after(scene):
    doc = json.loads(plan_recovery_index_migration(
        scene["empty"], scene["complete"]))
    assert doc["common"][0] == 0
    assert doc["common"][1] == 0
    assert doc["common"][2] == json.loads(scene["snaps"][1])
    assert doc["rollback"] == []
    assert [row[0] for row in doc["pending"]] == ["b0", "b1", "b2"]
    assert doc["snapshot"] == json.loads(scene["snaps"][4])
    assert doc["resume"] is None
    assert doc["complete"] is True


def test_full_before_empty_after(scene):
    doc = json.loads(plan_recovery_index_migration(
        scene["complete"], scene["empty"]))
    assert doc["common"][:2] == [0, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][1])
    assert [row[0] for row in doc["rollback"]] == ["b2", "b1", "b0"]
    assert doc["pending"] == []
    assert doc["snapshot"] is None
    assert doc["resume"] is None
    assert doc["complete"] is True


# ---------------------------------------------------------------------------
# identical / shared whole entries
# ---------------------------------------------------------------------------

def test_identical_indexes(scene):
    text = plan_recovery_index_migration(scene["complete"],
                                         scene["complete"])
    doc = json.loads(text)
    assert doc["common"][:2] == [2, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][4])
    assert doc["rollback"] == []
    assert doc["pending"] == []
    assert doc["complete"] is True
    assert text == _canonical(doc)


def test_shared_whole_entries_boundary_uses_last_end(scene):
    doc = json.loads(plan_recovery_index_migration(
        scene["prefix"], scene["complete"]))
    assert doc["common"][:2] == [1, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][3])
    assert doc["rollback"] == []
    assert [row[0] for row in doc["pending"]] == ["b2"]
    assert doc["complete"] is True


def test_dropping_entries_rolls_them_back(scene):
    doc = json.loads(plan_recovery_index_migration(
        scene["complete"], scene["prefix"]))
    assert doc["common"][:2] == [1, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][3])
    assert doc["rollback"] == [["b2", "applied"]]
    assert doc["pending"] == []
    assert doc["complete"] is True


# ---------------------------------------------------------------------------
# shared units inside one same-plan entry
# ---------------------------------------------------------------------------

def _single_plan_index(plan, states):
    history = build_recovery_history(
        plan, (recovery_fixtures._summary(plan, states),))
    return merge_recovery_indexes(
        (update_recovery_index(None, ((plan, history),)),)), history


def test_shared_units_inside_same_plan_entry(scene):
    plan = scene["plan_one"]
    before, h_before = _single_plan_index(
        plan, recovery_fixtures._states(plan, 1))
    after, h_after = _single_plan_index(
        plan, recovery_fixtures._states(plan, 2))
    doc = json.loads(plan_recovery_index_migration(before, after))
    assert doc["common"][:2] == [0, 1]
    assert doc["common"][2] == json.loads(scene["snaps"][2])
    assert doc["rollback"] == []
    assert doc["pending"] == [["b1", "applied"]]
    assert doc["snapshot"] == json.loads(scene["snaps"][3])
    assert doc["resume"] is None
    assert doc["complete"] is True
    # rollback arm of the same boundary
    doc = json.loads(plan_recovery_index_migration(after, before))
    assert doc["common"][:2] == [0, 1]
    assert doc["rollback"] == [["b1", "applied"]]
    assert doc["pending"] == []
    assert doc["snapshot"] == json.loads(scene["snaps"][2])
    assert doc["resume"] == json.loads(h_before)["resume"]
    assert doc["complete"] is False


def test_neither_side_has_confirmed_the_next_unit(scene):
    plan = scene["plan_one"]
    unstarted = build_recovery_history(plan, ())
    index_zero = merge_recovery_indexes(
        (update_recovery_index(None, ((plan, unstarted),)),))
    index_one, _ = _single_plan_index(
        plan, recovery_fixtures._states(plan, 1))
    doc = json.loads(plan_recovery_index_migration(
        index_zero, index_one))
    assert doc["common"][:2] == [0, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][1])
    assert doc["rollback"] == []
    assert doc["pending"] == [["b0", "applied"]]
    assert doc["complete"] is False
    doc = json.loads(plan_recovery_index_migration(
        index_zero, index_zero))
    assert doc["common"][:2] == [1, 0]
    assert doc["common"][2] == json.loads(scene["snaps"][1])
    assert doc["rollback"] == []
    assert doc["pending"] == []
    assert doc["resume"] == json.loads(unstarted)["resume"]
    assert doc["complete"] is False


# ---------------------------------------------------------------------------
# in-progress tail carries A's resume state
# ---------------------------------------------------------------------------

def test_resume_and_complete_come_from_after(scene):
    doc = json.loads(plan_recovery_index_migration(
        scene["complete"], scene["incomplete"]))
    assert doc["common"][:2] == [1, 0]
    assert [row[0] for row in doc["rollback"]] == ["b2"]
    assert doc["pending"] == []
    assert doc["snapshot"] == json.loads(scene["snaps"][3])
    assert doc["resume"] == json.loads(scene["started_two"])["resume"]
    assert doc["complete"] is False

    doc = json.loads(plan_recovery_index_migration(
        scene["incomplete"], scene["complete"]))
    assert doc["common"][:2] == [1, 0]
    assert doc["rollback"] == []
    assert doc["pending"] == [["b2", "applied"]]
    assert doc["complete"] is True
    assert doc["resume"] is None


# ---------------------------------------------------------------------------
# forks
# ---------------------------------------------------------------------------

def _forked_indexes(scene):
    log, snaps, ledgers = scene["log"], scene["snaps"], scene["ledgers"]
    checkpoint_two = merge_checkout_logs(log, ledgers[:2])
    checkpoint_three = merge_checkout_logs(log, ledgers[:3])
    plan_pending = plan_checkout_recovery(
        log, checkpoint_two, ledgers[:3])
    history_pending = build_recovery_history(
        plan_pending,
        (recovery_fixtures._summary(
            plan_pending, recovery_fixtures._states(plan_pending, 1)),))
    plan_missing = plan_checkout_recovery(
        log, checkpoint_three, ledgers[:2])
    history_missing = build_recovery_history(
        plan_missing,
        (recovery_fixtures._summary(
            plan_missing, recovery_fixtures._states(plan_missing, 1)),))
    assert plan_pending != plan_missing
    before = merge_recovery_indexes((update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),
               (plan_pending, history_pending))),))
    after = merge_recovery_indexes((update_recovery_index(
        None, ((scene["plan_one"], scene["completed_one"]),
               (plan_missing, history_missing))),))
    return before, after


def test_intersecting_post_boundary_ids_fork(scene):
    before, after = _forked_indexes(scene)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(after, before)


def test_nonempty_index_with_null_boundary(scene):
    # A plan with no units completes immediately with null plan snapshot
    # and null history start/end, so the shared boundary is null.
    log = scene["log"]
    plan = plan_checkout_recovery(log, None, ())
    history = build_recovery_history(plan, ())
    index = merge_recovery_indexes(
        (update_recovery_index(None, ((plan, history),)),))
    doc = json.loads(plan_recovery_index_migration(index, index))
    assert doc["common"] == [1, 0, None]
    assert doc["rollback"] == []
    assert doc["pending"] == []
    assert doc["snapshot"] is None
    assert doc["resume"] is None
    assert doc["complete"] is True


def test_different_starts_with_no_common_prefix_rejected(scene):
    # complete starts at snaps[1]; prefix-one-entry of plan_two starts
    # at snaps[3], so the two indexes cannot share a boundary.
    log, snaps, ledgers = scene["log"], scene["snaps"], scene["ledgers"]
    plan_two = scene["plan_two"]
    history = build_recovery_history(
        plan_two,
        (recovery_fixtures._summary(
            plan_two, recovery_fixtures._states(plan_two, 1)),))
    other = merge_recovery_indexes(
        (update_recovery_index(None, ((plan_two, history),)),))
    with pytest.raises(ValueError):
        plan_recovery_index_migration(scene["complete"], other)


# ---------------------------------------------------------------------------
# type and value validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    index = scene["complete"]
    for value in (None, 1, b"x", [], {}):
        with pytest.raises(TypeError):
            plan_recovery_index_migration(value, index)
        with pytest.raises(TypeError):
            plan_recovery_index_migration(index, value)


def test_malformed_and_non_canonical_indexes_rejected(scene):
    index = scene["complete"]
    for bad in ("", "not json", "{", "[]", "null", " " + index,
                json.dumps(json.loads(index), indent=2)):
        with pytest.raises(ValueError):
            plan_recovery_index_migration(bad, scene["empty"])
        with pytest.raises(ValueError):
            plan_recovery_index_migration(scene["empty"], bad)


def test_broken_derived_fields_rejected(scene):
    for key in ("audit", "snapshot", "complete"):
        document = json.loads(scene["complete"])
        if key == "audit":
            document["audit"] = document["audit"][1:]
        elif key == "snapshot":
            document["snapshot"] = None
        else:
            document["complete"] = not document["complete"]
        with pytest.raises(ValueError):
            plan_recovery_index_migration(_canonical(document),
                                          scene["empty"])
    document = json.loads(scene["incomplete"])
    document["resume"] = None
    with pytest.raises(ValueError):
        plan_recovery_index_migration(_canonical(document), scene["empty"])


def test_duplicate_batch_id_inside_index_rejected(scene):
    document = json.loads(scene["complete"])
    document["entries"].append(document["entries"][1])
    with pytest.raises(ValueError):
        plan_recovery_index_migration(_canonical(document), scene["empty"])


def test_broken_entry_chain_rejected(scene):
    document = json.loads(scene["complete"])
    document["entries"].reverse()
    with pytest.raises(ValueError):
        plan_recovery_index_migration(_canonical(document), scene["empty"])


# ---------------------------------------------------------------------------
# repeatability and input preservation
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    before, after = scene["incomplete"], scene["complete"]
    saved_before, saved_after = before, after
    one = plan_recovery_index_migration(before, after)
    two = plan_recovery_index_migration(before, after)
    assert one == two
    assert before == saved_before and after == saved_after
    three = plan_recovery_index_migration(after, before)
    assert plan_recovery_index_migration(after, before) == three
    assert not one.endswith("\n")
