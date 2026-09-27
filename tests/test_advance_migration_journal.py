"""Tests for :func:`lidar_scan.advance_migration_journal`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (advance_migration_journal,
                        build_recovery_history,
                        commit_migration_checkpoint,
                        execute_recovery_index_migration,
                        merge_checkout_logs, merge_recovery_indexes,
                        plan_checkout_recovery,
                        plan_recovery_index_migration, update_recovery_index)
from lidar_scan import tiles as tiles_module
import test_build_recovery_history as recovery_fixtures


def _states(plan, count):
    return recovery_fixtures._states(plan, count)


def _summary(plan, states):
    return recovery_fixtures._summary(plan, states)


def _completed_history(plan, units):
    states = _states(plan, units)
    return build_recovery_history(
        plan, tuple(_summary(plan, states[:i + 1])
                    for i in range(units)))


def _index(pairs):
    return merge_recovery_indexes((update_recovery_index(None, pairs),))


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p01 = plan_checkout_recovery(log, None, ledgers[:2])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    ledger_alt = ledgers[1].replace('"b1"', '"b1a"')
    p1_alt = plan_checkout_recovery(
        log, checkpoint_one, ledgers[:1] + (ledger_alt,))

    h0 = _completed_history(p0, 1)
    h01 = _completed_history(p01, 2)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)

    before = _index(((p0, h0), (p12, h12)))
    after = _index(((p0, h0), (p1_alt, h1_alt)))
    plan = plan_recovery_index_migration(before, after)

    return {
        "empty": merge_recovery_indexes(()),
        "before": before,
        "after": after,
        "plan": plan,
        "fork_left": _index(((p01, h01),)),
        "fork_right": _index(((p0, h0), (p1_alt, h1_alt))),
        "steps": [["rollback", "b2", "applied"],
                  ["rollback", "b1", "applied"],
                  ["pending", "b1a", "applied"]],
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.advance_migration_journal \
        is advance_migration_journal
    assert "advance_migration_journal" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# first commits
# ---------------------------------------------------------------------------

def test_first_zero_step_commit_records_initial_state(scene):
    text = advance_migration_journal(
        scene["before"], scene["after"], max_steps=0)
    document = json.loads(text)
    assert list(document) == [
        "plan", "states", "index", "complete", "receipt"]
    assert document["plan"] == json.loads(scene["plan"])
    assert document["index"] == json.loads(scene["before"])
    assert document["complete"] is False
    assert document["receipt"] is None
    assert len(document["states"]) == 1
    state = document["states"][0]
    assert list(state) == [
        "status", "confirmed", "steps", "index", "complete"]
    assert state == {
        "status": "pending",
        "confirmed": 0,
        "steps": [],
        "index": json.loads(scene["before"]),
        "complete": False}
    assert " " not in text and not text.endswith("\n")


def test_first_commit_advances_at_most_max_steps(scene):
    text = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    document = json.loads(text)
    assert document["complete"] is False
    assert document["receipt"] is None
    assert document["index"] == json.loads(scene["before"])
    assert len(document["states"]) == 1
    state = document["states"][0]
    assert state["status"] == "pending"
    assert state["confirmed"] == 1
    assert state["steps"] == scene["steps"][:1]


def test_none_max_steps_runs_the_whole_migration(scene):
    text = advance_migration_journal(scene["before"], scene["after"])
    document = json.loads(text)
    assert document["complete"] is True
    assert document["receipt"] == [2, 1, 3]
    assert document["index"] == json.loads(scene["after"])
    assert len(document["states"]) == 1
    state = document["states"][0]
    assert state["status"] == "applied"
    assert state["confirmed"] == 3
    assert state["steps"] == scene["steps"]
    assert state["index"] == json.loads(scene["after"])


def test_zero_step_plan_first_commit_is_complete(scene):
    plan = plan_recovery_index_migration(scene["after"], scene["after"])
    text = advance_migration_journal(scene["after"], scene["after"])
    document = json.loads(text)
    assert document["plan"] == json.loads(plan)
    assert document["complete"] is True
    assert document["receipt"] == [0, 0, 0]
    assert document["index"] == json.loads(scene["after"])
    assert len(document["states"]) == 1
    state = document["states"][0]
    assert state == {
        "status": "unchanged",
        "confirmed": 0,
        "steps": [],
        "index": json.loads(scene["after"]),
        "complete": True}


def test_zero_step_plan_with_zero_bound_completes(scene):
    text = advance_migration_journal(
        scene["after"], scene["after"], max_steps=0)
    document = json.loads(text)
    assert document["complete"] is True
    assert document["receipt"] == [0, 0, 0]


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------

def test_continuation_resumes_from_last_state(scene):
    first = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    second = advance_migration_journal(
        scene["before"], scene["after"], journal=first, max_steps=1)
    third = advance_migration_journal(
        scene["before"], scene["after"], journal=second)
    for journal, confirmed in ((first, 1), (second, 2), (third, 3)):
        document = json.loads(journal)
        assert document["states"][-1]["confirmed"] == confirmed
    assert [len(json.loads(j)["states"]) for j in (first, second, third)] \
        == [1, 2, 3]
    final = json.loads(third)
    assert final["complete"] is True
    assert final["receipt"] == [2, 1, 3]
    assert final["index"] == json.loads(scene["after"])
    assert [state["steps"] for state in final["states"]] == [
        scene["steps"][:1], scene["steps"][:2], scene["steps"]]


def test_zero_bound_with_journal_appends_nothing(scene):
    first = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    again = advance_migration_journal(
        scene["before"], scene["after"], journal=first, max_steps=0)
    assert again == first


def test_completed_journal_returned_byte_identical(scene):
    journal = advance_migration_journal(scene["before"], scene["after"])
    assert advance_migration_journal(
        scene["before"], scene["after"], journal=journal) == journal
    assert advance_migration_journal(
        scene["before"], scene["after"], journal=journal,
        max_steps=0) == journal
    assert advance_migration_journal(
        scene["before"], scene["after"], journal=journal,
        max_steps=5) == journal


def test_continues_a_checkpoint_style_journal(scene):
    state = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"],
        scene["before"], max_steps=1)
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"],
        scene["before"], state)
    advanced = advance_migration_journal(
        scene["before"], scene["after"], journal=journal, max_steps=1)
    document = json.loads(advanced)
    assert len(document["states"]) == 2
    assert document["states"][-1]["confirmed"] == 2
    assert document["complete"] is False


def test_step_bound_cannot_overshoot_into_another_state(scene):
    first = advance_migration_journal(
        scene["before"], scene["after"], max_steps=2)
    second = advance_migration_journal(
        scene["before"], scene["after"], journal=first, max_steps=99)
    document = json.loads(second)
    assert document["complete"] is True
    assert len(document["states"]) == 2


# ---------------------------------------------------------------------------
# canonical form and input preservation
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    args = (scene["before"], scene["after"])
    saved = (scene["before"], scene["after"], None, None)
    one = advance_migration_journal(*args)
    two = advance_migration_journal(*args)
    assert one == two
    assert (scene["before"], scene["after"], None, None) == saved


def test_compact_non_ascii_encoding(scene):
    text = advance_migration_journal(scene["before"], scene["after"])
    assert " " not in text and "\n" not in text
    json.loads(text)


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_index_arguments_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        advance_migration_journal(bad, scene["after"])
    with pytest.raises(TypeError):
        advance_migration_journal(scene["before"], bad)


@pytest.mark.parametrize("bad", [1, 1.5, [], b"x", {}])
def test_journal_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        advance_migration_journal(
            scene["before"], scene["after"], journal=bad)


def test_none_journal_accepted(scene):
    advance_migration_journal(
        scene["before"], scene["after"], journal=None)


@pytest.mark.parametrize("bad", [1.5, "1", [], b"1", {}])
def test_max_steps_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        advance_migration_journal(
            scene["before"], scene["after"], max_steps=bad)


def test_bool_max_steps_rejected(scene):
    with pytest.raises(TypeError):
        advance_migration_journal(
            scene["before"], scene["after"], max_steps=True)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_negative_max_steps_rejected(scene):
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"], max_steps=-1)


def test_malformed_indexes_rejected(scene):
    with pytest.raises(ValueError):
        advance_migration_journal("{", scene["after"])
    with pytest.raises(ValueError):
        advance_migration_journal(scene["before"] + " ", scene["after"])


def test_journal_must_match_recomputed_plan(scene):
    other = advance_migration_journal(scene["after"], scene["after"])
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"], journal=other)


def test_malformed_journal_rejected(scene):
    journal = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"], journal=journal + " ")
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"], journal="{")


def test_non_canonical_journal_rejected(scene):
    journal = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"],
            journal=json.dumps(json.loads(journal), indent=2))


def test_journal_plan_prefix_contradiction_rejected(scene):
    journal = advance_migration_journal(
        scene["before"], scene["after"], max_steps=1)
    document = json.loads(journal)
    document["plan"]["rollback"] = []
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["before"], scene["after"],
            journal=_canonical(document))


def test_forked_indexes_rejected(scene):
    # batch ids on both sides of the boundary intersect: no plan exists
    with pytest.raises(ValueError):
        advance_migration_journal(
            scene["fork_left"], scene["fork_right"])
