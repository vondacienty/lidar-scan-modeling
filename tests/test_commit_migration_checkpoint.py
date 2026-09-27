"""Tests for :func:`lidar_scan.commit_migration_checkpoint`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history,
                        commit_migration_checkpoint,
                        execute_recovery_index_migration,
                        merge_checkout_logs, merge_recovery_indexes,
                        plan_checkout_recovery,
                        plan_recovery_index_migration,
                        update_migration_checkpoint, update_recovery_index)
from lidar_scan import tiles as tiles_module
import test_build_recovery_history as recovery_fixtures


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


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


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    ledger_alt = ledgers[1].replace('"b1"', '"b1a"')
    p1_alt = plan_checkout_recovery(
        log, checkpoint_one, ledgers[:1] + (ledger_alt,))

    h0 = _completed_history(p0, 1)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)

    before = _index(((p0, h0), (p12, h12)))
    after = _index(((p0, h0), (p1_alt, h1_alt)))
    plan = plan_recovery_index_migration(before, after)

    s1 = execute_recovery_index_migration(
        before, after, plan, before, max_steps=1)
    s2 = execute_recovery_index_migration(
        before, after, plan, before, state=s1, max_steps=1)
    s3 = execute_recovery_index_migration(
        before, after, plan, before, state=s2)

    return {
        "snaps": snaps,
        "empty": merge_recovery_indexes(()),
        "before": before,
        "after": after,
        "plan": plan,
        "steps": [["rollback", "b2", "applied"],
                  ["rollback", "b1", "applied"],
                  ["pending", "b1a", "applied"]],
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.commit_migration_checkpoint \
        is commit_migration_checkpoint
    assert "commit_migration_checkpoint" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# first commit with no journal
# ---------------------------------------------------------------------------

def test_first_commit_unfinished(scene):
    text = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    document = json.loads(text)
    assert list(document) == [
        "plan", "states", "index", "complete", "receipt"]
    assert document["plan"] == json.loads(scene["plan"])
    assert document["states"] == [json.loads(scene["s1"])]
    assert document["index"] == json.loads(scene["before"])
    assert document["complete"] is False
    assert document["receipt"] is None
    assert " " not in text and not text.endswith("\n")


def test_first_commit_completed_carries_recomputed_receipt(scene):
    text = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        scene["s3"])
    document = json.loads(text)
    assert document["states"] == [json.loads(scene["s3"])]
    assert document["index"] == json.loads(scene["after"])
    assert document["complete"] is True
    assert document["receipt"] == [2, 1, 3]
    assert list(document) == [
        "plan", "states", "index", "complete", "receipt"]


def test_zero_step_migration_receipt(scene):
    plan = plan_recovery_index_migration(
        scene["after"], scene["after"])
    state = execute_recovery_index_migration(
        scene["after"], scene["after"], plan, scene["after"])
    text = commit_migration_checkpoint(
        scene["after"], scene["after"], plan, scene["after"], state)
    document = json.loads(text)
    assert document["complete"] is True
    assert document["receipt"] == [0, 0, 0]


# ---------------------------------------------------------------------------
# accumulation and idempotence
# ---------------------------------------------------------------------------

def test_states_accumulate_in_order(scene):
    j1 = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    j2 = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s2"], journal=j1)
    j3 = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        scene["s3"], journal=j2)
    assert [len(json.loads(j)["states"]) for j in (j1, j2, j3)] == [1, 2, 3]
    document = json.loads(j3)
    assert document["states"] == [
        json.loads(scene["s1"]), json.loads(scene["s2"]),
        json.loads(scene["s3"])]
    assert document["receipt"] == [2, 1, 3]


def test_re_entering_last_state_is_byte_identical(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    again = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"], journal=journal)
    assert again == journal
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s2"], journal=journal)
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        scene["s3"], journal=journal)
    assert commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        scene["s3"], journal=journal) == journal


def test_repeatable_and_inputs_unchanged(scene):
    args = (scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s1"])
    saved = tuple(args) + (None,)
    one = commit_migration_checkpoint(*args)
    two = commit_migration_checkpoint(*args)
    assert one == two
    assert tuple(args) + (None,) == saved


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_string_arguments_wrong_type(scene, bad):
    before, after, plan, current, state = (
        scene["before"], scene["after"], scene["plan"],
        scene["before"], scene["s1"])
    with pytest.raises(TypeError):
        commit_migration_checkpoint(bad, after, plan, current, state)
    with pytest.raises(TypeError):
        commit_migration_checkpoint(before, bad, plan, current, state)
    with pytest.raises(TypeError):
        commit_migration_checkpoint(before, after, bad, current, state)
    with pytest.raises(TypeError):
        commit_migration_checkpoint(before, after, plan, bad, state)
    with pytest.raises(TypeError):
        commit_migration_checkpoint(before, after, plan, current, bad)


@pytest.mark.parametrize("bad", [1, 1.5, [], b"x", {}])
def test_journal_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s1"], journal=bad)


def test_none_journal_accepted(scene):
    commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"], journal=None)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_plan_must_match_before_after(scene):
    other = plan_recovery_index_migration(scene["after"], scene["after"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], other, scene["before"],
            scene["s1"])


def test_current_must_equal_terminal_index(scene):
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"], scene["after"],
            scene["s1"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s3"])


def test_malformed_documents_rejected(scene):
    before, after, plan = (
        scene["before"], scene["after"], scene["plan"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            "{", after, plan, before, scene["s1"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            before, after, plan + " ", before, scene["s1"])


def test_state_must_continue_journal_with_strict_increase(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s2"])
    # rollback to an earlier confirmed prefix
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s1"], journal=journal)


def test_appending_after_completion_rejected(scene):
    journal = None
    for state, current in ((scene["s1"], scene["before"]),
                           (scene["s2"], scene["before"]),
                           (scene["s3"], scene["after"])):
        journal = commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"], current,
            state, journal=journal)
    unchanged = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"])
    assert unchanged != scene["s3"]
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"], scene["after"],
            unchanged, journal=journal)


def test_malformed_journal_rejected(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal=journal + " ")
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal="{")


def test_plain_checkpoint_is_not_a_journal(scene):
    checkpoint = update_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], None, scene["s1"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal=checkpoint)


def test_journal_from_another_plan_rejected(scene):
    other_plan = plan_recovery_index_migration(
        scene["after"], scene["after"])
    other_state = execute_recovery_index_migration(
        scene["after"], scene["after"], other_plan, scene["after"])
    foreign = commit_migration_checkpoint(
        scene["after"], scene["after"], other_plan, scene["after"],
        other_state)
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal=foreign)


def test_journal_receipt_must_match_plan(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        scene["s3"])
    document = json.loads(journal)
    document["receipt"] = [1, 1, 3]
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"], scene["after"],
            scene["s3"], journal=_canonical(document))


def test_unfinished_journal_receipt_must_be_null(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    document = json.loads(journal)
    document["receipt"] = [0, 0, 0]
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal=_canonical(document))


def test_non_canonical_journal_rejected(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"],
            journal=json.dumps(json.loads(journal), indent=2))


def test_journal_duplicate_confirmed_prefix_rejected(scene):
    journal = commit_migration_checkpoint(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        scene["s1"])
    document = json.loads(journal)
    document["states"].append(json.loads(scene["s1"]))
    with pytest.raises(ValueError):
        commit_migration_checkpoint(
            scene["before"], scene["after"], scene["plan"],
            scene["before"], scene["s2"], journal=_canonical(document))
