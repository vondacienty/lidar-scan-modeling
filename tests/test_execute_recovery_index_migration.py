"""Tests for :func:`lidar_scan.execute_recovery_index_migration`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        merge_recovery_indexes, plan_checkout_recovery,
                        plan_recovery_index_migration,
                        execute_recovery_index_migration,
                        update_recovery_index)
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


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p01 = plan_checkout_recovery(log, None, ledgers[:2])
    p1 = plan_checkout_recovery(log, checkpoint_one, ledgers[:2])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    ledger_alt = ledgers[1].replace('"b1"', '"b1a"')
    p1_alt = plan_checkout_recovery(
        log, checkpoint_one, ledgers[:1] + (ledger_alt,))

    h0 = _completed_history(p0, 1)
    h01 = _completed_history(p01, 2)
    h1 = _completed_history(p1, 1)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)

    states = _states(p01, 1)
    h01_partial = build_recovery_history(
        p01, (_summary(p01, states),))

    empty = merge_recovery_indexes(())
    p0_done = _index(((p0, h0),))
    p01_done = _index(((p01, h01),))
    p01_partial = _index(((p01, h01_partial),))
    p0_then_p1 = _index(((p0, h0), (p1, h1)))
    p0_then_p12 = _index(((p0, h0), (p12, h12)))
    p0_then_alt = _index(((p0, h0), (p1_alt, h1_alt)))

    return {
        "snaps": snaps,
        "empty": empty,
        "p0_done": p0_done,
        "p01_done": p01_done,
        "p01_partial": p01_partial,
        "p0_then_p1": p0_then_p1,
        "p0_then_p12": p0_then_p12,
        "p0_then_alt": p0_then_alt,
    }


def _run(scene, before_key, after_key, current_key=None, **kwargs):
    before = scene[before_key]
    after = scene[after_key]
    plan = plan_recovery_index_migration(before, after)
    current = scene[current_key] if current_key else before
    return execute_recovery_index_migration(
        before, after, plan, current, **kwargs), plan


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.execute_recovery_index_migration \
        is execute_recovery_index_migration
    assert "execute_recovery_index_migration" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_document_keys_and_compact_encoding(scene):
    text, _plan = _run(scene, "p0_then_p12", "p0_then_alt",
                       max_steps=1)
    assert list(json.loads(text)) == [
        "status", "confirmed", "steps", "index", "complete"]
    assert " " not in text and not text.endswith("\n")
    assert text == json.dumps(json.loads(text), ensure_ascii=False,
                              separators=(",", ":"))


# ---------------------------------------------------------------------------
# unchanged
# ---------------------------------------------------------------------------

def test_current_after_without_state_is_unchanged(scene):
    text, plan = _run(scene, "p0_then_p12", "p0_then_alt",
                      current_key="p0_then_alt")
    document = json.loads(text)
    planned = json.loads(plan)
    expected_steps = [["rollback", bid, status]
                      for bid, status in planned["rollback"]]
    expected_steps += [["pending", bid, status]
                       for bid, status in planned["pending"]]
    assert document["status"] == "unchanged"
    assert document["confirmed"] == len(expected_steps)
    assert document["steps"] == expected_steps
    assert document["index"] == json.loads(scene["p0_then_alt"])
    assert document["complete"] is True


def test_identical_indexes_unchanged_with_empty_steps(scene):
    text, _plan = _run(scene, "p0_done", "p0_done",
                       current_key="p0_done")
    document = json.loads(text)
    assert document == {
        "status": "unchanged",
        "confirmed": 0,
        "steps": [],
        "index": json.loads(scene["p0_done"]),
        "complete": True,
    }


def test_both_empty_indexes_unchanged(scene):
    text, _plan = _run(scene, "empty", "empty", current_key="empty")
    document = json.loads(text)
    assert document["status"] == "unchanged"
    assert document["confirmed"] == 0
    assert document["steps"] == []
    assert document["index"] == json.loads(scene["empty"])
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# full application
# ---------------------------------------------------------------------------

def test_full_application_lists_rollback_then_pending(scene):
    text, plan = _run(scene, "p0_then_p12", "p0_then_alt")
    document = json.loads(text)
    planned = json.loads(plan)
    assert document["status"] == "applied"
    assert document["steps"] == (
        [["rollback", bid, status]
         for bid, status in planned["rollback"]]
        + [["pending", bid, status]
           for bid, status in planned["pending"]])
    assert document["confirmed"] == len(document["steps"]) == 3
    assert document["index"] == json.loads(scene["p0_then_alt"])
    assert document["complete"] is True


def test_pending_only_migration(scene):
    text, _plan = _run(scene, "empty", "p0_done")
    document = json.loads(text)
    assert document["status"] == "applied"
    assert document["steps"] == [["pending", "b0", "applied"]]
    assert document["confirmed"] == 1
    assert document["complete"] is True
    assert document["index"] == json.loads(scene["p0_done"])


def test_rollback_only_migration(scene):
    text, _plan = _run(scene, "p0_done", "empty")
    document = json.loads(text)
    assert document["status"] == "applied"
    assert document["steps"] == [["rollback", "b0", "applied"]]
    assert document["confirmed"] == 1
    assert document["complete"] is True
    assert document["index"] == json.loads(scene["empty"])


def test_applied_into_unfinished_after_keeps_embedded_resume(scene):
    text, _plan = _run(scene, "empty", "p01_partial")
    document = json.loads(text)
    assert document["status"] == "applied"
    assert document["steps"] == [["pending", "b0", "applied"]]
    assert document["index"] == json.loads(scene["p01_partial"])
    assert document["complete"] is True
    assert document["index"]["complete"] is False
    assert document["index"]["resume"] is not None


def test_index_embedded_byte_for_byte(scene):
    text, _plan = _run(scene, "p0_then_p12", "p0_then_alt",
                       max_steps=1)
    assert ',"index":' + scene["p0_then_p12"] + ',"complete":' in text
    full, _plan = _run(scene, "p0_then_p12", "p0_then_alt")
    assert ',"index":' + scene["p0_then_alt"] + ',"complete":' in full


# ---------------------------------------------------------------------------
# stepwise execution with state and max_steps
# ---------------------------------------------------------------------------

def test_stepwise_execution_through_state(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)

    first = execute_recovery_index_migration(
        before, after, plan, before, None, 1)
    first_document = json.loads(first)
    assert first_document["status"] == "pending"
    assert first_document["confirmed"] == 1
    assert first_document["steps"] == [["rollback", "b2", "applied"]]
    assert first_document["index"] == json.loads(before)
    assert first_document["complete"] is False

    second = execute_recovery_index_migration(
        before, after, plan, before, first, 1)
    second_document = json.loads(second)
    assert second_document["status"] == "pending"
    assert second_document["confirmed"] == 2
    assert second_document["steps"] == [
        ["rollback", "b2", "applied"],
        ["rollback", "b1", "applied"]]
    assert second_document["index"] == json.loads(before)
    assert second_document["complete"] is False

    finished = execute_recovery_index_migration(
        before, after, plan, before, second, 1)
    finished_document = json.loads(finished)
    assert finished_document["status"] == "applied"
    assert finished_document["confirmed"] == 3
    assert finished_document["steps"] == [
        ["rollback", "b2", "applied"],
        ["rollback", "b1", "applied"],
        ["pending", "b1a", "applied"]]
    assert finished_document["index"] == json.loads(after)
    assert finished_document["complete"] is True

    # overshooting the remainder and no limit both finish identically
    assert execute_recovery_index_migration(
        before, after, plan, before, second, 99) == finished
    assert execute_recovery_index_migration(
        before, after, plan, before, second) == finished


def test_zero_steps_confirms_nothing(scene):
    text, _plan = _run(scene, "p0_then_p12", "p0_then_alt",
                       max_steps=0)
    document = json.loads(text)
    assert document["status"] == "pending"
    assert document["confirmed"] == 0
    assert document["steps"] == []
    assert document["index"] == json.loads(scene["p0_then_p12"])
    assert document["complete"] is False


def test_state_resumes_from_prefix(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    first = execute_recovery_index_migration(
        before, after, plan, before, None, 2)
    finished_one_shot = execute_recovery_index_migration(
        before, after, plan, before)
    finished_resumed = execute_recovery_index_migration(
        before, after, plan, before, first)
    assert finished_resumed == finished_one_shot


# ---------------------------------------------------------------------------
# completed-state reentry
# ---------------------------------------------------------------------------

def test_completed_applied_state_reenters_byte_identical(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    finished = execute_recovery_index_migration(
        before, after, plan, before)
    assert execute_recovery_index_migration(
        before, after, plan, after, finished) == finished
    assert execute_recovery_index_migration(
        before, after, plan, after, finished, 0) == finished
    assert execute_recovery_index_migration(
        before, after, plan, after, finished, 5) == finished


def test_unchanged_state_reenters_byte_identical(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    unchanged = execute_recovery_index_migration(
        before, after, plan, after)
    assert execute_recovery_index_migration(
        before, after, plan, after, unchanged) == unchanged
    assert execute_recovery_index_migration(
        before, after, plan, after, unchanged, 2) == unchanged


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    before = scene["p0_done"]
    after = scene["p0_then_p1"]
    plan = plan_recovery_index_migration(before, after)
    for value in (None, 1, [], b"x", {}):
        with pytest.raises(TypeError):
            execute_recovery_index_migration(value, after, plan, before)
        with pytest.raises(TypeError):
            execute_recovery_index_migration(before, value, plan, before)
        with pytest.raises(TypeError):
            execute_recovery_index_migration(before, after, value, before)
        with pytest.raises(TypeError):
            execute_recovery_index_migration(before, after, plan, value)
    for value in (1, b"x", [], {}):
        with pytest.raises(TypeError):
            execute_recovery_index_migration(
                before, after, plan, before, state=value)
    for value in (True, False, 1.0, "1", [], ()):
        with pytest.raises(TypeError):
            execute_recovery_index_migration(
                before, after, plan, before, max_steps=value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_negative_max_steps_rejected(scene):
    before = scene["p0_done"]
    after = scene["p0_then_p1"]
    plan = plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before, max_steps=-1)


def test_plan_mismatch_rejected(scene):
    before = scene["p0_done"]
    after = scene["p0_then_p1"]
    plan = plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, scene["p0_then_alt"], plan, before)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan + " ", before)


def test_malformed_documents_rejected(scene):
    before = scene["p0_done"]
    after = scene["p0_then_p1"]
    plan = plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        execute_recovery_index_migration("{", after, plan, before)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(before, "{", plan, before)
    pretty = json.dumps(json.loads(before), indent=2)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(before, after, plan, pretty)


def test_current_neither_before_nor_after_rejected(scene):
    before = scene["p0_done"]
    after = scene["p0_then_p1"]
    plan = plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, scene["p01_done"])


def test_malformed_or_non_canonical_state_rejected(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(before, after, plan, before, "{")
    state = execute_recovery_index_migration(
        before, after, plan, before, None, 1)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before,
            json.dumps(json.loads(state), indent=2))
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before, state + " ")


def test_tampered_state_rejected(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    state = execute_recovery_index_migration(
        before, after, plan, before, None, 1)
    tampered = state.replace('"b2"', '"b1"', 1)
    assert tampered != state
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before, tampered)


def test_state_from_another_plan_rejected(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    other_plan = plan_recovery_index_migration(
        scene["p0_done"], scene["empty"])
    foreign = execute_recovery_index_migration(
        scene["p0_done"], scene["empty"], other_plan,
        scene["p0_done"], None, 0)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before, foreign)


def test_pending_state_requires_current_before(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    pending = execute_recovery_index_migration(
        before, after, plan, before, None, 1)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, after, pending)


def test_completed_state_requires_current_after(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    finished = execute_recovery_index_migration(
        before, after, plan, before)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before, finished)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    before = scene["p0_then_p12"]
    after = scene["p0_then_alt"]
    plan = plan_recovery_index_migration(before, after)
    saved = (before, after, plan)
    one = execute_recovery_index_migration(before, after, plan, before,
                                           None, 1)
    two = execute_recovery_index_migration(before, after, plan, before,
                                           None, 1)
    assert one == two
    assert (before, after, plan) == saved
