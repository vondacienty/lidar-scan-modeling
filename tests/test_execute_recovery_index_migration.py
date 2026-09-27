"""Tests for :func:`lidar_scan.execute_recovery_index_migration`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, execute_recovery_index_migration,
                        merge_checkout_logs, merge_recovery_indexes,
                        plan_checkout_recovery, plan_recovery_index_migration,
                        update_recovery_index)
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
    p01 = plan_checkout_recovery(log, None, ledgers[:2])

    h0 = _completed_history(p0, 1)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)
    h01 = _completed_history(p01, 2)

    states = _states(p01, 1)
    h01_partial = build_recovery_history(
        p01, (_summary(p01, states),))

    before = _index(((p0, h0), (p12, h12)))
    after = _index(((p0, h0), (p1_alt, h1_alt)))
    plan = plan_recovery_index_migration(before, after)
    expected_steps = [["rollback", "b2", "applied"],
                      ["rollback", "b1", "applied"],
                      ["pending", "b1a", "applied"]]

    return {
        "snaps": snaps,
        "empty": merge_recovery_indexes(()),
        "before": before,
        "after": after,
        "plan": plan,
        "steps": expected_steps,
        "p01_done": _index(((p01, h01),)),
        "p01_partial": _index(((p01, h01_partial),)),
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.execute_recovery_index_migration \
        is execute_recovery_index_migration
    assert "execute_recovery_index_migration" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# one-shot application
# ---------------------------------------------------------------------------

def test_apply_all_steps_at_once(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    document = json.loads(text)
    assert list(document) == [
        "status", "confirmed", "steps", "index", "complete"]
    assert document["status"] == "applied"
    assert document["steps"] == scene["steps"]
    assert document["confirmed"] == 3
    assert document["index"] == json.loads(scene["after"])
    assert document["complete"] is True
    assert " " not in text and not text.endswith("\n")


def test_steps_rendered_as_three_item_arrays(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    assert text.count('["rollback",') == 2
    assert text.count('["pending",') == 1


def test_already_at_after_is_unchanged(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"])
    document = json.loads(text)
    assert document["status"] == "unchanged"
    assert document["steps"] == scene["steps"]
    assert document["confirmed"] == 3
    assert document["index"] == json.loads(scene["after"])
    assert document["complete"] is True


def test_empty_migration_between_identical_indexes(scene):
    after = scene["after"]
    plan = plan_recovery_index_migration(after, after)
    text = execute_recovery_index_migration(after, after, plan, after)
    assert json.loads(text) == {
        "status": "unchanged",
        "confirmed": 0,
        "steps": [],
        "index": json.loads(after),
        "complete": True,
    }
    assert list(json.loads(text)) == [
        "status", "confirmed", "steps", "index", "complete"]


def test_empty_indexes_migration(scene):
    empty = scene["empty"]
    plan = plan_recovery_index_migration(empty, empty)
    text = execute_recovery_index_migration(empty, empty, plan, empty)
    document = json.loads(text)
    assert document["status"] == "unchanged"
    assert document["steps"] == [] and document["confirmed"] == 0
    assert document["index"] == json.loads(empty)
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# stepwise confirmation
# ---------------------------------------------------------------------------

def test_partial_run_stays_pending_and_embeds_before(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    document = json.loads(text)
    assert document["status"] == "pending"
    assert document["steps"] == [scene["steps"][0]]
    assert document["confirmed"] == 1
    assert document["index"] == json.loads(scene["before"])
    assert document["complete"] is False


def test_steps_are_the_cumulative_prefix(scene):
    s1 = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    s2 = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        state=s1, max_steps=1)
    assert json.loads(s2)["steps"] == scene["steps"][:2]
    assert json.loads(s2)["confirmed"] == 2
    assert json.loads(s2)["index"] == json.loads(scene["before"])
    s3 = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        state=s2)
    document = json.loads(s3)
    assert document["status"] == "applied"
    assert document["steps"] == scene["steps"]
    assert document["confirmed"] == 3
    assert document["index"] == json.loads(scene["after"])
    assert document["complete"] is True


def test_completing_call_embeds_after_even_within_max_steps(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=100)
    document = json.loads(text)
    assert document["status"] == "applied"
    assert document["index"] == json.loads(scene["after"])


def test_max_steps_zero_confirms_nothing(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=0)
    document = json.loads(text)
    assert document["status"] == "pending"
    assert document["steps"] == []
    assert document["confirmed"] == 0
    assert document["index"] == json.loads(scene["before"])
    assert document["complete"] is False


def test_zero_step_pending_state_re_enters_byte_identical(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=0)
    again = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        state=text, max_steps=0)
    assert again == text


def test_pending_state_re_enters_byte_identical(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=2)
    again = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        state=text, max_steps=0)
    assert again == text


def test_completed_state_re_enters_byte_identical(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    again = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        state=text)
    assert again == text


def test_unchanged_state_re_enters_byte_identical(scene):
    text = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"])
    again = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"],
        state=text)
    assert again == text


def test_full_application_then_unchanged_have_same_prefix_and_index(scene):
    applied = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    unchanged = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["after"])
    applied_doc = json.loads(applied)
    unchanged_doc = json.loads(unchanged)
    assert applied_doc["steps"] == unchanged_doc["steps"]
    assert applied_doc["confirmed"] == unchanged_doc["confirmed"] == 3
    assert applied_doc["index"] == unchanged_doc["index"]
    assert applied_doc["status"] == "applied"
    assert unchanged_doc["status"] == "unchanged"


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_string_arguments_wrong_type(scene, bad):
    before, after, plan, current = (
        scene["before"], scene["after"], scene["plan"], scene["before"])
    with pytest.raises(TypeError):
        execute_recovery_index_migration(bad, after, plan, current)
    with pytest.raises(TypeError):
        execute_recovery_index_migration(before, bad, plan, current)
    with pytest.raises(TypeError):
        execute_recovery_index_migration(before, after, bad, current)
    with pytest.raises(TypeError):
        execute_recovery_index_migration(before, after, plan, bad)


@pytest.mark.parametrize("bad", [1, 1.5, [], b"x", {}])
def test_state_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", []])
def test_max_steps_wrong_type(scene, bad):
    with pytest.raises(TypeError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            max_steps=bad)


def test_none_arguments_accepted(scene):
    execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        state=None, max_steps=None)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_max_steps_negative_is_value_error(scene):
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            max_steps=-1)


def test_plan_must_match_before_after(scene):
    other = plan_recovery_index_migration(scene["after"], scene["after"])
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], other, scene["before"])


def test_malformed_documents_rejected(scene):
    before, after, plan = (
        scene["before"], scene["after"], scene["plan"])
    with pytest.raises(ValueError):
        execute_recovery_index_migration("{", after, plan, before)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan, before + " ")
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan,
            json.dumps(json.loads(before), indent=2))
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            before, after, plan + " ", before)


def test_current_must_equal_before_or_after_when_starting(scene):
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["empty"])


def test_unfinished_state_requires_current_before(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["after"],
            state=pending)


def test_completed_state_requires_current_after(scene):
    applied = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=applied)


def test_state_must_be_a_confirmed_prefix(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    document = json.loads(pending)
    document["confirmed"] = 2
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=_canonical(document))


def test_state_steps_must_match_plan_steps(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    document = json.loads(pending)
    document["steps"][0][1] = "b9"
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=_canonical(document))


def test_state_embedding_wrong_index_rejected(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    document = json.loads(pending)
    document["index"] = json.loads(scene["after"])
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=_canonical(document))


def test_state_flags_must_agree(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    document = json.loads(pending)
    document["complete"] = True
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=_canonical(document))
    applied = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"])
    document = json.loads(applied)
    document["status"] = "pending"
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["after"],
            state=_canonical(document))


def test_non_canonical_state_rejected(scene):
    pending = execute_recovery_index_migration(
        scene["before"], scene["after"], scene["plan"], scene["before"],
        max_steps=1)
    pretty = json.dumps(json.loads(pending), indent=2)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=pretty)


def test_state_from_another_plan_rejected(scene):
    other_plan = plan_recovery_index_migration(
        scene["p01_partial"], scene["p01_done"])
    foreign = execute_recovery_index_migration(
        scene["p01_partial"], scene["p01_done"], other_plan,
        scene["p01_partial"], max_steps=1)
    with pytest.raises(ValueError):
        execute_recovery_index_migration(
            scene["before"], scene["after"], scene["plan"], scene["before"],
            state=foreign)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    before, after, plan, current = (
        scene["before"], scene["after"], scene["plan"], scene["before"])
    saved = (before, after, plan, current)
    one = execute_recovery_index_migration(before, after, plan, current)
    two = execute_recovery_index_migration(before, after, plan, current)
    assert one == two
    assert (before, after, plan, current) == saved
