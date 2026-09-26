"""Tests for :func:`lidar_scan.update_recovery_index`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, build_recovery_receipt,
                        checkout_log, delivery_log, execute_checkout_recovery,
                        merge_checkout_logs, merge_recovery_receipts,
                        plan_checkout_recovery, plan_delivery_checkout,
                        plan_delivery_updates, update_recovery_index)
from lidar_scan import tiles as tiles_module
from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, merge_delivery_manifests,
                        merge_delivery_receipts)

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

EMPTY_CHANGES = '{"changes":[],"releasable":true}'
EMPTY_RECEIPTS = '{"products":[],"ready":true}'


def _empty_snapshot():
    return build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)


def _ready_p(batches):
    """Product ``p`` with the given (batch, version) pairs, all passing."""
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    changes = merge_delivery_manifests(
        tuple(build_delivery_manifest((item,)) for item in items))
    plan = build_delivery_plan(audit_delivery_changes(changes))
    receipts = merge_delivery_receipts(
        (build_delivery_receipt(plan, (("p", "succeeded", ""),)),))
    return build_delivery_snapshot(changes, receipts)


def _chain():
    """A log of four commits growing product ``p`` batch by batch."""
    snap0 = _empty_snapshot()
    snaps = [snap0] + [
        _ready_p(tuple((batch, str(batch + 1)) for batch in range(k + 1)))
        for k in range(4)]
    plans = [plan_delivery_updates(snaps[k], snaps[k + 1], (("p", k, k),))
             for k in range(4)]
    entries = tuple(
        ("c%d" % k, None if k == 0 else "c%d" % (k - 1), snaps[k],
         snaps[k + 1], plans[k])
        for k in range(4))
    return delivery_log(entries), snaps


def _ledgers(log, snaps, count):
    """One batch per ledger; ledger ``k`` applies c_k -> c_{k+1}."""
    checkout_plans = [plan_delivery_checkout(log, "c%d" % k, "c%d" % (k + 1))
                      for k in range(3)]
    return tuple(
        checkout_log(log, (("b%d" % k, snaps[k + 1],
                            (checkout_plans[k],)),))
        for k in range(count))


@pytest.fixture
def scene():
    log, snaps = _chain()
    ledgers = _ledgers(log, snaps, 3)
    return log, snaps, ledgers


def _states(plan, count):
    """States confirming one unit each up to ``count`` units."""
    states = []
    state = None
    for _ in range(count):
        state = execute_checkout_recovery(plan, state, 1)
        states.append(state)
    return states


def _summary(plan, states):
    return merge_recovery_receipts(
        plan, tuple(build_recovery_receipt(plan, tuple(states[:i + 1]))
                    for i in range(len(states))))


def _history(plan, states):
    return build_recovery_history(
        plan, tuple(_summary(plan, states[:i + 1])
                    for i in range(len(states))))


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.update_recovery_index is update_recovery_index
    assert "update_recovery_index" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# index contents
# ---------------------------------------------------------------------------

def test_empty_index_canonical(scene):
    text = update_recovery_index(None, ())
    assert text == '{"entries":[],"snapshot":null,"resume":null,' \
                   '"complete":true}'
    assert json.loads(text) == {
        "entries": [], "snapshot": None, "resume": None, "complete": True}


def test_empty_items_leaves_index_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    index = update_recovery_index(None, ((plan, history),))
    assert update_recovery_index(index, ()) == index
    assert update_recovery_index(None, ()) == update_recovery_index(
        update_recovery_index(None, ()), ())


def test_single_complete_entry(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    text = update_recovery_index(None, ((plan, history),))
    document = json.loads(text)
    assert list(document) == ["entries", "snapshot", "resume", "complete"]
    assert len(document["entries"]) == 1
    assert document["entries"][0] == [
        json.loads(plan), json.loads(history)]
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] is None
    assert document["complete"] is True
    assert text == _canonical(document)
    assert not text.endswith("\n") and "\n" not in text
    # plan and history embedded byte for byte in that order
    assert ('[' + plan + ',' + history + ']') in text


def test_incomplete_last_entry_carries_resume(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    history = _history(plan, states[:1])
    text = update_recovery_index(None, ((plan, history),))
    document = json.loads(text)
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["complete"] is False
    assert document["resume"] == [0, json.loads(states[0])]
    assert ('"resume":[0,' + states[0] + ']') in text


def test_replace_incomplete_entry_with_extension(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    partial = _history(plan, states[:1])
    full = _history(plan, states)
    index = update_recovery_index(None, ((plan, partial),))
    updated = update_recovery_index(index, ((plan, full),))
    document = json.loads(updated)
    assert len(document["entries"]) == 1
    assert document["entries"][0] == [
        json.loads(plan), json.loads(full)]
    assert document["resume"] is None
    assert document["complete"] is True


def test_chained_entries_keep_order_and_snapshot(scene):
    log, snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    first_plan = plan_checkout_recovery(log, None, ledgers[:2])
    first_history = _history(first_plan, _states(first_plan, 2))
    second_plan = plan_checkout_recovery(log, checkpoint, ledgers[:3])
    # the second entry is still in progress: no units confirmed yet
    second_history = build_recovery_history(second_plan, ())
    second_resume = execute_checkout_recovery(second_plan, None, 0)
    text = update_recovery_index(
        None, ((first_plan, first_history),
               (second_plan, second_history)))
    document = json.loads(text)
    assert [entry[0] for entry in document["entries"]] == [
        json.loads(first_plan), json.loads(second_plan)]
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] == [1, json.loads(second_resume)]
    assert document["complete"] is False
    assert text == _canonical(document)
    # feeding the index back with an empty tuple is byte-identical
    assert update_recovery_index(text, ()) == text


def test_extend_second_entry_after_round_trip(scene):
    log, snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    first_plan = plan_checkout_recovery(log, None, ledgers[:2])
    first_history = _history(first_plan, _states(first_plan, 2))
    second_plan = plan_checkout_recovery(log, checkpoint, ledgers[:3])
    pending_history = build_recovery_history(second_plan, ())
    index = update_recovery_index(
        None, ((first_plan, first_history),
               (second_plan, pending_history)))
    second_states = _states(second_plan, 1)
    second_history = _history(second_plan, second_states)
    updated = update_recovery_index(index, ((second_plan, second_history),))
    document = json.loads(updated)
    assert len(document["entries"]) == 2
    assert document["snapshot"] == json.loads(snaps[4])
    assert document["resume"] is None
    assert document["complete"] is True


def test_complete_second_entry_completes_index(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:1])
    first_plan = plan_checkout_recovery(log, None, ledgers[:1])
    first_history = _history(first_plan, _states(first_plan, 1))
    second_plan = plan_checkout_recovery(log, checkpoint, ledgers[:2])
    second_history = _history(second_plan, _states(second_plan, 1))
    text = update_recovery_index(
        None, ((first_plan, first_history),
               (second_plan, second_history)))
    document = json.loads(text)
    assert document["resume"] is None
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    for value in (1, b"x", [], {}):
        with pytest.raises(TypeError):
            update_recovery_index(value, ())
    for value in ([], None, "x", 1):
        with pytest.raises(TypeError):
            update_recovery_index(None, value)
    for value in ([plan, history], (plan,), (1, history), (plan, 1),
                  (plan, history, None)):
        with pytest.raises(TypeError):
            update_recovery_index(None, (value,))


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_plan_history_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    history = _history(plan, _states(plan, 2))
    with pytest.raises(ValueError):
        update_recovery_index(None, ((other, history),))


def test_non_canonical_inputs_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    with pytest.raises(ValueError):
        update_recovery_index("{", ())
    with pytest.raises(ValueError):
        update_recovery_index(None, (("{", history),))
    with pytest.raises(ValueError):
        update_recovery_index(None, ((plan, "{"),))
    index = update_recovery_index(None, ((plan, history),))
    with pytest.raises(ValueError):
        update_recovery_index(index + " ", ())
    with pytest.raises(ValueError):
        update_recovery_index(json.dumps(json.loads(index), indent=2), ())


def test_non_last_entry_must_be_complete(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    first_plan = plan_checkout_recovery(log, None, ledgers[:2])
    partial = _history(first_plan, _states(first_plan, 1))
    second_plan = plan_checkout_recovery(log, checkpoint, ledgers[:3])
    second_history = _history(second_plan, _states(second_plan, 1))
    with pytest.raises(ValueError):
        update_recovery_index(
            None, ((first_plan, partial), (second_plan, second_history)))


def test_chaining_requires_end_equals_next_start(scene):
    log, _snaps, ledgers = scene
    first_plan = plan_checkout_recovery(log, None, ledgers[:1])
    first_history = _history(first_plan, _states(first_plan, 1))
    # this plan starts from the empty snapshot, not the first end
    other = plan_checkout_recovery(log, None, ledgers[:2])
    other_history = _history(other, _states(other, 1))
    with pytest.raises(ValueError):
        update_recovery_index(
            None, ((first_plan, first_history), (other, other_history)))


def test_duplicate_batch_id_across_entries_rejected(scene):
    log, _snaps, ledgers = scene
    first_plan = plan_checkout_recovery(log, None, ledgers[:1])
    first_history = _history(first_plan, _states(first_plan, 1))
    second_plan = plan_checkout_recovery(log, None, ledgers[:1])
    second_history = _history(second_plan, _states(second_plan, 1))
    with pytest.raises(ValueError):
        update_recovery_index(
            None, ((first_plan, first_history),
                   (second_plan, second_history)))


def test_completed_entry_cannot_be_replaced(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    index = update_recovery_index(None, ((plan, history),))
    with pytest.raises(ValueError):
        update_recovery_index(index, ((plan, history),))


def test_incomplete_entry_replacement_needs_same_plan(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(log, None, ledgers[:2])
    assert other == plan  # canonical plans are identical
    partial = _history(plan, _states(plan, 1))
    # a genuinely different plan (different common prefix) replacing the
    # incomplete entry is rejected
    different = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:2]), ledgers[:3])
    different_history = _history(different, _states(different, 1))
    index = update_recovery_index(None, ((plan, partial),))
    with pytest.raises(ValueError):
        update_recovery_index(index, ((different, different_history),))


def test_replacement_must_strictly_extend_runs_and_audit(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    partial = _history(plan, states[:1])
    index = update_recovery_index(None, ((plan, partial),))
    # re-supplying the same history is not a strict extension
    with pytest.raises(ValueError):
        update_recovery_index(index, ((plan, partial),))
    tampered = partial.replace('"b0"', '"b9"', 1)
    assert tampered != partial
    with pytest.raises(ValueError):
        update_recovery_index(index, ((plan, tampered),))


def test_tampered_index_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:1])
    history = _history(plan, _states(plan, 1))
    index = update_recovery_index(None, ((plan, history),))
    with pytest.raises(ValueError):
        update_recovery_index(index.replace('"b0"', '"b9"', 1), ())


def test_tampered_index_fields_rejected(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    partial = _history(plan, states[:1])
    index = update_recovery_index(None, ((plan, partial),))
    # wrong resume position
    with pytest.raises(ValueError):
        update_recovery_index(index.replace('"resume":[0,', '"resume":[1,'),
                              ())
    # wrong complete flag (still canonical JSON otherwise)
    document = json.loads(index)
    document["complete"] = True
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(document), ())
    # wrong snapshot
    document = json.loads(index)
    document["snapshot"] = json.loads(snaps[3])
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(document), ())


def test_stored_chain_break_rejected(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:1])
    first_plan = plan_checkout_recovery(log, None, ledgers[:1])
    first_history = _history(first_plan, _states(first_plan, 1))
    second_plan = plan_checkout_recovery(log, checkpoint, ledgers[:2])
    second_history = _history(second_plan, _states(second_plan, 1))
    index = update_recovery_index(
        None, ((first_plan, first_history),
               (second_plan, second_history)))
    # swap the two entries so the later entry no longer chains from the
    # earlier one's end
    document = json.loads(index)
    document["entries"] = [document["entries"][1], document["entries"][0]]
    with pytest.raises(ValueError):
        update_recovery_index(_canonical(document), ())


def test_index_without_units_plan(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    history = build_recovery_history(plan, ())
    index = update_recovery_index(None, ((plan, history),))
    document = json.loads(index)
    assert document["snapshot"] is None
    assert document["resume"] is None
    assert document["complete"] is True
    assert update_recovery_index(index, ()) == index


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    history = _history(plan, _states(plan, 2))
    items = ((plan, history),)
    saved = (plan, history)
    one = update_recovery_index(None, items)
    two = update_recovery_index(None, items)
    assert one == two
    assert items[0] == saved
    assert update_recovery_index(one, ()) == one
