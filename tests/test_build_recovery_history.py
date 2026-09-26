"""Tests for :func:`lidar_scan.build_recovery_history`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, build_recovery_history,
                        build_recovery_receipt, checkout_log, delivery_log,
                        execute_checkout_recovery, merge_checkout_logs,
                        merge_delivery_manifests, merge_delivery_receipts,
                        merge_recovery_receipts, plan_checkout_recovery,
                        plan_delivery_checkout, plan_delivery_updates)
from lidar_scan import tiles as tiles_module

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


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.build_recovery_history is build_recovery_history
    assert "build_recovery_history" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def test_history_across_two_summaries(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first, second = _states(plan, 2)
    summary_one = _summary(plan, [first])
    summary_two = _summary(plan, [first, second])
    text = build_recovery_history(plan, (summary_one, summary_two))
    document = json.loads(text)
    assert list(document) == ["direction", "start", "end", "runs", "audit",
                             "resume", "complete"]
    assert document["direction"] == "pending"
    assert document["start"] == json.loads(snaps[1])
    assert document["end"] == json.loads(snaps[3])
    assert document["runs"] == [
        [1, json.loads(snaps[2]), False],
        [2, json.loads(snaps[3]), True],
    ]
    assert document["audit"] == [["b0", "applied"], ["b1", "applied"]]
    assert document["resume"] is None
    assert document["complete"] is True
    assert text == _canonical(document)
    assert "\n" not in text and not text.endswith("\n")


def test_run_end_snapshots_embedded_byte_for_byte(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first, second = _states(plan, 2)
    summary_two = _summary(plan, [first, second])
    end_raw = summary_two[
        summary_two.index('"end":') + 6:
        summary_two.index(',"receipts"')]
    text = build_recovery_history(plan, (_summary(plan, [first]),
                                         summary_two))
    assert ('[2,' + end_raw + ',true]') in text


def test_resume_state_at_confirmed_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first, second = _states(plan, 2)
    summary_one = _summary(plan, [first])
    summary_two = _summary(plan, [first, second])
    text = build_recovery_history(plan, (summary_one,))
    document = json.loads(text)
    assert document["complete"] is False
    assert document["resume"] == json.loads(first)
    assert '"resume":' + first in text
    text = build_recovery_history(plan, (summary_one, summary_two))
    assert json.loads(text)["resume"] is None


def test_empty_summaries_uses_start_and_initial_resume(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    text = build_recovery_history(plan, ())
    document = json.loads(text)
    assert document == {
        "direction": "pending",
        "start": json.loads(snaps[1]),
        "end": json.loads(snaps[1]),
        "runs": [],
        "audit": [],
        "resume": json.loads(execute_checkout_recovery(plan, None, 0)),
        "complete": False,
    }
    assert text == _canonical(document)


def test_no_units_is_complete(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    document = json.loads(build_recovery_history(plan, ()))
    assert document == {
        "direction": "none", "start": None, "end": None, "runs": [],
        "audit": [], "resume": None, "complete": True}
    empty = merge_recovery_receipts(plan, ())
    document = json.loads(build_recovery_history(plan, (empty,)))
    assert document["runs"] == [[0, None, True]]
    assert document["complete"] is True
    assert document["resume"] is None


def test_missing_direction_history(scene):
    log, snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    states = _states(plan, 2)
    text = build_recovery_history(
        plan, (_summary(plan, states[:1]), _summary(plan, states)))
    document = json.loads(text)
    assert document["direction"] == "missing"
    assert document["start"] == json.loads(snaps[2])
    assert document["end"] == json.loads(snaps[4])
    assert document["runs"][0] == [1, json.loads(snaps[3]), False]
    assert document["runs"][1] == [2, json.loads(snaps[4]), True]
    assert document["audit"] == [["b1", "applied"], ["b2", "applied"]]
    assert document["complete"] is True


def test_audit_follows_plan_units_for_multi_unit_summary(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first, second = _states(plan, 2)
    summary = _summary(plan, [first, second])
    document = json.loads(build_recovery_history(plan, (summary,)))
    assert document["runs"] == [[2, document["end"], True]]
    assert document["audit"] == [["b0", "applied"], ["b1", "applied"]]


def test_leading_empty_receipt_summary(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    empty = build_recovery_receipt(plan, ())
    receipt = build_recovery_receipt(plan, (first,))
    summary = merge_recovery_receipts(plan, (empty, receipt))
    document = json.loads(build_recovery_history(plan, (summary,)))
    assert document["runs"] == [[1, document["end"], False]]
    assert document["audit"] == [["b0", "applied"]]
    assert document["resume"] == json.loads(first)


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            build_recovery_history(value, ())
    for value in ([], None, "x", 1):
        with pytest.raises(TypeError):
            build_recovery_history(plan, value)
    state = execute_checkout_recovery(plan, None, 1)
    summary = _summary(plan, [state])
    with pytest.raises(TypeError):
        build_recovery_history(plan, (summary, 1))


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_plan_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for bad in ("", "not json", "{", "[]", "null",
                " " + plan,
                json.dumps(json.loads(plan), indent=2)):
        with pytest.raises(ValueError):
            build_recovery_history(bad, ())


def test_forked_plan_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    pending_rows = pending_plan[
        pending_plan.index('"pending":[') + 11:
        pending_plan.index('],"snapshot"')]
    missing_rows = missing_plan[
        missing_plan.index('"missing":[') + 11:
        missing_plan.index('],"pending"')]
    snapshot_raw = pending_plan[
        pending_plan.index('"snapshot":') + 11:-1]
    forked = ('{"common":0,"missing":[' + missing_rows + '],"pending":['
              + pending_rows + '],"snapshot":' + snapshot_raw + "}")
    with pytest.raises(ValueError):
        build_recovery_history(forked, ())


def test_malformed_or_non_canonical_summary_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    summary = _summary(plan, [state])
    with pytest.raises(ValueError):
        build_recovery_history(plan, ("{",))
    with pytest.raises(ValueError):
        build_recovery_history(
            plan, (json.dumps(json.loads(summary), indent=2),))
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary + " ",))


def test_direction_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    missing_state = execute_checkout_recovery(missing_plan, None, 1)
    missing_summary = _summary(missing_plan, [missing_state])
    with pytest.raises(ValueError):
        build_recovery_history(pending_plan, (missing_summary,))


def test_start_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    other_state = execute_checkout_recovery(other, None, 1)
    other_summary = _summary(other, [other_state])
    with pytest.raises(ValueError):
        build_recovery_history(plan, (other_summary,))


def test_summaries_must_strictly_extend(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    summary_one = _summary(plan, states[:1])
    summary_two = _summary(plan, states)
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary_one, summary_one))
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary_two, summary_one))


def test_tampered_prefix_receipt_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    summary_one = _summary(plan, states[:1])
    summary_two = _summary(plan, states)
    tampered_two = summary_two.replace('"b0"', '"b1"', 1)
    assert tampered_two != summary_two
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary_one, tampered_two))


def test_summary_whose_merge_changes_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    summary = _summary(plan, [state])
    document = json.loads(summary)
    document["complete"] = True
    with pytest.raises(ValueError):
        build_recovery_history(plan, (_canonical(document),))


def test_embedded_receipt_against_other_plan_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    other_state = execute_checkout_recovery(other, None, 1)
    other_receipt = build_recovery_receipt(other, (other_state,))
    document = json.loads(_summary(plan, [state]))
    document["receipts"] = [json.loads(other_receipt)]
    with pytest.raises(ValueError):
        build_recovery_history(plan, (_canonical(document),))


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    summaries = (_summary(plan, states[:1]), _summary(plan, states))
    saved = tuple(summaries)
    one = build_recovery_history(plan, summaries)
    two = build_recovery_history(plan, summaries)
    assert one == two
    assert summaries == saved
    assert plan == plan_checkout_recovery(log, None, ledgers[:2])
