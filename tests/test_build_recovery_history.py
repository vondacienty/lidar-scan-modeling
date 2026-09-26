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


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def _states(plan, count):
    """Confirm ``count`` units one at a time; return the state documents."""
    states = []
    state = None
    for _ in range(count):
        state = execute_checkout_recovery(plan, state, 1)
        states.append(state)
    return tuple(states)


def _summaries(plan, states):
    """merge_recovery_receipts summaries whose receipts strictly extend."""
    receipts = tuple(build_recovery_receipt(plan, states[:k + 1])
                     for k in range(len(states)))
    return tuple(merge_recovery_receipts(plan, receipts[:k + 1])
                 for k in range(len(receipts)))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.build_recovery_history is build_recovery_history
    assert "build_recovery_history" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# history building
# ---------------------------------------------------------------------------

def test_history_merges_stepwise_summaries(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:3])
    states = _states(plan, 3)
    summaries = _summaries(plan, states)
    text = build_recovery_history(plan, summaries)
    document = json.loads(text)
    assert list(document) == ["direction", "start", "end", "runs", "audit",
                              "resume", "complete"]
    assert document["direction"] == "pending"
    assert document["start"] == json.loads(snaps[1])
    assert document["end"] == json.loads(snaps[4])
    assert document["runs"] == [
        [1, json.loads(snaps[2]), False],
        [2, json.loads(snaps[3]), False],
        [3, json.loads(snaps[4]), True],
    ]
    assert document["audit"] == [
        ["b0", "applied"], ["b1", "applied"], ["b2", "applied"]]
    assert document["resume"] is None
    assert document["complete"] is True
    assert text == _canonical(document)
    assert "\n" not in text and not text.endswith("\n")


def test_resume_embeds_state_at_the_confirmed_prefix(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:3])
    states = _states(plan, 3)
    summaries = _summaries(plan, states[:2])
    resume = execute_checkout_recovery(plan, None, 2)
    text = build_recovery_history(plan, summaries)
    document = json.loads(text)
    assert document["runs"][-1] == [2, json.loads(snaps[3]), False]
    assert document["end"] == json.loads(snaps[3])
    assert document["audit"] == [["b0", "applied"], ["b1", "applied"]]
    assert document["resume"] == json.loads(resume)
    assert document["complete"] is False
    assert ('"resume":' + resume) in text


def test_empty_summaries_uses_start_as_end_and_initial_resume(scene):
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


def test_no_units_plan_is_complete_with_null_resume(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    text = build_recovery_history(plan, ())
    assert json.loads(text) == {
        "direction": "none", "start": None, "end": None, "runs": [],
        "audit": [], "resume": None, "complete": True}


def test_missing_direction_history(scene):
    log, snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    states = _states(plan, 2)
    summaries = _summaries(plan, states)
    document = json.loads(build_recovery_history(plan, summaries))
    assert document["direction"] == "missing"
    assert document["runs"][-1][0] == 2
    assert document["end"] == json.loads(snaps[4])
    assert document["complete"] is True
    assert document["resume"] is None


def test_one_summary_may_cover_multiple_units(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:3])
    states = _states(plan, 2)
    receipt = build_recovery_receipt(plan, states)
    summary = merge_recovery_receipts(plan, (receipt,))
    document = json.loads(build_recovery_history(plan, (summary,)))
    assert document["runs"] == [[2, document["end"], False]]
    assert document["audit"] == [["b0", "applied"], ["b1", "applied"]]
    assert document["resume"] == json.loads(states[-1])


def test_summary_with_leading_empty_progress_receipt(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    empty = build_recovery_receipt(plan, ())
    first = build_recovery_receipt(plan, states[:1])
    second = build_recovery_receipt(plan, states)
    summary_one = merge_recovery_receipts(plan, (empty, first))
    summary_two = merge_recovery_receipts(plan, (empty, first, second))
    document = json.loads(
        build_recovery_history(plan, (summary_one, summary_two)))
    assert [run[0] for run in document["runs"]] == [1, 2]
    assert document["complete"] is True


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
    summary = merge_recovery_receipts(
        plan, (build_recovery_receipt(plan, (state,)),))
    with pytest.raises(TypeError):
        build_recovery_history(plan, (summary, 1))


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_plan_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for bad in ("", "not json", "{", "[]", "null", " " + plan,
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
    summary = merge_recovery_receipts(
        plan, (build_recovery_receipt(plan, (state,)),))
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
    missing_summary = merge_recovery_receipts(
        missing_plan, (build_recovery_receipt(missing_plan,
                                              (missing_state,)),))
    with pytest.raises(ValueError):
        build_recovery_history(pending_plan, (missing_summary,))


def test_start_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    other_state = execute_checkout_recovery(other, None, 1)
    other_summary = merge_recovery_receipts(
        other, (build_recovery_receipt(other, (other_state,)),))
    with pytest.raises(ValueError):
        build_recovery_history(plan, (other_summary,))


def test_summary_receipts_must_strictly_extend(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:3])
    states = _states(plan, 3)
    receipt_one = build_recovery_receipt(plan, states[:1])
    receipt_two = build_recovery_receipt(plan, states[:2])
    summary_one = merge_recovery_receipts(plan, (receipt_one,))
    summary_two = merge_recovery_receipts(plan, (receipt_one, receipt_two,))
    # Repeating an identical summary does not extend the receipts.
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary_one, summary_one))
    # A receipt covering the same prefix without embedding the earlier
    # receipt advances the count but breaks the receipt prefix chain.
    summary_skip = merge_recovery_receipts(plan, (receipt_two,))
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summary_one, summary_skip))
    # And a strictly extending pair is accepted.
    assert json.loads(
        build_recovery_history(plan, (summary_one, summary_two)))["complete"] \
        is False


def test_summary_cannot_append_after_completion(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    states = _states(plan, 2)
    summaries = _summaries(plan, states)
    with pytest.raises(ValueError):
        build_recovery_history(
            plan, (summaries[-1], merge_recovery_receipts(plan, ())))
    with pytest.raises(ValueError):
        build_recovery_history(plan, (summaries[-1], summaries[-1]))


def test_no_units_plan_rejects_any_summary(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    empty_summary = merge_recovery_receipts(plan, ())
    with pytest.raises(ValueError):
        build_recovery_history(plan, (empty_summary,))


def test_tampered_end_or_complete_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    summary = merge_recovery_receipts(
        plan, (build_recovery_receipt(plan, (state,)),))
    bad_end = json.loads(summary)
    bad_end["end"] = bad_end["start"]
    with pytest.raises(ValueError):
        build_recovery_history(plan, (_canonical(bad_end),))
    bad_complete = json.loads(summary)
    bad_complete["complete"] = True
    with pytest.raises(ValueError):
        build_recovery_history(plan, (_canonical(bad_complete),))


def test_summary_cannot_confirm_more_units_than_the_plan(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    receipt = json.loads(build_recovery_receipt(plan, (state,)))
    # An embedded receipt claiming five confirmed units is invalid even
    # though the receipt decoder alone cannot see the plan's unit count.
    receipt["segments"] = [[0, 5, [["b0", "applied"]]]]
    receipt["complete"] = False
    outer = {
        "direction": "pending",
        "start": json.loads(snaps[1]),
        "end": json.loads(snaps[2]),
        "receipts": [receipt],
        "complete": False,
    }
    with pytest.raises(ValueError):
        build_recovery_history(plan, (_canonical(outer),))


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:3])
    states = _states(plan, 3)
    summaries = _summaries(plan, states)
    saved = summaries
    history_one = build_recovery_history(plan, summaries)
    history_two = build_recovery_history(plan, summaries)
    assert history_one == history_two
    assert summaries == saved
    assert plan == plan_checkout_recovery(log, None, ledgers[:3])
