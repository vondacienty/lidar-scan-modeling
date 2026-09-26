"""Tests for :func:`lidar_scan.build_recovery_receipt`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, build_recovery_receipt,
                        checkout_log, delivery_log,
                        execute_checkout_recovery, merge_checkout_logs,
                        merge_delivery_manifests, merge_delivery_receipts,
                        plan_checkout_recovery, plan_delivery_checkout,
                        plan_delivery_updates)
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


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.build_recovery_receipt is build_recovery_receipt
    assert "build_recovery_receipt" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# pending direction
# ---------------------------------------------------------------------------

def test_stepwise_states_segment_the_audit(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    text = build_recovery_receipt(plan, (first, second))
    document = json.loads(text)
    assert list(document) == ["direction", "start", "end", "segments",
                              "complete"]
    assert document["direction"] == "pending"
    assert document["start"] == json.loads(snaps[1])
    assert document["end"] == json.loads(snaps[3])
    assert document["segments"] == [
        [0, 1, [["b0", "applied"]]],
        [1, 2, [["b1", "applied"]]],
    ]
    assert document["complete"] is True
    assert text == _canonical(document)
    assert "\n" not in text and not text.endswith("\n")


def test_one_state_covering_several_units_flattens_every_audit(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    document = json.loads(build_recovery_receipt(plan, (finished,)))
    assert document["segments"] == [
        [0, 2, [["b0", "applied"], ["b1", "applied"]]],
    ]
    assert document["complete"] is True


def test_empty_states_uses_start_as_end(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    document = json.loads(build_recovery_receipt(plan, ()))
    assert document["direction"] == "pending"
    assert document["start"] == json.loads(snaps[1])
    assert document["end"] == document["start"]
    assert document["segments"] == []
    assert document["complete"] is False


def test_partial_progress_is_not_complete(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    document = json.loads(build_recovery_receipt(plan, (state,)))
    assert document["end"] == json.loads(snaps[2])
    assert document["segments"] == [[0, 1, [["b0", "applied"]]]]
    assert document["complete"] is False


def test_start_is_plan_snapshot_when_there_are_no_units(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    document = json.loads(build_recovery_receipt(plan, ()))
    assert document == {"direction": "none", "start": None, "end": None,
                        "segments": [], "complete": True}


def test_missing_direction_segments(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    plan_document = json.loads(plan)
    one = execute_checkout_recovery(plan, None, 1)
    done = execute_checkout_recovery(plan)
    document = json.loads(build_recovery_receipt(plan, (one, done)))
    assert document["direction"] == "missing"
    assert document["start"] == plan_document["missing"][0][0][3]
    assert document["end"] == plan_document["snapshot"]
    assert document["segments"] == [
        [0, 1, [["b1", "applied"]]],
        [1, 2, [["b2", "applied"]]],
    ]
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            build_recovery_receipt(value, ())
    for value in ([], None, "x", 1):
        with pytest.raises(TypeError):
            build_recovery_receipt(plan, value)
    state = execute_checkout_recovery(plan, None, 1)
    with pytest.raises(TypeError):
        build_recovery_receipt(plan, (state, 1))


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
            build_recovery_receipt(bad, ())


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
        build_recovery_receipt(forked, ())


def test_state_common_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    other_state = execute_checkout_recovery(other, None, 1)
    assert json.loads(state)["common"] == 0
    assert json.loads(other_state)["common"] == 1
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, (other_state,))


def test_state_direction_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    missing_state = execute_checkout_recovery(missing_plan, None, 1)
    with pytest.raises(ValueError):
        build_recovery_receipt(pending_plan, (missing_state,))


def test_confirmed_must_strictly_increase(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, (second, first))
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, (first, first))


def test_records_must_extend_the_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    finished = execute_checkout_recovery(plan)
    tampered = finished.replace('"b0"', '"b1"', 1)
    assert tampered != finished
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, (first, tampered))


def test_malformed_or_non_canonical_state_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, ("{",))
    with pytest.raises(ValueError):
        build_recovery_receipt(
            plan, (json.dumps(json.loads(state), indent=2),))
    with pytest.raises(ValueError):
        build_recovery_receipt(plan, (state + " ",))


def test_batch_ids_do_not_repeat_across_segments(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt = json.loads(build_recovery_receipt(plan, (first, second)))
    ids = [batch[0] for segment in receipt["segments"] for batch in
           segment[2]]
    assert len(ids) == len(set(ids))


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    assert build_recovery_receipt(plan, (first, second)) == \
        build_recovery_receipt(plan, (first, second))
    assert plan == plan_checkout_recovery(log, None, ledgers[:2])
