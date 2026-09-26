"""Tests for :func:`lidar_scan.merge_recovery_receipts`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, build_recovery_receipt,
                        checkout_log, delivery_log,
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


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.merge_recovery_receipts is merge_recovery_receipts
    assert "merge_recovery_receipts" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def test_stepwise_receipts_merge_in_order(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    text = merge_recovery_receipts(plan, (receipt_one, receipt_two))
    document = json.loads(text)
    assert list(document) == ["direction", "start", "end", "receipts",
                              "complete"]
    assert document["direction"] == "pending"
    assert document["start"] == json.loads(snaps[1])
    assert document["end"] == json.loads(snaps[3])
    assert document["receipts"] == [json.loads(receipt_one),
                                    json.loads(receipt_two)]
    assert document["complete"] is True
    assert text == _canonical(document)
    assert "\n" not in text and not text.endswith("\n")


def test_receipts_embedded_verbatim(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    text = merge_recovery_receipts(plan, (receipt_one, receipt_two))
    assert text.index(receipt_one) > 0
    assert text.index(receipt_two) > text.index(receipt_one)


def test_empty_receipts_uses_start_as_end(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    document = json.loads(merge_recovery_receipts(plan, ()))
    assert document == {"direction": "pending",
                        "start": json.loads(snaps[1]),
                        "end": json.loads(snaps[1]),
                        "receipts": [], "complete": False}


def test_empty_receipts_of_unitless_plan_is_complete(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    document = json.loads(merge_recovery_receipts(plan, ()))
    assert document == {"direction": "none", "start": None, "end": None,
                        "receipts": [], "complete": True}


def test_partial_last_receipt_is_not_complete(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (state,))
    document = json.loads(merge_recovery_receipts(plan, (receipt,)))
    assert document["end"] == json.loads(snaps[2])
    assert document["complete"] is False


def test_missing_direction_merge(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    plan_document = json.loads(plan)
    one = execute_checkout_recovery(plan, None, 1)
    done = execute_checkout_recovery(plan)
    receipt_one = build_recovery_receipt(plan, (one,))
    receipt_two = build_recovery_receipt(plan, (one, done))
    document = json.loads(merge_recovery_receipts(plan,
                                                  (receipt_one, receipt_two)))
    assert document["direction"] == "missing"
    assert document["start"] == plan_document["missing"][0][0][3]
    assert document["end"] == plan_document["snapshot"]
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            merge_recovery_receipts(value, ())
    for value in ([], None, "x", 1):
        with pytest.raises(TypeError):
            merge_recovery_receipts(plan, value)
    receipt = build_recovery_receipt(
        plan, (execute_checkout_recovery(plan, None, 1),))
    with pytest.raises(TypeError):
        merge_recovery_receipts(plan, (receipt, 1))


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
            merge_recovery_receipts(bad, ())


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
        merge_recovery_receipts(forked, ())


def test_malformed_or_non_canonical_receipt_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    receipt = build_recovery_receipt(
        plan, (execute_checkout_recovery(plan, None, 1),))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, ("{",))
    with pytest.raises(ValueError):
        merge_recovery_receipts(
            plan, (json.dumps(json.loads(receipt), indent=2),))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt + " ",))


def test_receipt_direction_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    missing_receipt = build_recovery_receipt(
        missing_plan, (execute_checkout_recovery(missing_plan, None, 1),))
    with pytest.raises(ValueError):
        merge_recovery_receipts(pending_plan, (missing_receipt,))


def test_receipt_start_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    receipt = build_recovery_receipt(
        plan, (execute_checkout_recovery(plan, None, 1),))
    other = plan_checkout_recovery(log, None, ledgers[1:3])
    other_receipt = build_recovery_receipt(
        other, (execute_checkout_recovery(other, None, 1),))
    assert json.loads(receipt)["start"] != json.loads(other_receipt)["start"]
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (other_receipt,))


def _tamper(receipt, old, new):
    tampered = receipt.replace(old, new, 1)
    assert tampered != receipt
    return tampered


def test_broken_segment_interval_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    receipt = build_recovery_receipt(plan, (finished,))
    assert '"segments":[[0,2,' in receipt
    with pytest.raises(ValueError):
        merge_recovery_receipts(
            plan, (_tamper(receipt, '"segments":[[0,2,',
                           '"segments":[[1,2,'),))


def test_duplicate_batch_id_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    receipt = build_recovery_receipt(plan, (finished,))
    tampered = receipt.replace('"b1"', '"b0"')
    assert tampered != receipt
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (tampered,))


def test_batches_must_equal_plan_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    receipt = build_recovery_receipt(plan, (finished,))
    with pytest.raises(ValueError):
        merge_recovery_receipts(
            plan, (_tamper(receipt, '["b0","applied"]', '["b0","unchanged"]'),))


def test_end_must_equal_plan_prefix(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    receipt = build_recovery_receipt(plan, (finished,))
    # Swap end for the start snapshot: no longer the confirmed prefix's end.
    start_raw = receipt[receipt.index('"start":') + 8:receipt.index(',"end"')]
    end_raw = receipt[receipt.index(',"end":') + 7:receipt.index(',"segments"')]
    assert start_raw != end_raw
    tampered = receipt.replace(',"end":' + end_raw, ',"end":' + start_raw, 1)
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (tampered,))


def test_complete_must_equal_plan_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (state,))
    assert receipt.endswith('"complete":false}')
    with pytest.raises(ValueError):
        merge_recovery_receipts(
            plan, (_tamper(receipt, '"complete":false', '"complete":true'),))


def test_confirmed_must_strictly_increase(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_two, receipt_one))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, receipt_one))


def test_segments_must_extend_the_previous_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    # Same confirmed prefix but a single flattened segment: not an
    # extension of receipt_one's two segments.
    flattened = build_recovery_receipt(plan, (second,))
    assert json.loads(receipt_one)["segments"] != \
        json.loads(flattened)["segments"]
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, flattened))


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    assert merge_recovery_receipts(plan, (receipt_one, receipt_two)) == \
        merge_recovery_receipts(plan, (receipt_one, receipt_two))
    assert plan == plan_checkout_recovery(log, None, ledgers[:2])
    assert receipt_one == build_recovery_receipt(plan, (first,))
