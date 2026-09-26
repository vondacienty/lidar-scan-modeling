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

def test_merge_embeds_every_receipt_in_order(scene):
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
    assert document["receipts"] == [
        json.loads(receipt_one), json.loads(receipt_two)]
    assert document["complete"] is True
    assert text == _canonical(document)
    assert "\n" not in text and not text.endswith("\n")


def test_receipts_are_embedded_byte_for_byte(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    text = merge_recovery_receipts(plan, (receipt_one, receipt_two))
    assert ('"receipts":[' + receipt_one + "," + receipt_two + "]") in text


def test_single_receipt_is_wrapped(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    receipt = build_recovery_receipt(plan, (finished,))
    document = json.loads(merge_recovery_receipts(plan, (receipt,)))
    assert document["direction"] == "pending"
    assert document["receipts"] == [json.loads(receipt)]
    assert document["end"] == json.loads(receipt)["end"]
    assert document["complete"] is True


def test_empty_receipts_uses_start_as_end(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    document = json.loads(merge_recovery_receipts(plan, ()))
    assert document == {
        "direction": "pending",
        "start": json.loads(snaps[1]),
        "end": json.loads(snaps[1]),
        "receipts": [],
        "complete": False,
    }


def test_empty_progress_receipt_may_lead(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    empty = build_recovery_receipt(plan, ())
    receipt = build_recovery_receipt(plan, (first,))
    document = json.loads(merge_recovery_receipts(plan, (empty, receipt)))
    assert document["receipts"] == [json.loads(empty), json.loads(receipt)]
    assert document["complete"] is False


def test_no_units_is_complete_with_or_without_receipts(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    assert json.loads(merge_recovery_receipts(plan, ())) == {
        "direction": "none", "start": None, "end": None, "receipts": [],
        "complete": True}
    empty = build_recovery_receipt(plan, ())
    assert json.loads(merge_recovery_receipts(plan, (empty,)))["complete"] \
        is True


def test_missing_direction_merge(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    one = execute_checkout_recovery(plan, None, 1)
    done = execute_checkout_recovery(plan)
    receipt_one = build_recovery_receipt(plan, (one,))
    receipt_two = build_recovery_receipt(plan, (one, done))
    document = json.loads(
        merge_recovery_receipts(plan, (receipt_one, receipt_two)))
    assert document["direction"] == "missing"
    assert document["receipts"] == [
        json.loads(receipt_one), json.loads(receipt_two)]
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
    state = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (state,))
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
    state = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (state,))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, ("{",))
    with pytest.raises(ValueError):
        merge_recovery_receipts(
            plan, (json.dumps(json.loads(receipt), indent=2),))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt + " ",))


def test_direction_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    missing_state = execute_checkout_recovery(missing_plan, None, 1)
    missing_receipt = build_recovery_receipt(missing_plan, (missing_state,))
    with pytest.raises(ValueError):
        merge_recovery_receipts(pending_plan, (missing_receipt,))


def test_start_mismatch_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    other = plan_checkout_recovery(
        log, merge_checkout_logs(log, ledgers[:1]), ledgers[:2])
    other_state = execute_checkout_recovery(other, None, 1)
    other_receipt = build_recovery_receipt(other, (other_state,))
    assert json.loads(other_receipt)["start"] != \
        json.loads(build_recovery_receipt(
            plan, (execute_checkout_recovery(plan, None, 1),)))["start"]
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (other_receipt,))


def test_confirmed_counts_must_strictly_increase(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    empty = build_recovery_receipt(plan, ())
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_two, receipt_one))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, receipt_one))
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (empty, empty))


def test_segment_interval_gap_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (first,))
    gapped = receipt.replace("[0,1,", "[1,2,", 1)
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (gapped,))


def test_earlier_segments_must_be_a_complete_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    tampered = receipt_two.replace('"b0"', '"b1"', 1)
    assert tampered != receipt_two
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, tampered))


def test_batches_must_match_the_plan_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    duplicated = receipt_two.replace('"b1"', '"b0"', 1)
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, duplicated))


def test_end_must_match_the_confirmed_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    document = json.loads(receipt_two)
    document["end"] = json.loads(first)["snapshot"]
    tampered = _canonical(document)
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (receipt_one, tampered))


def test_complete_must_match_the_confirmed_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (first,))
    document = json.loads(receipt)
    document["complete"] = True
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (_canonical(document),))


def test_receipt_cannot_confirm_more_units_than_the_plan(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    receipt = build_recovery_receipt(plan, (first,))
    document = json.loads(receipt)
    document["segments"] = [[0, 5, [["b0", "applied"]]]]
    document["complete"] = False
    with pytest.raises(ValueError):
        merge_recovery_receipts(plan, (_canonical(document),))


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, first, 1)
    receipt_one = build_recovery_receipt(plan, (first,))
    receipt_two = build_recovery_receipt(plan, (first, second))
    saved_one = receipt_one
    saved_two = receipt_two
    merged_one = merge_recovery_receipts(
        plan, (receipt_one, receipt_two))
    merged_two = merge_recovery_receipts(
        plan, (receipt_one, receipt_two))
    assert merged_one == merged_two
    assert receipt_one == saved_one
    assert receipt_two == saved_two
    assert plan == plan_checkout_recovery(log, None, ledgers[:2])
