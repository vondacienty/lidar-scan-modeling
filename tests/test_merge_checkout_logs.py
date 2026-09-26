"""Tests for :func:`lidar_scan.merge_checkout_logs`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, checkout_log,
                        delivery_log, merge_checkout_logs,
                        merge_delivery_manifests, merge_delivery_receipts,
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
    """A log of three commits growing product ``p`` batch by batch."""
    snap0 = _empty_snapshot()
    snap1 = _ready_p(((0, "1"),))
    snap2 = _ready_p(((0, "1"), (1, "2")))
    snap3 = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan1 = plan_delivery_updates(snap0, snap1, (("p", 0, 0),))
    plan2 = plan_delivery_updates(snap1, snap2, (("p", 1, 1),))
    plan3 = plan_delivery_updates(snap2, snap3, (("p", 2, 2),))
    entries = (("c1", None, snap0, snap1, plan1),
               ("c2", "c1", snap1, snap2, plan2),
               ("c3", "c2", snap2, snap3, plan3))
    return delivery_log(entries), (snap0, snap1, snap2, snap3)


def _forward(log, source, target):
    return (plan_delivery_checkout(log, source, target),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.merge_checkout_logs is merge_checkout_logs
    assert "merge_checkout_logs" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty ledgers
# ---------------------------------------------------------------------------

def test_no_ledgers():
    log, _snapshots = _chain()
    text = merge_checkout_logs(log, ())
    assert text == '{"ranges":[],"audit":[],"snapshot":null}'


def test_only_empty_ledgers():
    log, _snapshots = _chain()
    empty = checkout_log(log, ())
    text = merge_checkout_logs(log, (empty, empty))
    assert text == '{"ranges":[],"audit":[],"snapshot":null}'


# ---------------------------------------------------------------------------
# merged ranges and audit
# ---------------------------------------------------------------------------

def test_merge_chains_non_empty_ledgers_across_empty_ones():
    log, (_snap0, snap1, snap2, snap3) = _chain()
    empty = checkout_log(log, ())
    ledger_a = checkout_log(log, (("b1", snap1,
                                   _forward(log, "c1", "c3")),))
    ledger_b = checkout_log(log, (("b2", snap3,
                                   (plan_delivery_checkout(log, "c3", "c2"),
                                    plan_delivery_checkout(log, "c2",
                                                           "c1"))),))
    ledger_c = checkout_log(log, (("b3", snap1,
                                   _forward(log, "c1", "c1")),))
    text = merge_checkout_logs(log, (empty, ledger_a, empty, ledger_b,
                                     empty, ledger_c))
    document = json.loads(text)
    assert list(document) == ["ranges", "audit", "snapshot"]
    assert document["ranges"] == [
        ["b1", "b1", 1, json.loads(snap1), json.loads(snap3)],
        ["b2", "b2", 1, json.loads(snap3), json.loads(snap1)],
        ["b3", "b3", 1, json.loads(snap1), json.loads(snap1)]]
    assert document["audit"] == [["b1", "applied"], ["b2", "applied"],
                                 ["b3", "unchanged"]]
    assert document["snapshot"] == json.loads(snap1)


def test_range_spans_multiple_batches_of_one_ledger():
    log, (_snap0, snap1, snap2, snap3) = _chain()
    ledger = checkout_log(log, (
        ("b4", snap1, _forward(log, "c1", "c2")),
        ("b5", snap2, _forward(log, "c2", "c3"))))
    text = merge_checkout_logs(log, (ledger,))
    document = json.loads(text)
    assert document["ranges"] == [
        ["b4", "b5", 2, json.loads(snap1), json.loads(snap3)]]
    assert document["audit"] == [["b4", "applied"], ["b5", "applied"]]
    assert document["snapshot"] == json.loads(snap3)


def test_embedded_snapshots_are_raw_canonical_spans():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c3")),))
    text = merge_checkout_logs(log, (ledger,))
    assert text == ('{"ranges":[["b1","b1",1,' + snap1 + "," + snap3
                    + ']],"audit":[["b1","applied"]],"snapshot":' + snap3
                    + "}")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_boundary_mismatch_rejected():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    ledger_a = checkout_log(log, (("b1", snap1,
                                   _forward(log, "c1", "c3")),))
    ledger_c = checkout_log(log, (("b3", snap1,
                                   _forward(log, "c1", "c1")),))
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (ledger_a, ledger_c))
    # snap3 == snap3 here only documents that the mismatch was deliberate.
    assert snap3 != snap1


def test_empty_ledger_does_not_change_boundary():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    empty = checkout_log(log, ())
    ledger_a = checkout_log(log, (("b1", snap1,
                                   _forward(log, "c1", "c3")),))
    ledger_b = checkout_log(log, (
        ("b2", snap3, (plan_delivery_checkout(log, "c3", "c2"),
                       plan_delivery_checkout(log, "c2", "c1"))),))
    text = merge_checkout_logs(log, (ledger_a, empty, ledger_b))
    assert json.loads(text)["audit"] == [["b1", "applied"],
                                         ["b2", "applied"]]


def test_duplicate_batch_id_rejected():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    ledger_a = checkout_log(log, (("b1", snap1,
                                   _forward(log, "c1", "c3")),))
    ledger_b = checkout_log(log, (
        ("b1", snap3, (plan_delivery_checkout(log, "c3", "c2"),
                       plan_delivery_checkout(log, "c2", "c1"))),))
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (ledger_a, ledger_b))


def test_malformed_ledger_rejected():
    log, _snapshots = _chain()
    for bad in ("", "not json", "{", '{"batches":[]', "[1,2]",
                '{"batches":[],"snapshot":null} extra'):
        with pytest.raises(ValueError):
            merge_checkout_logs(log, (bad,))


def test_non_canonical_ledger_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c2")),))
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (ledger + " ",))
    pretty = json.dumps(json.loads(ledger), indent=2)
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (pretty,))


def test_tampered_ledger_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c2")),))
    tampered = ledger.replace('"applied"', '"unchanged"', 1)
    assert tampered != ledger
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (tampered,))


def test_tampered_log_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c2")),))
    tampered = log.replace('"committed":true', '"committed":false', 1)
    assert tampered != log
    with pytest.raises(ValueError):
        merge_checkout_logs(tampered, (ledger,))


def test_ledger_from_other_log_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    other_snap = _ready_p(((0, "1"),))
    other_plan = plan_delivery_updates(_empty_snapshot(), other_snap,
                                       (("p", 0, 0),))
    other_log = delivery_log((("q1", None, _empty_snapshot(), other_snap,
                               other_plan),))
    foreign = checkout_log(
        other_log, (("b1", other_snap,
                     _forward(other_log, "q1", "q1")),))
    with pytest.raises(ValueError):
        merge_checkout_logs(log, (foreign,))


# ---------------------------------------------------------------------------
# type validation and encoding properties
# ---------------------------------------------------------------------------

def test_type_errors():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c2")),))
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            merge_checkout_logs(value, (ledger,))
        with pytest.raises(TypeError):
            merge_checkout_logs(log, value)
        with pytest.raises(TypeError):
            merge_checkout_logs(log, (value,))


def test_compact_encoding_without_trailing_newline():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    empty = checkout_log(log, ())
    ledger = checkout_log(log, (("b1", snap1, _forward(log, "c1", "c3")),))
    ledger_b = checkout_log(log, (
        ("b2", snap3, (plan_delivery_checkout(log, "c3", "c2"),
                       plan_delivery_checkout(log, "c2", "c1"))),))
    for text in (merge_checkout_logs(log, ()),
                 merge_checkout_logs(log, (empty,)),
                 merge_checkout_logs(log, (ledger,)),
                 merge_checkout_logs(log, (ledger, empty, ledger_b))):
        assert not text.endswith("\n")
        assert "\n" not in text and "\t" not in text
        assert text == json.dumps(json.loads(text), ensure_ascii=False,
                                  separators=(",", ":"))


def test_repeatable_and_inputs_unchanged():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    ledger_a = checkout_log(log, (("b1", snap1,
                                   _forward(log, "c1", "c3")),))
    ledger_b = checkout_log(log, (
        ("b2", snap3, (plan_delivery_checkout(log, "c3", "c2"),
                       plan_delivery_checkout(log, "c2", "c1"))),))
    ledgers = (ledger_a, ledger_b)
    first = merge_checkout_logs(log, ledgers)
    second = merge_checkout_logs(log, ledgers)
    assert first == second
    assert ledgers == (ledger_a, ledger_b)
