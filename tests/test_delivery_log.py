"""Tests for :func:`lidar_scan.delivery_log`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, commit_delivery_updates,
                        delivery_log, merge_delivery_manifests,
                        merge_delivery_receipts, plan_delivery_updates)
from lidar_scan import tiles as tiles_module

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

EMPTY_CHANGES = '{"changes":[],"releasable":true}'
EMPTY_RECEIPTS = '{"products":[],"ready":true}'
EMPTY_PLAN = '{"operations":[],"changed":false}'


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
    """Three linked commits growing product ``p`` batch by batch."""
    snap0 = _empty_snapshot()
    snap1 = _ready_p(((0, "1"),))
    snap2 = _ready_p(((0, "1"), (1, "2")))
    plan1 = plan_delivery_updates(snap0, snap1, (("p", 0, 0),))
    plan2 = plan_delivery_updates(snap1, snap2, (("p", 1, 1),))
    entries = (("c1", None, snap0, snap1, plan1),
               ("c2", "c1", snap1, snap2, plan2))
    return entries, (snap0, snap1, snap2)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.delivery_log is delivery_log
    assert "delivery_log" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# output shape and encoding
# ---------------------------------------------------------------------------

def test_empty_entries():
    assert delivery_log(()) == '{"commits":[],"head":null}'


def test_single_commit_byte_exact():
    snapshot = _empty_snapshot()
    result = commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN)
    expected = ('{"commits":[["c1",null,' + snapshot + ',' + snapshot
                + ',' + EMPTY_PLAN + ',' + result + ']],"head":"c1"}')
    assert delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),)) \
        == expected


def test_chain_shape_and_key_order():
    entries, (snap0, snap1, snap2) = _chain()
    log = delivery_log(entries)
    assert ": " not in log and ", " not in log
    assert not log.endswith("\n")
    document = json.loads(log)
    assert list(document) == ["commits", "head"]
    assert document["head"] == "c2"
    rows = document["commits"]
    assert [row[0] for row in rows] == ["c1", "c2"]
    assert rows[0][1] is None
    assert rows[1][1] == "c1"
    assert rows[0][2] == json.loads(snap0)
    assert rows[0][3] == json.loads(snap1)
    assert rows[1][2] == json.loads(snap1)
    assert rows[1][3] == json.loads(snap2)
    for row, (commit_id, _parent, snapshot, target, plan) in zip(
            rows, entries):
        assert row[4] == json.loads(plan)
        assert row[5] == json.loads(
            commit_delivery_updates(snapshot, target, plan))


def test_deterministic_and_inputs_unchanged():
    entries, _snapshots = _chain()
    first = delivery_log(entries)
    second = delivery_log(entries)
    assert first == second
    assert entries[0][0] == "c1" and len(entries) == 2


def test_non_ascii_text_kept_unescaped():
    items = ((0, "p", "1", PASSING_GATE),)
    changes = merge_delivery_manifests(
        tuple(build_delivery_manifest((item,)) for item in items))
    plan = build_delivery_plan(audit_delivery_changes(changes))
    receipts = merge_delivery_receipts(
        (build_delivery_receipt(plan, (("p", "failed", "原因"),)),))
    snapshot = build_delivery_snapshot(changes, receipts)
    log = delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),))
    assert "原因" in log
    assert "\\u" not in log


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_entries_type_errors():
    snapshot = _empty_snapshot()
    entry = ("c1", None, snapshot, snapshot, EMPTY_PLAN)
    for value in (1, "x", None, b"x", [entry], {"c1": entry}):
        with pytest.raises(TypeError):
            delivery_log(value)


def test_entry_type_errors():
    snapshot = _empty_snapshot()
    good = ("c1", None, snapshot, snapshot, EMPTY_PLAN)
    for entry in ([None] * 5, ("c1", None, snapshot, snapshot),
                  ("c1", None, snapshot, snapshot, EMPTY_PLAN, "x"), "c1",
                  1, None):
        with pytest.raises(TypeError):
            delivery_log((entry,))
    with pytest.raises(TypeError):
        delivery_log((good, ["c2", "c1", snapshot, snapshot, EMPTY_PLAN]))


def test_field_type_errors():
    snapshot = _empty_snapshot()
    good = ("c1", None, snapshot, snapshot, EMPTY_PLAN)
    for index, value in ((0, 1), (0, None), (1, 1), (2, 1),
                         (2, None), (3, b"x"), (4, 1.5), (4, None)):
        entry = list(good)
        entry[index] = value
        with pytest.raises(TypeError):
            delivery_log((tuple(entry),))


def test_invalid_and_duplicate_ids():
    snapshot = _empty_snapshot()
    for bad_id in ("", "bad id", "a/b", "a+b", "提交"):
        with pytest.raises(ValueError):
            delivery_log(((bad_id, None, snapshot, snapshot, EMPTY_PLAN),))
    entry = ("c1", None, snapshot, snapshot, EMPTY_PLAN)
    with pytest.raises(ValueError):
        delivery_log((entry, ("c1", "c1", snapshot, snapshot, EMPTY_PLAN)))


def test_parent_chain_errors():
    snapshot = _empty_snapshot()
    with pytest.raises(ValueError):
        delivery_log((("c1", "c0", snapshot, snapshot, EMPTY_PLAN),))
    with pytest.raises(ValueError):
        delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),
                      ("c2", "c3", snapshot, snapshot, EMPTY_PLAN)))
    with pytest.raises(ValueError):
        delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),
                      ("c2", None, snapshot, snapshot, EMPTY_PLAN)))


def test_snapshot_linkage_error():
    entries, (snap0, snap1, snap2) = _chain()
    broken = (entries[0], ("c2", "c1", snap2, snap2, EMPTY_PLAN))
    with pytest.raises(ValueError):
        delivery_log(broken)


def test_document_value_errors():
    snapshot = _empty_snapshot()
    with pytest.raises(ValueError):
        delivery_log((("c1", None, "not json", snapshot, EMPTY_PLAN),))
    with pytest.raises(ValueError):
        delivery_log((("c1", None, snapshot, "not json", EMPTY_PLAN),))
    with pytest.raises(ValueError):
        delivery_log((("c1", None, snapshot, snapshot, "not json"),))
    with pytest.raises(ValueError):
        delivery_log((("c1", None, snapshot, snapshot,
                       '{"results":[],"changed":false}'),))
