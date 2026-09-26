"""Tests for :func:`lidar_scan.plan_delivery_checkout`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, commit_delivery_updates,
                        delivery_log, merge_delivery_manifests,
                        merge_delivery_receipts, plan_delivery_checkout,
                        plan_delivery_updates)
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


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.plan_delivery_checkout is plan_delivery_checkout
    assert "plan_delivery_checkout" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# forward checkout
# ---------------------------------------------------------------------------

def test_forward_checkout_shape_and_order():
    log, (snap0, snap1, snap2, snap3) = _chain()
    text = plan_delivery_checkout(log, "c1", "c3")
    document = json.loads(text)
    assert list(document) == ["source", "target", "direction", "steps",
                              "snapshot"]
    assert document["source"] == "c1"
    assert document["target"] == "c3"
    assert document["direction"] == "forward"
    assert document["snapshot"] == json.loads(snap3)
    assert document["steps"] == [
        ["c2", "forward", json.loads(snap1), json.loads(snap2)],
        ["c3", "forward", json.loads(snap2), json.loads(snap3)],
    ]
    assert snap0 == _empty_snapshot()


def test_forward_single_step():
    log, (_snap0, snap1, snap2, _snap3) = _chain()
    document = json.loads(plan_delivery_checkout(log, "c1", "c2"))
    assert document["direction"] == "forward"
    assert document["steps"] == [
        ["c2", "forward", json.loads(snap1), json.loads(snap2)]]
    assert document["snapshot"] == json.loads(snap2)


def test_forward_from_root():
    log, snapshots = _chain()
    document = json.loads(plan_delivery_checkout(log, "c1", "c1"))
    assert document["steps"] == []
    assert document["snapshot"] == json.loads(snapshots[1])


# ---------------------------------------------------------------------------
# rollback checkout
# ---------------------------------------------------------------------------

def test_rollback_checkout_shape_and_order():
    log, (_snap0, snap1, snap2, snap3) = _chain()
    text = plan_delivery_checkout(log, "c3", "c1")
    document = json.loads(text)
    assert document["source"] == "c3"
    assert document["target"] == "c1"
    assert document["direction"] == "rollback"
    assert document["snapshot"] == json.loads(snap1)
    assert document["steps"] == [
        ["c3", "rollback", json.loads(snap3), json.loads(snap2)],
        ["c2", "rollback", json.loads(snap2), json.loads(snap1)],
    ]


def test_rollback_single_step():
    log, (_snap0, snap1, snap2, _snap3) = _chain()
    document = json.loads(plan_delivery_checkout(log, "c2", "c1"))
    assert document["direction"] == "rollback"
    assert document["steps"] == [
        ["c2", "rollback", json.loads(snap2), json.loads(snap1)]]
    assert document["snapshot"] == json.loads(snap1)


# ---------------------------------------------------------------------------
# same commit
# ---------------------------------------------------------------------------

def test_same_commit_is_none_with_empty_steps():
    log, (_snap0, _snap1, snap2, _snap3) = _chain()
    text = plan_delivery_checkout(log, "c2", "c2")
    assert text == (
        '{"source":"c2","target":"c2","direction":"none","steps":[],'
        '"snapshot":' + snap2 + "}")
    document = json.loads(text)
    assert document["direction"] == "none"
    assert document["steps"] == []
    assert document["snapshot"] == json.loads(snap2)


# ---------------------------------------------------------------------------
# encoding properties
# ---------------------------------------------------------------------------

def test_compact_encoding_without_trailing_newline():
    log, _snapshots = _chain()
    for text in (plan_delivery_checkout(log, "c1", "c3"),
                 plan_delivery_checkout(log, "c3", "c1"),
                 plan_delivery_checkout(log, "c2", "c2")):
        assert not text.endswith("\n")
        assert "\n" not in text and "\t" not in text
        assert text == json.dumps(json.loads(text), ensure_ascii=False,
                                  separators=(",", ":"))


def test_repeatable_and_inputs_unchanged():
    log, _snapshots = _chain()
    first = plan_delivery_checkout(log, "c1", "c3")
    second = plan_delivery_checkout(log, "c1", "c3")
    assert first == second
    log_back, _ = _chain()
    assert log == log_back


def test_empty_plan_commit_chain():
    snapshot = _ready_p(((0, "1"),))
    log = delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),
                        ("c2", "c1", snapshot, snapshot, EMPTY_PLAN)))
    forward = json.loads(plan_delivery_checkout(log, "c1", "c2"))
    assert forward["steps"] == [
        ["c2", "forward", json.loads(snapshot), json.loads(snapshot)]]
    rollback = json.loads(plan_delivery_checkout(log, "c2", "c1"))
    assert rollback["steps"] == [
        ["c2", "rollback", json.loads(snapshot), json.loads(snapshot)]]


def test_conflict_commit_uses_unchanged_snapshot():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan = plan_delivery_updates(before, after, (("p", 0, 0),))
    other = _ready_p(((0, "5"),))
    result = commit_delivery_updates(other, after, plan)
    assert json.loads(result)["committed"] is False
    log = delivery_log((("c1", None, other, after, plan),))
    document = json.loads(plan_delivery_checkout(log, "c1", "c1"))
    assert document["snapshot"] == json.loads(other)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    log, _snapshots = _chain()
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            plan_delivery_checkout(value, "c1", "c2")
        with pytest.raises(TypeError):
            plan_delivery_checkout(log, value, "c2")
        with pytest.raises(TypeError):
            plan_delivery_checkout(log, "c1", value)


def test_invalid_and_unknown_ids():
    log, _snapshots = _chain()
    for bad_id in ("", "bad id", "a/b"):
        with pytest.raises(ValueError):
            plan_delivery_checkout(log, bad_id, "c1")
        with pytest.raises(ValueError):
            plan_delivery_checkout(log, "c1", bad_id)
    with pytest.raises(ValueError):
        plan_delivery_checkout(log, "c3", "c4")
    with pytest.raises(ValueError):
        plan_delivery_checkout(log, "c4", "c1")
    with pytest.raises(ValueError):
        plan_delivery_checkout(delivery_log(()), "c1", "c1")


def test_malformed_log():
    for log in ("", "not json", "{", '{"commits":[]', "[1,2]",
                '{"commits":[],"head":null} extra',
                '{"commits":[],"head":null}{}'):
        with pytest.raises(ValueError):
            plan_delivery_checkout(log, "c1", "c2")


def test_non_canonical_log_rejected():
    log, _snapshots = _chain()
    pretty = json.dumps(json.loads(log), indent=2)
    with pytest.raises(ValueError):
        plan_delivery_checkout(pretty, "c1", "c2")
    with pytest.raises(ValueError):
        plan_delivery_checkout(log + " ", "c1", "c2")
    with pytest.raises(ValueError):
        plan_delivery_checkout(" " + log, "c1", "c2")


def test_recomputed_result_mismatch():
    log, _snapshots = _chain()
    tampered = log.replace('"committed":true', '"committed":false', 1)
    assert tampered != log
    with pytest.raises(ValueError):
        plan_delivery_checkout(tampered, "c1", "c2")


def test_broken_chain_rejected():
    snapshot = _empty_snapshot()
    result = commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN)
    row1 = '["c1",null,' + snapshot + ',' + snapshot + ',' + EMPTY_PLAN \
        + ',' + result + ']'
    row2 = '["c2","c9",' + snapshot + ',' + snapshot + ',' + EMPTY_PLAN \
        + ',' + result + ']'
    with pytest.raises(ValueError):
        plan_delivery_checkout(
            '{"commits":[' + row1 + ',' + row2 + '],"head":"c2"}',
            "c1", "c2")


def test_dot_dash_underscore_ids_supported():
    _log, (_snap0, snap1, snap2, _snap3) = _chain()
    snapshot0 = _empty_snapshot()
    plan1 = plan_delivery_updates(snapshot0, snap1, (("p", 0, 0),))
    plan2 = plan_delivery_updates(snap1, snap2, (("p", 1, 1),))
    log = delivery_log((("a.b-1_x", None, snapshot0, snap1, plan1),
                        ("a.b-2_x", "a.b-1_x", snap1, snap2, plan2)))
    document = json.loads(
        plan_delivery_checkout(log, "a.b-2_x", "a.b-1_x"))
    assert document["direction"] == "rollback"
    assert [step[0] for step in document["steps"]] == ["a.b-2_x"]
