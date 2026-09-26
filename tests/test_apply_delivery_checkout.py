"""Tests for :func:`lidar_scan.apply_delivery_checkout`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (apply_delivery_checkout, audit_delivery_changes,
                        build_delivery_manifest, build_delivery_plan,
                        build_delivery_receipt, build_delivery_snapshot,
                        commit_delivery_updates, delivery_log,
                        merge_delivery_manifests, merge_delivery_receipts,
                        plan_delivery_checkout, plan_delivery_updates)
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
    assert tiles_module.apply_delivery_checkout is apply_delivery_checkout
    assert "apply_delivery_checkout" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# applied checkouts
# ---------------------------------------------------------------------------

def test_forward_checkout_applied():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    text = apply_delivery_checkout(log, snap1, plan)
    document = json.loads(text)
    assert list(document) == ["source", "target", "direction", "status",
                              "snapshot"]
    assert document["source"] == "c1"
    assert document["target"] == "c3"
    assert document["direction"] == "forward"
    assert document["status"] == "applied"
    assert document["snapshot"] == json.loads(snap3)


def test_rollback_checkout_applied():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    plan = plan_delivery_checkout(log, "c3", "c1")
    document = json.loads(apply_delivery_checkout(log, snap3, plan))
    assert document["direction"] == "rollback"
    assert document["status"] == "applied"
    assert document["snapshot"] == json.loads(snap1)


def test_exact_forward_output_bytes():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    assert apply_delivery_checkout(log, snap1, plan) == (
        '{"source":"c1","target":"c3","direction":"forward",'
        '"status":"applied","snapshot":' + snap3 + "}")


# ---------------------------------------------------------------------------
# unchanged checkouts
# ---------------------------------------------------------------------------

def test_current_at_target_is_unchanged():
    log, (_snap0, _snap1, _snap2, snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    document = json.loads(apply_delivery_checkout(log, snap3, plan))
    assert document["status"] == "unchanged"
    assert document["snapshot"] == json.loads(snap3)


def test_same_commit_checkout_is_unchanged():
    log, (_snap0, _snap1, snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c2", "c2")
    text = apply_delivery_checkout(log, snap2, plan)
    assert text == ('{"source":"c2","target":"c2","direction":"none",'
                    '"status":"unchanged","snapshot":' + snap2 + "}")


def test_unchanged_wins_when_source_and_target_snapshots_match():
    snapshot = _ready_p(((0, "1"),))
    log = delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),
                        ("c2", "c1", snapshot, snapshot, EMPTY_PLAN)))
    plan = plan_delivery_checkout(log, "c1", "c2")
    document = json.loads(apply_delivery_checkout(log, snapshot, plan))
    assert document["status"] == "unchanged"
    assert document["snapshot"] == json.loads(snapshot)


# ---------------------------------------------------------------------------
# stale or conflicting current
# ---------------------------------------------------------------------------

def test_current_matching_neither_side_rejected():
    log, (_snap0, _snap1, snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap2, plan)
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, _empty_snapshot(), plan)


# ---------------------------------------------------------------------------
# encoding properties
# ---------------------------------------------------------------------------

def test_compact_encoding_without_trailing_newline():
    log, (_snap0, snap1, _snap2, snap3) = _chain()
    texts = [
        apply_delivery_checkout(log, snap1,
                                plan_delivery_checkout(log, "c1", "c3")),
        apply_delivery_checkout(log, snap3,
                                plan_delivery_checkout(log, "c3", "c1")),
        apply_delivery_checkout(log, snap3,
                                plan_delivery_checkout(log, "c1", "c3")),
    ]
    for text in texts:
        assert not text.endswith("\n")
        assert "\n" not in text and "\t" not in text
        assert text == json.dumps(json.loads(text), ensure_ascii=False,
                                  separators=(",", ":"))


def test_repeatable_and_inputs_unchanged():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    first = apply_delivery_checkout(log, snap1, plan)
    second = apply_delivery_checkout(log, snap1, plan)
    assert first == second
    log_back, _ = _chain()
    assert log == log_back
    assert plan == plan_delivery_checkout(log, "c1", "c3")


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            apply_delivery_checkout(value, snap1, plan)
        with pytest.raises(TypeError):
            apply_delivery_checkout(log, value, plan)
        with pytest.raises(TypeError):
            apply_delivery_checkout(log, snap1, value)


def test_malformed_documents_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    for bad in ("", "not json", "{", "[1,2]", '{"source":"c1"', "null"):
        with pytest.raises(ValueError):
            apply_delivery_checkout(bad, snap1, plan)
        with pytest.raises(ValueError):
            apply_delivery_checkout(log, bad, plan)
        with pytest.raises(ValueError):
            apply_delivery_checkout(log, snap1, bad)


def test_non_canonical_documents_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    with pytest.raises(ValueError):
        apply_delivery_checkout(log + " ", snap1, plan)
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, json.dumps(json.loads(snap1),
                                                indent=2), plan)
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap1, json.dumps(json.loads(plan),
                                                       indent=2))
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap1, plan + " ")


def test_plan_mismatch_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    other = plan_delivery_checkout(log, "c1", "c2")
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap1,
                                other.replace('"target":"c2"',
                                              '"target":"c3"'))
    tampered = plan_delivery_checkout(log, "c1", "c3")
    tampered = tampered.replace('"direction":"forward"',
                                '"direction":"rollback"')
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap1, tampered)


def test_plan_unknown_commit_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    bad = plan.replace('"target":"c3"', '"target":"c9"')
    with pytest.raises(ValueError):
        apply_delivery_checkout(log, snap1, bad)


def test_recomputed_log_mismatch_rejected():
    log, (_snap0, snap1, _snap2, _snap3) = _chain()
    plan = plan_delivery_checkout(log, "c1", "c3")
    tampered = log.replace('"committed":true', '"committed":false', 1)
    assert tampered != log
    with pytest.raises(ValueError):
        apply_delivery_checkout(tampered, snap1, plan)


def test_conflict_commit_target_snapshot_used():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan_updates = plan_delivery_updates(before, after, (("p", 0, 0),))
    other = _ready_p(((0, "5"),))
    result = commit_delivery_updates(other, after, plan_updates)
    assert json.loads(result)["committed"] is False
    log = delivery_log((("c1", None, other, after, plan_updates),))
    plan = plan_delivery_checkout(log, "c1", "c1")
    document = json.loads(apply_delivery_checkout(log, other, plan))
    assert document["status"] == "unchanged"
    assert document["snapshot"] == json.loads(other)
