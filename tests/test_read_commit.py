"""Tests for :func:`lidar_scan.read_commit`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, commit_delivery_updates,
                        delivery_log, merge_delivery_manifests,
                        merge_delivery_receipts, plan_delivery_updates,
                        read_commit)
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
    plan1 = plan_delivery_updates(snap0, snap1, (("p", 0, 0),))
    plan2 = plan_delivery_updates(snap1, snap2, (("p", 1, 1),))
    entries = (("c1", None, snap0, snap1, plan1),
               ("c2", "c1", snap1, snap2, plan2))
    return delivery_log(entries), (snap0, snap1, snap2)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.read_commit is read_commit
    assert "read_commit" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------

def test_read_result_snapshot_by_default():
    log, (snap0, snap1, snap2) = _chain()
    assert read_commit(log, "c1") == snap1
    assert read_commit(log, "c2") == snap2
    assert read_commit(log, "c2", False) == snap2
    assert read_commit(log, "c2", before=False) == snap2


def test_read_entry_snapshot_with_before():
    log, (snap0, snap1, snap2) = _chain()
    assert read_commit(log, "c1", before=True) == snap0
    assert read_commit(log, "c2", True) == snap1
    assert read_commit(log, "c2", before=True) == snap1


def test_read_empty_plan_commit():
    snapshot = _ready_p(((0, "1"),))
    log = delivery_log((("c1", None, snapshot, snapshot, EMPTY_PLAN),))
    assert read_commit(log, "c1") == snapshot
    assert read_commit(log, "c1", before=True) == snapshot


def test_read_conflict_commit_returns_input_snapshot():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan = plan_delivery_updates(before, after, (("p", 0, 0),))
    other = _ready_p(((0, "5"),))
    result = commit_delivery_updates(other, after, plan)
    assert json.loads(result)["committed"] is False
    log = delivery_log((("c1", None, other, after, plan),))
    assert read_commit(log, "c1") == other
    assert read_commit(log, "c1", before=True) == other


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    log, _snapshots = _chain()
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            read_commit(value, "c1")
        with pytest.raises(TypeError):
            read_commit(log, value)
    for value in (1, "x", None, [], 0.5):
        with pytest.raises(TypeError):
            read_commit(log, "c1", value)


def test_invalid_and_unknown_ids():
    log, _snapshots = _chain()
    for bad_id in ("", "bad id", "a/b"):
        with pytest.raises(ValueError):
            read_commit(log, bad_id)
    with pytest.raises(ValueError):
        read_commit(log, "c3")
    with pytest.raises(ValueError):
        read_commit('{"commits":[],"head":null}', "c1")


# ---------------------------------------------------------------------------
# log validation
# ---------------------------------------------------------------------------

def test_malformed_json():
    for log in ("", "not json", "{", '{"commits":[]', "[1,2]",
                '{"commits":[],"head":null} extra',
                '{"commits":[],"head":null}{}'):
        with pytest.raises(ValueError):
            read_commit(log, "c1")


def test_document_structure_errors():
    snapshot = _empty_snapshot()
    row = '["c1",null,' + snapshot + ',' + snapshot + ',' + EMPTY_PLAN \
        + ',' + commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN) + ']'
    for log in ('{"commits":[]}',
                '{"head":null,"commits":[]}',
                '{"commits":[],"head":null,"extra":1}',
                '{"commits":{},"head":null}',
                '{"commits":[["c1"]],"head":"c1"}',
                '{"commits":[["c1",null,{},' + snapshot + ','
                + EMPTY_PLAN + ',{}]],"head":"c1"}',
                '{"commits":[[' + row[1:-1] + ']],"head":"c2"}',
                '{"commits":[[' + row[1:-1] + ']],"head":null}',
                '{"commits":[[' + row[1:-1] + ']],"head":1}'):
        with pytest.raises(ValueError):
            read_commit(log, "c1")


def test_field_type_errors_inside_log():
    snapshot = _empty_snapshot()
    result = commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN)
    good = ["c1", None, snapshot, snapshot, EMPTY_PLAN, result]
    for index, value in ((0, 1), (1, 1), (2, "x"), (3, 1), (4, []),
                         (5, "x")):
        row = list(good)
        row[index] = value
        log = '{"commits":[[' + ",".join(
            item if isinstance(item, str) and item.startswith("{")
            else json.dumps(item) for item in row) + ']],"head":"c1"}'
        with pytest.raises(ValueError):
            read_commit(log, "c1")


def test_recomputed_result_mismatch():
    log, _snapshots = _chain()
    tampered = log.replace('"committed":true', '"committed":false', 1)
    assert tampered != log
    with pytest.raises(ValueError):
        read_commit(tampered, "c1")


def test_broken_chain_rejected():
    snapshot = _empty_snapshot()
    result = commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN)
    row1 = '["c1",null,' + snapshot + ',' + snapshot + ',' + EMPTY_PLAN \
        + ',' + result + ']'
    row2 = '["c2","c9",' + snapshot + ',' + snapshot + ',' + EMPTY_PLAN \
        + ',' + result + ']'
    with pytest.raises(ValueError):
        read_commit('{"commits":[' + row1 + ',' + row2 + '],"head":"c2"}',
                    "c2")


def test_non_canonical_log_rejected():
    log, _snapshots = _chain()
    pretty = json.dumps(json.loads(log), indent=2)
    with pytest.raises(ValueError):
        read_commit(pretty, "c1")
    with pytest.raises(ValueError):
        read_commit(log + " ", "c1")
    with pytest.raises(ValueError):
        read_commit(" " + log, "c1")


def test_log_not_modified_and_repeatable():
    log, (snap0, snap1, snap2) = _chain()
    assert read_commit(log, "c2") == snap2
    assert read_commit(log, "c2") == snap2
