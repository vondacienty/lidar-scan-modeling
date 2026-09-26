"""Tests for :func:`lidar_scan.apply_delivery_updates`."""

from __future__ import annotations

import copy
import json

import pytest

from lidar_scan import (apply_delivery_updates, audit_delivery_changes,
                         build_delivery_manifest, build_delivery_plan,
                         build_delivery_receipt, build_delivery_snapshot,
                         merge_delivery_manifests, merge_delivery_receipts,
                         plan_delivery_updates, query_delivery_range,
                         replay_delivery_updates)
from lidar_scan import tiles as tiles_module

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

EMPTY_CHANGES = '{"changes":[],"releasable":true}'
EMPTY_RECEIPTS = '{"products":[],"ready":true}'
EMPTY_PLAN = '{"operations":[],"changed":false}'


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


def _plan(changes):
    return build_delivery_plan(audit_delivery_changes(changes))


def _merged(changes, results, product="p"):
    receipt = build_delivery_receipt(_plan(changes),
                                     tuple((product,) + result
                                           for result in results))
    return merge_delivery_receipts((receipt,))


def _snapshot(items, result=("succeeded", "")):
    changes = _changes(*(_manifest(item) for item in items))
    product = items[0][1]
    receipts = _merged(changes, (result,), product) if result else EMPTY_RECEIPTS
    return build_delivery_snapshot(changes, receipts)


def _ready_p(batches):
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    changes = _changes(*(_manifest(item) for item in items))
    receipts = _merged(changes, (("succeeded", ""),))
    return build_delivery_snapshot(changes, receipts)


def _empty_snapshot():
    return build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)


def _range_json(snapshot, product, batch_min, batch_max):
    value = query_delivery_range(snapshot, product, batch_min, batch_max)
    return json.loads(json.dumps(
        value, default=lambda item: list(item) if isinstance(item, tuple)
        else item))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.apply_delivery_updates is apply_delivery_updates
    assert "apply_delivery_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    snapshot = _ready_p(((0, "1"),))
    for value in (1, None, [], b"snapshot", {}):
        with pytest.raises(TypeError):
            apply_delivery_updates(value, EMPTY_PLAN)
    for value in (1, None, [], b"plan", {}):
        with pytest.raises(TypeError):
            apply_delivery_updates(snapshot, value)


def test_value_errors():
    snapshot = _ready_p(((0, "1"),))
    with pytest.raises(ValueError):
        apply_delivery_updates("not json", EMPTY_PLAN)
    with pytest.raises(ValueError):
        apply_delivery_updates(snapshot, "not json")
    with pytest.raises(ValueError):
        apply_delivery_updates(snapshot, '{"results":[],"changed":false}')


def test_plan_must_pass_replay():
    snapshot = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(snapshot, snapshot, (("p", 0, 0),))
    tampered = json.dumps(json.loads(plan), indent=2)
    with pytest.raises(ValueError):
        apply_delivery_updates(snapshot, tampered)


def test_snapshot_must_be_canonical():
    plan = plan_delivery_updates(
        _ready_p(((0, "1"),)), _ready_p(((0, "1"),)), (("p", 0, 0),))
    with pytest.raises(ValueError):
        apply_delivery_updates(EMPTY_PLAN, plan)


# ---------------------------------------------------------------------------
# empty plan and output encoding
# ---------------------------------------------------------------------------

def test_empty_plan_applies():
    snapshot = _ready_p(((0, "1"),))
    assert apply_delivery_updates(snapshot, EMPTY_PLAN) == \
        '{"results":[],"applied":true}'


def test_output_shape_and_encoding():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    result = apply_delivery_updates(before, plan)
    assert ": " not in result and ", " not in result
    assert not result.endswith("\n")
    document = json.loads(result)
    assert list(document) == ["results", "applied"]
    for row in document["results"]:
        assert len(row) == 6


def test_results_keep_plan_order():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2), ("p", 0, 0))
    plan = plan_delivery_updates(before, after, ranges)
    document = json.loads(apply_delivery_updates(before, plan))
    assert [[row[0], row[1], row[2]] for row in document["results"]] == \
        [[row[0], row[1], row[2]]
         for row in json.loads(plan)["operations"]]


def test_deterministic_and_inputs_unchanged():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan = plan_delivery_updates(before, after, (("p", 0, 0),))
    snapshot_before = before
    plan_before = plan
    first = apply_delivery_updates(before, plan)
    second = apply_delivery_updates(before, plan)
    assert first == second
    assert before == snapshot_before and plan == plan_before


# ---------------------------------------------------------------------------
# applied / unchanged
# ---------------------------------------------------------------------------

def test_applied_from_base():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))
    target = json.loads(plan)["operations"][0][7]

    document = json.loads(apply_delivery_updates(before, plan))
    assert document["applied"] is True
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "applied", ""]
    assert row[5] == target


def test_unchanged_at_target():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))

    document = json.loads(apply_delivery_updates(after, plan))
    assert document["applied"] is True
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "unchanged", ""]
    assert row[5] == json.loads(plan)["operations"][0][7]


def test_mixed_applied_and_unchanged():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    ranges = (("p", 0, 2), ("p", 3, 3), ("q", 0, 0))
    plan = plan_delivery_updates(before, after, ranges)

    document = json.loads(apply_delivery_updates(before, plan))
    assert document["applied"] is True
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("p", 0, 2)] == ("applied", "")
    assert statuses[("p", 3, 3)] == ("applied", "")
    assert statuses[("q", 0, 0)] == ("unchanged", "")
    for row, operation in zip(document["results"],
                              json.loads(plan)["operations"]):
        assert row[5] == operation[7]


def test_equal_base_and_target_is_unchanged():
    snapshot = _ready_p(((0, "1"), (1, "2")))
    plan = plan_delivery_updates(snapshot, snapshot,
                                 (("p", 0, 2), ("q", 5, 6)))
    document = json.loads(apply_delivery_updates(snapshot, plan))
    assert document["applied"] is True
    assert all(row[3] == "unchanged" and row[4] == ""
               for row in document["results"])


# ---------------------------------------------------------------------------
# conflicts and preflight
# ---------------------------------------------------------------------------

def test_conflict_rows():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    divergent = _ready_p(((0, "1"), (1, "2"), (2, "9")))
    plan = plan_delivery_updates(before, after, (("p", 0, 3),))

    document = json.loads(apply_delivery_updates(divergent, plan))
    assert document["applied"] is False
    row = document["results"][0]
    assert row[:5] == ["p", 0, 3, "conflict", "mismatch"]
    assert row[5] == _range_json(divergent, "p", 0, 3)


def test_conflict_aborts_every_other_operation():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    ranges = (("p", 0, 2), ("p", 3, 3), ("q", 0, 0))
    plan = plan_delivery_updates(before, after, ranges)
    # 'p' diverges from both base and target; 'q' is absent everywhere.
    divergent = _ready_p(((0, "1"), (1, "2"), (2, "9")))

    document = json.loads(apply_delivery_updates(divergent, plan))
    assert document["applied"] is False
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("p", 0, 2)] == ("conflict", "mismatch")
    assert statuses[("p", 3, 3)] == ("conflict", "mismatch")
    assert statuses[("q", 0, 0)] == ("aborted", "conflict")
    for row in document["results"]:
        assert row[5] == _range_json(divergent, row[0], row[1], row[2])


def test_applicable_operations_aborted_when_another_conflicts():
    empty = _empty_snapshot()
    target_q = _snapshot(((0, "q", "q0", PASSING_GATE),))
    plan = plan_delivery_updates(empty, target_q,
                                 (("q", 0, 0), ("z", 0, 0)))
    # Snapshot already past the plan: q diverges, z would be applicable.
    divergent_q = _snapshot(((0, "q", "qx", PASSING_GATE),))

    document = json.loads(apply_delivery_updates(divergent_q, plan))
    assert document["applied"] is False
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("q", 0, 0)] == ("conflict", "mismatch")
    assert statuses[("z", 0, 0)] == ("aborted", "conflict")
    for row in document["results"]:
        assert row[5] == _range_json(divergent_q, row[0], row[1], row[2])


def test_replay_still_roundtrips_alongside_apply():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))
    replay_document = json.loads(replay_delivery_updates(plan))
    apply_document = json.loads(apply_delivery_updates(before, plan))
    assert apply_document["applied"] is True
    for replay_row, apply_row in zip(replay_document["results"],
                                     apply_document["results"]):
        assert replay_row[:3] == apply_row[:3]
        assert replay_row[4] == apply_row[5]
