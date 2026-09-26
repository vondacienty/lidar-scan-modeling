"""Tests for :func:`lidar_scan.commit_delivery_updates`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, commit_delivery_updates,
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


def _snapshot(items, results=None):
    changes = _changes(*(_manifest(item) for item in items))
    if results is None:
        results = tuple((item[1], "succeeded", "") for item in items[:1])
    receipts = merge_delivery_receipts((build_delivery_receipt(
        _plan(changes), results),)) if results else EMPTY_RECEIPTS
    return build_delivery_snapshot(changes, receipts)


def _ready_p(batches):
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    return _snapshot(items)


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
    assert tiles_module.commit_delivery_updates is commit_delivery_updates
    assert "commit_delivery_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    snapshot = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(snapshot, snapshot, (("p", 0, 0),))
    for value in (1, None, [], b"snapshot", {}):
        with pytest.raises(TypeError):
            commit_delivery_updates(value, snapshot, plan)
        with pytest.raises(TypeError):
            commit_delivery_updates(snapshot, value, plan)
        with pytest.raises(TypeError):
            commit_delivery_updates(snapshot, snapshot, value)


def test_value_errors():
    snapshot = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(snapshot, snapshot, (("p", 0, 0),))
    with pytest.raises(ValueError):
        commit_delivery_updates("not json", snapshot, plan)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, "not json", plan)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, snapshot, "not json")
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, snapshot,
                                '{"results":[],"changed":false}')


def test_plan_must_pass_replay():
    snapshot = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(snapshot, snapshot, (("p", 0, 0),))
    tampered = json.dumps(json.loads(plan), indent=2)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, snapshot, tampered)


def test_snapshots_must_be_canonical():
    snapshot = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(snapshot, snapshot, (("p", 0, 0),))
    with pytest.raises(ValueError):
        commit_delivery_updates(EMPTY_PLAN, snapshot, plan)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, EMPTY_PLAN, plan)


# ---------------------------------------------------------------------------
# empty plan and output encoding
# ---------------------------------------------------------------------------

def test_empty_plan_commits_identical_snapshots():
    snapshot = _ready_p(((0, "1"),))
    document = json.loads(commit_delivery_updates(snapshot, snapshot,
                                                  EMPTY_PLAN))
    assert document["committed"] is True
    assert document["results"] == []
    assert document["snapshot"] == json.loads(snapshot)


def test_empty_plan_requires_equal_snapshots():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, after, EMPTY_PLAN)


def test_output_shape_and_encoding():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    result = commit_delivery_updates(before, after, plan)
    assert ": " not in result and ", " not in result
    assert not result.endswith("\n")
    document = json.loads(result)
    assert list(document) == ["snapshot", "results", "committed"]
    for row in document["results"]:
        assert len(row) == 6


def test_results_keep_plan_order():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2), ("p", 0, 0))
    plan = plan_delivery_updates(before, after, ranges)
    document = json.loads(commit_delivery_updates(before, after, plan))
    assert [[row[0], row[1], row[2]] for row in document["results"]] == \
        [[row[0], row[1], row[2]]
         for row in json.loads(plan)["operations"]]


def test_deterministic_and_inputs_unchanged():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan = plan_delivery_updates(before, after, (("p", 0, 0),))
    snapshot_before, target_before, plan_before = before, after, plan
    first = commit_delivery_updates(before, after, plan)
    second = commit_delivery_updates(before, after, plan)
    assert first == second
    assert before == snapshot_before and after == target_before
    assert plan == plan_before


# ---------------------------------------------------------------------------
# committed snapshots
# ---------------------------------------------------------------------------

def test_commit_adopts_target():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))

    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "applied", ""]
    assert row[5] == json.loads(plan)["operations"][0][7]


def test_commit_unchanged_at_target():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))

    document = json.loads(commit_delivery_updates(after, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "unchanged", ""]
    assert row[5] == json.loads(plan)["operations"][0][7]


def test_commit_mixed_applied_and_unchanged():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    ranges = (("p", 0, 2), ("p", 3, 3), ("q", 0, 0))
    plan = plan_delivery_updates(before, after, ranges)

    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("p", 0, 2)] == ("applied", "")
    assert statuses[("p", 3, 3)] == ("applied", "")
    assert statuses[("q", 0, 0)] == ("unchanged", "")


def test_commit_keeps_uncovered_products():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (1, "p", "2", PASSING_GATE),
                        (0, "q", "q1", PASSING_GATE)),
                       (("p", "succeeded", ""), ("q", "succeeded", "")))
    after = _snapshot(((0, "p", "1", PASSING_GATE),
                       (1, "p", "2", PASSING_GATE),
                       (2, "p", "3", PASSING_GATE),
                       (0, "q", "q1", PASSING_GATE)),
                      (("p", "succeeded", ""), ("q", "succeeded", "")))
    plan = plan_delivery_updates(before, after, (("p", 2, 2),))

    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)


def test_commit_removes_product():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (0, "q", "q1", PASSING_GATE)),
                       (("p", "succeeded", ""), ("q", "succeeded", "")))
    after = _snapshot(((0, "q", "q1", PASSING_GATE),),
                      (("q", "succeeded", ""),))
    plan = plan_delivery_updates(before, after, (("p", 0, 0),))

    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)


# ---------------------------------------------------------------------------
# target consistency validation
# ---------------------------------------------------------------------------

def test_target_range_must_match_plan_target():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    wrong = _ready_p(((0, "1"), (1, "2"), (2, "9")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, wrong, plan)


def test_target_uncovered_items_must_match_snapshot():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _snapshot(((0, "p", "1", PASSING_GATE),
                       (1, "p", "2", PASSING_GATE),
                       (2, "p", "3", PASSING_GATE),
                       (0, "q", "q1", PASSING_GATE)),
                      (("p", "succeeded", ""), ("q", "succeeded", "")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, after, plan)


def test_overlapping_product_ranges_rejected():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 3), ("p", 3, 4)))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, after, plan)


def test_adjacent_product_ranges_allowed():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2), ("p", 3, 4)))
    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)


# ---------------------------------------------------------------------------
# conflicts and preflight
# ---------------------------------------------------------------------------

def test_conflict_keeps_snapshot():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    divergent = _ready_p(((0, "1"), (1, "2"), (2, "9")))
    plan = plan_delivery_updates(before, after, (("p", 0, 3),))

    document = json.loads(commit_delivery_updates(divergent, after, plan))
    assert document["committed"] is False
    assert document["snapshot"] == json.loads(divergent)
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

    document = json.loads(commit_delivery_updates(divergent, after, plan))
    assert document["committed"] is False
    assert document["snapshot"] == json.loads(divergent)
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("p", 0, 2)] == ("conflict", "mismatch")
    assert statuses[("p", 3, 3)] == ("conflict", "mismatch")
    assert statuses[("q", 0, 0)] == ("aborted", "conflict")
    for row in document["results"]:
        assert row[5] == _range_json(divergent, row[0], row[1], row[2])


def test_applicable_operations_aborted_when_another_conflicts():
    empty = _empty_snapshot()
    target_q = _snapshot(((0, "q", "q0", PASSING_GATE),),
                         (("q", "succeeded", ""),))
    plan = plan_delivery_updates(empty, target_q,
                                 (("q", 0, 0), ("z", 0, 0)))
    # Snapshot already past the plan: q diverges, z would be applicable.
    divergent_q = _snapshot(((0, "q", "qx", PASSING_GATE),),
                            (("q", "succeeded", ""),))

    document = json.loads(commit_delivery_updates(divergent_q, target_q,
                                                  plan))
    assert document["committed"] is False
    assert document["snapshot"] == json.loads(divergent_q)
    statuses = {(row[0], row[1], row[2]): (row[3], row[4])
                for row in document["results"]}
    assert statuses[("q", 0, 0)] == ("conflict", "mismatch")
    assert statuses[("z", 0, 0)] == ("aborted", "conflict")


def test_replay_still_roundtrips_alongside_commit():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))
    replay_document = json.loads(replay_delivery_updates(plan))
    commit_document = json.loads(commit_delivery_updates(before, after,
                                                         plan))
    assert commit_document["committed"] is True
    for replay_row, commit_row in zip(replay_document["results"],
                                      commit_document["results"]):
        assert replay_row[:3] == commit_row[:3]
        assert replay_row[4] == commit_row[5]
