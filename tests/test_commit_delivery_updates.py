"""Tests for :func:`lidar_scan.commit_delivery_updates`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                         build_delivery_plan, build_delivery_receipt,
                         build_delivery_snapshot, commit_delivery_updates,
                         merge_delivery_manifests, merge_delivery_receipts,
                         plan_delivery_updates, query_delivery_range)
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
    assert tiles_module.commit_delivery_updates is commit_delivery_updates
    assert "commit_delivery_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    snapshot = _ready_p(((0, "1"),))
    for value in (1, None, [], b"snapshot", {}):
        with pytest.raises(TypeError):
            commit_delivery_updates(value, snapshot, EMPTY_PLAN)
    for value in (1, None, [], b"target", {}):
        with pytest.raises(TypeError):
            commit_delivery_updates(snapshot, value, EMPTY_PLAN)
    for value in (1, None, [], b"plan", {}):
        with pytest.raises(TypeError):
            commit_delivery_updates(snapshot, snapshot, value)


def test_value_errors():
    snapshot = _ready_p(((0, "1"),))
    with pytest.raises(ValueError):
        commit_delivery_updates("not json", snapshot, EMPTY_PLAN)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, "not json", EMPTY_PLAN)
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, snapshot, "not json")
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, snapshot,
                                '{"results":[],"changed":false}')


def test_plan_must_pass_replay():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"), (1, "2")))
    plan = plan_delivery_updates(before, after, (("p", 0, 1),))
    tampered = json.dumps(json.loads(plan), indent=2)
    with pytest.raises(ValueError):
        commit_delivery_updates(before, after, tampered)


def test_snapshots_must_be_canonical():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"), (1, "2")))
    plan = plan_delivery_updates(before, after, (("p", 0, 1),))
    with pytest.raises(ValueError):
        commit_delivery_updates(EMPTY_PLAN, after, plan)
    with pytest.raises(ValueError):
        commit_delivery_updates(before, EMPTY_PLAN, plan)


def test_overlapping_ranges_rejected():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2), ("p", 2, 3)))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, after, plan)


def test_target_must_match_plan_target_per_range():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    other = _ready_p(((0, "1"), (1, "2"), (2, "9")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, other, plan)


def test_uncovered_product_must_be_identical():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"), (1, "2")))
    plan = plan_delivery_updates(before, after, (("p", 1, 1),))
    # 'target' carries an extra uncovered product the snapshot lacks.
    items = ((0, "p", "1", PASSING_GATE), (1, "p", "2", PASSING_GATE),
             (0, "q", "1", PASSING_GATE))
    changes = _changes(*(_manifest(item) for item in items))
    receipts = merge_delivery_receipts(
        (build_delivery_receipt(_plan(changes),
                                (("p", "succeeded", ""),
                                 ("q", "succeeded", ""))),))
    tampered_target = build_delivery_snapshot(changes, receipts)
    with pytest.raises(ValueError):
        commit_delivery_updates(before, tampered_target, plan)


def test_uncovered_version_must_be_identical():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan = plan_delivery_updates(before, after, (("p", 2, 2),))
    # 'target' rewrites the uncovered batch 1 as well.
    tampered_target = _ready_p(((0, "1"), (1, "9"), (2, "3")))
    with pytest.raises(ValueError):
        commit_delivery_updates(before, tampered_target, plan)


# ---------------------------------------------------------------------------
# empty plan and output encoding
# ---------------------------------------------------------------------------

def test_empty_plan_requires_identical_snapshots():
    snapshot = _ready_p(((0, "1"),))
    other = _ready_p(((0, "1"), (1, "2")))
    with pytest.raises(ValueError):
        commit_delivery_updates(snapshot, other, EMPTY_PLAN)
    with pytest.raises(ValueError):
        commit_delivery_updates(other, snapshot, EMPTY_PLAN)


def test_empty_plan_commits():
    snapshot = _ready_p(((0, "1"),))
    assert commit_delivery_updates(snapshot, snapshot, EMPTY_PLAN) == \
        '{"snapshot":' + snapshot + ',"results":[],"committed":true}'


def test_output_shape_and_encoding():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan = plan_delivery_updates(before, after, (("p", 0, 2),))
    result = commit_delivery_updates(before, after, plan)
    assert ": " not in result and ", " not in result
    assert not result.endswith("\n")
    document = json.loads(result)
    assert list(document) == ["snapshot", "results", "committed"]
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)
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
# committed / unchanged
# ---------------------------------------------------------------------------

def test_applied_from_base():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))
    target = json.loads(plan)["operations"][0][7]

    document = json.loads(commit_delivery_updates(before, after, plan))
    assert document["committed"] is True
    assert document["snapshot"] == json.loads(after)
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "applied", ""]
    assert row[5] == target


def test_unchanged_at_target():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan = plan_delivery_updates(before, after, (("p", 0, 4),))

    document = json.loads(commit_delivery_updates(after, after, plan))
    assert document["committed"] is True
    row = document["results"][0]
    assert row[:5] == ["p", 0, 4, "unchanged", ""]
    assert row[5] == json.loads(plan)["operations"][0][7]


def test_mixed_applied_and_unchanged():
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
    for row, operation in zip(document["results"],
                              json.loads(plan)["operations"]):
        assert row[5] == operation[7]


# ---------------------------------------------------------------------------
# conflicts and preflight
# ---------------------------------------------------------------------------

def test_conflict_keeps_snapshot():
    before = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (3, "4")))
    divergent = _ready_p(((0, "1"), (1, "2"), (2, "9")))
    plan = plan_delivery_updates(before, after, (("p", 0, 3),))
    # The target still matches the plan; only the snapshot diverged.
    target = after

    document = json.loads(commit_delivery_updates(divergent, target, plan))
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
