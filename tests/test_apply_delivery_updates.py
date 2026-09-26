"""Tests for :func:`lidar_scan.apply_delivery_updates`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (apply_delivery_updates, build_delivery_manifest,
                         build_delivery_snapshot, merge_delivery_manifests,
                         plan_delivery_updates, query_delivery_range)
from lidar_scan import tiles as tiles_module
from lidar_scan.tiles import _ranges_result_to_json

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

EMPTY_RECEIPTS = '{"products":[],"ready":true}'


def _snapshot(items):
    manifests = tuple(
        build_delivery_manifest(((batch, product, version, PASSING_GATE),))
        for batch, product, version in items)
    changes = merge_delivery_manifests(manifests)
    return build_delivery_snapshot(changes, EMPTY_RECEIPTS)


def _plan(before, after, ranges):
    return plan_delivery_updates(before, after, ranges)


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

def test_type_error_for_non_str():
    before = _snapshot(((0, "p", "1"),))
    plan = _plan(before, before, ())
    for value in (1, None, [], b"x", {}):
        with pytest.raises(TypeError):
            apply_delivery_updates(value, plan)
        with pytest.raises(TypeError):
            apply_delivery_updates(before, value)


def test_invalid_plan_raises_value_error():
    before = _snapshot(((0, "p", "1"),))
    with pytest.raises(ValueError):
        apply_delivery_updates(before, "not json")
    plan = _plan(before, before, ())
    with pytest.raises(ValueError):
        apply_delivery_updates(before, plan + "\n")


def test_invalid_snapshot_raises_value_error():
    before = _snapshot(((0, "p", "1"),))
    plan = _plan(before, before, ())
    with pytest.raises(ValueError):
        apply_delivery_updates("not json", plan)


def test_inputs_are_not_modified():
    before = _snapshot(((0, "p", "1"),))
    after = _snapshot(((0, "p", "2"),))
    plan_text = _plan(before, after, (("p", 0, 0),))
    snapshot_kept = plan_text
    before_kept = before
    apply_delivery_updates(before, plan_text)
    assert plan_text == snapshot_kept
    assert before == before_kept


# ---------------------------------------------------------------------------
# empty plan and encoding
# ---------------------------------------------------------------------------

def test_empty_plan_applies():
    before = _snapshot(((0, "p", "1"),))
    assert apply_delivery_updates(
        before, _plan(before, before, ())) == '{"results":[],"applied":true}'


def test_compact_encoding_and_key_order():
    before = _snapshot(((0, "p", "1"),))
    after = _snapshot(((0, "p", "2"),))
    result = apply_delivery_updates(
        before, _plan(before, after, (("p", 0, 0),)))
    assert ": " not in result and ", " not in result
    assert not result.endswith("\n")
    assert list(json.loads(result)) == ["results", "applied"]


# ---------------------------------------------------------------------------
# applied / unchanged
# ---------------------------------------------------------------------------

def test_current_equals_base_is_applied_with_target_state():
    before = _snapshot(((0, "p", "1"), (1, "p", "2")))
    after = _snapshot(((0, "p", "1"), (1, "p", "2"), (2, "p", "3")))
    plan_text = _plan(before, after, (("p", 0, 2),))
    document = json.loads(apply_delivery_updates(before, plan_text))
    assert document["applied"] is True
    row = document["results"][0]
    assert row == ["p", 0, 2, "applied", "",
                   json.loads(plan_text)["operations"][0][7]]


def test_current_equals_target_is_unchanged_with_target_state():
    before = _snapshot(((0, "p", "1"), (1, "p", "2")))
    after = _snapshot(((0, "p", "1"), (1, "p", "2"), (2, "p", "3")))
    plan_text = _plan(before, after, (("p", 0, 2),))
    document = json.loads(apply_delivery_updates(after, plan_text))
    assert document["applied"] is True
    row = document["results"][0]
    assert row[3] == "unchanged" and row[4] == ""
    assert row[5] == json.loads(plan_text)["operations"][0][7]


def test_mixed_applied_and_unchanged():
    before = _snapshot(((0, "p", "1"), (0, "q", "a")))
    after = _snapshot(((0, "p", "2"), (0, "q", "b")))
    plan_text = _plan(before, after, (("p", 0, 0), ("q", 0, 0)))
    current = _snapshot(((0, "p", "1"), (0, "q", "b")))
    document = json.loads(apply_delivery_updates(current, plan_text))
    assert document["applied"] is True
    rows = {row[0]: row for row in document["results"]}
    assert rows["p"][3:5] == ["applied", ""]
    assert rows["q"][3:5] == ["unchanged", ""]
    targets = {op[0]: op[7]
               for op in json.loads(plan_text)["operations"]}
    for row in document["results"]:
        assert row[5] == targets[row[0]]


def test_results_keep_operation_order_and_fields():
    before = _snapshot(((0, "p", "1"), (0, "q", "a")))
    after = _snapshot(((0, "p", "2"), (0, "q", "b")))
    plan_text = _plan(before, after, (("q", 0, 0), ("p", 0, 0)))
    document = json.loads(apply_delivery_updates(before, plan_text))
    assert [row[:3] for row in document["results"]] == \
        [op[:3] for op in json.loads(plan_text)["operations"]]
    for row in document["results"]:
        assert len(row) == 6


# ---------------------------------------------------------------------------
# conflicts and pre-flight
# ---------------------------------------------------------------------------

def test_conflict_is_not_applied_with_current_state():
    before = _snapshot(((0, "p", "1"), (1, "p", "2")))
    after = _snapshot(((0, "p", "1"), (1, "p", "2"), (2, "p", "3")))
    plan_text = _plan(before, after, (("p", 0, 2),))
    diverged = _snapshot(((0, "p", "1"), (1, "p", "9")))
    document = json.loads(apply_delivery_updates(diverged, plan_text))
    assert document["applied"] is False
    row = document["results"][0]
    assert row[3:5] == ["conflict", "mismatch"]
    assert row[5] == _ranges_result_to_json(
        query_delivery_range(diverged, "p", 0, 2))


def test_one_conflict_aborts_every_other_operation():
    before = _snapshot(((0, "p", "1"), (0, "q", "a")))
    after = _snapshot(((0, "p", "2"), (0, "q", "b")))
    plan_text = _plan(before, after, (("p", 0, 0), ("q", 0, 0)))
    # p diverges (conflict); q still sits at its base (would apply).
    current = _snapshot(((0, "p", "9"), (0, "q", "a")))
    document = json.loads(apply_delivery_updates(current, plan_text))
    assert document["applied"] is False
    rows = {row[0]: row for row in document["results"]}
    assert rows["p"][3:5] == ["conflict", "mismatch"]
    assert rows["q"][3:5] == ["aborted", "conflict"]
    for row in document["results"]:
        assert row[5] == _ranges_result_to_json(
            query_delivery_range(current, row[0], 0, 0))


def test_absent_product_conflict_uses_null_state():
    before = _snapshot(((0, "p", "1"), (0, "q", "a")))
    after = _snapshot(((0, "p", "1"), (0, "q", "b")))
    plan_text = _plan(before, after, (("q", 0, 0),))
    # q has a base and target in the plan but is absent from the current
    # snapshot -> conflict whose state is null.
    current = _snapshot(((0, "p", "1"),))
    document = json.loads(apply_delivery_updates(current, plan_text))
    assert document["applied"] is False
    assert document["results"][0][3:5] == ["conflict", "mismatch"]
    assert document["results"][0][5] is None


def test_plan_must_pass_replay():
    before = _snapshot(((0, "p", "1"), (1, "p", "2")))
    after = _snapshot(((0, "p", "1"), (1, "p", "2"), (2, "p", "3")))
    plan_text = _plan(before, after, (("p", 0, 2),))
    document = json.loads(plan_text)
    document["operations"][0][7][0][1][1] = "WRONG"
    bad_plan = json.dumps(document, separators=(",", ":"))
    with pytest.raises(ValueError):
        apply_delivery_updates(before, bad_plan)
