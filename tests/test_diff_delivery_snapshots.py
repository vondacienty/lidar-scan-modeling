"""Tests for :func:`lidar_scan.diff_delivery_snapshots`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, diff_delivery_snapshots,
                        merge_delivery_manifests, merge_delivery_receipts)
from lidar_scan import tiles as tiles_module

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

FAILING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],false],'
    '[0,100,100,109,109,null,false]],"passed":false}'
)

EMPTY_CHANGES = '{"changes":[],"releasable":true}'
EMPTY_RECEIPTS = '{"products":[],"ready":true}'


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


def _plan(changes):
    return build_delivery_plan(audit_delivery_changes(changes))


def _merged(changes, results):
    receipt = build_delivery_receipt(_plan(changes), tuple(results))
    return merge_delivery_receipts((receipt,))


def _snapshot(items, receipt_results=()):
    """Build a snapshot from (batch, product, version, gate) items."""
    changes = _changes(*(_manifest(item) for item in items))
    receipts = (
        _merged(changes, receipt_results) if receipt_results
        else EMPTY_RECEIPTS
    )
    return build_delivery_snapshot(changes, receipts)


def _ready_p(batches):
    """Product ``p`` with the given (batch, version) pairs, all passing."""
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    changes = _changes(*(_manifest(item) for item in items))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    return build_delivery_snapshot(changes, receipts)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.diff_delivery_snapshots is diff_delivery_snapshots
    assert "diff_delivery_snapshots" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(1, after, ())
    with pytest.raises(TypeError):
        diff_delivery_snapshots(None, after, ())
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, 1, ())
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, None, ())
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, [])
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (["p", 0, 1],))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", 0, 1, 2),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, ((1, 0, 1),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", True, 1),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", 0, False),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", "0", 1),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", 0, 1.5),))
    with pytest.raises(TypeError):
        diff_delivery_snapshots(before, after, (("p", None, 1),))


def test_value_errors():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    with pytest.raises(ValueError):
        diff_delivery_snapshots("not json", after, ())
    with pytest.raises(ValueError):
        diff_delivery_snapshots(before, "not json", ())
    with pytest.raises(ValueError):
        diff_delivery_snapshots(before, after, (("p", -1, 1),))
    with pytest.raises(ValueError):
        diff_delivery_snapshots(before, after, (("p", 0, -1),))
    with pytest.raises(ValueError):
        diff_delivery_snapshots(before, after, (("p", 2, 1),))
    with pytest.raises(ValueError):
        diff_delivery_snapshots(
            before, after, (("p", 0, 1), ("p", 0, 1)))


def test_non_canonical_snapshot_rejected():
    good = _ready_p(((0, "1"),))
    document = json.loads(good)
    rebuilt = json.dumps(document, indent=2)
    with pytest.raises(ValueError):
        diff_delivery_snapshots(rebuilt, good, ())
    with pytest.raises(ValueError):
        diff_delivery_snapshots(good, rebuilt, ())


# ---------------------------------------------------------------------------
# output shape and sorting
# ---------------------------------------------------------------------------

def test_empty_ranges_document():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    assert diff_delivery_snapshots(before, after, ()) == \
        '{"ranges":[],"changed":false}'


def test_top_level_key_order_and_sorting():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    text = diff_delivery_snapshots(
        before, after, (("q", 0, 0), ("p", 2, 2), ("p", 0, 0)))
    assert text.index('"ranges"') < text.index('"changed"')
    document = json.loads(text)
    assert list(document) == ["ranges", "changed"]
    assert [row[:3] for row in document["ranges"]] == [
        ["p", 0, 0], ["p", 2, 2], ["q", 0, 0]]


def test_reordering_inputs_is_byte_identical():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2), ("p", 0, 0))
    first = diff_delivery_snapshots(before, after, ranges)
    second = diff_delivery_snapshots(before, after, tuple(reversed(ranges)))
    assert first == second


def test_row_has_eight_fields():
    before = _ready_p(((0, "1"),))
    after = before
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 0),)))
    row = document["ranges"][0]
    assert len(row) == 8
    assert row[:3] == ["p", 0, 0]


def test_compact_encoding_has_no_whitespace_or_newline():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    text = diff_delivery_snapshots(before, after, (("p", 0, 2),))
    assert ": " not in text
    assert ", " not in text
    assert not text.endswith("\n")


# ---------------------------------------------------------------------------
# B/A sides
# ---------------------------------------------------------------------------

def test_b_and_a_are_range_query_results_as_arrays():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 1),)))
    b_side, a_side = document["ranges"][0][3], document["ranges"][0][4]
    assert b_side == [
        [[0, None, "1", True], [1, "1", "2", True]],
        [],
        ["publish", "2", "2", [["succeeded", ""]], "succeeded"],
        True,
    ]
    assert a_side == [
        [[0, None, "1", True], [1, "1", "2", True]],
        [],
        ["publish", "3", "3", [["succeeded", ""]], "succeeded"],
        True,
    ]


def test_absent_product_sides_are_null_treated_as_empty_versions():
    before = _ready_p(((0, "1"),))
    after = _snapshot(
        ((0, "q", "a", FAILING_GATE),),
        (("q", "blocked", "held"),))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("q", 0, 0), ("z", 0, 0))))
    q_row, z_row = document["ranges"]
    assert q_row[3] is None
    assert q_row[4] is not None
    assert q_row[4][0] == [[0, None, "a", False]]
    assert z_row[3] is None and z_row[4] is None
    assert z_row[5:8] == [[], [], []]


# ---------------------------------------------------------------------------
# added / removed / modified
# ---------------------------------------------------------------------------

def test_added_lists_after_only_batches_in_order():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 4),)))
    row = document["ranges"][0]
    assert row[5] == [[2, "2", "3", True], [4, "3", "5", True]]
    assert row[6] == []
    assert row[7] == []


def test_removed_lists_before_only_batches_in_order():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (2, "p", "2", PASSING_GATE)))
    after = _snapshot(((0, "p", "1", PASSING_GATE),))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 2),)))
    row = document["ranges"][0]
    assert row[5] == []
    assert row[6] == [[2, "1", "2", True]]
    assert row[7] == []


def test_entire_product_removed():
    before = _ready_p(((0, "1"),))
    after = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 0),)))
    row = document["ranges"][0]
    assert row[3] is not None and row[4] is None
    assert row[5] == []
    assert row[6] == [[0, None, "1", True]]
    assert row[7] == []


def test_modified_shared_batch_with_different_record():
    before = _snapshot(((0, "m", "1", PASSING_GATE),))
    after = _snapshot(((0, "m", "1", FAILING_GATE),))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("m", 0, 0),)))
    row = document["ranges"][0]
    assert row[5] == [] and row[6] == []
    assert row[7] == [
        [0, [0, None, "1", True], [0, None, "1", False]]]


def test_modified_sorted_with_added_and_removed():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (1, "p", "2", PASSING_GATE)))
    after = _snapshot(((0, "p", "9", PASSING_GATE),
                       (2, "p", "3", PASSING_GATE)))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 2),)))
    added, removed, modified = document["ranges"][0][5:8]
    assert added == [[2, "9", "3", True]]
    assert removed == [[1, "1", "2", True]]
    assert modified == [
        [0, [0, None, "1", True], [0, None, "9", True]]]


def test_differences_only_within_requested_range():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (1, "p", "2", PASSING_GATE)))
    after = _snapshot(((0, "p", "1", PASSING_GATE),
                       (1, "p", "2", PASSING_GATE),
                       (2, "p", "3", PASSING_GATE)))
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 1),)))
    row = document["ranges"][0]
    # Batch 2 exists only in after but lies outside the requested range.
    assert row[5] == [] and row[6] == [] and row[7] == []
    assert document["changed"] is False


# ---------------------------------------------------------------------------
# changed flag
# ---------------------------------------------------------------------------

def test_changed_false_for_identical_snapshots():
    before = _ready_p(((0, "1"), (1, "2")))
    document = json.loads(diff_delivery_snapshots(
        before, before, (("p", 0, 2), ("q", 5, 6))))
    assert document["changed"] is False


def test_changed_true_when_only_receipt_differs():
    before = _snapshot(((0, "p", "1", PASSING_GATE),))
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    after = build_delivery_snapshot(
        changes, _merged(changes, (("p", "succeeded", ""),)))
    # Versions identical; receipt/re readiness differ.
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 0),)))
    row = document["ranges"][0]
    assert row[5] == [] and row[6] == [] and row[7] == []
    assert document["changed"] is True


def test_changed_true_for_any_unequal_range():
    before = _snapshot(((0, "p", "1", PASSING_GATE),))
    after = _snapshot(((0, "p", "1", PASSING_GATE),
                       (1, "p", "2", PASSING_GATE)))
    # Sorted first, ("p", 5, 9) is empty with identical gaps on both sides;
    # ("p", 0, 0) is equal too, but ("p", 0, 1) picks up the new batch.
    document = json.loads(diff_delivery_snapshots(
        before, after, (("p", 0, 1), ("p", 5, 9))))
    assert document["ranges"][0][3] != document["ranges"][0][4]
    assert document["ranges"][1][3] == document["ranges"][1][4]
    assert document["changed"] is True
