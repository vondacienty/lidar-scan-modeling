"""Tests for :func:`lidar_scan.plan_delivery_updates`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, merge_delivery_manifests,
                        merge_delivery_receipts, plan_delivery_updates)
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


def _gaps(records, batch_min, batch_max):
    gaps = []
    expected = batch_min
    for record in records:
        batch = record[0]
        if batch > expected:
            gaps.append([expected, batch - 1])
        expected = batch + 1
    if expected <= batch_max:
        gaps.append([expected, batch_max])
    return gaps


def _replay(row):
    """Replay the planned operations on base and return the new side."""
    _product, batch_min, batch_max, base, actions, receipt, ready, \
        target = row
    records = {} if base is None else {record[0]: list(record)
                                       for record in base[0]}
    for action in actions:
        if action[0] == "add":
            records[action[1][0]] = list(action[1])
        elif action[0] == "remove":
            del records[action[1][0]]
        else:
            assert action[0] == "replace"
            records[action[2][0]] = list(action[2])
    if target is None:
        assert not records
        assert receipt[1] is None and ready[1] is None
        return None
    replayed = [
        [records[batch] for batch in sorted(records)],
        _gaps([records[batch] for batch in sorted(records)],
              batch_min, batch_max),
        receipt[1],
        ready[1],
    ]
    assert replayed == target
    return replayed


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.plan_delivery_updates is plan_delivery_updates
    assert "plan_delivery_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    with pytest.raises(TypeError):
        plan_delivery_updates(1, after, ())
    with pytest.raises(TypeError):
        plan_delivery_updates(None, after, ())
    with pytest.raises(TypeError):
        plan_delivery_updates(before, 1, ())
    with pytest.raises(TypeError):
        plan_delivery_updates(before, None, ())
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, [])
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (["p", 0, 1],))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", 0, 1, 2),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, ((1, 0, 1),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", True, 1),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", 0, False),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", "0", 1),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", 0, 1.5),))
    with pytest.raises(TypeError):
        plan_delivery_updates(before, after, (("p", None, 1),))


def test_value_errors():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    with pytest.raises(ValueError):
        plan_delivery_updates("not json", after, ())
    with pytest.raises(ValueError):
        plan_delivery_updates(before, "not json", ())
    with pytest.raises(ValueError):
        plan_delivery_updates(before, after, (("p", -1, 1),))
    with pytest.raises(ValueError):
        plan_delivery_updates(before, after, (("p", 0, -1),))
    with pytest.raises(ValueError):
        plan_delivery_updates(before, after, (("p", 2, 1),))
    with pytest.raises(ValueError):
        plan_delivery_updates(
            before, after, (("p", 0, 1), ("p", 0, 1)))


def test_non_canonical_snapshot_rejected():
    good = _ready_p(((0, "1"),))
    document = json.loads(good)
    rebuilt = json.dumps(document, indent=2)
    with pytest.raises(ValueError):
        plan_delivery_updates(rebuilt, good, ())
    with pytest.raises(ValueError):
        plan_delivery_updates(good, rebuilt, ())


# ---------------------------------------------------------------------------
# output shape and sorting
# ---------------------------------------------------------------------------

def test_empty_ranges_document():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"),))
    assert plan_delivery_updates(before, after, ()) == \
        '{"operations":[],"changed":false}'


def test_top_level_key_order_and_sorting():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    text = plan_delivery_updates(
        before, after, (("q", 0, 0), ("p", 2, 2), ("p", 0, 0)))
    assert text.index('"operations"') < text.index('"changed"')
    document = json.loads(text)
    assert list(document) == ["operations", "changed"]
    assert [row[:3] for row in document["operations"]] == [
        ["p", 0, 0], ["p", 2, 2], ["q", 0, 0]]


def test_reordering_inputs_is_byte_identical():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2), ("p", 0, 0))
    first = plan_delivery_updates(before, after, ranges)
    second = plan_delivery_updates(before, after, tuple(reversed(ranges)))
    assert first == second


def test_row_has_eight_fields_in_order():
    before = _ready_p(((0, "1"),))
    document = json.loads(plan_delivery_updates(
        before, before, (("p", 0, 0),)))
    row = document["operations"][0]
    assert len(row) == 8
    assert row[:3] == ["p", 0, 0]
    assert row[3] == row[7]


def test_compact_encoding_has_no_whitespace_or_newline():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    text = plan_delivery_updates(before, after, (("p", 0, 2),))
    assert ": " not in text
    assert ", " not in text
    assert not text.endswith("\n")


def test_inputs_not_modified():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2))
    plan_delivery_updates(before, after, ranges)
    assert ranges == (("q", 0, 0), ("p", 2, 2))


# ---------------------------------------------------------------------------
# base / target and receipt / ready pairs
# ---------------------------------------------------------------------------

def test_base_and_target_are_range_query_results():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 1),)))
    row = document["operations"][0]
    assert row[3] == [
        [[0, None, "1", True], [1, "1", "2", True]],
        [],
        ["publish", "2", "2", [["succeeded", ""]], "succeeded"],
        True,
    ]
    assert row[7] == [
        [[0, None, "1", True], [1, "1", "2", True]],
        [],
        ["publish", "3", "3", [["succeeded", ""]], "succeeded"],
        True,
    ]


def test_receipt_and_ready_pairs():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 1),)))
    row = document["operations"][0]
    assert row[5] == [
        ["publish", "2", "2", [["succeeded", ""]], "succeeded"],
        ["publish", "3", "3", [["succeeded", ""]], "succeeded"],
    ]
    assert row[6] == [True, True]


def test_absent_product_sides_are_null():
    before = _ready_p(((0, "1"),))
    after = _snapshot(
        ((0, "q", "a", FAILING_GATE),),
        (("q", "blocked", "held"),))
    document = json.loads(plan_delivery_updates(
        before, after, (("q", 0, 0), ("z", 0, 0))))
    q_row, z_row = document["operations"]
    assert q_row[3] is None and q_row[7] is not None
    assert q_row[5] == [
        None, ["block", "a", None, [["blocked", "held"]], "blocked"]]
    assert q_row[6] == [None, False]
    assert z_row[3] is None and z_row[7] is None
    assert z_row[4] == []
    assert z_row[5] == [None, None]
    assert z_row[6] == [None, None]


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def test_add_actions_in_batch_order():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 4),)))
    assert document["operations"][0][4] == [
        ["add", [2, "2", "3", True]],
        ["add", [4, "3", "5", True]],
    ]


def test_remove_actions_in_batch_order():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (2, "p", "2", PASSING_GATE)))
    after = _snapshot(((0, "p", "1", PASSING_GATE),))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 2),)))
    assert document["operations"][0][4] == [
        ["remove", [2, "1", "2", True]]]


def test_entire_product_removed():
    before = _ready_p(((0, "1"),))
    after = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 0),)))
    row = document["operations"][0]
    assert row[3] is not None and row[7] is None
    assert row[4] == [["remove", [0, None, "1", True]]]
    assert row[5][0] is not None and row[5][1] is None
    assert row[6] == [True, None]


def test_replace_action_for_modified_record():
    before = _snapshot(((0, "m", "1", PASSING_GATE),))
    after = _snapshot(((0, "m", "1", FAILING_GATE),))
    document = json.loads(plan_delivery_updates(
        before, after, (("m", 0, 0),)))
    assert document["operations"][0][4] == [
        ["replace", [0, None, "1", True], [0, None, "1", False]]]


def test_actions_sorted_combined():
    before = _snapshot(((0, "p", "1", PASSING_GATE),
                        (1, "p", "2", PASSING_GATE)))
    after = _snapshot(((0, "p", "9", PASSING_GATE),
                       (2, "p", "3", PASSING_GATE)))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 2),)))
    assert document["operations"][0][4] == [
        ["replace", [0, None, "1", True], [0, None, "9", True]],
        ["remove", [1, "1", "2", True]],
        ["add", [2, "9", "3", True]],
    ]


def test_empty_range_with_gaps():
    before = _snapshot(((5, "p", "1", PASSING_GATE),))
    after = _snapshot(((9, "p", "1", PASSING_GATE),))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 2),)))
    row = document["operations"][0]
    # Neither side has a version in the range; gaps are equal.
    assert row[4] == []
    assert row[3][1] == [[0, 2]]
    assert row[7][1] == [[0, 2]]


# ---------------------------------------------------------------------------
# replay semantics
# ---------------------------------------------------------------------------

def test_replaying_actions_reproduces_target():
    scenarios = [
        (_ready_p(((0, "1"), (1, "2"))),
         _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5"))),
         (("p", 0, 4),)),
        (_snapshot(((0, "p", "1", PASSING_GATE),
                    (2, "p", "2", PASSING_GATE))),
         _snapshot(((0, "p", "1", PASSING_GATE),)),
         (("p", 0, 2),)),
        (_snapshot(((0, "m", "1", PASSING_GATE),)),
         _snapshot(((0, "m", "1", FAILING_GATE),)),
         (("m", 0, 0),)),
        (_snapshot(((0, "p", "1", PASSING_GATE),
                    (1, "p", "2", PASSING_GATE))),
         _snapshot(((0, "p", "9", PASSING_GATE),
                    (2, "p", "3", PASSING_GATE))),
         (("p", 0, 2),)),
        (_ready_p(((0, "1"),)),
         build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS),
         (("p", 0, 0),)),
    ]
    for before, after, ranges in scenarios:
        document = json.loads(plan_delivery_updates(before, after, ranges))
        for row in document["operations"]:
            _replay(row)


# ---------------------------------------------------------------------------
# changed flag
# ---------------------------------------------------------------------------

def test_changed_false_for_identical_snapshots():
    before = _ready_p(((0, "1"), (1, "2")))
    document = json.loads(plan_delivery_updates(
        before, before, (("p", 0, 2), ("q", 5, 6))))
    assert document["changed"] is False


def test_changed_true_when_only_receipt_differs():
    before = _snapshot(((0, "p", "1", PASSING_GATE),))
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    after = build_delivery_snapshot(
        changes, _merged(changes, (("p", "succeeded", ""),)))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 0),)))
    row = document["operations"][0]
    assert row[4] == []
    assert document["changed"] is True


def test_changed_true_for_any_unequal_range():
    before = _snapshot(((0, "p", "1", PASSING_GATE),))
    after = _snapshot(((0, "p", "1", PASSING_GATE),
                       (1, "p", "2", PASSING_GATE)))
    document = json.loads(plan_delivery_updates(
        before, after, (("p", 0, 1), ("p", 5, 9))))
    assert document["operations"][0][3] != document["operations"][0][7]
    assert document["operations"][1][3] == document["operations"][1][7]
    assert document["changed"] is True
