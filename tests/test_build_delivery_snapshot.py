"""Tests for :func:`lidar_scan.build_delivery_snapshot` and
:func:`lidar_scan.query_delivery_snapshot`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, gate_report,
                        merge_delivery_manifests, merge_delivery_receipts,
                        query_delivery_snapshot)
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


def _audit(changes):
    return audit_delivery_changes(changes)


def _plan(changes):
    return build_delivery_plan(_audit(changes))


def _merged(changes, results):
    receipt = build_delivery_receipt(_plan(changes), tuple(results))
    return merge_delivery_receipts((receipt,))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.build_delivery_snapshot is build_delivery_snapshot
    assert tiles_module.query_delivery_snapshot is query_delivery_snapshot
    import lidar_scan
    assert "build_delivery_snapshot" in lidar_scan.__all__
    assert "query_delivery_snapshot" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# build: basics and document shape
# ---------------------------------------------------------------------------

def test_empty_inputs():
    assert build_delivery_snapshot(EMPTY_CHANGES,
                                   EMPTY_RECEIPTS) == (
        '{"products":[],"ready":true}'
    )


def test_type_errors():
    with pytest.raises(TypeError):
        build_delivery_snapshot(1, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        build_delivery_snapshot(None, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        build_delivery_snapshot(EMPTY_CHANGES, 1)
    with pytest.raises(TypeError):
        build_delivery_snapshot(EMPTY_CHANGES, None)


def test_non_canonical_inputs_raise_value_error():
    with pytest.raises(ValueError):
        build_delivery_snapshot("not json", EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        build_delivery_snapshot(EMPTY_CHANGES, "not json")
    with pytest.raises(ValueError):
        build_delivery_snapshot(
            '{"changes": [], "releasable": true}', EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        build_delivery_snapshot(
            EMPTY_CHANGES, '{"products": [], "ready": true}')


def test_top_level_key_order():
    text = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    assert list(json.loads(text)) == ["products", "ready"]
    assert text == '{"products":[],"ready":true}'


def test_products_sorted_by_name():
    changes = _changes(
        _manifest((0, "c", "1", PASSING_GATE),
                  (0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)))
    receipts = _merged(
        changes,
        (("a", "succeeded", ""), ("b", "succeeded", ""),
         ("c", "succeeded", "")))
    snapshot = build_delivery_snapshot(changes, receipts)
    names = [row[0] for row in json.loads(snapshot)["products"]]
    assert names == ["a", "b", "c"]


def test_item_shape_and_version_rows():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((1, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    item = json.loads(snapshot)["products"][0]
    assert item[0] == "p"
    assert item[1] == [[0, None, "1", True], [1, "1", "2", True]]
    assert item[2] == []
    assert item[3] == ["publish", "2", "2", [], [["succeeded", ""]],
                       "succeeded", []]


def test_missing_receipt_is_null():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = build_delivery_snapshot(changes, EMPTY_RECEIPTS)
    item = json.loads(snapshot)["products"][0]
    assert item[3] is None


def test_receipt_history_preserved():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    plan = _plan(changes)
    failed = build_delivery_receipt(plan, (("p", "failed", "boom"),))
    succeeded = build_delivery_receipt(plan, (("p", "succeeded", ""),))
    receipts = merge_delivery_receipts((failed, succeeded))
    snapshot = build_delivery_snapshot(changes, receipts)
    item = json.loads(snapshot)["products"][0]
    assert item[3][4] == [["failed", "boom"], ["succeeded", ""]]
    assert item[3][5] == "succeeded"


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------

def test_single_gap():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((2, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    item = json.loads(build_delivery_snapshot(changes, receipts))["products"][0]
    assert item[2] == [[1, 1]]


def test_gaps_compressed_into_maximal_closed_intervals():
    changes = _changes(_manifest((0, "z", "1", PASSING_GATE)),
                       _manifest((4, "z", "2", PASSING_GATE),
                                 (7, "z", "3", PASSING_GATE)))
    receipts = _merged(changes, (("z", "succeeded", ""),))
    item = json.loads(build_delivery_snapshot(changes, receipts))["products"][0]
    assert item[2] == [[1, 3], [5, 6]]


def test_gaps_start_from_zero_when_batch_zero_absent():
    changes = _changes(_manifest((3, "p", "1", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    item = json.loads(build_delivery_snapshot(changes, receipts))["products"][0]
    assert item[2] == [[0, 2]]


# ---------------------------------------------------------------------------
# ready
# ---------------------------------------------------------------------------

def test_ready_true_for_contiguous_succeeded_publishes():
    changes = _changes(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)),
        _manifest((1, "a", "2", PASSING_GATE)))
    receipts = _merged(changes, (("a", "succeeded", ""),
                                 ("b", "succeeded", "")))
    snapshot = build_delivery_snapshot(changes, receipts)
    assert snapshot.endswith('"ready":true}')


def test_ready_false_when_gaps_present():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((2, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    assert snapshot.endswith('"ready":false}')


def test_ready_false_without_receipt():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = build_delivery_snapshot(changes, EMPTY_RECEIPTS)
    assert snapshot.endswith('"ready":false}')


def test_ready_false_when_final_change_fails():
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    receipts = _merged(changes, (("p", "blocked", "held"),))
    snapshot = build_delivery_snapshot(changes, receipts)
    assert snapshot.endswith('"ready":false}')


def test_ready_false_for_block_and_rollback_actions():
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    blocked = _merged(changes, (("p", "blocked", "held"),))
    assert build_delivery_snapshot(changes, blocked).endswith(
        '"ready":false}')

    changes_rb = _changes(_manifest((0, "r", "1", PASSING_GATE)),
                          _manifest((1, "r", "2", FAILING_GATE)))
    plan_rb = _plan(changes_rb)
    receipt_rb = build_delivery_receipt(
        plan_rb, (("r", "succeeded", ""),))
    rolled_back = merge_delivery_receipts((receipt_rb,))
    assert build_delivery_snapshot(changes_rb, rolled_back).endswith(
        '"ready":false}')


def test_ready_false_when_final_status_failed():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    plan = _plan(changes)
    failed = build_delivery_receipt(plan, (("p", "failed", "boom"),))
    receipts = merge_delivery_receipts((failed,))
    assert build_delivery_snapshot(changes, receipts).endswith(
        '"ready":false}')


# ---------------------------------------------------------------------------
# cross-document consistency
# ---------------------------------------------------------------------------

def test_unknown_receipt_product_raises():
    changes = _changes(_manifest((0, "a", "1", PASSING_GATE)))
    other = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts = _merged(other, (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        build_delivery_snapshot(changes, receipts)


def test_receipt_current_must_equal_final_version():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                           _manifest((1, "p", "2", PASSING_GATE)))
    receipts = _merged(changes_one, (("p", "succeeded", ""),))
    changes_two = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                           _manifest((1, "p", "9", PASSING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_snapshot(changes_two, receipts)


def test_failure_rows_preserved_with_six_decimal_metrics():
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    receipts = _merged(changes, (("p", "blocked", "held"),))
    snapshot = build_delivery_snapshot(changes, receipts)
    failures = json.loads(snapshot)["products"][0][3][6]
    assert failures == [
        [0, "1", 0, 0, 0, 9, 9, [-1.0, 2.0, 1.581139, 1.0, 1]],
        [0, "1", 0, 100, 100, 109, 109, None],
    ]
    assert "-1.000000" in snapshot
    assert "2.000000" in snapshot


def test_build_is_deterministic_and_does_not_modify_inputs():
    changes = _changes(_manifest((0, "a", "1", PASSING_GATE)),
                       _manifest((0, "b", "1", PASSING_GATE)))
    receipts = _merged(changes, (("a", "succeeded", ""),
                                 ("b", "succeeded", "")))
    changes_copy = changes
    receipts_copy = receipts
    first = build_delivery_snapshot(changes, receipts)
    second = build_delivery_snapshot(changes_copy, receipts_copy)
    assert first == second
    assert changes == changes_copy and receipts == receipts_copy


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def test_query_type_errors():
    with pytest.raises(TypeError):
        query_delivery_snapshot(1, "p")
    with pytest.raises(TypeError):
        query_delivery_snapshot(None, "p")
    snapshot = build_delivery_snapshot(
        _changes(_manifest((0, "p", "1", PASSING_GATE))), EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        query_delivery_snapshot(snapshot, 1)
    with pytest.raises(TypeError):
        query_delivery_snapshot(snapshot, None)


def test_query_invalid_snapshot_raises_value_error():
    with pytest.raises(ValueError):
        query_delivery_snapshot("not json", "p")
    with pytest.raises(ValueError):
        query_delivery_snapshot(
            '{"products": [] , "ready": true}', "p")


def test_query_miss_returns_none():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = build_delivery_snapshot(changes, EMPTY_RECEIPTS)
    assert query_delivery_snapshot(snapshot, "zzz") is None


def test_query_empty_snapshot_returns_none():
    snapshot = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    assert query_delivery_snapshot(snapshot, "p") is None


def test_query_hit_returns_nested_tuples():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((2, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    item = query_delivery_snapshot(snapshot, "p")
    assert item == (
        "p",
        ((0, None, "1", True), (2, "1", "2", True)),
        ((1, 1),),
        ("publish", "2", "2", (), (("succeeded", ""),), "succeeded", ()),
    )
    assert isinstance(item, tuple)
    assert all(isinstance(version, tuple) for version in item[1])
    assert all(isinstance(gap, tuple) for gap in item[2])
    assert isinstance(item[3], tuple)


def test_query_null_receipt_becomes_none():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = build_delivery_snapshot(changes, EMPTY_RECEIPTS)
    item = query_delivery_snapshot(snapshot, "p")
    assert item[3] is None


def test_query_failure_metrics_become_floats():
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    receipts = _merged(changes, (("p", "blocked", "held"),))
    snapshot = build_delivery_snapshot(changes, receipts)
    item = query_delivery_snapshot(snapshot, "p")
    summary = item[3][6][0][7]
    assert summary == (-1.0, 2.0, 1.581139, 1.0, 1)
    assert all(isinstance(value, float) for value in summary[:4])
    assert isinstance(summary[4], int)


def test_query_hit_tampered_ready_flag_rejected():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    document = json.loads(snapshot)
    tampered = json.dumps({"products": document["products"], "ready": False})
    with pytest.raises(ValueError):
        query_delivery_snapshot(tampered, "p")


def test_query_rejects_bad_gap_compression():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((2, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    document = json.loads(snapshot)
    document["products"][0][2] = [[1, 2]]
    with pytest.raises(ValueError):
        query_delivery_snapshot(json.dumps(document), "p")


def test_query_rejects_current_version_mismatch():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = build_delivery_snapshot(changes, receipts)
    document = json.loads(snapshot)
    document["products"][0][3][1] = "9"
    with pytest.raises(ValueError):
        query_delivery_snapshot(json.dumps(document), "p")
