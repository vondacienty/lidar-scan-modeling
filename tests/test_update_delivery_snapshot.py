"""Tests for :func:`lidar_scan.update_delivery_snapshot`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot,
                        merge_delivery_manifests, merge_delivery_receipts,
                        update_delivery_snapshot)
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
EMPTY_SNAPSHOT = '{"products":[],"ready":true}'


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


def _plan(changes):
    return build_delivery_plan(audit_delivery_changes(changes))


def _merged(changes, results):
    receipt = build_delivery_receipt(_plan(changes), tuple(results))
    return merge_delivery_receipts((receipt,))


def _snapshot(changes, receipts):
    return build_delivery_snapshot(changes, receipts)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.update_delivery_snapshot is update_delivery_snapshot
    assert lidar_scan.update_delivery_snapshot is update_delivery_snapshot
    assert "update_delivery_snapshot" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# type and canonical-encoding validation
# ---------------------------------------------------------------------------

def test_type_errors():
    with pytest.raises(TypeError):
        update_delivery_snapshot(1, EMPTY_CHANGES, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        update_delivery_snapshot(None, EMPTY_CHANGES, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, 1, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, None, EMPTY_RECEIPTS)
    with pytest.raises(TypeError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES, 1)
    with pytest.raises(TypeError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES, None)


def test_non_canonical_inputs_raise_value_error():
    with pytest.raises(ValueError):
        update_delivery_snapshot("not json", EMPTY_CHANGES, EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(
            '{"products": [] , "ready": true}', EMPTY_CHANGES, EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, "not json", EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(
            EMPTY_SNAPSHOT, '{"changes": [], "releasable": true}',
            EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES, "not json")
    with pytest.raises(ValueError):
        update_delivery_snapshot(
            EMPTY_SNAPSHOT, EMPTY_CHANGES, '{"products": [], "ready": true}')


# ---------------------------------------------------------------------------
# empty and from-scratch equivalence with build
# ---------------------------------------------------------------------------

def test_empty_documents_stay_empty():
    assert update_delivery_snapshot(
        EMPTY_SNAPSHOT, EMPTY_CHANGES, EMPTY_RECEIPTS) == EMPTY_SNAPSHOT


def test_building_from_empty_matches_build_delivery_snapshot():
    changes = _changes(_manifest((0, "a", "1", PASSING_GATE),
                                 (0, "b", "1", PASSING_GATE)))
    receipts = _merged(changes, (("a", "succeeded", ""),
                                 ("b", "succeeded", "")))
    assert update_delivery_snapshot(
        EMPTY_SNAPSHOT, changes, receipts
    ) == build_delivery_snapshot(changes, receipts)


# ---------------------------------------------------------------------------
# union and previous recomputation
# ---------------------------------------------------------------------------

def test_changes_extend_versions_with_recomputed_previous():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((2, "p", "3", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)
    item = json.loads(result)["products"][0]
    assert item[1] == [[0, None, "1", True], [2, "1", "3", True]]
    assert item[2] == [[1, 1]]


def test_union_orders_batches_regardless_of_change_order():
    changes_one = _changes(_manifest((3, "p", "4", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((0, "p", "1", PASSING_GATE),
                                     (1, "p", "2", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)
    versions = json.loads(result)["products"][0][1]
    assert [row[0] for row in versions] == [0, 1, 3]
    assert [row[1] for row in versions] == [None, "1", "2"]
    assert [row[2] for row in versions] == ["1", "2", "4"]


def test_new_products_are_merged_in_sorted_order():
    changes_p = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_p, EMPTY_RECEIPTS)
    changes_a = _changes(_manifest((0, "a", "1", PASSING_GATE),
                                   (0, "z", "1", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_a, EMPTY_RECEIPTS)
    names = [row[0] for row in json.loads(result)["products"]]
    assert names == ["a", "p", "z"]


def test_identical_shared_key_is_idempotent():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE),
                                 (1, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes, receipts)
    only_first = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    assert update_delivery_snapshot(
        snapshot, only_first, EMPTY_RECEIPTS) == snapshot
    assert update_delivery_snapshot(snapshot, changes, receipts) == snapshot


def test_conflicting_version_for_same_key_raises():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((0, "p", "2", PASSING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)


def test_conflicting_passed_for_same_key_raises():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)


def test_version_reused_across_batches_raises():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((1, "p", "1", PASSING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)


def test_same_version_may_be_used_by_different_products():
    changes_p = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_p, EMPTY_RECEIPTS)
    changes_q = _changes(_manifest((0, "q", "1", PASSING_GATE),
                                   (1, "q", "2", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_q, EMPTY_RECEIPTS)
    names = [row[0] for row in json.loads(result)["products"]]
    assert names == ["p", "q"]


# ---------------------------------------------------------------------------
# receipt replacement / retention / nulling
# ---------------------------------------------------------------------------

def test_new_receipt_replaces_old_receipt():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts_one = _merged(changes_one, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes_one, receipts_one)

    plan_one = _plan(changes_one)
    failed = build_delivery_receipt(plan_one, (("p", "failed", "boom"),))
    succeeded = build_delivery_receipt(plan_one, (("p", "succeeded", ""),))
    replaced = merge_delivery_receipts((failed, succeeded))

    result = update_delivery_snapshot(snapshot, EMPTY_CHANGES, replaced)
    receipt = json.loads(result)["products"][0][3]
    assert receipt[4] == [["failed", "boom"], ["succeeded", ""]]
    assert receipt[5] == "succeeded"


def test_old_receipt_kept_when_current_still_final_without_new_receipt():
    changes_p = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts_p = _merged(changes_p, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes_p, receipts_p)
    changes_q = _changes(_manifest((0, "q", "1", PASSING_GATE)))
    receipts_q = _merged(changes_q, (("q", "succeeded", ""),))
    result = update_delivery_snapshot(snapshot, changes_q, receipts_q)
    products = {row[0]: row for row in json.loads(result)["products"]}
    assert products["p"][3][1] == "1"
    assert products["q"][3][1] == "1"


def test_old_receipt_nulled_when_final_version_moves_on():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts_one = _merged(changes_one, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes_one, receipts_one)
    changes_two = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)
    item = json.loads(result)["products"][0]
    assert item[1][-1][2] == "2"
    assert item[3] is None


def test_null_old_receipt_stays_null():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)
    assert json.loads(result)["products"][0][3] is None


def test_new_receipt_current_must_equal_merged_final_version():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    stale_receipts = _merged(changes_one, (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes_two, stale_receipts)


def test_receipt_for_unknown_product_raises():
    changes_p = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_p, EMPTY_RECEIPTS)
    changes_q = _changes(_manifest((0, "q", "1", PASSING_GATE)))
    receipts_z = _merged(
        _changes(_manifest((0, "z", "1", PASSING_GATE))),
        (("z", "succeeded", ""),))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes_q, receipts_z)


# ---------------------------------------------------------------------------
# ready and canonical output
# ---------------------------------------------------------------------------

def test_ready_true_after_contiguous_succeeded_publish_update():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    full_changes = _changes(_manifest((0, "p", "1", PASSING_GATE),
                                      (1, "p", "2", PASSING_GATE)))
    receipts = _merged(full_changes, (("p", "succeeded", ""),))
    result = update_delivery_snapshot(snapshot, changes_two, receipts)
    assert result.endswith('"ready":true}')
    assert result == build_delivery_snapshot(full_changes, receipts)


def test_ready_false_when_gaps_introduced():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts_one = _merged(changes_one, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes_one, receipts_one)
    changes_two = _changes(_manifest((2, "p", "2", PASSING_GATE)))
    result = update_delivery_snapshot(snapshot, changes_two, EMPTY_RECEIPTS)
    assert result.endswith('"ready":false}')


def test_ready_false_when_final_change_fails():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts_one = _merged(changes_one, (("p", "succeeded", ""),))
    snapshot = _snapshot(changes_one, receipts_one)
    changes_two = _changes(_manifest((1, "p", "2", FAILING_GATE)))
    full_changes = _changes(_manifest((0, "p", "1", PASSING_GATE),
                                      (1, "p", "2", FAILING_GATE)))
    receipts = _merged(full_changes, (("p", "failed", "boom"),))
    result = update_delivery_snapshot(snapshot, changes_two, receipts)
    assert result.endswith('"ready":false}')


def test_output_is_canonical_compact_json():
    changes_one = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = _snapshot(changes_one, EMPTY_RECEIPTS)
    changes_two = _changes(_manifest((2, "p", "2", FAILING_GATE)))
    full_changes = _changes(_manifest((0, "p", "1", PASSING_GATE),
                                      (2, "p", "2", FAILING_GATE)))
    receipts = _merged(full_changes, (("p", "failed", "boom"),))
    result = update_delivery_snapshot(snapshot, changes_two, receipts)
    assert list(json.loads(result)) == ["products", "ready"]
    assert " " not in result
    assert not result.endswith("\n")
    assert result == build_delivery_snapshot(full_changes, receipts)
    assert "-1.000000" in result and "2.000000" in result


def test_result_is_deterministic_and_inputs_unchanged():
    changes_one = _changes(_manifest((0, "a", "1", PASSING_GATE)))
    snapshot = _snapshot(
        changes_one, _merged(changes_one, (("a", "succeeded", ""),)))
    changes_two = _changes(_manifest((1, "a", "2", PASSING_GATE),
                                     (0, "b", "1", PASSING_GATE)))
    full_changes = _changes(_manifest((0, "a", "1", PASSING_GATE),
                                      (1, "a", "2", PASSING_GATE),
                                      (0, "b", "1", PASSING_GATE)))
    receipts = _merged(full_changes,
                       (("a", "succeeded", ""), ("b", "succeeded", "")))
    snapshot_before = snapshot
    changes_before = changes_two
    receipts_before = receipts
    first = update_delivery_snapshot(snapshot, changes_two, receipts)
    second = update_delivery_snapshot(snapshot, changes_two, receipts)
    assert first == second
    assert snapshot == snapshot_before
    assert changes_two == changes_before
    assert receipts == receipts_before
