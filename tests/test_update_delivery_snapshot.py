"""Tests for :func:`lidar_scan.update_delivery_snapshot`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, merge_delivery_manifests,
                        merge_delivery_receipts, update_delivery_snapshot)
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


def _snapshot(*manifests, results=()):
    changes = _changes(*manifests)
    receipts = (_merged(changes, results) if results
                else EMPTY_RECEIPTS)
    return build_delivery_snapshot(changes, receipts)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.update_delivery_snapshot is update_delivery_snapshot
    import lidar_scan
    assert "update_delivery_snapshot" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
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
        update_delivery_snapshot(EMPTY_SNAPSHOT, "not json", EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES, "not json")
    with pytest.raises(ValueError):
        update_delivery_snapshot('{"products": [], "ready": true}',
                                 EMPTY_CHANGES, EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(EMPTY_SNAPSHOT,
                                 '{"changes": [], "releasable": true}',
                                 EMPTY_RECEIPTS)
    with pytest.raises(ValueError):
        update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES,
                                 '{"products": [], "ready": true}')


# ---------------------------------------------------------------------------
# identity and union behaviour
# ---------------------------------------------------------------------------

def test_empty_inputs_round_trip():
    assert update_delivery_snapshot(EMPTY_SNAPSHOT, EMPTY_CHANGES,
                                    EMPTY_RECEIPTS) == EMPTY_SNAPSHOT


def test_no_changes_or_receipts_is_identity():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    assert update_delivery_snapshot(snapshot, EMPTY_CHANGES,
                                    EMPTY_RECEIPTS) == snapshot


def test_snapshot_with_receipt_unchanged_without_new_inputs():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    assert update_delivery_snapshot(snapshot, EMPTY_CHANGES,
                                    EMPTY_RECEIPTS) == snapshot


def test_new_changes_extend_existing_product():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    item = json.loads(updated)["products"][0]
    assert item[1] == [[0, None, "1", True], [1, "1", "2", True]]
    # The old receipt's current no longer matches the final version.
    assert item[3] is None
    assert updated.endswith('"ready":false}')


def test_new_changes_add_new_product():
    snapshot = _snapshot(_manifest((0, "a", "1", PASSING_GATE)),
                       results=(("a", "succeeded", ""),))
    changes = _changes(_manifest((0, "b", "1", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    names = [row[0] for row in json.loads(updated)["products"]]
    assert names == ["a", "b"]


def test_overlapping_identical_entries_are_deduplicated():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((1, "p", "2", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    item = json.loads(updated)["products"][0]
    assert item[1] == [[0, None, "1", True], [1, "1", "2", True]]


def test_conflicting_version_raises():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    changes = _changes(_manifest((0, "p", "9", PASSING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)


def test_conflicting_passed_raises():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)


def test_version_reused_across_batches_raises():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    changes = _changes(_manifest((1, "p", "1", PASSING_GATE)))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)


def test_previous_recomputed_in_batch_order():
    snapshot = _snapshot(_manifest((1, "p", "2", PASSING_GATE)))
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    item = json.loads(updated)["products"][0]
    assert item[1] == [[0, None, "1", True], [1, "1", "2", True]]


# ---------------------------------------------------------------------------
# gaps and ready
# ---------------------------------------------------------------------------

def test_gaps_recomputed_after_update():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((3, "p", "2", PASSING_GATE)))
    item = json.loads(snapshot)["products"][0]
    assert item[2] == [[1, 2]]
    changes = _changes(_manifest((2, "p", "3", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    item = json.loads(updated)["products"][0]
    assert item[2] == [[1, 1]]


def test_ready_recomputed_after_update():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    assert snapshot.endswith('"ready":true}')
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    assert updated.endswith('"ready":false}')


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------

def test_new_receipt_replaces_old():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                       results=(("p", "succeeded", ""),))
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    receipts = _merged(_changes(_manifest((0, "p", "1", PASSING_GATE)),
                              _manifest((1, "p", "2", PASSING_GATE))),
                       (("p", "succeeded", ""),))
    updated = update_delivery_snapshot(snapshot, changes, receipts)
    item = json.loads(updated)["products"][0]
    assert item[3] == ["publish", "2", "2", [], [["succeeded", ""]],
                       "succeeded", []]
    assert updated.endswith('"ready":true}')


def test_old_receipt_kept_when_current_still_final():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    updated = update_delivery_snapshot(snapshot, EMPTY_CHANGES,
                                       EMPTY_RECEIPTS)
    assert json.loads(updated)["products"][0][3] == [
        "publish", "1", "1", [], [["succeeded", ""]], "succeeded", []]


def test_old_receipt_dropped_when_current_stale():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    updated = update_delivery_snapshot(snapshot, changes, EMPTY_RECEIPTS)
    assert json.loads(updated)["products"][0][3] is None


def test_unknown_receipt_product_raises():
    snapshot = _snapshot(_manifest((0, "a", "1", PASSING_GATE)))
    other = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    receipts = _merged(other, (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, EMPTY_CHANGES, receipts)


def test_receipt_current_mismatch_raises():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)))
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    receipts = _merged(_changes(_manifest((0, "p", "1", PASSING_GATE))),
                     (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        update_delivery_snapshot(snapshot, changes, receipts)


# ---------------------------------------------------------------------------
# determinism and input immutability
# ---------------------------------------------------------------------------

def test_deterministic_and_inputs_unmodified():
    snapshot = _snapshot(_manifest((0, "p", "1", PASSING_GATE)),
                         results=(("p", "succeeded", ""),))
    changes = _changes(_manifest((1, "p", "2", PASSING_GATE)))
    receipts = _merged(_changes(_manifest((0, "p", "1", PASSING_GATE)),
                              _manifest((1, "p", "2", PASSING_GATE))),
                       (("p", "succeeded", ""),))
    first = update_delivery_snapshot(snapshot, changes, receipts)
    second = update_delivery_snapshot(snapshot, changes, receipts)
    assert first == second
    assert update_delivery_snapshot(snapshot, EMPTY_CHANGES,
                                    EMPTY_RECEIPTS) == snapshot
