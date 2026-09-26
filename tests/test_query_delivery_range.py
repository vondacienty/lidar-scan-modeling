"""Tests for :func:`lidar_scan.query_delivery_range`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, merge_delivery_manifests,
                        merge_delivery_receipts, query_delivery_range)
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


def _ready_snapshot():
    """Product ``p`` with batches 0, 1, 2 and a succeeded publish receipt."""
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((1, "p", "2", PASSING_GATE)),
                       _manifest((2, "p", "3", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    return build_delivery_snapshot(changes, receipts)


def _gapped_snapshot():
    """Product ``p`` with batches 0 and 2 and a succeeded publish receipt."""
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)),
                       _manifest((2, "p", "2", PASSING_GATE)))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    return build_delivery_snapshot(changes, receipts)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.query_delivery_range is query_delivery_range
    import lidar_scan
    assert "query_delivery_range" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_errors():
    snapshot = _ready_snapshot()
    with pytest.raises(TypeError):
        query_delivery_range(1, "p", 0, 1)
    with pytest.raises(TypeError):
        query_delivery_range(None, "p", 0, 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, 1, 0, 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, None, 0, 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", True, 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", 0, False)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", "0", 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", 0, 1.5)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", None, 1)
    with pytest.raises(TypeError):
        query_delivery_range(snapshot, "p", 0, None)


def test_bound_value_errors():
    snapshot = _ready_snapshot()
    with pytest.raises(ValueError):
        query_delivery_range(snapshot, "p", -1, 1)
    with pytest.raises(ValueError):
        query_delivery_range(snapshot, "p", 0, -1)
    with pytest.raises(ValueError):
        query_delivery_range(snapshot, "p", 2, 1)


def test_invalid_snapshot_raises_value_error():
    with pytest.raises(ValueError):
        query_delivery_range("not json", "p", 0, 1)
    with pytest.raises(ValueError):
        query_delivery_range(
            '{"products": [] , "ready": true}', "p", 0, 1)


def test_tampered_derived_values_rejected():
    snapshot = _gapped_snapshot()
    document = json.loads(snapshot)

    tampered_ready = json.dumps(
        {"products": document["products"], "ready": True})
    with pytest.raises(ValueError):
        query_delivery_range(tampered_ready, "p", 0, 2)

    bad_gaps = json.loads(snapshot)
    bad_gaps["products"][0][2] = []
    with pytest.raises(ValueError):
        query_delivery_range(json.dumps(bad_gaps), "p", 0, 2)

    bad_previous = json.loads(snapshot)
    bad_previous["products"][0][1][1][1] = None
    with pytest.raises(ValueError):
        query_delivery_range(json.dumps(bad_previous), "p", 0, 2)


# ---------------------------------------------------------------------------
# misses
# ---------------------------------------------------------------------------

def test_unknown_product_returns_none():
    assert query_delivery_range(_ready_snapshot(), "zzz", 0, 2) is None


def test_empty_snapshot_returns_none():
    snapshot = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    assert query_delivery_range(snapshot, "p", 0, 0) is None


# ---------------------------------------------------------------------------
# versions and gaps
# ---------------------------------------------------------------------------

def test_full_range_returns_all_versions_and_no_gaps():
    versions, gaps, receipt, ready = query_delivery_range(
        _ready_snapshot(), "p", 0, 2)
    assert versions == (
        (0, None, "1", True),
        (1, "1", "2", True),
        (2, "2", "3", True),
    )
    assert gaps == ()
    assert ready is True


def test_subrange_filters_versions_and_preserves_order():
    versions, gaps, _receipt, ready = query_delivery_range(
        _ready_snapshot(), "p", 1, 1)
    assert versions == ((1, "1", "2", True),)
    assert gaps == ()
    assert ready is True


def test_range_beyond_last_batch_adds_trailing_gap():
    versions, gaps, _receipt, ready = query_delivery_range(
        _ready_snapshot(), "p", 1, 5)
    assert versions == ((1, "1", "2", True), (2, "2", "3", True))
    assert gaps == ((3, 5),)
    assert ready is True


def test_empty_interval_yields_whole_range_gap():
    versions, gaps, receipt, ready = query_delivery_range(
        _ready_snapshot(), "p", 5, 7)
    assert versions == ()
    assert gaps == ((5, 7),)
    assert ready is True


def test_gapped_chain_reports_inner_gap():
    versions, gaps, _receipt, ready = query_delivery_range(
        _gapped_snapshot(), "p", 0, 2)
    assert versions == ((0, None, "1", True), (2, "1", "2", True))
    assert gaps == ((1, 1),)
    assert ready is False


def test_gapped_chain_head_and_tail_gaps():
    _versions, gaps, _receipt, _ready = query_delivery_range(
        _gapped_snapshot(), "p", 0, 4)
    assert gaps == ((1, 1), (3, 4))

    versions, gaps, _receipt, _ready = query_delivery_range(
        _gapped_snapshot(), "p", 2, 3)
    assert versions == ((2, "1", "2", True),)
    assert gaps == ((3, 3),)


def test_ready_ignores_range_truncation():
    # The gapped chain is not ready even over a gap-free subrange.
    _versions, gaps, _receipt, ready = query_delivery_range(
        _gapped_snapshot(), "p", 0, 0)
    assert gaps == ()
    assert ready is False


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

def test_receipt_fields_and_history_tuples():
    _versions, _gaps, receipt, _ready = query_delivery_range(
        _ready_snapshot(), "p", 0, 2)
    assert receipt == (
        "publish", "3", "3", (("succeeded", ""),), "succeeded")
    assert isinstance(receipt[3], tuple)
    assert all(isinstance(entry, tuple) for entry in receipt[3])


def test_null_receipt_becomes_none():
    changes = _changes(_manifest((0, "p", "1", PASSING_GATE)))
    snapshot = build_delivery_snapshot(changes, EMPTY_RECEIPTS)
    versions, gaps, receipt, ready = query_delivery_range(
        snapshot, "p", 0, 0)
    assert versions == ((0, None, "1", True),)
    assert gaps == ()
    assert receipt is None
    assert ready is False


def test_blocked_receipt_history_preserves_order():
    changes = _changes(_manifest((0, "p", "1", FAILING_GATE)))
    receipts = _merged(changes, (("p", "blocked", "held"),))
    snapshot = build_delivery_snapshot(changes, receipts)
    _versions, _gaps, receipt, ready = query_delivery_range(
        snapshot, "p", 0, 0)
    assert receipt == ("block", "1", None, (("blocked", "held"),),
                       "blocked")
    assert ready is False


def test_result_is_nested_tuples():
    result = query_delivery_range(_ready_snapshot(), "p", 0, 2)
    assert isinstance(result, tuple) and len(result) == 4
    versions, gaps, receipt, ready = result
    assert all(isinstance(version, tuple) for version in versions)
    assert all(isinstance(gap, tuple) for gap in gaps)
    assert isinstance(receipt, tuple)
    assert isinstance(ready, bool)
