"""Tests for :func:`lidar_scan.merge_delivery_manifests`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import build_delivery_manifest, merge_delivery_manifests
from lidar_scan import tiles as tiles_module


EMPTY_GATE = '{"windows":[],"passed":true}'

FAILING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],false],'
    '[0,100,100,109,109,null,false]],"passed":false}'
)

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.merge_delivery_manifests is merge_delivery_manifests
    import lidar_scan
    assert "merge_delivery_manifests" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_manifests_change_log():
    assert merge_delivery_manifests(()) == '{"changes":[],"releasable":true}'


def test_empty_manifest_contributes_no_changes():
    assert merge_delivery_manifests((_manifest(),)) == (
        '{"changes":[],"releasable":true}')


def test_change_shape_and_key_order():
    text = merge_delivery_manifests((_manifest((0, "prod", "1.0.0",
                                                EMPTY_GATE)),))
    assert list(json.loads(text)) == ["changes", "releasable"]
    assert text == (
        '{"changes":[[0,"prod",null,"1.0.0",true,[]]],"releasable":true}'
    )


def test_changes_sorted_by_batch_then_product():
    manifests = (
        _manifest((2, "b", "1", EMPTY_GATE), (0, "z", "1", EMPTY_GATE)),
        _manifest((1, "a", "2", EMPTY_GATE), (0, "a", "1", EMPTY_GATE)),
    )
    doc = json.loads(merge_delivery_manifests(manifests))
    assert [(c[0], c[1]) for c in doc["changes"]] == [
        (0, "a"), (0, "z"), (1, "a"), (2, "b")
    ]


def test_previous_version_chains_per_product():
    manifests = (
        _manifest((0, "a", "1.0", EMPTY_GATE), (2, "a", "1.2", EMPTY_GATE)),
        _manifest((1, "a", "1.1", EMPTY_GATE), (0, "b", "3.0", EMPTY_GATE)),
    )
    doc = json.loads(merge_delivery_manifests(manifests))
    by_key = {(c[0], c[1]): c for c in doc["changes"]}
    assert by_key[(0, "a")][2] is None
    assert by_key[(1, "a")][2] == "1.0"
    assert by_key[(2, "a")][2] == "1.1"
    assert by_key[(0, "b")][2] is None


def test_releasable_and_of_last_batch_per_product():
    manifests = (
        _manifest((0, "a", "1", FAILING_GATE), (1, "a", "2", PASSING_GATE)),
        _manifest((0, "b", "1", PASSING_GATE), (1, "b", "2", FAILING_GATE)),
        _manifest((0, "c", "1", PASSING_GATE)),
    )
    # a's last batch passes, b's last batch fails, c's only batch passes.
    assert json.loads(merge_delivery_manifests(manifests))["releasable"] is False
    assert json.loads(merge_delivery_manifests(manifests[:1]))["releasable"] is True


def test_failing_entry_payload():
    text = merge_delivery_manifests((_manifest((7, "prod", "v2",
                                                FAILING_GATE)),))
    assert text == (
        '{"changes":[[7,"prod",null,"v2",false,'
        '[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,null]]]],"releasable":false}'
    )


def test_failures_keep_manifest_window_order():
    gate = (
        '{"windows":[[2,0,0,0,0,null,false],[0,0,0,0,0,null,false]],'
        '"passed":false}'
    )
    doc = json.loads(
        merge_delivery_manifests((_manifest((0, "p", "1", gate)),)))
    assert [window[0] for window in doc["changes"][0][5]] == [2, 0]


# ---------------------------------------------------------------------------
# merging semantics
# ---------------------------------------------------------------------------

def test_identical_entries_are_deduplicated():
    manifest = _manifest((0, "a", "1", EMPTY_GATE), (1, "b", "2", EMPTY_GATE))
    merged = merge_delivery_manifests((manifest, manifest))
    assert merged == merge_delivery_manifests((manifest,))


def test_overlapping_manifests_deduplicate_shared_entries():
    first = _manifest((0, "a", "1", EMPTY_GATE), (1, "a", "2", EMPTY_GATE))
    second = _manifest((1, "a", "2", EMPTY_GATE), (2, "a", "3", EMPTY_GATE))
    doc = json.loads(merge_delivery_manifests((first, second)))
    assert [(c[0], c[1]) for c in doc["changes"]] == [
        (0, "a"), (1, "a"), (2, "a")
    ]


def test_conflicting_entries_raise():
    first = _manifest((0, "a", "1", EMPTY_GATE))
    second = _manifest((0, "a", "9", EMPTY_GATE))
    with pytest.raises(ValueError, match="conflicting"):
        merge_delivery_manifests((first, second))


def test_conflicting_passed_flag_raises():
    first = _manifest((0, "a", "1", EMPTY_GATE))
    second = _manifest((0, "a", "1", FAILING_GATE))
    with pytest.raises(ValueError, match="conflicting"):
        merge_delivery_manifests((first, second))


def test_version_reuse_within_product_raises():
    first = _manifest((0, "a", "1", EMPTY_GATE))
    second = _manifest((1, "a", "1", EMPTY_GATE))
    with pytest.raises(ValueError, match="reused"):
        merge_delivery_manifests((first, second))


def test_same_version_for_different_products_is_allowed():
    manifests = (_manifest((0, "a", "1", EMPTY_GATE)),
                 _manifest((0, "b", "1", EMPTY_GATE)))
    doc = json.loads(merge_delivery_manifests(manifests))
    assert len(doc["changes"]) == 2


def test_reordering_manifests_is_byte_identical():
    manifests = (
        _manifest((0, "a", "1", FAILING_GATE), (2, "b", "1", EMPTY_GATE)),
        _manifest((1, "a", "2", PASSING_GATE)),
        _manifest((0, "b", "0", EMPTY_GATE), (3, "a", "3", EMPTY_GATE)),
    )
    assert merge_delivery_manifests(manifests) == merge_delivery_manifests(
        tuple(reversed(manifests)))


def test_input_tuple_is_not_modified():
    manifests = (_manifest((0, "a", "1", EMPTY_GATE)),)
    snapshot = manifests[0]
    merge_delivery_manifests(manifests)
    assert manifests[0] == snapshot


# ---------------------------------------------------------------------------
# type errors
# ---------------------------------------------------------------------------

def test_non_tuple_manifests_raise_type_error():
    manifest = _manifest((0, "a", "1", EMPTY_GATE))
    for bad in ([manifest], "manifest", None, 0, True):
        with pytest.raises(TypeError):
            merge_delivery_manifests(bad)


def test_non_str_member_raises_type_error():
    for bad in (1, None, b"bytes", ("nested",), object()):
        with pytest.raises(TypeError):
            merge_delivery_manifests((bad,))


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------

def test_invalid_json_raises_value_error():
    for bad in ("", "not json", "{", "[1,]", "null", "[]", '"text"', "7"):
        with pytest.raises(ValueError):
            merge_delivery_manifests((bad,))


def test_non_finite_literals_raise_value_error():
    with pytest.raises(ValueError):
        merge_delivery_manifests(
            ('{"entries":[[0,"a","1",true,[[0,0,0,0,0,'
             '[NaN,0.000000,0.000000,0.000000,1]]]]],"releasable":true}',))
    with pytest.raises(ValueError):
        merge_delivery_manifests(
            ('{"entries":[[0,"a","1",true,[[0,0,0,0,0,'
             '[Infinity,0.000000,0.000000,0.000000,1]]]]],'
             '"releasable":true}',))


def test_wrong_top_level_shape_raises_value_error():
    for bad in ('{}',
                '{"entries":[]}',
                '{"releasable":true}',
                '{"entries":[],"releasable":true,"extra":0}',
                '{"entries":{},"releasable":true}',
                '{"entries":[],"releasable":1}'):
        with pytest.raises(ValueError):
            merge_delivery_manifests((bad,))


def test_non_canonical_encoding_raises_value_error():
    for bad in ('{ "entries":[],"releasable":true}',          # whitespace
                '{"releasable":true,"entries":[]}',           # key order
                '{"entries":[],"releasable":true}\n',         # trailing
                '{"entries":[[0,"a","1",true,[]],'
                '[0,"b","1",true,[]]],"releasable":true}\t'):
        with pytest.raises(ValueError):
            merge_delivery_manifests((bad,))


def test_unsorted_entries_raise_value_error():
    with pytest.raises(ValueError):
        merge_delivery_manifests(
            ('{"entries":[[1,"a","1",true,[]],[0,"a","1",true,[]]],'
             '"releasable":true}',))


def test_duplicate_entry_key_raises_value_error():
    with pytest.raises(ValueError):
        merge_delivery_manifests(
            ('{"entries":[[0,"a","1",true,[]],[0,"a","1",true,[]]],'
             '"releasable":true}',))


def test_releasable_mismatch_raises_value_error():
    for bad in ('{"entries":[],"releasable":false}',
                '{"entries":[[0,"a","1",true,[]]],"releasable":false}',
                '{"entries":[[0,"a","1",false,[]]],"releasable":true}'):
        with pytest.raises(ValueError):
            merge_delivery_manifests((bad,))


def test_bad_entry_fields_raise_value_error():
    for entry in ('[0,"a","1",true]',                    # too short
                  '[0,"a","1",true,[],0]',               # too long
                  '[-1,"a","1",true,[]]',                # negative batch
                  '[true,"a","1",true,[]]',              # bool batch
                  '[0,"","1",true,[]]',                  # empty product
                  '[0,"a b","1",true,[]]',               # illegal product
                  '[0,"a","",true,[]]',                  # empty version
                  '[0,"a","v v",true,[]]',               # illegal version
                  '[0,"a","1",1,[]]',                    # non-bool passed
                  '[0,"a","1",true,{}]'):                # non-array failures
        with pytest.raises(ValueError):
            merge_delivery_manifests(
                ('{"entries":[' + entry + '],"releasable":true}',))


def test_bad_failure_windows_raise_value_error():
    failures = (
        '[0,0,0,0,0]',                                   # too short
        '[0,0,0,0,0,null,false]',                        # too long
        '[-1,0,0,0,0,null]',                             # negative level
        '[0,2,0,1,0,null]',                              # ix_min > ix_max
        '[0,0,2,0,1,null]',                              # iy_min > iy_max
        '[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000]]',   # short R
        '[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,0]]',  # n = 0
        '[0,0,0,0,0,[0.000000,0.000000,-1.000000,0.000000,1]]',  # rmse < 0
        '[0,0,0,0,0,[0.000000,0.000000,0.000000,-1.000000,1]]',  # amean < 0
        '[0,0,0,0,0,[-0.000000,0.000000,0.000000,0.000000,1]]',  # -0
        '[0,0,0,0,0,[0.0,0.000000,0.000000,0.000000,1]]',  # not 6dp
        '[0,0,0,0,0,[1e0,0.000000,0.000000,0.000000,1]]',  # exponent
    )
    for failure in failures:
        with pytest.raises(ValueError):
            merge_delivery_manifests(
                ('{"entries":[[0,"a","1",false,[' + failure + ']]],'
                 '"releasable":false}',))


def test_more_than_six_failures_raise_value_error():
    failures = ",".join("[0,0,0,0,0,null]" for _ in range(7))
    with pytest.raises(ValueError):
        merge_delivery_manifests(
            ('{"entries":[[0,"a","1",false,[' + failures + ']]],'
             '"releasable":false}',))


def test_round_trip_through_build_delivery_manifest():
    manifests = (
        _manifest((0, "a", "1", FAILING_GATE), (1, "a", "2", PASSING_GATE)),
        _manifest((0, "b", "1", EMPTY_GATE)),
    )
    # Every build_delivery_manifest output is accepted verbatim.
    merge_delivery_manifests(manifests)
    for manifest in manifests:
        assert merge_delivery_manifests((manifest,)) == (
            merge_delivery_manifests((manifest, manifest)))
