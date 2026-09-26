"""Tests for :func:`lidar_scan.merge_delivery_manifests`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_delivery_manifest, gate_report,
                        merge_delivery_manifests)
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

def test_empty_manifests():
    assert merge_delivery_manifests(()) == '{"changes":[],"releasable":true}'


def test_change_shape_and_key_order():
    text = merge_delivery_manifests((
        _manifest((0, "prod", "1.0.0", EMPTY_GATE)),
    ))
    assert list(json.loads(text)) == ["changes", "releasable"]
    assert text == (
        '{"changes":[[0,"prod",null,"1.0.0",true,[]]],"releasable":true}'
    )


def test_changes_sorted_by_batch_then_product():
    text = merge_delivery_manifests((
        _manifest((2, "b", "1", EMPTY_GATE)),
        _manifest((0, "z", "1", EMPTY_GATE), (1, "a", "2", EMPTY_GATE),
                  (0, "a", "1", EMPTY_GATE)),
    ))
    doc = json.loads(text)
    assert [(c[0], c[1]) for c in doc["changes"]] == [
        (0, "a"), (0, "z"), (1, "a"), (2, "b")
    ]


def test_previous_is_prior_batch_version_or_null():
    text = merge_delivery_manifests((
        _manifest((0, "a", "1.0", EMPTY_GATE), (0, "b", "x", EMPTY_GATE)),
        _manifest((1, "a", "1.1", EMPTY_GATE)),
        _manifest((2, "a", "2.0", EMPTY_GATE), (1, "c", "c1", EMPTY_GATE)),
    ))
    assert [(c[1], c[2], c[3]) for c in json.loads(text)["changes"]] == [
        ("a", None, "1.0"),
        ("b", None, "x"),
        ("a", "1.0", "1.1"),
        ("c", None, "c1"),
        ("a", "1.1", "2.0"),
    ]


def test_previous_follows_batch_order_not_merge_order():
    forward = (
        _manifest((0, "p", "1", EMPTY_GATE)),
        _manifest((1, "p", "2", EMPTY_GATE)),
        _manifest((2, "p", "3", EMPTY_GATE)),
    )
    backward = tuple(reversed(forward))
    assert merge_delivery_manifests(forward) == merge_delivery_manifests(
        backward)


def test_failures_preserved_with_metrics():
    text = merge_delivery_manifests((
        _manifest((7, "prod", "v2", FAILING_GATE)),
    ))
    assert text == (
        '{"changes":[[7,"prod",null,"v2",false,'
        '[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,null]]]],"releasable":false}'
    )
    change = json.loads(text)["changes"][0]
    assert len(change[5][0]) == 6


def test_failures_keep_gate_window_order():
    report = (
        '{"windows":[[0,0,0,0,0,null],'
        '[1,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,2]],'
        '[2,0,0,0,0,null]]}'
    )
    gate = gate_report(report, (1, 1, 1, 2))
    manifest = _manifest((0, "p", "1", gate))
    doc = json.loads(merge_delivery_manifests((manifest,)))
    assert [window[0] for window in doc["changes"][0][5]] == [0, 2]


def test_releasable_uses_final_batch_per_product():
    # Product "a": batch 1 fails, batch 2 passes -> releasable.
    # Product "b": batch 0 passes, batch 3 fails -> blocks release.
    text = merge_delivery_manifests((
        _manifest((1, "a", "1", FAILING_GATE), (0, "b", "1", PASSING_GATE)),
        _manifest((2, "a", "2", PASSING_GATE), (3, "b", "2", FAILING_GATE)),
    ))
    assert json.loads(text)["releasable"] is False

    text = merge_delivery_manifests((
        _manifest((1, "a", "1", FAILING_GATE), (0, "b", "1", PASSING_GATE)),
        _manifest((2, "a", "2", PASSING_GATE)),
    ))
    assert json.loads(text)["releasable"] is True


def test_passing_history_does_not_make_final_failure_releasable():
    text = merge_delivery_manifests((
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
    ))
    assert json.loads(text)["releasable"] is False


# ---------------------------------------------------------------------------
# deduplication and conflicts
# ---------------------------------------------------------------------------

def test_identical_entries_are_deduplicated():
    manifest = _manifest((0, "p", "1", FAILING_GATE),
                         (1, "q", "2", PASSING_GATE))
    assert merge_delivery_manifests((manifest, manifest, manifest)) == (
        merge_delivery_manifests((manifest,)))


def test_conflicting_entries_raise_value_error():
    first = _manifest((0, "p", "1", EMPTY_GATE))
    for other_text in (
        _manifest((0, "p", "2", EMPTY_GATE)),
        _manifest((0, "p", "1", FAILING_GATE)),
    ):
        with pytest.raises(ValueError):
            merge_delivery_manifests((first, other_text))


def test_same_version_same_batch_product_is_exact_dedupe():
    # Different gate text for the same key is a conflict, not a dedupe.
    first = _manifest((0, "p", "1", PASSING_GATE))
    same = _manifest((0, "p", "1", PASSING_GATE))
    assert merge_delivery_manifests((first, same)) == (
        merge_delivery_manifests((first,)))


# ---------------------------------------------------------------------------
# version reuse
# ---------------------------------------------------------------------------

def test_version_reuse_across_batches_raises_value_error():
    with pytest.raises(ValueError):
        merge_delivery_manifests((
            _manifest((0, "p", "1.0", EMPTY_GATE)),
            _manifest((5, "p", "1.0", EMPTY_GATE)),
        ))


def test_same_version_for_different_products_is_allowed():
    text = merge_delivery_manifests((
        _manifest((0, "a", "1.0", EMPTY_GATE), (0, "b", "1.0", EMPTY_GATE)),
    ))
    assert json.loads(text)["releasable"] is True


# ---------------------------------------------------------------------------
# determinism and immutability
# ---------------------------------------------------------------------------

def test_reordering_manifests_is_byte_identical():
    manifests = (
        _manifest((2, "b", "1", FAILING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE), (1, "a", "3", EMPTY_GATE)),
        _manifest((0, "a", "4", FAILING_GATE)),
    )
    assert merge_delivery_manifests(manifests) == merge_delivery_manifests(
        tuple(reversed(manifests)))


def test_inputs_are_not_modified():
    manifests = (
        _manifest((2, "b", "1", FAILING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE)),
    )
    snapshot = tuple(manifests)
    merge_delivery_manifests(manifests)
    assert manifests == snapshot


def test_no_whitespace_or_trailing_newline():
    text = merge_delivery_manifests((
        _manifest((0, "p", "1", FAILING_GATE), (1, "q", "2", PASSING_GATE)),
    ))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    text = merge_delivery_manifests((
        _manifest((batch, "p", "1", EMPTY_GATE)),
    ))
    assert "[" + str(batch) + "," in text
    assert json.loads(text)["changes"][0][0] == batch


# ---------------------------------------------------------------------------
# argument validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_manifests", [
    None, [], {}, "x", 1, 1.5,
])
def test_manifests_wrong_type_raises_type_error(bad_manifests):
    with pytest.raises(TypeError):
        merge_delivery_manifests(bad_manifests)


@pytest.mark.parametrize("bad_member", [
    None, 1, 1.5, b"", [], {},
])
def test_member_wrong_type_raises_type_error(bad_member):
    with pytest.raises(TypeError):
        merge_delivery_manifests((_manifest((0, "p", "1", EMPTY_GATE)),
                                  bad_member))


# ---------------------------------------------------------------------------
# argument validation: malformed manifest strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_manifest", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"entries":[]}',
    '{"changes":[],"releasable":true}',
    '{"entries":[],"releasable":1}',
    '{"entries":{},"releasable":true}',
    '{"entries":null,"releasable":true}',
    '{"releasable":true,"entries":[]}',                # wrong key order
    '{"entries":[],"releasable":true,"x":1}',          # extra key
    '{"entries":[],"releasable":true} ',               # trailing whitespace
    ' { "entries":[],"releasable":true}',              # leading whitespace
    '{"entries":[],"releasable":false}',               # inconsistent AND
    '{"entries":[[]],"releasable":false}',             # empty entry
    '{"entries":[[0,"p","1",true]],"releasable":true}',  # four-value entry
    '{"entries":[[0,"p","1",true,[]]],"releasable":true} ',
    '{"entries":[[true,"p","1",true,[]]],"releasable":false}',
    '{"entries":[[-1,"p","1",true,[]]],"releasable":false}',
    '{"entries":[[0,"p/x","1",true,[]]],"releasable":false}',
    '{"entries":[[0,"p","",true,[]]],"releasable":false}',
    '{"entries":[[0,"p","1","yes",[]]],"releasable":false}',
    '{"entries":[[0,"p","1",true,{}]],"releasable":false}',
    '{"entries":[[0,"p","1",false,[[0,0,0,0,0]]]],'
    '"releasable":false}',                             # five-value failure
    '{"entries":[[0,"p","1",false,[[0,0,0,0,0,null],'
    '[0,0,0,0,0,null],[0,0,0,0,0,null],[0,0,0,0,0,null],'
    '[0,0,0,0,0,null],[0,0,0,0,0,null],[0,0,0,0,0,null]]]],'
    '"releasable":false}',                             # seven failures
    '{"entries":[[0,"p","1",false,'
    '[[0,0,0,0,0,[-0.000000,0.000000,0.000000,0.000000,1]]]]'
    ',"releasable":false}',                            # negative zero
    '{"entries":[[0,"p","1",false,'
    '[[0,0,0,0,0,[0.00000,0.000000,0.000000,0.000000,1]]]]'
    ',"releasable":false}',                            # five decimals
    '{"entries":[[0,"p","1",false,'
    '[[0,0,0,0,0,[0.000000,0.000000,-1.000000,0.000000,1]]]]'
    ',"releasable":false}',                            # negative rmse
    '{"entries":[[0,"p","1",false,'
    '[[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,0]]]]'
    ',"releasable":false}',                            # n zero
    '{"entries":[[1,"p","1",true,[]],[0,"p","1",true,[]]],'
    '"releasable":true}',                              # unsorted entries
    '{"entries":[[0,"p","1",true,[]],[0,"p","2",true,[]]],'
    '"releasable":true}',                              # duplicate key
])
def test_bad_manifest_raises_value_error(bad_manifest):
    with pytest.raises(ValueError):
        merge_delivery_manifests((bad_manifest,))


def test_nan_and_infinity_are_rejected():
    with pytest.raises(ValueError):
        merge_delivery_manifests((
            '{"entries":[],"releasable":NaN}',
        ))
    with pytest.raises(ValueError):
        merge_delivery_manifests((
            '{"entries":[[0,"p","1",false,'
            '[[0,0,0,0,0,[Infinity,0.000000,0.000000,0.000000,1]]]]'
            ',"releasable":false}',
        ))


def test_bad_manifest_in_later_tuple_member_rejected():
    with pytest.raises(ValueError):
        merge_delivery_manifests((
            _manifest((0, "p", "1", EMPTY_GATE)),
            "not json",
        ))
