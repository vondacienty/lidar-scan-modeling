"""Tests for :func:`lidar_scan.audit_delivery_changes`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        gate_report, merge_delivery_manifests)
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

SECOND_FAILING_GATE = (
    '{"windows":[[3,0,0,1,1,null,false],'
    '[4,2,2,3,3,[2.000000,2.000000,2.000000,2.000000,4],false]],'
    '"passed":false}'
)


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.audit_delivery_changes is audit_delivery_changes
    import lidar_scan
    assert "audit_delivery_changes" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_changes():
    assert audit_delivery_changes(_changes()) == (
        '{"products":[],"releasable":true}'
    )


def test_document_key_order_and_product_row_shape():
    text = audit_delivery_changes(
        _changes(_manifest((0, "p", "1.0", PASSING_GATE))))
    assert list(json.loads(text)) == ["products", "releasable"]
    assert text == (
        '{"products":[["p","1.0",true,null,[],[]]],"releasable":true}'
    )


def test_products_sorted_by_product_name():
    text = audit_delivery_changes(_changes(
        _manifest((0, "z", "1", EMPTY_GATE), (0, "a", "1", EMPTY_GATE)),
    ))
    assert [row[0] for row in json.loads(text)["products"]] == ["a", "z"]


def test_current_and_passed_come_from_final_batch():
    text = audit_delivery_changes(_changes(
        _manifest((1, "p", "1", FAILING_GATE)),
        _manifest((2, "p", "2", PASSING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[0] == "p" and row[1] == "2" and row[2] is True


def test_passing_final_change_clears_rollback_affected_and_failures():
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", FAILING_GATE)),
        _manifest((1, "p", "2", PASSING_GATE)),
    ))
    assert text == (
        '{"products":[["p","2",true,null,[],[]]],"releasable":true}'
    )


# ---------------------------------------------------------------------------
# rollback and affected batches
# ---------------------------------------------------------------------------

def test_never_passed_product_has_null_rollback_and_all_failing_batches():
    text = audit_delivery_changes(_changes(
        _manifest((1, "p", "1", FAILING_GATE)),
        _manifest((2, "p", "2", SECOND_FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[:5] == ["p", "2", False, None, [1, 2]]
    assert [failure[0] for failure in row[5]] == [1, 1, 2, 2]
    assert [failure[1] for failure in row[5]] == ["1", "1", "2", "2"]


def test_rollback_is_nearest_earlier_passing_version():
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
        _manifest((2, "p", "3", PASSING_GATE)),
        _manifest((3, "p", "4", FAILING_GATE)),
        _manifest((4, "p", "5", SECOND_FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[1] == "5" and row[2] is False
    assert row[3] == "3"
    assert row[4] == [3, 4]
    assert [failure[0] for failure in row[5]] == [3, 3, 4, 4]
    assert [failure[1] for failure in row[5]] == ["4", "4", "5", "5"]


def test_failures_before_rollback_pass_are_excluded():
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", FAILING_GATE)),
        _manifest((1, "p", "2", PASSING_GATE)),
        _manifest((2, "p", "3", FAILING_GATE)),
        _manifest((3, "p", "4", PASSING_GATE)),
        _manifest((4, "p", "5", SECOND_FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[3] == "4"
    assert row[4] == [4]
    assert {failure[0] for failure in row[5]} == {4}


def test_non_adjacent_batch_numbers_are_respected():
    text = audit_delivery_changes(_changes(
        _manifest((10, "p", "1", PASSING_GATE)),
        _manifest((20, "p", "2", FAILING_GATE)),
        _manifest((30, "p", "3", PASSING_GATE)),
        _manifest((40, "p", "4", FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[3] == "3" and row[4] == [40]


# ---------------------------------------------------------------------------
# failure expansion
# ---------------------------------------------------------------------------

def test_failure_window_fields_and_r_preserved():
    text = audit_delivery_changes(_changes(
        _manifest((7, "p", "v2", FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert row[5][0] == [
        7, "v2", 0, 0, 0, 9, 9,
        [-1.0, 2.0, 1.581139, 1.0, 1],
    ]
    assert row[5][1] == [7, "v2", 0, 100, 100, 109, 109, None]


def test_failures_follow_batch_then_source_window_order():
    report = (
        '{"windows":[[0,0,0,0,0,null],'
        '[1,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,2]],'
        '[2,0,0,0,0,null]]}'
    )
    gate = gate_report(report, (1, 1, 1, 2))
    text = audit_delivery_changes(_changes(
        _manifest((5, "p", "1", gate)),
        _manifest((6, "p", "2", FAILING_GATE)),
    ))
    row = json.loads(text)["products"][0]
    assert [(failure[0], failure[2]) for failure in row[5]] == [
        (5, 0), (5, 2), (6, 0), (6, 0),
    ]


def test_at_most_six_failures_per_change_survive():
    windows = ",".join(
        f"[{level},0,0,0,0,null]" for level in range(7)
    )
    report = '{"windows":[' + windows + "]} "
    gate = gate_report(report.rstrip(), (1, 1, 1, 1))
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", gate)),
    ))
    row = json.loads(text)["products"][0]
    assert [failure[2] for failure in row[5]] == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# releasable echoes the input document
# ---------------------------------------------------------------------------

def test_releasable_echoes_input_flag():
    releasable = audit_delivery_changes(_changes(
        _manifest((0, "a", "1", PASSING_GATE)),
    ))
    assert releasable.endswith('"releasable":true}')

    blocked = audit_delivery_changes(_changes(
        _manifest((0, "a", "1", FAILING_GATE)),
    ))
    assert blocked.endswith('"releasable":false}')


def test_releasable_true_even_with_failing_history_when_final_passes():
    text = audit_delivery_changes(_changes(
        _manifest((0, "a", "1", FAILING_GATE)),
        _manifest((1, "a", "2", PASSING_GATE)),
        _manifest((0, "b", "1", PASSING_GATE)),
    ))
    assert json.loads(text)["releasable"] is True


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_compact_json_without_whitespace_or_trailing_newline():
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", FAILING_GATE)),
    ))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_metrics_keep_six_decimals_and_zero_formatting():
    # Window metrics are within the limits but n < min_n, so it fails while
    # carrying canonical zero metrics.
    report = (
        '{"windows":[[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,9]]]}'
    )
    gate = gate_report(report, (1, 1, 1, 10))
    text = audit_delivery_changes(_changes(
        _manifest((0, "p", "1", gate)),
    ))
    assert "[0.000000,0.000000,0.000000,0.000000,9]" in text
    assert "-0.000000" not in text


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    text = audit_delivery_changes(_changes(
        _manifest((batch, "p", "1", FAILING_GATE)),
    ))
    assert str(batch) in text
    assert json.loads(text)["products"][0][4] == [batch]


def test_deterministic_and_does_not_modify_input():
    changes = _changes(
        _manifest((2, "b", "1", FAILING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE)),
    )
    snapshot = changes
    assert audit_delivery_changes(changes) == audit_delivery_changes(changes)
    assert changes == snapshot


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_changes", [
    None, 1, 1.5, b"", [], {}, (),
])
def test_non_str_raises_type_error(bad_changes):
    with pytest.raises(TypeError):
        audit_delivery_changes(bad_changes)


@pytest.mark.parametrize("bad_changes", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"products":[],"releasable":true}',          # audit output is not input
    '{"entries":[],"releasable":true}',           # manifest shape
    '{"changes":[]}',                             # missing releasable
    '{"changes":[],"releasable":1}',
    '{"changes":{},"releasable":true}',
    '{"changes":null,"releasable":true}',
    '{"releasable":true,"changes":[]}',           # wrong key order
    '{"changes":[],"releasable":true,"x":1}',     # extra key
    '{"changes":[],"releasable":false}',          # inconsistent AND
    '{"changes":[],"releasable":true} ',          # trailing whitespace
    ' {"changes":[],"releasable":true}',          # leading whitespace
    '{"changes":[[]],"releasable":false}',        # empty change
    '{"changes":[[0,"p",null,"1",true,[]]],"releasable":false}',
    '{"changes":[[0,"p",null,"1",false,[]]],"releasable":false}',
    '{"changes":[[0,"p",null,"1",true,'
    '[[0,0,0,0,0,null]]],"releasable":false}',    # passing change w/ failure
    '{"changes":[[true,"p",null,"1",true,[]]],"releasable":true}',
    '{"changes":[[-1,"p",null,"1",true,[]]],"releasable":true}',
    '{"changes":[[0,"p/x",null,"1",true,[]]],"releasable":true}',
    '{"changes":[[0,"p",null,"",true,[]]],"releasable":true}',
    '{"changes":[[0,"p","1","1",true,[]]],"releasable":true}',  # bad previous
    '{"changes":[[1,"p",null,"1",true,[]],[0,"p","1","2",true,[]]],'
    '"releasable":true}',                                       # unsorted
    '{"changes":[[0,"p",null,"1",true,[]],[0,"p",null,"1",true,[]]],'
    '"releasable":true}',                                       # duplicate key
])
def test_malformed_changes_raise_value_error(bad_changes):
    with pytest.raises(ValueError):
        audit_delivery_changes(bad_changes)


def test_nan_and_infinity_are_rejected():
    with pytest.raises(ValueError):
        audit_delivery_changes('{"changes":[],"releasable":NaN}')
    with pytest.raises(ValueError):
        audit_delivery_changes(
            '{"changes":[[0,"p",null,"1",false,'
            '[[0,0,0,0,0,[Infinity,0.000000,0.000000,0.000000,1]]]],'
            '"releasable":false}'
        )


def test_accepts_every_merge_output():
    changes = _changes(
        _manifest((2, "b", "1", FAILING_GATE), (5, "a", "9", PASSING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE), (1, "a", "3", EMPTY_GATE)),
        _manifest((0, "a", "4", FAILING_GATE)),
    )
    json.loads(audit_delivery_changes(changes))
