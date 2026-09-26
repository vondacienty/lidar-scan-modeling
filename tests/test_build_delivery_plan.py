"""Tests for :func:`lidar_scan.build_delivery_plan`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, gate_report,
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

SECOND_FAILING_GATE = (
    '{"windows":[[3,0,0,1,1,null,false],'
    '[4,2,2,3,3,[2.000000,2.000000,2.000000,2.000000,4],false]],'
    '"passed":false}'
)


def _manifest(*items):
    return build_delivery_manifest(tuple(items))


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


def _audit(*manifests):
    return audit_delivery_changes(_changes(*manifests))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.build_delivery_plan is build_delivery_plan
    import lidar_scan
    assert "build_delivery_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_products():
    assert build_delivery_plan(_audit()) == (
        '{"operations":[],"releasable":true}'
    )


def test_document_key_order_and_operation_shape():
    text = build_delivery_plan(_audit(
        _manifest((0, "p", "1.0", PASSING_GATE))))
    assert list(json.loads(text)) == ["operations", "releasable"]
    assert text == (
        '{"operations":[["p","publish","1.0","1.0",[],[]]],'
        '"releasable":true}'
    )


def test_operations_follow_product_order():
    text = build_delivery_plan(_audit(
        _manifest((0, "z", "1", EMPTY_GATE), (0, "a", "1", EMPTY_GATE)),
    ))
    assert [row[0] for row in json.loads(text)["operations"]] == ["a", "z"]


def test_passing_product_publishes_current_with_empty_arrays():
    text = build_delivery_plan(_audit(
        _manifest((0, "p", "1", FAILING_GATE)),
        _manifest((1, "p", "2", PASSING_GATE)),
    ))
    assert json.loads(text)["operations"][0] == (
        ["p", "publish", "2", "2", [], []]
    )


# ---------------------------------------------------------------------------
# rollback and block actions
# ---------------------------------------------------------------------------

def test_failing_product_with_rollback_rolls_back():
    text = build_delivery_plan(_audit(
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
        _manifest((2, "p", "3", PASSING_GATE)),
        _manifest((3, "p", "4", FAILING_GATE)),
    ))
    operation = json.loads(text)["operations"][0]
    assert operation[:4] == ["p", "rollback", "4", "3"]
    assert operation[4] == [3]
    assert [failure[0] for failure in operation[5]] == [3, 3]


def test_never_passed_product_is_blocked_with_null_target():
    text = build_delivery_plan(_audit(
        _manifest((1, "p", "1", FAILING_GATE)),
        _manifest((2, "p", "2", SECOND_FAILING_GATE)),
    ))
    operation = json.loads(text)["operations"][0]
    assert operation[:4] == ["p", "block", "2", None]
    assert operation[4] == [1, 2]


def test_failing_operation_keeps_affected_and_failures_verbatim():
    audit = _audit(
        _manifest((7, "p", "v2", FAILING_GATE)),
    )
    audit_row = json.loads(audit)["products"][0]
    operation = json.loads(build_delivery_plan(audit))["operations"][0]
    assert operation[4] == audit_row[4]
    assert operation[5] == audit_row[5]
    assert operation[5][0] == [
        7, "v2", 0, 0, 0, 9, 9,
        [-1.0, 2.0, 1.581139, 1.0, 1],
    ]
    assert operation[5][1] == [7, "v2", 0, 100, 100, 109, 109, None]


def test_mixed_products_yield_mixed_actions():
    text = build_delivery_plan(_audit(
        _manifest((0, "a", "1", PASSING_GATE), (0, "b", "1", PASSING_GATE),
                  (0, "c", "1", FAILING_GATE)),
        _manifest((1, "b", "2", FAILING_GATE)),
    ))
    operations = json.loads(text)["operations"]
    assert [(row[0], row[1], row[3]) for row in operations] == [
        ("a", "publish", "1"),
        ("b", "rollback", "1"),
        ("c", "block", None),
    ]


# ---------------------------------------------------------------------------
# releasable echoes the input document
# ---------------------------------------------------------------------------

def test_releasable_echoes_input_flag():
    assert build_delivery_plan(_audit(
        _manifest((0, "a", "1", PASSING_GATE)),
    )).endswith('"releasable":true}')
    assert build_delivery_plan(_audit(
        _manifest((0, "a", "1", FAILING_GATE)),
    )).endswith('"releasable":false}')


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_compact_json_without_whitespace_or_trailing_newline():
    text = build_delivery_plan(_audit(
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
    text = build_delivery_plan(_audit(
        _manifest((0, "p", "1", gate)),
    ))
    assert "[0.000000,0.000000,0.000000,0.000000,9]" in text
    assert "-0.000000" not in text


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    text = build_delivery_plan(_audit(
        _manifest((batch, "p", "1", FAILING_GATE)),
    ))
    assert str(batch) in text
    assert json.loads(text)["operations"][0][4] == [batch]


def test_deterministic_and_does_not_modify_input():
    audit = _audit(
        _manifest((2, "b", "1", FAILING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE)),
    )
    snapshot = audit
    assert build_delivery_plan(audit) == build_delivery_plan(audit)
    assert audit == snapshot


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_audit", [
    None, 1, 1.5, b"", [], {}, (),
])
def test_non_str_raises_type_error(bad_audit):
    with pytest.raises(TypeError):
        build_delivery_plan(bad_audit)


@pytest.mark.parametrize("bad_audit", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"operations":[],"releasable":true}',         # plan output is not input
    '{"changes":[],"releasable":true}',            # changes shape
    '{"entries":[],"releasable":true}',            # manifest shape
    '{"products":[]}',                             # missing releasable
    '{"products":[],"releasable":1}',
    '{"products":{},"releasable":true}',
    '{"products":null,"releasable":true}',
    '{"releasable":true,"products":[]}',           # wrong key order
    '{"products":[],"releasable":true,"x":1}',     # extra key
    '{"products":[],"releasable":false}',          # inconsistent AND
    '{"products":[],"releasable":true} ',          # trailing whitespace
    ' {"products":[],"releasable":true}',          # leading whitespace
    '{"products":[[]],"releasable":false}',        # empty product row
    '{"products":[["p","1",true,null,[],[]]],"releasable":false}',
    '{"products":[["p","1",false,null,[],[]]],"releasable":false}',
    '{"products":[["p","1",true,"0",[],[]]],"releasable":true}',
    '{"products":[["p","1",true,null,[],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":true}',
    '{"products":[["p","1",false,null,[],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p","1",false,null,[0],[]]],"releasable":false}',
    '{"products":[["p","1",false,null,[1],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":false}',  # affected != failures
    '{"products":[["p","1",false,null,[0,1],'
    '[[1,"1",0,0,0,0,0,null],[0,"1",0,0,0,0,0,null]]]],'
    '"releasable":false}',                              # failures unordered
    '{"products":[["p","1",false,null,[1,0],'
    '[[0,"1",0,0,0,0,0,null],[1,"1",0,0,0,0,0,null]]]],'
    '"releasable":false}',                              # affected unsorted
    '{"products":[["p","1",false,null,[-1],'
    '[[-1,"1",0,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p","1",false,null,[true],'
    '[[true,"1",0,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p/x","1",true,null,[],[]]],"releasable":true}',
    '{"products":[["p","",true,null,[],[]]],"releasable":true}',
    '{"products":[["p","1",true,null,[0],[]]],"releasable":true}',
    '{"products":[["p","1",1,null,[],[]]],"releasable":true}',
    '{"products":[["p","1",false,"",[0],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p","1",false,null,[0],'
    '[[0,"",0,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p","1",false,null,[0],'
    '[[0,"1",-1,0,0,0,0,null]]]],"releasable":false}',
    '{"products":[["p","1",false,null,[0],'
    '[[0,"1",0,2,0,1,0,null]]]],"releasable":false}',  # ix_min > ix_max
    '{"products":[["p","1",false,null,[0],'
    '[[0,"1",0,0,0,0,0,[0.000000,0.000000,-1.000000,0.000000,1]]]],'
    '"releasable":false}',                              # negative rmse
    '{"products":[["p","1",false,null,[0],'
    '[[0,"1",0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,0]]]],'
    '"releasable":false}',                              # n < 1
    '{"products":[["p","1",false,null,[0],'
    '[[0,"1",0,0,0,0,0,[0.0,0.000000,0.000000,0.000000,1]]]],'
    '"releasable":false}',                              # non-six-decimal
    '{"products":[["b","1",true,null,[],[]],'
    '["a","1",true,null,[],[]]],"releasable":true}',    # unsorted products
    '{"products":[["a","1",true,null,[],[]],'
    '["a","2",true,null,[],[]]],"releasable":true}',    # duplicate product
])
def test_malformed_audit_raises_value_error(bad_audit):
    with pytest.raises(ValueError):
        build_delivery_plan(bad_audit)


def test_nan_and_infinity_are_rejected():
    with pytest.raises(ValueError):
        build_delivery_plan('{"products":[],"releasable":NaN}')
    with pytest.raises(ValueError):
        build_delivery_plan(
            '{"products":[["p","1",false,null,[0],'
            '[[0,"1",0,0,0,0,0,'
            '[Infinity,0.000000,0.000000,0.000000,1]]]]],'
            '"releasable":false}'
        )


def test_accepts_every_audit_output():
    audit = _audit(
        _manifest((2, "b", "1", FAILING_GATE), (5, "a", "9", PASSING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE), (1, "a", "3", EMPTY_GATE)),
        _manifest((0, "a", "4", FAILING_GATE)),
    )
    json.loads(build_delivery_plan(audit))
