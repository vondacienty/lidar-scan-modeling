"""Tests for :func:`lidar_scan.build_delivery_receipt`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
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


def _audit(*manifests):
    return audit_delivery_changes(_changes(*manifests))


def _plan(*manifests):
    return build_delivery_plan(_audit(*manifests))


def _mixed_plan():
    # a -> publish, b -> rollback, c -> block (see build_delivery_plan tests)
    return _plan(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE),
                  (0, "c", "1", FAILING_GATE)),
        _manifest((1, "b", "2", FAILING_GATE)),
    )


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.build_delivery_receipt is build_delivery_receipt
    import lidar_scan
    assert "build_delivery_receipt" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_plan():
    assert build_delivery_receipt(
        '{"operations":[],"releasable":true}', ()
    ) == '{"receipts":[],"ready":true}'


def test_document_key_order_and_receipt_shape():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    text = build_delivery_receipt(plan, (("p", "succeeded", ""),))
    assert list(json.loads(text)) == ["receipts", "ready"]
    assert text == (
        '{"receipts":[["p","publish","1","1",[],"succeeded","",[]]],'
        '"ready":true}'
    )


def test_receipts_follow_plan_order_regardless_of_result_order():
    plan = _mixed_plan()
    shuffled = (("c", "blocked", "never released"),
                ("a", "succeeded", ""),
                ("b", "failed", "deploy failed"))
    text = build_delivery_receipt(plan, shuffled)
    assert [row[0] for row in json.loads(text)["receipts"]] == [
        "a", "b", "c"]
    assert build_delivery_receipt(plan, tuple(reversed(shuffled))) == text


def test_receipt_preserves_planned_fields_and_failure_rows():
    plan = _plan(
        _manifest((0, "p", "v2", FAILING_GATE)),
    )
    text = build_delivery_receipt(plan, (("p", "blocked", "hold"),))
    receipt = json.loads(text)["receipts"][0]
    assert receipt == [
        "p", "block", "v2", None, [0], "blocked", "hold",
        [[0, "v2", 0, 0, 0, 9, 9,
          [-1.0, 2.0, 1.581139, 1.0, 1]],
         [0, "v2", 0, 100, 100, 109, 109, None]],
    ]
    assert receipt[3] is None


def test_rollback_fields_preserved():
    plan = _plan(
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
    )
    receipt = json.loads(build_delivery_receipt(
        plan, (("p", "failed", "x"),)))["receipts"][0]
    assert receipt[:4] == ["p", "rollback", "2", "1"]
    assert receipt[4] == [1]
    assert [failure[0] for failure in receipt[7]] == [1, 1]


# ---------------------------------------------------------------------------
# status rules
# ---------------------------------------------------------------------------

def test_publish_accepts_succeeded_or_failed():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    assert json.loads(build_delivery_receipt(
        plan, (("p", "succeeded", ""),)))["receipts"][0][5] == "succeeded"
    assert json.loads(build_delivery_receipt(
        plan, (("p", "failed", "boom"),)))["receipts"][0][5] == "failed"


def test_rollback_accepts_succeeded_or_failed():
    plan = _plan(
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
    )
    assert json.loads(build_delivery_receipt(
        plan, (("p", "succeeded", ""),)))["receipts"][0][5] == "succeeded"
    assert json.loads(build_delivery_receipt(
        plan, (("p", "failed", "boom"),)))["receipts"][0][5] == "failed"


def test_block_accepts_only_blocked():
    plan = _plan(_manifest((1, "p", "1", FAILING_GATE)))
    assert json.loads(build_delivery_receipt(
        plan, (("p", "blocked", "hold"),)))["receipts"][0][5] == "blocked"


@pytest.mark.parametrize("status", ["blocked", "pending", "SUCCEEDED", ""])
def test_publish_rejects_other_statuses(status):
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    reason = "x" if status else "x"
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (("p", status, reason),))


def test_block_rejects_succeeded_and_failed():
    plan = _plan(_manifest((1, "p", "1", FAILING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (("p", "failed", "x"),))


def test_succeeded_requires_empty_reason():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (("p", "succeeded", "x"),))


@pytest.mark.parametrize("status,reason", [
    ("failed", ""),
    ("blocked", ""),
])
def test_failed_and_blocked_require_nonempty_reason(status, reason):
    if status == "failed":
        plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    else:
        plan = _plan(_manifest((1, "p", "1", FAILING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (("p", status, reason),))


# ---------------------------------------------------------------------------
# ready
# ---------------------------------------------------------------------------

def test_ready_true_only_when_all_publish_succeeded():
    plan = _plan(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)),
    )
    text = build_delivery_receipt(
        plan, (("b", "succeeded", ""), ("a", "succeeded", "")))
    assert text.endswith('"ready":true}')


@pytest.mark.parametrize("results", [
    (("a", "failed", "x"), ("b", "succeeded", "")),       # a publish failed
    (("a", "succeeded", ""), ("b", "failed", "x")),       # b publish failed
    (("a", "failed", "x"), ("b", "failed", "y")),         # both failed
])
def test_ready_false_when_any_publish_failed(results):
    plan = _plan(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)),
    )
    assert build_delivery_receipt(plan, results).endswith('"ready":false}')


def test_ready_false_when_rollback_succeeds():
    plan = _mixed_plan()
    text = build_delivery_receipt(plan, (("a", "succeeded", ""),
                                         ("b", "succeeded", ""),
                                         ("c", "blocked", "x")))
    assert text.endswith('"ready":false}')


def test_ready_false_when_any_block_present():
    plan = _mixed_plan()
    text = build_delivery_receipt(plan, (("a", "succeeded", ""),
                                         ("b", "failed", "x"),
                                         ("c", "blocked", "x")))
    assert text.endswith('"ready":false}')


# ---------------------------------------------------------------------------
# result matching
# ---------------------------------------------------------------------------

def test_missing_result_raises_value_error():
    plan = _mixed_plan()
    with pytest.raises(ValueError):
        build_delivery_receipt(
            plan, (("a", "succeeded", ""), ("b", "failed", "x")))


def test_unknown_result_raises_value_error():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_receipt(
            plan, (("p", "succeeded", ""), ("q", "failed", "x")))


def test_duplicate_result_raises_value_error():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(ValueError):
        build_delivery_receipt(
            plan, (("p", "succeeded", ""), ("p", "failed", "x")))


def test_empty_plan_rejects_any_result():
    with pytest.raises(ValueError):
        build_delivery_receipt(
            '{"operations":[],"releasable":true}',
            (("p", "succeeded", ""),))


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_compact_json_without_whitespace_or_trailing_newline():
    plan = _mixed_plan()
    text = build_delivery_receipt(plan, (("a", "succeeded", ""),
                                         ("b", "failed", "x"),
                                         ("c", "blocked", "y")))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_unicode_reason_is_not_ascii_escaped():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    text = build_delivery_receipt(plan, (("p", "failed", "部署失败"),))
    assert "部署失败" in text
    assert "\\u" not in text
    json.loads(text)


def test_metrics_keep_six_decimals_and_zero_formatting():
    report = (
        '{"windows":[[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,9]]]}'
    )
    gate = gate_report(report, (1, 1, 1, 10))
    plan = _plan(_manifest((0, "p", "1", gate)))
    text = build_delivery_receipt(plan, (("p", "blocked", "x"),))
    assert "[0.000000,0.000000,0.000000,0.000000,9]" in text
    assert "-0.000000" not in text


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    plan = _plan(_manifest((batch, "p", "1", FAILING_GATE)))
    text = build_delivery_receipt(plan, (("p", "blocked", "x"),))
    assert str(batch) in text
    assert json.loads(text)["receipts"][0][4] == [batch]


def test_deterministic_and_does_not_modify_inputs():
    plan = _mixed_plan()
    results = (("c", "blocked", "x"), ("a", "succeeded", ""),
               ("b", "failed", "y"))
    snapshot = tuple(results)
    assert build_delivery_receipt(plan, results) == (
        build_delivery_receipt(plan, results))
    assert results == snapshot
    assert plan == _mixed_plan()


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_plan", [None, 1, 1.5, b"", [], {}, ()])
def test_non_str_plan_raises_type_error(bad_plan):
    with pytest.raises(TypeError):
        build_delivery_receipt(bad_plan, ())


@pytest.mark.parametrize("bad_results", [
    None, 1, 1.5, b"", [], {}, set(),
])
def test_non_tuple_results_raises_type_error(bad_results):
    plan = '{"operations":[],"releasable":true}'
    with pytest.raises(TypeError):
        build_delivery_receipt(plan, bad_results)


@pytest.mark.parametrize("bad_item", [
    ["p", "succeeded", ""],       # list, not tuple
    ("p", "succeeded"),           # two values
    ("p", "succeeded", "", 1),    # four values
    "p",
])
def test_bad_result_shape_raises_type_error(bad_item):
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(TypeError):
        build_delivery_receipt(plan, (bad_item,))


@pytest.mark.parametrize("bad_item", [
    (1, "succeeded", ""),
    (None, "succeeded", ""),
    ("p", 0, ""),
    ("p", None, ""),
    ("p", "succeeded", 1),
    ("p", "succeeded", None),
    (b"p", "succeeded", ""),
])
def test_non_str_field_raises_type_error(bad_item):
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(TypeError):
        build_delivery_receipt(plan, (bad_item,))


@pytest.mark.parametrize("bad_plan", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"products":[],"releasable":true}',           # audit shape
    '{"changes":[],"releasable":true}',            # changes shape
    '{"entries":[],"releasable":true}',            # manifest shape
    '{"receipts":[],"ready":true}',                # receipt output
    '{"operations":[]}',                           # missing releasable
    '{"operations":null,"releasable":true}',
    '{"operations":{},"releasable":true}',
    '{"releasable":true,"operations":[]}',         # wrong key order
    '{"operations":[],"releasable":1}',
    '{"operations":[],"releasable":true,"x":1}',   # extra key
    '{"operations":[],"releasable":false}',        # inconsistent AND
    '{"operations":[],"releasable":true} ',        # trailing whitespace
    ' {"operations":[],"releasable":true}',        # leading whitespace
    '{"operations":[[]],"releasable":false}',      # empty operation row
    '{"operations":[["p","publish","1","1",[],[]],'
    '["a","publish","1","1",[],[]]],"releasable":true}',   # unsorted
    '{"operations":[["p","publish","1","1",[],[]],'
    '["p","publish","2","2",[],[]]],"releasable":true}',   # duplicate
    '{"operations":[["p","ship","1","1",[],[]]],'
    '"releasable":false}',                                 # bad action
    '{"operations":[["p","publish","1","2",[],[]]],'
    '"releasable":false}',                                 # publish retarget
    '{"operations":[["p","publish","1","1",[0],[]]],'
    '"releasable":false}',                                 # publish affected
    '{"operations":[["p","block","1",null,[],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":false}',      # block no affected
    '{"operations":[["p","rollback","1",null,[0],'
    '[[0,"1",0,0,0,0,0,null]]]],"releasable":false}',      # rollback null target
    '{"operations":[["p","block","1","2",[0],'
    '[[0,"2",0,0,0,0,0,null]]]],"releasable":false}',      # block with target
])
def test_malformed_plan_raises_value_error(bad_plan):
    with pytest.raises(ValueError):
        build_delivery_receipt(bad_plan, ())


def test_nan_and_infinity_in_plan_are_rejected():
    with pytest.raises(ValueError):
        build_delivery_receipt(
            '{"operations":[],"releasable":NaN}', ())
    with pytest.raises(ValueError):
        build_delivery_receipt(
            '{"operations":[],"releasable":Infinity}', ())


def test_accepts_every_plan_output_with_matching_results():
    audit = _audit(
        _manifest((2, "b", "1", FAILING_GATE),
                  (5, "a", "9", PASSING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE),
                  (1, "a", "3", EMPTY_GATE)),
        _manifest((0, "a", "4", FAILING_GATE)),
    )
    plan = build_delivery_plan(audit)
    rows = json.loads(plan)["operations"]
    results = []
    for product, action in ((row[0], row[1]) for row in rows):
        if action == "block":
            results.append((product, "blocked", "held"))
        else:
            results.append((product, "succeeded", ""))
    json.loads(build_delivery_receipt(plan, tuple(reversed(results))))
