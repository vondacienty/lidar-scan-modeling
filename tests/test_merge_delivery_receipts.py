"""Tests for :func:`lidar_scan.merge_delivery_receipts`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        gate_report, merge_delivery_manifests,
                        merge_delivery_receipts)
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

EMPTY_RECEIPT = '{"receipts":[],"ready":true}'


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


def _mixed_results(status_a="succeeded", reason_a="",
                   status_b="failed", reason_b="deploy failed",
                   status_c="blocked", reason_c="held"):
    return (("a", status_a, reason_a),
            ("b", status_b, reason_b),
            ("c", status_c, reason_c))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.merge_delivery_receipts is merge_delivery_receipts
    import lidar_scan
    assert "merge_delivery_receipts" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_input():
    assert merge_delivery_receipts(()) == '{"products":[],"ready":true}'


def test_empty_receipts_merge_to_empty_products():
    assert merge_delivery_receipts(
        (EMPTY_RECEIPT, EMPTY_RECEIPT)) == '{"products":[],"ready":true}'


def test_single_receipt_single_product_byte_exact():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    receipt = build_delivery_receipt(plan, (("p", "succeeded", ""),))
    assert merge_delivery_receipts((receipt,)) == (
        '{"products":[["p","publish","1","1",[],[["succeeded",""]],'
        '"succeeded",[]]],"ready":true}'
    )


def test_document_key_order_and_product_shape():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    receipt = build_delivery_receipt(plan, (("p", "succeeded", ""),))
    document = json.loads(merge_delivery_receipts((receipt,)))
    assert list(document) == ["products", "ready"]
    assert list(json.loads(merge_delivery_receipts((receipt,)))
                ["products"][0]) == [
        "p", "publish", "1", "1", [], [["succeeded", ""]], "succeeded", []]


def test_products_sorted_by_name_regardless_of_plan_order():
    plan = _mixed_plan()
    receipt = build_delivery_receipt(plan, _mixed_results())
    products = json.loads(merge_delivery_receipts((receipt,)))["products"]
    assert [row[0] for row in products] == ["a", "b", "c"]


def test_history_follows_receipt_input_order_and_final_is_last():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    receipts = tuple(
        build_delivery_receipt(plan, (("p", status, reason),))
        for status, reason in (("failed", "first"),
                               ("succeeded", ""),
                               ("failed", "third")))
    product = json.loads(merge_delivery_receipts(receipts))["products"][0]
    assert product[5] == [["failed", "first"], ["succeeded", ""],
                          ["failed", "third"]]
    assert product[6] == "failed"


def test_failures_keep_original_order():
    plan = _plan(_manifest((0, "p", "v2", FAILING_GATE)))
    receipt = build_delivery_receipt(plan, (("p", "blocked", "hold"),))
    product = json.loads(merge_delivery_receipts((receipt,)))["products"][0]
    assert product[7] == [
        [0, "v2", 0, 0, 0, 9, 9, [-1.0, 2.0, 1.581139, 1.0, 1]],
        [0, "v2", 0, 100, 100, 109, 109, None],
    ]


def test_mixed_plan_merge():
    plan = _mixed_plan()
    first = build_delivery_receipt(plan, _mixed_results())
    second = build_delivery_receipt(
        plan, _mixed_results(status_b="succeeded", reason_b="",
                             reason_c="still held"))
    products = json.loads(
        merge_delivery_receipts((first, second)))["products"]
    assert products[0] == ["a", "publish", "1", "1", [],
                           [["succeeded", ""], ["succeeded", ""]],
                           "succeeded", []]
    assert products[1][:5] == ["b", "rollback", "2", "1", [1]]
    assert products[1][5] == [["failed", "deploy failed"],
                              ["succeeded", ""]]
    assert products[1][6] == "succeeded"
    assert products[2][:5] == ["c", "block", "1", None, [0]]
    assert products[2][5] == [["blocked", "held"], ["blocked", "still held"]]
    assert products[2][6] == "blocked"


# ---------------------------------------------------------------------------
# ready
# ---------------------------------------------------------------------------

def test_ready_true_when_every_product_is_a_succeeded_publish():
    plan = _plan(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)),
    )
    receipt = build_delivery_receipt(
        plan, (("a", "succeeded", ""), ("b", "succeeded", "")))
    assert merge_delivery_receipts((receipt,)).endswith('"ready":true}')


def test_ready_false_when_a_final_status_is_failed():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    ok = build_delivery_receipt(plan, (("p", "succeeded", ""),))
    bad = build_delivery_receipt(plan, (("p", "failed", "boom"),))
    merged = merge_delivery_receipts((ok, bad))
    assert merged.endswith('"ready":false}')
    assert merge_delivery_receipts((bad, ok)).endswith('"ready":true}')


def test_ready_false_with_rollback_or_block_products():
    plan = _mixed_plan()
    receipt = build_delivery_receipt(
        plan, _mixed_results(status_b="succeeded", reason_b=""))
    assert merge_delivery_receipts((receipt,)).endswith('"ready":false}')


# ---------------------------------------------------------------------------
# cross-receipt consistency
# ---------------------------------------------------------------------------

def test_different_product_sets_raise_value_error():
    plan_p = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    plan_q = _plan(_manifest((0, "q", "1", PASSING_GATE)))
    receipt_p = build_delivery_receipt(plan_p, (("p", "succeeded", ""),))
    receipt_q = build_delivery_receipt(plan_q, (("q", "succeeded", ""),))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_p, receipt_q))


def test_different_product_counts_raise_value_error():
    plan_one = _plan(_manifest((0, "a", "1", PASSING_GATE)))
    plan_two = _plan(
        _manifest((0, "a", "1", PASSING_GATE),
                  (0, "b", "1", PASSING_GATE)))
    receipt_one = build_delivery_receipt(plan_one, (("a", "succeeded", ""),))
    receipt_two = build_delivery_receipt(
        plan_two, (("a", "succeeded", ""), ("b", "succeeded", "")))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_one, receipt_two))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_two, receipt_one))


def test_conflicting_action_raises_value_error():
    publish_plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    block_plan = _plan(_manifest((0, "p", "1", FAILING_GATE)))
    publish = build_delivery_receipt(publish_plan, (("p", "succeeded", ""),))
    block = build_delivery_receipt(block_plan, (("p", "blocked", "held"),))
    with pytest.raises(ValueError):
        merge_delivery_receipts((publish, block))


def test_conflicting_current_raises_value_error():
    plan_one = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    plan_two = _plan(_manifest((0, "p", "2", PASSING_GATE)))
    receipt_one = build_delivery_receipt(plan_one, (("p", "succeeded", ""),))
    receipt_two = build_delivery_receipt(plan_two, (("p", "succeeded", ""),))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_one, receipt_two))


def test_conflicting_target_raises_value_error():
    # Same product, current, affected and failures; only the rollback
    # target (the previous version) differs.
    plan_one = _plan(
        _manifest((0, "p", "1", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
    )
    plan_two = _plan(
        _manifest((0, "p", "0", PASSING_GATE)),
        _manifest((1, "p", "2", FAILING_GATE)),
    )
    receipt_one = build_delivery_receipt(plan_one, (("p", "failed", "x"),))
    receipt_two = build_delivery_receipt(plan_two, (("p", "failed", "x"),))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_one, receipt_two))


def test_conflicting_failures_raise_value_error():
    plan_one = _plan(_manifest((0, "p", "1", FAILING_GATE)))
    plan_two = _plan(_manifest((0, "p", "1", SECOND_FAILING_GATE)))
    receipt_one = build_delivery_receipt(plan_one, (("p", "blocked", "x"),))
    receipt_two = build_delivery_receipt(plan_two, (("p", "blocked", "x"),))
    with pytest.raises(ValueError):
        merge_delivery_receipts((receipt_one, receipt_two))


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_compact_json_without_whitespace_or_trailing_newline():
    plan = _mixed_plan()
    receipt = build_delivery_receipt(
        plan, _mixed_results(reason_b="boom", reason_c="held"))
    text = merge_delivery_receipts((receipt, receipt))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_unicode_reason_is_not_ascii_escaped():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    receipt = build_delivery_receipt(plan, (("p", "failed", "部署失败"),))
    text = merge_delivery_receipts((receipt,))
    assert "部署失败" in text
    assert "\\u" not in text
    json.loads(text)


def test_metrics_keep_six_decimals_and_zero_formatting():
    report = (
        '{"windows":[[0,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,9]]]}'
    )
    gate = gate_report(report, (1, 1, 1, 10))
    plan = _plan(_manifest((0, "p", "1", gate)))
    receipt = build_delivery_receipt(plan, (("p", "blocked", "x"),))
    text = merge_delivery_receipts((receipt,))
    assert "[0.000000,0.000000,0.000000,0.000000,9]" in text
    assert "-0.000000" not in text


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    plan = _plan(_manifest((batch, "p", "1", FAILING_GATE)))
    receipt = build_delivery_receipt(plan, (("p", "blocked", "x"),))
    text = merge_delivery_receipts((receipt,))
    assert str(batch) in text
    assert json.loads(text)["products"][0][4] == [batch]


def test_deterministic_and_does_not_modify_inputs():
    plan = _mixed_plan()
    first = build_delivery_receipt(plan, _mixed_results())
    second = build_delivery_receipt(
        plan, _mixed_results(status_b="succeeded", reason_b=""))
    receipts = (first, second)
    snapshot = tuple(receipts)
    assert merge_delivery_receipts(receipts) == merge_delivery_receipts(
        receipts)
    assert receipts == snapshot


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_receipts", [None, 1, 1.5, b"", [], {}, "x"])
def test_non_tuple_receipts_raises_type_error(bad_receipts):
    with pytest.raises(TypeError):
        merge_delivery_receipts(bad_receipts)


@pytest.mark.parametrize("bad_item", [None, 1, 1.5, b"", [], {}, ()])
def test_non_str_receipt_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        merge_delivery_receipts((EMPTY_RECEIPT, bad_item))


@pytest.mark.parametrize("bad_receipt", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"products":[],"ready":true}',              # merged shape
    '{"operations":[],"releasable":true}',       # plan shape
    '{"entries":[],"releasable":true}',          # manifest shape
    '{"receipts":[]}',                           # missing ready
    '{"receipts":null,"ready":true}',
    '{"receipts":{},"ready":true}',
    '{"ready":true,"receipts":[]}',              # wrong key order
    '{"receipts":[],"ready":1}',
    '{"receipts":[],"ready":true,"x":1}',        # extra key
    '{"receipts":[],"ready":false}',             # inconsistent AND
    '{"receipts":[],"ready":true} ',             # trailing whitespace
    ' {"receipts":[],"ready":true}',             # leading whitespace
    '{"receipts":[[]],"ready":false}',           # empty receipt row
    '{"receipts":[["p","publish","1","1",[],"succeeded","",[]],'
    '["a","publish","1","1",[],"succeeded","",[]]],"ready":true}',  # unsorted
    '{"receipts":[["p","publish","1","1",[],"succeeded","",[]],'
    '["p","publish","1","1",[],"succeeded","",[]]],"ready":true}',  # duplicate
    '{"receipts":[["p","ship","1","1",[],"succeeded","",[]]],'
    '"ready":false}',                            # bad action
    '{"receipts":[["p","publish","1","1",[],"done","",[]]],'
    '"ready":false}',                            # bad status
    '{"receipts":[["p","publish","1","1",[],"blocked","x",[]]],'
    '"ready":false}',                            # publish blocked
    '{"receipts":[["p","block","1",null,[0],'
    '[[0,"1",0,0,0,0,0,null]],"succeeded","",'
    '[[0,"1",0,0,0,0,0,null]]]],"ready":false}',  # block succeeded
    '{"receipts":[["p","publish","1","1",[],"succeeded","oops",[]]],'
    '"ready":true}',                             # succeeded with reason
    '{"receipts":[["p","publish","1","1",[],"failed","",[]]],'
    '"ready":false}',                            # failed without reason
    '{"receipts":[["p","publish","1","1",[],"succeeded","",[]]],'
    '"ready":false}',                            # ready flag too strict
    '{"receipts":[["p","publish","1","1",[],"failed","x",[]]],'
    '"ready":true}',                             # ready flag too lax
    '{"receipts":[["p","publish","1","2",[],"succeeded","",[]]],'
    '"ready":true}',                             # publish retarget
    '{"receipts":[["p","rollback","1",null,[0],'
    '[[0,"1",0,0,0,0,0,null]],"failed","x",'
    '[[0,"1",0,0,0,0,0,null]]]],"ready":false}',  # rollback null target
    '{"receipts":[["p","block","1","2",[0],'
    '[[0,"2",0,0,0,0,0,null]],"blocked","x",'
    '[[0,"2",0,0,0,0,0,null]]]],"ready":false}',  # block with target
])
def test_malformed_receipt_raises_value_error(bad_receipt):
    with pytest.raises(ValueError):
        merge_delivery_receipts((bad_receipt,))


def test_nan_and_infinity_in_receipt_are_rejected():
    with pytest.raises(ValueError):
        merge_delivery_receipts(('{"receipts":[],"ready":NaN}',))
    with pytest.raises(ValueError):
        merge_delivery_receipts(('{"receipts":[],"ready":Infinity}',))


def test_accepts_every_receipt_output():
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
    receipt = build_delivery_receipt(plan, tuple(results))
    merged = json.loads(merge_delivery_receipts((receipt, receipt)))
    assert [row[0] for row in merged["products"]] == sorted(
        row[0] for row in rows)
    assert all(len(row[5]) == 2 for row in merged["products"])
