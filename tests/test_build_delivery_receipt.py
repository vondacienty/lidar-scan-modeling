"""Tests for :func:`lidar_scan.build_delivery_receipt`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
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


def _changes(*manifests):
    return merge_delivery_manifests(tuple(manifests))


def _audit(*manifests):
    return audit_delivery_changes(_changes(*manifests))


def _plan(*manifests):
    return build_delivery_plan(_audit(*manifests))


def _mixed_plan():
    return _plan(
        _manifest((0, "pub", "1", PASSING_GATE)),
        _manifest((1, "rb", "1", PASSING_GATE),
                  (2, "rb", "2", FAILING_GATE)),
        _manifest((3, "blk", "1", FAILING_GATE)),
    )


_MIXED_RESULTS = (
    ("blk", "blocked", "never passed"),
    ("rb", "failed", "rollback failed"),
    ("pub", "succeeded", ""),
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

def test_empty_plan_is_ready():
    assert build_delivery_receipt(_plan(), ()) == (
        '{"receipts":[],"ready":true}'
    )


def test_document_key_order_and_receipt_shape():
    text = build_delivery_receipt(
        _plan(_manifest((0, "p", "1", PASSING_GATE))),
        (("p", "succeeded", ""),),
    )
    assert list(json.loads(text)) == ["receipts", "ready"]
    assert text == (
        '{"receipts":[["p","publish","1","1",[],"succeeded","",[]]],'
        '"ready":true}'
    )


def test_receipts_follow_plan_order_regardless_of_result_order():
    text = build_delivery_receipt(_mixed_plan(), _MIXED_RESULTS)
    assert [row[0] for row in json.loads(text)["receipts"]] == [
        "blk", "pub", "rb",
    ]


def test_receipt_fields_in_contract_order():
    text = build_delivery_receipt(_mixed_plan(), _MIXED_RESULTS)
    rows = {row[0]: row for row in json.loads(text)["receipts"]}
    assert len(rows["pub"]) == 8
    assert rows["pub"] == ["pub", "publish", "1", "1", [],
                           "succeeded", "", []]
    assert rows["rb"][:7] == ["rb", "rollback", "2", "1", [2],
                              "failed", "rollback failed"]
    assert rows["rb"][7] == [
        [2, "2", 0, 0, 0, 9, 9,
         [-1.0, 2.0, 1.581139, 1.0, 1]],
        [2, "2", 0, 100, 100, 109, 109, None],
    ]
    assert rows["blk"][:7] == ["blk", "block", "1", None, [3],
                               "blocked", "never passed"]
    assert [failure[0] for failure in rows["blk"][7]] == [3, 3]


def test_plan_fields_preserved_verbatim():
    plan = _mixed_plan()
    operations = {row[0]: row for row in json.loads(plan)["operations"]}
    text = build_delivery_receipt(plan, _MIXED_RESULTS)
    for receipt in json.loads(text)["receipts"]:
        operation = operations[receipt[0]]
        assert receipt[:5] == operation[:5]
        assert receipt[7] == operation[5]


# ---------------------------------------------------------------------------
# status and reason rules
# ---------------------------------------------------------------------------

def test_publish_and_rollback_admit_succeeded_or_failed():
    plan = _mixed_plan()
    text = build_delivery_receipt(plan, (
        ("pub", "failed", "boom"),
        ("rb", "succeeded", ""),
        ("blk", "blocked", "held"),
    ))
    rows = {row[0]: row for row in json.loads(text)["receipts"]}
    assert rows["pub"][5:7] == ["failed", "boom"]
    assert rows["rb"][5:7] == ["succeeded", ""]


@pytest.mark.parametrize("status", ["succeeded", "failed", "weird"])
def test_block_only_admits_blocked(status):
    with pytest.raises(ValueError):
        build_delivery_receipt(_mixed_plan(), (
            ("pub", "succeeded", ""),
            ("rb", "failed", "x"),
            ("blk", status, "x"),
        ))


@pytest.mark.parametrize("status", ["blocked", "weird"])
def test_publish_and_rollback_reject_other_statuses(status):
    plan = _mixed_plan()
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", status, "x"),
            ("rb", "failed", "x"),
            ("blk", "blocked", "x"),
        ))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "succeeded", ""),
            ("rb", status, "x"),
            ("blk", "blocked", "x"),
        ))


def test_succeeded_requires_empty_reason():
    plan = _mixed_plan()
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "succeeded", "note"),
            ("rb", "failed", "x"),
            ("blk", "blocked", "x"),
        ))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "succeeded", ""),
            ("rb", "succeeded", "note"),
            ("blk", "blocked", "x"),
        ))


def test_failed_and_blocked_require_nonempty_reason():
    plan = _mixed_plan()
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "failed", ""),
            ("rb", "failed", "x"),
            ("blk", "blocked", "x"),
        ))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "succeeded", ""),
            ("rb", "failed", ""),
            ("blk", "blocked", "x"),
        ))
    with pytest.raises(ValueError):
        build_delivery_receipt(plan, (
            ("pub", "succeeded", ""),
            ("rb", "failed", "x"),
            ("blk", "blocked", ""),
        ))


# ---------------------------------------------------------------------------
# product correspondence
# ---------------------------------------------------------------------------

def test_results_may_be_any_order_but_render_by_plan_order():
    plan = _mixed_plan()
    assert build_delivery_receipt(plan, _MIXED_RESULTS) == (
        build_delivery_receipt(plan, tuple(reversed(_MIXED_RESULTS)))
    )


def test_missing_product_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_receipt(_mixed_plan(), (
            ("pub", "succeeded", ""),
            ("rb", "failed", "x"),
        ))


def test_unknown_product_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_receipt(_mixed_plan(), _MIXED_RESULTS + (
            ("zzz", "succeeded", ""),
        ))


def test_duplicate_product_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_receipt(_mixed_plan(), (
            ("pub", "succeeded", ""),
            ("pub", "failed", "x"),
            ("rb", "failed", "x"),
            ("blk", "blocked", "x"),
        ))


def test_empty_results_for_nonempty_plan_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_receipt(_mixed_plan(), ())


# ---------------------------------------------------------------------------
# ready flag
# ---------------------------------------------------------------------------

def test_ready_true_only_when_all_publish_succeeded():
    plan = _plan(
        _manifest((0, "a", "1", PASSING_GATE)),
        _manifest((0, "b", "1", PASSING_GATE)),
    )
    text = build_delivery_receipt(plan, (
        ("a", "succeeded", ""),
        ("b", "succeeded", ""),
    ))
    assert text.endswith('"ready":true}')


def test_failed_publish_makes_ready_false():
    plan = _plan(_manifest((0, "a", "1", PASSING_GATE)))
    assert build_delivery_receipt(plan, (("a", "failed", "boom"),)).endswith(
        '"ready":false}')


def test_succeeded_rollback_makes_ready_false():
    assert build_delivery_receipt(_mixed_plan(), (
        ("pub", "succeeded", ""),
        ("rb", "succeeded", ""),
        ("blk", "blocked", "x"),
    )).endswith('"ready":false}')


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_compact_json_without_whitespace_or_trailing_newline():
    results = (
        ("blk", "blocked", "held"),
        ("rb", "failed", "rollback-failed"),
        ("pub", "succeeded", ""),
    )
    text = build_delivery_receipt(_mixed_plan(), results)
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_metrics_keep_six_decimals():
    text = build_delivery_receipt(_mixed_plan(), _MIXED_RESULTS)
    assert "[-1.000000,2.000000,1.581139,1.000000,1]" in text
    assert "-0.000000" not in text


def test_reasons_are_encoded_without_ascii_escaping():
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    text = build_delivery_receipt(plan, (("p", "failed", "原因"),))
    assert "原因" in text
    assert "\\u" not in text
    json.loads(text)


def test_deterministic_and_does_not_modify_inputs():
    plan = _mixed_plan()
    snapshot_plan = plan
    snapshot_results = _MIXED_RESULTS
    assert (build_delivery_receipt(plan, _MIXED_RESULTS)
            == build_delivery_receipt(plan, _MIXED_RESULTS))
    assert plan == snapshot_plan
    assert _MIXED_RESULTS == snapshot_results


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_plan", [None, 1, 1.5, b"", [], {}, ()])
def test_non_str_plan_raises_type_error(bad_plan):
    with pytest.raises(TypeError):
        build_delivery_receipt(bad_plan, ())


@pytest.mark.parametrize("bad_results", [[], None, "x", 1, {}])
def test_non_tuple_results_raises_type_error(bad_results):
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(TypeError):
        build_delivery_receipt(plan, bad_results)


@pytest.mark.parametrize("bad_item", [
    ["p", "succeeded", ""],
    ("p", "succeeded"),
    ("p", "succeeded", "", "extra"),
    (1, "succeeded", ""),
    ("p", 1, ""),
    ("p", "succeeded", 1),
    ("p", "succeeded", None),
    (None, "succeeded", ""),
])
def test_bad_result_item_shape_or_type_raises_type_error(bad_item):
    plan = _plan(_manifest((0, "p", "1", PASSING_GATE)))
    with pytest.raises(TypeError):
        build_delivery_receipt(plan, (bad_item,))


@pytest.mark.parametrize("bad_plan", [
    "",
    "not json",
    "null",
    "[]",
    "{}",
    '{"products":[],"releasable":true}',                 # audit shape
    '{"entries":[],"releasable":true}',                  # manifest shape
    '{"receipts":[],"ready":true}',                      # receipt shape
    '{"operations":[]}',                                 # missing releasable
    '{"operations":[],"releasable":1}',
    '{"operations":{},"releasable":true}',
    '{"operations":null,"releasable":true}',
    '{"releasable":true,"operations":[]}',               # wrong key order
    '{"operations":[],"releasable":true,"x":1}',         # extra key
    '{"operations":[],"releasable":false}',              # inconsistent AND
    '{"operations":[],"releasable":true} ',               # trailing space
    ' {"operations":[],"releasable":true}',              # leading space
])
def test_malformed_plan_raises_value_error(bad_plan):
    with pytest.raises(ValueError):
        build_delivery_receipt(bad_plan, ())


def test_nan_and_infinity_are_rejected():
    with pytest.raises(ValueError):
        build_delivery_receipt(
            '{"operations":[],"releasable":NaN}', ())
    with pytest.raises(ValueError):
        build_delivery_receipt(
            '{"operations":[],"releasable":Infinity}', ())


def test_accepts_every_plan_output():
    plan = _plan(
        _manifest((2, "b", "1", FAILING_GATE),
                  (5, "a", "9", PASSING_GATE)),
        _manifest((0, "z", "2", PASSING_GATE),
                  (1, "a", "3", EMPTY_GATE)),
        _manifest((0, "a", "4", FAILING_GATE)),
    )
    results = tuple(
        (row[0], "succeeded", "") if row[1] == "publish"
        else (row[0], "failed" if row[1] == "rollback" else "blocked", "x")
        for row in json.loads(plan)["operations"]
    )
    json.loads(build_delivery_receipt(plan, results))
