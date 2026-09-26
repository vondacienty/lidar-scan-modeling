"""Tests for :func:`lidar_scan.replay_delivery_updates`."""

from __future__ import annotations

import copy
import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                         build_delivery_plan, build_delivery_receipt,
                         build_delivery_snapshot, merge_delivery_manifests,
                         merge_delivery_receipts, plan_delivery_updates,
                         replay_delivery_updates)
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


def _snapshot(items, receipt_results=()):
    changes = _changes(*(_manifest(item) for item in items))
    receipts = (
        _merged(changes, receipt_results) if receipt_results
        else EMPTY_RECEIPTS
    )
    return build_delivery_snapshot(changes, receipts)


def _ready_p(batches):
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    changes = _changes(*(_manifest(item) for item in items))
    receipts = _merged(changes, (("p", "succeeded", ""),))
    return build_delivery_snapshot(changes, receipts)


def _replay(before, after, ranges):
    plan = plan_delivery_updates(before, after, ranges)
    return plan, replay_delivery_updates(plan)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.replay_delivery_updates is replay_delivery_updates
    assert "replay_delivery_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_type_error_for_non_str():
    for value in (1, None, [], b"plan", {}):
        with pytest.raises(TypeError):
            replay_delivery_updates(value)


def test_value_errors():
    before = _ready_p(((0, "1"),))
    plan = plan_delivery_updates(before, before, ())
    with pytest.raises(ValueError):
        replay_delivery_updates("not json")
    with pytest.raises(ValueError):
        replay_delivery_updates(
            json.dumps(json.loads(plan), indent=2))
    with pytest.raises(ValueError):
        replay_delivery_updates(
            json.dumps(json.loads(plan), separators=(", ", ": ")))
    with pytest.raises(ValueError):
        replay_delivery_updates(plan + "\n")
    with pytest.raises(ValueError):
        replay_delivery_updates('{"operations":[],"changed":1}')
    with pytest.raises(ValueError):
        replay_delivery_updates('{"results":[],"changed":false}')
    with pytest.raises(ValueError):
        replay_delivery_updates(
            '{"operations":[],"changed":false,"extra":1}')


# ---------------------------------------------------------------------------
# empty plan and output encoding
# ---------------------------------------------------------------------------

def test_empty_plan():
    assert replay_delivery_updates(
        '{"operations":[],"changed":false}') == \
        '{"results":[],"changed":false}'


def test_compact_encoding():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    _plan_text, result = _replay(before, after, (("p", 0, 2),))
    assert ": " not in result
    assert ", " not in result
    assert not result.endswith("\n")
    document = json.loads(result)
    assert list(document) == ["results", "changed"]


def test_deterministic_and_does_not_modify_input():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 0),))
    snapshot = plan_text
    first = replay_delivery_updates(plan_text)
    second = replay_delivery_updates(plan_text)
    assert first == second
    assert plan_text == snapshot


# ---------------------------------------------------------------------------
# result rows
# ---------------------------------------------------------------------------

def test_results_follow_operations_order():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    ranges = (("q", 0, 0), ("p", 2, 2), ("p", 0, 0))
    plan_text, result = _replay(before, after, ranges)
    plan_doc = json.loads(plan_text)
    result_doc = json.loads(result)
    assert [row[:3] for row in result_doc["results"]] == \
        [row[:3] for row in plan_doc["operations"]]


def test_result_row_shape_target_and_action_count():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan_text, result = _replay(before, after, (("p", 0, 4),))
    plan_doc = json.loads(plan_text)
    row = plan_doc["operations"][0]
    out_row = json.loads(result)["results"][0]
    assert out_row == ["p", 0, 4, 2, row[7]]
    assert out_row[4] == [
        [[0, None, "1", True], [1, "1", "2", True],
         [2, "2", "3", True], [4, "3", "5", True]],
        [[3, 3]],
        ["publish", "5", "5", [["succeeded", ""]], "succeeded"],
        False,
    ]


def test_roundtrip_scenarios():
    scenarios = [
        (_ready_p(((0, "1"), (1, "2"))),
         _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5"))),
         (("p", 0, 4),)),
        (_snapshot(((0, "p", "1", PASSING_GATE),
                    (2, "p", "2", PASSING_GATE))),
         _snapshot(((0, "p", "1", PASSING_GATE),)),
         (("p", 0, 2),)),
        (_snapshot(((0, "m", "1", PASSING_GATE),)),
         _snapshot(((0, "m", "1", FAILING_GATE),)),
         (("m", 0, 0),)),
        (_snapshot(((0, "p", "1", PASSING_GATE),
                    (1, "p", "2", PASSING_GATE))),
         _snapshot(((0, "p", "9", PASSING_GATE),
                    (2, "p", "3", PASSING_GATE))),
         (("p", 0, 2),)),
        (_ready_p(((0, "1"),)),
         build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS),
         (("p", 0, 0),)),
        (_snapshot(((5, "p", "1", PASSING_GATE),)),
         _snapshot(((9, "p", "1", PASSING_GATE),)),
         (("p", 0, 2),)),
    ]
    for before, after, ranges in scenarios:
        plan_text = plan_delivery_updates(before, after, ranges)
        plan_doc = json.loads(plan_text)
        result_doc = json.loads(
            replay_delivery_updates(plan_text))
        for row, result_row in zip(
                plan_doc["operations"], result_doc["results"]):
            assert result_row[0] == row[0]
            assert result_row[1] == row[1]
            assert result_row[2] == row[2]
            assert result_row[3] == len(row[4])
            assert result_row[4] == row[7]
        assert result_doc["changed"] == plan_doc["changed"]


def test_absent_products_roundtrip():
    before = _ready_p(((0, "1"),))
    after = _snapshot(
        ((0, "q", "a", FAILING_GATE),),
        (("q", "blocked", "held"),))
    plan_text = plan_delivery_updates(
        before, after, (("q", 0, 0), ("z", 0, 0)))
    result = json.loads(replay_delivery_updates(plan_text))
    plan_doc = json.loads(plan_text)
    assert [row[4] for row in result["results"]] == \
        [row[7] for row in plan_doc["operations"]]
    assert result["changed"] is True


def test_changed_false_roundtrip():
    before = _ready_p(((0, "1"), (1, "2")))
    _plan_text, result = _replay(
        before, before, (("p", 0, 2), ("q", 5, 6)))
    assert json.loads(result)["changed"] is False


# ---------------------------------------------------------------------------
# malformed operations
# ---------------------------------------------------------------------------

def _tampered(plan_text, mutate):
    document = copy.deepcopy(json.loads(plan_text))
    mutate(document)
    return json.dumps(document, ensure_ascii=False,
                     separators=(",", ":"))


def test_bad_operation_fields_rejected():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 4),))
    row = lambda doc: doc["operations"][0]

    bad_plans = [
        lambda doc: row(doc).append(1),
        lambda doc: row(doc).__setitem__(0, 5),
        lambda doc: row(doc).__setitem__(0, "p x"),
        lambda doc: row(doc).__setitem__(1, True),
        lambda doc: row(doc).__setitem__(1, -1),
        lambda doc: row(doc).__setitem__(2, 0),
        lambda doc: row(doc).__setitem__(4, "add"),
    ]
    for mutate in bad_plans:
        with pytest.raises(ValueError):
            replay_delivery_updates(_tampered(plan_text, mutate))


def test_bad_actions_rejected():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 4),))
    row = lambda doc: doc["operations"][0]

    bad_plans = [
        lambda doc: row(doc)[4][0].__setitem__(0, "insert"),
        lambda doc: row(doc)[4][0].append([2, "2", "3", True]),
        lambda doc: row(doc)[4].append(
            ["remove", [4, "3", "5", True], [4, "3", "5", True]]),
        lambda doc: row(doc)[4].__setitem__(
            1, ["add", [2, "2", "3", True]]),
        lambda doc: row(doc)[4].__setitem__(
            0, ["add", [0, None, "1", True]]),
        lambda doc: row(doc)[4][0].__setitem__(
            1, [5, "3", "6", True]),
        lambda doc: row(doc)[4][0][1].__setitem__(3, 1),
        lambda doc: row(doc)[4].__setitem__(
            0, ["remove", [3, None, "x", True]]),
    ]
    for mutate in bad_plans:
        with pytest.raises(ValueError):
            replay_delivery_updates(_tampered(plan_text, mutate))


def test_replace_rules_rejected():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "9"),))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 0),))
    action = lambda doc: doc["operations"][0][4][0]

    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: action(doc)[1].__setitem__(2, "2")))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: action(doc)[2].__setitem__(0, 1)))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: doc["operations"][0][4].__setitem__(
                0, ["replace", [0, None, "1", True]])))


def test_target_mismatch_rejected():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 4),))
    row = lambda doc: doc["operations"][0]

    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text, lambda doc: row(doc)[7][1].append([9, 9])))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: row(doc)[7][0].__setitem__(
                0, [0, None, "1", False])))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: row(doc)[7][2].__setitem__(1, "zzz")))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: row(doc).__setitem__(
                7, [row(doc)[7][0], row(doc)[7][1],
                     row(doc)[7][2], True])))


def test_pair_first_items_must_match_base():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3"), (4, "5")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 4),))
    row = lambda doc: doc["operations"][0]

    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: row(doc)[5][0].__setitem__(1, "zzz")))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: row(doc)[6].__setitem__(0, False)))


def test_null_sides_must_keep_null_pairs():
    before = _ready_p(((0, "1"),))
    plan_text = plan_delivery_updates(
        before, before, (("z", 0, 0), ("p", 0, 0),))
    z_row = lambda doc: next(row for row in doc["operations"]
                             if row[0] == "z")

    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: z_row(doc)[5].__setitem__(
                0, ["publish", "1", "1", [["succeeded", ""]],
                     "succeeded"])))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: z_row(doc)[6].__setitem__(0, True)))

    after_empty = build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)
    remove_plan = plan_delivery_updates(
        before, after_empty, (("p", 0, 0),))
    p_row = lambda doc: doc["operations"][0]
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            remove_plan,
            lambda doc: p_row(doc)[5].__setitem__(
                1, ["publish", "1", "1", [["succeeded", ""]],
                     "succeeded"])))


def test_changed_flag_must_match():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 2),))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text, lambda doc: doc.__setitem__("changed", False)))

    equal_plan = plan_delivery_updates(before, before, (("p", 0, 2),))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            equal_plan, lambda doc: doc.__setitem__("changed", True)))


def test_operations_must_be_strictly_sorted():
    before = _ready_p(((0, "1"),))
    after = _ready_p(((0, "1"), (1, "2")))
    plan_text = plan_delivery_updates(
        before, after, (("q", 0, 0), ("p", 0, 0)))
    document = json.loads(plan_text)
    document["operations"].reverse()
    with pytest.raises(ValueError):
        replay_delivery_updates(json.dumps(
            document, separators=(",", ":")))


def test_side_versions_must_chain_previous():
    before = _ready_p(((0, "1"), (1, "2")))
    after = _ready_p(((0, "1"), (1, "2"), (2, "3")))
    plan_text = plan_delivery_updates(before, after, (("p", 0, 2),))

    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: doc["operations"][0][7][0][1].__setitem__(1, "x")))
    with pytest.raises(ValueError):
        replay_delivery_updates(_tampered(
            plan_text,
            lambda doc: doc["operations"][0][3][0][1].__setitem__(1, "x")))
