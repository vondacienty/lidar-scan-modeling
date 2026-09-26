"""Tests for :func:`lidar_scan.build_delivery_manifest`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import build_delivery_manifest, gate_report
from lidar_scan import tiles as tiles_module

PASS_EMPTY = '{"windows":[],"passed":true}'

PASS_ONE = (
    '{"windows":[[1,0,0,4,4,[0.000000,0.500000,0.250000,0.500000,3],true]]'
    ',"passed":true}'
)

FAIL_NULL = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true],'
    '[0,100,100,109,109,null,false]],"passed":false}'
)


def _gate(window_specs, passed):
    """Build a canonical gate report string from ``(bounds, flag)`` pairs."""
    windows = []
    for bounds, summary, flag in window_specs:
        summary_text = "null" if summary is None else (
            "[" + ",".join(summary) + "]"
        )
        bounds_text = ",".join(str(value) for value in bounds)
        windows.append(f"[{bounds_text},{summary_text},"
                       f"{'true' if flag else 'false'}]")
    return ('{"windows":[' + ",".join(windows) + '],"passed":'
            + ("true}" if passed else "false}"))


def _item(batch=1, product="prod", version="1.0.0", gate=PASS_EMPTY):
    return (batch, product, version, gate)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.build_delivery_manifest is build_delivery_manifest
    import lidar_scan
    assert "build_delivery_manifest" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_items_are_releasable():
    assert build_delivery_manifest(()) == (
        '{"entries":[],"releasable":true}'
    )


def test_output_shape_and_key_order():
    text = build_delivery_manifest((_item(gate=PASS_ONE),))
    assert not text.endswith("\n")
    document = json.loads(text)
    assert list(document) == ["entries", "releasable"]
    entry = document["entries"][0]
    assert len(entry) == 5
    assert entry == [1, "prod", "1.0.0", True, []]


def test_passing_entry_is_releasable():
    text = build_delivery_manifest((_item(gate=PASS_ONE),))
    assert json.loads(text)["releasable"] is True


def test_failing_entry_is_not_releasable():
    text = build_delivery_manifest((_item(gate=FAIL_NULL),))
    document = json.loads(text)
    assert document["entries"][0][3] is False
    assert document["releasable"] is False


def test_failures_collect_false_windows_in_order():
    specs = [
        ((0, 0, 0, 1, 1), None, False),
        ((0, 2, 2, 3, 3),
         ("1.000000", "2.000000", "1.000000", "1.000000", "2"), True),
        ((1, 4, 4, 5, 5),
         ("1.000000", "2.000000", "1.000000", "1.000000", "2"), False),
    ]
    gate = _gate(specs, False)
    text = build_delivery_manifest((_item(gate=gate),))
    failures = json.loads(text)["entries"][0][4]
    assert failures == [[0, 0, 0, 1, 1], [1, 4, 4, 5, 5]]


def test_failures_keep_only_first_six_false_windows():
    specs = []
    for index in range(9):
        # failing windows at 0,1,3,4,6,7,8; passing at 2,5
        passing = index in (2, 5)
        summary = None
        if passing:
            summary = ("1.000000", "2.000000", "1.000000", "1.000000", "2")
        specs.append(((0, index, index, index, index), summary, passing))
    gate = _gate(specs, False)
    failures = json.loads(build_delivery_manifest(
        (_item(gate=gate),)))["entries"][0][4]
    assert failures == [[0, i, i, i, i] for i in (0, 1, 3, 4, 6, 7)]
    assert len(failures) == 6


def test_entries_are_sorted_by_batch_then_product():
    items = (
        _item(batch=2, product="b", gate=PASS_EMPTY),
        _item(batch=1, product="z", gate=PASS_EMPTY),
        _item(batch=0, product="m", gate=PASS_ONE),
        _item(batch=1, product="a", gate=PASS_EMPTY),
    )
    keys = [(entry[0], entry[1])
            for entry in json.loads(build_delivery_manifest(items))["entries"]]
    assert keys == [(0, "m"), (1, "a"), (1, "z"), (2, "b")]


def test_releasable_is_logical_and_across_entries():
    items = (
        _item(batch=0, product="ok", gate=PASS_ONE),
        _item(batch=1, product="bad", gate=FAIL_NULL),
        _item(batch=2, product="fine", gate=PASS_EMPTY),
    )
    assert json.loads(build_delivery_manifest(items))["releasable"] is False


def test_all_passing_entries_are_releasable():
    items = (
        _item(batch=0, product="ok", gate=PASS_ONE),
        _item(batch=1, product="fine", gate=PASS_EMPTY),
    )
    assert json.loads(build_delivery_manifest(items))["releasable"] is True


def test_accepts_gate_report_output_end_to_end():
    finalized = (
        '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,null]]}'
    )
    gate = gate_report(finalized, (5, 5, 5, 1))
    text = build_delivery_manifest((_item(gate=gate),))
    entry = json.loads(text)["entries"][0]
    assert entry[3] is False
    assert entry[4] == [[0, 100, 100, 109, 109]]


def test_no_whitespace_or_non_finite_literals():
    text = build_delivery_manifest((_item(gate=FAIL_NULL),))
    assert " " not in text
    assert "\n" not in text and "\t" not in text
    assert "NaN" not in text and "Infinity" not in text


def test_inputs_are_not_modified():
    items = (
        _item(batch=2, product="b", gate=PASS_EMPTY),
        _item(batch=1, product="a", gate=FAIL_NULL),
    )
    snapshot = tuple((batch, product, version, gate)
                     for batch, product, version, gate in items)
    build_delivery_manifest(items)
    assert items == snapshot


def test_reordering_items_is_byte_identical():
    items = (
        _item(batch=2, product="b", gate=PASS_EMPTY),
        _item(batch=1, product="a", gate=FAIL_NULL),
        _item(batch=0, product="c", version="2.0", gate=PASS_ONE),
    )
    assert build_delivery_manifest(items) == build_delivery_manifest(
        (items[2], items[0], items[1])
    )


def test_repeated_calls_are_byte_identical():
    items = (_item(gate=FAIL_NULL),)
    assert build_delivery_manifest(items) == build_delivery_manifest(items)


def test_legal_identifier_characters_are_accepted():
    text = build_delivery_manifest(
        (_item(batch=0, product="A_z.1-9", version="v.2_0-1"),)
    )
    document = json.loads(text)
    assert document["entries"][0][1] == "A_z.1-9"
    assert document["entries"][0][2] == "v.2_0-1"


def test_batch_zero_and_large_integers_are_decimal():
    text = build_delivery_manifest(
        (_item(batch=12345678901234567890, gate=PASS_EMPTY),)
    )
    assert "12345678901234567890" in text
    assert "12345678901234567890.0" not in text
    assert json.loads(text)["entries"][0][0] == 12345678901234567890


# ---------------------------------------------------------------------------
# argument validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_items", [None, [], {}, set(), ""])
def test_items_wrong_type_raises_type_error(bad_items):
    with pytest.raises(TypeError):
        build_delivery_manifest(bad_items)


@pytest.mark.parametrize("bad_item", [
    [1, "p", "1", PASS_EMPTY],
    (1, "p", "1"),
    (1, "p", "1", PASS_EMPTY, "extra"),
    "1p1",
    None,
])
def test_item_shape_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


@pytest.mark.parametrize("bad_batch", [True, False, 1.0, "1", None, 1 + 0j])
def test_batch_type_raises_type_error(bad_batch):
    with pytest.raises(TypeError):
        build_delivery_manifest((_item(batch=bad_batch),))


@pytest.mark.parametrize("bad_product", [b"p", 1, 1.0, None, ["p"]])
def test_product_type_raises_type_error(bad_product):
    with pytest.raises(TypeError):
        build_delivery_manifest((_item(product=bad_product),))


@pytest.mark.parametrize("bad_version", [b"1", 1, 1.0, None, ["1"]])
def test_version_type_raises_type_error(bad_version):
    with pytest.raises(TypeError):
        build_delivery_manifest((_item(version=bad_version),))


@pytest.mark.parametrize("bad_gate", [None, 0, 1.5, b"", [], {}])
def test_gate_type_raises_type_error(bad_gate):
    with pytest.raises(TypeError):
        build_delivery_manifest((_item(gate=bad_gate),))


# ---------------------------------------------------------------------------
# argument validation: values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_batch", [-1, -100])
def test_negative_batch_raises_value_error(bad_batch):
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(batch=bad_batch),))


@pytest.mark.parametrize("bad_identifier", [
    "", " ", "a b", "a/b", "a\\b", "a:b", "a@b", "pét", "naïve",
    "p#1", "a,b", "x(y)", "日本",

])
def test_illegal_identifier_raises_value_error(bad_identifier):
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(product=bad_identifier),))
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(version=bad_identifier),))


def test_duplicate_batch_product_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_manifest(
            (_item(batch=1, product="p", version="1", gate=PASS_EMPTY),
             _item(batch=1, product="p", version="2", gate=PASS_ONE))
        )


def test_same_product_different_batch_is_allowed():
    text = build_delivery_manifest(
        (_item(batch=0, product="p", gate=PASS_EMPTY),
         _item(batch=1, product="p", gate=PASS_EMPTY))
    )
    assert json.loads(text)["releasable"] is True


# ---------------------------------------------------------------------------
# gate validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_gate", [
    '{',
    'null',
    '[]',
    '{}',
    '{"windows":[]}',
    '{"passed":true}',
    '{"windows":[],"passed":true,"extra":1}',
    '{"passed":true,"windows":[]}',
    '{"windows":{},"passed":true}',
    '{"windows":null,"passed":true}',
    '{"windows":[],"passed":"true"}',
    '{"windows":[[]],"passed":false}',
    '{"windows":[[0,0,0,9,9,null]],"passed":false}',
    '{"windows":[[0,0,0,9,9,null,true,false]],"passed":false}',
    '{"windows":[["0",0,0,9,9,null,true]],"passed":true}',
    '{"windows":[[true,0,0,9,9,null,true]],"passed":true}',
    '{"windows":[[-1,0,0,9,9,null,true]],"passed":true}',
    '{"windows":[[0,9,0,0,9,null,false]],"passed":false}',
    '{"windows":[[0,0,0,9,9,null,1]],"passed":true}',
    '{"windows":[[0,0,0,9,9,null,true]],"passed":true}',
    '{"windows":[[0,0,0,9,9,[1,2,1,1,1],true]],"passed":true}',
    '{"windows":[[0,0,0,9,9,[NaN,2,1,1,1],true]],"passed":true}',
    '{"windows" :[],"passed":true}',
    '{ "windows":[],"passed":true}',
    "{'windows':[],'passed':true}",
])
def test_bad_gate_raises_value_error(bad_gate):
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(gate=bad_gate),))


@pytest.mark.parametrize("bad_gate", [
    # empty window set must report passed=true
    '{"windows":[],"passed":false}',
    # all window flags true but top-level false
    ('{"windows":[[0,0,0,9,9,null,true]],"passed":false}'),
    # a failing window but top-level true
    ('{"windows":[[0,0,0,9,9,null,false]],"passed":true}'),
    # mixed windows disagreeing with the top level
    ('{"windows":[[0,0,0,9,9,null,false],'
     '[1,0,0,4,4,[0.000000,0.500000,0.250000,0.500000,3],true]],'
     '"passed":true}'),
])
def test_inconsistent_top_level_passed_raises_value_error(bad_gate):
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(gate=bad_gate),))


def test_pre_gate_finalization_report_is_rejected():
    # The six-value finalization format (no per-window flag / no top-level
    # passed) is not a gate_report output.
    finalized = '{"windows":[[0,0,0,9,9,null]]}'
    with pytest.raises(ValueError):
        build_delivery_manifest((_item(gate=finalized),))


def test_error_in_later_item_still_raises():
    items = (_item(batch=0, product="ok"),
             _item(batch=1, product="bad/", gate=PASS_EMPTY))
    with pytest.raises(ValueError):
        build_delivery_manifest(items)
