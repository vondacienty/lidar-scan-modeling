"""Tests for :func:`lidar_scan.build_delivery_manifest`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import build_delivery_manifest, gate_report
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


def _failing_gate(levels):
    """Build a gate report in which every (null-summary) window fails."""
    windows = ",".join(f"[{level},0,0,0,0,null]" for level in levels)
    return gate_report('{"windows":[' + windows + "]}", (1, 1, 1, 1))


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

def test_empty_items_manifest():
    assert build_delivery_manifest(()) == '{"entries":[],"releasable":true}'


def test_entry_shape_and_key_order():
    text = build_delivery_manifest(((0, "prod", "1.0.0", EMPTY_GATE),))
    assert list(json.loads(text)) == ["entries", "releasable"]
    assert text == (
        '{"entries":[[0,"prod","1.0.0",true,[]]],"releasable":true}'
    )


def test_entries_sorted_by_batch_then_product():
    items = (
        (2, "b", "1", EMPTY_GATE),
        (0, "z", "1", EMPTY_GATE),
        (1, "a", "1", EMPTY_GATE),
        (0, "a", "1", EMPTY_GATE),
    )
    doc = json.loads(build_delivery_manifest(items))
    assert [(e[0], e[1]) for e in doc["entries"]] == [
        (0, "a"), (0, "z"), (1, "a"), (2, "b")
    ]


def test_reordering_input_is_byte_identical():
    items = (
        (2, "b", "1", FAILING_GATE),
        (0, "z", "2", PASSING_GATE),
        (1, "a", "3", EMPTY_GATE),
        (0, "a", "4", FAILING_GATE),
    )
    assert build_delivery_manifest(items) == build_delivery_manifest(
        tuple(reversed(items)))


def test_failing_gate_entry_payload():
    text = build_delivery_manifest(((7, "prod", "v2", FAILING_GATE),))
    assert text == (
        '{"entries":[[7,"prod","v2",false,'
        '[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,null]]]],"releasable":false}'
    )
    doc = json.loads(text)
    entry = doc["entries"][0]
    assert entry[:4] == [7, "prod", "v2", False]
    assert len(entry[4][0]) == 6


def test_passing_gate_has_empty_failures():
    doc = json.loads(
        build_delivery_manifest(((0, "p", "1", PASSING_GATE),)))
    assert doc["entries"][0][3] is True
    assert doc["entries"][0][4] == []
    assert doc["releasable"] is True


def test_failures_keep_gate_window_order():
    report = (
        '{"windows":[[0,0,0,0,0,null],'
        '[1,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,2]],'
        '[2,0,0,0,0,null]]}'
    )
    gate = gate_report(report, (1, 1, 1, 2))
    doc = json.loads(build_delivery_manifest(((0, "p", "1", gate),)))
    assert [window[0] for window in doc["entries"][0][4]] == [0, 2]


def test_only_first_six_failing_windows_are_listed():
    gate = _failing_gate(range(7))
    doc = json.loads(build_delivery_manifest(((0, "p", "1", gate),)))
    failures = doc["entries"][0][4]
    assert len(failures) == 6
    assert [window[0] for window in failures] == [0, 1, 2, 3, 4, 5]


def test_six_failures_does_not_skip_seventh_passing_window():
    report = (
        '{"windows":['
        + ",".join(f"[{i},0,0,0,0,null]" for i in range(6))
        + ',[6,0,0,0,0,[0.000000,0.000000,0.000000,0.000000,2]]'
        + ',[7,0,0,0,0,null]]}'
    )
    gate = gate_report(report, (1, 1, 1, 2))
    doc = json.loads(build_delivery_manifest(((0, "p", "1", gate),)))
    assert [window[0] for window in doc["entries"][0][4]] == [0, 1, 2, 3, 4, 5]


def test_releasable_is_logical_and_of_entries():
    doc = json.loads(build_delivery_manifest((
        (0, "a", "1", PASSING_GATE),
        (1, "b", "1", EMPTY_GATE),
        (2, "c", "1", FAILING_GATE),
    )))
    assert doc["releasable"] is False
    doc = json.loads(build_delivery_manifest((
        (0, "a", "1", PASSING_GATE),
        (1, "b", "1", EMPTY_GATE),
    )))
    assert doc["releasable"] is True


def test_allowed_identifier_characters():
    text = build_delivery_manifest((
        (0, "aZ09._-", "0.1_2-3.x", EMPTY_GATE),
    ))
    doc = json.loads(text)
    assert doc["entries"][0][1] == "aZ09._-"
    assert doc["entries"][0][2] == "0.1_2-3.x"


def test_large_batch_integer_is_decimal():
    batch = 10 ** 40
    text = build_delivery_manifest(((batch, "p", "1", EMPTY_GATE),))
    assert "[" + str(batch) + "," in text
    assert json.loads(text)["entries"][0][0] == batch


def test_no_whitespace_or_trailing_newline():
    text = build_delivery_manifest((
        (0, "p", "1", FAILING_GATE),
        (1, "q", "2", PASSING_GATE),
    ))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text
    json.loads(text)


def test_repeated_calls_are_byte_identical():
    items = ((0, "p", "1", FAILING_GATE), (1, "q", "2", PASSING_GATE))
    assert build_delivery_manifest(items) == build_delivery_manifest(items)


def test_inputs_are_not_modified():
    items = ((2, "p", "1", FAILING_GATE), (0, "q", "2", PASSING_GATE))
    snapshot = tuple((batch, product, version, gate)
                     for batch, product, version, gate in items)
    build_delivery_manifest(items)
    assert items == snapshot


# ---------------------------------------------------------------------------
# argument validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_items", [
    None, [], {}, "x", 1, 1.5,
])
def test_items_wrong_type_raises_type_error(bad_items):
    with pytest.raises(TypeError):
        build_delivery_manifest(bad_items)


@pytest.mark.parametrize("bad_item", [
    [0, "p", "1", EMPTY_GATE],
    (0, "p", "1"),
    (0, "p", "1", EMPTY_GATE, 9),
    (),
])
def test_item_shape_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


@pytest.mark.parametrize("bad_item", [
    (True, "p", "1", EMPTY_GATE),
    ("0", "p", "1", EMPTY_GATE),
    (1.0, "p", "1", EMPTY_GATE),
    (None, "p", "1", EMPTY_GATE),
])
def test_batch_type_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


@pytest.mark.parametrize("bad_item", [
    (0, 1, "1", EMPTY_GATE),
    (0, None, "1", EMPTY_GATE),
    (0, b"p", "1", EMPTY_GATE),
])
def test_product_type_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


@pytest.mark.parametrize("bad_item", [
    (0, "p", 1, EMPTY_GATE),
    (0, "p", None, EMPTY_GATE),
])
def test_version_type_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


@pytest.mark.parametrize("bad_item", [
    (0, "p", "1", None),
    (0, "p", "1", 1),
    (0, "p", "1", b""),
    (0, "p", "1", []),
])
def test_gate_type_raises_type_error(bad_item):
    with pytest.raises(TypeError):
        build_delivery_manifest((bad_item,))


# ---------------------------------------------------------------------------
# argument validation: values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [-1, -10 ** 6])
def test_negative_batch_raises_value_error(batch):
    with pytest.raises(ValueError):
        build_delivery_manifest(((batch, "p", "1", EMPTY_GATE),))


@pytest.mark.parametrize("name", [
    "", "p/x", "p x", "p:q", "p#1", "p.q@x", "café", "p™", "p\nq",
])
def test_bad_product_raises_value_error(name):
    with pytest.raises(ValueError):
        build_delivery_manifest(((0, name, "1", EMPTY_GATE),))


@pytest.mark.parametrize("name", [
    "", "1/2", "1 2", "v1+2", "v1β",
])
def test_bad_version_raises_value_error(name):
    with pytest.raises(ValueError):
        build_delivery_manifest(((0, "p", name, EMPTY_GATE),))


def test_duplicate_batch_product_raises_value_error():
    with pytest.raises(ValueError):
        build_delivery_manifest((
            (0, "p", "1", EMPTY_GATE),
            (0, "p", "2", PASSING_GATE),
        ))


def test_same_product_different_batch_is_allowed():
    doc = json.loads(build_delivery_manifest((
        (0, "p", "1", EMPTY_GATE),
        (1, "p", "1", EMPTY_GATE),
    )))
    assert len(doc["entries"]) == 2


@pytest.mark.parametrize("bad_gate", [
    '{"windows":[]}',                                  # finalize format
    '{"passed":true,"windows":[]}',                    # wrong key order
    '{"windows":[],"passed":false}',                   # inconsistent AND
    '{"windows":[],"passed":true} ',                   # trailing whitespace
    '{"windows":[],"passed":true,"x":1}',              # extra key
    '{"windows":[],"passed":1}',                       # non-bool passed
    '{ "windows":[],"passed":true}',                   # leading whitespace
    '',
    'not json',
    'null',
    '[]',
    '{}',
    '{"windows":null,"passed":true}',
    '{"windows":[[]],"passed":true}',
    '{"windows":[[0,0,0,9,9,null]],"passed":false}',   # six-value window
    '{"windows":[[0,0,0,9,9,null,false],'
    '[0,100,100,109,109,null,true]],"passed":true}',   # flag/AND mismatch
    '{"windows":[[0,0,0,9,9,null,true]],"passed":true}',  # null marked pass
])
def test_bad_gate_raises_value_error(bad_gate):
    with pytest.raises(ValueError):
        build_delivery_manifest(((0, "p", "1", bad_gate),))


def test_gate_with_all_passing_windows_requires_top_level_true():
    gate = (
        '{"windows":[[0,0,0,9,9,'
        '[-1.000000,2.000000,1.581139,1.000000,1],true]],"passed":false}'
    )
    with pytest.raises(ValueError):
        build_delivery_manifest(((0, "p", "1", gate),))


def test_duplicate_detected_before_gate_validation():
    # A duplicate (batch, product) is rejected even though the second gate is
    # also malformed.
    with pytest.raises(ValueError):
        build_delivery_manifest((
            (0, "p", "1", EMPTY_GATE),
            (0, "p", "2", "not json"),
        ))
