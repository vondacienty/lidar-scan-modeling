"""Tests for :func:`lidar_scan.finalize_reconciliation_state`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (accumulate_reconciliation,
                        finalize_reconciliation_state,
                        merge_reconciliation_states)
from lidar_scan import tiles as tiles_module


WINDOWS = ((0, 0, 0, 9, 9), (0, 100, 100, 109, 109))


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=9, iy1=9,
                  err_zmin=1.0, err_zmax=2.0, err_count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["err_zmin"],
            fields["err_zmax"], fields["err_count"])


def _state(assessment, windows=WINDOWS):
    return accumulate_reconciliation(None, assessment, windows)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.finalize_reconciliation_state
            is finalize_reconciliation_state)
    import lidar_scan
    assert "finalize_reconciliation_state" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_document():
    assert finalize_reconciliation_state('{"windows":[]}', ()) == (
        '{"windows":[]}'
    )


def test_null_summaries_report_null():
    state = '{"windows":[[0,0,0,9,9,null],[0,100,100,109,109,null]]}'
    assert finalize_reconciliation_state(state, WINDOWS) == state


def test_report_values_are_derived_per_window():
    state = (
        '{"windows":[[0,0,0,9,9,[-2.000000,5.000000,20000000000000,10,5]],'
        '[0,100,100,109,109,[1.000000,2.000000,5000000000000,1,1]]]}'
    )
    assert finalize_reconciliation_state(state, WINDOWS) == (
        '{"windows":[[0,0,0,9,9,'
        '[-2.000000,5.000000,1.414214,2.000000,5]],'
        '[0,100,100,109,109,'
        '[1.000000,2.000000,1.581139,1.000000,1]]]}'
    )


def test_finalize_matches_accumulated_state():
    assessment = ((_tile(err_zmin=1.0, err_zmax=2.0, err_count=1),
                   _tile(tx=1, err_zmin=-3.0, err_zmax=5.0, err_count=-2)),)
    windows = ((0, 0, 0, 9, 9),)
    state = _state(assessment, windows)
    assert state == (
        '{"windows":[[0,0,0,9,9,[-3.000000,5.000000,39000000000000,3,2]]]}'
    )
    assert finalize_reconciliation_state(state, windows) == (
        '{"windows":[[0,0,0,9,9,'
        '[-3.000000,5.000000,3.122499,1.500000,2]]]}'
    )


def test_finalize_of_merged_states():
    batch_a = ((_tile(tx=0, err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(tx=1, err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    batch_c = ((_tile(tx=2, err_zmin=0.5, err_zmax=4.5, err_count=3),),)
    windows = ((0, 0, 0, 9, 9),)
    states = [accumulate_reconciliation(None, b, windows)
              for b in (batch_a, batch_b, batch_c)]
    merged = merge_reconciliation_states(tuple(states), windows)
    assert finalize_reconciliation_state(merged, windows) == (
        finalize_reconciliation_state(
            _state((batch_a[0] + batch_b[0] + batch_c[0],), windows),
            windows)
    )


def test_negative_zero_is_normalized():
    state = '{"windows":[[0,0,0,9,9,[0.000000,0.000000,0,0,1]]]}'
    windows = ((0, 0, 0, 9, 9),)
    assert finalize_reconciliation_state(state, windows) == (
        '{"windows":[[0,0,0,9,9,[0.000000,0.000000,0.000000,0.000000,1]]]}'
    )


def test_inputs_are_not_modified():
    state = _state(((_tile(err_zmin=-1.0, err_zmax=2.0),),))
    snapshot = (state, json.dumps(WINDOWS))
    finalize_reconciliation_state(state, WINDOWS)
    assert (state, json.dumps(WINDOWS)) == snapshot


def test_repeated_calls_are_byte_identical():
    state = _state(((_tile(err_zmin=-1.0, err_zmax=2.0),),))
    assert (finalize_reconciliation_state(state, WINDOWS)
            == finalize_reconciliation_state(state, WINDOWS))


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_state", [None, 0, 1.5, b"", [], {}])
def test_state_wrong_type_raises_type_error(bad_state):
    with pytest.raises(TypeError):
        finalize_reconciliation_state(bad_state, ())


def test_windows_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        finalize_reconciliation_state('{"windows":[]}', [])


def test_window_container_and_length_type_errors():
    with pytest.raises(TypeError):
        finalize_reconciliation_state('{"windows":[]}', ([0, 0, 0, 9, 9],))
    with pytest.raises(TypeError):
        finalize_reconciliation_state('{"windows":[]}', ((0, 0, 0, 9),))
    with pytest.raises(TypeError):
        finalize_reconciliation_state('{"windows":[]}', ((0, 0, 0, 9, 9, 0),))


@pytest.mark.parametrize("bad_window", [
    (True, 0, 0, 9, 9),
    (0, 1.5, 0, 9, 9),
    (0, 0, "0", 9, 9),
    (0, 0, 0, 9.0, 9),
    (0, 0, 0, 9, None),
])
def test_window_field_types_raise_type_error(bad_window):
    with pytest.raises(TypeError):
        finalize_reconciliation_state('{"windows":[]}', (bad_window,))


def test_negative_level_raises_value_error():
    with pytest.raises(ValueError):
        finalize_reconciliation_state('{"windows":[]}', ((-1, 0, 0, 9, 9),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        finalize_reconciliation_state('{"windows":[]}', ((0, 9, 0, 0, 9),))
    with pytest.raises(ValueError):
        finalize_reconciliation_state('{"windows":[]}', ((0, 0, 9, 9, 0),))


@pytest.mark.parametrize("bad_state", [
    '{',
    'null',
    '[]',
    '{}',
    '{"levels":[]}',
    '{"windows":{}}',
    '{"windows":null}',
    '{"windows":[[0,0,0,9,9]]}',
    '{"windows":[[0,0,0,9,9,[1,2,3,4]]]}',
    '{"windows":[[0,0,0,9,9,{}]]}',
    '{"windows":[["0",0,0,9,9,null]]}',
    '{"windows":[[0,0,0,9,9,[1,2,0,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.0,2.000000,0,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0,0.5,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0,0,0]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,-1,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0,-1,1]]]}',
    '{"windows":[[0,0,0,9,9,[NaN,2.000000,0,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,Infinity,0,0,1]]]}',
    '{ "windows":[]}',
    '{"windows" :[]}',
    "{'windows':[]}",
    '{"windows":[],"extra":1}',
])
def test_noncanonical_state_raises_value_error(bad_state):
    windows = ((0, 0, 0, 9, 9),)
    with pytest.raises(ValueError):
        finalize_reconciliation_state(bad_state, windows)


def test_state_window_count_mismatch_raises_value_error():
    state = '{"windows":[[0,0,0,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        finalize_reconciliation_state(state, ((0, 0, 0, 9, 9),))


def test_state_window_keys_must_match_in_order():
    state = '{"windows":[[0,5,5,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        finalize_reconciliation_state(
            state, ((0, 0, 0, 4, 4), (0, 5, 5, 9, 9)))
