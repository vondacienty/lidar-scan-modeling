"""Tests for :func:`lidar_scan.merge_reconciliation_states`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import accumulate_reconciliation, merge_reconciliation_states
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
    assert (tiles_module.merge_reconciliation_states
            is merge_reconciliation_states)
    import lidar_scan
    assert "merge_reconciliation_states" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_states_gives_null_for_every_window():
    assert merge_reconciliation_states((), WINDOWS) == (
        '{"windows":[[0,0,0,9,9,null],[0,100,100,109,109,null]]}'
    )


def test_empty_windows_returns_empty_document():
    assert merge_reconciliation_states((), ()) == '{"windows":[]}'
    assert merge_reconciliation_states(('{"windows":[]}',), ()) == (
        '{"windows":[]}'
    )


def test_single_state_roundtrips_byte_identical():
    state = _state((( _tile(),),))
    assert merge_reconciliation_states((state,), WINDOWS) == state


def test_all_null_summaries_stay_null():
    state = '{"windows":[[0,0,0,9,9,null],[0,100,100,109,109,null]]}'
    assert merge_reconciliation_states((state, state), WINDOWS) == state


def test_null_summaries_do_not_contribute():
    populated = _state(((_tile(),),))
    all_null = '{"windows":[[0,0,0,9,9,null],[0,100,100,109,109,null]]}'
    assert merge_reconciliation_states((all_null, populated), WINDOWS) == (
        populated
    )
    assert merge_reconciliation_states((populated, all_null), WINDOWS) == (
        populated
    )


def test_extremes_and_exact_totals_merge_per_window():
    a = '{"windows":[[0,0,0,9,9,[-2.000000,3.000000,13000000000000,4,2]],' \
        '[0,100,100,109,109,null]]}'
    b = '{"windows":[[0,0,0,9,9,[-1.500000,5.000000,7000000000000,6,3]],' \
        '[0,100,100,109,109,[1.000000,2.000000,5000000000000,1,1]]]}'
    assert merge_reconciliation_states((a, b), WINDOWS) == (
        '{"windows":[[0,0,0,9,9,[-2.000000,5.000000,20000000000000,10,5]],'
        '[0,100,100,109,109,[1.000000,2.000000,5000000000000,1,1]]]}'
    )


def test_result_matches_accumulating_batches():
    first = ((_tile(tx=0, err_zmin=-2.5, err_zmax=0.25, err_count=-3),),)
    second = ((_tile(tx=1, ix0=10, ix1=19, err_zmin=1.5,
                     err_zmax=3.25, err_count=7),),)
    whole = ((first[0][0], second[0][0]),)
    windows = ((0, 0, 0, 19, 9), (0, 100, 100, 109, 109))

    state_a = accumulate_reconciliation(None, first, windows)
    state_b = accumulate_reconciliation(None, second, windows)
    expected = accumulate_reconciliation(None, whole, windows)
    assert merge_reconciliation_states((state_a, state_b), windows) == expected


def test_order_and_grouping_do_not_change_result():
    batch_a = ((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    batch_c = ((_tile(err_zmin=0.5, err_zmax=4.5, err_count=3),),)
    windows = ((0, 0, 0, 9, 9),)
    states = [accumulate_reconciliation(None, b, windows)
              for b in (batch_a, batch_b, batch_c)]

    expected = '{"windows":[[0,0,0,9,9,[-3.000000,5.000000,' \
               '59500000000000,6,3]]]}'
    assert merge_reconciliation_states(tuple(states), windows) == expected
    assert merge_reconciliation_states(tuple(reversed(states)), windows) == (
        expected
    )
    grouped = merge_reconciliation_states((states[0],), windows)
    assert merge_reconciliation_states(
        (grouped, states[1], states[2]), windows) == expected


def test_inputs_are_not_modified():
    a = _state(((_tile(err_zmin=-1.0, err_zmax=2.0),),))
    b = _state(((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),))
    snapshot = (a, b, json.dumps(WINDOWS))
    merge_reconciliation_states((a, b), WINDOWS)
    assert (a, b, json.dumps(WINDOWS)) == snapshot


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_states", [[], None, 1, "x", {}])
def test_states_wrong_type_raises_type_error(bad_states):
    with pytest.raises(TypeError):
        merge_reconciliation_states(bad_states, ())


@pytest.mark.parametrize("bad_member", [None, 0, 1.5, b"", [], {}])
def test_state_member_wrong_type_raises_type_error(bad_member):
    with pytest.raises(TypeError):
        merge_reconciliation_states((bad_member,), ())


def test_windows_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        merge_reconciliation_states((), [])


def test_window_container_and_length_type_errors():
    with pytest.raises(TypeError):
        merge_reconciliation_states((), ([0, 0, 0, 9, 9],))
    with pytest.raises(TypeError):
        merge_reconciliation_states((), ((0, 0, 0, 9),))
    with pytest.raises(TypeError):
        merge_reconciliation_states((), ((0, 0, 0, 9, 9, 0),))


@pytest.mark.parametrize("bad_window", [
    (True, 0, 0, 9, 9),
    (0, 1.5, 0, 9, 9),
    (0, 0, "0", 9, 9),
    (0, 0, 0, 9.0, 9),
    (0, 0, 0, 9, None),
])
def test_window_field_types_raise_type_error(bad_window):
    with pytest.raises(TypeError):
        merge_reconciliation_states((), (bad_window,))


def test_negative_level_raises_value_error():
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((-1, 0, 0, 9, 9),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((0, 9, 0, 0, 9),))
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((0, 0, 9, 9, 0),))


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
        merge_reconciliation_states((bad_state,), windows)


def test_state_window_count_mismatch_raises_value_error():
    state = '{"windows":[[0,0,0,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        merge_reconciliation_states((state,), ((0, 0, 0, 9, 9),))


def test_state_window_keys_must_match_in_order():
    state = '{"windows":[[0,5,5,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        merge_reconciliation_states(
            (state,), ((0, 0, 0, 4, 4), (0, 5, 5, 9, 9)))
