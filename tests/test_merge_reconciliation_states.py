"""Tests for :func:`lidar_scan.merge_reconciliation_states`."""

from __future__ import annotations

import pytest

from lidar_scan import accumulate_reconciliation, merge_reconciliation_states
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=9, iy1=9,
                  err_zmin=1.0, err_zmax=2.0, err_count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["err_zmin"],
            fields["err_zmax"], fields["err_count"])


WINDOWS = ((0, 0, 0, 9, 9), (0, 100, 100, 109, 109))


def _state(assessment):
    return accumulate_reconciliation(None, assessment, WINDOWS)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.merge_reconciliation_states
            is merge_reconciliation_states)
    import lidar_scan
    assert "merge_reconciliation_states" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# merging basics
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_document():
    assert merge_reconciliation_states((), ()) == '{"windows":[]}'
    state = accumulate_reconciliation(None, (), ())
    assert merge_reconciliation_states((state,), ()) == '{"windows":[]}'


def test_empty_states_gives_null_for_every_window():
    assert merge_reconciliation_states((), WINDOWS) == (
        '{"windows":[[0,0,0,9,9,null],[0,100,100,109,109,null]]}'
    )


def test_all_null_states_merge_to_null():
    null_state = _state(((),))
    assert merge_reconciliation_states((null_state,), WINDOWS) == null_state
    assert merge_reconciliation_states(
        (null_state, null_state), WINDOWS) == null_state


def test_null_states_do_not_contribute():
    null_state = _state(((),))
    real = _state(((_tile(err_zmin=-1.0, err_zmax=4.0, err_count=2),),))
    assert merge_reconciliation_states((null_state, real), WINDOWS) == real
    assert merge_reconciliation_states((real, null_state), WINDOWS) == real


def test_single_state_round_trips():
    state = _state(((_tile(err_zmin=-2.5, err_zmax=0.25, err_count=-3),
                     _tile(tx=1, ix0=10, ix1=19, err_zmin=1.5,
                           err_zmax=3.25, err_count=7)),))
    assert merge_reconciliation_states((state,), WINDOWS) == state


def test_merge_extremes_and_exact_integer_totals():
    batch_a = ((_tile(ty=0, err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(ty=1, err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    batch_c = ((_tile(ty=2, err_zmin=0.5, err_zmax=4.5, err_count=3),),)
    states = tuple(_state(batch) for batch in (batch_a, batch_b, batch_c))
    whole = _state(((batch_a[0][0], batch_b[0][0], batch_c[0][0]),))
    assert merge_reconciliation_states(states, WINDOWS) == whole
    assert merge_reconciliation_states(states, WINDOWS) == (
        '{"windows":[[0,0,0,9,9,[-3.000000,5.000000,'
        '59500000000000,6,3]],[0,100,100,109,109,null]]}'
    )


def test_merge_partial_windows():
    first_tile = _tile(tx=0, err_zmin=-1.0, err_zmax=1.0, err_count=2)
    second_w0 = _tile(tx=1, err_zmin=0.0, err_zmax=1.0, err_count=1)
    second_w1 = _tile(tx=2, ix0=100, iy0=100, ix1=109, iy1=109,
                      err_zmin=3.0, err_zmax=4.0, err_count=5)
    only_first = _state(((first_tile,),))
    only_second = _state(((second_w0, second_w1),))
    expected = _state(((first_tile, second_w0, second_w1),))
    assert merge_reconciliation_states(
        (only_first, only_second), WINDOWS) == expected


# ---------------------------------------------------------------------------
# order / grouping independence and purity
# ---------------------------------------------------------------------------

def test_result_independent_of_state_order():
    batch_a = ((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    state_a = _state(batch_a)
    state_b = _state(batch_b)
    assert (merge_reconciliation_states((state_a, state_b), WINDOWS)
            == merge_reconciliation_states((state_b, state_a), WINDOWS))


def test_result_independent_of_grouping():
    tiles_ = (
        _tile(tx=0, err_zmin=-2.5, err_zmax=0.25, err_count=-3),
        _tile(tx=1, ix0=10, ix1=19, err_zmin=1.5, err_zmax=3.25,
              err_count=7),
        _tile(tx=2, ix0=20, ix1=29, err_zmin=-0.5, err_zmax=2.0,
              err_count=4),
    )
    whole = _state((tiles_,))
    g1 = _state(((tiles_[0], tiles_[1]),))
    g2 = _state(((tiles_[2],),))
    g3 = _state(((tiles_[0],),))
    g4 = _state(((tiles_[1], tiles_[2]),))
    assert merge_reconciliation_states((g1, g2), WINDOWS) == whole
    assert merge_reconciliation_states((g3, g4), WINDOWS) == whole


def test_merge_matches_accumulation_chaining():
    batch_a = ((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    state_a = _state(batch_a)
    state_b = _state(batch_b)
    chained = accumulate_reconciliation(state_a, batch_b, WINDOWS)
    assert merge_reconciliation_states(
        (state_a, state_b), WINDOWS) == chained


def test_inputs_are_not_modified():
    state_a = _state(((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=1),),))
    state_b = _state(((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),))
    snapshot = (state_a, state_b)
    merge_reconciliation_states((state_a, state_b), WINDOWS)
    merge_reconciliation_states((state_a, state_b), WINDOWS)
    assert (state_a, state_b) == snapshot


def test_repeated_merge_is_stable():
    state = _state(((_tile(err_zmin=-1.0, err_zmax=1.0, err_count=2),),))
    once = merge_reconciliation_states((state, state), WINDOWS)
    assert merge_reconciliation_states((once,), WINDOWS) == once
    assert merge_reconciliation_states((once, state), WINDOWS) == (
        merge_reconciliation_states((state, once), WINDOWS))


# ---------------------------------------------------------------------------
# state validation
# ---------------------------------------------------------------------------

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
    with pytest.raises(ValueError):
        merge_reconciliation_states((bad_state,), WINDOWS)


def test_one_bad_state_among_many_raises_value_error():
    good = _state(((),))
    with pytest.raises(ValueError):
        merge_reconciliation_states((good, '{'), WINDOWS)


def test_empty_state_with_nonempty_windows_raises_value_error():
    with pytest.raises(ValueError):
        merge_reconciliation_states(('{"windows":[]}',), WINDOWS)


def test_state_window_count_mismatch_raises_value_error():
    state = '{"windows":[[0,0,0,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        merge_reconciliation_states((state,), WINDOWS)


def test_state_window_keys_must_match_in_order():
    state = '{"windows":[[0,5,5,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        merge_reconciliation_states(
            (state,), ((0, 0, 0, 4, 4), (0, 5, 5, 9, 9)))
    good = '{"windows":[[0,0,0,4,4,null],[0,5,5,9,9,null]]}'
    assert merge_reconciliation_states(
        (good,), ((0, 0, 0, 4, 4), (0, 5, 5, 9, 9))) == good


def test_state_with_wrong_key_value_raises_value_error():
    state = '{"windows":[[1,0,0,9,9,null]]}'
    with pytest.raises(ValueError):
        merge_reconciliation_states((state,), ((0, 0, 0, 9, 9),))


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("states", [[], None, 0, "", b"", {}])
def test_states_wrong_type_raises_type_error(states):
    with pytest.raises(TypeError):
        merge_reconciliation_states(states, WINDOWS)


@pytest.mark.parametrize("member", [0, 1, 1.5, b"", None, [], {}])
def test_state_member_wrong_type_raises_type_error(member):
    with pytest.raises(TypeError):
        merge_reconciliation_states((member,), WINDOWS)


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


def test_negative_level_raises_value_error_without_assessment():
    # Unlike accumulate_reconciliation there is no assessment, so any
    # negative level is rejected (no upper-bound level-range check).
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((-1, 0, 0, 9, 9),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((0, 9, 0, 0, 9),))
    with pytest.raises(ValueError):
        merge_reconciliation_states((), ((0, 0, 9, 9, 0),))


def test_type_errors_are_raised_before_value_errors():
    # A non-str member must surface TypeError even if windows are also bad.
    with pytest.raises(TypeError):
        merge_reconciliation_states((1,), ((-1, 0, 0, 9, 9),))
