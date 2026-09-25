"""Tests for :func:`lidar_scan.accumulate_reconciliation`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import accumulate_reconciliation
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=9, iy1=9,
                  err_zmin=1.0, err_zmax=2.0, err_count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["err_zmin"],
            fields["err_zmax"], fields["err_count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.accumulate_reconciliation is accumulate_reconciliation
    import lidar_scan
    assert "accumulate_reconciliation" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding basics
# ---------------------------------------------------------------------------

def test_empty_windows_encodes_empty_document():
    assert accumulate_reconciliation(None, (), ()) == '{"windows":[]}'
    assert accumulate_reconciliation(None, ((), ()), ()) == '{"windows":[]}'


def test_empty_state_string_is_accepted_for_empty_windows():
    assert (accumulate_reconciliation('{"windows":[]}', (), ())
            == '{"windows":[]}')


def test_no_match_window_is_null_and_others_keep_keys():
    assessment = ((_tile(),),)
    windows = ((0, 0, 0, 9, 9), (0, 100, 100, 109, 109))
    text = accumulate_reconciliation(None, assessment, windows)
    assert text == (
        '{"windows":[[0,0,0,9,9,[1.000000,2.000000,5000000000000,1,1]],'
        '[0,100,100,109,109,null]]}'
    )
    document = json.loads(text)
    assert set(document) == {"windows"}


def test_summary_fields_quantized_and_scaled_to_microunits():
    # err_zmin 0.0000005 rounds to 0.000000 (ROUND_HALF_EVEN tie);
    # err_zmax 1.0000006 rounds up to 1.000001.
    assessment = (((0, 0, 0, 0, 9, 9, 0.0000005, 1.0000006, -4),),)
    text = accumulate_reconciliation(None, assessment, ((0, 0, 0, 9, 9),))
    assert text == (
        '{"windows":[[0,0,0,9,9,[0.000000,1.000001,1000002000001,4,1]]]}'
    )


def test_extremes_qsum_asum_and_count_over_multiple_tiles():
    assessment = ((
        _tile(tx=0, err_zmin=-2.5, err_zmax=0.25, err_count=-3),
        _tile(tx=1, ix0=10, ix1=19, err_zmin=1.5, err_zmax=3.25,
              err_count=7),
    ),)
    text = accumulate_reconciliation(None, assessment, ((0, 0, 0, 19, 9),))
    qsum = (2_500_000 ** 2 + 250_000 ** 2
            + 1_500_000 ** 2 + 3_250_000 ** 2)
    assert text == (
        '{"windows":[[0,0,0,19,9,[-2.500000,3.250000,'
        f'{qsum},10,2]]]}}'
    )


def test_negative_zero_extreme_is_formatted_as_zero():
    assessment = (((0, 0, 0, 0, 9, 9, -0.0000001, -0.0000001, 0),),)
    text = accumulate_reconciliation(None, assessment, ((0, 0, 0, 9, 9),))
    assert text == '{"windows":[[0,0,0,9,9,[0.000000,0.000000,0,0,1]]]}'


def test_intersection_uses_closed_intervals():
    # Tile covers ix 0..9, iy 0..9; boundary windows still match.
    assessment = ((_tile(),),)
    text = accumulate_reconciliation(
        None, assessment, ((0, 9, 9, 20, 20),))
    assert "[0,9,9,20,20,[1.000000,2.000000,5000000000000,1,1]]" in text
    text = accumulate_reconciliation(
        None, assessment, ((0, 10, 10, 20, 20),))
    assert "[0,10,10,20,20,null]" in text


def test_output_is_compact_without_whitespace_or_nonfinite_literals():
    assessment = ((_tile(),),)
    text = accumulate_reconciliation(None, assessment, ((0, 0, 0, 9, 9),))
    assert " " not in text
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# accumulation / batching
# ---------------------------------------------------------------------------

def _split_assessment():
    first = ((_tile(tx=0, err_zmin=-2.5, err_zmax=0.25, err_count=-3),),)
    second = ((_tile(tx=1, ix0=10, ix1=19, err_zmin=1.5,
                     err_zmax=3.25, err_count=7),),)
    whole = ((first[0][0], second[0][0]),)
    return first, second, whole


def test_batch_split_matches_single_batch():
    first, second, whole = _split_assessment()
    windows = ((0, 0, 0, 19, 9), (0, 100, 100, 109, 109))
    expected = accumulate_reconciliation(None, whole, windows)

    state = accumulate_reconciliation(None, first, windows)
    state = accumulate_reconciliation(state, second, windows)
    assert state == expected


def test_batch_order_does_not_change_result():
    first, second, whole = _split_assessment()
    windows = ((0, 0, 0, 19, 9),)
    expected = accumulate_reconciliation(None, whole, windows)

    state = accumulate_reconciliation(None, second, windows)
    state = accumulate_reconciliation(state, first, windows)
    assert state == expected


def test_no_match_batch_leaves_prior_state_unchanged():
    first, _, _ = _split_assessment()
    windows = ((0, 0, 0, 19, 9), (0, 100, 100, 109, 109))
    state = accumulate_reconciliation(None, first, windows)
    assert accumulate_reconciliation(state, ((),), windows) == state
    # An assessment whose tiles miss every window also leaves it unchanged.
    misses = ((_tile(tx=9, ix0=200, iy0=200, ix1=209, iy1=209),),)
    assert accumulate_reconciliation(state, misses, windows) == state


def test_repeated_accumulation_is_idempotent_per_window():
    assessment = ((_tile(err_zmin=-1.0, err_zmax=1.0),),)
    windows = ((0, 0, 0, 9, 9),)
    first = accumulate_reconciliation(None, assessment, windows)
    # Feeding an empty batch with the state must be stable.
    assert accumulate_reconciliation(first, ((),), windows) == first


def test_null_summary_gets_filled_by_later_batch():
    windows = ((0, 0, 0, 9, 9),)
    state = accumulate_reconciliation(None, ((),), windows)
    assert state == '{"windows":[[0,0,0,9,9,null]]}'
    assessment = ((_tile(err_zmin=-1.0, err_zmax=4.0, err_count=2),),)
    state = accumulate_reconciliation(state, assessment, windows)
    assert state == (
        '{"windows":[[0,0,0,9,9,[-1.000000,4.000000,17000000000000,2,1]]]}')


def test_extremes_merge_across_batches():
    windows = ((0, 0, 0, 9, 9),)
    batch_a = ((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=1),),)
    batch_b = ((_tile(err_zmin=-3.0, err_zmax=5.0, err_count=2),),)
    batch_c = ((_tile(err_zmin=0.5, err_zmax=4.5, err_count=3),),)
    state = accumulate_reconciliation(None, batch_a, windows)
    state = accumulate_reconciliation(state, batch_b, windows)
    state = accumulate_reconciliation(state, batch_c, windows)
    assert state == (
        '{"windows":[[0,0,0,9,9,[-3.000000,5.000000,'
        '59500000000000,6,3]]]}')


def test_multi_window_and_multi_level_batching():
    level0 = (
        _tile(tx=0, err_zmin=0.1, err_zmax=0.2, err_count=1),
        _tile(tx=1, ix0=10, ix1=19, err_zmin=0.3, err_zmax=0.4,
              err_count=2),
    )
    level1 = (_tile(tx=0, err_zmin=-0.5, err_zmax=0.5, err_count=4),)
    windows = ((0, 0, 0, 9, 9), (0, 0, 0, 19, 9), (1, 0, 0, 9, 9))

    state = accumulate_reconciliation(None, (level0, ()), windows)
    state = accumulate_reconciliation(state, ((), level1), windows)
    assert state == (
        '{"windows":['
        '[0,0,0,9,9,[0.100000,0.200000,50000000000,1,1]],'
        '[0,0,0,19,9,[0.100000,0.400000,300000000000,3,2]],'
        '[1,0,0,9,9,[-0.500000,0.500000,500000000000,4,1]]'
        ']}')


def test_inputs_are_not_modified():
    assessment = ((_tile(),),)
    windows = ((0, 0, 0, 9, 9),)
    snapshot = (json.dumps(assessment), json.dumps(windows))
    state = accumulate_reconciliation(None, assessment, windows)
    accumulate_reconciliation(state, assessment, windows)
    assert (json.dumps(assessment), json.dumps(windows)) == snapshot


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
    '{"windows":[[0,0,0,9,9]]}',                      # entry too short
    '{"windows":[[0,0,0,9,9,[1,2,3,4]]]}',            # summary too short
    '{"windows":[[0,0,0,9,9,{}]]}',                   # bad summary type
    '{"windows":[["0",0,0,9,9,null]]}',               # non-int key
    '{"windows":[[0,0,0,9,9,[1,2,0,0,1]]]}',          # int emin (not 6dp)
    '{"windows":[[0,0,0,9,9,[1.0,2.000000,0,0,1]]]}', # emin not six decimals
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0,0.5,1]]]}',  # float n
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0,0,0]]]}',    # n == 0
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,-1,0,1]]]}',   # qsum < 0
    '{"windows":[[0,0,0,9,9,[NaN,2.000000,0,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,Infinity,0,0,1]]]}',
    '{ "windows":[]}',                                # whitespace
    '{"windows" :[]}',
    "{'windows':[]}",                                 # not JSON
    '{"windows":[],"extra":1}',
])
def test_noncanonical_state_raises_value_error(bad_state):
    windows = ((0, 0, 0, 9, 9),)
    with pytest.raises(ValueError):
        accumulate_reconciliation(bad_state, ((),), windows)


def test_empty_state_with_nonempty_windows_raises_value_error():
    with pytest.raises(ValueError):
        accumulate_reconciliation('{"windows":[]}', ((),),
                                  ((0, 0, 0, 9, 9),))


def test_state_window_count_mismatch_raises_value_error():
    assessment = ((),)
    state = '{"windows":[[0,0,0,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        accumulate_reconciliation(state, assessment, ((0, 0, 0, 9, 9),))


def test_state_window_keys_must_match_in_order():
    assessment = ((),)
    state = '{"windows":[[0,5,5,9,9,null],[0,0,0,4,4,null]]}'
    with pytest.raises(ValueError):
        accumulate_reconciliation(state, assessment,
                                  ((0, 0, 0, 4, 4), (0, 5, 5, 9, 9)))
    # The same keys in the matching order are accepted.
    good = '{"windows":[[0,0,0,4,4,null],[0,5,5,9,9,null]]}'
    assert (accumulate_reconciliation(good, assessment,
                                      ((0, 0, 0, 4, 4),
                                       (0, 5, 5, 9, 9)))
            == good)


def test_state_with_wrong_key_value_raises_value_error():
    state = '{"windows":[[1,0,0,9,9,null]]}'
    with pytest.raises(ValueError):
        accumulate_reconciliation(state, ((), ()), ((0, 0, 0, 9, 9),))


# ---------------------------------------------------------------------------
# argument validation (mirrors query_tile_reconciliation_windows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", [0, 1, 1.5, b"", [], {}, object()])
def test_state_wrong_type_raises_type_error(state):
    with pytest.raises(TypeError):
        accumulate_reconciliation(state, ((),), ((0, 0, 0, 9, 9),))


def test_assessment_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, [], ())
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, [], ())


def test_windows_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, (), [])


def test_window_container_and_length_type_errors():
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, (), ([0, 0, 0, 9, 9],))
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, (), ((0, 0, 0, 9),))
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, (), ((0, 0, 0, 9, 9, 0),))


@pytest.mark.parametrize("bad_window", [
    (True, 0, 0, 9, 9),
    (0, 1.5, 0, 9, 9),
    (0, 0, "0", 9, 9),
    (0, 0, 0, 9.0, 9),
    (0, 0, 0, 9, None),
])
def test_window_field_types_raise_type_error(bad_window):
    with pytest.raises(TypeError):
        accumulate_reconciliation(None, ((),), (bad_window,))


def test_level_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        accumulate_reconciliation(None, ((),), ((1, 0, 0, 9, 9),))
    with pytest.raises(ValueError):
        accumulate_reconciliation(None, ((),), ((-1, 0, 0, 9, 9),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        accumulate_reconciliation(None, ((),), ((0, 9, 0, 0, 9),))
    with pytest.raises(ValueError):
        accumulate_reconciliation(None, ((),), ((0, 0, 9, 9, 0),))


def test_bad_assessment_structure_raises_value_error():
    good_tile = _tile()
    with pytest.raises(ValueError):
        accumulate_reconciliation(None, ([good_tile],), ((0, 0, 0, 9, 9),))
    with pytest.raises(ValueError):
        # duplicate (tx, ty)
        accumulate_reconciliation(None, ((good_tile, good_tile),),
                                  ((0, 0, 0, 9, 9),))
    with pytest.raises(ValueError):
        # non-finite error
        accumulate_reconciliation(
            None, (((0, 0, 0, 0, 9, 9, float("nan"), 2.0, 0),),),
            ((0, 0, 0, 9, 9),))
