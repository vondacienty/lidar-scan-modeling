"""Tests for :func:`lidar_scan.rollback_tile_pyramid_delta_windows`."""

from __future__ import annotations

import copy
import math

import pytest

from lidar_scan import (assess_tile_pyramid_deltas, build_tile_pyramid,
                        rollback_tile_pyramid_delta_windows)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _delta(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  dzmin=0.0, dzmax=0.0, dcount=0)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["dzmin"],
            fields["dzmax"], fields["dcount"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.rollback_tile_pyramid_delta_windows
            is rollback_tile_pyramid_delta_windows)
    import lidar_scan
    assert "rollback_tile_pyramid_delta_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert rollback_tile_pyramid_delta_windows((), (), ()) == ()
    base = ((_tile(),),)
    deltas = ((_delta(),),)
    assert rollback_tile_pyramid_delta_windows(base, deltas, ()) == ()


def test_empty_deltas_queries_base_unchanged():
    base = (
        (_tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),
    )
    result = rollback_tile_pyramid_delta_windows(
        base, (), ((0, 0, 0, 600, 600),))
    assert result == (
        (0, 0, 0, 600, 600, base[0]),
    )


def test_window_without_matches_keeps_window_and_empty_tiles():
    base = ((),)
    assert rollback_tile_pyramid_delta_windows(
        base, (), ((0, 0, 0, 10, 10),)) == ((0, 0, 0, 10, 10, ()),)


def test_empty_levels_are_allowed():
    base = ((), (), ())
    deltas = ((), (), ())
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 10, 10), (2, -5, -5, 5, 5)))
    assert result == (
        (0, 0, 0, 10, 10, ()),
        (2, -5, -5, 5, 5, ()),
    )


def test_rollback_with_shuffled_deltas_matches_by_coordinate():
    base = ((
        _tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=4),
    ),)
    # Deltas presented in reverse order with swapped geometry so positional
    # matching would fail.
    deltas = ((
        _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=1.0, dcount=4),
        _delta(dzmin=2.0, dzmax=-2.0, dcount=3),
    ),)
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 256, 0, 300, 255),))
    assert result == (
        (0, 256, 0, 300, 255,
         ((1, 0, 256, 0, 511, 255, 2, 4, 0),)),
    )


def test_result_independent_of_delta_order():
    base = ((
        _tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmin=1.0, zmax=2.0, count=2),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=4),
    ),)
    ordered = ((
        _delta(dzmin=2.0, dzmax=-2.0, dcount=3),
        _delta(tx=0, ty=1, iy0=256, iy1=511,
               dzmin=0.5, dzmax=0.5, dcount=1),
        _delta(tx=1, ty=0, ix0=256, ix1=511,
               dzmin=1.0, dzmax=1.0, dcount=4),
    ),)
    shuffled = ((ordered[0][2], ordered[0][0], ordered[0][1]),)
    windows = ((0, 0, 0, 600, 600),)
    assert (rollback_tile_pyramid_delta_windows(base, ordered, windows)
            == rollback_tile_pyramid_delta_windows(base, shuffled, windows))


def test_closed_intervals_match_at_endpoints():
    base = ((_tile(zmin=1.0, zmax=1.0, count=1),),)
    deltas = ((_delta(),),)
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 255, 255, 255, 255),))
    assert result[0][5] == ((0, 0, 0, 0, 255, 255, 1, 1, 1),)
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 256, 0, 300, 255),))
    assert result[0][5] == ()


def test_window_echoes_its_arguments():
    base = ((), (), (_tile(ix0=0, iy0=0, ix1=1023, iy1=1023),))
    deltas = ((), (), (_delta(ix0=0, iy0=0, ix1=1023, iy1=1023),))
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    base = ((_tile(),),)
    deltas = ((_delta(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = rollback_tile_pyramid_delta_windows(base, deltas, windows)
    assert tuple(window[:5] for window in result) == windows


def test_tiles_preserve_base_order():
    base = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=1, iy0=256, iy1=511),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    deltas = ((
        _delta(tx=1, ty=0, ix0=256, ix1=511,
               dzmin=-1.0, dzmax=-1.0, dcount=0),
        _delta(tx=0, ty=1, iy0=256, iy1=511,
               dzmin=0.0, dzmax=0.0, dcount=1),
        _delta(dzmin=1.0, dzmax=1.0, dcount=0),
    ),)
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 600, 600),))
    assert [tile[0:2] for tile in result[0][5]] == [
        (0, 0), (0, 1), (1, 0)]
    assert [tile[6:9] for tile in result[0][5]] == [
        (0, 1, 1), (1, 2, 0), (2, 3, 1)]


def test_multiple_levels_select_the_right_level():
    base = (
        (_tile(),),
        (_tile(ix1=511, iy1=511, zmin=4.0, zmax=4.0, count=2),),
    )
    deltas = (
        (_delta(dzmin=0.0, dzmax=0.0, dcount=1),),
        (_delta(ix1=511, iy1=511, dzmin=1.0, dzmax=1.0, dcount=2),),
    )
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511)))
    assert result[0][5] == ((0, 0, 0, 0, 255, 255, 1, 2, 0),)
    assert result[1][5] == ((0, 0, 0, 0, 511, 511, 3, 3, 0),)


def test_non_integral_values_become_quantized_floats():
    base = ((_tile(zmin=0.3, zmax=0.5, count=1),),)
    deltas = ((_delta(dzmin=0.1, dzmax=0.2, dcount=0),),)
    tile = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0.2
    assert tile[7] == 0.3
    assert isinstance(tile[6], float)
    assert isinstance(tile[7], float)


def test_integral_values_are_exact_ints():
    base = ((_tile(zmin=10.0, zmax=20.0, count=8),),)
    deltas = ((_delta(dzmin=2.0, dzmax=-2.0, dcount=3),),)
    tile = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert isinstance(tile[6], int)
    assert isinstance(tile[7], int)
    assert isinstance(tile[8], int)


def test_zero_float_result_normalized_to_positive_zero():
    base = ((_tile(zmin=1.0, zmax=1.0, count=1),),)
    deltas = ((_delta(dzmin=1.0, dzmax=1.0, dcount=0),),)
    tile = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0 and math.copysign(1.0, float(tile[6])) == 1.0
    assert tile[7] == 0 and math.copysign(1.0, float(tile[7])) == 1.0


def test_inputs_are_not_modified_and_calls_are_stable():
    base = ((_tile(zmin=1.5, zmax=3.0, count=4),
             _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0,
                   count=2)),)
    deltas = ((_delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=2.0,
                      dcount=1),
               _delta(dzmin=0.5, dzmax=0.0, dcount=1)),)
    windows = ((0, 0, 0, 600, 600),)
    base_before = copy.deepcopy(base)
    deltas_before = copy.deepcopy(deltas)
    first = rollback_tile_pyramid_delta_windows(base, deltas, windows)
    second = rollback_tile_pyramid_delta_windows(base, deltas, windows)
    assert first == second
    assert base == base_before
    assert deltas == deltas_before


def test_inverts_assess_tile_pyramid_deltas_via_windows_with_shuffle():
    estimate = build_tile_pyramid(
        [(0.0, 0.0, 3.0, 1, 1.0), (300.0, 0.0, 2.0, 1, 1.0)], levels=2)
    reference = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 1, 1.0), (300.0, 0.0, 5.0, 1, 1.0)], levels=2)
    deltas = assess_tile_pyramid_deltas(estimate, reference)
    shuffled = tuple(
        (tuple(reversed(level)) if level else level) for level in deltas)
    windows = ((0, 0, 0, 600, 600), (1, 0, 0, 600, 600))
    result = rollback_tile_pyramid_delta_windows(
        estimate, shuffled, windows)
    assert tuple(window[5] for window in result) == reference


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows(bad, (), ())
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows((), bad, ())
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows((), (), bad)


@pytest.mark.parametrize("bad", [[], None, [0, 0, 0, 1, 1]])
def test_window_not_tuple_raises_type_error(bad):
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows((), (), (bad,))


@pytest.mark.parametrize("bad", [(0, 0, 0, 1), (0, 0, 0, 1, 1, 1)])
def test_window_wrong_length_raises_type_error(bad):
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows((), (), (bad,))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None, 1.0])
def test_window_fields_must_be_non_bool_ints(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        rollback_tile_pyramid_delta_windows((), (), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

def test_level_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((), ()), (), ((2, 0, 0, 1, 1),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((), ()), (), ((-1, 0, 0, 1, 1),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((),), (), ((0, 2, 0, 1, 1),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((),), (), ((0, 0, 2, 1, 1),))


# ---------------------------------------------------------------------------
# ValueError: base/deltas contract
# ---------------------------------------------------------------------------

def test_bad_base_structure_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ([_tile()],), ((_delta(),),), ())


def test_bad_delta_structure_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(),),), ([_delta()],), ())


def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(),),), ((), ()), ((0, 0, 0, 1, 1),))


def test_different_coordinate_sets_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(tx=0, ty=0),),),
            ((_delta(tx=1, ty=0, ix0=256, ix1=511),),),
            ((0, 0, 0, 600, 600),))


def test_extra_delta_coordinate_raises_value_error():
    base = ((_tile(tx=0, ty=0),),)
    deltas = ((_delta(), _delta(tx=1, ty=0, ix0=256, ix1=511)),)
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            base, deltas, ((0, 0, 0, 600, 600),))


def test_empty_delta_level_against_nonempty_base_raises_value_error():
    base = ((_tile(),),)
    deltas = ((),)
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            base, deltas, ((0, 0, 0, 600, 600),))


def test_duplicate_delta_coordinate_raises_value_error():
    base = ((_tile(tx=0, ty=0), _tile(tx=1, ty=0, ix0=256, ix1=511)),)
    deltas = ((_delta(), _delta()),)
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            base, deltas, ((0, 0, 0, 600, 600),))


def test_same_coordinate_disagreeing_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(tx=0, ty=0),),),
            ((_delta(ix0=1, ix1=255),),),
            ((0, 0, 0, 600, 600),))


def test_negative_resulting_count_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(zmin=1.0, zmax=2.0, count=1),),),
            ((_delta(dzmin=0.0, dzmax=0.0, dcount=2),),),
            ((0, 0, 0, 255, 255),))


def test_inverted_resulting_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_delta_windows(
            ((_tile(zmin=1.0, zmax=1.0, count=1),),),
            ((_delta(dzmin=-1.0, dzmax=1.0, dcount=0),),),
            ((0, 0, 0, 255, 255),))
