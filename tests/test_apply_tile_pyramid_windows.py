"""Tests for :func:`lidar_scan.apply_tile_pyramid_windows`."""

from __future__ import annotations

import copy
import math

import pytest

from lidar_scan import (build_tile_pyramid,
                        assess_tile_pyramid_deltas,
                        apply_tile_pyramid_windows,
                        query_tile_pyramid_windows)
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
    assert (tiles_module.apply_tile_pyramid_windows
            is apply_tile_pyramid_windows)
    import lidar_scan
    assert "apply_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert apply_tile_pyramid_windows((), (), ()) == ()
    base = ((_tile(),),)
    deltas = ((_delta(),),)
    assert apply_tile_pyramid_windows(base, deltas, ()) == ()


def test_empty_deltas_queries_base_unchanged():
    base = (
        (_tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),
    )
    result = apply_tile_pyramid_windows(base, (), ((0, 0, 0, 600, 600),))
    assert result == (
        (0, 0, 0, 600, 600, base[0]),
    )


def test_window_without_matches_keeps_window_and_empty_tiles():
    base = ((),)
    assert apply_tile_pyramid_windows(base, (), ((0, 0, 0, 10, 10),)) == \
        ((0, 0, 0, 10, 10, ()),)


def test_empty_levels_are_allowed():
    base = ((), (), ())
    result = apply_tile_pyramid_windows(
        base, (), ((0, 0, 0, 10, 10), (2, -5, -5, 5, 5)))
    assert result == (
        (0, 0, 0, 10, 10, ()),
        (2, -5, -5, 5, 5, ()),
    )


def test_apply_then_window_selection():
    base = ((
        _tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=4),
    ),)
    deltas = ((
        _delta(dzmin=2.0, dzmax=-2.0, dcount=3),
        _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=1.0, dcount=4),
    ),)
    result = apply_tile_pyramid_windows(
        base, deltas, ((0, 256, 0, 300, 255),))
    assert result == (
        (0, 256, 0, 300, 255,
         ((1, 0, 256, 0, 511, 255, 4, 6, 8),)),
    )


def test_apply_then_window_matches_expected_selection():
    base = ((
        _tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmin=1.0, zmax=2.0, count=2),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=4),
    ),)
    deltas = ((
        _delta(dzmin=2.0, dzmax=-2.0, dcount=3),
        _delta(tx=0, ty=1, iy0=256, iy1=511, dzmin=0.5, dzmax=0.5, dcount=1),
        _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=1.0, dcount=4),
    ),)
    windows = ((0, 256, 0, 300, 600), (0, 0, 0, 10, 10))
    result = apply_tile_pyramid_windows(base, deltas, windows)
    assert result[0][:5] == (0, 256, 0, 300, 600)
    # Tile (0, 1) has ix1 = 255 < ix_min = 256, so only tile (1, 0) matches.
    assert result[0][5] == (
        (1, 0, 256, 0, 511, 255, 4, 6, 8),
    )
    assert result[1] == (
        0, 0, 0, 10, 10,
        ((0, 0, 0, 0, 255, 255, 12, 18, 11),))


def test_closed_intervals_match_at_endpoints():
    base = ((_tile(zmin=1.0, zmax=1.0, count=1),),)
    deltas = ((_delta(),),)
    result = apply_tile_pyramid_windows(
        base, deltas, ((0, 255, 255, 255, 255),))
    assert result[0][5] == ((0, 0, 0, 0, 255, 255, 1, 1, 1),)
    result = apply_tile_pyramid_windows(
        base, deltas, ((0, 256, 0, 300, 255),))
    assert result[0][5] == ()


def test_window_echoes_its_arguments():
    base = ((), (), (_tile(ix0=0, iy0=0, ix1=1023, iy1=1023),))
    deltas = ((), (), (_delta(ix0=0, iy0=0, ix1=1023, iy1=1023),))
    result = apply_tile_pyramid_windows(
        base, deltas, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    base = ((_tile(),),)
    deltas = ((_delta(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = apply_tile_pyramid_windows(base, deltas, windows)
    assert tuple(window[:5] for window in result) == windows


def test_tiles_preserve_base_order():
    base = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=1, iy0=256, iy1=511),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    deltas = ((
        _delta(dzmin=1.0, dzmax=1.0, dcount=0),
        _delta(tx=0, ty=1, iy0=256, iy1=511, dzmin=0.0, dzmax=0.0, dcount=1),
        _delta(tx=1, ty=0, ix0=256, ix1=511,
               dzmin=-1.0, dzmax=-1.0, dcount=0),
    ),)
    result = apply_tile_pyramid_windows(
        base, deltas, ((0, 0, 0, 600, 600),))
    assert [tile[0:2] for tile in result[0][5]] == [
        (0, 0), (0, 1), (1, 0)]
    assert [tile[6:9] for tile in result[0][5]] == [
        (2, 3, 1), (1, 2, 2), (0, 1, 1)]


def test_multiple_levels_select_the_right_level():
    base = (
        (_tile(),),
        (_tile(ix1=511, iy1=511, zmin=4.0, zmax=4.0, count=2),),
    )
    deltas = (
        (_delta(dzmin=0.0, dzmax=0.0, dcount=1),),
        (_delta(ix1=511, iy1=511, dzmin=1.0, dzmax=2.0, dcount=2),),
    )
    result = apply_tile_pyramid_windows(
        base, deltas, ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511)))
    assert result[0][5] == ((0, 0, 0, 0, 255, 255, 1, 2, 2),)
    assert result[1][5] == ((0, 0, 0, 0, 511, 511, 5, 6, 4),)


def test_non_integral_values_become_quantized_floats():
    base = ((_tile(zmin=0.3, zmax=0.5, count=1),),)
    deltas = ((_delta(dzmin=0.1, dzmax=0.2, dcount=0),),)
    tile = apply_tile_pyramid_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0.4
    assert tile[7] == 0.7
    assert isinstance(tile[6], float)
    assert isinstance(tile[7], float)


def test_integral_results_are_floats_not_ints():
    base = ((_tile(zmin=10.0, zmax=20.0, count=8),),)
    deltas = ((_delta(dzmin=2.0, dzmax=-2.0, dcount=3),),)
    tile = apply_tile_pyramid_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert isinstance(tile[6], float)
    assert isinstance(tile[7], float)
    assert isinstance(tile[8], int) and not isinstance(tile[8], bool)
    assert tile[6:9] == (12.0, 18.0, 11)


def test_zero_float_result_normalized_to_positive_zero():
    base = ((_tile(zmin=1.0, zmax=1.0, count=1),),)
    deltas = ((_delta(dzmin=-1.0, dzmax=-1.0, dcount=0),),)
    tile = apply_tile_pyramid_windows(
        base, deltas, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0 and math.copysign(1.0, float(tile[6])) == 1.0
    assert tile[7] == 0 and math.copysign(1.0, float(tile[7])) == 1.0


def test_inputs_are_not_modified_and_calls_are_stable():
    base = ((_tile(zmin=1.5, zmax=3.0, count=4),
             _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),)
    deltas = ((_delta(dzmin=0.5, dzmax=0.0, dcount=1),
               _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=2.0,
                      dcount=1)),)
    windows = ((0, 0, 0, 600, 600),)
    base_before = copy.deepcopy(base)
    deltas_before = copy.deepcopy(deltas)
    first = apply_tile_pyramid_windows(base, deltas, windows)
    second = apply_tile_pyramid_windows(base, deltas, windows)
    assert first == second
    assert base == base_before
    assert deltas == deltas_before


def test_apply_recovers_estimate_from_reference_and_deltas_via_windows():
    estimate = build_tile_pyramid(
        [(0.0, 0.0, 3.0, 1, 1.0), (300.0, 0.0, 2.0, 1, 1.0)], levels=2)
    reference = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 1, 1.0), (300.0, 0.0, 5.0, 1, 1.0)], levels=2)
    deltas = assess_tile_pyramid_deltas(estimate, reference)
    windows = ((0, 0, 0, 600, 600), (1, 0, 0, 600, 600))
    result = apply_tile_pyramid_windows(reference, deltas, windows)
    assert result == query_tile_pyramid_windows(estimate, windows)


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows(bad, (), ())
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows((), bad, ())
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows((), (), bad)


@pytest.mark.parametrize("bad", [[], None, [0, 0, 0, 1, 1]])
def test_window_not_tuple_raises_type_error(bad):
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows((), (), (bad,))


@pytest.mark.parametrize("bad", [(0, 0, 0, 1), (0, 0, 0, 1, 1, 1)])
def test_window_wrong_length_raises_type_error(bad):
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows((), (), (bad,))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None, 1.0])
def test_window_fields_must_be_non_bool_ints(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        apply_tile_pyramid_windows((), (), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

def test_level_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(((), ()), (), ((2, 0, 0, 1, 1),))
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(((), ()), (), ((-1, 0, 0, 1, 1),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(((),), (), ((0, 2, 0, 1, 1),))
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(((),), (), ((0, 0, 2, 1, 1),))


# ---------------------------------------------------------------------------
# ValueError: base/deltas contract
# ---------------------------------------------------------------------------

def test_bad_base_structure_raises_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(([_tile()],), ((_delta(),),), ())


def test_bad_delta_structure_raises_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(((_tile(),),), ([_delta()],), ())


def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(),),), ((), ()), ((0, 0, 0, 1, 1),))


def test_different_coordinate_sets_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(tx=0, ty=0),),),
            ((_delta(tx=1, ty=0, ix0=256, ix1=511),),),
            ((0, 0, 0, 600, 600),))


def test_mismatched_cell_bounds_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(),),),
            ((_delta(ix1=127),),),
            ())


def test_base_inverted_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(zmin=2.0, zmax=1.0),),),
            ((_delta(),),),
            ())


def test_base_negative_count_raises_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(count=-1),),),
            ((_delta(),),),
            ())


def test_negative_resulting_count_raises_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(zmin=1.0, zmax=2.0, count=1),),),
            ((_delta(dzmin=0.0, dzmax=0.0, dcount=-2),),),
            ((0, 0, 0, 255, 255),))


def test_inverted_resulting_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        apply_tile_pyramid_windows(
            ((_tile(zmin=1.0, zmax=1.0, count=1),),),
            ((_delta(dzmin=1.0, dzmax=-1.0, dcount=0),),),
            ((0, 0, 0, 255, 255),))
