"""Tests for :func:`lidar_scan.aggregate_tile_pyramid_delta_windows`."""

from __future__ import annotations

import pytest

from lidar_scan import aggregate_tile_pyramid_delta_windows
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  dzmin=1.0, dzmax=2.0, dcount=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["dzmin"], fields["dzmax"],
            fields["dcount"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.aggregate_tile_pyramid_delta_windows
            is aggregate_tile_pyramid_delta_windows)
    import lidar_scan
    assert "aggregate_tile_pyramid_delta_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert aggregate_tile_pyramid_delta_windows((), ()) == ()
    assert aggregate_tile_pyramid_delta_windows(((), (), ()), ()) == ()


def test_window_without_matches_gets_none_summary():
    assessment = ((),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, None),)


def test_empty_assessment_levels_are_allowed():
    assessment = ((), (), ())
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 10, 10), (2, -5, -5, 5, 5)))
    assert result == (
        (0, 0, 0, 10, 10, None),
        (2, -5, -5, 5, 5, None),
    )


def test_single_match_summary():
    assessment = ((_tile(dzmin=-1.5, dzmax=3.25, dcount=7),),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 255, 255),))
    assert result == ((0, 0, 0, 255, 255, (-1.5, 3.25, 7, 1)),)


def test_multiple_matches_aggregate_extrema_sum_and_count():
    assessment = ((
        _tile(tx=0, ty=0, dzmin=-2.0, dzmax=4.0, dcount=3),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-5.0, dzmax=1.0,
              dcount=-2),
        _tile(tx=2, ty=0, ix0=512, ix1=767, dzmin=0.5, dzmax=9.0,
              dcount=10),
    ),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 600, 255),))
    # Tiles tx=0 and tx=1 intersect [0, 600]; tx=2 starts at ix0=512 and also
    # intersects (512 <= 600), so all three match.
    assert result[0] == (0, 0, 0, 600, 255, (-5.0, 9.0, 11, 3))


def test_only_intersecting_tiles_are_aggregated():
    assessment = ((
        _tile(tx=0, ty=0, dzmin=-2.0, dzmax=4.0, dcount=3),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-5.0, dzmax=1.0,
              dcount=-2),
    ),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 256, 0, 300, 255),))
    assert result[0] == (0, 256, 0, 300, 255, (-5.0, 1.0, -2, 1))


def test_closed_intervals_match_at_endpoints():
    assessment = ((_tile(dzmin=1.0, dzmax=2.0, dcount=4),),)
    # Window touching the tile only at (255, 255) still intersects.
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (1.0, 2.0, 4, 1)
    # Windows starting just past the tile do not intersect.
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 256, 0, 300, 255),))
    assert result[0][5] is None
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 256, 255, 300),))
    assert result[0][5] is None


def test_window_echoes_its_arguments():
    assessment = ((), (), (_tile(),))
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    assessment = ((_tile(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 300, 300, 400, 400),
               (0, 100, 100, 120, 120))
    result = aggregate_tile_pyramid_delta_windows(assessment, windows)
    assert tuple(window[:5] for window in result) == windows
    assert result[0][5] == (1.0, 2.0, 1, 1)
    assert result[1][5] is None
    assert result[2][5] == (1.0, 2.0, 1, 1)


def test_multiple_levels_select_the_right_level():
    assessment = (
        (_tile(dzmin=-1.0, dzmax=1.0, dcount=2),),
        (_tile(ix1=511, iy1=511, dzmin=-3.0, dzmax=3.0, dcount=5),),
    )
    result = aggregate_tile_pyramid_delta_windows(
        assessment,
        ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511),
         (0, 300, 300, 400, 400)))
    assert result[0][5] == (-1.0, 1.0, 2, 1)
    assert result[1][5] == (-3.0, 3.0, 5, 1)
    assert result[2][5] is None


def test_duplicate_windows_are_allowed_and_independent():
    assessment = ((_tile(),),)
    windows = ((0, 0, 0, 255, 255), (0, 0, 0, 255, 255))
    result = aggregate_tile_pyramid_delta_windows(assessment, windows)
    assert result[0] == result[1]


def test_decimal_quantization_half_even_and_negative_zero():
    assessment = ((
        _tile(tx=0, ty=0, dzmin=1.2345675, dzmax=-0.0000002, dcount=1),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.2345685,
              dzmax=2.0000005, dcount=2),
    ),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 600, 255),))
    min_dzmin, max_dzmax, sum_dcount, match_count = result[0][5]
    # 1.2345675 and 1.2345685 both str-convert to ~7-decimal values; the
    # minimum is quantized with ROUND_HALF_EVEN (ties go to even last digit).
    assert min_dzmin == 1.234568
    assert max_dzmax == 2.0
    assert sum_dcount == 3
    assert match_count == 2
    # Negative zero is normalized to positive zero.
    assessment_neg = ((_tile(dzmin=-0.0, dzmax=-0.0),),)
    neg_result = aggregate_tile_pyramid_delta_windows(
        assessment_neg, ((0, 0, 0, 255, 255),))
    assert neg_result[0][5][0] == 0.0
    assert neg_result[0][5][1] == 0.0


def test_extrema_compare_as_decimal_str():
    # 0.1 + 0.2-style floats must compare through Decimal(str(v)), not raw
    # float order surprises: pick values whose str spelling orders clearly.
    assessment = ((
        _tile(tx=0, ty=0, dzmin=-0.3, dzmax=0.1),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-0.299999,
              dzmax=0.300001),
    ),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 600, 255),))
    assert result[0][5][:2] == (-0.3, 0.300001)


def test_repeated_calls_are_deterministic():
    assessment = ((
        _tile(tx=0, ty=0, dzmin=-2.0, dzmax=4.0, dcount=3),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-5.0, dzmax=1.0,
              dcount=-2),
    ),)
    windows = ((0, 0, 0, 600, 255),)
    first = aggregate_tile_pyramid_delta_windows(assessment, windows)
    second = aggregate_tile_pyramid_delta_windows(assessment, windows)
    assert first == second


def test_inputs_are_not_modified():
    tiles = (
        _tile(tx=0, ty=0, dzmin=-2.0, dzmax=4.0, dcount=3),
        _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-5.0, dzmax=1.0,
              dcount=-2),
    )
    assessment = (tiles,)
    snapshot = tuple(tuple(t) for t in tiles)
    aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 0, 0, 600, 255),))
    assert tuple(tuple(t) for t in assessment[0]) == snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_assessment_raises_type_error(bad):
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows(bad, ())


@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_windows_raises_type_error(bad):
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows((), bad)


@pytest.mark.parametrize("bad_window", [
    [0, 0, 0, 1, 1],
    None,
    42,
])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows((), (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows((), ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows((), ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        aggregate_tile_pyramid_delta_windows((), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: assessment structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(([_tile()],), ())


@pytest.mark.parametrize("bad_tile", [
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),            # too few fields
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 0),      # too many fields
    [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],         # list, not tuple
])
def test_tile_wrong_shape_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((bad_tile,),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [1.5, "1", None])
def test_non_int_index_or_dcount_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
def test_bool_index_or_dcount_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = True
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, "1.0", None])
def test_non_float_dz_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_dz_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((tuple(tile),),), ())


def test_inverted_tile_bounds_raise_value_error():
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(
            ((_tile(ix0=256, ix1=200),),), ())
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(
            ((_tile(iy0=256, iy1=200),),), ())


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(unsorted, ())

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(duplicate, ())


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(
            ((),), ((level, 0, 0, 1, 1),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((),), ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        aggregate_tile_pyramid_delta_windows(((),), ((0, 0, 5, 1, 1),))


def test_equal_window_bounds_are_allowed():
    assessment = ((_tile(dzmin=1.0, dzmax=2.0, dcount=4),),)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (1.0, 2.0, 4, 1)
