"""Tests for :func:`lidar_scan.assess_tile_pyramid_windows`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import assess_tile_pyramid_windows
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, zmean=1.5, zsigma=0.5, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["zmean"], fields["zsigma"], fields["count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.assess_tile_pyramid_windows is assess_tile_pyramid_windows
    import lidar_scan
    assert "assess_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_everything_returns_empty_tuple():
    assert assess_tile_pyramid_windows((), (), ()) == ()


def test_window_without_matches_keeps_window_and_empty_tiles():
    estimate = ((),)
    reference = ((),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, ()),)


def test_single_match_metrics():
    estimate = ((_tile(zmean=12.0, zsigma=3.0, count=2),),)
    reference = ((_tile(zmean=9.0, zsigma=4.0, count=5),),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))
    assert len(result) == 1
    level, ix_min, iy_min, ix_max, iy_max, tiles = result[0]
    assert (level, ix_min, iy_min, ix_max, iy_max) == (0, 0, 0, 255, 255)
    assert tiles == (
        (0, 0, 0, 0, 255, 255, 3.0, 3.0, 5.0, 0.6, -3),
    )


def test_window_echoes_its_arguments():
    empty_level = ()
    estimate = (empty_level, empty_level, (_tile(),))
    reference = (empty_level, empty_level, (_tile(),))
    result = assess_tile_pyramid_windows(
        estimate, reference, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    estimate = ((_tile(),),)
    reference = ((_tile(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = assess_tile_pyramid_windows(estimate, reference, windows)
    assert tuple(window[:5] for window in result) == windows


def test_tiles_preserve_estimate_order():
    estimate = ((
        _tile(tx=0, ty=0, zmean=2.0, zsigma=1.0, count=2),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmean=3.0, zsigma=1.0, count=3),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=4.0, zsigma=1.0, count=4),
    ),)
    reference = ((
        _tile(tx=0, ty=0, zmean=1.0, zsigma=1.0, count=7),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmean=1.0, zsigma=1.0, count=8),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=1.0, zsigma=1.0, count=9),
    ),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 600, 600),))
    tiles = result[0][5]
    assert [tile[0:2] for tile in tiles] == [(0, 0), (0, 1), (1, 0)]
    assert [tile[10] for tile in tiles] == [-5, -5, -5]
    assert [tile[6] for tile in tiles] == [1.0, 2.0, 3.0]


def test_only_tiles_intersecting_the_window_are_returned():
    estimate = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    reference = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 256, 0, 300, 255),))
    assert [tile[0:2] for tile in result[0][5]] == [(1, 0)]


def test_closed_intervals_match_at_endpoints():
    estimate = ((_tile(),),)
    reference = ((_tile(),),)
    # Window touching the tile only at (255, 255) still intersects.
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 255, 255, 255, 255),))
    assert len(result[0][5]) == 1
    # Window starting just past the tile does not intersect.
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 256, 0, 300, 255),))
    assert result[0][5] == ()


def test_tile_on_only_one_side_is_not_a_match():
    estimate = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=2.0, zsigma=1.0),
    ),)
    reference = ((_tile(tx=0, ty=0, zmean=1.0, zsigma=1.0),),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 600, 600),))
    assert [tile[0:2] for tile in result[0][5]] == [(0, 0)]
    # A window containing the estimate-only tile yields no match.
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 256, 0, 600, 255),))
    assert result[0][5] == ()


def test_matching_tile_outside_window_does_not_raise_on_bounds_mismatch():
    estimate = ((_tile(ix1=200),),)
    reference = ((_tile(ix1=255),),)
    result = assess_tile_pyramid_windows(
        estimate, reference, ((0, 256, 256, 300, 300),))
    assert result[0][5] == ()


def test_negative_bias_gives_negative_z_score():
    estimate = ((_tile(zmean=9.0, zsigma=3.0),),)
    reference = ((_tile(zmean=12.0, zsigma=4.0),),)
    tiles = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))[0][5]
    _, _, _, _, _, _, bias, abs_error, combined, z_score, count_delta = tiles[0]
    assert bias == -3.0
    assert abs_error == 3.0
    assert combined == pytest.approx(5.0, abs=1e-6)
    assert z_score == pytest.approx(-0.6, abs=1e-6)
    assert count_delta == 0


def test_count_delta_is_estimate_minus_reference():
    estimate = ((_tile(count=3),),)
    reference = ((_tile(count=10),),)
    tiles = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))[0][5]
    assert tiles[0][10] == -7


def test_zero_bias_normalized_to_positive_zero():
    estimate = ((_tile(zmean=10.0, zsigma=2.0),),)
    reference = ((_tile(zmean=10.0, zsigma=2.0),),)
    tile = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0.0 and math.copysign(1.0, tile[6]) == 1.0
    assert tile[7] == 0.0
    assert tile[9] == 0.0 and math.copysign(1.0, tile[9]) == 1.0


def test_quantized_to_six_decimal_places():
    estimate = ((_tile(zmean=1 / 3, zsigma=1.0),),)
    reference = ((_tile(zmean=0.0, zsigma=1.0),),)
    tile = assess_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))[0][5][0]
    assert tile[6] == 0.333333
    assert tile[8] == pytest.approx(math.sqrt(2.0), abs=5e-7)
    assert tile[9] == pytest.approx((1 / 3) / math.sqrt(2.0), abs=5e-7)


def test_multiple_levels_select_the_right_level():
    estimate = (
        (_tile(),),
        (_tile(ix1=511, iy1=511, zmean=4.0, zsigma=1.0),),
    )
    reference = (
        (_tile(zmean=1.0, zsigma=1.0),),
        (_tile(ix1=511, iy1=511, zmean=1.0, zsigma=1.0),),
    )
    result = assess_tile_pyramid_windows(
        estimate, reference,
        ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511)))
    assert result[0][5][0][6] == pytest.approx(0.5, abs=1e-6)
    assert result[1][5][0][3:6] == (0, 511, 511)
    assert result[1][5][0][6] == 3.0


def test_inputs_not_modified():
    estimate = ((_tile(tx=0, ty=0),
                 _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=3.0)),)
    reference = ((_tile(tx=0, ty=0, zmean=1.0),
                  _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=1.0)),)
    est_snapshot = tuple(tuple(t) for t in estimate[0])
    ref_snapshot = tuple(tuple(t) for t in reference[0])
    windows = ((0, 0, 0, 600, 255), (0, 0, 0, 0, 0))
    assess_tile_pyramid_windows(estimate, reference, windows)
    assert tuple(tuple(t) for t in estimate[0]) == est_snapshot
    assert tuple(tuple(t) for t in reference[0]) == ref_snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows(bad, (), ())
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), bad, ())
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), (), bad)


@pytest.mark.parametrize("bad_window", [
    [0, 0, 0, 1, 1],
    None,
    42,
])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), (), (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), (), ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), (), ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        assess_tile_pyramid_windows((), (), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: pyramid structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(([_tile()],), ((_tile(),),), ())


@pytest.mark.parametrize("which", ["estimate", "reference"])
def test_wrong_field_count_raises_value_error(which):
    bad = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1.5, 0.5),),)
    good = ((_tile(),),)
    args = (bad, good, ()) if which == "estimate" else (good, bad, ())
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(*args)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_non_int_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1.5
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(bad, good, ())
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(good, bad, ())


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_non_float_statistics_raise_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(bad, good, ())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_statistics_raise_value_error(bad_value):
    tile = list(_tile())
    tile[8] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(bad, good, ())


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_nonpositive_zsigma_raises_value_error(bad_sigma):
    bad = ((_tile(zsigma=bad_sigma),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(bad, good, ())


def test_inverted_tile_bounds_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            ((_tile(ix0=256, ix1=200),),), ((_tile(),),), ())
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            ((_tile(),),), ((_tile(iy0=256, iy1=200),),), ())


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(unsorted, unsorted, ())

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(duplicate, duplicate, ())


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            ((),), ((),), ((level, 0, 0, 1, 1),))


def test_level_checked_against_both_pyramids():
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            ((_tile(),),), (), ((0, 0, 0, 255, 255),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            (), ((_tile(),),), ((0, 0, 0, 255, 255),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(((),), ((),), ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(((),), ((),), ((0, 0, 5, 1, 1),))


# ---------------------------------------------------------------------------
# ValueError: matched tiles disagree on bounds
# ---------------------------------------------------------------------------

def test_same_coordinate_mismatched_bounds_in_window_raise_value_error():
    estimate = ((_tile(ix1=200),),)
    reference = ((_tile(ix1=255),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_windows(
            estimate, reference, ((0, 0, 0, 255, 255),))
