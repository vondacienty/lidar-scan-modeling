"""Tests for :func:`lidar_scan.summarize_tile_pyramid_windows`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import summarize_tile_pyramid_windows
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
    assert tiles_module.summarize_tile_pyramid_windows is (
        summarize_tile_pyramid_windows)
    import lidar_scan
    assert "summarize_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert summarize_tile_pyramid_windows((), (), ()) == ()
    assert summarize_tile_pyramid_windows(((),), ((),), ()) == ()


def test_window_without_matches_gets_none_summary():
    result = summarize_tile_pyramid_windows(
        ((),), ((),), ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, None),)


def test_window_covering_no_shared_tile_gets_none_summary():
    pyramid = ((_tile(), _tile(tx=1, ty=0, ix0=256, ix1=511)),)
    result = summarize_tile_pyramid_windows(
        pyramid, pyramid, ((0, 600, 0, 700, 700),))
    assert result == ((0, 600, 0, 700, 700, None),)


def test_single_match_summary():
    # b = 12 - 9 = 3; a = 3; combined sigma = 5; q = 0.6; d = 2 - 5 = -3
    estimate = ((_tile(zmean=12.0, zsigma=3.0, count=2),),)
    reference = ((_tile(zmean=9.0, zsigma=4.0, count=5),),)
    result = summarize_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))
    assert result == ((0, 0, 0, 255, 255, (3.0, 3.0, 3.0, 0.6, -3)),)


def test_window_echoes_its_arguments():
    empty_level = ()
    estimate = (empty_level, empty_level, (_tile(),))
    reference = (empty_level, empty_level, (_tile(),))
    result = summarize_tile_pyramid_windows(
        estimate, reference, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    pyramid = ((_tile(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 300, 300, 400, 400),
               (0, 256, 0, 300, 255))
    result = summarize_tile_pyramid_windows(pyramid, pyramid, windows)
    assert tuple(window[:5] for window in result) == windows
    assert result[0][5] is not None
    assert result[1][5] is None
    assert result[2][5] is None


def test_only_tiles_intersecting_the_window_are_summarized():
    pyramid = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=2.0),
    ),)
    result = summarize_tile_pyramid_windows(
        pyramid, pyramid, ((0, 256, 0, 300, 255),))
    # Only tile (1, 0) with zmean 2.0 on both sides: b = 0, d = 0.
    assert result[0][5] == (0.0, 0.0, 0.0, 0.0, 0)


def test_closed_intervals_match_at_endpoints():
    pyramid = ((_tile(),),)
    result = summarize_tile_pyramid_windows(
        pyramid, pyramid, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (0.0, 0.0, 0.0, 0.0, 0)
    result = summarize_tile_pyramid_windows(
        pyramid, pyramid, ((0, 256, 0, 300, 255),))
    assert result[0][5] is None


def test_multiple_matches_aggregate():
    estimate = ((
        _tile(tx=0, ty=0, zmean=2.0, zsigma=1.0, count=2),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=4.0, zsigma=1.0, count=4),
    ),)
    reference = ((
        _tile(tx=0, ty=0, zmean=1.0, zsigma=1.0, count=7),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=1.0, zsigma=1.0, count=9),
    ),)
    # b = (1, 3): min 1, max 3, mean(a) = 2;
    # q = (1/sqrt(2), 3/sqrt(2)): rms = sqrt((0.5 + 4.5) / 2) = sqrt(2.5);
    # sum(d) = (2 - 7) + (4 - 9) = -10
    summary = summarize_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 600, 600),))[0][5]
    assert summary[0:3] == (1.0, 3.0, 2.0)
    assert summary[3] == pytest.approx(math.sqrt(2.5), abs=5e-7)
    assert summary[4] == -10


def test_bias_min_and_max_span_negative_and_positive():
    estimate = ((
        _tile(tx=0, ty=0, zmean=12.0, zsigma=3.0, count=2),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmean=6.0, zsigma=3.0, count=2),
    ),)
    reference = ((
        _tile(tx=0, ty=0, zmean=9.0, zsigma=4.0, count=5),
        _tile(tx=0, ty=1, iy0=256, iy1=511, zmean=9.0, zsigma=4.0, count=5),
    ),)
    # b = (3, -3): q = (0.6, -0.6); sum(d) = -3 + -3 = -6
    summary = summarize_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 600, 600),))[0][5]
    assert summary == (-3.0, 3.0, 3.0, 0.6, -6)


def test_count_delta_sum_is_exact_int():
    estimate = ((
        _tile(tx=0, ty=0, count=10**40),
        _tile(tx=0, ty=1, iy0=256, iy1=511, count=3),
    ),)
    reference = ((
        _tile(tx=0, ty=0, count=1),
        _tile(tx=0, ty=1, iy0=256, iy1=511, count=10**40 + 2),
    ),)
    summary = summarize_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 600, 600),))[0][5]
    assert isinstance(summary[4], int)
    assert summary[4] == 0


def test_zero_bias_normalized_to_positive_zero():
    pyramid = ((_tile(zmean=10.0, zsigma=2.0),),)
    summary = summarize_tile_pyramid_windows(
        pyramid, pyramid, ((0, 0, 0, 255, 255),))[0][5]
    for value in summary[0:4]:
        assert isinstance(value, float)
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0
    assert summary[4] == 0


def test_quantized_to_six_decimal_places():
    estimate = ((_tile(zmean=1 / 3, zsigma=1.0),),)
    reference = ((_tile(zmean=0.0, zsigma=1.0),),)
    summary = summarize_tile_pyramid_windows(
        estimate, reference, ((0, 0, 0, 255, 255),))[0][5]
    assert summary[0] == 0.333333
    assert summary[1] == 0.333333
    assert summary[2] == 0.333333
    assert summary[3] == pytest.approx((1 / 3) / math.sqrt(2.0), abs=5e-7)


def test_multiple_levels_select_the_right_level():
    estimate = (
        (_tile(),),
        (_tile(ix1=511, iy1=511, zmean=4.0, zsigma=1.0),),
    )
    reference = (
        (_tile(zmean=1.0, zsigma=1.0),),
        (_tile(ix1=511, iy1=511, zmean=1.0, zsigma=1.0),),
    )
    result = summarize_tile_pyramid_windows(
        estimate, reference,
        ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511)))
    # Level 0: b = 1.5 - 1.0 = 0.5, combined sigma = sqrt(0.5**2 + 1.0**2).
    assert result[0][5] == (
        0.5, 0.5, 0.5,
        pytest.approx(0.5 / math.sqrt(1.25), abs=5e-7), 0)
    assert result[1][:5] == (1, 0, 0, 511, 511)
    assert result[1][5][0:3] == (3.0, 3.0, 3.0)


def test_inputs_not_modified():
    estimate = ((_tile(tx=0, ty=0),
                 _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=3.0)),)
    reference = ((_tile(tx=0, ty=0, zmean=1.0),
                  _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=1.0)),)
    est_snapshot = tuple(tuple(t) for t in estimate[0])
    ref_snapshot = tuple(tuple(t) for t in reference[0])
    windows = ((0, 0, 0, 600, 255), (0, 0, 0, 0, 0))
    summarize_tile_pyramid_windows(estimate, reference, windows)
    assert tuple(tuple(t) for t in estimate[0]) == est_snapshot
    assert tuple(tuple(t) for t in reference[0]) == ref_snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows(bad, (), ())
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), bad, ())
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), (), bad)


@pytest.mark.parametrize("bad_window", [
    [0, 0, 0, 1, 1],
    None,
    42,
])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), (), (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), (), ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), (), ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        summarize_tile_pyramid_windows((), (), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: pyramid structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(([_tile()],), ((_tile(),),), ())


@pytest.mark.parametrize("which", ["estimate", "reference"])
def test_wrong_field_count_raises_value_error(which):
    bad = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1.5, 0.5),),)
    good = ((_tile(),),)
    args = (bad, good, ()) if which == "estimate" else (good, bad, ())
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(*args)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_non_int_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1.5
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(bad, good, ())
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(good, bad, ())


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_non_float_statistics_raise_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(bad, good, ())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_statistics_raise_value_error(bad_value):
    tile = list(_tile())
    tile[8] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(bad, good, ())


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_nonpositive_zsigma_raises_value_error(bad_sigma):
    bad = ((_tile(zsigma=bad_sigma),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(bad, good, ())


def test_inverted_tile_bounds_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(ix0=256, ix1=200),),), ((_tile(),),), ())
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(),),), ((_tile(iy0=256, iy1=200),),), ())


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(unsorted, unsorted, ())

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(duplicate, duplicate, ())


# ---------------------------------------------------------------------------
# ValueError: side consistency
# ---------------------------------------------------------------------------

def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((), (_tile(),)), ((_tile(),),), ())


def test_different_tile_coordinates_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(tx=1, ty=0, ix0=256, ix1=511),),),
            ((_tile(),),), ())
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(), _tile(tx=0, ty=1, iy0=256, iy1=511)),),
            ((_tile(),),), ())


def test_mismatched_bounds_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(ix1=200),),), ((_tile(ix1=255),),), ())


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((),), ((),), ((level, 0, 0, 1, 1),))


def test_level_checked_against_both_pyramids():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            ((_tile(),),), (), ((0, 0, 0, 255, 255),))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(
            (), ((_tile(),),), ((0, 0, 0, 255, 255),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(((),), ((),), ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_windows(((),), ((),), ((0, 0, 5, 1, 1),))
