"""Tests for :func:`lidar_scan.assess_tile_pyramid_stats`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import assess_tile_pyramid_stats, build_tile_pyramid_stats
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
    assert tiles_module.assess_tile_pyramid_stats is assess_tile_pyramid_stats
    import lidar_scan
    assert "assess_tile_pyramid_stats" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_double_empty_returns_empty_tuple():
    assert assess_tile_pyramid_stats((), ()) == ()


def test_empty_levels_preserved():
    estimate = ((), ())
    reference = ((), ())
    assert assess_tile_pyramid_stats(estimate, reference) == ((), ())


def test_single_level_single_tile_metrics():
    estimate = ((_tile(zmean=12.0, zsigma=3.0, count=2),),)
    reference = ((_tile(zmean=9.0, zsigma=4.0, count=5),),)
    result = assess_tile_pyramid_stats(estimate, reference)
    assert len(result) == 1
    assert result[0] == (
        (0, 0, 0, 0, 255, 255, 3.0, 3.0, 5.0, 0.6, -3),
    )


def test_multiple_levels_in_level_order():
    estimate = (
        (_tile(zmean=2.0, zsigma=1.0),),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511,
               zmean=4.0, zsigma=1.0),),
    )
    reference = (
        (_tile(zmean=1.0, zsigma=1.0, count=8),),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511,
               zmean=1.0, zsigma=1.0, count=9),),
    )
    result = assess_tile_pyramid_stats(estimate, reference)
    assert len(result) == 2
    assert result[0][0][6] == 1.0
    assert result[0][0][10] == -7
    assert result[1][0][6] == 3.0
    assert result[1][0][10] == -8
    assert result[1][0][2:6] == (0, 0, 511, 511)


def test_negative_bias_gives_negative_z_score():
    estimate = ((_tile(zmean=9.0, zsigma=3.0),),)
    reference = ((_tile(zmean=12.0, zsigma=4.0),),)
    row = assess_tile_pyramid_stats(estimate, reference)[0][0]
    bias, abs_error, combined, z_score, count_delta = row[6:11]
    assert bias == -3.0
    assert abs_error == 3.0
    assert combined == pytest.approx(5.0, abs=1e-6)
    assert z_score == pytest.approx(-0.6, abs=1e-6)
    assert count_delta == 0


def test_multiple_tiles_preserve_input_order():
    estimate = (
        (
            _tile(tx=0, ty=0, zmean=2.0, zsigma=1.0),
            _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=4.0, zsigma=1.0),
        ),
    )
    reference = (
        (
            _tile(tx=0, ty=0, zmean=1.0, zsigma=1.0, count=8),
            _tile(tx=1, ty=0, ix0=256, ix1=511, zmean=1.0, zsigma=1.0, count=9),
        ),
    )
    result = assess_tile_pyramid_stats(estimate, reference)
    assert [(row[0], row[1]) for row in result[0]] == [(0, 0), (1, 0)]
    assert result[0][0][6] == 1.0
    assert result[0][1][6] == 3.0


def test_zero_bias_normalized_to_positive_zero():
    estimate = ((_tile(zmean=10.0, zsigma=2.0, count=1),),)
    reference = ((_tile(zmean=10.0, zsigma=2.0, count=7),),)
    row = assess_tile_pyramid_stats(estimate, reference)[0][0]
    bias, abs_error, combined, z_score, count_delta = row[6:11]
    assert bias == 0.0 and math.copysign(1.0, bias) == 1.0
    assert abs_error == 0.0
    assert combined == pytest.approx(math.sqrt(8.0), abs=1e-6)
    assert z_score == 0.0 and math.copysign(1.0, z_score) == 1.0
    assert count_delta == -6


def test_quantized_to_six_decimal_places():
    estimate = ((_tile(zmean=1 / 3, zsigma=1.0),),)
    reference = ((_tile(zmean=0.0, zsigma=1.0),),)
    row = assess_tile_pyramid_stats(estimate, reference)[0][0]
    bias, _, combined, z_score, _ = row[6:11]
    assert bias == 0.333333
    assert combined == pytest.approx(math.sqrt(2.0), abs=5e-7)
    assert z_score == pytest.approx((1 / 3) / math.sqrt(2.0), abs=5e-7)


def test_inputs_not_modified():
    estimate = (
        (_tile(tx=0, ty=0, zmean=2.0, count=3),
         _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511,
               zmean=3.0, count=4)),
    )
    reference = (
        (_tile(tx=0, ty=0, zmean=1.0, count=2),
         _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511,
               zmean=0.0, count=5)),
    )
    est_snapshot = tuple(tuple(t) for t in estimate[0])
    ref_snapshot = tuple(tuple(t) for t in reference[0])
    assess_tile_pyramid_stats(estimate, reference)
    assert tuple(estimate[0]) == est_snapshot
    assert tuple(reference[0]) == ref_snapshot


def test_agrees_with_built_pyramids():
    est_points = [(0.0, 0.0, 12.0, 1, 3.0), (300.0, 0.0, 14.0, 1, 3.0)]
    ref_points = [(0.0, 0.0, 9.0, 1, 4.0), (300.0, 0.0, 8.0, 1, 4.0)]
    estimate = build_tile_pyramid_stats(est_points, levels=2)
    reference = build_tile_pyramid_stats(ref_points, levels=2)
    result = assess_tile_pyramid_stats(estimate, reference)
    assert len(result) == 2
    # At level 1 both points fall in the same tile, so its level-1 row is the
    # only entry; metrics are finite and count_delta is 0 (2 - 2).
    level1 = result[1]
    assert len(level1) == 1
    assert level1[0][10] == 0
    assert math.isfinite(level1[0][9])


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        assess_tile_pyramid_stats(bad, ())
    with pytest.raises(TypeError):
        assess_tile_pyramid_stats((), bad)


# ---------------------------------------------------------------------------
# ValueError: per-input structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["estimate", "reference"])
def test_level_not_tuple_raises_value_error(which):
    bad = ([_tile()],)
    good = ((_tile(),),)
    args = (bad, good) if which == "estimate" else (good, bad)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(*args)


@pytest.mark.parametrize("which", ["estimate", "reference"])
def test_wrong_field_count_raises_value_error(which):
    bad = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1.5, 0.5),),)
    good = ((_tile(),),)
    args = (bad, good) if which == "estimate" else (good, bad)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(*args)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_non_int_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1.5
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(good, bad)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_bool_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = True
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, good)


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_non_float_statistics_raise_value_error(pos):
    tile = list(_tile())
    tile[pos] = 1
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(good, bad)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_statistics_raise_value_error(bad_value):
    tile = list(_tile())
    tile[8] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(good, bad)


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_nonpositive_zsigma_raises_value_error(bad_sigma):
    bad = ((_tile(zsigma=bad_sigma),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(good, bad)


def test_inverted_tile_bounds_raise_value_error():
    good = ((_tile(),),)
    bad_x = ((_tile(ix1=254),),)
    bad_y = ((_tile(iy1=254),),)
    for bad in (bad_x, bad_y):
        with pytest.raises(ValueError):
            assess_tile_pyramid_stats(bad, good)
        with pytest.raises(ValueError):
            assess_tile_pyramid_stats(good, bad)


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    good = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    bad = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(bad, bad)

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(duplicate, duplicate)


# ---------------------------------------------------------------------------
# ValueError: cross-input agreement
# ---------------------------------------------------------------------------

def test_different_level_counts_raise_value_error():
    estimate = ((_tile(),),)
    reference = ((_tile(),), (_tile(ix0=0, iy0=0, ix1=511, iy1=511),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(reference, estimate)


def test_different_lengths_raise_value_error():
    estimate = ((_tile(),),)
    reference = ((_tile(), _tile(tx=1, ty=0, ix0=256, ix1=511)),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(reference, estimate)


def test_one_empty_one_nonempty_level_raises_value_error():
    good = ((_tile(),),)
    empty = ((),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(good, empty)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(empty, good)


def test_mismatched_coordinates_raise_value_error():
    estimate = ((_tile(tx=0, ty=0),),)
    reference = ((_tile(tx=0, ty=1),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)


def test_mismatched_bounds_raise_value_error():
    estimate = ((_tile(ix1=255),),)
    reference = ((_tile(ix1=200),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)
    estimate = ((_tile(iy1=255),),)
    reference = ((_tile(iy1=200),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)


def test_different_counts_are_allowed():
    estimate = ((_tile(count=3),),)
    reference = ((_tile(count=99),),)
    result = assess_tile_pyramid_stats(estimate, reference)
    assert result[0][0][10] == -96


def test_mismatch_on_second_level_only_raises_value_error():
    estimate = (
        (_tile(),),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511),),
    )
    reference = (
        (_tile(),),
        (_tile(tx=1, ty=0, ix0=512, iy0=0, ix1=1023, iy1=511),),
    )
    with pytest.raises(ValueError):
        assess_tile_pyramid_stats(estimate, reference)
