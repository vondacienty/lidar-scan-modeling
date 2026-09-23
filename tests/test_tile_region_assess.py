"""Tests for :func:`lidar_scan.tiles.assess_tile_region_stats`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import assess_tile_region_stats
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

def test_module_and_package_exports():
    assert tiles_module.assess_tile_region_stats is assess_tile_region_stats
    import lidar_scan
    assert "assess_tile_region_stats" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_empty_pair_returns_empty_tuple():
    assert assess_tile_region_stats((), ()) == ()


def test_single_tile_metrics():
    estimate = (_tile(zmean=12.0, zsigma=3.0, count=2),)
    reference = (_tile(zmean=9.0, zsigma=4.0, count=5),)
    result = assess_tile_region_stats(estimate, reference)
    tx, ty, bias, abs_error, combined, z_score = result[0]
    assert (tx, ty) == (0, 0)
    assert bias == 3.0
    assert abs_error == 3.0
    assert combined == pytest.approx(5.0, abs=1e-6)
    assert z_score == pytest.approx(0.6, abs=1e-6)


def test_negative_bias_gives_negative_z_score():
    estimate = (_tile(zmean=9.0, zsigma=3.0),)
    reference = (_tile(zmean=12.0, zsigma=4.0),)
    _, _, bias, abs_error, _, z_score = assess_tile_region_stats(
        estimate, reference)[0]
    assert bias == -3.0
    assert abs_error == 3.0
    assert z_score == pytest.approx(-0.6, abs=1e-6)


def test_zero_bias_normalized_to_positive_zero():
    estimate = (_tile(zmean=10.0, zsigma=2.0),)
    reference = (_tile(zmean=10.0, zsigma=2.0),)
    _, _, bias, abs_error, combined, z_score = assess_tile_region_stats(
        estimate, reference)[0]
    assert bias == 0.0 and math.copysign(1.0, bias) == 1.0
    assert abs_error == 0.0
    assert combined == pytest.approx(math.sqrt(8.0), abs=1e-6)
    assert z_score == 0.0 and math.copysign(1.0, z_score) == 1.0


def test_multiple_tiles_preserve_order():
    estimate = (
        _tile(tx=-1, ty=5, ix0=-256, iy0=1280, ix1=-1, iy1=1535,
              zmean=2.0, zsigma=1.0),
        _tile(tx=0, ty=0, zmean=3.0, zsigma=1.0),
        _tile(tx=2, ty=0, ix0=512, ix1=767, zmean=4.0, zsigma=1.0),
    )
    reference = tuple(
        _tile(tx=t[0], ty=t[1], ix0=t[2], iy0=t[3], ix1=t[4], iy1=t[5],
              zmean=0.0, zsigma=1.0, count=9)
        for t in estimate
    )
    result = assess_tile_region_stats(estimate, reference)
    assert [(row[0], row[1]) for row in result] == [(-1, 5), (0, 0), (2, 0)]
    assert [row[2] for row in result] == [2.0, 3.0, 4.0]


def test_count_may_differ():
    estimate = (_tile(count=3),)
    reference = (_tile(count=9999),)
    result = assess_tile_region_stats(estimate, reference)
    assert result[0][2] == 0.0


def test_zmin_zmax_are_not_compared():
    # Only coordinates, bounds, zmean and zsigma matter; zmin/zmax differ.
    estimate = (_tile(zmin=-99.0, zmax=99.0, zmean=1.0, zsigma=1.0),)
    reference = (_tile(zmin=0.0, zmax=2.0, zmean=0.0, zsigma=1.0),)
    result = assess_tile_region_stats(estimate, reference)
    assert result[0][2] == 1.0


def test_quantized_to_six_decimal_places():
    estimate = (_tile(zmean=1 / 3, zsigma=1.0),)
    reference = (_tile(zmean=0.0, zsigma=1.0),)
    _, _, bias, _, combined, z_score = assess_tile_region_stats(
        estimate, reference)[0]
    assert bias == 0.333333
    assert combined == pytest.approx(math.sqrt(2.0), abs=5e-7)
    assert z_score == pytest.approx((1 / 3) / math.sqrt(2.0), abs=5e-7)


def test_half_even_rounding():
    # bias = 0.0000005 exactly -> ROUND_HALF_EVEN -> 0.000000
    estimate = (_tile(zmean=0.0000005, zsigma=1.0),)
    reference = (_tile(zmean=0.0, zsigma=1.0),)
    _, _, bias, _, _, _ = assess_tile_region_stats(estimate, reference)[0]
    assert bias == 0.0


def test_inputs_not_modified():
    estimate = (_tile(tx=0, zmean=2.0, zsigma=2.0, count=3),
                _tile(tx=1, ix0=256, ix1=511, zmean=1.0, zsigma=1.0))
    reference = (_tile(tx=0, zmean=0.0, zsigma=1.0, count=4),
                 _tile(tx=1, ix0=256, ix1=511, zmean=0.0, zsigma=1.0))
    est_snapshot = tuple(t for t in estimate)
    ref_snapshot = tuple(t for t in reference)
    assess_tile_region_stats(estimate, reference)
    assert estimate == est_snapshot
    assert reference == ref_snapshot


def test_result_is_tuple_of_tuples():
    result = assess_tile_region_stats((_tile(),), (_tile(),))
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 6


# ---------------------------------------------------------------------------
# mismatched coordinates / bounds
# ---------------------------------------------------------------------------

def test_length_mismatch_raises_value_error():
    estimate = (_tile(tx=0), _tile(tx=1, ix0=256, ix1=511))
    reference = (_tile(tx=0),)
    with pytest.raises(ValueError):
        assess_tile_region_stats(estimate, reference)
    with pytest.raises(ValueError):
        assess_tile_region_stats(reference, estimate)


def test_mismatched_coordinates_raise_value_error():
    estimate = (_tile(tx=0, ty=1),)
    reference = (_tile(tx=0, ty=2, iy0=256, iy1=511),)
    with pytest.raises(ValueError):
        assess_tile_region_stats(estimate, reference)


def test_mismatched_bounds_raise_value_error():
    good = (_tile(),)
    for overrides in (dict(ix0=1), dict(iy0=1), dict(ix1=254), dict(iy1=254)):
        bad = (_tile(**overrides),)
        with pytest.raises(ValueError):
            assess_tile_region_stats(bad, good)
        with pytest.raises(ValueError):
            assess_tile_region_stats(good, bad)


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------

def test_non_tuple_input_raises_type_error():
    good = (_tile(),)
    for bad in ([], [_tile()], None, 42, ""):
        with pytest.raises(TypeError):
            assess_tile_region_stats(bad, good)
        with pytest.raises(TypeError):
            assess_tile_region_stats(good, bad)


def test_non_tuple_element_raises_value_error():
    good = (_tile(),)
    with pytest.raises(ValueError):
        assess_tile_region_stats(([_tile()],), good)
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, (["abc"],))


def test_wrong_field_count_raises_value_error():
    good = (_tile(),)
    with pytest.raises(ValueError):
        assess_tile_region_stats(((_tile()[:10]),), good)
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, (_tile() + (9,),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_non_int_index_or_count_raises_value_error(pos):
    good = (_tile(),)
    fields = list(_tile())
    fields[pos] = 1.5
    with pytest.raises(ValueError):
        assess_tile_region_stats((tuple(fields),), good)
    fields[pos] = "1"
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, (tuple(fields),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_bool_index_or_count_rejected(pos):
    good = (_tile(),)
    fields = list(_tile())
    fields[pos] = True
    with pytest.raises(ValueError):
        assess_tile_region_stats((tuple(fields),), good)


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_non_float_statistic_raises_value_error(pos):
    good = (_tile(),)
    fields = list(_tile())
    fields[pos] = 1
    with pytest.raises(ValueError):
        assess_tile_region_stats((tuple(fields),), good)
    fields[pos] = "1.0"
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, (tuple(fields),))


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_nonfinite_statistic_raises_value_error(pos):
    good = (_tile(),)
    for bad in float("nan"), float("inf"), float("-inf"):
        fields = list(_tile())
        fields[pos] = bad
        with pytest.raises(ValueError):
            assess_tile_region_stats((tuple(fields),), good)
        with pytest.raises(ValueError):
            assess_tile_region_stats(good, (tuple(fields),))


def test_non_positive_zsigma_raises_value_error():
    good = (_tile(),)
    with pytest.raises(ValueError):
        assess_tile_region_stats((_tile(zsigma=0.0),), good)
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, (_tile(zsigma=-1.5),))


def test_unsorted_tiles_raise_value_error():
    tiles = (_tile(tx=1, ix0=256, ix1=511), _tile(tx=0))
    with pytest.raises(ValueError):
        assess_tile_region_stats(tiles, tiles)


def test_duplicate_coordinates_raise_value_error():
    tiles = (_tile(), _tile())
    with pytest.raises(ValueError):
        assess_tile_region_stats(tiles, tiles)


def test_reference_still_validated_when_estimate_empty():
    good = (_tile(),)
    structurally_bad = ((1, 2, 3),)
    with pytest.raises(ValueError):
        assess_tile_region_stats((), structurally_bad)
    with pytest.raises(ValueError):
        assess_tile_region_stats(structurally_bad, ())
    # Structurally valid but non-empty against an empty input: the tile
    # correspondence check must still raise.
    with pytest.raises(ValueError):
        assess_tile_region_stats((), good)
    with pytest.raises(ValueError):
        assess_tile_region_stats(good, ())
