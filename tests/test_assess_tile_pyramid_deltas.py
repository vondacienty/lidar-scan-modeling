"""Tests for :func:`lidar_scan.assess_tile_pyramid_deltas`."""

from __future__ import annotations

import copy
import math

import pytest

from lidar_scan import assess_tile_pyramid_deltas
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.assess_tile_pyramid_deltas is assess_tile_pyramid_deltas
    import lidar_scan
    assert "assess_tile_pyramid_deltas" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_double_empty_returns_empty_tuple():
    assert assess_tile_pyramid_deltas((), ()) == ()


def test_empty_levels_are_preserved():
    assert assess_tile_pyramid_deltas(((), (), ()), ((), (), ())) == \
        ((), (), ())


def test_mixed_empty_and_populated_levels():
    estimate = ((), (_tile(),))
    reference = ((), (_tile(zmin=2.0, zmax=3.0, count=4),))
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result[0] == ()
    assert result[1] == ((0, 0, 0, 0, 255, 255, -1.0, -1.0, -3),)


def test_single_tile_deltas():
    estimate = ((_tile(zmin=12.0, zmax=20.0, count=8),),)
    reference = ((_tile(zmin=10.0, zmax=22.0, count=5),),)
    assert assess_tile_pyramid_deltas(estimate, reference) == \
        (((0, 0, 0, 0, 255, 255, 2.0, -2.0, 3),),)


def test_result_echoes_coordinates_and_cell_bounds():
    estimate = ((_tile(tx=2, ty=-3, ix0=512, iy0=-768, ix1=767,
                       iy1=-513, zmin=4.0, zmax=5.0, count=2),),)
    reference = ((_tile(tx=2, ty=-3, ix0=512, iy0=-768, ix1=767,
                        iy1=-513, zmin=1.0, zmax=2.0, count=9),),)
    tile = assess_tile_pyramid_deltas(estimate, reference)[0][0]
    assert tile[:6] == (2, -3, 512, -768, 767, -513)
    assert tile[6:] == (3.0, 3.0, -7)


def test_level_order_and_tile_order_follow_estimate():
    estimate = (
        (_tile(tx=0, ty=0, zmin=2.0, zmax=4.0),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0)),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, zmin=9.0, zmax=9.0),),
    )
    reference = (
        (_tile(tx=0, ty=0, zmin=1.0, zmax=4.0),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=1.0, zmax=5.0)),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, zmin=4.0, zmax=9.0),),
    )
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert [t[:2] for t in result[0]] == [(0, 0), (1, 0)]
    assert [t[6] for t in result[0]] == [1.0, 2.0]
    assert result[1][0][6] == 5.0


def test_count_delta_is_exact_int_even_for_huge_values():
    huge = 10 ** 40
    result = assess_tile_pyramid_deltas(
        ((_tile(count=huge),),), ((_tile(count=7),),))
    dcount = result[0][0][8]
    assert dcount == huge - 7
    assert isinstance(dcount, int)


def test_zero_float_delta_normalized_to_positive_zero():
    tile = assess_tile_pyramid_deltas(
        ((_tile(zmin=-0.0, zmax=-0.0),),),
        ((_tile(zmin=0.0, zmax=0.0),),))[0][0]
    assert tile[6] == 0.0 and math.copysign(1.0, tile[6]) == 1.0
    assert tile[7] == 0.0 and math.copysign(1.0, tile[7]) == 1.0


def test_quantized_to_six_decimal_places_with_half_even_rounding():
    # 1.0000015 - 0 -> 1.000002 (round half to even); 1.0000005 -> 1.000000
    estimate = ((_tile(zmin=1.0000005, zmax=1.0000015),),)
    reference = ((_tile(zmin=0.0, zmax=0.0),),)
    tile = assess_tile_pyramid_deltas(estimate, reference)[0][0]
    assert tile[6] == 1.0
    assert tile[7] == 1.000002


def test_decimal_str_conversion_avoids_binary_artifact():
    estimate = ((_tile(zmin=0.3, zmax=0.3),),)
    reference = ((_tile(zmin=0.1, zmax=0.2),),)
    tile = assess_tile_pyramid_deltas(estimate, reference)[0][0]
    # Plain float subtraction would yield 0.09999999999999998 / 0.199999...9
    assert tile[6] == 0.2
    assert tile[7] == 0.1


def test_tiles_are_paired_by_coordinate_not_position():
    # Both levels are valid strictly-sorted sequences with identical key sets,
    # so positional zip and coordinate pairing coincide; an unsorted side is
    # rejected outright (see validation tests) rather than silently mispaired.
    estimate = ((_tile(tx=0, ty=0, zmin=1.5, zmax=3.0),
                 _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=2.0, zmax=9.0)),)
    reference = ((_tile(tx=0, ty=0, zmin=1.0, zmax=3.0),
                  _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=2.0, zmax=2.5)),)
    assert [t[6:8] for t in assess_tile_pyramid_deltas(
        estimate, reference)[0]] == [(0.5, 0.0), (0.0, 6.5)]


def test_inputs_are_not_modified_and_calls_are_stable():
    estimate = (
        (_tile(tx=0, ty=0, zmin=1.5, zmax=3.0, count=4),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),
    )
    reference = (
        (_tile(tx=0, ty=0, zmin=1.0, zmax=3.0, count=1),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=2.0, zmax=5.0, count=9)),
    )
    est_before = copy.deepcopy(estimate)
    ref_before = copy.deepcopy(reference)
    first = assess_tile_pyramid_deltas(estimate, reference)
    second = assess_tile_pyramid_deltas(estimate, reference)
    assert first == second
    assert estimate == est_before
    assert reference == ref_before


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        assess_tile_pyramid_deltas(bad, ())
    with pytest.raises(TypeError):
        assess_tile_pyramid_deltas((), bad)


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(([_tile()],), ((_tile(),),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((_tile(),),), ([_tile()],))


@pytest.mark.parametrize("which", ["estimate", "reference"])
def test_tile_wrong_container_or_length_raises_value_error(which):
    good = ((_tile(),),)
    bad_container = ((list(_tile()),),)
    bad_short = (((0, 0, 0, 0, 255, 255, 1.0, 2.0),),)
    bad_long = ((_tile() + (99,),),)
    for bad in (bad_container, bad_short, bad_long):
        args = (bad, good) if which == "estimate" else (good, bad)
        with pytest.raises(ValueError):
            assess_tile_pyramid_deltas(*args)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_integer_fields_wrong_type_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(good, bad)


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, True, "1.0", None])
def test_z_fields_must_be_floats(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, good)


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_z_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    bad = ((tuple(tile),),)
    good = ((_tile(),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, good)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(good, bad)


def test_unsorted_coordinates_raise_value_error():
    unsorted = ((_tile(tx=1, ty=0, ix0=256, ix1=511), _tile(tx=0, ty=0)),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(unsorted, unsorted)


def test_duplicate_coordinates_raise_value_error():
    duplicate = ((_tile(tx=0, ty=0),
                  _tile(tx=0, ty=0, ix0=256, ix1=511)),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(duplicate, duplicate)


def test_inverted_cell_bounds_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((_tile(ix0=256, ix1=200),),),
                                   ((_tile(),),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((_tile(),),),
                                   ((_tile(iy0=256, iy1=200),),))


def test_inverted_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(zmin=2.0, zmax=1.0),),), ((_tile(),),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(),),), ((_tile(zmin=2.0, zmax=1.0),),))


def test_equal_z_bounds_are_allowed():
    result = assess_tile_pyramid_deltas(
        ((_tile(zmin=1.0, zmax=1.0),),),
        ((_tile(zmin=1.0, zmax=1.0),),))
    assert result[0][0][6:8] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# ValueError: estimate/reference agreement
# ---------------------------------------------------------------------------

def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(),),), ((), ()))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            (), ((_tile(),),))


def test_different_coordinate_sets_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(tx=0, ty=0),),),
            ((_tile(tx=1, ty=0, ix0=256, ix1=511),),))
    with pytest.raises(ValueError):  # tile present only on one side
        assess_tile_pyramid_deltas(
            ((_tile(tx=0, ty=0), _tile(tx=1, ty=0, ix0=256, ix1=511)),),
            ((_tile(tx=0, ty=0),),))


def test_matching_coordinates_with_different_bounds_raise_value_error():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(ix1=200),),), ((_tile(ix1=255),),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(
            ((_tile(iy0=1, iy1=255),),), ((_tile(),),))


def test_identical_coords_and_bounds_with_different_count_are_allowed():
    result = assess_tile_pyramid_deltas(
        ((_tile(count=3),),), ((_tile(count=10),),))
    assert result[0][0][8] == -7


def test_coordinate_mismatch_at_second_level_raises_value_error():
    estimate = (
        (_tile(),),
        (_tile(tx=0, ty=0),),
    )
    reference = (
        (_tile(),),
        (_tile(tx=1, ty=0, ix0=256, ix1=511),),
    )
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, reference)
