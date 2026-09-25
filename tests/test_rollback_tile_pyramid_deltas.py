"""Tests for :func:`lidar_scan.rollback_tile_pyramid_deltas`."""

from __future__ import annotations

import copy
import math

import pytest

from lidar_scan import rollback_tile_pyramid_deltas
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
    assert tiles_module.rollback_tile_pyramid_deltas is rollback_tile_pyramid_deltas
    import lidar_scan
    assert "rollback_tile_pyramid_deltas" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_double_empty_returns_empty_tuple():
    assert rollback_tile_pyramid_deltas((), ()) == ()


def test_empty_deltas_returns_base_identical_object():
    base = ((_tile(),),)
    assert rollback_tile_pyramid_deltas(base, ()) is base


def test_empty_levels_are_preserved():
    assert rollback_tile_pyramid_deltas(((), (), ()), ((), (), ())) == \
        ((), (), ())


def test_single_tile_rollback():
    base = ((_tile(zmin=10.0, zmax=20.0, count=8),),)
    deltas = ((_delta(dzmin=2.0, dzmax=-2.0, dcount=3),),)
    assert rollback_tile_pyramid_deltas(base, deltas) == \
        (((0, 0, 0, 0, 255, 255, 8, 22, 5),),)


def test_integral_values_are_exact_ints():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=10.0, zmax=20.0, count=8),),),
        ((_delta(dzmin=2.0, dzmax=-2.0, dcount=3),),))
    tile = result[0][0]
    assert isinstance(tile[6], int)
    assert isinstance(tile[7], int)
    assert isinstance(tile[8], int)


def test_non_integral_values_become_quantized_floats():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=0.3, zmax=0.5, count=1),),),
        ((_delta(dzmin=0.1, dzmax=0.2, dcount=0),),))
    tile = result[0][0]
    assert tile[6] == 0.2
    assert tile[7] == 0.3
    assert isinstance(tile[6], float)
    assert isinstance(tile[7], float)


def test_geometry_is_copied_unchanged():
    base = ((_tile(tx=2, ty=-3, ix0=512, iy0=-768, ix1=767, iy1=-513,
                   zmin=4.0, zmax=5.0, count=2),),)
    deltas = ((_delta(tx=2, ty=-3, ix0=512, iy0=-768, ix1=767, iy1=-513,
                      dzmin=1.0, dzmax=1.0, dcount=1),),)
    tile = rollback_tile_pyramid_deltas(base, deltas)[0][0]
    assert tile[:6] == (2, -3, 512, -768, 767, -513)
    assert tile[8] == 1


def test_level_and_tile_order_follow_base():
    base = (
        (_tile(tx=0, ty=0, zmin=2.0, zmax=4.0, count=4),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, zmin=9.0, zmax=9.0, count=3),),
    )
    deltas = (
        (_delta(dzmin=1.0, dzmax=0.0, dcount=1),
         _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=0.0, dzmax=2.0, dcount=1)),
        (_delta(ix0=0, iy0=0, ix1=511, iy1=511, dzmin=4.0, dzmax=0.0,
                dcount=2),),
    )
    result = rollback_tile_pyramid_deltas(base, deltas)
    assert [t[:2] for t in result[0]] == [(0, 0), (1, 0)]
    assert [t[6:9] for t in result[0]] == [(1.0, 4.0, 3), (3.0, 3.0, 1)]
    assert result[1][0][6:9] == (5, 9, 1)


def test_zero_float_result_normalized_to_positive_zero():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=1.0, zmax=1.0, count=1),),),
        ((_delta(dzmin=1.0, dzmax=1.0, dcount=0),),))[0][0]
    assert result[6] == 0.0 and math.copysign(1.0, result[6]) == 1.0
    assert result[7] == 0.0 and math.copysign(1.0, result[7]) == 1.0
    assert isinstance(result[6], int) and isinstance(result[7], int)


def test_quantized_to_six_decimal_places_with_half_even_rounding():
    # 1.0000005 - 0 -> 1.0 (round half to even); 1.0000015 -> 1.000002
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=1.0000005, zmax=1.0000015, count=1),),),
        ((_delta(dzmin=0.0, dzmax=0.0, dcount=0),),))[0][0]
    assert result[6] == 1.0
    assert result[7] == 1.000002


def test_decimal_str_conversion_avoids_binary_artifact():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=0.3, zmax=0.5, count=1),),),
        ((_delta(dzmin=0.1, dzmax=0.2, dcount=0),),))[0][0]
    assert result[6] == 0.2
    assert result[7] == 0.3


def test_count_is_exact_int_even_for_huge_values():
    huge = 10 ** 40
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=1.0, zmax=1.0, count=huge),),),
        ((_delta(dzmin=0.0, dzmax=0.0, dcount=7),),))[0][0]
    assert result[8] == huge - 7
    assert isinstance(result[8], int)


def test_inputs_are_not_modified_and_calls_are_stable():
    base = ((_tile(zmin=1.5, zmax=3.0, count=4),
             _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=2)),)
    deltas = ((_delta(dzmin=0.5, dzmax=0.0, dcount=1),
               _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=2.0,
                      dcount=1)),)
    base_before = copy.deepcopy(base)
    deltas_before = copy.deepcopy(deltas)
    first = rollback_tile_pyramid_deltas(base, deltas)
    second = rollback_tile_pyramid_deltas(base, deltas)
    assert first == second
    assert base == base_before
    assert deltas == deltas_before


def test_inverts_assess_tile_pyramid_deltas():
    from lidar_scan import assess_tile_pyramid_deltas, build_tile_pyramid

    estimate = build_tile_pyramid(
        [(0.0, 0.0, 3.0, 1, 1.0), (300.0, 0.0, 2.0, 1, 1.0)], levels=2)
    reference = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 1, 1.0), (300.0, 0.0, 5.0, 1, 1.0)], levels=2)
    deltas = assess_tile_pyramid_deltas(estimate, reference)
    assert rollback_tile_pyramid_deltas(estimate, deltas) == reference


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        rollback_tile_pyramid_deltas(bad, ())
    with pytest.raises(TypeError):
        rollback_tile_pyramid_deltas((), bad)


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(([_tile()],), ((_delta(),),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((_tile(),),), ([_delta()],))


@pytest.mark.parametrize("which", ["base", "deltas"])
def test_tile_wrong_container_or_length_raises_value_error(which):
    good_base = ((_tile(),),)
    good_deltas = ((_delta(),),)
    if which == "base":
        bads = (
            ((list(_tile()),),),
            (((0, 0, 0, 0, 255, 255, 1.0, 2.0),),),
            ((_tile() + (99,),),),
        )
        for bad in bads:
            with pytest.raises(ValueError):
                rollback_tile_pyramid_deltas(bad, good_deltas)
    else:
        bads = (
            ((list(_delta()),),),
            (((0, 0, 0, 0, 255, 255, 0.0, 0.0),),),
            ((_delta() + (99,),),),
        )
        for bad in bads:
            with pytest.raises(ValueError):
                rollback_tile_pyramid_deltas(good_base, bad)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_base_integer_fields_wrong_type_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((tuple(tile),),), ((_delta(),),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_delta_integer_fields_wrong_type_raise_value_error(pos, bad_value):
    tile = list(_delta())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((_tile(),),), ((tuple(tile),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, True, "1.0", None])
def test_base_z_fields_must_be_floats(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((tuple(tile),),), ((_delta(),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, True, "0.0", None])
def test_delta_z_fields_must_be_floats(pos, bad_value):
    tile = list(_delta())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((_tile(),),), ((tuple(tile),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_base_z_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((tuple(tile),),), ((_delta(),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_delta_z_raises_value_error(pos, bad_value):
    tile = list(_delta())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((_tile(),),), ((tuple(tile),),))


def test_unsorted_coordinates_raise_value_error():
    unsorted_base = ((_tile(tx=1, ty=0, ix0=256, ix1=511), _tile(tx=0, ty=0)),)
    unsorted_deltas = (
        (_delta(tx=1, ty=0, ix0=256, ix1=511), _delta(tx=0, ty=0)),)
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(unsorted_base, unsorted_deltas)


def test_duplicate_coordinates_raise_value_error():
    duplicate = ((_tile(tx=0, ty=0),
                  _tile(tx=0, ty=0, ix0=256, ix1=511)),)
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(duplicate, duplicate)


def test_inverted_cell_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(ix0=256, ix1=200),),),
            ((_delta(ix0=256, ix1=200),),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(iy0=256, iy1=200),),),
            ((_delta(iy0=256, iy1=200),),))


def test_inverted_base_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(zmin=2.0, zmax=1.0),),), ((_delta(),),))


def test_negative_base_count_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(count=-1),),), ((_delta(),),))


def test_equal_base_z_bounds_are_allowed():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=1.0, zmax=1.0, count=1),),),
        ((_delta(dzmin=0.0, dzmax=0.0, dcount=0),),))
    assert result[0][0][6:8] == (1, 1)


# ---------------------------------------------------------------------------
# ValueError: base/deltas agreement
# ---------------------------------------------------------------------------

def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(((_tile(),),), ((), ()))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas((), ((_delta(),),))


def test_different_coordinate_sets_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(tx=0, ty=0),),),
            ((_delta(tx=1, ty=0, ix0=256, ix1=511),),))
    with pytest.raises(ValueError):  # delta tile present, base tile missing
        rollback_tile_pyramid_deltas(
            ((_tile(tx=0, ty=0),),),
            ((_delta(tx=0, ty=0), _delta(tx=1, ty=0, ix0=256, ix1=511)),))
    with pytest.raises(ValueError):  # base tile present, delta tile missing
        rollback_tile_pyramid_deltas(
            ((_tile(tx=0, ty=0), _tile(tx=1, ty=0, ix0=256, ix1=511)),),
            ((_delta(tx=0, ty=0),),))


def test_matching_coordinates_with_different_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(ix1=200),),), ((_delta(),),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(),),), ((_delta(iy0=1, iy1=255),),))


def test_coordinate_mismatch_at_second_level_raises_value_error():
    base = ((_tile(),), (_tile(tx=0, ty=0),))
    deltas = ((_delta(),), (_delta(tx=1, ty=0, ix0=256, ix1=511),))
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(base, deltas)


# ---------------------------------------------------------------------------
# ValueError: resulting tiles
# ---------------------------------------------------------------------------

def test_negative_resulting_count_raises_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(zmin=1.0, zmax=2.0, count=1),),),
            ((_delta(dzmin=0.0, dzmax=0.0, dcount=2),),))


def test_inverted_resulting_z_bounds_raise_value_error():
    with pytest.raises(ValueError):
        rollback_tile_pyramid_deltas(
            ((_tile(zmin=1.0, zmax=1.0, count=1),),),
            ((_delta(dzmin=-1.0, dzmax=1.0, dcount=0),),))


def test_zero_resulting_count_is_allowed():
    result = rollback_tile_pyramid_deltas(
        ((_tile(zmin=1.0, zmax=2.0, count=3),),),
        ((_delta(dzmin=0.0, dzmax=0.0, dcount=3),),))
    assert result[0][0][8] == 0
