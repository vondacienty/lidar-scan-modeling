"""Tests for :func:`lidar_scan.merge_tile_pyramid_deltas`."""

from __future__ import annotations

import copy
import math

import pytest

from lidar_scan import merge_tile_pyramid_deltas
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  dzmin=0.0, dzmax=0.0, dcount=0)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["dzmin"], fields["dzmax"],
            fields["dcount"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.merge_tile_pyramid_deltas is merge_tile_pyramid_deltas
    import lidar_scan
    assert "merge_tile_pyramid_deltas" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_tuple_returns_empty_tuple():
    assert merge_tile_pyramid_deltas(()) == ()


def test_empty_levels_are_preserved():
    assert merge_tile_pyramid_deltas((((), (), ()),)) == ((), (), ())


def test_single_assessment_is_reproduced():
    assessment = (
        (_tile(tx=0, ty=0, dzmin=1.5, dzmax=-2.25, dcount=3),
         _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-0.5, dzmax=4.0,
               dcount=-7)),
        (),
    )
    assert merge_tile_pyramid_deltas((assessment,)) == assessment
    assert type(merge_tile_pyramid_deltas((assessment,))) is tuple


def test_matching_tiles_sum_their_deltas():
    first = ((_tile(dzmin=1.0, dzmax=2.0, dcount=3),),)
    second = ((_tile(dzmin=-4.0, dzmax=0.5, dcount=-8),),)
    third = ((_tile(dzmin=2.5, dzmax=-1.5, dcount=2),),)
    assert merge_tile_pyramid_deltas((first, second, third)) == \
        (((0, 0, 0, 0, 255, 255, -0.5, 1.0, -3),),)


def test_missing_coordinates_contribute_zero():
    first = ((_tile(tx=0, ty=0, dzmin=1.0, dzmax=2.0, dcount=3),
              _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=5.0, dzmax=6.0,
                    dcount=4)),)
    second = ((_tile(tx=0, ty=0, dzmin=-1.0, dzmax=-2.0, dcount=-3),),)
    result = merge_tile_pyramid_deltas((first, second))
    assert result == (
        ((0, 0, 0, 0, 255, 255, 0.0, 0.0, 0),
         (1, 0, 256, 0, 511, 255, 5.0, 6.0, 4)),
    )


def test_coordinates_are_unioned_and_sorted():
    first = ((_tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=1.0,
                    dcount=1),),)
    second = ((_tile(tx=0, ty=0, dzmin=2.0, dzmax=2.0, dcount=2),
               _tile(tx=2, ty=0, ix0=512, ix1=767, dzmin=3.0, dzmax=3.0,
                     dcount=3)),)
    result = merge_tile_pyramid_deltas((first, second))
    assert [t[:2] for t in result[0]] == [(0, 0), (1, 0), (2, 0)]
    assert [t[8] for t in result[0]] == [2, 1, 3]


def test_level_order_is_preserved():
    first = (
        (_tile(dzmin=1.0, dcount=1),),
        (),
        (_tile(ix0=0, iy0=0, ix1=1023, iy1=1023, dzmin=4.0, dcount=4),),
    )
    second = (
        (),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, dzmin=2.0, dcount=2),),
        (_tile(ix0=0, iy0=0, ix1=1023, iy1=1023, dzmin=-1.0, dcount=-1),),
    )
    result = merge_tile_pyramid_deltas((first, second))
    assert len(result) == 3
    assert result[0][0][6] == 1.0
    assert result[1][0][6] == 2.0
    assert result[2][0][6] == 3.0
    assert result[2][0][8] == 3


def test_dcount_is_exact_int_even_for_huge_values():
    huge = 10 ** 40
    result = merge_tile_pyramid_deltas(
        (((_tile(dcount=huge),),), ((_tile(dcount=-7),),)))
    dcount = result[0][0][8]
    assert dcount == huge - 7
    assert isinstance(dcount, int)


def test_zero_float_sum_normalized_to_positive_zero():
    tile = merge_tile_pyramid_deltas(
        (((_tile(dzmin=-0.0, dzmax=1.0),),),
         ((_tile(dzmin=0.0, dzmax=-1.0),),)))[0][0]
    assert tile[6] == 0.0 and math.copysign(1.0, tile[6]) == 1.0
    assert tile[7] == 0.0 and math.copysign(1.0, tile[7]) == 1.0


def test_quantized_to_six_decimal_places_with_half_even_rounding():
    first = ((_tile(dzmin=0.0000005, dzmax=0.0000015),),)
    second = ((_tile(dzmin=1.0, dzmax=1.0),),)
    tile = merge_tile_pyramid_deltas((first, second))[0][0]
    # 1 + 0.0000005 -> 1.000000 (half to even); 1 + 0.0000015 -> 1.000002
    assert tile[6] == 1.0
    assert tile[7] == 1.000002


def test_decimal_str_conversion_avoids_binary_artifact():
    first = ((_tile(dzmin=0.1, dzmax=0.2),),)
    second = ((_tile(dzmin=0.2, dzmax=0.1),),)
    tile = merge_tile_pyramid_deltas((first, second))[0][0]
    # Plain float addition would yield 0.30000000000000004.
    assert tile[6] == 0.3
    assert tile[7] == 0.3


def test_result_does_not_depend_on_input_order():
    a = ((_tile(tx=0, ty=0, dzmin=1.0, dzmax=3.0, dcount=2),
          _tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=-4.0, dzmax=0.5,
                dcount=-9)),)
    b = ((_tile(tx=0, ty=0, dzmin=-1.0, dzmax=2.0, dcount=7),
          _tile(tx=2, ty=0, ix0=512, ix1=767, dzmin=0.25, dzmax=0.75,
                dcount=1)),)
    c = ((_tile(tx=1, ty=0, ix0=256, ix1=511, dzmin=4.0, dzmax=-0.5,
                dcount=9),),)
    forward = merge_tile_pyramid_deltas((a, b, c))
    reverse = merge_tile_pyramid_deltas((c, b, a))
    assert forward == reverse
    assert forward == (
        ((0, 0, 0, 0, 255, 255, 0.0, 5.0, 9),
         (1, 0, 256, 0, 511, 255, 0.0, 0.0, 0),
         (2, 0, 512, 0, 767, 255, 0.25, 0.75, 1)),
    )


def test_inputs_are_not_modified_and_calls_are_stable():
    a = ((_tile(dzmin=1.0, dzmax=2.0, dcount=3),),)
    b = ((_tile(dzmin=-1.0, dzmax=0.5, dcount=2),),)
    a_before = copy.deepcopy(a)
    b_before = copy.deepcopy(b)
    first = merge_tile_pyramid_deltas((a, b))
    second = merge_tile_pyramid_deltas((a, b))
    assert first == second
    assert a == a_before
    assert b == b_before


def test_output_is_all_tuples():
    result = merge_tile_pyramid_deltas(
        (((_tile(dzmin=1.0),),), ((_tile(tx=1, ty=1, ix0=256, iy0=256,
                                         ix1=511, iy1=511),),)))
    assert type(result) is tuple
    for level in result:
        assert type(level) is tuple
        for tile in level:
            assert type(tile) is tuple


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_outer_input_raises_type_error(bad):
    with pytest.raises(TypeError):
        merge_tile_pyramid_deltas(bad)


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_member_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((_tile(),),), [(_tile(),)]))


def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((([_tile()],),))


@pytest.mark.parametrize("bad_tile", [
    list(_tile()),
    (0, 0, 0, 0, 255, 255, 0.0, 0.0),
    _tile() + (99,),
])
def test_tile_wrong_container_or_length_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((bad_tile,),),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_integer_fields_wrong_type_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((tuple(tile),),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, True, "0.0", None])
def test_delta_fields_must_be_floats(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((tuple(tile),),),))


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_delta_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((tuple(tile),),),))


def test_unsorted_coordinates_raise_value_error():
    bad = ((_tile(tx=1, ty=0, ix0=256, ix1=511), _tile(tx=0, ty=0)),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((bad,))


def test_duplicate_coordinates_raise_value_error():
    bad = ((_tile(tx=0, ty=0), _tile(tx=0, ty=0)),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((bad,))


def test_inverted_cell_bounds_raise_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((_tile(ix0=256, ix1=200),),),))
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((_tile(iy0=256, iy1=200),),),))


# ---------------------------------------------------------------------------
# ValueError: agreement between assessments
# ---------------------------------------------------------------------------

def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((((_tile(),),), ((), ())))
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas(((), ((_tile(),),)))


def test_matching_coordinates_with_different_bounds_raise_value_error():
    first = ((_tile(ix1=200),),)
    second = ((_tile(ix1=255),),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((first, second))
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas(
            (first, ((_tile(iy0=1, iy1=255),),)))


def test_different_bounds_at_second_level_raise_value_error():
    first = (
        (_tile(),),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511),),
    )
    second = (
        (_tile(),),
        (_tile(ix0=0, iy0=0, ix1=1023, iy1=511),),
    )
    with pytest.raises(ValueError):
        merge_tile_pyramid_deltas((first, second))


def test_different_dcounts_at_same_coordinate_are_allowed():
    result = merge_tile_pyramid_deltas(
        (((_tile(dcount=3),),), ((_tile(dcount=10),),)))
    assert result[0][0][8] == 13
