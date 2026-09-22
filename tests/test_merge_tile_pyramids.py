"""Tests for :func:`lidar_scan.tiles.merge_tile_pyramids`."""

from __future__ import annotations

import pytest

from lidar_scan import build_tile_pyramid, merge_tile_pyramids
from lidar_scan import tiles as tiles_module
import lidar_scan


def _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
          zmin=1.0, zmax=2.0, count=1):
    return (tx, ty, ix0, iy0, ix1, iy1, zmin, zmax, count)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.merge_tile_pyramids is merge_tile_pyramids
    assert lidar_scan.merge_tile_pyramids is merge_tile_pyramids
    assert "merge_tile_pyramids" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic behaviour
# ---------------------------------------------------------------------------

def test_empty_tuple_returns_empty_tuple():
    result = merge_tile_pyramids(())
    assert result == ()
    assert isinstance(result, tuple)


def test_zero_level_pyramids():
    assert merge_tile_pyramids(((), ())) == ()
    assert merge_tile_pyramids(((), (), ())) == ()


def test_single_pyramid_returned_equal():
    pyramid = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0.0, 1.0), (300.0, 0.0, 5.0, 0.0, 1.0)], levels=2)
    assert merge_tile_pyramids((pyramid,)) == pyramid


def test_union_of_disjoint_coordinates():
    p1 = build_tile_pyramid([(0.0, 0.0, 1.0, 0.0, 1.0)], levels=1)
    p2 = build_tile_pyramid([(600.0, 600.0, 4.0, 0.0, 1.0)], levels=1)
    result = merge_tile_pyramids((p1, p2))
    assert len(result) == 1
    assert [tile[:2] for tile in result[0]] == [(0, 0), (2, 2)]
    assert result[0][0][6:9] == (1.0, 1.0, 1)
    assert result[0][1][6:9] == (4.0, 4.0, 1)


def test_shared_coordinate_aggregated():
    p1 = (((0, 0, 0, 0, 255, 255, 1.0, 4.0, 3),),)
    p2 = (((0, 0, 0, 0, 255, 255, 0.5, 9.0, 2),),)
    result = merge_tile_pyramids((p1, p2))
    assert result == (((0, 0, 0, 0, 255, 255, 0.5, 9.0, 5),),)


def test_geometry_taken_from_first_input():
    p1 = (((1, -2, 256, -512, 511, -257, 1.0, 1.0, 1),),)
    p2 = (((1, -2, 256, -512, 511, -257, 2.0, 2.0, 1),),)
    result = merge_tile_pyramids((p1, p2))
    assert result[0][0][2:6] == (256, -512, 511, -257)
    assert result[0][0][6:9] == (1.0, 2.0, 2)


def test_tiles_sorted_by_coordinates():
    p1 = build_tile_pyramid([(600.0, 0.0, 1.0, 0.0, 1.0)], levels=1)
    p2 = build_tile_pyramid([(0.0, 600.0, 1.0, 0.0, 1.0),
                             (-300.0, 0.0, 1.0, 0.0, 1.0)], levels=1)
    result = merge_tile_pyramids((p1, p2))
    keys = [tile[:2] for tile in result[0]]
    assert keys == sorted(keys)
    assert keys == [(-2, 0), (0, 2), (2, 0)]


def test_empty_levels_preserved():
    p1 = build_tile_pyramid([], levels=3)
    p2 = build_tile_pyramid([(0.0, 0.0, 1.0, 0.0, 1.0)], levels=3)
    result = merge_tile_pyramids((p1, p2))
    assert len(result) == 3
    assert isinstance(result[0], tuple)
    assert isinstance(result[1], tuple)
    assert isinstance(result[2], tuple)


def test_level_count_and_order_preserved():
    p1 = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0.0, 1.0), (300.0, 300.0, 2.0, 0.0, 1.0)], levels=3)
    p2 = build_tile_pyramid([(600.0, 0.0, 3.0, 0.0, 1.0)], levels=3)
    result = merge_tile_pyramids((p1, p2))
    assert len(result) == 3
    for level, level_tiles in enumerate(result):
        assert isinstance(level_tiles, tuple)
        assert all(isinstance(tile, tuple) for tile in level_tiles)
        width = 256 * 2 ** level
        for tile in level_tiles:
            assert tile[2] == tile[0] * width
            assert tile[4] == tile[0] * width + width - 1


def test_result_independent_of_input_order():
    p1 = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0.0, 1.0), (300.0, 0.0, 5.0, 0.0, 1.0)], levels=2)
    p2 = build_tile_pyramid(
        [(0.0, 0.0, 3.0, 0.0, 1.0), (600.0, 0.0, -2.0, 0.0, 1.0)], levels=2)
    p3 = build_tile_pyramid([(0.0, 900.0, 7.0, 0.0, 1.0)], levels=2)
    assert (merge_tile_pyramids((p1, p2, p3))
            == merge_tile_pyramids((p3, p1, p2))
            == merge_tile_pyramids((p2, p3, p1)))


def test_inputs_not_modified():
    p1 = build_tile_pyramid([(0.0, 0.0, 1.0, 0.0, 1.0)], levels=2)
    p2 = build_tile_pyramid([(0.0, 0.0, 3.0, 0.0, 1.0)], levels=2)
    snapshot = tuple(tuple(t) for t in p1)
    merge_tile_pyramids((p1, p2))
    assert tuple(tuple(t) for t in p1) == snapshot


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, "x", {}, [()], iter(())])
def test_non_tuple_argument_type_error(bad):
    with pytest.raises(TypeError):
        merge_tile_pyramids(bad)


def test_member_not_a_tuple_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids(([()],))


def test_differing_level_counts_value_error():
    p1 = build_tile_pyramid([(0.0, 0.0, 1.0, 0.0, 1.0)], levels=2)
    p2 = build_tile_pyramid([(0.0, 0.0, 1.0, 0.0, 1.0)], levels=1)
    with pytest.raises(ValueError):
        merge_tile_pyramids((p1, p2))


def test_level_not_a_tuple_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids(((([_tile()]),),))


def test_tile_not_a_nine_tuple_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids((((_tile()[:8],),),))
    with pytest.raises(ValueError):
        merge_tile_pyramids(((([_tile()],),),))


def test_bool_integer_fields_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids(
            (((_tile(tx=True),),),))
    with pytest.raises(ValueError):
        merge_tile_pyramids(
            (((_tile(count=False),),),))


def test_non_float_z_fields_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids((((_tile(zmin=1),),),))
    with pytest.raises(ValueError):
        merge_tile_pyramids((((_tile(zmax="2.0"),),),))


@pytest.mark.parametrize("zmin,zmax", [
    (float("nan"), 2.0),
    (1.0, float("inf")),
    (float("-inf"), 2.0),
])
def test_nonfinite_z_fields_value_error(zmin, zmax):
    with pytest.raises(ValueError):
        merge_tile_pyramids((((_tile(zmin=zmin, zmax=zmax),),),))


def test_unsorted_or_duplicate_coordinates_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramids(
            (((_tile(tx=1), _tile(tx=0)),),))
    with pytest.raises(ValueError):
        merge_tile_pyramids(
            (((_tile(tx=0, ty=1), _tile(tx=0, ty=1)),),))


def test_geometry_mismatch_value_error():
    p1 = (((_tile(ix1=255),),),)
    p2 = (((_tile(ix1=200),),),)
    with pytest.raises(ValueError):
        merge_tile_pyramids((p1, p2))


def test_geometry_mismatch_on_second_level():
    p1 = (((_tile(),),), ((_tile(ix1=255),),))
    p2 = (((_tile(),),), ((_tile(ix1=200),),))
    with pytest.raises(ValueError):
        merge_tile_pyramids((p1, p2))
