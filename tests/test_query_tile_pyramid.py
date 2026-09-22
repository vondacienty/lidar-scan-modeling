"""Tests for :func:`lidar_scan.tiles.query_tile_pyramid`."""

from __future__ import annotations

import copy

import pytest

from lidar_scan import (build_tile_pyramid, query_tile_pyramid,
                        __version__)

GOOD = (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1)


@pytest.fixture
def pyramid():
    points = [(0.1, 0.1, 3.0, 1.0, 1.0),
              (600.0, 600.0, 1.0, 1.0, 1.0),
              (-5.0, 10.0, 2.0, 1.0, 1.0)]
    return build_tile_pyramid(points, tile_cells=4, levels=3)


def test_hit_returns_original_tile_tuple(pyramid):
    for level, tiles in enumerate(pyramid):
        for tile in tiles:
            found = query_tile_pyramid(pyramid, level, tile[0], tile[1])
            assert found is tile
            assert found == tile


def test_miss_returns_none(pyramid):
    assert query_tile_pyramid(pyramid, 0, 99, 99) is None
    assert query_tile_pyramid(pyramid, 0, -99, -99) is None


def test_empty_levels_are_queryable():
    empty = build_tile_pyramid([], levels=2)
    assert query_tile_pyramid(empty, 0, 0, 0) is None
    assert query_tile_pyramid(empty, 1, -1, -1) is None


def test_input_not_modified_or_reordered(pyramid):
    snapshot = copy.deepcopy(pyramid)
    for level, tiles in enumerate(pyramid):
        if tiles:
            query_tile_pyramid(pyramid, level, tiles[0][0], tiles[0][1])
    query_tile_pyramid(pyramid, 0, 999, 999)
    assert pyramid == snapshot


@pytest.mark.parametrize("bad_pyramid", [[], [GOOD], None, 123, "x"])
def test_pyramid_must_be_tuple(bad_pyramid):
    with pytest.raises(TypeError):
        query_tile_pyramid(bad_pyramid, 0, 0, 0)


@pytest.mark.parametrize("level,tx,ty", [
    (True, 0, 0),
    (0, 1.0, 0),
    (0, 0, False),
    ("0", 0, 0),
    (0, [0], 0),
])
def test_key_arguments_must_be_non_bool_ints(level, tx, ty):
    with pytest.raises(TypeError):
        query_tile_pyramid(((GOOD,),), level, tx, ty)


def test_level_out_of_range_value_error(pyramid):
    with pytest.raises(ValueError):
        query_tile_pyramid(pyramid, -1, 0, 0)
    with pytest.raises(ValueError):
        query_tile_pyramid(pyramid, 3, 0, 0)


def test_level_must_be_tuple():
    with pytest.raises(ValueError):
        query_tile_pyramid(((GOOD,), [GOOD]), 1, 0, 0)


def test_tile_must_be_tuple():
    with pytest.raises(ValueError):
        query_tile_pyramid(((GOOD, [0] * 9),), 0, 0, 0)


@pytest.mark.parametrize("tile", [
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),            # too short
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 9),      # too long
])
def test_tile_must_have_nine_fields(tile):
    with pytest.raises(ValueError):
        query_tile_pyramid(((tile,),), 0, 0, 0)


@pytest.mark.parametrize("index,bad_value", [
    (0, True), (1, True), (2, 1.0), (3, 1.0),
    (4, 1.0), (5, 1.0), (8, True), (8, 1.0),
])
def test_integer_fields_must_be_non_bool_ints(index, bad_value):
    tile = list(GOOD)
    tile[index] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid(((tuple(tile),),), 0, 0, 0)


@pytest.mark.parametrize("index,bad_value", [
    (6, 1), (7, 2),
    (6, True),
    (6, float("nan")), (7, float("inf")), (7, float("-inf")),
])
def test_z_fields_must_be_finite_floats(index, bad_value):
    tile = list(GOOD)
    tile[index] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid(((tuple(tile),),), 0, 0, 0)


def test_unsorted_level_value_error():
    tiles = ((1, 0, 0, 0, 255, 255, 1.0, 2.0, 1),
             (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1))
    with pytest.raises(ValueError):
        query_tile_pyramid((tiles,), 0, 0, 0)


def test_duplicate_coordinates_value_error():
    tiles = ((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1),
             (0, 0, 0, 0, 255, 255, 9.0, 9.0, 2))
    with pytest.raises(ValueError):
        query_tile_pyramid((tiles,), 0, 0, 0)


def test_only_target_level_validated(pyramid):
    # Querying a valid level must succeed even if another level is malformed.
    mixed = (pyramid[0], "not-a-tuple")
    tile = pyramid[0][0]
    assert query_tile_pyramid(mixed, 0, tile[0], tile[1]) == tile


def test_version_unchanged():
    assert __version__ == "0.1.0"
