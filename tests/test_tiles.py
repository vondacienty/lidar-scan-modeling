"""Tests for :func:`lidar_scan.tiles.build_tile_index`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import __version__, build_tile_index


def test_empty_returns_empty_tuple():
    assert build_tile_index([]) == ()
    assert build_tile_index(iter(())) == ()


def test_single_point_tile():
    result = build_tile_index([(1.0, -2.0, 3.5, 10.0, 0.5)])
    assert result == ((0, -1, 0, -256, 255, -1, 3.5, 3.5, 1),)


def test_cell_indices_floor_toward_negative_infinity():
    result = build_tile_index([(-0.1, 0.0, 1.0, 1.0, 1.0),
                               (0.1, 0.0, 2.0, 1.0, 1.0)], tile_cells=4)
    assert [tile[:2] for tile in result] == [(-1, 0), (0, 0)]
    assert result[0][2:6] == (-4, 0, -1, 3)
    assert result[1][2:6] == (0, 0, 3, 3)


def test_cell_size_divides_coordinates():
    result = build_tile_index([(3.9, 0.0, 1.0, 1.0, 1.0),
                               (4.1, 0.0, 2.0, 1.0, 1.0)],
                              cell_size=2, tile_cells=2)
    # ix = 1 and ix = 2 fall into tiles tx = 0 and tx = 1
    assert [tile[:2] for tile in result] == [(0, 0), (1, 0)]
    assert result[0][2:6] == (0, 0, 1, 1)
    assert result[1][2:6] == (2, 0, 3, 1)


def test_tile_bounds_from_tile_cells():
    result = build_tile_index([(256.0, -257.0, 0.0, 0.0, 1.0)])
    assert result[0][:2] == (1, -2)
    assert result[0][2:6] == (256, -512, 511, -257)


def test_zmin_zmax_count_aggregated_per_tile():
    result = build_tile_index([(0.1, 0.1, 3.0, 1.0, 1.0),
                               (0.2, 0.2, -1.5, 1.0, 1.0),
                               (0.3, 0.3, 2.25, 1.0, 1.0)])
    assert result == ((0, 0, 0, 0, 255, 255, -1.5, 3.0, 3),)


def test_results_sorted_lexicographically():
    result = build_tile_index([(500.0, 0.0, 0.0, 0.0, 1.0),
                               (0.0, 500.0, 0.0, 0.0, 1.0),
                               (-1.0, 0.0, 0.0, 0.0, 1.0),
                               (0.0, 0.0, 0.0, 0.0, 1.0)])
    assert [tile[:2] for tile in result] == [(-1, 0), (0, 0), (0, 1), (1, 0)]


def test_results_independent_of_input_order():
    points = [(0.1, 0.2, 0.3, 1.0, 0.3),
              (300.4, 0.3, 0.2, 9.0, 0.7),
              (0.25, 0.25, -0.25, 5.0, 1.1)]
    assert build_tile_index(points) == build_tile_index(list(reversed(points)))


def test_accepts_tuples_lists_and_iterators():
    result = build_tile_index(iter([(0.0, 0.0, 1.0, 0.0, 1.0),
                                    [0.2, 0.2, 2.0, 2.0, 1.0]]))
    assert result[0][8] == 2
    assert result[0][6:8] == (1.0, 2.0)


def test_zmin_zmax_quantized_to_six_decimals():
    result = build_tile_index([(0.0, 0.0, 1 / 3, 0.0, 1.0),
                               (0.0, 0.0, -1 / 7, 0.0, 1.0)])
    assert result[0][6] == round(-1 / 7, 6)
    assert result[0][7] == round(1 / 3, 6)


def test_negative_zero_normalized():
    result = build_tile_index([(0.0, 0.0, -0.0, 0.0, 1.0)])
    for index in (6, 7):
        assert result[0][index] == 0.0
        assert math.copysign(1.0, result[0][index]) == 1.0


def test_return_types_are_tuples():
    result = build_tile_index([(0.0, 0.0, 0.0, 0.0, 1.0)])
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)


@pytest.mark.parametrize("bad_points", [
    123,
    None,
    3.14,
    True,
])
def test_non_iterable_points_type_error(bad_points):
    with pytest.raises(TypeError):
        build_tile_index(bad_points)


@pytest.mark.parametrize("point", [
    123,
    (1, 2, 3, 4),
    (1, 2, 3, 4, 5, 6),
    [1, 2, 3],
    {0, 0, 0, 0, 1},
    b"12345",
    "12345",
])
def test_bad_point_container_or_length_type_error(point):
    with pytest.raises(TypeError):
        build_tile_index([point])


@pytest.mark.parametrize("point", [
    ("1", 0, 0, 0, 1),
    (0, [0], 0, 0, 1),
    (0, 0, None, 0, 1),
    (0, 0, 0, {1: 2}, 1),
    (0, 0, 0, 0, 1 + 0j),
    (True, 0, 0, 0, 1),
    (0, 0, 0, False, 1),
    (0, 0, 0, 0, True),
])
def test_illegal_scalar_type_error(point):
    with pytest.raises(TypeError):
        build_tile_index([point])


@pytest.mark.parametrize("cell_size", ["1", [1.0], None, True])
def test_bad_cell_size_type(cell_size):
    with pytest.raises(TypeError):
        build_tile_index([(0, 0, 0, 0, 1.0)], cell_size=cell_size)


@pytest.mark.parametrize("cell_size", [0, -1, -1.5, float("nan"),
                                       float("inf"), float("-inf")])
def test_bad_cell_size_value(cell_size):
    with pytest.raises(ValueError):
        build_tile_index([(0, 0, 0, 0, 1.0)], cell_size=cell_size)


@pytest.mark.parametrize("tile_cells", ["256", 256.0, [256], None, True])
def test_bad_tile_cells_type(tile_cells):
    with pytest.raises(TypeError):
        build_tile_index([(0, 0, 0, 0, 1.0)], tile_cells=tile_cells)


@pytest.mark.parametrize("tile_cells", [0, -1, -256])
def test_bad_tile_cells_value(tile_cells):
    with pytest.raises(ValueError):
        build_tile_index([(0, 0, 0, 0, 1.0)], tile_cells=tile_cells)


@pytest.mark.parametrize("sigma", [0, -1, -0.5, float("nan"),
                                   float("inf"), float("-inf")])
def test_bad_sigma_value(sigma):
    with pytest.raises(ValueError):
        build_tile_index([(0, 0, 0, 0, sigma)])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_value_error(value):
    with pytest.raises(ValueError):
        build_tile_index([(value, 0, 0, 0, 1.0)])


def test_version_unchanged():
    assert __version__ == "0.1.0"
