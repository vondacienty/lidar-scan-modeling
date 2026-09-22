"""Tests for :func:`lidar_scan.chm.build_chm`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import __version__, build_chm

GROUND = [(0, 0, 10.0, 0.5, 3), (-1, -1, 0.0, 1.0, 1), (1, 0, -2.0, 0.25, 2)]


def test_version_unchanged():
    assert __version__ == "0.1.0"


def test_empty_points_returns_empty_tuple():
    assert build_chm([], GROUND) == ()
    assert build_chm(iter(()), iter(GROUND)) == ()


def test_single_point_height_and_sigma():
    result = build_chm([(0.5, 0.5, 12.5, 1.0, 0.5)], [(0, 0, 10.0, 0.5, 1)])
    assert result == ((0, 0, 2.5, pytest.approx(math.sqrt(0.5)), 1),)


def test_height_clamped_at_zero():
    result = build_chm([(0.5, 0.5, 8.0, 1.0, 1.0)], [(0, 0, 10.0, 1.0, 1)])
    assert result == ((0, 0, 0.0, pytest.approx(math.sqrt(2.0)), 1),)


def test_highest_point_wins_per_cell():
    result = build_chm([(0.1, 0.1, 11.0, 1.0, 1.0),
                        (0.2, 0.2, 13.0, 2.0, 1.0),
                        (0.3, 0.3, 12.0, 3.0, 1.0)],
                       [(0, 0, 10.0, 1.0, 1)])
    assert result == ((0, 0, 3.0, pytest.approx(math.sqrt(2.0)), 3),)


def test_ties_broken_by_sigma_intensity_x_y():
    # equal z: lowest sigma wins
    result = build_chm([(0.1, 0.1, 12.0, 1.0, 2.0),
                        (0.2, 0.2, 12.0, 1.0, 1.0)],
                       [(0, 0, 10.0, 1.0, 1)])
    assert result[0][3] == pytest.approx(math.sqrt(2.0))
    # equal z and sigma: lowest intensity wins
    result = build_chm([(0.1, 0.1, 12.0, 5.0, 1.0),
                        (0.2, 0.2, 12.0, 3.0, 1.0)],
                       [(0, 0, 10.0, 1.0, 1)])
    assert result[0][2] == 2.0
    # equal z, sigma, intensity: lowest x wins
    result = build_chm([(0.9, 0.1, 12.0, 1.0, 1.0),
                        (0.1, 0.2, 12.0, 1.0, 1.0)],
                       [(0, 0, 10.0, 1.0, 1)])
    assert result[0][2] == 2.0


def test_cell_size_divides_coordinates():
    result = build_chm([(1.9, 0.0, 5.0, 1.0, 1.0)], [(0, 0, 1.0, 1.0, 1)],
                       cell_size=2)
    assert result == ((0, 0, 4.0, pytest.approx(math.sqrt(2.0)), 1),)
    # exactly on a boundary belongs to the higher cell
    boundary = build_chm([(2.0, 0.0, 5.0, 1.0, 1.0)], [(1, 0, 1.0, 1.0, 1)],
                         cell_size=2)
    assert boundary[0][:2] == (1, 0)


def test_indices_floor_toward_negative_infinity():
    result = build_chm([(-0.1, -0.1, 1.0, 1.0, 1.0)], [(-1, -1, 0.0, 1.0, 1)])
    assert result[0][:2] == (-1, -1)


def test_results_sorted_lexicographically():
    result = build_chm([(0.5, 0.5, 12.0, 1.0, 1.0),
                        (-0.5, -0.5, 1.0, 1.0, 1.0),
                        (1.5, 0.5, 0.0, 1.0, 1.0)], GROUND)
    assert [row[:2] for row in result] == [(-1, -1), (0, 0), (1, 0)]
    assert [row[4] for row in result] == [1, 1, 1]


def test_six_decimal_quantization():
    result = build_chm([(0.5, 0.5, 10.123456789, 1.0, 1.0)],
                       [(0, 0, 10.0, 1.0, 1)])
    assert result[0][2] == 0.123457


@pytest.mark.parametrize("cell_size", ["1", None, True, [1.0]])
def test_cell_size_type_error(cell_size):
    with pytest.raises(TypeError):
        build_chm([], GROUND, cell_size=cell_size)


@pytest.mark.parametrize("cell_size", [0.0, -1.0, 0, float("inf"), float("nan")])
def test_cell_size_value_error(cell_size):
    with pytest.raises(ValueError):
        build_chm([], GROUND, cell_size=cell_size)


@pytest.mark.parametrize("points", [None, 1, 1.5, "points"])
def test_points_not_iterable_type_error(points):
    with pytest.raises(TypeError):
        build_chm(points, GROUND)


@pytest.mark.parametrize("ground", [None, 1, 1.5, "ground"])
def test_ground_not_iterable_type_error(ground):
    with pytest.raises(TypeError):
        build_chm([], ground)


@pytest.mark.parametrize("point", [(1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                                   "12345", None])
def test_bad_point_container_type_error(point):
    with pytest.raises(TypeError):
        build_chm([point], GROUND)


@pytest.mark.parametrize("cell", [(0, 0, 1.0, 1.0), (0, 0, 1.0, 1.0, 1, 2),
                                  "12345", None])
def test_bad_ground_container_type_error(cell):
    with pytest.raises(TypeError):
        build_chm([], [cell])


@pytest.mark.parametrize("value", [True, "1", None, [1.0]])
def test_point_field_type_error(value):
    point = [0.5, 0.5, 12.0, 1.0, 1.0]
    for i in range(5):
        bad = list(point)
        bad[i] = value
        with pytest.raises(TypeError):
            build_chm([tuple(bad)], GROUND)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_point_field_value_error(value):
    point = [0.5, 0.5, 12.0, 1.0, 1.0]
    for i in range(5):
        bad = list(point)
        bad[i] = value
        with pytest.raises(ValueError):
            build_chm([tuple(bad)], GROUND)


@pytest.mark.parametrize("sigma", [0.0, -1.0, 0])
def test_point_sigma_value_error(sigma):
    with pytest.raises(ValueError):
        build_chm([(0.5, 0.5, 12.0, 1.0, sigma)], GROUND)


@pytest.mark.parametrize("field", [0, 1, 4])
@pytest.mark.parametrize("value", [True, 1.5, "1", None])
def test_ground_index_and_count_type_error(field, value):
    cell = [0, 0, 10.0, 1.0, 1]
    cell[field] = value
    with pytest.raises(TypeError):
        build_chm([], [tuple(cell)])


@pytest.mark.parametrize("field", [2, 3])
@pytest.mark.parametrize("value", [True, "1", None])
def test_ground_z_sigma_type_error(field, value):
    cell = [0, 0, 10.0, 1.0, 1]
    cell[field] = value
    with pytest.raises(TypeError):
        build_chm([], [tuple(cell)])


@pytest.mark.parametrize("field", [2, 3])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_ground_z_sigma_value_error(field, value):
    cell = [0, 0, 10.0, 1.0, 1]
    cell[field] = value
    with pytest.raises(ValueError):
        build_chm([], [tuple(cell)])


@pytest.mark.parametrize("sigma", [0.0, -1.0, 0])
def test_ground_sigma_value_error(sigma):
    with pytest.raises(ValueError):
        build_chm([], [(0, 0, 10.0, sigma, 1)])


@pytest.mark.parametrize("count", [0, -1])
def test_ground_count_value_error(count):
    with pytest.raises(ValueError):
        build_chm([], [(0, 0, 10.0, 1.0, count)])


def test_duplicate_ground_index_value_error():
    with pytest.raises(ValueError):
        build_chm([], [(0, 0, 10.0, 1.0, 1), (0, 0, 11.0, 1.0, 1)])


def test_point_cell_without_ground_value_error():
    with pytest.raises(ValueError):
        build_chm([(5.5, 5.5, 1.0, 1.0, 1.0)], GROUND)


def test_inputs_consumed_as_single_pass_iterables():
    class SinglePass:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            raise TypeError("no len")

    points = SinglePass([(0.5, 0.5, 12.0, 1.0, 1.0)])
    ground = SinglePass([(0, 0, 10.0, 1.0, 1)])
    result = build_chm(points, ground)
    assert result[0][:2] == (0, 0)
