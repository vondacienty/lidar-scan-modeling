"""Tests for :func:`lidar_scan.chm.build_chm`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import __version__, build_chm

GROUND = ((0, 0, 1.0, 0.5, 4),)


def test_empty_points_returns_empty_tuple():
    assert build_chm([], GROUND) == ()
    assert build_chm(iter(()), iter(GROUND)) == ()


def test_ground_consumed_even_when_points_empty():
    # ground validation still happens for an empty points iterable
    with pytest.raises(ValueError):
        build_chm([], ((0, 0, 0.0, 1.0, 1), (0, 0, 0.0, 2.0, 1)))


def test_single_cell_height_and_sigma():
    result = build_chm([(0.2, 0.2, 3.0, 10.0, 0.5)], GROUND)
    assert result == ((0, 0, 2.0, pytest.approx(math.sqrt(0.25 + 0.25)), 1),)


def test_highest_z_is_kept_and_count_counts_all_points():
    points = [(0.2, 0.2, 3.0, 10.0, 0.5),
              (0.8, 0.8, 5.0, 4.0, 0.7),
              (0.1, 0.1, 5.0, 9.0, 0.7)]
    result = build_chm(points, GROUND)
    # z=5 tie: sigma equal, intensity 4 < 9 -> intensity-4 point wins
    assert result == ((0, 0, 4.0, pytest.approx(math.sqrt(0.49 + 0.25)), 3),)


def test_height_clamped_to_zero():
    result = build_chm([(0.0, 0.0, 0.5, 0.0, 1.0)], ((0, 0, 1.0, 1.0, 1),))
    assert result[0][2] == 0.0
    assert math.copysign(1.0, result[0][2]) == 1.0


def test_indices_floor_toward_negative_infinity():
    result = build_chm([(-0.1, -0.1, 1.0, 0.0, 1.0)],
                       ((-1, -1, 0.0, 1.0, 1),))
    assert result[0][:2] == (-1, -1)


def test_cell_size_divides_coordinates():
    result = build_chm([(3.0, 3.0, 5.0, 0.0, 1.0)],
                       ((1, 1, 1.0, 1.0, 1),), cell_size=2)
    assert result == ((1, 1, 4.0, pytest.approx(math.sqrt(2)), 1),)
    # exactly on a boundary belongs to the higher cell
    boundary = build_chm([(2.0, 0.0, 1.0, 0.0, 1.0)],
                         ((1, 0, 0.0, 1.0, 1),), cell_size=2)
    assert boundary[0][:2] == (1, 0)


def test_tie_resolved_by_sigma_then_intensity_then_x_then_y():
    ground = ((0, 0, 0.0, 0.4, 1),)
    # equal z: smaller sigma wins
    result = build_chm([(0.1, 0.1, 5.0, 0.0, 0.9),
                        (0.2, 0.2, 5.0, 0.0, 0.3)], ground)
    assert result[0][3] == pytest.approx(0.5)
    # equal z and sigma: smaller intensity wins
    points = [(0.2, 0.2, 5.0, 9.0, 0.3), (0.1, 0.1, 5.0, 4.0, 0.3)]
    assert build_chm(points, ground) == build_chm(list(reversed(points)), ground)
    # equal z, sigma and intensity: smaller x then y wins -> order independent
    points = [(0.8, 0.8, 5.0, 4.0, 0.3), (0.1, 0.2, 5.0, 4.0, 0.3)]
    assert build_chm(points, ground) == build_chm(list(reversed(points)), ground)


def test_results_sorted_lexicographically():
    ground = ((-2, -2, 0.0, 1.0, 1), (0, 0, 0.0, 1.0, 1), (-2, 1, 0.0, 1.0, 1))
    points = [(-1.5, -1.5, 1.0, 0.0, 1.0),
              (0.5, 0.5, 1.0, 0.0, 1.0),
              (-1.5, 1.5, 1.0, 0.0, 1.0)]
    assert [cell[:2] for cell in build_chm(points, ground)] == [
        (-2, -2), (-2, 1), (0, 0),
    ]


def test_accepts_tuples_lists_and_single_pass_iterators():
    class SinglePass:
        def __init__(self, items):
            self._items = iter(items)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._items)

    result = build_chm(SinglePass([[0.0, 0.0, 2.0, 0.0, 1.0]]),
                       SinglePass([[0, 0, 0.0, 1.0, 1]]))
    assert result == ((0, 0, 2.0, pytest.approx(math.sqrt(2)), 1),)


def test_outputs_quantized_to_six_decimals():
    result = build_chm([(0.0, 0.0, 1 / 3, 0.0, 1.0)],
                       ((0, 0, 0.0, 1.0, 1),))
    assert result[0][2] == round(1 / 3, 6)
    assert result[0][3] == round(math.sqrt(2), 6) == pytest.approx(math.sqrt(2), abs=1e-6)


def test_return_types_are_tuples():
    result = build_chm([(0.0, 0.0, 1.0, 0.0, 1.0)], ((0, 0, 0.0, 1.0, 1),))
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)


@pytest.mark.parametrize("bad_points", [123, None, 3.14, True])
def test_non_iterable_points_type_error(bad_points):
    with pytest.raises(TypeError):
        build_chm(bad_points, GROUND)


@pytest.mark.parametrize("bad_ground", [123, None, 3.14, True, "abc"])
def test_non_iterable_ground_type_error(bad_ground):
    with pytest.raises(TypeError):
        build_chm([], bad_ground)


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
        build_chm([point], GROUND)


@pytest.mark.parametrize("cell", [
    123,
    (0, 0, 0, 1),
    (0, 0, 0, 1, 1, 2),
    [0, 0, 0],
    {0, 0, 0, 1, 1},
    b"12345",
    "12345",
])
def test_bad_ground_container_or_length_type_error(cell):
    with pytest.raises(TypeError):
        build_chm([], [cell])


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
def test_illegal_point_scalar_type_error(point):
    with pytest.raises(TypeError):
        build_chm([point], GROUND)


@pytest.mark.parametrize("cell", [
    ("0", 0, 0, 1.0, 1),
    (0, [0], 0, 1.0, 1),
    (0, 0, None, 1.0, 1),
    (0, 0, 0, 1.0 + 0j, 1),
    (True, 0, 0, 1.0, 1),
    (0, False, 0, 1.0, 1),
    (0, 0, 0, 1.0, 1.0),
    (0, 0, 0, 1.0, True),
])
def test_illegal_ground_scalar_type_error(cell):
    with pytest.raises(TypeError):
        build_chm([], [cell])


@pytest.mark.parametrize("cell_size", ["1", [1.0], None])
def test_bad_cell_size_type(cell_size):
    with pytest.raises(TypeError):
        build_chm([(0, 0, 0, 0, 1.0)], GROUND, cell_size=cell_size)


def test_bool_cell_size_type_error():
    with pytest.raises(TypeError):
        build_chm([(0, 0, 0, 0, 1.0)], GROUND, cell_size=True)


@pytest.mark.parametrize("cell_size", [0, -1, -1.5, float("nan"),
                                      float("inf"), float("-inf")])
def test_bad_cell_size_value(cell_size):
    with pytest.raises(ValueError):
        build_chm([(0, 0, 0, 0, 1.0)], GROUND, cell_size=cell_size)


@pytest.mark.parametrize("sigma", [0, -1, -0.5, float("nan"),
                                   float("inf"), float("-inf")])
def test_bad_point_sigma_value(sigma):
    with pytest.raises(ValueError):
        build_chm([(0, 0, 0, 0, sigma)], GROUND)


@pytest.mark.parametrize("sigma", [0, -1, -0.5, float("nan"),
                                   float("inf"), float("-inf")])
def test_bad_ground_sigma_value(sigma):
    with pytest.raises(ValueError):
        build_chm([], ((0, 0, 0.0, sigma, 1),))


@pytest.mark.parametrize("count", [0, -1])
def test_bad_ground_count_value(count):
    with pytest.raises(ValueError):
        build_chm([], ((0, 0, 0.0, 1.0, count),))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_point_coordinates_value_error(value):
    with pytest.raises(ValueError):
        build_chm([(value, 0, 0, 0, 1.0)], GROUND)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_ground_z_value_error(value):
    with pytest.raises(ValueError):
        build_chm([], ((0, 0, value, 1.0, 1),))


def test_duplicate_ground_index_value_error():
    with pytest.raises(ValueError):
        build_chm([], ((0, 0, 0.0, 1.0, 1), (0, 0, 2.0, 1.0, 1)))


def test_point_cell_without_ground_value_error():
    with pytest.raises(ValueError):
        build_chm([(5, 5, 0.0, 0.0, 1.0)], GROUND)


def test_version_unchanged():
    assert __version__ == "0.1.0"
