"""Tests for :func:`lidar_scan.voxel.fuse_voxels`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import __version__, fuse_voxels


def test_empty_returns_empty_tuple():
    assert fuse_voxels([]) == ()
    assert fuse_voxels(iter(())) == ()


def test_single_point_passthrough():
    result = fuse_voxels([(1.0, -2.0, 3.5, 10.0, 0.5)])
    assert result == ((1, -2, 3, 1.0, -2.0, 3.5, 10.0, 0.5, 1),)


def test_indices_floor_toward_negative_infinity():
    result = fuse_voxels([(-0.1, -1.0, 2.9, 1.0, 1.0),
                          (0.1, -1.0, -2.9, 2.0, 1.0)])
    assert [point[:3] for point in result] == [(-1, -1, 2), (0, -1, -3)]


def test_voxel_size_divides_coordinates():
    result = fuse_voxels([(0.9, 0.0, 0.0, 1.0, 1.0),
                          (1.9, 0.0, 0.0, 2.0, 1.0)], voxel_size=2)
    assert result == ((0, 0, 0, 1.4, 0.0, 0.0, 1.5,
                       pytest.approx(1 / math.sqrt(2)), 2),)
    # exactly on a boundary belongs to the higher voxel
    boundary = fuse_voxels([(2.0, 0.0, 0.0, 0.0, 1.0)], voxel_size=2)
    assert boundary[0][:3] == (1, 0, 0)


def test_weighted_mean_and_fused_sigma_equal_weights():
    result = fuse_voxels([(0.0, 0.0, 0.0, 0.0, 2.0),
                          (0.5, 0.5, 0.5, 10.0, 2.0)])
    assert result == ((0, 0, 0, 0.25, 0.25, 0.25, 5.0,
                       pytest.approx(1 / math.sqrt(0.5)), 2),)


def test_weighted_mean_unequal_weights():
    # sigma 1.0 -> w=1, sigma 0.5 -> w=4
    result = fuse_voxels([(0.0, 0.0, 0.0, 0.0, 1.0),
                          (0.5, 0.5, 0.5, 10.0, 0.5)])
    point = result[0]
    assert point[:3] == (0, 0, 0)
    assert point[3:7] == (0.4, 0.4, 0.4, 8.0)
    assert point[7] == pytest.approx(math.sqrt(0.2), abs=1e-6)
    assert point[8] == 2


def test_results_sorted_lexicographically():
    result = fuse_voxels([(5, 5, 5, 0, 1),
                          (1, 1, 1, 0, 1),
                          (1, 0, 1, 0, 1),
                          (1, 0, 0, 0, 1)])
    assert [point[:3] for point in result] == [
        (1, 0, 0), (1, 0, 1), (1, 1, 1), (5, 5, 5),
    ]


def test_results_independent_of_input_order():
    points = [(0.1, 0.2, 0.3, 1.0, 0.3),
              (0.4, 0.3, 0.2, 9.0, 0.7),
              (0.25, 0.25, 0.25, 5.0, 1.1)]
    assert fuse_voxels(points) == fuse_voxels(list(reversed(points)))


def test_accepts_tuples_lists_and_iterators():
    result = fuse_voxels(iter([(0.0, 0.0, 0.0, 0.0, 1.0),
                               [0.2, 0.2, 0.2, 2.0, 1.0]]))
    assert result[0][8] == 2
    assert result[0][6] == 1.0


def test_outputs_quantized_to_six_decimals():
    result = fuse_voxels([(1 / 3, 1 / 7, 0.0, 0.0, 1.0)])
    assert result[0][3] == round(1 / 3, 6)
    assert result[0][4] == round(1 / 7, 6)


def test_negative_zero_normalized():
    result = fuse_voxels([(-0.0, 0.0, -0.0, -0.0, 1)])
    for index in (3, 4, 5, 6):
        assert result[0][index] == 0.0
        assert math.copysign(1.0, result[0][index]) == 1.0


def test_return_types_are_tuples():
    result = fuse_voxels([(0.0, 0.0,0.0, 0.0, 1.0)])
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
        fuse_voxels(bad_points)


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
        fuse_voxels([point])


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
        fuse_voxels([point])


@pytest.mark.parametrize("voxel_size", ["1", [1.0], None, True])
def test_bad_voxel_size_type(voxel_size):
    with pytest.raises(TypeError):
        fuse_voxels([(0, 0, 0, 0, 1.0)], voxel_size=voxel_size)


@pytest.mark.parametrize("voxel_size", [0, -1, -1.5, float("nan"),
                                        float("inf"), float("-inf")])
def test_bad_voxel_size_value(voxel_size):
    with pytest.raises(ValueError):
        fuse_voxels([(0, 0, 0, 0, 1.0)], voxel_size=voxel_size)


@pytest.mark.parametrize("sigma", [0, -1, -0.5, float("nan"),
                                   float("inf"), float("-inf")])
def test_bad_sigma_value(sigma):
    with pytest.raises(ValueError):
        fuse_voxels([(0, 0, 0, 0, sigma)])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_value_error(value):
    with pytest.raises(ValueError):
        fuse_voxels([(value, 0, 0, 0, 1.0)])


def test_version_unchanged():
    assert __version__ == "0.1.0"
