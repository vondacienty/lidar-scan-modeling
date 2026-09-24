"""Tests for :func:`lidar_scan.query_tile_pyramid_delta`."""

from __future__ import annotations

import copy

import pytest

from lidar_scan import (assess_tile_pyramid_deltas, query_tile_pyramid_delta)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _assessment():
    estimate = (
        (_tile(tx=0, ty=0, zmin=3.0, zmax=5.0, count=8),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=4.0, zmax=6.0, count=3)),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, zmin=9.0, zmax=9.0),),
    )
    reference = (
        (_tile(tx=0, ty=0, zmin=1.0, count=2),
         _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=2.0, zmax=5.0, count=7)),
        (_tile(ix0=0, iy0=0, ix1=511, iy1=511, zmin=4.0, zmax=9.0),),
    )
    return assess_tile_pyramid_deltas(estimate, reference)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.query_tile_pyramid_delta is query_tile_pyramid_delta
    import lidar_scan
    assert "query_tile_pyramid_delta" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_hit_returns_the_stored_tile_object():
    assessment = _assessment()
    assert query_tile_pyramid_delta(assessment, 0, 0, 0) is assessment[0][0]
    assert query_tile_pyramid_delta(assessment, 0, 1, 0) is assessment[0][1]
    assert query_tile_pyramid_delta(assessment, 1, 0, 0) is assessment[1][0]
    assert query_tile_pyramid_delta(assessment, 0, 0, 0) == \
        (0, 0, 0, 0, 255, 255, 2.0, 3.0, 6)


def test_missing_coordinate_returns_none():
    assessment = _assessment()
    assert query_tile_pyramid_delta(assessment, 0, 2, 0) is None
    assert query_tile_pyramid_delta(assessment, 0, 0, 1) is None
    assert query_tile_pyramid_delta(assessment, 1, 1, 0) is None


def test_empty_level_returns_none():
    assessment = assess_tile_pyramid_deltas(
        ((), (_tile(),)), ((), (_tile(zmin=2.0, zmax=3.0, count=4),)))
    assert query_tile_pyramid_delta(assessment, 0, 0, 0) is None
    assert query_tile_pyramid_delta(assessment, 1, 0, 0) is assessment[1][0]


def test_repeated_calls_are_stable_and_input_is_untouched():
    assessment = _assessment()
    before = copy.deepcopy(assessment)
    first = query_tile_pyramid_delta(assessment, 0, 0, 0)
    second = query_tile_pyramid_delta(assessment, 0, 0, 0)
    assert first is second
    assert query_tile_pyramid_delta(assessment, 0, 1, 0) == \
        query_tile_pyramid_delta(assessment, 0, 1, 0)
    assert assessment == before


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_assessment_raises_type_error(bad):
    with pytest.raises(TypeError):
        query_tile_pyramid_delta(bad, 0, 0, 0)


@pytest.mark.parametrize("bad", [True, 1.5, "0", None])
def test_non_int_arguments_raise_type_error(bad):
    assessment = _assessment()
    with pytest.raises(TypeError):
        query_tile_pyramid_delta(assessment, bad, 0, 0)
    with pytest.raises(TypeError):
        query_tile_pyramid_delta(assessment, 0, bad, 0)
    with pytest.raises(TypeError):
        query_tile_pyramid_delta(assessment, 0, 0, bad)


# ---------------------------------------------------------------------------
# ValueError: level range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 2, 3])
def test_level_out_of_range_raises_value_error(level):
    assessment = _assessment()
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(assessment, level, 0, 0)


def test_empty_assessment_level_always_out_of_range():
    with pytest.raises(ValueError):
        query_tile_pyramid_delta((), 0, 0, 0)


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    tile = _assessment()[0][0]
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(([tile],), 0, 0, 0)


@pytest.mark.parametrize("bad_tile", [
    [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 99),
])
def test_tile_container_or_length_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((bad_tile,),), 0, 0, 0)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_integer_fields_wrong_type_raise_value_error(pos, bad_value):
    tile = list(_assessment()[0][0])
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tuple(tile),),), 0, 0, 0)


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, True, "1.0", None])
def test_delta_fields_must_be_floats(pos, bad_value):
    tile = list(_assessment()[0][0])
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tuple(tile),),), 0, 0, 0)


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_delta_raises_value_error(pos, bad_value):
    tile = list(_assessment()[0][0])
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tuple(tile),),), 0, 0, 0)


def test_unsorted_coordinates_raise_value_error():
    assessment = _assessment()
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(
            ((assessment[0][1], assessment[0][0]),), 0, 0, 0)


def test_duplicate_coordinates_raise_value_error():
    tile = _assessment()[0][0]
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tile, tile),), 0, 0, 0)


def test_inverted_cell_bounds_raise_value_error():
    tile = list(_assessment()[0][0])
    tile[4] = tile[2] - 1
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tuple(tile),),), 0, 0, 0)
    tile = list(_assessment()[0][0])
    tile[5] = tile[3] - 1
    with pytest.raises(ValueError):
        query_tile_pyramid_delta(((tuple(tile),),), 0, 0, 0)
