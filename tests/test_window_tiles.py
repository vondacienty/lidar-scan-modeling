"""Tests for :func:`lidar_scan.tiles.window_tiles`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import window_tiles
from lidar_scan import tiles as tiles_module


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.window_tiles is window_tiles
    import lidar_scan
    assert "window_tiles" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

POINTS = [(0.5, 0.5, 1.0, 0, 1.0),
          (3.9, 3.9, 5.0, 0, 1.0),
          (7.5, 7.5, 9.0, 0, 1.0),
          (8.1, 1.5, -2.0, 0, 1.0)]
WINDOWS = ((0, 1, 1, 3, 3),
           (1, 4, 4, 7, 7),
           (1, 8, 0, 9, 0),
           (0, 100, 100, 101, 101))


def test_empty_windows_returns_empty_tuple():
    assert window_tiles(POINTS, (), tile_cells=4, levels=2) == ()


def test_empty_points_gives_empty_tiles_per_window():
    result = window_tiles((), WINDOWS, tile_cells=4, levels=2)
    assert result == tuple(w + ((),) for w in WINDOWS)


def test_windows_echoed_in_order():
    result = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    assert [r[0:5] for r in result] == list(WINDOWS)


def test_level_zero_window_intersection():
    result = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    # Window [1..3] x [1..3] intersects only tile (0, 0) covering cells
    # [0..3] x [0..3]; the (8.1, 1.5) point belongs to tile (2, 0).
    assert result[0] == (
        0, 1, 1, 3, 3,
        ((0, 0, 0, 0, 3, 3, 1.0, 5.0, 2),),
    )


def test_coarse_level_window_intersection():
    result = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    # Level 1 width 8: window [4..7] x [4..7] intersects tile (0, 0) which
    # contains the first three points.
    assert result[1] == (
        1, 4, 4, 7, 7,
        ((0, 0, 0, 0, 7, 7, 1.0, 9.0, 3),),
    )
    # Window [8..9] x [0..0] intersects tile (1, 0) covering [8..15]x[0..7].
    assert result[2] == (
        1, 8, 0, 9, 0,
        ((1, 0, 8, 0, 15, 7, -2.0, -2.0, 1),),
    )


def test_no_match_window_has_empty_tiles():
    result = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    assert result[3] == (0, 100, 100, 101, 101, ())


def test_touching_boundary_is_closed_interval_intersection():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    # Window starts exactly on the tile's last cell: still intersects.
    result = window_tiles(points, ((0, 3, 3, 3, 3),), tile_cells=4, levels=1)
    assert result[0][5] == ((0, 0, 0, 0, 3, 3, 1.0, 1.0, 1),)
    # One cell beyond no longer intersects.
    result = window_tiles(points, ((0, 4, 4, 4, 4),), tile_cells=4, levels=1)
    assert result[0][5] == ()


def test_negative_cell_coordinates_floor():
    result = window_tiles([(-0.1, -0.1, 2.0, 0, 1.0)],
                          ((0, -4, -4, -1, -1),), tile_cells=4, levels=1)
    assert result[0][5] == ((-1, -1, -4, -4, -1, -1, 2.0, 2.0, 1),)


def test_tiles_sorted_and_order_independent():
    windows = ((1, 0, 0, 15, 15),)
    forward = window_tiles(POINTS, windows, tile_cells=4, levels=2)
    backward = window_tiles(list(reversed(POINTS)), windows,
                            tile_cells=4, levels=2)
    assert forward == backward
    keys = [tile[:2] for tile in forward[0][5]]
    assert keys == sorted(keys)


def test_shared_tile_between_windows_counted_once_each():
    result = window_tiles(POINTS, ((0, 0, 0, 0, 0), (0, 3, 3, 3, 3)),
                          tile_cells=4, levels=1)
    tile = (0, 0, 0, 0, 3, 3, 1.0, 5.0, 2)
    assert result[0][5] == (tile,)
    assert result[1][5] == (tile,)


def test_accepts_iterator_and_consumed_single_pass():
    result = window_tiles(iter(POINTS), WINDOWS, tile_cells=4, levels=2)
    expected = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    assert result == expected


def test_zmin_zmax_quantized_six_decimals_and_negative_zero():
    result = window_tiles([(0.0, 0.0, -1e-12, 0, 1.0)],
                          ((0, 0, 0, 0, 0),), tile_cells=4, levels=1)
    tile = result[0][5][0]
    assert tile[6] == 0.0
    assert math.copysign(1.0, tile[6]) == 1.0
    result = window_tiles([(0.0, 0.0, 0.0000005, 0, 1.0),
                           (0.0, 0.0, 0.0000015, 0, 1.0)],
                          ((0, 0, 0, 0, 0),), tile_cells=4, levels=1)
    assert result[0][5][0][6:8] == (0.0, 0.000002)


def test_return_types_are_tuples():
    result = window_tiles(POINTS, WINDOWS, tile_cells=4, levels=2)
    assert isinstance(result, tuple)
    for window in result:
        assert isinstance(window, tuple) and len(window) == 6
        assert isinstance(window[5], tuple)
        for tile in window[5]:
            assert isinstance(tile, tuple) and len(tile) == 9


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_points", [None, 1, 1.5, True, object()])
def test_non_iterable_points_type_error(bad_points):
    with pytest.raises(TypeError):
        window_tiles(bad_points, ((0, 0, 0, 1, 1),), tile_cells=4)


@pytest.mark.parametrize("windows", [None, [], [()], ((),)])
def test_windows_container_type_error(windows):
    with pytest.raises(TypeError):
        window_tiles(POINTS, windows, tile_cells=4)


@pytest.mark.parametrize("window", [
    [0, 0, 0, 1, 1],          # not a tuple
    (0, 0, 0, 1),             # too short
    (0, 0, 0, 1, 1, 0),       # too long
])
def test_bad_window_structure_type_error(window):
    with pytest.raises(TypeError):
        window_tiles(POINTS, (window,), tile_cells=4)


@pytest.mark.parametrize("window", [
    (0.0, 0, 0, 1, 1),
    (0, 0.0, 0, 1, 1),
    (0, 0, True, 1, 1),
    (False, 0, 0, 1, 1),
    (0, 0, 0, "1", 1),
])
def test_bad_window_field_type_error(window):
    with pytest.raises(TypeError):
        window_tiles(POINTS, (window,), tile_cells=4)


@pytest.mark.parametrize("level", [-1, 3])
def test_level_out_of_range_value_error(level):
    with pytest.raises(ValueError):
        window_tiles(POINTS, ((level, 0, 0, 1, 1),), tile_cells=4, levels=3)


@pytest.mark.parametrize("window", [
    (0, 3, 0, 2, 1),          # ix_min > ix_max
    (0, 0, 3, 1, 2),          # iy_min > iy_max
])
def test_inverted_window_bounds_value_error(window):
    with pytest.raises(ValueError):
        window_tiles(POINTS, (window,), tile_cells=4)


@pytest.mark.parametrize("point", [
    None, 1, (1, 1, 1, 1), [1, 1, 1, 1], (1, 1, 1, 1, 1, 1),
])
def test_bad_point_structure_type_error(point):
    with pytest.raises(TypeError):
        window_tiles([point], ((0, 0, 0, 1, 1),), tile_cells=4)


@pytest.mark.parametrize("point", [
    ("1", 0, 0, 0, 1),
    (0, None, 0, 0, 1),
    (0, 0, True, 0, 1),
    (0, 0, 0, 0, False),
])
def test_bad_point_field_type_error(point):
    with pytest.raises(TypeError):
        window_tiles([point], ((0, 0, 0, 1, 1),), tile_cells=4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"),
                                   float("-inf")])
def test_nonfinite_point_field_value_error(value):
    with pytest.raises(ValueError):
        window_tiles([(value, 0, 0, 0, 1)], ((0, 0, 0, 1, 1),),
                     tile_cells=4)


@pytest.mark.parametrize("sigma", [0, -1, 0.0, -0.5])
def test_sigma_must_be_positive(sigma):
    with pytest.raises(ValueError):
        window_tiles([(0, 0, 0, 0, sigma)], ((0, 0, 0, 1, 1),),
                     tile_cells=4)


@pytest.mark.parametrize("cell_size", [True, "1", None])
def test_bad_cell_size_type(cell_size):
    with pytest.raises(TypeError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), cell_size=cell_size)


@pytest.mark.parametrize("cell_size", [0, -1.0, float("nan"), float("inf")])
def test_bad_cell_size_value(cell_size):
    with pytest.raises(ValueError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), cell_size=cell_size)


@pytest.mark.parametrize("tile_cells", [True, 1.0, 4.0, "4"])
def test_bad_tile_cells_type(tile_cells):
    with pytest.raises(TypeError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), tile_cells=tile_cells)


@pytest.mark.parametrize("tile_cells", [0, -1])
def test_bad_tile_cells_value(tile_cells):
    with pytest.raises(ValueError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), tile_cells=tile_cells)


@pytest.mark.parametrize("levels", [True, 1.0, 3.0, "3"])
def test_bad_levels_type(levels):
    with pytest.raises(TypeError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), levels=levels)


@pytest.mark.parametrize("levels", [0, -2])
def test_bad_levels_value(levels):
    with pytest.raises(ValueError):
        window_tiles(POINTS, ((0, 0, 0, 1, 1),), levels=levels)


def test_version_unchanged():
    from lidar_scan import __version__
    assert __version__ == "0.1.0"
