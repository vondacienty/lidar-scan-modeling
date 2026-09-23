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

def test_empty_windows_return_empty_tuple():
    assert window_tiles([(0.0, 0.0, 1.0, 0, 1.0)], ()) == ()
    assert window_tiles(iter(()), ()) == ()


def test_empty_points_keep_windows_with_empty_tiles():
    result = window_tiles([], ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, ()),)


def test_window_without_matches_has_empty_tiles():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    result = window_tiles(points, ((0, 300, 300, 400, 400),))
    assert result == ((0, 300, 300, 400, 400, ()),)


def test_single_point_tile():
    result = window_tiles(
        [(1.0, -2.0, 3.5, 10.0, 0.5)], ((0, -10, -10, 10, 10),))
    assert result == (
        (0, -10, -10, 10, 10,
         ((0, -1, 0, -256, 255, -1, 3.5, 3.5, 1),)),
    )


def test_window_echoes_its_arguments():
    result = window_tiles(
        [(0.0, 0.0, 1.0, 0, 1.0)], ((2, -10, -20, 30, 40),),
        tile_cells=256, levels=3)
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = window_tiles(points, windows)
    assert tuple(window[:5] for window in result) == windows


def test_aggregate_zmin_zmax_count_in_a_tile():
    points = [(0.0, 0.0, 3.0, 0, 1.0),
              (10.0, 20.0, -1.5, 0, 1.0),
              (255.0, 255.0, 7.25, 0, 1.0)]
    result = window_tiles(points, ((0, 0, 0, 255, 255),))
    assert result[0][5] == (
        (0, 0, 0, 0, 255, 255, -1.5, 7.25, 3),
    )


def test_tiles_are_sorted_by_tx_ty():
    points = [(600.0, 600.0, 1.0, 0, 1.0),
              (-10.0, -10.0, 1.0, 0, 1.0),
              (300.0, 0.0, 1.0, 0, 1.0)]
    result = window_tiles(
        points, ((0, -256, -256, 1023, 1023),))
    assert [tile[:2] for tile in result[0][5]] == [
        (-1, -1), (1, 0), (2, 2)
    ]


def test_only_tiles_intersecting_the_window_are_returned():
    points = [(0.0, 0.0, 1.0, 0, 1.0),
              (300.0, 0.0, 2.0, 0, 1.0)]
    result = window_tiles(points, ((0, 256, 0, 300, 255),))
    tiles = result[0][5]
    assert [tile[:2] for tile in tiles] == [(1, 0)]
    assert tiles[0][2:6] == (256, 0, 511, 255)


def test_closed_intervals_match_at_endpoints():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    # Window touching the tile only at (255, 255) still intersects.
    result = window_tiles(points, ((0, 255, 255, 300, 300),))
    assert len(result[0][5]) == 1
    # Window starting just past the tile does not intersect.
    result = window_tiles(points, ((0, 256, 256, 300, 300),))
    assert result[0][5] == ()


def test_multiple_levels_use_doubled_widths():
    points = [(600.0, 0.0, 1.0, 0, 1.0)]
    result = window_tiles(
        points, ((0, 0, 0, 600, 255), (1, 0, 0, 600, 511)),
        tile_cells=256, levels=2)
    # Level 0: point ix = 600 -> tx = 2, two tiles cannot merge.
    level0_tiles = result[0][5]
    assert [tile[:2] for tile in level0_tiles] == [(2, 0)]
    assert level0_tiles[0][2:6] == (512, 0, 767, 255)
    # Level 1: width 512 -> tx = 1, the single tile spans 512..1023.
    level1_tiles = result[1][5]
    assert [tile[:2] for tile in level1_tiles] == [(1, 0)]
    assert level1_tiles[0][2:6] == (512, 0, 1023, 511)


def test_points_across_levels_aggregate_simultaneously():
    points = [(0.0, 0.0, 1.0, 0, 1.0),
              (300.0, 0.0, 3.0, 0, 1.0)]
    result = window_tiles(
        points, ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511)),
        tile_cells=256, levels=2)
    assert result[0][5] == (
        (0, 0, 0, 0, 255, 255, 1.0, 1.0, 1),
    )
    assert result[1][5] == (
        (0, 0, 0, 0, 511, 511, 1.0, 3.0, 2),
    )


def test_cell_indices_floor_toward_negative_infinity():
    result = window_tiles(
        [(-0.1, 0.0, 1.0, 0, 1.0), (0.1, 0.0, 2.0, 0, 1.0)],
        ((0, -10, -10, 10, 10),), tile_cells=4)
    coords = [tile[:2] for tile in result[0][5]]
    assert coords == [(-1, 0), (0, 0)]


def test_cell_size_divides_coordinates():
    result = window_tiles(
        [(3.9, 0.0, 1.0, 0, 1.0), (4.1, 0.0, 2.0, 0, 1.0)],
        ((0, 0, 0, 3, 3),), cell_size=2, tile_cells=2)
    assert [tile[:2] for tile in result[0][5]] == [(0, 0), (1, 0)]


def test_quantized_to_six_decimal_places():
    result = window_tiles(
        [(0.0, 0.0, 1 / 3, 0, 1.0)], ((0, 0, 0, 255, 255),))
    tile = result[0][5][0]
    assert tile[6] == 0.333333
    assert tile[7] == 0.333333


def test_negative_zero_normalized():
    result = window_tiles(
        [(-0.0, 0.0, -0.0, 0, 1.0)], ((0, 0, 0, 255, 255),))
    tile = result[0][5][0]
    assert tile[6] == 0.0 and math.copysign(1.0, tile[6]) == 1.0
    assert tile[7] == 0.0 and math.copysign(1.0, tile[7]) == 1.0


def test_points_consumed_in_a_single_pass():
    class OneShotPoints:
        def __init__(self, values):
            self._values = values
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            yield from self._values

    points = OneShotPoints([(0.0, 0.0, 1.0, 0, 1.0),
                            (300.0, 0.0, 2.0, 0, 1.0)])
    result = window_tiles(
        points, ((0, 0, 0, 255, 255), (0, 256, 0, 511, 255)))
    assert points.iter_calls == 1
    assert len(result[0][5]) == 1
    assert len(result[1][5]) == 1


def test_results_match_build_tile_pyramid_then_window_query():
    points = [(0.0, 0.0, 1.0, 0, 1.0),
              (300.0, 600.0, 4.0, 0, 0.5),
              (-5.0, -5.0, -2.0, 0, 2.0),
              (700.0, 200.0, 9.0, 0, 1.0)]
    windows = ((0, 0, 0, 255, 255),
               (0, 256, 0, 511, 1023),
               (1, -512, -512, 1023, 1023))
    result = window_tiles(points, windows, tile_cells=256, levels=2)
    pyramid = tiles_module.build_tile_pyramid(
        points, tile_cells=256, levels=2)
    for (level, ix_min, iy_min, ix_max, iy_max, tiles), window in zip(
            result, windows):
        expected = tiles_module.query_tile_window(
            pyramid, level, ix_min, iy_min, ix_max, iy_max)
        assert tiles == expected


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 42, 1.5])
def test_points_not_iterable_raises_type_error(bad):
    with pytest.raises(TypeError):
        window_tiles(bad, ())


@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_windows_not_tuple_raises_type_error(bad):
    with pytest.raises(TypeError):
        window_tiles([], bad)


@pytest.mark.parametrize("bad_window", [[0, 0, 0, 1, 1], None, 42])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        window_tiles([], (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        window_tiles([], ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        window_tiles([], ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        window_tiles([], (tuple(window),))


@pytest.mark.parametrize("bad_point", [None, 42, (1, 2, 3)])
def test_point_wrong_container_or_length_raises_type_error(bad_point):
    with pytest.raises(TypeError):
        window_tiles(iter([bad_point]), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, "1.0", None])
def test_point_field_wrong_type_raises_type_error(pos, bad_value):
    point = [0.0, 0.0, 0.0, 0.0, 1.0]
    point[pos] = bad_value
    with pytest.raises(TypeError):
        window_tiles(iter([tuple(point)]), ())


@pytest.mark.parametrize("bad_value", [True, "1.0", None])
def test_cell_size_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        window_tiles([], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1", None])
def test_tile_cells_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        window_tiles([], (), tile_cells=bad_value)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1", None])
def test_levels_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        window_tiles([], (), levels=bad_value)


# ---------------------------------------------------------------------------
# ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 3, 100])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        window_tiles([], ((level, 0, 0, 1, 1),), levels=3)


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        window_tiles([], ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        window_tiles([], ((0, 0, 5, 1, 1),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_nonfinite_point_field_raises_value_error(pos, bad_value):
    point = [0.0, 0.0, 0.0, 0.0, 1.0]
    point[pos] = bad_value
    with pytest.raises(ValueError):
        window_tiles(iter([tuple(point)]), ())


@pytest.mark.parametrize("bad_sigma", [0, 0.0, -1, -0.5])
def test_nonpositive_sigma_raises_value_error(bad_sigma):
    with pytest.raises(ValueError):
        window_tiles(iter([(0.0, 0.0, 0.0, 0.0, bad_sigma)]), ())


@pytest.mark.parametrize("bad_value", [0, 0.0, -1, -2.5])
def test_nonpositive_cell_size_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        window_tiles([], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_nonfinite_cell_size_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        window_tiles([], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, -256])
def test_nonpositive_tile_cells_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        window_tiles([], (), tile_cells=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, -3])
def test_nonpositive_levels_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        window_tiles([], ((0, 0, 0, 1, 1),), levels=bad_value)


def test_invalid_point_during_iteration_propagates_error():
    def points():
        yield (0.0, 0.0, 1.0, 0.0, 1.0)
        yield (0.0, 0.0, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sigma"):
        window_tiles(points(), ())
