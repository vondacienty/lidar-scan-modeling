"""Tests for :func:`lidar_scan.update_tile_pyramid_windows`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import (build_tile_pyramid, query_tile_pyramid_windows,
                        update_tile_pyramid, update_tile_pyramid_windows)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.update_tile_pyramid_windows
            is update_tile_pyramid_windows)
    import lidar_scan
    assert "update_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert update_tile_pyramid_windows(
        ((), (), ()), [(0.0, 0.0, 1.0, 0, 1.0)], ()) == ()
    assert update_tile_pyramid_windows(((), (), ()), iter(()), ()) == ()


def test_empty_points_queries_the_existing_pyramid():
    pyramid = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0, 1.0)], tile_cells=256, levels=2)
    windows = ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511))
    result = update_tile_pyramid_windows(
        pyramid, (), windows, tile_cells=256, levels=2)
    assert result == query_tile_pyramid_windows(pyramid, windows)
    # The empty-points path queries the pyramid itself; tiles are the same
    # objects.
    assert all(a is b for a, b in zip(result[0][5], pyramid[0]))


def test_window_without_matches_has_empty_tiles():
    pyramid = ((),)
    result = update_tile_pyramid_windows(
        pyramid, [(0.0, 0.0, 1.0, 0, 1.0)],
        ((0, 300, 300, 400, 400),), levels=1)
    assert result == ((0, 300, 300, 400, 400, ()),)


def test_new_points_appear_in_matched_tiles():
    result = update_tile_pyramid_windows(
        ((),),
        [(1.0, -2.0, 3.5, 10.0, 0.5)],
        ((0, -10, -10, 10, 10),),
        levels=1)
    assert result == (
        (0, -10, -10, 10, 10,
         ((0, -1, 0, -256, 255, -1, 3.5, 3.5, 1),)),
    )


def test_new_points_merge_with_existing_tiles():
    pyramid = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0, 1.0)], tile_cells=256, levels=1)
    points = [(255.0, 255.0, 7.25, 0, 1.0),
              (10.0, 20.0, -1.5, 0, 1.0)]
    result = update_tile_pyramid_windows(
        pyramid, points, ((0, 0, 0, 255, 255),), levels=1)
    assert result[0][5] == (
        (0, 0, 0, 0, 255, 255, -1.5, 7.25, 3),
    )


def test_matches_update_then_query_equivalence():
    initial = [(0.0, 0.0, 1.0, 0, 1.0),
               (300.0, 600.0, 4.0, 0, 0.5)]
    pyramid = build_tile_pyramid(initial, tile_cells=256, levels=2)
    points = [(-5.0, -5.0, -2.0, 0, 2.0),
              (700.0, 200.0, 9.0, 0, 1.0),
              (300.0, 10.0, 0.25, 0, 1.0)]
    windows = ((0, 0, 0, 255, 255),
               (0, 256, 0, 511, 1023),
               (1, -512, -512, 1023, 1023))
    result = update_tile_pyramid_windows(
        pyramid, points, windows, tile_cells=256, levels=2)
    updated = update_tile_pyramid(
        pyramid, points, tile_cells=256, levels=2)
    assert result == query_tile_pyramid_windows(updated, windows)


def test_windows_preserve_window_order():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = update_tile_pyramid_windows(((),), points, windows, levels=1)
    assert tuple(window[:5] for window in result) == windows


def test_multiple_levels_select_the_right_level():
    pyramid = build_tile_pyramid(
        [(10.0, 10.0, 1.0, 0, 1.0)], tile_cells=256, levels=2)
    points = [(600.0, 600.0, 5.0, 0, 1.0)]
    windows = ((0, 0, 0, 255, 255),
               (0, 512, 512, 767, 767),
               (1, 0, 0, 511, 511))
    result = update_tile_pyramid_windows(
        pyramid, points, windows, tile_cells=256, levels=2)
    assert [tile[:2] for tile in result[0][5]] == [(0, 0)]
    assert [tile[:2] for tile in result[1][5]] == [(2, 2)]
    # At level 1 the new point (600, 600) lands in tile (1, 1), so the
    # window covering 0..511 only contains the existing point's tile.
    assert result[2][5] == (
        (0, 0, 0, 0, 511, 511, 1.0, 1.0, 1),
    )


def test_closed_intervals_match_at_endpoints():
    points = [(0.0, 0.0, 1.0, 0, 1.0)]
    result = update_tile_pyramid_windows(
        ((),), points, ((0, 255, 255, 300, 300),), levels=1)
    assert len(result[0][5]) == 1
    result = update_tile_pyramid_windows(
        ((),), points, ((0, 256, 256, 300, 300),), levels=1)
    assert result[0][5] == ()


def test_quantized_to_six_decimal_places():
    result = update_tile_pyramid_windows(
        ((),), [(0.0, 0.0, 1 / 3, 0, 1.0)],
        ((0, 0, 0, 255, 255),), levels=1)
    tile = result[0][5][0]
    assert tile[6] == 0.333333
    assert tile[7] == 0.333333


def test_negative_zero_normalized():
    result = update_tile_pyramid_windows(
        ((),), [(-0.0, 0.0, -0.0, 0, 1.0)],
        ((0, 0, 0, 255, 255),), levels=1)
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
    result = update_tile_pyramid_windows(
        ((),), points,
        ((0, 0, 0, 255, 255), (0, 256, 0, 511, 255)),
        levels=1)
    assert points.iter_calls == 1
    assert len(result[0][5]) == 1
    assert len(result[1][5]) == 1


def test_results_independent_of_point_order():
    pyramid = build_tile_pyramid(
        [(10.0, 10.0, 3.0, 0, 1.0)], tile_cells=256, levels=1)
    points_a = [(0.0, 0.0, 1.0, 0, 1.0),
                (255.0, 255.0, 9.0, 0, 1.0),
                (20.0, 20.0, -4.0, 0, 1.0)]
    windows = ((0, 0, 0, 255, 255),)
    first = update_tile_pyramid_windows(
        pyramid, iter(points_a), windows, levels=1)
    second = update_tile_pyramid_windows(
        pyramid, iter(list(reversed(points_a))), windows, levels=1)
    assert first == second


def test_repeated_calls_are_deterministic():
    pyramid = build_tile_pyramid(
        [(5.0, 5.0, 2.0, 0, 1.0)], tile_cells=256, levels=2)
    points = [(x * 100.0, y * 100.0, float(x - y), 0, 1.0)
              for x in range(6) for y in range(6)]
    windows = ((0, 0, 0, 300, 300), (1, -50, -50, 900, 200))
    first = update_tile_pyramid_windows(
        pyramid, points, windows, tile_cells=256, levels=2)
    second = update_tile_pyramid_windows(
        pyramid, points, windows, tile_cells=256, levels=2)
    assert first == second


def test_inputs_are_not_modified():
    pyramid = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 0, 1.0)], tile_cells=256, levels=1)
    snapshot = tuple(
        tuple(tile) for level in pyramid for tile in level)
    points = [(255.0, 255.0, 7.0, 0, 1.0)]
    points_snapshot = tuple(points)
    windows = ((0, 0, 0, 255, 255),)
    update_tile_pyramid_windows(pyramid, points, windows, levels=1)
    assert tuple(
        tile for level in pyramid for tile in level) == snapshot
    assert tuple(points) == points_snapshot
    # The original pyramid's tiles keep their original count.
    assert pyramid[0][0][8] == 1


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_pyramid_raises_type_error(bad):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows(bad, (), ())


@pytest.mark.parametrize("bad", [None, 42, 1.5])
def test_points_not_iterable_raises_type_error(bad):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows(((), (), ()), bad, ())


@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_windows_not_tuple_raises_type_error(bad):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], bad)


@pytest.mark.parametrize("bad_window", [[0, 0, 0, 1, 1], None, 42])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], (tuple(window),))


@pytest.mark.parametrize("bad_point", [None, 42, (1, 2, 3)])
def test_point_wrong_container_or_length_raises_type_error(bad_point):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows(((), (), ()), iter([bad_point]), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, "1.0", None])
def test_point_field_wrong_type_raises_type_error(pos, bad_value):
    point = [0.0, 0.0, 0.0, 0.0, 1.0]
    point[pos] = bad_value
    with pytest.raises(TypeError):
        update_tile_pyramid_windows(((), (), ()), iter([tuple(point)]), ())


@pytest.mark.parametrize("bad_value", [True, "1.0", None])
def test_cell_size_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1", None])
def test_tile_cells_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], (), tile_cells=bad_value)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1", None])
def test_levels_wrong_type_raises_type_error(bad_value):
    with pytest.raises(TypeError):
        update_tile_pyramid_windows((), [], (), levels=bad_value)


# ---------------------------------------------------------------------------
# ValueError: parameters and points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            ((),), [], ((level, 0, 0, 1, 1),), levels=1)


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((),), [], ((0, 5, 0, 1, 1),), levels=1)
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((),), [], ((0, 0, 5, 1, 1),), levels=1)


@pytest.mark.parametrize("pos", [0, 1, 2, 3])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_nonfinite_point_field_raises_value_error(pos, bad_value):
    point = [0.0, 0.0, 0.0, 0.0, 1.0]
    point[pos] = bad_value
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((), (), ()), iter([tuple(point)]), ())


@pytest.mark.parametrize("bad_sigma", [0, 0.0, -1, -0.5])
def test_nonpositive_sigma_raises_value_error(bad_sigma):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            ((), (), ()),
            iter([(0.0, 0.0, 0.0, 0.0, bad_sigma)]), ())


@pytest.mark.parametrize("bad_value", [0, 0.0, -1, -2.5])
def test_nonpositive_cell_size_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows((), [], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_nonfinite_cell_size_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows((), [], (), cell_size=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, -256])
def test_nonpositive_tile_cells_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows((), [], (), tile_cells=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, -3])
def test_nonpositive_levels_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            ((),), [], ((0, 0, 0, 1, 1),), levels=bad_value)


def test_invalid_point_during_iteration_propagates_error():
    def points():
        yield (0.0, 0.0, 1.0, 0.0, 1.0)
        yield (0.0, 0.0, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sigma"):
        update_tile_pyramid_windows(((), (), ()), points(), ())


# ---------------------------------------------------------------------------
# ValueError: pyramid structure / level count / tile bounds
# ---------------------------------------------------------------------------

def test_level_count_mismatch_raises_value_error():
    pyramid = ((), ())
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            pyramid, [], ((0, 0, 0, 1, 1),), levels=3)


def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(([_tile()],), [], (), levels=1)


@pytest.mark.parametrize("bad_tile", [
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),            # too few fields
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 0),      # too many fields
    [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],         # list, not tuple
])
def test_tile_wrong_shape_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((bad_tile,),), [], (), levels=1)


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [1.5, "1", None])
def test_non_int_index_or_count_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((tuple(tile),),), [], (), levels=1)


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [1, "1.0", None, float("nan"), float("inf")])
def test_bad_z_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(((tuple(tile),),), [], (), levels=1)


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(unsorted, [], (), levels=1)

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(duplicate, [], (), levels=1)


def test_tile_bounds_inconsistent_with_tile_cells_raise_value_error():
    # A level-0 tile claiming a width of 4 cells.
    bad = ((_tile(ix1=3, iy1=3),),)
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            bad, [], ((0, 0, 0, 1, 1),), tile_cells=256, levels=1)


def test_tile_bounds_inconsistent_with_level_raise_value_error():
    # Two levels with the right count, but the level-1 tile claims a width of
    # 256 cells where level 1 must cover 512.
    bad = (
        (_tile(),),
        (_tile(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255),),
    )
    with pytest.raises(ValueError):
        update_tile_pyramid_windows(
            bad, [], ((1, 0, 0, 1, 1),), tile_cells=256, levels=2)
