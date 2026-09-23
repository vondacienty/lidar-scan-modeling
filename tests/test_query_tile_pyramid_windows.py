"""Tests for :func:`lidar_scan.query_tile_pyramid_windows`."""

from __future__ import annotations

import pytest

from lidar_scan import (build_tile_pyramid, query_tile_pyramid,
                        query_tile_pyramid_windows, query_tile_window)
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
    assert tiles_module.query_tile_pyramid_windows is query_tile_pyramid_windows
    import lidar_scan
    assert "query_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_windows_returns_empty_tuple():
    assert query_tile_pyramid_windows((), ()) == ()
    assert query_tile_pyramid_windows(((), (), ()), ()) == ()


def test_window_without_matches_keeps_window_and_empty_tiles():
    pyramid = ((),)
    result = query_tile_pyramid_windows(pyramid, ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, ()),)


def test_empty_pyramid_levels_are_allowed():
    pyramid = ((), (), ())
    result = query_tile_pyramid_windows(
        pyramid, ((0, 0, 0, 10, 10), (2, -5, -5, 5, 5)))
    assert result == (
        (0, 0, 0, 10, 10, ()),
        (2, -5, -5, 5, 5, ()),
    )


def test_single_window_returns_matching_tiles():
    pyramid = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    result = query_tile_pyramid_windows(pyramid, ((0, 0, 0, 600, 600),))
    assert result == (
        (0, 0, 0, 600, 600, pyramid[0]),
    )


def test_only_tiles_intersecting_the_window_are_returned():
    pyramid = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    result = query_tile_pyramid_windows(pyramid, ((0, 256, 0, 300, 255),))
    assert result[0][5] == (_tile(tx=1, ty=0, ix0=256, ix1=511),)


def test_closed_intervals_match_at_endpoints():
    pyramid = ((_tile(),),)
    # Window touching the tile only at (255, 255) still intersects.
    result = query_tile_pyramid_windows(
        pyramid, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (_tile(),)
    # Window starting just past the tile does not intersect.
    result = query_tile_pyramid_windows(pyramid, ((0, 256, 0, 300, 255),))
    assert result[0][5] == ()
    result = query_tile_pyramid_windows(pyramid, ((0, 0, 256, 255, 300),))
    assert result[0][5] == ()


def test_window_echoes_its_arguments():
    pyramid = ((), (), (_tile(),))
    result = query_tile_pyramid_windows(
        pyramid, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    pyramid = ((_tile(),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = query_tile_pyramid_windows(pyramid, windows)
    assert tuple(window[:5] for window in result) == windows


def test_tiles_preserve_pyramid_order():
    pyramid = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=1, iy0=256, iy1=511),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    result = query_tile_pyramid_windows(pyramid, ((0, 0, 0, 600, 600),))
    assert result[0][5] == pyramid[0]
    assert [tile[0:2] for tile in result[0][5]] == [
        (0, 0), (0, 1), (1, 0)]


def test_multiple_levels_select_the_right_level():
    pyramid = (
        (_tile(),),
        (_tile(ix1=511, iy1=511),),
    )
    result = query_tile_pyramid_windows(
        pyramid,
        ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511),
         (0, 300, 300, 400, 400)))
    assert result[0][5] == (_tile(),)
    assert result[1][5] == (_tile(ix1=511, iy1=511),)
    assert result[2][5] == ()


def test_duplicate_windows_are_allowed_and_independent():
    pyramid = ((_tile(),),)
    windows = ((0, 0, 0, 255, 255), (0, 0, 0, 255, 255))
    result = query_tile_pyramid_windows(pyramid, windows)
    assert result[0] == result[1]
    assert result[0][5] is not result[1][5]


def test_returned_tiles_are_the_original_tile_objects():
    pyramid = ((_tile(), _tile(tx=1, ty=0, ix0=256, ix1=511)),)
    result = query_tile_pyramid_windows(pyramid, ((0, 0, 0, 600, 600),))
    assert tuple(result[0][5]) == pyramid[0]
    assert all(a is b for a, b in zip(result[0][5], pyramid[0]))


def test_matches_single_window_query():
    points = [(x * 100.0, y * 100.0, float(x + y), 0, 1.0)
              for x in range(6) for y in range(6)]
    pyramid = build_tile_pyramid(points, tile_cells=256, levels=3)
    windows = ((0, 100, 100, 400, 400),
               (1, 0, 0, 511, 255),
               (2, -1000, -1000, 1000, 1000))
    result = query_tile_pyramid_windows(pyramid, windows)
    assert len(result) == 3
    for (level, ix_min, iy_min, ix_max, iy_max, tiles), window in zip(
            result, windows):
        assert (level, ix_min, iy_min, ix_max, iy_max) == window
        expected = query_tile_window(
            pyramid, level, ix_min, iy_min, ix_max, iy_max)
        assert tiles == expected


def test_repeated_calls_are_deterministic():
    points = [(x * 100.0, y * 100.0, float(x - y), 0, 1.0)
              for x in range(8) for y in range(8)]
    pyramid = build_tile_pyramid(points, tile_cells=256, levels=3)
    windows = ((0, 0, 0, 300, 300), (2, -50, -50, 900, 200))
    first = query_tile_pyramid_windows(pyramid, windows)
    second = query_tile_pyramid_windows(pyramid, windows)
    assert first == second
    assert query_tile_pyramid_windows(pyramid, windows) == first


def test_inputs_are_not_modified():
    pyramid = ((
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511),
    ),)
    snapshot = tuple(tuple(t) for t in pyramid[0])
    windows = ((0, 0, 0, 600, 255), (0, 0, 0, 0, 0))
    query_tile_pyramid_windows(pyramid, windows)
    assert tuple(tuple(t) for t in pyramid[0]) == snapshot
    assert query_tile_pyramid(pyramid, 0, 1, 0) is pyramid[0][1]


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_pyramid_raises_type_error(bad):
    with pytest.raises(TypeError):
        query_tile_pyramid_windows(bad, ())


@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_windows_raises_type_error(bad):
    with pytest.raises(TypeError):
        query_tile_pyramid_windows((), bad)


@pytest.mark.parametrize("bad_window", [
    [0, 0, 0, 1, 1],
    None,
    42,
])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        query_tile_pyramid_windows((), (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        query_tile_pyramid_windows((), ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        query_tile_pyramid_windows((), ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        query_tile_pyramid_windows((), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: pyramid structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(([_tile()],), ())


def test_outer_level_entries_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(([_tile()],), ((0, 0, 0, 1, 1),))


@pytest.mark.parametrize("bad_tile", [
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),            # too few fields
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 0),      # too many fields
    [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],         # list, not tuple
])
def test_tile_wrong_shape_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((bad_tile,),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [1.5, "1", None])
def test_non_int_index_or_count_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
def test_bool_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = True
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, "1.0", None])
def test_non_float_z_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((tuple(tile),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_z_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((tuple(tile),),), ())


def test_inverted_tile_bounds_raise_value_error():
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(
            ((_tile(ix0=256, ix1=200),),), ())
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(
            ((_tile(iy0=256, iy1=200),),), ())


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(unsorted, ())

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(duplicate, ())


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(
            ((),), ((level, 0, 0, 1, 1),))


def test_inverted_window_bounds_raise_value_error():
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((),), ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        query_tile_pyramid_windows(((),), ((0, 0, 5, 1, 1),))


def test_equal_window_bounds_are_allowed():
    pyramid = ((_tile(),),)
    result = query_tile_pyramid_windows(pyramid, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (_tile(),)
