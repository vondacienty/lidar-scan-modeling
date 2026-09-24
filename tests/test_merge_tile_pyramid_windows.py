"""Tests for :func:`lidar_scan.merge_tile_pyramid_windows`."""

from __future__ import annotations

import pytest

from lidar_scan import (build_tile_pyramid, merge_tile_pyramid_windows,
                        merge_tile_pyramids, query_tile_pyramid_windows)
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
    assert tiles_module.merge_tile_pyramid_windows is merge_tile_pyramid_windows
    import lidar_scan
    assert "merge_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_empty_pyramids_and_empty_windows_return_empty_tuple():
    assert merge_tile_pyramid_windows((), ()) == ()


def test_empty_pyramids_with_windows_raises_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((), ((0, 0, 0, 10, 10),))


def test_empty_windows_over_pyramids_returns_empty_tuple():
    pyramids = (((_tile(),),), ((_tile(tx=1, ty=0, ix0=256, ix1=511),),))
    assert merge_tile_pyramid_windows(pyramids, ()) == ()


def test_zero_level_pyramids_support_only_empty_windows():
    assert merge_tile_pyramid_windows(((), ()), ()) == ()
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows(((), ()), ((0, 0, 0, 1, 1),))


def test_window_without_matches_keeps_window_and_empty_tiles():
    pyramids = (((),),)
    result = merge_tile_pyramid_windows(pyramids, ((0, 0, 0, 10, 10),))
    assert result == ((0, 0, 0, 10, 10, ()),)


def test_tiles_from_all_pyramids_are_unioned():
    first = ((
        _tile(tx=0, ty=0, zmin=1.0, zmax=3.0, count=2),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=0.5, count=4),
    ),)
    second = ((
        _tile(tx=0, ty=0, zmin=0.0, zmax=5.0, count=7),
        _tile(tx=2, ty=0, ix0=512, ix1=767, zmin=2.0, zmax=2.0, count=1),
    ),)
    result = merge_tile_pyramid_windows((first, second), ((0, 0, 0, 767, 255),))
    assert result == ((0, 0, 0, 767, 255, (
        _tile(tx=0, ty=0, zmin=0.0, zmax=5.0, count=9),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=0.5, zmax=2.0, count=4),
        _tile(tx=2, ty=0, ix0=512, ix1=767, zmin=2.0, zmax=2.0, count=1),
    )),)


def test_only_tiles_intersecting_the_window_are_returned():
    first = ((_tile(tx=0, ty=0),),)
    second = ((_tile(tx=1, ty=0, ix0=256, ix1=511),),)
    result = merge_tile_pyramid_windows(
        (first, second), ((0, 256, 0, 300, 255),))
    assert result[0][5] == (_tile(tx=1, ty=0, ix0=256, ix1=511),)


def test_closed_intervals_match_at_endpoints():
    pyramids = (((_tile(),),),)
    result = merge_tile_pyramid_windows(
        pyramids, ((0, 255, 255, 255, 255),))
    assert result[0][5] == (_tile(),)
    result = merge_tile_pyramid_windows(pyramids, ((0, 256, 0, 300, 255),))
    assert result[0][5] == ()
    result = merge_tile_pyramid_windows(pyramids, ((0, 0, 256, 255, 300),))
    assert result[0][5] == ()


def test_window_echoes_its_arguments():
    pyramids = (((), (), (_tile(),)),)
    result = merge_tile_pyramid_windows(pyramids, ((2, -10, -20, 30, 40),))
    assert result[0][:5] == (2, -10, -20, 30, 40)


def test_windows_preserve_window_order():
    pyramids = (((_tile(),),),)
    windows = ((0, 200, 200, 210, 210),
               (0, 0, 0, 10, 10),
               (0, 100, 100, 120, 120))
    result = merge_tile_pyramid_windows(pyramids, windows)
    assert tuple(window[:5] for window in result) == windows


def test_result_tiles_are_sorted_by_coordinates():
    first = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=2, ty=0, ix0=512, ix1=767),
    ),)
    second = ((_tile(tx=0, ty=0),
               _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511)),)
    result = merge_tile_pyramid_windows((first, second), ((0, 0, 0, 800, 800),))
    assert [tile[0:2] for tile in result[0][5]] == [
        (0, 0), (1, 0), (1, 1), (2, 0)]


def test_multiple_levels_select_the_right_level():
    first = ((_tile(),), ())
    second = ((), (_tile(ix1=511, iy1=511),))
    result = merge_tile_pyramid_windows(
        (first, second),
        ((0, 0, 0, 255, 255), (1, 0, 0, 511, 511),
         (0, 300, 300, 400, 400)))
    assert result[0][5] == (_tile(),)
    assert result[1][5] == (_tile(ix1=511, iy1=511),)
    assert result[2][5] == ()


def test_matches_merge_then_query():
    points_a = [(x * 100.0, y * 100.0, float(x + y), 0, 1.0)
                for x in range(4) for y in range(4)]
    points_b = [(x * 100.0, y * 100.0, float(x - y), 0, 1.0)
                for x in range(2, 6) for y in range(2, 6)]
    pyramids = (
        build_tile_pyramid(points_a, tile_cells=256, levels=3),
        build_tile_pyramid(points_b, tile_cells=256, levels=3),
    )
    windows = ((0, 100, 100, 400, 400),
               (1, 0, 0, 511, 255),
               (2, -1000, -1000, 1000, 1000))
    result = merge_tile_pyramid_windows(pyramids, windows)
    expected = query_tile_pyramid_windows(merge_tile_pyramids(pyramids), windows)
    assert result == expected


def test_result_is_independent_of_pyramid_order():
    first = ((
        _tile(tx=0, ty=0, zmin=1.0, zmax=3.0, count=2),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=0.5, count=4),
    ),)
    second = ((
        _tile(tx=0, ty=0, zmin=0.0, zmax=5.0, count=7),
        _tile(tx=2, ty=0, ix0=512, ix1=767, count=3),
    ),)
    windows = ((0, 0, 0, 800, 300),)
    ab = merge_tile_pyramid_windows((first, second), windows)
    ba = merge_tile_pyramid_windows((second, first), windows)
    assert ab == ba


def test_inputs_are_not_modified():
    first = ((_tile(tx=0, ty=0, count=2),
              _tile(tx=1, ty=0, ix0=256, ix1=511, count=3)),)
    second = ((_tile(tx=0, ty=0, count=5),),)
    snapshot = (
        tuple(tuple(t) for t in first[0]),
        tuple(tuple(t) for t in second[0]),
    )
    merge_tile_pyramid_windows((first, second), ((0, 0, 0, 600, 255),))
    assert (tuple(tuple(t) for t in first[0]),
            tuple(tuple(t) for t in second[0])) == snapshot


def test_merged_tiles_are_new_objects():
    first = ((_tile(tx=0, ty=0, count=2),),)
    second = ((_tile(tx=0, ty=0, count=5),),)
    result = merge_tile_pyramid_windows((first, second), ((0, 0, 0, 255, 255),))
    assert result[0][5][0] is not first[0][0]
    assert result[0][5][0] is not second[0][0]


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_pyramids_raises_type_error(bad):
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows(bad, ())


@pytest.mark.parametrize("bad", [[], None, 42, "x"])
def test_non_tuple_windows_raises_type_error(bad):
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows((), bad)


@pytest.mark.parametrize("bad_window", [[0, 0, 0, 1, 1], None, 42])
def test_window_wrong_container_raises_type_error(bad_window):
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows((), (bad_window,))


def test_window_wrong_length_raises_type_error():
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows((), ((0, 0, 0, 1),))
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows((), ((0, 0, 0, 1, 1, 0),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1", None])
def test_window_field_wrong_type_raises_type_error(pos, bad_value):
    window = [0, 0, 0, 1, 1]
    window[pos] = bad_value
    with pytest.raises(TypeError):
        merge_tile_pyramid_windows((), (tuple(window),))


# ---------------------------------------------------------------------------
# ValueError: pyramids
# ---------------------------------------------------------------------------

def test_pyramid_member_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows(([_tile()],), ())


def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((([_tile()],),), ())


@pytest.mark.parametrize("bad_tile", [
    (0, 0, 0, 0, 255, 255, 1.0, 2.0),
    (0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 0),
    [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],
])
def test_tile_wrong_shape_raises_value_error(bad_tile):
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((bad_tile,),),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("bad_value", [1.5, "1", None])
def test_non_int_index_or_count_raises_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((tuple(tile),),),), ())


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 8])
def test_bool_index_or_count_raises_value_error(pos):
    tile = list(_tile())
    tile[pos] = True
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((tuple(tile),),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value", [1, "1.0", None])
def test_non_float_z_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((tuple(tile),),),), ())


@pytest.mark.parametrize("pos", [6, 7])
@pytest.mark.parametrize("bad_value",
                         [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_z_bounds_raise_value_error(pos, bad_value):
    tile = list(_tile())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((tuple(tile),),),), ())


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((unsorted,), ())

    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((duplicate,), ())


def test_different_level_counts_raise_value_error():
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows(
            ((_tile(),), ((_tile(),), (_tile(),))), ())


def test_same_coordinate_with_different_bounds_raises_value_error():
    first = ((_tile(tx=0, ty=0, ix1=200),),)
    second = ((_tile(tx=0, ty=0, ix1=255),),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((first, second), ())


# ---------------------------------------------------------------------------
# ValueError: windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [-1, 1, 5])
def test_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows((((),),), ((level, 0, 0, 1, 1),))


def test_inverted_window_bounds_raise_value_error():
    pyramids = (((),),)
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows(pyramids, ((0, 5, 0, 1, 1),))
    with pytest.raises(ValueError):
        merge_tile_pyramid_windows(pyramids, ((0, 0, 5, 1, 1),))
