"""Tests for :func:`lidar_scan.tiles.query_tile_window`."""

from __future__ import annotations

import pytest

from lidar_scan import build_tile_pyramid, query_tile_window
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _pyramid(*tiles, levels=1, at_level=0):
    level_values = [() for _ in range(levels)]
    level_values[at_level] = tuple(tiles)
    return tuple(level_values)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.query_tile_window is query_tile_window
    import lidar_scan
    assert "query_tile_window" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def test_window_selects_intersecting_tiles_inclusive_and_ordered():
    pyramid = _pyramid(
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=5, ix0=0, iy0=1280, ix1=255, iy1=1535, count=10),
        _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511, count=3),
        _tile(tx=5, ty=0, ix0=1280, iy0=0, ix1=1535, iy1=255, count=9),
    )
    # Window [255, 256] x [255, 511]: corner-touches tile (0, 0) via
    # ix1=255/iy1=255, fully contains tile (1, 1) and excludes the far ones.
    result = query_tile_window(pyramid, 0, 255, 255, 256, 511)
    assert result == (
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511, count=3),
    )


def test_window_boundary_touch_is_inclusive():
    tile = _tile(ix0=10, iy0=20, ix1=30, iy1=40)
    pyramid = _pyramid(tile)
    # Single-cell windows touching each edge/corner still intersect.
    assert query_tile_window(pyramid, 0, 30, 40, 30, 40) == (tile,)
    assert query_tile_window(pyramid, 0, 10, 20, 10, 20) == (tile,)
    assert query_tile_window(pyramid, 0, 30, 0, 60, 20) == (tile,)
    assert query_tile_window(pyramid, 0, 0, 40, 10, 80) == (tile,)


def test_window_no_overlap_returns_empty():
    pyramid = _pyramid(_tile(ix0=0, iy0=0, ix1=255, iy1=255))
    # tile ix1=255 < ix_min=256: tile entirely to the right of the window
    assert query_tile_window(pyramid, 0, 256, 0, 300, 255) == ()
    # tile ix0=0 > ix_max=-1: tile entirely to the left
    assert query_tile_window(pyramid, 0, -10, 0, -1, 255) == ()
    # tile iy1=255 < iy_min=256: tile entirely above
    assert query_tile_window(pyramid, 0, 0, 256, 255, 300) == ()
    # tile iy0=0 > iy_max=-1: tile entirely below
    assert query_tile_window(pyramid, 0, 0, -10, 255, -1) == ()


def test_window_empty_level_returns_empty():
    pyramid = ((), (_tile(),))
    assert query_tile_window(pyramid, 0, -10 ** 6, -10 ** 6,
                             10 ** 6, 10 ** 6) == ()


def test_window_uses_the_requested_level():
    tile = _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=511, iy1=511)
    pyramid = ((_tile(),), (tile,))
    assert query_tile_window(pyramid, 1, 500, 0, 511, 255) == (tile,)
    assert query_tile_window(pyramid, 0, 500, 0, 511, 255) == ()


def test_window_does_not_modify_input():
    tile = _tile(tx=1, ty=2)
    pyramid = _pyramid(tile, levels=2)
    query_tile_window(pyramid, 0, 0, 0, 9, 9)
    assert pyramid == ((tile,), ())


def test_window_matches_built_pyramid():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    result = query_tile_window(pyramid, 2, -100, -100, 600, 300)
    expected = tuple(
        tile for tile in pyramid[2]
        if tile[4] >= -100 and tile[2] <= 600
        and tile[5] >= -100 and tile[3] <= 300
    )
    assert result == expected
    # Returned tiles are the stored 9-tuples themselves.
    assert all(tile in pyramid[2] for tile in result)


def test_window_returns_tuple():
    result = query_tile_window(_pyramid(_tile()), 0, 0, 0, 0, 0)
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# argument types and ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [(_tile(),)],
    {"levels": []},
    1,
    1.5,
    True,
    False,
    None,
    "((),)",
])
def test_window_non_tuple_pyramid_raises_type_error(bad):
    with pytest.raises(TypeError):
        query_tile_window(bad, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("name,bad", [
    ("level", 1.0), ("level", True), ("level", False),
    ("level", None), ("level", "0"), ("level", 1 + 0j),
    ("ix_min", 1.0), ("iy_min", True), ("ix_max", None), ("iy_max", "1"),
])
def test_window_non_int_arguments_raise_type_error(name, bad):
    kwargs = dict(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0)
    kwargs[name] = bad
    with pytest.raises(TypeError):
        query_tile_window(_pyramid(_tile()), **kwargs)


@pytest.mark.parametrize("level", [-1, 2, 100])
def test_window_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        query_tile_window(_pyramid(_tile(), levels=2), level, 0, 0, 0, 0)


def test_window_inverted_bounds_raise_value_error():
    pyramid = _pyramid(_tile())
    with pytest.raises(ValueError):
        query_tile_window(pyramid, 0, 2, 0, 1, 0)
    with pytest.raises(ValueError):
        query_tile_window(pyramid, 0, 0, 2, 0, 1)


# ---------------------------------------------------------------------------
# pyramid structure validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_pyramid", [
    ([_tile()],),                                  # level is a list
    (([_tile()],),),                               # tile is a list
    _pyramid((1, 2)),                              # 2-tuple tile
    _pyramid(_tile()[:8]),                         # 8 fields
    _pyramid(_tile() + (9,)),                      # 10 fields
    _pyramid(123),                                 # non-tuple tile
    _pyramid(_tile(tx=1.0)),
    _pyramid(_tile(ty=True)),
    _pyramid(_tile(ix0=False)),
    _pyramid(_tile(iy1="255")),
    _pyramid(_tile(count=1.0)),
    _pyramid(_tile(count=True)),
    _pyramid(_tile(zmin=1)),
    _pyramid(_tile(zmax=2.0 + 0j)),
    _pyramid(_tile(zmin=float("nan"))),
    _pyramid(_tile(zmax=float("inf"))),
    _pyramid(_tile(zmin=float("-inf"))),
    _pyramid(_tile(tx=0, ty=1), _tile(tx=0, ty=0)),     # unsorted
    _pyramid(_tile(tx=1, ty=1), _tile(tx=1, ty=1)),     # duplicate
    _pyramid(_tile(ix0=256, ix1=255)),                  # ix0 > ix1
    _pyramid(_tile(iy0=256, iy1=255)),                  # iy0 > iy1
    ((_tile(),), [_tile(tx=1, ty=1)]),                  # later level is list
])
def test_window_bad_structure_raises_value_error(bad_pyramid):
    with pytest.raises(ValueError):
        query_tile_window(bad_pyramid, 0, -(10 ** 100), -(10 ** 100),
                          10 ** 100, 10 ** 100)


def test_window_validates_all_levels_not_only_target():
    # The queried level is fine; a bad tile on another level must still raise.
    bad_pyramid = ((_tile(),), (_tile(ix0=256, ix1=255),))
    with pytest.raises(ValueError):
        query_tile_window(bad_pyramid, 0, -(10 ** 100), -(10 ** 100),
                          10 ** 100, 10 ** 100)
