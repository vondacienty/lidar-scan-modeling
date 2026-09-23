"""Tests for :func:`encode_tile_window`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid, encode_tile_window,
                        query_tile_window)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _pyramid(*tiles, levels=1):
    empty = ((),) * (levels - 1)
    return empty + (tuple(tiles),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.encode_tile_window is encode_tile_window
    import lidar_scan
    assert "encode_tile_window" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_window_canonical_text_and_key_order():
    pyramid = _pyramid(
        _tile(tx=-5, ty=-7, zmin=-3.25, zmax=9.0, count=42),
        _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023, count=7),
    )
    text = encode_tile_window(pyramid, 0, -10, -10, 600, 1100)
    assert text == (
        '{"level":0,"ix_min":-10,"iy_min":-10,"ix_max":600,"iy_max":1100,'
        '"tiles":[[-5,-7,0,0,255,255,-3.250000,9.000000,42],'
        '[2,3,512,768,767,1023,1.000000,2.000000,7]]}'
    )
    assert list(json.loads(text)) == [
        "level", "ix_min", "iy_min", "ix_max", "iy_max", "tiles"]


def test_encode_window_echoes_arguments():
    pyramid = _pyramid(_tile(), levels=3)
    text = encode_tile_window(pyramid, 2, -4, -5, 6, 7)
    document = json.loads(text)
    assert document["level"] == 2
    assert (document["ix_min"], document["iy_min"],
            document["ix_max"], document["iy_max"]) == (-4, -5, 6, 7)


@pytest.mark.parametrize("ix_min,ix_max", [(255, 300), (0, 0), (-5, 0)])
def test_encode_window_inclusive_intersection_x(ix_min, ix_max):
    pyramid = _pyramid(_tile())
    text = encode_tile_window(pyramid, 0, ix_min, 0, ix_max, 255)
    assert len(json.loads(text)["tiles"]) == 1


@pytest.mark.parametrize("iy_min,iy_max", [(255, 300), (0, 0), (-5, 0)])
def test_encode_window_inclusive_intersection_y(iy_min, iy_max):
    pyramid = _pyramid(_tile())
    text = encode_tile_window(pyramid, 0, 0, iy_min, 255, iy_max)
    assert len(json.loads(text)["tiles"]) == 1


def test_encode_window_filters_closed_intervals_and_preserves_order():
    pyramid = _pyramid(
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, ix0=256, ix1=511, count=2),
        _tile(tx=1, ty=1, ix0=256, iy0=256, ix1=511, iy1=511, count=3),
        _tile(tx=2, ty=1, ix0=512, iy0=256, ix1=767, iy1=511, count=4),
    )
    text = encode_tile_window(pyramid, 0, 256, 0, 511, 511)
    assert json.loads(text)["tiles"] == [
        [1, 0, 256, 0, 511, 255, 1.0, 2.0, 2],
        [1, 1, 256, 256, 511, 511, 1.0, 2.0, 3],
    ]


def test_encode_window_matches_query_tile_window():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    text = encode_tile_window(pyramid, 2, -100, -100, 400, 400)
    expected = [list(tile) for tile in query_tile_window(
        pyramid, 2, -100, -100, 400, 400)]
    assert json.loads(text)["tiles"] == expected


def test_encode_window_no_match_is_empty_array():
    pyramid = _pyramid(_tile(tx=0, ty=0))
    assert encode_tile_window(pyramid, 0, 256, 256, 300, 300) == (
        '{"level":0,"ix_min":256,"iy_min":256,"ix_max":300,"iy_max":300,'
        '"tiles":[]}')


def test_encode_window_empty_level():
    pyramid = ((), (_tile(),))
    assert encode_tile_window(pyramid, 0, 0, 0, 1, 1) == (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":1,"iy_max":1,"tiles":[]}')


def test_encode_window_no_whitespace():
    text = encode_tile_window(_pyramid(_tile()), 0, 0, 0, 0, 0)
    assert not any(ch in text for ch in " \t\n\r")
    assert ":" in text and "," in text


def test_encode_window_six_decimals_and_negative_zero():
    pyramid = _pyramid(_tile(zmin=-1 / 7, zmax=math.copysign(0.0, -1.0)))
    text = encode_tile_window(pyramid, 0, 0, 0, 255, 255)
    assert "-0.142857" in text
    assert "-0.000000" not in text
    assert "0.000000" in text


def test_encode_window_large_integers_decimal():
    pyramid = _pyramid(
        _tile(tx=10 ** 40, ty=-10 ** 40, ix0=10 ** 40, iy0=-10 ** 40,
              ix1=10 ** 40 + 255, iy1=-10 ** 40 + 255, count=10 ** 80))
    text = encode_tile_window(pyramid, 0, -(10 ** 100), -(10 ** 100),
                              10 ** 100, 10 ** 100)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert str(10 ** 80) in text


def test_encode_window_does_not_modify_input():
    tile = _tile(tx=1, ty=2, ix0=256, iy0=512, ix1=511, iy1=767)
    pyramid = _pyramid(tile, levels=2)
    encode_tile_window(pyramid, 1, 0, 0, 999, 999)
    assert pyramid == ((), (tile,))


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
def test_encode_window_non_tuple_pyramid_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_window(bad, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("name,bad", [
    ("level", 1.0), ("level", True), ("level", False),
    ("level", None), ("level", "0"), ("level", 1 + 0j),
    ("ix_min", 1.0), ("iy_min", True), ("ix_max", None), ("iy_max", "1"),
])
def test_encode_window_non_int_arguments_raise_type_error(name, bad):
    kwargs = dict(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0)
    kwargs[name] = bad
    with pytest.raises(TypeError):
        encode_tile_window(_pyramid(_tile()), **kwargs)


@pytest.mark.parametrize("level", [-1, 2, 100])
def test_encode_window_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        encode_tile_window(_pyramid(_tile(), levels=2), level, 0, 0, 0, 0)


def test_encode_window_inverted_bounds_raise_value_error():
    pyramid = _pyramid(_tile())
    with pytest.raises(ValueError):
        encode_tile_window(pyramid, 0, 2, 0, 1, 0)
    with pytest.raises(ValueError):
        encode_tile_window(pyramid, 0, 0, 2, 0, 1)


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
    ((_tile(),), [_tile(tx=1, ty=1)]),                  # later level is list
])
def test_encode_window_bad_structure_raises_value_error(bad_pyramid):
    with pytest.raises(ValueError):
        encode_tile_window(bad_pyramid, 0, -(10 ** 100), -(10 ** 100),
                           10 ** 100, 10 ** 100)


@pytest.mark.parametrize("bad_tile", [
    _tile(ix1=-1),
    _tile(iy1=-1),
    _tile(ix0=256, ix1=255),
    _tile(iy0=256, iy1=255),
])
def test_encode_window_inverted_tile_bounds_raise_value_error(bad_tile):
    with pytest.raises(ValueError):
        encode_tile_window(_pyramid(bad_tile), 0, -(10 ** 100),
                           -(10 ** 100), 10 ** 100, 10 ** 100)
