"""Tests for :func:`decode_tile_window`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid, decode_tile_window,
                        encode_tile_window)
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
    assert tiles_module.decode_tile_window is decode_tile_window
    import lidar_scan
    assert "decode_tile_window" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# decoding / round trips
# ---------------------------------------------------------------------------

def test_decode_canonical_text_and_key_order():
    text = (
        '{"level":0,"ix_min":-10,"iy_min":-10,"ix_max":300,"iy_max":300,'
        '"tiles":[[-5,-7,0,0,255,255,-3.250000,9.000000,42]]}'
    )
    result = decode_tile_window(text)
    assert result == (
        0, -10, -10, 300, 300,
        ((-5, -7, 0, 0, 255, 255, -3.25, 9.0, 42),),
    )
    assert list(json.loads(text)) == [
        "level", "ix_min", "iy_min", "ix_max", "iy_max", "tiles"]


def test_decode_returns_tuples_preserving_order():
    text = encode_tile_window(
        _pyramid(
            _tile(tx=-5, ty=-7, count=42),
            _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023, count=7),
        ),
        0, -10, -10, 600, 1100)
    result = decode_tile_window(text)
    assert isinstance(result, tuple) and len(result) == 6
    level, ix_min, iy_min, ix_max, iy_max, tiles = result
    assert (level, ix_min, iy_min, ix_max, iy_max) == (0, -10, -10, 600, 1100)
    assert isinstance(tiles, tuple) and len(tiles) == 2
    assert all(isinstance(tile, tuple) and len(tile) == 9 for tile in tiles)
    assert tiles[0][:2] == (-5, -7)
    assert tiles[1][:2] == (2, 3)
    for name, value in zip(("tx", "ty", "ix0", "iy0", "ix1", "iy1", "count"),
                           tiles[0][:6] + (tiles[0][8],)):
        assert isinstance(value, int) and not isinstance(value, bool), name
    for value in tiles[0][6:8]:
        assert isinstance(value, float) and math.isfinite(value)


def test_decode_empty_tiles():
    text = '{"level":2,"ix_min":5,"iy_min":5,"ix_max":9,"iy_max":9,"tiles":[]}'
    assert decode_tile_window(text) == (2, 5, 5, 9, 9, ())


def test_decode_window_round_trips_against_encoder():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    text = encode_tile_window(pyramid, 2, -100, -100, 400, 400)
    level, ix_min, iy_min, ix_max, iy_max, tiles = decode_tile_window(text)
    assert (level, ix_min, iy_min, ix_max, iy_max) == (2, -100, -100, 400, 400)
    assert tiles == tuple(
        tile for tile in pyramid[2]
        if (tile[4] >= ix_min and tile[2] <= ix_max
            and tile[5] >= iy_min and tile[3] <= iy_max))
    assert encode_tile_window(pyramid, level, ix_min, iy_min,
                              ix_max, iy_max) == text


@pytest.mark.parametrize("ix_min,ix_max", [(255, 300), (0, 0), (-5, 0)])
def test_decode_inclusive_window_bounds_x(ix_min, ix_max):
    text = (
        f'{{"level":0,"ix_min":{ix_min},"iy_min":0,"ix_max":{ix_max},'
        '"iy_max":255,'
        '"tiles":[[0,0,0,0,255,255,1.000000,2.000000,3]]}')
    assert len(decode_tile_window(text)[5]) == 1


@pytest.mark.parametrize("iy_min,iy_max", [(255, 300), (0, 0), (-5, 0)])
def test_decode_inclusive_window_bounds_y(iy_min, iy_max):
    text = (
        f'{{"level":0,"ix_min":0,"iy_min":{iy_min},"ix_max":255,'
        f'"iy_max":{iy_max},'
        '"tiles":[[0,0,0,0,255,255,1.000000,2.000000,3]]}')
    assert len(decode_tile_window(text)[5]) == 1


def test_decode_six_decimals_and_negative_zero_normalized():
    text = (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":255,"iy_max":255,'
        '"tiles":[[0,0,0,0,255,255,-0.142857,0.000000,1]]}'
    )
    tile = decode_tile_window(text)[5][0]
    assert tile[6] == pytest.approx(-1 / 7, abs=1e-6)
    assert tile[7] == 0.0 and math.copysign(1.0, tile[7]) == 1.0


def test_decode_large_integers_decimal():
    text = encode_tile_window(
        _pyramid(_tile(tx=10 ** 40, ty=-10 ** 40, ix0=0, iy0=0,
                       ix1=255, iy1=255, count=10 ** 80)),
        0, -(10 ** 40), -(10 ** 40), 10 ** 40, 10 ** 40)
    level, ix_min, iy_min, ix_max, iy_max, tiles = decode_tile_window(text)
    assert tiles[0][0] == 10 ** 40 and tiles[0][1] == -(10 ** 40)
    assert tiles[0][8] == 10 ** 80
    assert ix_min == -(10 ** 40)


def test_decode_does_not_modify_input():
    text = (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":255,"iy_max":255,'
        '"tiles":[[0,0,0,0,255,255,1.000000,2.000000,1]]}')
    snapshot = text
    decode_tile_window(text)
    assert text == snapshot


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"...", True, False, [], {}])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_tile_window(bad)


# ---------------------------------------------------------------------------
# malformed documents
# ---------------------------------------------------------------------------

_VALID_TILE = "[0,0,0,0,255,255,1.000000,2.000000,1]"


def _doc(bounds='"level":0,"ix_min":0,"iy_min":0,"ix_max":255,"iy_max":255',
         tiles=None):
    if tiles is None:
        tiles = _VALID_TILE
    return "{" + bounds + ',"tiles":[' + tiles + "]}"


@pytest.mark.parametrize("bad", ["", "{", "[]", "null", "1", '"x"',
                                 '{"a":1}', "{...}"])
def test_decode_bad_json_syntax_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode_tile_window(bad)


@pytest.mark.parametrize("bad", [
    '{"tiles":[]}',
    '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,"extra":1,'
    '"tiles":[]}',
    '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"tiles":[]}',
    '{"level":0,"ix_min":0,"iy_min":0,"iy_max":0,"ix_max":0,"tiles":[]}',
    '{"tiles":[],"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0}',
    '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,'
    '"tiles":[],"tiles":[]}',
])
def test_decode_bad_top_level_keys_raise_value_error(bad):
    with pytest.raises(ValueError):
        decode_tile_window(bad)


@pytest.mark.parametrize("bad", [
    _doc(bounds='"level":-1,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":true,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":false,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":0.0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":"0","ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":null,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":0,"ix_min":true,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":0,"ix_min":1,"iy_min":0,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":0,"ix_min":0,"iy_min":1,"ix_max":0,"iy_max":0'),
    _doc(bounds='"level":0,"ix_min":0,"iy_min":0,"ix_max":false,"iy_max":0'),
    _doc(bounds='"level":0,"ix_min":0,"iy_min":0,"ix_max":0.5,"iy_max":0'),
])
def test_decode_bad_bounds_raise_value_error(bad):
    with pytest.raises(ValueError):
        decode_tile_window(bad)


def test_decode_tiles_not_array_raises_value_error():
    with pytest.raises(ValueError):
        decode_tile_window(
            '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,'
            '"tiles":{}}')


@pytest.mark.parametrize("bad_tile", [
    "0",
    "{}",
    "[]",
    "[0,0,0,0,255,255,1.000000,2.000000]",       # 8 fields
    "[0,0,0,0,255,255,1.000000,2.000000,1,2]",  # 10 fields
    "[0.0,0,0,0,255,255,1.000000,2.000000,1]",
    "[0,true,0,0,255,255,1.000000,2.000000,1]",
    "[0,0,0,0,255,false,1.000000,2.000000,1]",
    "[0,0,0,0,255,255,1.000000,2.000000,true]",
    "[0,0,0,0,255,255,1.000000,2.000000,1.0]",
    '[0,0,0,0,255,255,1,2.000000,1]',
    '[0,0,0,0,255,255,1.000000,2,1]',
    '[0,0,0,0,255,255,1.000000,null,1]',
])
def test_decode_bad_tile_fields_raise_value_error(bad_tile):
    with pytest.raises(ValueError):
        decode_tile_window(_doc(tiles=bad_tile))


@pytest.mark.parametrize("bad_tile", [
    # ix0 > ix1
    "[0,0,256,0,255,255,1.000000,2.000000,1]",
    # iy0 > iy1
    "[0,0,0,256,255,255,1.000000,2.000000,1]",
])
def test_decode_inverted_tile_bounds_raise_value_error(bad_tile):
    with pytest.raises(ValueError):
        decode_tile_window(_doc(tiles=bad_tile))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_decode_non_finite_literals_raise_value_error(literal):
    bad_tile = f"[0,0,0,0,255,255,1.000000,{literal},1]"
    with pytest.raises(ValueError):
        decode_tile_window(_doc(tiles=bad_tile))


def test_decode_unsorted_tiles_raise_value_error():
    bad = (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":600,"iy_max":300,'
        '"tiles":[[1,0,256,0,511,255,1.000000,2.000000,1],'
        '[0,0,0,0,255,255,1.000000,2.000000,1]]}')
    with pytest.raises(ValueError):
        decode_tile_window(bad)


def test_decode_duplicate_tile_coordinates_raise_value_error():
    bad = (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":255,"iy_max":300,'
        '"tiles":[[0,1,0,256,255,511,1.000000,2.000000,1],'
        '[0,1,0,256,255,511,3.000000,4.000000,2]]}')
    with pytest.raises(ValueError):
        decode_tile_window(bad)


@pytest.mark.parametrize("bounds", [
    # tile ix1=255 < ix_min=256: tile entirely to the right of the window
    '"level":0,"ix_min":256,"iy_min":0,"ix_max":300,"iy_max":255',
    # tile ix0=0 > ix_max=-1: tile entirely to the left
    '"level":0,"ix_min":-10,"iy_min":0,"ix_max":-1,"iy_max":255',
    # tile iy1=255 < iy_min=256: tile entirely above
    '"level":0,"ix_min":0,"iy_min":256,"ix_max":255,"iy_max":300',
    # tile iy0=0 > iy_max=-1: tile entirely below
    '"level":0,"ix_min":0,"iy_min":-10,"ix_max":255,"iy_max":-1',
    # no intersection on either axis
    '"level":0,"ix_min":256,"iy_min":256,"ix_max":300,"iy_max":300',
])
def test_decode_non_intersecting_tile_raises_value_error(bounds):
    with pytest.raises(ValueError):
        decode_tile_window(_doc(bounds=bounds))


@pytest.mark.parametrize("bad", [
    '{"level": 0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,"tiles":[]}',
    '{ "level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,"tiles":[]}',
    '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,"tiles": []}',
    '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,"tiles":[ ]}',
])
def test_decode_whitespace_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode_tile_window(bad)


@pytest.mark.parametrize("bad_tile", [
    "[0,0,0,0,255,255,1,2.000000,1]",
    "[0,0,0,0,255,255,1.0,2.000000,1]",
    "[0,0,0,0,255,255,1.00000,2.000000,1]",
    "[0,0,0,0,255,255,1.0000000,2.000000,1]",
    "[0,0,0,0,255,255,-0.000000,2.000000,1]",
    "[0,0,0,0,255,255,+1.000000,2.000000,1]",
    "[0,0,0,0,255,255,1.000000e0,2.000000,1]",
])
def test_decode_non_canonical_numbers_raise_value_error(bad_tile):
    with pytest.raises(ValueError):
        decode_tile_window(_doc(tiles=bad_tile))


@pytest.mark.parametrize("bad_bounds", [
    '"level":00,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0',
    '"level":0,"ix_min":+0,"iy_min":0,"ix_max":0,"iy_max":0',
    '"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":-00',
])
def test_decode_non_canonical_integer_spelling_raises_value_error(bad_bounds):
    with pytest.raises(ValueError):
        decode_tile_window(_doc(bounds=bad_bounds))


def test_decode_duplicate_json_keys_raise_value_error():
    bad = (
        '{"level":0,"ix_min":0,"iy_min":0,"ix_max":0,"iy_max":0,'
        '"level":1,"tiles":[]}')
    with pytest.raises(ValueError):
        decode_tile_window(bad)
