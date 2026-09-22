"""Tests for :func:`encode_tile_region`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid, encode_tile_region,
                        query_tile_region)
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
    assert tiles_module.encode_tile_region is encode_tile_region
    import lidar_scan
    assert "encode_tile_region" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_region_canonical_text_and_key_order():
    pyramid = _pyramid(
        _tile(tx=-5, ty=-7, zmin=-3.25, zmax=9.0, count=42),
        _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023, count=7),
    )
    text = encode_tile_region(pyramid, 0, -10, -10, 3, 3)
    assert text == (
        '{"level":0,"tx_min":-10,"ty_min":-10,"tx_max":3,"ty_max":3,'
        '"tiles":[[-5,-7,0,0,255,255,-3.250000,9.000000,42],'
        '[2,3,512,768,767,1023,1.000000,2.000000,7]]}'
    )
    assert list(json.loads(text)) == [
        "level", "tx_min", "ty_min", "tx_max", "ty_max", "tiles"]


def test_encode_region_echoes_arguments():
    pyramid = _pyramid(_tile(), levels=3)
    text = encode_tile_region(pyramid, 2, -4, -5, 6, 7)
    document = json.loads(text)
    assert document["level"] == 2
    assert (document["tx_min"], document["ty_min"],
            document["tx_max"], document["ty_max"]) == (-4, -5, 6, 7)


def test_encode_region_filters_inclusive_and_preserves_order():
    pyramid = _pyramid(
        _tile(tx=0, ty=0),
        _tile(tx=1, ty=0, count=2),
        _tile(tx=1, ty=1, count=3),
        _tile(tx=2, ty=1, count=4),
    )
    text = encode_tile_region(pyramid, 0, 1, 0, 2, 1)
    assert json.loads(text)["tiles"] == [
        [1, 0, 0, 0, 255, 255, 1.0, 2.0, 2],
        [1, 1, 0, 0, 255, 255, 1.0, 2.0, 3],
        [2, 1, 0, 0, 255, 255, 1.0, 2.0, 4],
    ]


def test_encode_region_matches_query_tile_region():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    text = encode_tile_region(pyramid, 2, -2, -2, 1, 1)
    expected = [list(tile) for tile in query_tile_region(
        pyramid, 2, -2, -2, 1, 1)]
    assert json.loads(text)["tiles"] == expected


def test_encode_region_no_match_is_empty_array():
    pyramid = _pyramid(_tile(tx=0, ty=0))
    assert encode_tile_region(pyramid, 0, 5, 5, 9, 9) == (
        '{"level":0,"tx_min":5,"ty_min":5,"tx_max":9,"ty_max":9,"tiles":[]}')


def test_encode_region_empty_level():
    pyramid = ((), (_tile(),))
    assert encode_tile_region(pyramid, 0, 0, 0, 1, 1) == (
        '{"level":0,"tx_min":0,"ty_min":0,"tx_max":1,"ty_max":1,"tiles":[]}')


def test_encode_region_no_whitespace():
    text = encode_tile_region(_pyramid(_tile()), 0, 0, 0, 0, 0)
    assert not any(ch in text for ch in " \t\n\r")
    assert ":" in text and "," in text


def test_encode_region_six_decimals_and_negative_zero():
    pyramid = _pyramid(_tile(zmin=-1 / 7, zmax=math.copysign(0.0, -1.0)))
    text = encode_tile_region(pyramid, 0, 0, 0, 0, 0)
    assert "-0.142857" in text
    assert "-0.000000" not in text
    assert "0.000000" in text


def test_encode_region_large_integers_decimal():
    pyramid = _pyramid(_tile(tx=10 ** 40, ty=-10 ** 40, count=10 ** 80))
    text = encode_tile_region(pyramid, 0, -(10 ** 40), -(10 ** 40),
                              10 ** 40, 10 ** 40)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert str(10 ** 80) in text


def test_encode_region_does_not_modify_input():
    tile = _tile(tx=1, ty=2)
    pyramid = _pyramid(tile, levels=2)
    encode_tile_region(pyramid, 1, 0, 0, 9, 9)
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
def test_encode_region_non_tuple_pyramid_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_region(bad, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("name,bad", [
    ("level", 1.0), ("level", True), ("level", False),
    ("level", None), ("level", "0"), ("level", 1 + 0j),
    ("tx_min", 1.0), ("ty_min", True), ("tx_max", None), ("ty_max", "1"),
])
def test_encode_region_non_int_arguments_raise_type_error(name, bad):
    kwargs = dict(level=0, tx_min=0, ty_min=0, tx_max=0, ty_max=0)
    kwargs[name] = bad
    with pytest.raises(TypeError):
        encode_tile_region(_pyramid(_tile()), **kwargs)


@pytest.mark.parametrize("level", [-1, 2, 100])
def test_encode_region_level_out_of_range_raises_value_error(level):
    with pytest.raises(ValueError):
        encode_tile_region(_pyramid(_tile(), levels=2), level, 0, 0, 0, 0)


def test_encode_region_inverted_bounds_raise_value_error():
    pyramid = _pyramid(_tile())
    with pytest.raises(ValueError):
        encode_tile_region(pyramid, 0, 2, 0, 1, 0)
    with pytest.raises(ValueError):
        encode_tile_region(pyramid, 0, 0, 2, 0, 1)


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
def test_encode_region_bad_structure_raises_value_error(bad_pyramid):
    with pytest.raises(ValueError):
        encode_tile_region(bad_pyramid, 0, -(10 ** 100), -(10 ** 100),
                           10 ** 100, 10 ** 100)
