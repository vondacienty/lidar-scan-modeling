"""Tests for :func:`encode_tile_pyramid_stats` /
:func:`decode_tile_pyramid_stats`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid_stats, decode_tile_pyramid_stats,
                        encode_tile_pyramid_stats)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, zmean=1.5, zsigma=0.5, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["zmean"], fields["zsigma"], fields["count"])


def _pyramid(*tiles):
    return (tuple(tiles),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.encode_tile_pyramid_stats is encode_tile_pyramid_stats
    assert tiles_module.decode_tile_pyramid_stats is decode_tile_pyramid_stats
    import lidar_scan
    assert "encode_tile_pyramid_stats" in lidar_scan.__all__
    assert "decode_tile_pyramid_stats" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_single_tile_canonical_text():
    text = encode_tile_pyramid_stats(_pyramid(_tile()))
    assert text == (
        '{"levels":[[[0,0,0,0,255,255,1.000000,2.000000,1.500000,0.500000,1]]]}'
    )


def test_encode_empty_outer_pyramid():
    assert encode_tile_pyramid_stats(()) == '{"levels":[]}'


def test_encode_empty_levels_preserved():
    assert encode_tile_pyramid_stats(((), ())) == '{"levels":[[],[]]}'


def test_encode_from_built_pyramid_round_trips():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid_stats(points, levels=3)
    text = encode_tile_pyramid_stats(pyramid)
    assert decode_tile_pyramid_stats(text) == pyramid


def test_encode_keeps_level_and_tile_order():
    pyramid = (
        (),
        (_tile(tx=-5, ty=-7, zmin=-3.25, zmax=9.0, zmean=2.0,
               zsigma=1.25, count=42),
         _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023,
               zmin=0.0, zmax=1.5, zmean=0.75, zsigma=0.1, count=7)),
    )
    text = encode_tile_pyramid_stats(pyramid)
    assert text == (
        '{"levels":[[],['
        '[-5,-7,0,0,255,255,-3.250000,9.000000,2.000000,1.250000,42],'
        '[2,3,512,768,767,1023,0.000000,1.500000,0.750000,0.100000,7]]]}'
    )


def test_encode_uses_compact_separators_without_whitespace():
    text = encode_tile_pyramid_stats(_pyramid(_tile()))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    pyramid = _pyramid(_tile(zmin=-1 / 7, zmean=1 / 3, zsigma=1 / 9))
    text = encode_tile_pyramid_stats(pyramid)
    assert "-0.142857" in text and "0.333333" in text and "0.111111" in text
    assert list(json.loads(text)) == ["levels"]


def test_encode_negative_zero_written_positive():
    text = encode_tile_pyramid_stats(
        _pyramid(_tile(zmin=-0.0, zmax=math.copysign(0.0, -1.0),
                       zmean=-0.0))
    )
    assert "-0.000000" not in text
    assert "0.000000,0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    pyramid = _pyramid(_tile(tx=10 ** 40, ty=-10 ** 40, count=10 ** 80))
    text = encode_tile_pyramid_stats(pyramid)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert decode_tile_pyramid_stats(text) == pyramid


def test_encode_top_level_has_only_levels_key():
    text = encode_tile_pyramid_stats(_pyramid(_tile()))
    assert list(json.loads(text).keys()) == ["levels"]


@pytest.mark.parametrize("bad", [
    [[[_tile()]]],
    {"levels": []},
    "((),)",
    1,
    1.5,
    True,
    False,
    None,
    {(_tile(),)},
])
def test_encode_non_tuple_outer_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_pyramid_stats(bad)


@pytest.mark.parametrize("bad_pyramid", [
    ([_tile()],),                                  # level is a list
    (([_tile()],),),                               # tile is a list
    _pyramid(_tile()[:10]),                        # 10 fields
    _pyramid(_tile() + (9,)),                      # 12 fields
    _pyramid(123),                                 # non-tuple tile
    _pyramid(_tile(tx=1.0)),
    _pyramid(_tile(ty=True)),
    _pyramid(_tile(ix0=False)),
    _pyramid(_tile(iy1="255")),
    _pyramid(_tile(count=1.0)),
    _pyramid(_tile(count=True)),
    _pyramid(_tile(zmin=1)),
    _pyramid(_tile(zmax=2.0 + 0j)),
    _pyramid(_tile(zmean="1.5")),
    _pyramid(_tile(zsigma=1)),
    _pyramid(_tile(zmin=float("nan"))),
    _pyramid(_tile(zmax=float("inf"))),
    _pyramid(_tile(zmean=float("-inf"))),
    _pyramid(_tile(zsigma=float("nan"))),
    _pyramid(_tile(zsigma=0.0)),
    _pyramid(_tile(zsigma=-0.5)),
    _pyramid(_tile(tx=0, ty=1), _tile(tx=0, ty=0)),     # unsorted
    _pyramid(_tile(tx=1, ty=1), _tile(tx=1, ty=1)),     # duplicate
    ((_tile(),), [_tile(tx=1, ty=1)]),                  # later level is list
])
def test_encode_bad_structure_raises_value_error(bad_pyramid):
    with pytest.raises(ValueError):
        encode_tile_pyramid_stats(bad_pyramid)


def test_encode_does_not_modify_input():
    tile = _tile()
    pyramid = _pyramid(tile)
    encode_tile_pyramid_stats(pyramid)
    assert pyramid == _pyramid(tile)


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_returns_tuples_and_preserves_order():
    pyramid = (
        (),
        (_tile(tx=-5, ty=-7), _tile(tx=2, ty=3, ix0=512, iy0=768,
                                    ix1=767, iy1=1023)),
    )
    result = decode_tile_pyramid_stats(encode_tile_pyramid_stats(pyramid))
    assert result == pyramid
    assert isinstance(result, tuple)
    assert all(isinstance(level, tuple) for level in result)
    assert all(isinstance(tile, tuple) for level in result for tile in level)


def test_decode_empty_variants():
    assert decode_tile_pyramid_stats('{"levels":[]}') == ()
    assert decode_tile_pyramid_stats('{"levels":[[]]}') == ((),)
    assert decode_tile_pyramid_stats('{"levels":[[],[]]}') == ((), ())


def test_decode_stats_are_floats_and_counts_are_ints():
    result = decode_tile_pyramid_stats(
        '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3]]]}'
    )
    tile = result[0][0]
    for index in (6, 7, 8, 9):
        assert type(tile[index]) is float
    for index in range(6):
        assert isinstance(tile[index], int) and not isinstance(tile[index], bool)
    assert type(tile[10]) is int


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"...", [], {}, True, object()])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_tile_pyramid_stats(bad)


@pytest.mark.parametrize("text", [
    "",
    "{",
    "}",
    "null",
    "[]",
    "42",
    '"levels"',
    "true",
    "[{}]",
    '{"levels":[',
    '{"levels":}',
    '{"levels":[}',
    '{"levels":null}',
    '{"levels":{}}',
    '{"levels":[{}]}',
    '{"levels":[null]}',
    '{"levels":[42]}',
    '{"levels":[["x"]]}',
    '{"levels":[[null]]}',
    '{"levels":[[[]]]}',
    # numeric formatting
    '{"levels":[[[0,0,0,0,255,255,1.0,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.5,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.5000000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000e0,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000E0,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,+1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,-00.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,-0.000000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.5,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.00000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.5,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.5000000,3]]]}',
    # field count / types
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3,4]]]}',
    '{"levels":[[[0,0,0,0,255,255,"a",2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,true,1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,null,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255.0,255,1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,0255,255,1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3.0]]]}',
    '{"levels":[[[true,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.00000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.0000001,0.500000,3]]]}',
    # non-positive zsigma
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.000000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,-0.500000,3]]]}',
    # non-finite literals
    '{"levels":[[[0,0,0,0,255,255,NaN,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,Infinity,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,-Infinity,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,NaN,3]]]}',
    # ordering / duplicates
    '{"levels":[[[0,1,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3],'
    '[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3]]]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3],'
    '[0,0,0,0,255,255,9.000000,9.000000,9.000000,0.500000,1]]]}',
    # whitespace / extra keys / trailing junk
    ' {"levels":[]}',
    '{"levels":[]} ',
    '{"levels" :[]}',
    '\t{"levels":[]}',
    '{"levels":[] ,}',
    '{,"levels":[]}',
    '{"levels":[,]}',
    '{"levels":[[] ,]}',
    '{"levels":[[ ,]]}',
    '{"LEVELS":[]}',
    '{"levels":[],"x":1}',
    '{"x":1,"levels":[]}',
    '{"levels":[],}',
    '{"levels":[]}{}',
    '{"levels":[]}{"levels":[]}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,1]]]}}',
    '{"levels":[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3]]]x}',
    '{"levels" :[[[0,0,0,0,255,255,1.500000,2.500000,2.000000,0.500000,3]]]}',
])
def test_decode_bad_text_raises_value_error(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid_stats(text)


def test_decode_rejects_non_canonical_but_semantically_equal_text():
    canonical = encode_tile_pyramid_stats(_pyramid(_tile()))
    variants = [
        canonical.replace(":", " : "),
        canonical.replace(",", ", "),
        canonical.replace('{"levels":', '{"levels" :'),
        '{"levels": [[[0, 0, 0, 0, 255, 255, 1.000000, 2.000000, '
        '1.500000, 0.500000, 1]]]}',
    ]
    for variant in variants:
        assert variant != canonical
        with pytest.raises(ValueError):
            decode_tile_pyramid_stats(variant)


def test_encode_decode_idempotent():
    pyramid = build_tile_pyramid_stats(
        [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0)],
        levels=2)
    once = encode_tile_pyramid_stats(pyramid)
    assert encode_tile_pyramid_stats(decode_tile_pyramid_stats(once)) == once
