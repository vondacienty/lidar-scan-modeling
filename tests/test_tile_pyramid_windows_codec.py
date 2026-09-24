"""Tests for :func:`encode_tile_pyramid_windows` /
:func:`decode_tile_pyramid_windows`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid, decode_tile_pyramid_windows,
                        encode_tile_pyramid_windows,
                        merge_tile_pyramid_windows)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _window(*fields):
    """Build a 6-tuple window; default empty tiles."""
    if len(fields) == 5:
        fields = fields + ((),)
    return tuple(fields)


def _result(*windows):
    return tuple(windows)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.encode_tile_pyramid_windows is encode_tile_pyramid_windows
    assert tiles_module.decode_tile_pyramid_windows is decode_tile_pyramid_windows
    import lidar_scan
    assert "encode_tile_pyramid_windows" in lidar_scan.__all__
    assert "decode_tile_pyramid_windows" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_single_window_single_tile_canonical_text():
    text = encode_tile_pyramid_windows(
        _result(_window(0, 0, 0, 255, 255, (_tile(),))))
    assert text == (
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.000000,2.000000,1]]]]}'
    )


def test_encode_empty_result():
    assert encode_tile_pyramid_windows(()) == '{"windows":[]}'


def test_encode_empty_tiles_preserved():
    text = encode_tile_pyramid_windows(
        _result(_window(1, -3, -4, 2, 8), _window(0, 0, 0, 0, 0)))
    assert text == '{"windows":[[1,-3,-4,2,8,[]],[0,0,0,0,0,[]]]}'


def test_encode_from_merge_round_trips():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    result = merge_tile_pyramid_windows(
        (pyramid,), ((0, -5, -5, 300, 300), (2, 0, 0, 100, 100)))
    text = encode_tile_pyramid_windows(result)
    assert decode_tile_pyramid_windows(text) == result


def test_encode_keeps_window_and_tile_order():
    result = _result(
        _window(2, 0, 0, 9, 9, ()),
        _window(0, -10, -10, 1000, 1000, (
            _tile(tx=-5, ty=-7, zmin=-3.25, zmax=9.0, count=42),
            _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023,
                  zmin=0.0, zmax=1.5, count=7),
        )),
    )
    text = encode_tile_pyramid_windows(result)
    assert text == (
        '{"windows":[[2,0,0,9,9,[]],[0,-10,-10,1000,1000,['
        '[-5,-7,0,0,255,255,-3.250000,9.000000,42],'
        '[2,3,512,768,767,1023,0.000000,1.500000,7]]]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_tile_pyramid_windows(
        _result(_window(0, 0, 0, 255, 255, (_tile(),))))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    result = _result(_window(0, 0, 0, 255, 255,
                             (_tile(zmin=-1 / 7, zmax=1 / 3),)))
    text = encode_tile_pyramid_windows(result)
    assert "-0.142857" in text and "0.333333" in text
    assert list(json.loads(text)) == ["windows"]


def test_encode_negative_zero_written_positive():
    result = _result(_window(
        0, 0, 0, 255, 255,
        (_tile(zmin=-0.0, zmax=math.copysign(0.0, -1.0)),)))
    text = encode_tile_pyramid_windows(result)
    assert "-0.000000" not in text
    assert "0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    result = _result(_window(
        10 ** 40, -10 ** 40, 0, 10 ** 40, 10 ** 40,
        (_tile(tx=10 ** 40, ty=-10 ** 40, count=10 ** 80),)))
    text = encode_tile_pyramid_windows(result)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert decode_tile_pyramid_windows(text) == result


def test_encode_top_level_has_only_windows_key():
    text = encode_tile_pyramid_windows(_result(_window(0, 0, 0, 0, 0)))
    assert list(json.loads(text).keys()) == ["windows"]


@pytest.mark.parametrize("bad", [
    [_window(0, 0, 0, 0, 0)],
    {"windows": []},
    "((),)",
    1,
    1.5,
    True,
    False,
    None,
    {_window(0, 0, 0, 0, 0)},
])
def test_encode_non_tuple_outer_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_pyramid_windows(bad)


@pytest.mark.parametrize("bad_result", [
    _result([0, 0, 0, 0, 0, ()]),                      # window is a list
    _result(_window(0, 0, 0, 0, 0)[:5]),              # 5-tuple window
    _result(_window(0, 0, 0, 0, 0, (), ())),          # 7-tuple window
    _result(123),                                      # non-tuple window
    _result(_window(True, 0, 0, 0, 0)),
    _result(_window(0, 1.0, 0, 0, 0)),
    _result(_window(0, 0, False, 0, 0)),
    _result(_window(0, 0, 0, "255", 0)),
    _result(_window(0, 0, 0, 0, None)),
    _result(_window(0, 1, 0, 0, 0)),                   # ix_min > ix_max
    _result(_window(0, 0, 1, 0, 0)),                   # iy_min > iy_max
    _result(_window(0, 0, 0, 0, 0, [_tile()])),        # tiles is a list
    _result(_window(0, 0, 0, 0, 0, (list(_tile()),))),  # tile is a list
    _result(_window(0, 0, 0, 0, 0, (_tile()[:8],))),   # 8-field tile
    _result(_window(0, 0, 0, 0, 0, (_tile() + (9,),))),  # 10-field tile
    _result(_window(0, 0, 0, 0, 0, (123,))),
    _result(_window(0, 0, 0, 0, 0, (_tile(tx=1.0),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(ty=True),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(ix0=False),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(count=1.0),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(count=True),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(zmin=1),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(zmax=2.0 + 0j),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(zmin=float("nan")),))),
    _result(_window(0, 0, 0, 0, 0, (_tile(zmax=float("inf")),))),
    _result(_window(0, 0, 0, 0, 0,
                    (_tile(ix0=1, ix1=0),))),          # inverted ix bounds
    _result(_window(0, 0, 0, 0, 0,
                    (_tile(iy0=1, iy1=0),))),          # inverted iy bounds
    _result(_window(0, 0, 0, 0, 0,
                    (_tile(tx=0, ty=1), _tile(tx=0, ty=0)))),  # unsorted
    _result(_window(0, 0, 0, 0, 0,
                    (_tile(tx=1, ty=1), _tile(tx=1, ty=1)))),  # duplicate
])
def test_encode_bad_structure_raises_value_error(bad_result):
    with pytest.raises(ValueError):
        encode_tile_pyramid_windows(bad_result)


def test_encode_does_not_modify_input():
    window = _window(0, 0, 0, 255, 255, (_tile(),))
    result = _result(window)
    encode_tile_pyramid_windows(result)
    assert result == _result(window)


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_returns_tuples_and_preserves_order():
    result = _result(
        _window(2, 0, 0, 9, 9, ()),
        _window(0, -10, -10, 1000, 1000, (
            _tile(tx=-5, ty=-7),
            _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023),
        )),
    )
    decoded = decode_tile_pyramid_windows(
        encode_tile_pyramid_windows(result))
    assert decoded == result
    assert isinstance(decoded, tuple)
    assert all(isinstance(window, tuple) for window in decoded)
    assert all(isinstance(window[5], tuple) for window in decoded)
    assert all(isinstance(tile, tuple)
               for window in decoded for tile in window[5])


def test_decode_z_values_are_floats_and_indices_are_ints():
    decoded = decode_tile_pyramid_windows(
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.500000,2.500000,3]]]]}'
    )
    tile = decoded[0][5][0]
    assert isinstance(tile[6], float) and isinstance(tile[7], float)
    for index in (0, 1, 2, 3, 4, 5, 8):
        assert isinstance(tile[index], int)
        assert not isinstance(tile[index], bool)


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"...", [], {}, True, object()])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_tile_pyramid_windows(bad)


@pytest.mark.parametrize("text", [
    "",
    "{",
    "}",
    "null",
    "[]",
    "42",
    '"windows"',
    "true",
    "[{}]",
    '{"windows":[',
    '{"windows":}',
    '{"windows":[}',
    '{"windows":null}',
    '{"windows":{}}',
    '{"windows":[{}]}',
    '{"windows":[null]}',
    '{"windows":[42]}',
    '{"windows":["x"]}',
    '{"windows":[[null]]}',
    '{"windows":[[[]]]}',
    '{"windows":[[0,0,0,0,0]]}',                # five fields
    '{"windows":[[0,0,0,0,0,[],1]]}',           # seven fields
    '{"windows":[[0,0,0,0,0]]}',
    '{"windows":[[0,0,0,0,0,{}]]}',
    '{"windows":[[0,0,0,0,0,[42]]]}',
    '{"windows":[[0,0,0,0,0,[null]]]}',
    # field types / bools
    '{"windows":[[true,0,0,0,0,[]]]}',
    '{"windows":[[0,1.0,0,0,0,[]]]}',
    '{"windows":[[0,0,false,0,0,[]]]}',
    '{"windows":[[0,0,0,"9",0,[]]]}',
    '{"windows":[[0,0,0,0,null,[]]]}',
    # bounds
    '{"windows":[[0,1,0,0,0,[]]]}',             # ix_min > ix_max
    '{"windows":[[0,0,1,0,0,[]]]}',             # iy_min > iy_max
    # tile shape / types
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,2.500000]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,2.500000,1,2]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,true,1.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,"a",2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,null,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[true,0,0,0,255,255,1.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,2.500000,1.0]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,2.500000,true]]]]}',
    # tile bounds inverted
    '{"windows":[[0,0,0,255,255,[[0,0,256,0,255,255,1.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,256,255,255,1.500000,2.500000,1]]]]}',
    # non-finite literals
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,NaN,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,Infinity,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,-Infinity,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,NaN,1]]]]}',
    # tile ordering / duplicates
    '{"windows":[[0,0,0,255,255,['
    '[0,1,0,0,255,255,1.500000,2.500000,1],'
    '[0,0,0,0,255,255,1.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,['
    '[0,0,0,0,255,255,1.500000,2.500000,1],'
    '[0,0,0,0,255,255,9.000000,9.000000,2]]]]}',
    # numeric formatting
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.0,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.5000000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000e0,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,00.500000,2.500000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255,1.500000,-0.000000,1]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,0255,255,1.500000,2.500000,1]]]]}',
    '{"windows":[[00,0,0,255,255,[[0,0,0,0,255,255,1.500000,2.500000,1]]]]}',
    # whitespace / extra keys / trailing junk
    ' {"windows":[]}',
    '{"windows":[]} ',
    '{"windows" :[]}',
    '\t{"windows":[]}',
    '{"windows":[] ,}',
    '{,"windows":[]}',
    '{"windows":[,]}',
    '{"windows":[[0,0,0,0,0,[]] ,]}',
    '{"WINDOWS":[]}',
    '{"levels":[]}',
    '{"windows":[],"x":1}',
    '{"x":1,"windows":[]}',
    '{"windows":[]}{}',
    '{"windows":[]}{"windows":[]}',
    '{"windows":[[0,0,0,0,0,[]]]}}',
    '{"windows":[[0,0,0,0,0,[]]]x}',
])
def test_decode_bad_text_raises_value_error(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid_windows(text)


def test_decode_rejects_non_canonical_but_semantically_equal_text():
    canonical = encode_tile_pyramid_windows(
        _result(_window(0, 0, 0, 255, 255, (_tile(),))))
    variants = [
        canonical.replace(":", " : "),
        canonical.replace(",", ", "),
        canonical.replace('{"windows":', '{"windows" :'),
        '{"windows": [[0, 0, 0, 255, 255, '
        '[[0, 0, 0, 0, 255, 255, 1.000000, 2.000000, 1]]]]}',
    ]
    for variant in variants:
        assert variant != canonical
        with pytest.raises(ValueError):
            decode_tile_pyramid_windows(variant)


def test_encode_decode_idempotent():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=2)
    result = merge_tile_pyramid_windows(
        (pyramid,), ((0, -5, -5, 300, 300), (1, 0, 0, 100, 100)))
    once = encode_tile_pyramid_windows(result)
    assert encode_tile_pyramid_windows(
        decode_tile_pyramid_windows(once)) == once
