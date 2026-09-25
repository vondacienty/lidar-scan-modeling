"""Tests for :func:`encode_rollback_tile_pyramid_delta_windows` /
:func:`decode_rollback_tile_pyramid_delta_windows`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid,
                        assess_tile_pyramid_deltas,
                        decode_rollback_tile_pyramid_delta_windows,
                        encode_rollback_tile_pyramid_delta_windows,
                        rollback_tile_pyramid_delta_windows)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


def _window(level=0, ix_min=0, iy_min=0, ix_max=255, iy_max=255, tiles=()):
    return (level, ix_min, iy_min, ix_max, iy_max, tuple(tiles))


def _delta(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  dzmin=0.0, dzmax=0.0, dcount=0)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["dzmin"],
            fields["dzmax"], fields["dcount"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert (tiles_module.encode_rollback_tile_pyramid_delta_windows
            is encode_rollback_tile_pyramid_delta_windows)
    assert (tiles_module.decode_rollback_tile_pyramid_delta_windows
            is decode_rollback_tile_pyramid_delta_windows)
    import lidar_scan
    assert ("encode_rollback_tile_pyramid_delta_windows"
            in lidar_scan.__all__)
    assert ("decode_rollback_tile_pyramid_delta_windows"
            in lidar_scan.__all__)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_empty_result():
    assert encode_rollback_tile_pyramid_delta_windows(()) == '{"windows":[]}'


def test_encode_window_with_empty_tiles():
    text = encode_rollback_tile_pyramid_delta_windows((_window(tiles=()),))
    assert text == '{"windows":[[0,0,0,255,255,[]]]}'


def test_encode_single_tile_canonical_text():
    text = encode_rollback_tile_pyramid_delta_windows(
        (_window(tiles=[_tile()]),))
    assert text == (
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.000000,2.000000,1]]]]}'
    )


def test_encode_from_rollback_result_round_trips():
    base = ((
        _tile(tx=0, ty=0, zmin=10.0, zmax=20.0, count=8),
        _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=5.0, count=4),
    ),)
    deltas = ((
        _delta(dzmin=2.0, dzmax=-2.0, dcount=3),
        _delta(tx=1, ty=0, ix0=256, ix1=511, dzmin=1.0, dzmax=1.0, dcount=4),
    ),)
    result = rollback_tile_pyramid_delta_windows(
        base, deltas, ((0, 256, 0, 300, 255),))
    text = encode_rollback_tile_pyramid_delta_windows(result)
    assert decode_rollback_tile_pyramid_delta_windows(text) == result


def test_encode_from_built_pyramids_round_trips():
    estimate = build_tile_pyramid(
        [(0.0, 0.0, 3.0, 1, 1.0), (300.0, 0.0, 2.0, 1, 1.0)], levels=2)
    reference = build_tile_pyramid(
        [(0.0, 0.0, 1.0, 1, 1.0), (300.0, 0.0, 5.0, 1, 1.0)], levels=2)
    deltas = assess_tile_pyramid_deltas(estimate, reference)
    windows = ((0, 0, 0, 600, 600), (1, 0, 0, 600, 600))
    result = rollback_tile_pyramid_delta_windows(estimate, deltas, windows)
    text = encode_rollback_tile_pyramid_delta_windows(result)
    assert decode_rollback_tile_pyramid_delta_windows(text) == result


def test_encode_keeps_window_and_tile_order():
    result = (
        _window(level=1, ix_min=-5, iy_min=-7, ix_max=9, iy_max=11,
                tiles=[_tile(tx=-5, ty=-7, zmin=-3.25, zmax=9.0, count=42),
                       _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767,
                             iy1=1023, zmin=0.0, zmax=1.5, count=7)]),
        _window(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0, tiles=[]),
    )
    text = encode_rollback_tile_pyramid_delta_windows(result)
    assert text == (
        '{"windows":['
        '[1,-5,-7,9,11,'
        '[[-5,-7,0,0,255,255,-3.250000,9.000000,42],'
        '[2,3,512,768,767,1023,0.000000,1.500000,7]]],'
        '[0,0,0,0,0,[]]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_rollback_tile_pyramid_delta_windows(
        (_window(tiles=[_tile()]),))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_rollback_tile_pyramid_delta_windows(
        (_window(tiles=[_tile(zmin=-1 / 7, zmax=1 / 3)]),))
    assert "-0.142857" in text and "0.333333" in text
    assert list(json.loads(text)) == ["windows"]


def test_encode_negative_zero_written_positive():
    text = encode_rollback_tile_pyramid_delta_windows(
        (_window(tiles=[_tile(zmin=-0.0, zmax=math.copysign(0.0, -1.0))]),))
    assert "-0.000000" not in text
    assert "0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    result = (_window(level=10 ** 20, ix_min=-10 ** 40, iy_min=-10 ** 40,
                      ix_max=10 ** 40, iy_max=10 ** 40,
                      tiles=[_tile(tx=10 ** 40, ty=-10 ** 40,
                                   count=10 ** 80)]),)
    text = encode_rollback_tile_pyramid_delta_windows(result)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert decode_rollback_tile_pyramid_delta_windows(text) == result


def test_encode_top_level_has_only_windows_key():
    text = encode_rollback_tile_pyramid_delta_windows((_window(),))
    assert list(json.loads(text).keys()) == ["windows"]


@pytest.mark.parametrize("bad", [
    [_window()],
    {"windows": []},
    "((),)",
    1,
    1.5,
    True,
    False,
    None,
    {_window()},
])
def test_encode_non_tuple_outer_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_rollback_tile_pyramid_delta_windows(bad)


@pytest.mark.parametrize("bad_result", [
    (_window()[:5],),                                   # 5-tuple window
    (_window() + (1,),),                                # 7-tuple window
    ([0, 0, 0, 255, 255, ()],),                         # window as list
    ((0, 0, 0, 255, 255, [_tile()]),),                  # tiles as list
    (_window(tiles=[_tile()[:8]]),),                    # 8-field tile
    (_window(tiles=[_tile() + (9,)]),),                 # 10-field tile
    (_window(tiles=[123]),),                            # non-tuple tile
    (_window(level=-1),),                               # negative level
    (_window(level=1.0),),
    (_window(level=True),),
    (_window(ix_min=False),),
    (_window(iy_min="0"),),
    (_window(ix_max=255.0),),
    (_window(tiles=[_tile(tx=1.0)]),),
    (_window(tiles=[_tile(ty=True)]),),
    (_window(tiles=[_tile(ix0=False)]),),
    (_window(tiles=[_tile(iy1="255")]),),
    (_window(tiles=[_tile(count=1.0)]),),
    (_window(tiles=[_tile(count=True)]),),
    (_window(tiles=[_tile(count=-1)]),),                # negative count
    (_window(tiles=[_tile(zmin=1)]),),
    (_window(tiles=[_tile(zmax=2.0 + 0j)]),),
    (_window(tiles=[_tile(zmin=float("nan"))]),),
    (_window(tiles=[_tile(zmax=float("inf"))]),),
    (_window(tiles=[_tile(zmin=float("-inf"))]),),
    (_window(ix_min=300, ix_max=255),),                 # inverted ix bounds
    (_window(iy_min=300, iy_max=255),),                 # inverted iy bounds
    (_window(tiles=[_tile(zmin=3.0, zmax=2.0)]),),      # zmin > zmax
    (_window(tiles=[_tile(tx=0, ty=1),
                    _tile(tx=0, ty=0)]),),              # unsorted tiles
    (_window(tiles=[_tile(tx=1, ty=1),
                    _tile(tx=1, ty=1)]),),              # duplicate tile key
    (_window(tiles=[_tile(ix0=10, ix1=9)]),),           # ix1 < ix0
    (_window(tiles=[_tile(iy0=10, iy1=9)]),),           # iy1 < iy0
])
def test_encode_bad_structure_raises_value_error(bad_result):
    with pytest.raises(ValueError):
        encode_rollback_tile_pyramid_delta_windows(bad_result)


def test_encode_does_not_modify_input():
    result = (_window(tiles=[_tile()]),)
    snapshot = (_window(tiles=[_tile()]),)
    encode_rollback_tile_pyramid_delta_windows(result)
    assert result == snapshot


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_returns_tuples_and_preserves_order():
    result = (
        _window(level=1, ix_min=-5, iy_min=-7, ix_max=9, iy_max=11,
                tiles=[_tile(tx=-5, ty=-7),
                       _tile(tx=2, ty=3, ix0=512, iy0=768,
                             ix1=767, iy1=1023)]),
        _window(level=0, tiles=[]),
    )
    decoded = decode_rollback_tile_pyramid_delta_windows(
        encode_rollback_tile_pyramid_delta_windows(result))
    assert decoded == result
    assert isinstance(decoded, tuple)
    assert all(isinstance(window, tuple) for window in decoded)
    assert all(isinstance(window[5], tuple) for window in decoded)
    assert all(isinstance(tile, tuple)
               for window in decoded for tile in window[5])


def test_decode_empty_variants():
    assert decode_rollback_tile_pyramid_delta_windows(
        '{"windows":[]}') == ()
    assert decode_rollback_tile_pyramid_delta_windows(
        '{"windows":[[0,0,0,255,255,[]]]}') == (_window(tiles=()),)
    assert decode_rollback_tile_pyramid_delta_windows(
        '{"windows":[[0,0,0,255,255,[]],[1,0,0,9,9,[]]]}'
    ) == (_window(tiles=()), _window(level=1, ix_max=9, iy_max=9, tiles=()))


def test_decode_field_python_types():
    result = decode_rollback_tile_pyramid_delta_windows(
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.500000,2.500000,3]]]]}'
    )
    window = result[0]
    for index in range(5):
        assert type(window[index]) is int
    tile = window[5][0]
    for index in range(6):
        assert type(tile[index]) is int
    assert type(tile[6]) is float and type(tile[7]) is float
    assert type(tile[8]) is int


def test_decode_zero_count_is_allowed():
    result = decode_rollback_tile_pyramid_delta_windows(
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.500000,2.500000,0]]]]}'
    )
    assert result[0][5][0][8] == 0


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"...", [], {}, True, object()])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_rollback_tile_pyramid_delta_windows(bad)


@pytest.mark.parametrize("text", [
    "",
    "{",
    "}",
    "null",
    "[]",
    "42",
    '"windows"',
    "true",
    "{}",
    '{"levels":[]}',
    '[{}]',
    '{"windows":[',
    '{"windows":}',
    '{"windows":[}',
    '{"windows":null}',
    '{"windows":{}}',
    '{"windows":42}',
    '{"windows":[{}]}',
    '{"windows":[[]]}',
    '{"windows":[[0,0,0,255,255]]}',
    '{"windows":[[0,0,0,255,255,null]]}',
    '{"windows":[[0,0,0,255,255,[null]]]}',
    '{"windows":[[0,0,0,255,255,[[]]]]}',
    '{"windows":[[0,0,0,255,255,[[0,0,0,0,255,255]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.500000,2.500000,3,4]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,"a",2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,true,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,null,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255.0,255,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,0255,255,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.500000,2.500000,3.0]]]]}',
    '{"windows":[[true,0,0,255,255,[]]]}',
    '{"windows":[[0,0,0,255,255.0,[]]]}',
    # numeric formatting
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.0,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.5000000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.500000e0,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,+1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,-0.000000,2.500000,3]]]]}',
    # non-finite literals
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,NaN,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,Infinity,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,-Infinity,2.500000,3]]]]}',
    # level / bounds / ordering / duplicates / z / count
    '{"windows":[[-1,0,0,255,255,[]]]}',
    '{"windows":[[0,300,0,255,255,[]]]}',
    '{"windows":[[0,0,300,255,255,[]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,10,0,9,255,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,10,255,9,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,2.500000,1.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.500000,2.500000,-1]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,1,0,0,255,255,1.500000,2.500000,3],'
    '[0,0,0,0,255,255,1.500000,2.500000,3]]]]}',
    '{"windows":[[0,0,0,255,255,'
    '[[0,0,0,0,255,255,1.500000,2.500000,3],'
    '[0,0,0,0,255,255,9.000000,9.000000,1]]]]}',
    # whitespace / extra keys / trailing junk
    ' {"windows":[]}',
    '{"windows":[]} ',
    '{"windows" :[]}',
    '\t{"windows":[]}',
    '{"windows":[] ,}',
    '{,"windows":[]}',
    '{"windows":[,]}',
    '{"WINDOWS":[]}',
    '{"windows":[],"x":1}',
    '{"x":1,"windows":[]}',
    '{"windows":[]}{}',
    '{"windows":[]}{"windows":[]}',
])
def test_decode_bad_text_raises_value_error(text):
    with pytest.raises(ValueError):
        decode_rollback_tile_pyramid_delta_windows(text)


def test_decode_rejects_non_canonical_but_semantically_equal_text():
    canonical = encode_rollback_tile_pyramid_delta_windows(
        (_window(tiles=[_tile()]),))
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
            decode_rollback_tile_pyramid_delta_windows(variant)


def test_decode_does_not_modify_input():
    text = ('{"windows":[[0,0,0,255,255,'
            '[[0,0,0,0,255,255,1.000000,2.000000,1]]]]}')
    snapshot = text
    decode_rollback_tile_pyramid_delta_windows(text)
    assert text == snapshot


def test_encode_decode_idempotent():
    estimate = build_tile_pyramid(
        [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0)],
        levels=2)
    reference = build_tile_pyramid(
        [(0.1, 0.2, 0.5, 1.0, 0.5), (300.0, -400.0, -3.5, 0.0, 1.0)],
        levels=2)
    deltas = assess_tile_pyramid_deltas(estimate, reference)
    result = rollback_tile_pyramid_delta_windows(
        estimate, deltas, ((0, -10, -410, 300, 0), (1, 0, 0, 1000, 1000)))
    once = encode_rollback_tile_pyramid_delta_windows(result)
    assert encode_rollback_tile_pyramid_delta_windows(
        decode_rollback_tile_pyramid_delta_windows(once)) == once
