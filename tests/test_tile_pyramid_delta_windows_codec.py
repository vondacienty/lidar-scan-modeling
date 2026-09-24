"""Tests for :func:`encode_tile_pyramid_delta_windows`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (assess_tile_pyramid_deltas, build_tile_pyramid,
                        encode_tile_pyramid_delta_windows,
                        query_tile_pyramid_delta_windows)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  dzmin=1.0, dzmax=2.0, dcount=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["dzmin"], fields["dzmax"],
            fields["dcount"])


def _window(level=0, ix_min=0, iy_min=0, ix_max=255, iy_max=255, tiles=()):
    return (level, ix_min, iy_min, ix_max, iy_max, tuple(tiles))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.encode_tile_pyramid_delta_windows
            is encode_tile_pyramid_delta_windows)
    import lidar_scan
    assert "encode_tile_pyramid_delta_windows" in lidar_scan.__all__
    assert (lidar_scan.encode_tile_pyramid_delta_windows
            is encode_tile_pyramid_delta_windows)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_empty_result():
    assert encode_tile_pyramid_delta_windows(()) == '{"windows":[]}'


def test_encode_window_with_empty_tiles():
    text = encode_tile_pyramid_delta_windows((_window(tiles=()),))
    assert text == '{"windows":[[0,0,0,255,255,[]]]}'


def test_encode_single_tile_canonical_text():
    text = encode_tile_pyramid_delta_windows((_window(tiles=[_tile()]),))
    assert text == (
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.000000,2.000000,1]]]]}'
    )


def test_encode_from_delta_windows_query_round_trips_through_json():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    assessment = assess_tile_pyramid_deltas(pyramid, pyramid)
    windows = ((0, -10, -410, 300, 0), (2, 0, 0, 1000, 1000),
               (1, 5000, 5000, 6000, 6000))
    result = query_tile_pyramid_delta_windows(assessment, windows)
    text = encode_tile_pyramid_delta_windows(result)
    assert json.loads(text)["windows"] and text.startswith('{"windows":')
    # The documented text re-parses into the same numeric structure.
    decoded = json.loads(text)["windows"]
    assert decoded[0][0:5] == [0, -10, -410, 300, 0]
    assert len(decoded[1][5]) == len(result[1][5])
    assert decoded[2][5] == []


def test_encode_keeps_window_and_tile_order():
    result = (
        _window(level=1, ix_min=-512, iy_min=-512, ix_max=767, iy_max=1023,
                tiles=[_tile(tx=-5, ty=-7, ix0=-512, iy0=-512, ix1=-257,
                             iy1=-257, dzmin=-3.25, dzmax=9.0, dcount=42),
                       _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767,
                             iy1=1023, dzmin=0.0, dzmax=1.5, dcount=7)]),
        _window(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0, tiles=[]),
    )
    text = encode_tile_pyramid_delta_windows(result)
    assert text == (
        '{"windows":['
        '[1,-512,-512,767,1023,'
        '[[-5,-7,-512,-512,-257,-257,-3.250000,9.000000,42],'
        '[2,3,512,768,767,1023,0.000000,1.500000,7]]],'
        '[0,0,0,0,0,[]]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_tile_pyramid_delta_windows((_window(tiles=[_tile()]),))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_tile_pyramid_delta_windows(
        (_window(tiles=[_tile(dzmin=-1 / 7, dzmax=1 / 3)]),))
    assert "-0.142857" in text and "0.333333" in text
    assert list(json.loads(text)) == ["windows"]


def test_encode_negative_zero_written_positive():
    text = encode_tile_pyramid_delta_windows(
        (_window(tiles=[_tile(dzmin=-0.0,
                              dzmax=math.copysign(0.0, -1.0))]),))
    assert "-0.000000" not in text
    assert "0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    result = (_window(level=10 ** 20, ix_min=-10 ** 40, iy_min=-10 ** 40,
                      ix_max=10 ** 40, iy_max=10 ** 40,
                      tiles=[_tile(tx=10 ** 40, ty=-10 ** 40,
                                   ix0=10 ** 40, iy0=-10 ** 40,
                                   ix1=10 ** 40 + 255,
                                   iy1=-10 ** 40 + 255,
                                   dcount=10 ** 80)]),)
    text = encode_tile_pyramid_delta_windows(result)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert json.loads(text)["windows"][0][0] == 10 ** 20


def test_encode_top_level_has_only_windows_key():
    text = encode_tile_pyramid_delta_windows((_window(),))
    assert list(json.loads(text).keys()) == ["windows"]


def test_encode_does_not_emit_non_finite_tokens():
    text = encode_tile_pyramid_delta_windows(
        (_window(tiles=[_tile()]),))
    assert "NaN" not in text and "Infinity" not in text


def test_encode_accepts_tile_merely_touching_window_boundary():
    # Closed intervals: a tile whose ix0 equals ix_max still intersects.
    result = (_window(ix_min=0, iy_min=768, ix_max=512, iy_max=1023,
                      tiles=[_tile(tx=2, ty=3, ix0=512, iy0=768,
                                   ix1=767, iy1=1023)]),)
    assert encode_tile_pyramid_delta_windows(result).startswith(
        '{"windows":')


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
        encode_tile_pyramid_delta_windows(bad)


@pytest.mark.parametrize("bad_result", [
    (_window()[:5],),                                   # 5-tuple window
    (_window() + (1,),),                                # 7-tuple window
    ([0, 0, 0, 255, 255, ()],),                         # window as list
    ((0, 0, 0, 255, 255, [_tile()]),),                  # tiles as list
    (_window(tiles=[_tile()[:8]]),),                    # 8-field tile
    (_window(tiles=[_tile() + (9,)]),),                 # 10-field tile
    (_window(tiles=[123]),),                            # non-tuple tile
    (_window(level=1.0),),
    (_window(level=True),),
    (_window(level=-1),),                               # negative level
    (_window(ix_min=False),),
    (_window(iy_min="0"),),
    (_window(ix_max=255.0),),
    (_window(tiles=[_tile(tx=1.0)]),),
    (_window(tiles=[_tile(ty=True)]),),
    (_window(tiles=[_tile(ix0=False)]),),
    (_window(tiles=[_tile(iy1="255")]),),
    (_window(tiles=[_tile(dcount=1.0)]),),
    (_window(tiles=[_tile(dcount=True)]),),
    (_window(tiles=[_tile(dzmin=1)]),),
    (_window(tiles=[_tile(dzmax=2.0 + 0j)]),),
    (_window(tiles=[_tile(dzmin=float("nan"))]),),
    (_window(tiles=[_tile(dzmax=float("inf"))]),),
    (_window(tiles=[_tile(dzmin=float("-inf"))]),),
    (_window(ix_min=300, ix_max=255),),                 # inverted ix bounds
    (_window(iy_min=300, iy_max=255),),                 # inverted iy bounds
    (_window(tiles=[_tile(tx=0, ty=1),
                    _tile(tx=0, ty=0)]),),              # unsorted tiles
    (_window(tiles=[_tile(), _tile()]),),               # duplicate tile key
    (_window(tiles=[_tile(ix0=10, ix1=9)]),),           # ix1 < ix0
    (_window(tiles=[_tile(iy0=10, iy1=9)]),),           # iy1 < iy0
    (_window(ix_max=255,
             tiles=[_tile(tx=1, ty=0, ix0=256, iy0=0,
                          ix1=511, iy1=255)]),),        # tile past ix_max
    (_window(ix_min=256,
             tiles=[_tile(tx=0, ty=0, ix0=0, iy0=0,
                          ix1=255, iy1=255)]),),        # tile before ix_min
    (_window(iy_max=255,
             tiles=[_tile(tx=0, ty=1, ix0=0, iy0=256,
                          ix1=255, iy1=511)]),),        # tile past iy_max
    (_window(iy_min=256,
             tiles=[_tile(tx=0, ty=0, ix0=0, iy0=0,
                          ix1=255, iy1=255)]),),        # tile before iy_min
])
def test_encode_bad_structure_raises_value_error(bad_result):
    with pytest.raises(ValueError):
        encode_tile_pyramid_delta_windows(bad_result)


def test_encode_does_not_modify_input():
    result = (_window(tiles=[_tile()]),)
    snapshot = (_window(tiles=[_tile()]),)
    encode_tile_pyramid_delta_windows(result)
    assert result == snapshot


def test_encode_repeated_calls_are_identical():
    points = [(0.1, 0.2, 1.0, 1.0, 0.5), (300.0, -400.0, 2.0, 0.0, 1.0),
              (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=2)
    assessment = assess_tile_pyramid_deltas(pyramid, pyramid)
    result = query_tile_pyramid_delta_windows(
        assessment, ((0, -10, -410, 300, 0), (1, 0, 0, 1000, 1000)))
    once = encode_tile_pyramid_delta_windows(result)
    assert encode_tile_pyramid_delta_windows(result) == once
