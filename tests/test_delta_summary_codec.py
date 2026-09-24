"""Tests for :func:`encode_delta_summary` and :func:`decode_delta_summary`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (aggregate_tile_pyramid_delta_windows,
                        assess_tile_pyramid_deltas, build_tile_pyramid,
                        decode_delta_summary, encode_delta_summary)
from lidar_scan import tiles as tiles_module


def _summary(min_dzmin=1.0, max_dzmax=2.0, sum_dcount=4, match_count=2):
    return (min_dzmin, max_dzmax, sum_dcount, match_count)


def _window(level=0, ix_min=0, iy_min=0, ix_max=255, iy_max=255, summary=None):
    return (level, ix_min, iy_min, ix_max, iy_max, summary)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.encode_delta_summary is encode_delta_summary
    assert tiles_module.decode_delta_summary is decode_delta_summary
    import lidar_scan
    assert "encode_delta_summary" in lidar_scan.__all__
    assert "decode_delta_summary" in lidar_scan.__all__
    assert lidar_scan.encode_delta_summary is encode_delta_summary
    assert lidar_scan.decode_delta_summary is decode_delta_summary


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_empty_result():
    assert encode_delta_summary(()) == '{"windows":[]}'


def test_encode_null_summary():
    assert encode_delta_summary((_window(),)) == '{"windows":[[0,0,0,255,255,null]]}'


def test_encode_summary_canonical_text():
    text = encode_delta_summary((_window(summary=_summary()),))
    assert text == '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4,2]]]}'


def test_encode_keeps_window_order():
    result = (
        _window(level=1, ix_min=-512, iy_min=-512, ix_max=767, iy_max=1023,
                summary=_summary(-3.25, 9.0, 42, 7)),
        _window(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0),
    )
    assert encode_delta_summary(result) == (
        '{"windows":['
        '[1,-512,-512,767,1023,[-3.250000,9.000000,42,7]],'
        '[0,0,0,0,0,null]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_delta_summary((_window(summary=_summary()),))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_delta_summary(
        (_window(summary=_summary(-1 / 7, 1 / 3)),))
    assert "-0.142857" in text and "0.333333" in text


def test_encode_negative_zero_written_positive():
    text = encode_delta_summary(
        (_window(summary=_summary(-0.0, math.copysign(0.0, -1.0))),))
    assert "-0.000000" not in text
    assert "0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    result = (_window(level=10 ** 20, ix_min=-10 ** 40, iy_min=-10 ** 40,
                      ix_max=10 ** 40, iy_max=10 ** 40,
                      summary=_summary(sum_dcount=10 ** 80,
                                       match_count=10 ** 20)),)
    text = encode_delta_summary(result)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert str(10 ** 80) in text
    assert json.loads(text)["windows"][0][0] == 10 ** 20


def test_encode_top_level_has_only_windows_key():
    text = encode_delta_summary((_window(),))
    assert list(json.loads(text).keys()) == ["windows"]


def test_encode_does_not_emit_non_finite_tokens():
    text = encode_delta_summary((_window(summary=_summary()),))
    assert "NaN" not in text and "Infinity" not in text


def test_encode_from_aggregate_round_trips_through_json():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    assessment = assess_tile_pyramid_deltas(pyramid, pyramid)
    windows = ((0, -10, -410, 300, 0), (2, 0, 0, 1000, 1000),
               (1, 5000, 5000, 6000, 6000))
    result = aggregate_tile_pyramid_delta_windows(assessment, windows)
    text = encode_delta_summary(result)
    assert text.startswith('{"windows":')
    decoded = json.loads(text)["windows"]
    assert decoded[2][5] is None
    assert decoded[0][5][3] == result[0][5][3]
    assert decoded[1][5] is not None


@pytest.mark.parametrize("bad", [
    [_window()],
    {"windows": []},
    "((),)",
    1,
    1.5,
    True,
    False,
    None,
    [_window()],
    {_window()},
])
def test_encode_non_tuple_outer_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_delta_summary(bad)


@pytest.mark.parametrize("bad_result", [
    (_window()[:5],),                                   # 5-tuple window
    (_window() + (1,),),                                # 7-tuple window
    (_window(level=1.0),),
    (_window(level=True),),
    (_window(level=-1),),                               # negative level
    (_window(ix_min=False),),
    (_window(iy_min="0"),),
    (_window(ix_max=255.0),),
    (_window(ix_min=300, ix_max=255),),                 # inverted ix bounds
    (_window(iy_min=300, iy_max=255),),                 # inverted iy bounds
    (_window(summary=[]),),                             # summary empty list
    (_window(summary=()),),                             # summary empty tuple
    (_window(summary=_summary()[:3]),),                 # 3-field summary
    (_window(summary=_summary() + (1,)),),              # 5-field summary
    (_window(summary=[1.0, 2.0, 4, 2]),),               # summary as list
    (_window(summary=_summary(min_dzmin=1)),),          # int min_dzmin
    (_window(summary=_summary(max_dzmax=2)),),          # int max_dzmax
    (_window(summary=_summary(min_dzmin=2.0 + 0j)),),
    (_window(summary=_summary(max_dzmax=2.0 + 0j)),),
    (_window(summary=_summary(min_dzmin=float("nan")))),
    (_window(summary=_summary(max_dzmax=float("inf")))),
    (_window(summary=_summary(min_dzmin=float("-inf")))),
    (_window(summary=_summary(min_dzmin=2.0,
                              max_dzmax=1.0)),),         # min > max
    (_window(summary=_summary(sum_dcount=4.0)),),
    (_window(summary=_summary(sum_dcount=True)),),
    (_window(summary=_summary(match_count=False)),),
    (_window(summary=_summary(match_count=2.0)),),
    (_window(summary=_summary(match_count=0)),),
    (_window(summary=_summary(match_count=-1)),),
])
def test_encode_bad_structure_raises_value_error(bad_result):
    with pytest.raises(ValueError):
        encode_delta_summary(bad_result)


def test_encode_does_not_modify_input():
    result = (_window(summary=_summary()), _window())
    snapshot = (_window(summary=_summary()), _window())
    encode_delta_summary(result)
    assert result == snapshot


def test_encode_repeated_calls_are_identical():
    points = [(0.1, 0.2, 1.0, 1.0, 0.5), (300.0, -400.0, 2.0, 0.0, 1.0),
              (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=2)
    assessment = assess_tile_pyramid_deltas(pyramid, pyramid)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, -10, -410, 300, 0), (1, 0, 0, 1000, 1000)))
    once = encode_delta_summary(result)
    assert encode_delta_summary(result) == once


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_empty_document():
    assert decode_delta_summary('{"windows":[]}') == ()


def test_decode_null_summary():
    assert decode_delta_summary('{"windows":[[0,0,0,255,255,null]]}') == (
        _window(),)


def test_decode_summary():
    text = '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4,2]]]}'
    assert decode_delta_summary(text) == (_window(summary=_summary()),)


def test_decode_preserves_order_and_returns_tuples():
    text = (
        '{"windows":['
        '[1,-512,-512,767,1023,[-3.250000,9.000000,42,7]],'
        '[0,0,0,0,0,null]]}'
    )
    result = decode_delta_summary(text)
    assert result == (
        _window(level=1, ix_min=-512, iy_min=-512, ix_max=767, iy_max=1023,
                summary=(-3.25, 9.0, 42, 7)),
        (0, 0, 0, 0, 0, None),
    )
    assert isinstance(result, tuple)
    assert all(isinstance(window, tuple) for window in result)
    assert isinstance(result[0][5], tuple)
    assert result[1][5] is None


def test_round_trip_through_real_aggregation():
    points = [(0.1, 0.2, 1 / 3, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
              (-0.1, 0.0, -0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    pyramid = build_tile_pyramid(points, levels=3)
    assessment = assess_tile_pyramid_deltas(pyramid, pyramid)
    result = aggregate_tile_pyramid_delta_windows(
        assessment, ((0, -10, -410, 300, 0), (2, 0, 0, 1000, 1000),
                     (1, 5000, 5000, 6000, 6000)))
    assert decode_delta_summary(encode_delta_summary(result)) == result


@pytest.mark.parametrize("bad", [
    1,
    1.5,
    True,
    None,
    [],
    {},
    b'{"windows":[]}',
])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_delta_summary(bad)


@pytest.mark.parametrize("bad_text", [
    '',
    'not json',
    '{}',
    '[]',
    'null',
    '{"foo":[]}',
    '{"windows":[],"x":1}',
    '{"windows":[],"windows":[]}',
    '{"Windows":[]}',
    ' {"windows":[]}',
    '{"windows":[]} ',
    '{"windows" :[]}',
    '{"windows":{}}',
    '{"windows":null}',
    '{"windows":[{}]}',
    '{"windows":[[0,0,0,255,255]]}',
    '{"windows":[[0,0,0,255,255,null,1]]}',
    '{"windows":[[0,0,0,255,255,true]]}',
    '{"windows":[[0,0,0,255,255,[]]]}',
    '{"windows":[[0,0,0,255,255,{}]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4]]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4,2,9]]]}',
    '{"windows":[[0,0,0,255,255,[1,2,4,2]]]}',
    '{"windows":[[0,0,0,255,255,[1.0,2.0,4,2]]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4.0,2]]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4,2.0]]]}',
    '{"windows":[[0,0,0,255,255,[NaN,2.000000,4,2]]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,Infinity,4,2]]]}',
    '{"windows":[[0,0,0,255,255,[2.000000,1.000000,4,2]]]}',
    '{"windows":[[0,0,0,255,255,[1.000000,2.000000,4,0]]]}',
    '{"windows":[[-1,0,0,255,255,null]]}',
    '{"windows":[[true,0,0,255,255,null]]}',
    '{"windows":[[0.5,0,0,255,255,null]]}',
    '{"windows":[[0,"0",0,255,255,null]]}',
    '{"windows":[[0,300,0,255,255,null]]}',
    '{"windows":[[0,0,300,255,255,null]]}',
    '{"windows":[[0,0,0,255,255,null ]]}',
    '{"windows":[ [0,0,0,255,255,null]]}',
    '{"windows":[[0,0,0,255,255,null],]}',
])
def test_decode_bad_text_raises_value_error(bad_text):
    with pytest.raises(ValueError):
        decode_delta_summary(bad_text)


def test_decode_rejects_negative_zero_spelling():
    with pytest.raises(ValueError):
        decode_delta_summary(
            '{"windows":[[0,0,0,255,255,[-0.000000,2.000000,4,2]]]}')
    # The same value with normalized spelling decodes fine.
    result = decode_delta_summary(
        '{"windows":[[0,0,0,255,255,[0.000000,2.000000,4,2]]]}')
    assert result[0][5][0] == 0.0


def test_decode_does_not_canonicalize_silently():
    # Whitespace and alternate float spellings must be rejected outright.
    with pytest.raises(ValueError):
        decode_delta_summary('{"windows": [[0,0,0,255,255,null]]}')
    with pytest.raises(ValueError):
        decode_delta_summary(
            '{"windows":[[0,0,0,255,255,[1,2.000000,4,2]]]}')
