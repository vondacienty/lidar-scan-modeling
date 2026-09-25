"""Tests for :func:`encode_summary` and :func:`decode_summary`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (build_tile_pyramid_stats, decode_summary,
                        encode_summary, summarize_tile_pyramid_windows)
from lidar_scan import tiles as tiles_module


def _window(level=0, ix_min=0, iy_min=0, ix_max=255, iy_max=255, summary=None):
    return (level, ix_min, iy_min, ix_max, iy_max, summary)


def _summary(bias_min=-1.0, bias_max=2.0, abs_error_mean=0.5,
             z_score_rms=0.25, count_delta=3):
    return (bias_min, bias_max, abs_error_mean, z_score_rms, count_delta)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.encode_summary is encode_summary
    assert tiles_module.decode_summary is decode_summary
    import lidar_scan
    assert "encode_summary" in lidar_scan.__all__
    assert "decode_summary" in lidar_scan.__all__
    assert lidar_scan.encode_summary is encode_summary
    assert lidar_scan.decode_summary is decode_summary


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_empty_result():
    assert encode_summary(()) == '{"windows":[]}'


def test_encode_null_summary():
    assert encode_summary((_window(),)) == '{"windows":[[0,0,0,255,255,null]]}'


def test_encode_single_window_canonical_text():
    text = encode_summary((_window(summary=_summary()),))
    assert text == (
        '{"windows":[[0,0,0,255,255,'
        '[-1.000000,2.000000,0.500000,0.250000,3]]]}'
    )


def test_encode_keeps_window_order():
    result = (
        _window(level=1, ix_min=-512, iy_min=-512, ix_max=767, iy_max=1023,
                summary=_summary(bias_min=-3.25, bias_max=9.0,
                                 abs_error_mean=1.5, z_score_rms=2.0,
                                 count_delta=-42)),
        _window(level=0, ix_min=0, iy_min=0, ix_max=0, iy_max=0),
    )
    text = encode_summary(result)
    assert text == (
        '{"windows":['
        '[1,-512,-512,767,1023,[-3.250000,9.000000,1.500000,2.000000,-42]],'
        '[0,0,0,0,0,null]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_summary((_window(summary=_summary()),))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_summary(
        (_window(summary=_summary(bias_min=-1 / 7, bias_max=1 / 3,
                                  abs_error_mean=1 / 9, z_score_rms=2 / 3)),))
    assert "-0.142857" in text and "0.333333" in text
    assert "0.111111" in text and "0.666667" in text
    assert list(json.loads(text)) == ["windows"]


def test_encode_negative_zero_written_positive():
    text = encode_summary(
        (_window(summary=_summary(
            bias_min=-0.0, bias_max=math.copysign(0.0, -1.0),
            abs_error_mean=math.copysign(0.0, -1.0),
            z_score_rms=-0.0)),))
    assert "-0.000000" not in text
    assert "[0.000000,0.000000,0.000000,0.000000,3]" in text


def test_encode_integers_are_decimal_and_exact():
    result = (_window(level=10 ** 20, ix_min=-10 ** 40, iy_min=-10 ** 40,
                      ix_max=10 ** 40, iy_max=10 ** 40,
                      summary=_summary(count_delta=-(10 ** 80))),)
    text = encode_summary(result)
    assert str(10 ** 20) in text and str(-10 ** 40) in text
    assert str(-(10 ** 80)) in text
    assert json.loads(text)["windows"][0][0] == 10 ** 20


def test_encode_top_level_has_only_windows_key():
    text = encode_summary((_window(),))
    assert list(json.loads(text).keys()) == ["windows"]


def test_encode_does_not_emit_non_finite_tokens():
    text = encode_summary((_window(summary=_summary()),))
    assert "NaN" not in text and "Infinity" not in text


def test_encode_accepts_negative_zero_as_non_negative():
    result = (_window(summary=_summary(
        bias_min=math.copysign(0.0, -1.0),
        abs_error_mean=math.copysign(0.0, -1.0),
        z_score_rms=math.copysign(0.0, -1.0))),)
    assert encode_summary(result).startswith('{"windows":')


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
        encode_summary(bad)


@pytest.mark.parametrize("bad_result", [
    (_window()[:5],),                                    # 5-tuple window
    (_window() + (1,),),                                 # 7-tuple window
    ([0, 0, 0, 255, 255, None],),                        # window as list
    (_window(level=1.0),),
    (_window(level=True),),
    (_window(level=False),),
    (_window(level=-1),),                                # negative level
    (_window(ix_min=False),),
    (_window(iy_min="0"),),
    (_window(ix_max=255.0),),
    (_window(iy_max=True),),
    (_window(ix_min=300, ix_max=255),),                  # inverted ix
    (_window(iy_min=300, iy_max=255),),                  # inverted iy
    (_window(summary=[]),),
    (_window(summary=list(_summary())),),                # summary as list
    (_window(summary=_summary()[:4]),),                  # 4-tuple summary
    (_window(summary=_summary() + (0,)),),               # 6-tuple summary
    (_window(summary=(0, 2.0, 0.5, 0.25, 3)),),          # int bias_min
    (_window(summary=(True, 2.0, 0.5, 0.25, 3)),),       # bool bias_min
    (_window(summary=(-1.0, 2, 0.5, 0.25, 3)),),         # int bias_max
    (_window(summary=(-1.0, 2.0, 1, 0.25, 3)),),         # int abs mean
    (_window(summary=(-1.0, 2.0, 0.5, 1, 3)),),          # int z rms
    (_window(summary=(-1.0, 2.0 + 0j, 0.5, 0.25, 3)),),  # complex
    (_window(summary=(float("nan"), 2.0, 0.5, 0.25, 3)),),
    (_window(summary=(-1.0, float("inf"), 0.5, 0.25, 3)),),
    (_window(summary=(-1.0, float("-inf"), 0.5, 0.25, 3)),),
    (_window(summary=(-1.0, 2.0, float("nan"), 0.25, 3)),),
    (_window(summary=(-1.0, 2.0, 0.5, float("inf"), 3)),),
    (_window(summary=(2.0, 1.0, 0.5, 0.25, 3)),),        # bias_min > max
    (_window(summary=(-1.0, 2.0, -0.25, 0.25, 3)),),     # negative abs mean
    (_window(summary=(-1.0, 2.0, 0.5, -0.5, 3)),),       # negative z rms
    (_window(summary=(-1.0, 2.0, -1e-12, 0.25, 3)),),    # below-zero (not -0)
    (_window(summary=(-1.0, 2.0, 0.5, 0.25, 3.0)),),     # float count_delta
    (_window(summary=(-1.0, 2.0, 0.5, 0.25, True)),),    # bool count_delta
    (_window(summary="x"),),
])
def test_encode_bad_structure_raises_value_error(bad_result):
    with pytest.raises(ValueError):
        encode_summary(bad_result)


def test_encode_does_not_modify_input():
    result = (_window(summary=_summary()),)
    snapshot = (_window(summary=_summary()),)
    encode_summary(result)
    assert result == snapshot


def test_encode_repeated_calls_are_identical():
    result = (_window(summary=_summary()),)
    assert encode_summary(result) == encode_summary(result)


def test_encode_round_trips_real_summary_output():
    points_e = [(0.0, 0.0, 1.0, 1.0, 0.5), (10.0, 10.0, 2.0, 1.0, 0.5),
                (300.0, 300.0, 5.0, 1.0, 0.5)]
    points_r = [(0.0, 0.0, 1.5, 1.0, 0.5), (10.0, 10.0, 1.5, 1.0, 0.5),
                (300.0, 300.0, 4.5, 1.0, 0.5)]
    est = build_tile_pyramid_stats(points_e, levels=2)
    ref = build_tile_pyramid_stats(points_r, levels=2)
    windows = ((0, 0, 0, 255, 255), (1, 0, 0, 1000, 1000))
    result = summarize_tile_pyramid_windows(est, ref, windows)
    text = encode_summary(result)
    assert text.startswith('{"windows":')
    assert decode_summary(text) == result


# ---------------------------------------------------------------------------
# decoding / round trips
# ---------------------------------------------------------------------------

def test_decode_empty_document():
    assert decode_summary('{"windows":[]}') == ()


def test_decode_canonical_text_and_key_order():
    text = (
        '{"windows":[[1,-512,-512,767,1023,'
        '[-3.250000,9.000000,1.500000,2.000000,-42]],'
        '[0,0,0,0,0,null]]}'
    )
    result = decode_summary(text)
    assert result == (
        (1, -512, -512, 767, 1023, (-3.25, 9.0, 1.5, 2.0, -42)),
        (0, 0, 0, 0, 0, None),
    )
    assert list(json.loads(text)) == ["windows"]


def test_decode_returns_tuples_preserving_order():
    result = (
        _window(level=1, ix_min=-5, iy_min=-7, ix_max=300, iy_max=400,
                summary=_summary()),
        _window(level=0),
    )
    decoded = decode_summary(encode_summary(result))
    assert isinstance(decoded, tuple) and len(decoded) == 2
    assert all(isinstance(window, tuple) and len(window) == 6
               for window in decoded)
    summary = decoded[0][5]
    assert isinstance(summary, tuple) and len(summary) == 5
    assert decoded[1][5] is None
    for value in decoded[0][:5]:
        assert isinstance(value, int) and not isinstance(value, bool)
    for value in summary[:4]:
        assert isinstance(value, float) and math.isfinite(value)
    assert isinstance(summary[4], int) and not isinstance(summary[4], bool)


def test_decode_round_trips_against_encoder():
    result = (
        _window(level=2, ix_min=-(10 ** 20), iy_min=5, ix_max=10 ** 20,
                iy_max=9, summary=_summary(
                    bias_min=-0.125, bias_max=0.5,
                    abs_error_mean=0.0, z_score_rms=2.0,
                    count_delta=-(10 ** 60))),
        _window(level=0, summary=None),
    )
    text = encode_summary(result)
    assert decode_summary(text) == result
    assert encode_summary(decode_summary(text)) == text


def test_decode_six_decimals_and_positive_zero():
    text = (
        '{"windows":[[0,0,0,255,255,'
        '[-0.142857,0.000000,0.000000,0.000000,0]]]}'
    )
    summary = decode_summary(text)[0][5]
    assert summary[0] == pytest.approx(-1 / 7, abs=1e-6)
    for value in summary[1:4]:
        assert value == 0.0 and math.copysign(1.0, value) == 1.0


def test_decode_negative_zero_spelling_raises_value_error():
    text = (
        '{"windows":[[0,0,0,255,255,'
        '[-0.142857,-0.000000,0.500000,0.250000,3]]]}'
    )
    with pytest.raises(ValueError):
        decode_summary(text)


def test_decode_does_not_modify_input():
    text = '{"windows":[]}'
    decode_summary(text)
    assert text == '{"windows":[]}'


@pytest.mark.parametrize("bad", [None, 1, 1.5, b"...", True, False, [], {}])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_summary(bad)


# ---------------------------------------------------------------------------
# malformed documents
# ---------------------------------------------------------------------------

_VALID_SUMMARY = "[-1.000000,2.000000,0.500000,0.250000,3]"


def _doc(window_body=None):
    if window_body is None:
        window_body = f"0,0,0,255,255,{_VALID_SUMMARY}"
    return '{"windows":[[' + window_body + "]]}"


@pytest.mark.parametrize("bad", ["", "{", "[]", "null", "1", '"x"',
                                 '{"a":1}', "{...}"])
def test_decode_bad_json_syntax_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode_summary(bad)


@pytest.mark.parametrize("bad", [
    '{"levels":[]}',
    '{"windows":[],"x":1}',
    '{"x":1,"windows":[]}',
    '{"windows":[],"windows":[]}',
])
def test_decode_bad_top_level_keys_raise_value_error(bad):
    with pytest.raises(ValueError):
        decode_summary(bad)


def test_decode_windows_not_array_raises_value_error():
    with pytest.raises(ValueError):
        decode_summary('{"windows":{}}')


@pytest.mark.parametrize("body", [
    "0",
    "{}",
    "[]",
    "0,0,0,255",                                  # 4 values
    "0,0,0,255,255",                              # 5 values
    "0,0,0,255,255,null,1",                       # 7 values
    "0.0,0,0,255,255,null",
    "true,0,0,255,255,null",
    '"0",0,0,255,255,null',
    "-1,0,0,255,255,null",
    "0,300,0,255,255,null",
    "0,0,300,255,255,null",
    "0,0,0,255,255,0",                            # int summary
    "0,0,0,255,255,{}",
    "0,0,0,255,255,[]",                           # 0-value summary
    "0,0,0,255,255,[-1.000000,2.000000,0.500000,0.250000]",   # 4
    "0,0,0,255,255,[-1.000000,2.000000,0.500000,0.250000,3,4]",  # 6
    "0,0,0,255,255,[-1,2.000000,0.500000,0.250000,3]",
    "0,0,0,255,255,[-1.000000,true,0.500000,0.250000,3]",
    "0,0,0,255,255,[-1.000000,2.000000,null,0.250000,3]",
    "0,0,0,255,255,[NaN,2.000000,0.500000,0.250000,3]",
    "0,0,0,255,255,[-1.000000,Infinity,0.500000,0.250000,3]",
    "0,0,0,255,255,[2.000000,1.000000,0.500000,0.250000,3]",
    "0,0,0,255,255,[-1.000000,2.000000,-0.000001,0.250000,3]",
    "0,0,0,255,255,[-1.000000,2.000000,0.500000,-0.500000,3]",
    "0,0,0,255,255,[-1.000000,2.000000,0.500000,0.250000,true]",
    "0,0,0,255,255,[-1.000000,2.000000,0.500000,0.250000,3.0]",
])
def test_decode_bad_window_raises_value_error(body):
    with pytest.raises(ValueError):
        decode_summary(_doc(body))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_decode_non_finite_literals_raise_value_error(literal):
    with pytest.raises(ValueError):
        decode_summary(
            f'{{"windows":[[0,0,0,1,1,[{literal},2.0,0.5,0.25,3]]]}}')


@pytest.mark.parametrize("bad", [
    ' {"windows":[]}',
    '{"windows":[]} ',
    '{"windows" :[]}',
    '{"windows": [ ]}',
    '{"windows": [[0,0,0,1,1,null]]}',
])
def test_decode_whitespace_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode_summary(bad)


@pytest.mark.parametrize("summary", [
    "[-1,2.000000,0.500000,0.250000,3]",
    "[-1.0,2.000000,0.500000,0.250000,3]",
    "[-1.00000,2.000000,0.500000,0.250000,3]",
    "[-1.0000000,2.000000,0.500000,0.250000,3]",
    "[-0.000000,2.000000,0.500000,0.250000,3]",
    "[+1.000000,2.000000,0.500000,0.250000,3]",
    "[-1.000000e0,2.000000,0.500000,0.250000,3]",
])
def test_decode_non_canonical_numbers_raise_value_error(summary):
    with pytest.raises(ValueError):
        decode_summary(f'{{"windows":[[0,0,0,1,1,{summary}]]}}')


@pytest.mark.parametrize("bad", [
    '{"windows":[[00,0,0,1,1,null]]}',
    '{"windows":[[+0,0,0,1,1,null]]}',
    '{"windows":[[0,-00,0,1,1,null]]}',
])
def test_decode_non_canonical_integer_spelling_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode_summary(bad)


def test_decode_duplicate_json_keys_raise_value_error():
    with pytest.raises(ValueError):
        decode_summary('{"windows":[],"windows":[]}')
