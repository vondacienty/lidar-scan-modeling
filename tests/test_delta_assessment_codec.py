"""Tests for :func:`encode_delta_assessment` and :func:`decode_delta_assessment`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (assess_delta_summary, decode_delta_assessment,
                        encode_delta_assessment)
from lidar_scan import tiles as tiles_module


def _window(level=0, ix_min=0, iy_min=0, ix_max=255, iy_max=255, delta=None):
    return (level, ix_min, iy_min, ix_max, iy_max, delta)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_functions_exported_from_module_and_package():
    assert tiles_module.encode_delta_assessment is encode_delta_assessment
    assert tiles_module.decode_delta_assessment is decode_delta_assessment
    import lidar_scan
    assert "encode_delta_assessment" in lidar_scan.__all__
    assert "decode_delta_assessment" in lidar_scan.__all__
    assert lidar_scan.encode_delta_assessment is encode_delta_assessment
    assert lidar_scan.decode_delta_assessment is decode_delta_assessment


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_empty_assessment():
    assert encode_delta_assessment(()) == '{"windows":[]}'


def test_encode_window_with_null_delta():
    text = encode_delta_assessment((_window(delta=None),))
    assert text == '{"windows":[[0,0,0,255,255,null]]}'


def test_encode_window_with_int_deltas():
    text = encode_delta_assessment((_window(delta=(1, -2, 3, -4)),))
    assert text == '{"windows":[[0,0,0,255,255,[1,-2,3,-4]]]}'


def test_encode_window_with_float_deltas():
    text = encode_delta_assessment((_window(delta=(1.5, -2.25, 3, -4)),))
    assert text == '{"windows":[[0,0,0,255,255,[1.500000,-2.250000,3,-4]]]}'


def test_encode_mixed_int_and_float_deltas():
    text = encode_delta_assessment((_window(delta=(1, -2.5, 0, 0)),))
    assert text == '{"windows":[[0,0,0,255,255,[1,-2.500000,0,0]]]}'


def test_encode_from_assess_delta_summary_round_trips():
    estimate = (
        (0, 0, 0, 255, 255, (1.5, 9.25, 40, 4)),
        (0, 256, 0, 511, 255, None),
        (1, 0, 0, 511, 511, (-2, 7, 3, 1)),
    )
    reference = (
        (0, 0, 0, 255, 255, (0.5, 10.0, 42, 6)),
        (0, 256, 0, 511, 255, None),
        (1, 0, 0, 511, 511, (-5, 7, 1, 2)),
    )
    assessment = assess_delta_summary(estimate, reference)
    text = encode_delta_assessment(assessment)
    assert text == (
        '{"windows":['
        '[0,0,0,255,255,[1.000000,-0.750000,-2,-2]],'
        '[0,256,0,511,255,null],'
        '[1,0,0,511,511,[3,0,2,-1]]]}'
    )
    assert decode_delta_assessment(text) == assessment


def test_encode_keeps_window_order():
    assessment = (
        _window(level=1, ix_min=-512, iy_min=-512, ix_max=767, iy_max=1023,
                delta=(-3.25, 9.0, 42, 7)),
        _window(level=2, ix_min=0, iy_min=0, ix_max=0, iy_max=0, delta=None),
    )
    text = encode_delta_assessment(assessment)
    assert text == (
        '{"windows":['
        '[1,-512,-512,767,1023,[-3.250000,9.000000,42,7]],'
        '[2,0,0,0,0,null]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_delta_assessment((_window(delta=(1.0, 2.0, 3, 4)),))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_delta_assessment((_window(delta=(-1 / 7, 1 / 3, 0, 0)),))
    assert "-0.142857" in text and "0.333333" in text
    assert list(json.loads(text)) == ["windows"]


def test_encode_negative_zero_written_positive():
    text = encode_delta_assessment((_window(delta=(-0.0, 0.0, 0, 0)),))
    assert "-0.000000" not in text
    assert "0.000000,0.000000" in text


def test_encode_integers_are_decimal_and_exact():
    assessment = (_window(level=10 ** 20, ix_min=-10 ** 40, iy_min=-10 ** 40,
                          ix_max=10 ** 40, iy_max=10 ** 40,
                          delta=(10 ** 80, -10 ** 80, 10 ** 60, -10 ** 60)),)
    text = encode_delta_assessment(assessment)
    assert str(10 ** 80) in text and str(-10 ** 80) in text
    assert json.loads(text)["windows"][0][0] == 10 ** 20


def test_encode_top_level_has_only_windows_key():
    text = encode_delta_assessment((_window(),))
    assert list(json.loads(text).keys()) == ["windows"]


def test_encode_does_not_emit_non_finite_tokens():
    text = encode_delta_assessment((_window(delta=(1.0, 2.0, 3, 4)),))
    assert "NaN" not in text and "Infinity" not in text


def test_encode_allows_reversed_delta_extrema():
    # Delta values may appear in either order (no min <= max constraint).
    text = encode_delta_assessment((_window(delta=(5.0, -5.0, 0, 0)),))
    assert "[5.000000,-5.000000,0,0]" in text


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
        encode_delta_assessment(bad)


@pytest.mark.parametrize("bad_assessment", [
    (_window()[:5],),                                   # 5-tuple window
    (_window() + (1,),),                                # 7-tuple window
    ([0, 0, 0, 255, 255, None],),                       # window as list
    (_window(level=1.0),),
    (_window(level=True),),
    (_window(level=-1),),                               # negative level
    (_window(ix_min=False),),
    (_window(iy_min="0"),),
    (_window(ix_max=255.0),),
    (_window(ix_min=300, ix_max=255),),                 # inverted ix bounds
    (_window(iy_min=300, iy_max=255),),                 # inverted iy bounds
    (_window(delta=(1, 2, 3)),),                        # 3-field delta
    (_window(delta=(1, 2, 3, 4, 5)),),                  # 5-field delta
    (_window(delta=[1, 2, 3, 4]),),                     # delta as list
    (_window(delta=1),),                                # non-tuple delta
    (_window(delta=(True, 2, 3, 4)),),
    (_window(delta=(1, False, 3, 4)),),
    (_window(delta=("1", 2, 3, 4)),),
    (_window(delta=(1.0 + 0j, 2, 3, 4)),),
    (_window(delta=(float("nan"), 2, 3, 4)),),
    (_window(delta=(1, float("inf"), 3, 4)),),
    (_window(delta=(1, float("-inf"), 3, 4)),),
    (_window(delta=(1, 2, 3.0, 4)),),                   # dsum_dcount float
    (_window(delta=(1, 2, 3, 4.0)),),                   # dmatch_count float
    (_window(delta=(1, 2, True, 4)),),
    (_window(delta=(1, 2, 3, False)),),
    (_window(level=0), _window(level=0)),               # duplicate keys
    (_window(level=1), _window(level=0)),               # unsorted keys
    (_window(ix_min=0, ix_max=255),
     _window(ix_min=0, ix_max=255)),                    # duplicate keys
    (_window(ix_min=0, ix_max=255),
     _window(ix_min=0, ix_max=254)),                    # unsorted keys
])
def test_encode_bad_structure_raises_value_error(bad_assessment):
    with pytest.raises(ValueError):
        encode_delta_assessment(bad_assessment)


def test_encode_does_not_modify_input():
    assessment = (_window(delta=(1.5, -2.25, 3, -4)), _window(level=1))
    snapshot = (_window(delta=(1.5, -2.25, 3, -4)), _window(level=1))
    encode_delta_assessment(assessment)
    assert assessment == snapshot


def test_encode_repeated_calls_are_identical():
    assessment = (_window(delta=(1 / 3, -2.25, 3, -4)), _window(level=1))
    once = encode_delta_assessment(assessment)
    assert encode_delta_assessment(assessment) == once


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def test_decode_empty_document():
    assert decode_delta_assessment('{"windows":[]}') == ()


def test_decode_null_delta():
    assert decode_delta_assessment(
        '{"windows":[[0,0,0,255,255,null]]}') == (_window(delta=None),)


def test_decode_int_and_float_deltas_keep_types():
    result = decode_delta_assessment(
        '{"windows":[[0,0,0,255,255,[1,-2.500000,3,-4]]]}')
    assert result == (_window(delta=(1, -2.5, 3, -4)),)
    delta = result[0][5]
    assert isinstance(delta[0], int) and not isinstance(delta[0], bool)
    assert isinstance(delta[1], float)
    assert isinstance(delta[2], int) and isinstance(delta[3], int)


def test_decode_encode_round_trip():
    assessment = (
        _window(level=0, ix_min=-10, iy_min=-410, ix_max=300, iy_max=0,
                delta=(1.5, -0.75, -2, 2)),
        _window(level=0, ix_min=0, iy_min=1, ix_max=300, iy_max=400,
                delta=None),
        _window(level=2, ix_min=0, iy_min=0, ix_max=1000, iy_max=1000,
                delta=(3, 0, 10 ** 40, -10 ** 40)),
    )
    text = encode_delta_assessment(assessment)
    assert decode_delta_assessment(text) == assessment


def test_decode_returns_tuples():
    result = decode_delta_assessment(
        '{"windows":[[0,0,0,255,255,[1.000000,2.000000,3,4]]]}')
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][5], tuple)


@pytest.mark.parametrize("bad", [
    ["windows"],
    {"windows": []},
    1,
    1.5,
    True,
    False,
    None,
    b'{"windows":[]}',
])
def test_decode_non_str_raises_type_error(bad):
    with pytest.raises(TypeError):
        decode_delta_assessment(bad)


@pytest.mark.parametrize("bad_text", [
    "",
    "not json",
    "{",
    '{"windows":[]',
    "[1,2,3]",
    "null",
    "42",
    '"text"',
    "true",
    "{}",                                        # missing windows key
    '{"windows":{}',                             # malformed
    '{"windows":[],"extra":1}',                  # extra key
    '{"extra":1,"windows":[]}',                  # extra key before
    '{"Windows":[]}',                            # wrong key
    '{"windows":{}}',                            # windows not an array
    '{"windows":null}',
    '{"windows":[null]}',
    '{"windows":[[0,0,0,255,255]]}',             # 5-value window
    '{"windows":[[0,0,0,255,255,null,1]]}',      # 7-value window
    '{"windows":[[0,0,0,255,255,[]]]}',          # empty delta array
    '{"windows":[[0,0,0,255,255,[1,2,3]]]}',     # 3-value delta
    '{"windows":[[0,0,0,255,255,[1,2,3,4,5]]]}',  # 5-value delta
    '{"windows":[[0,0,0,255,255,{}]]}',          # delta as object
    '{"windows":[[0,0,0,255,255,1]]}',           # delta as number
    '{"windows":[[0,0,0,255,255,"null"]]}',      # delta as string
    '{"windows":[[-1,0,0,255,255,null]]}',       # negative level
    '{"windows":[[1.0,0,0,255,255,null]]}',      # float level
    '{"windows":[[true,0,0,255,255,null]]}',     # bool level
    '{"windows":[[0,300,0,255,255,null]]}',        # inverted ix bounds
    '{"windows":[[0,0,1,255,0,null]]}',          # inverted iy bounds
    '{"windows":[[0,0,0,255,255,[null,2,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,null,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[true,2,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,false,3,4]]]}',
    '{"windows":[[0,0,0,255,255,["1",2,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[NaN,2,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,Infinity,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,-Infinity,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,2,3.0,4]]]}',   # float dsum_dcount
    '{"windows":[[0,0,0,255,255,[1,2,3,4.0]]]}',   # float dmatch_count
    '{"windows":[[0,0,0,255,255,[1,2,true,4]]]}',
    '{"windows":[[0,0,0,255,255,[1,2,3,false]]]}',
    '{"windows":[[0,0,0,255,255,null],[0,0,0,255,255,null]]}',  # dup keys
    '{"windows":[[1,0,0,255,255,null],[0,0,0,255,255,null]]}',  # unsorted
    '{"windows": [[0,0,0,255,255,null]]}',         # whitespace
    '{ "windows":[]}',                             # whitespace
    '{"windows":[] }',
    '{"windows":[[0,0,0,255,255,[1.5,2,3,4]]]}',   # non-six-decimal float
    '{"windows":[[0,0,0,255,255,[1.5000000,2,3,4]]]}',
    '{"windows":[[0,0,0,255,255,[-0.000000,2,3,4]]]}',  # negative zero
    '{"windows":[[0,0,0,255,255,[1e0,2,3,4]]]}',   # exponent
    '{"windows":[[0,0,0,255,255,[+1,2,3,4]]]}',    # leading plus
    '{"windows":[[01,0,0,255,255,null]]}',         # leading zero
    '{"windows":[[0,0,0,255,255,[1.,2,3,4]]]}',    # trailing dot
    '{"windows":[[0,0,0,255,255,[.5,2,3,4]]]}',    # leading dot
    '{"windows":[[0,0,0,255,255,],[0,0,0,255,255,null]]}',  # malformed
])
def test_decode_bad_document_raises_value_error(bad_text):
    with pytest.raises(ValueError):
        decode_delta_assessment(bad_text)


def test_decode_rejects_reordered_top_level_keys():
    with pytest.raises(ValueError):
        decode_delta_assessment('{"windows":[],"windows":[]}')


def test_decode_does_not_modify_input():
    text = '{"windows":[[0,0,0,255,255,[1.000000,-2.250000,3,-4]]]}'
    decode_delta_assessment(text)
    assert text == '{"windows":[[0,0,0,255,255,[1.000000,-2.250000,3,-4]]]}'


def test_decode_repeated_calls_are_identical():
    text = ('{"windows":[[0,0,0,255,255,[1.000000,-2.250000,3,-4]],'
            '[1,0,0,0,0,null]]}')
    once = decode_delta_assessment(text)
    assert decode_delta_assessment(text) == once
