"""Tests for :func:`lidar_scan.tiles.encode_tile_pyramid` and
:func:`lidar_scan.tiles.decode_tile_pyramid`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import (build_tile_pyramid, decode_tile_pyramid,
                        encode_tile_pyramid)
from lidar_scan import tiles


def _tile(tx=0, ty=0, zmin=1.0, zmax=2.0, count=1):
    return (tx, ty, tx * 256, ty * 256, tx * 256 + 255, ty * 256 + 255,
            zmin, zmax, count)


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def test_encode_empty_pyramid():
    assert encode_tile_pyramid(()) == '{"levels":[]}'


def test_encode_empty_levels():
    assert encode_tile_pyramid(((), ())) == '{"levels":[[],[]]}'


def test_encode_canonical_text():
    pyramid = build_tile_pyramid(
        [(0.1, 0.2, 1.0, 1.0, 1.0),
         (300.0, -5.0, -2.5, 1.0, 1.0)],
        tile_cells=256, levels=2)
    text = encode_tile_pyramid(pyramid)
    assert text == (
        '{"levels":[['
        '[0,0,0,0,255,255,1.000000,1.000000,1],'
        '[1,-1,256,-256,511,-1,-2.500000,-2.500000,1]'
        '],['
        '[0,-1,0,-512,511,-1,-2.500000,-2.500000,1],'
        '[0,0,0,0,511,511,1.000000,1.000000,1]'
        ']]}'
    )
    # compact: no whitespace
    assert " " not in text
    assert "\t" not in text and "\n" not in text


def test_encode_fixed_six_decimals():
    text = encode_tile_pyramid(((_tile(zmin=1 / 3, zmax=-1 / 7),),))
    assert "0.333333" in text
    assert "-0.142857" in text


def test_encode_negative_zero_written_positive():
    text = encode_tile_pyramid(((_tile(zmin=-0.0, zmax=-0.0),),))
    assert "-0.000000" not in text
    assert "0.000000" in text


def test_encode_returns_str():
    assert isinstance(encode_tile_pyramid(()), str)


def test_encode_tile_order_preserved_within_level():
    pyramid = (
        (_tile(tx=0, ty=0), _tile(tx=1, ty=0), _tile(tx=1, ty=2)),
    )
    text = encode_tile_pyramid(pyramid)
    assert text.index("[0,0,") < text.index("[1,0,") < text.index("[1,2,")


@pytest.mark.parametrize("bad", [
    [],
    {},
    None,
    "((),)",
    123,
])
def test_encode_outer_non_tuple_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_pyramid(bad)


def test_encode_level_must_be_tuple():
    with pytest.raises(ValueError):
        encode_tile_pyramid(([_tile()],))


def test_encode_tile_must_be_tuple():
    with pytest.raises(ValueError):
        encode_tile_pyramid(([list(_tile())],))


def test_encode_tile_must_have_nine_items():
    good = _tile()
    with pytest.raises(ValueError):
        encode_tile_pyramid((good[:8],))
    with pytest.raises(ValueError):
        encode_tile_pyramid((good + (1,),))


@pytest.mark.parametrize("position", range(6))
def test_encode_int_fields_reject_non_ints(position):
    tile = list(_tile())
    tile[position] = 1.0
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile),),))


@pytest.mark.parametrize("position", range(6))
def test_encode_int_fields_reject_bools(position):
    tile = list(_tile())
    tile[position] = True
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile),),))


def test_encode_count_must_be_non_bool_int():
    tile_bad = list(_tile())
    tile_bad[8] = 1.0
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile_bad),),))
    tile_bool = list(_tile())
    tile_bool[8] = False
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile_bool),),))


@pytest.mark.parametrize("position", (6, 7))
@pytest.mark.parametrize("bad", [1, 0, -2, True, "1.0", None])
def test_encode_z_bounds_must_be_finite_float(position, bad):
    tile = list(_tile())
    tile[position] = bad
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile),),))


@pytest.mark.parametrize("position", (6, 7))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_encode_z_bounds_reject_nonfinite(position, bad):
    tile = list(_tile())
    tile[position] = bad
    with pytest.raises(ValueError):
        encode_tile_pyramid(((tuple(tile),),))


def test_encode_levels_must_be_sorted():
    with pytest.raises(ValueError):
        encode_tile_pyramid(((_tile(tx=1, ty=0), _tile(tx=0, ty=0)),))


def test_encode_rejects_equal_keys_via_ty():
    with pytest.raises(ValueError):
        encode_tile_pyramid(((_tile(tx=1, ty=1), _tile(tx=1, ty=0)),))


def test_encode_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        encode_tile_pyramid(((_tile(tx=1, ty=1), _tile(tx=1, ty=1)),))


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def test_decode_round_trip_built_pyramid():
    points = [(0.1, 0.2, 1 / 3, 1.0, 1.0),
              (300.0, -5.0, -0.0, 1.0, 1.0),
              (700.0, 700.0, -2.5, 1.0, 1.0)]
    pyramid = build_tile_pyramid(points, tile_cells=256, levels=3)
    text = encode_tile_pyramid(pyramid)
    result = decode_tile_pyramid(text)
    assert result == pyramid
    assert isinstance(result, tuple)
    for level in result:
        assert isinstance(level, tuple)
        for tile in level:
            assert isinstance(tile, tuple)
            assert len(tile) == 9


def test_decode_empty_pyramid():
    assert decode_tile_pyramid('{"levels":[]}') == ()


def test_decode_empty_levels():
    assert decode_tile_pyramid('{"levels":[[],[]]}') == ((), ())


def test_decode_preserves_order():
    pyramid = (
        (_tile(tx=0, ty=0), _tile(tx=1, ty=0), _tile(tx=1, ty=2)),
        (_tile(tx=5, ty=-3),),
    )
    assert decode_tile_pyramid(encode_tile_pyramid(pyramid)) == pyramid


@pytest.mark.parametrize("bad", [
    b'{"levels":[]}',
    123,
    None,
    bytearray(b'{"levels":[]}'),
    1.0,
])
def test_decode_non_str_type_error(bad):
    with pytest.raises(TypeError):
        decode_tile_pyramid(bad)


@pytest.mark.parametrize("text", [
    "",
    "{",
    "not json",
    '{"levels":[',
    '{"levels":[[[1,0,0,0,255,255,1.000000,2.000000,1]',
])
def test_decode_syntax_errors(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


@pytest.mark.parametrize("text", [
    "{}",
    "[]",
    "5",
    "null",
    '"levels"',
    '{"levels":[],"x":1}',
    '{"x":[]}',
    '{"LEVELS":[]}',
])
def test_decode_bad_top_level(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


def test_decode_duplicate_top_level_key_rejected():
    with pytest.raises(ValueError):
        decode_tile_pyramid('{"levels":[],"levels":[]}')


@pytest.mark.parametrize("text", [
    '{"levels":{}}',
    '{"levels":5}',
    '{"levels":null}',
])
def test_decode_levels_must_be_array(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


@pytest.mark.parametrize("text", [
    '{"levels":[{}]}',
    '{"levels":[5]}',
    '{"levels":[null]}',
])
def test_decode_level_must_be_array(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


def test_decode_tile_must_be_array():
    with pytest.raises(ValueError):
        decode_tile_pyramid('{"levels":[[5]]}')


@pytest.mark.parametrize("text", [
    '{"levels":[[[1,0,0,0,255,255,1.000000,2.000000]]]}',
    '{"levels":[[[1,0,0,0,255,255,1.000000,2.000000,1,2]]]}',
    '{"levels":[[[1,0,0,0,255,255,1.000000,2.000000,null]]]}',
])
def test_decode_tile_shape_and_field_types(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


@pytest.mark.parametrize("text", [
    '{"levels":[[[1.0,0,0,0,255,255,1.000000,2.000000,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,1.000000,2.000000,true]]]}',
    '{"levels":[[[false,0,0,0,255,255,1.000000,2.000000,1]]]}',
])
def test_decode_numeric_field_kind(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


@pytest.mark.parametrize("text", [
    '{"levels":[[[1,0,0,0,255,255,1.0,2.000000,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,1.0000000,2.000000,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,1,2,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,1.00000e0,2.000000,1]]]}',
])
def test_decode_z_bounds_need_six_decimals(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


def test_decode_negative_zero_rejected():
    with pytest.raises(ValueError):
        decode_tile_pyramid(
            '{"levels":[[[1,0,0,0,255,255,-0.000000,2.000000,1]]]}')


@pytest.mark.parametrize("text", [
    '{"levels":[[[1,0,0,0,255,255,NaN,2.000000,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,Infinity,2.000000,1]]]}',
    '{"levels":[[[1,0,0,0,255,255,-Infinity,2.000000,1]]]}',
])
def test_decode_rejects_nonstandard_constants(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


def test_decode_unsorted_level_rejected():
    with pytest.raises(ValueError):
        decode_tile_pyramid(
            '{"levels":[[[1,0,256,0,511,255,1.000000,2.000000,1],'
            '[0,0,0,0,255,255,1.000000,2.000000,1]]]}')


def test_decode_duplicate_tile_key_rejected():
    with pytest.raises(ValueError):
        decode_tile_pyramid(
            '{"levels":[[[1,0,256,0,511,255,1.000000,2.000000,1],'
            '[1,0,256,0,511,255,3.000000,4.000000,1]]]}')


@pytest.mark.parametrize("text", [
    '{"levels": [[ [1,0,0,0,255,255,1.000000,2.000000,1] ]]}',
    '{ "levels":[]}',
    '{"levels" :[]}',
    '{"levels":[]}\n',
    '{"levels":[]} ',
])
def test_decode_requires_canonical_compact_text(text):
    with pytest.raises(ValueError):
        decode_tile_pyramid(text)


def test_decode_then_reencode_is_identity_text():
    pyramid = build_tile_pyramid(
        [(12.5, -8.25, 4.0, 9.0, 0.25), (900.0, 900.0, -7.125, 1.0, 1.0)],
        levels=2)
    text = encode_tile_pyramid(pyramid)
    assert encode_tile_pyramid(decode_tile_pyramid(text)) == text


def test_module_exports_match_root():
    assert tiles.encode_tile_pyramid is encode_tile_pyramid
    assert tiles.decode_tile_pyramid is decode_tile_pyramid
    import lidar_scan
    assert "encode_tile_pyramid" in lidar_scan.__all__
    assert "decode_tile_pyramid" in lidar_scan.__all__


def test_decoded_values_have_expected_types():
    text = '{"levels":[[[0,-1,0,-256,255,-1,-2.500000,3.250000,7]]]}'
    result = decode_tile_pyramid(text)
    tile = result[0][0]
    assert tile[:6] == (0, -1, 0, -256, 255, -1)
    assert all(isinstance(v, int) and not isinstance(v, bool)
               for v in tile[:6])
    assert tile[6] == -2.5 and isinstance(tile[6], float)
    assert tile[7] == 3.25 and isinstance(tile[7], float)
    assert tile[8] == 7 and isinstance(tile[8], int)
    assert math.copysign(1.0, tile[6]) == -1.0
