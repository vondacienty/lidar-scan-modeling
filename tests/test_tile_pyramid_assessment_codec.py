"""Tests for :func:`encode_tile_pyramid_assessment`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (assess_tile_pyramid_stats,
                        build_tile_pyramid_stats,
                        encode_tile_pyramid_assessment)
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  bias=0.1, abs_error=0.1, combined_sigma=1.5,
                  z_score=0.066667, count_delta=0)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["bias"],
            fields["abs_error"], fields["combined_sigma"],
            fields["z_score"], fields["count_delta"])


def _assessment(*tiles):
    return (tuple(tiles),)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.encode_tile_pyramid_assessment
            is encode_tile_pyramid_assessment)
    import lidar_scan
    assert "encode_tile_pyramid_assessment" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def test_encode_single_tile_canonical_text():
    text = encode_tile_pyramid_assessment(_assessment(_tile()))
    assert text == (
        '{"levels":[[[0,0,0,0,255,255,0.100000,0.100000,1.500000,'
        '0.066667,0]]]}'
    )


def test_encode_empty_assessment():
    assert encode_tile_pyramid_assessment(()) == '{"levels":[]}'


def test_encode_empty_levels_preserved():
    assert encode_tile_pyramid_assessment(((), ())) == '{"levels":[[],[]]}'


def test_encode_from_real_assessment_round_trips_json():
    points_a = [(0.1, 0.2, 1.0, 1.0, 0.5), (300.0, -400.0, -2.5, 0.0, 1.0),
                (-0.1, 0.0, 0.0, 0.0, 2.0), (600.0, 600.0, 7.25, 0.0, 1.0)]
    points_b = [(0.4, 0.3, 1.2, 1.0, 0.75), (300.0, -400.0, -2.0, 0.0, 1.0),
                (-0.1, 0.0, 0.1, 0.0, 2.0), (600.0, 600.0, 7.0, 0.0, 1.0)]
    assessment = assess_tile_pyramid_stats(
        build_tile_pyramid_stats(points_a, levels=3),
        build_tile_pyramid_stats(points_b, levels=3))
    text = encode_tile_pyramid_assessment(assessment)

    document = json.loads(text)
    assert list(document) == ["levels"]
    assert len(document["levels"]) == 3
    for level, level_tiles in zip(document["levels"], assessment):
        assert len(level) == len(level_tiles)
        for encoded, tile in zip(level, level_tiles):
            assert encoded == list(tile)


def test_encode_keeps_level_and_tile_order():
    assessment = (
        (),
        (_tile(tx=-5, ty=-7, bias=-3.25, abs_error=3.25,
               combined_sigma=9.0, z_score=-0.361111, count_delta=42),
         _tile(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023,
               bias=0.0, abs_error=0.0, combined_sigma=1.5,
               z_score=1.5, count_delta=-7)),
    )
    text = encode_tile_pyramid_assessment(assessment)
    assert text == (
        '{"levels":[[],['
        '[-5,-7,0,0,255,255,-3.250000,3.250000,9.000000,-0.361111,42],'
        '[2,3,512,768,767,1023,0.000000,0.000000,1.500000,1.500000,-7]]]}'
    )


def test_encode_uses_compact_separators_without_whitespace():
    text = encode_tile_pyramid_assessment(_assessment(_tile()))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    text = encode_tile_pyramid_assessment(
        _assessment(_tile(bias=-1 / 7, abs_error=1 / 7,
                          combined_sigma=1 / 3, z_score=2 / 3)))
    assert "-0.142857" in text and "0.142857" in text
    assert "0.333333" in text and "0.666667" in text


def test_encode_negative_zero_written_positive():
    text = encode_tile_pyramid_assessment(
        _assessment(_tile(bias=-0.0, z_score=math.copysign(0.0, -1.0))))
    assert "-0.000000" not in text
    assert "0.000000,0.100000,1.500000,0.000000,0" in text


def test_encode_integers_are_decimal_and_exact():
    assessment = _assessment(_tile(tx=10 ** 40, ty=-10 ** 40,
                                  count_delta=10 ** 80))
    text = encode_tile_pyramid_assessment(assessment)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert str(10 ** 80) in text
    assert json.loads(text)["levels"][0][0][0] == 10 ** 40


def test_encode_top_level_has_only_levels_key():
    text = encode_tile_pyramid_assessment(_assessment(_tile()))
    assert list(json.loads(text).keys()) == ["levels"]


@pytest.mark.parametrize("bad", [
    [[_tile()]],
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
        encode_tile_pyramid_assessment(bad)


@pytest.mark.parametrize("bad_assessment", [
    ([_tile()],),                                       # level is a list
    (([_tile()],),),                                    # tile is a list
    _assessment((1, 2)),                                # 2-tuple tile
    _assessment(_tile()[:10]),                          # 10 fields
    _assessment(_tile() + (9,)),                        # 12 fields
    _assessment(123),                                   # non-tuple tile
    _assessment(_tile(tx=1.0)),
    _assessment(_tile(ty=True)),
    _assessment(_tile(ix0=False)),
    _assessment(_tile(iy1="255")),
    _assessment(_tile(count_delta=1.0)),
    _assessment(_tile(count_delta=True)),
    _assessment(_tile(bias=1)),
    _assessment(_tile(abs_error=2.0 + 0j)),
    _assessment(_tile(bias=float("nan"))),
    _assessment(_tile(abs_error=float("inf"))),
    _assessment(_tile(z_score=float("-inf"))),
    _assessment(_tile(combined_sigma=0.0)),
    _assessment(_tile(combined_sigma=-1.0)),
    _assessment(_tile(ix1=0, ix0=255)),                 # ix0 > ix1
    _assessment(_tile(iy0=255, iy1=0)),                 # iy0 > iy1
    _assessment(_tile(tx=0, ty=1), _tile(tx=0, ty=0)),  # unsorted
    _assessment(_tile(tx=1, ty=1), _tile(tx=1, ty=1)),  # duplicate
    ((_tile(),), [_tile(tx=1, ty=1)]),                  # later level is list
])
def test_encode_bad_structure_raises_value_error(bad_assessment):
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(bad_assessment)


def test_encode_does_not_modify_input():
    tile = _tile()
    assessment = _assessment(tile)
    encode_tile_pyramid_assessment(assessment)
    assert assessment == _assessment(tile)
