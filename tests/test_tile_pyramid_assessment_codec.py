"""Tests for :func:`encode_tile_pyramid_assessment`."""

from __future__ import annotations

import json
import math

import pytest

from lidar_scan import (assess_tile_pyramid_stats, build_tile_pyramid_stats,
                        encode_tile_pyramid_assessment)
from lidar_scan import tiles as tiles_module


def _entry(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  bias=0.5, abs_error=0.5, combined_sigma=1.0, z_score=0.5,
                  count_delta=0)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["bias"],
            fields["abs_error"], fields["combined_sigma"],
            fields["z_score"], fields["count_delta"])


def _assessment(*entries):
    return (tuple(entries),)


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

def test_encode_single_entry_canonical_text():
    text = encode_tile_pyramid_assessment(_assessment(_entry()))
    assert text == (
        '{"levels":[[[0,0,0,0,255,255,0.500000,0.500000,1.000000,0.500000,0]]]}'
    )


def test_encode_empty_assessment():
    assert encode_tile_pyramid_assessment(()) == '{"levels":[]}'


def test_encode_empty_levels_preserved():
    assert encode_tile_pyramid_assessment(((), ())) == '{"levels":[[],[]]}'


def test_encode_from_assess_output():
    points = [(0.1, 0.2, 1.0, 1, 0.5), (300.0, 5.0, 4.0, 0, 1.0),
              (600.0, 600.0, -2.0, 0, 2.0)]
    estimate = build_tile_pyramid_stats(points, levels=3)
    reference = build_tile_pyramid_stats(
        [(0.4, 0.9, 0.5, 0, 0.7), (300.0, 5.0, 5.0, 0, 1.0),
         (600.0, 600.0, -1.0, 0, 1.5)], levels=3)
    assessment = assess_tile_pyramid_stats(estimate, reference)
    text = encode_tile_pyramid_assessment(assessment)
    document = json.loads(text)
    assert list(document) == ["levels"]
    assert len(document["levels"]) == len(assessment)
    for level, level_tiles in zip(document["levels"], assessment):
        assert level == [list(tile) for tile in level_tiles]


def test_encode_keeps_level_and_tile_order():
    assessment = (
        (),
        (_entry(tx=-5, ty=-7, ix0=0, iy0=0, ix1=511, iy1=511,
                bias=-3.25, abs_error=3.25, combined_sigma=4.0,
                z_score=-0.8125, count_delta=42),
         _entry(tx=2, ty=3, ix0=512, iy0=768, ix1=767, iy1=1023,
                bias=0.0, abs_error=0.0, combined_sigma=1.5,
                z_score=0.0, count_delta=-7)),
    )
    text = encode_tile_pyramid_assessment(assessment)
    assert text == (
        '{"levels":[[],['
        '[-5,-7,0,0,511,511,-3.250000,3.250000,4.000000,-0.812500,42],'
        '[2,3,512,768,767,1023,0.000000,0.000000,1.500000,0.000000,-7]]]}'
    )


def test_encode_compact_separators_without_whitespace():
    text = encode_tile_pyramid_assessment(_assessment(_entry()))
    assert not any(ch in text for ch in " \t\n\r")
    assert "," in text and ":" in text


def test_encode_fixed_six_decimals():
    assessment = _assessment(
        _entry(bias=1 / 3, abs_error=1 / 7, combined_sigma=math.sqrt(2.0),
               z_score=-1 / 3))
    text = encode_tile_pyramid_assessment(assessment)
    assert "0.333333" in text and "0.142857" in text
    assert "1.414214" in text and "-0.333333" in text


def test_encode_negative_zero_written_positive():
    assessment = _assessment(_entry(bias=-0.0, z_score=math.copysign(0.0, -1.0)))
    text = encode_tile_pyramid_assessment(assessment)
    assert "-0.000000" not in text


def test_encode_integers_are_decimal_and_exact():
    assessment = _assessment(
        _entry(tx=10 ** 40, ty=-10 ** 40, ix0=10 ** 40, ix1=10 ** 40 + 255,
               count_delta=-10 ** 80))
    text = encode_tile_pyramid_assessment(assessment)
    assert str(10 ** 40) in text and str(-10 ** 40) in text
    assert str(-10 ** 80) in text


def test_encode_does_not_modify_input():
    import copy
    points = [(0.1, 0.2, 1.0, 1, 0.5), (300.0, 5.0, 4.0, 0, 1.0)]
    estimate = build_tile_pyramid_stats(points, levels=2)
    reference = build_tile_pyramid_stats(
        [(0.4, 0.9, 0.5, 0, 0.7), (300.0, 5.0, 5.0, 0, 1.0)], levels=2)
    assessment = assess_tile_pyramid_stats(estimate, reference)
    snapshot = copy.deepcopy(assessment)
    encode_tile_pyramid_assessment(assessment)
    assert assessment == snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x", {}])
def test_non_tuple_raises_type_error(bad):
    with pytest.raises(TypeError):
        encode_tile_pyramid_assessment(bad)


# ---------------------------------------------------------------------------
# ValueError
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(([_entry()],))


def test_entry_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment((([0, 0, 0, 0, 255, 255,
                                          1.0, 1.0, 1.0, 1.0, 0],),))


@pytest.mark.parametrize("length", [10, 12])
def test_wrong_field_count_raises_value_error(length):
    tile = (0,) * length
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tile,),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_non_int_fields_raise_value_error(pos):
    tile = list(_entry())
    tile[pos] = 1.5
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tuple(tile),),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4, 5, 10])
def test_bool_fields_raise_value_error(pos):
    tile = list(_entry())
    tile[pos] = True
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tuple(tile),),))


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
def test_non_float_metrics_raise_value_error(pos):
    tile = list(_entry())
    tile[pos] = 1
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tuple(tile),),))


@pytest.mark.parametrize("pos", [6, 7, 8, 9])
@pytest.mark.parametrize("bad_value", [
    float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_raise_value_error(pos, bad_value):
    tile = list(_entry())
    tile[pos] = bad_value
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tuple(tile),),))


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_nonpositive_combined_sigma_raises_value_error(bad_sigma):
    tile = _entry(combined_sigma=bad_sigma)
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((tile,),))


def test_inverted_bounds_raise_value_error():
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((_entry(ix0=256, ix1=254),),))
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment(((_entry(iy0=256, iy1=254),),))


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = (
        _entry(tx=1, ty=0, ix0=256, ix1=511),
        _entry(tx=0, ty=0),
    )
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment((unsorted,))

    duplicate = (_entry(), _entry(ix0=256, ix1=511))
    with pytest.raises(ValueError):
        encode_tile_pyramid_assessment((duplicate,))
