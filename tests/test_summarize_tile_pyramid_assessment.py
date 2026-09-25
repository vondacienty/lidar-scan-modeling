"""Tests for :func:`lidar_scan.summarize_tile_pyramid_assessment`."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import pytest

from lidar_scan import (assess_tile_pyramid_stats, build_tile_pyramid_stats,
                        summarize_tile_pyramid_assessment)
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
    assert (tiles_module.summarize_tile_pyramid_assessment
            is summarize_tile_pyramid_assessment)
    import lidar_scan
    assert "summarize_tile_pyramid_assessment" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty inputs
# ---------------------------------------------------------------------------

def test_empty_assessment_returns_empty_tuple():
    assert summarize_tile_pyramid_assessment(()) == ()


def test_empty_level_returns_none_fields_and_zero_sum():
    assert summarize_tile_pyramid_assessment(((),)) == (
        (0, None, None, None, None, 0),
    )


def test_mixed_empty_and_nonempty_levels():
    assessment = (
        (),
        (_entry(tx=0, ty=0, ix0=256, iy0=256, ix1=511, iy1=511,
                bias=1.0, abs_error=1.0, z_score=-2.0, count_delta=4),),
        (),
    )
    assert summarize_tile_pyramid_assessment(assessment) == (
        (0, None, None, None, None, 0),
        (1, 1.0, 1.0, 1.0, 2.0, 4),
        (2, None, None, None, None, 0),
    )


# ---------------------------------------------------------------------------
# summary metrics
# ---------------------------------------------------------------------------

def test_single_tile_metrics():
    assessment = _assessment(_entry(bias=-3.0, abs_error=3.0,
                                    combined_sigma=5.0, z_score=-0.6,
                                    count_delta=-7))
    assert summarize_tile_pyramid_assessment(assessment) == (
        (0, -3.0, -3.0, 3.0, 0.6, -7),
    )


def test_bias_min_max_abs_error_mean_and_rms_over_tiles():
    assessment = _assessment(
        _entry(tx=0, ty=0, bias=-2.0, abs_error=2.0, z_score=-3.0,
               count_delta=-5),
        _entry(tx=1, ty=0, ix0=256, ix1=511, bias=1.5, abs_error=1.5,
               z_score=4.0, count_delta=7),
        _entry(tx=2, ty=0, ix0=512, ix1=767, bias=3.0, abs_error=3.0,
               z_score=0.0, count_delta=-2),
    )
    level, bias_min, bias_max, abs_mean, z_rms, count_sum = (
        summarize_tile_pyramid_assessment(assessment)[0])
    assert level == 0
    assert bias_min == -2.0
    assert bias_max == 3.0
    assert abs_mean == pytest.approx((2.0 + 1.5 + 3.0) / 3, abs=5e-7)
    assert z_rms == pytest.approx(math.sqrt((9.0 + 16.0) / 3), abs=5e-7)
    assert count_sum == 0


def test_metrics_match_decimal_reference():
    entries = [
        _entry(tx=0, ty=0, bias=1 / 3, abs_error=1 / 7, z_score=1 / 9),
        _entry(tx=1, ty=0, ix0=256, ix1=511, bias=-2 / 3, abs_error=2 / 7,
               z_score=-1 / 6),
        _entry(tx=2, ty=0, ix0=512, ix1=767, bias=1 / 11, abs_error=1 / 13,
               z_score=2 / 3),
    ]
    result = summarize_tile_pyramid_assessment((tuple(entries),))[0]
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_HALF_EVEN
        biases = [Decimal(str(e[6])) for e in entries]
        abs_errors = [Decimal(str(e[7])) for e in entries]
        scores = [Decimal(str(e[9])) for e in entries]
        n = Decimal(len(entries))
        expected_abs_mean = float(
            (sum(sorted(abs_errors)) / n).quantize(Decimal("0.000001")))
        expected_rms = float(
            (sum(sorted(s * s for s in scores)) / n).sqrt()
            .quantize(Decimal("0.000001")))
    assert result[1] == float(min(biases).quantize(Decimal("0.000001")))
    assert result[2] == float(max(biases).quantize(Decimal("0.000001")))
    assert result[3] == expected_abs_mean
    assert result[4] == expected_rms


def test_result_independent_of_which_coordinates_hold_the_metrics():
    values = [
        dict(bias=-1.25, abs_error=1.25, z_score=-0.5, count_delta=3),
        dict(bias=2.5, abs_error=2.5, z_score=1.5, count_delta=-8),
        dict(bias=0.25, abs_error=0.25, z_score=0.0, count_delta=5),
    ]
    assessment_a = _assessment(
        _entry(tx=0, ty=0, **values[0]),
        _entry(tx=1, ty=0, ix0=256, ix1=511, **values[1]),
        _entry(tx=2, ty=0, ix0=512, ix1=767, **values[2]),
    )
    assessment_b = _assessment(
        _entry(tx=0, ty=0, **values[2]),
        _entry(tx=1, ty=0, ix0=256, ix1=511, **values[0]),
        _entry(tx=2, ty=0, ix0=512, ix1=767, **values[1]),
    )
    assert (summarize_tile_pyramid_assessment(assessment_a)
            == summarize_tile_pyramid_assessment(assessment_b))


def test_round_half_even_quantization_of_abs_error_mean():
    assessment_odd = _assessment(_entry(abs_error=1.5e-6))
    assert summarize_tile_pyramid_assessment(assessment_odd)[0][3] == 0.000002
    assessment_even = _assessment(_entry(abs_error=2.5e-6))
    assert summarize_tile_pyramid_assessment(assessment_even)[0][3] == 0.000002


def test_negative_zero_normalized_for_mean_and_rms():
    result = summarize_tile_pyramid_assessment(
        _assessment(_entry(bias=0.0, abs_error=0.0, z_score=-0.0)))[0]
    assert result[1] == 0.0 and math.copysign(1.0, result[1]) == 1.0
    assert result[2] == 0.0 and math.copysign(1.0, result[2]) == 1.0
    assert result[3] == 0.0 and math.copysign(1.0, result[3]) == 1.0
    assert result[4] == 0.0 and math.copysign(1.0, result[4]) == 1.0


def test_bias_extrema_quantized_to_six_decimal_places():
    result = summarize_tile_pyramid_assessment(
        _assessment(_entry(bias=1 / 3, abs_error=1.0, z_score=1.0),
                    _entry(tx=1, ty=0, ix0=256, ix1=511,
                           bias=-4 / 3, abs_error=1.0, z_score=1.0)))[0]
    assert result[1] == -1.333333
    assert result[2] == 0.333333


def test_from_assess_tile_pyramid_stats_output():
    estimate = build_tile_pyramid_stats(
        [(0.0, 0.0, 10.0, 0, 2.0)], levels=3)
    reference = build_tile_pyramid_stats(
        [(0.0, 0.0, 7.0, 0, 2.0)], levels=3)
    assessment = assess_tile_pyramid_stats(estimate, reference)
    summaries = summarize_tile_pyramid_assessment(assessment)
    assert len(summaries) == 3
    for level, summary in enumerate(summaries):
        assert summary == (level, 3.0, 3.0, 3.0, 1.060660, 0)


def test_input_not_modified():
    assessment = _assessment(
        _entry(bias=-1.0, abs_error=1.0, z_score=-1.0, count_delta=2),
        _entry(tx=1, ty=0, ix0=256, ix1=511,
               bias=2.0, abs_error=2.0, z_score=2.0, count_delta=-3),
    )
    snapshot = tuple(tuple(tile) for tile in assessment[0])
    summarize_tile_pyramid_assessment(assessment)
    assert assessment[0] == snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[], None, 42, "x", {}])
def test_non_tuple_assessment_raises_type_error(bad):
    with pytest.raises(TypeError):
        summarize_tile_pyramid_assessment(bad)


# ---------------------------------------------------------------------------
# ValueError
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(([_entry()],))


def test_tile_wrong_length_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((_entry()[:10],),))


@pytest.mark.parametrize("field_index,bad_value", [
    (0, True), (1, True), (2, 1.0), (3, False), (4, 1.0), (5, 2.0),
    (10, True),
])
def test_non_bool_int_fields_raise_value_error(field_index, bad_value):
    tile = list(_entry())
    tile[field_index] = bad_value
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tuple(tile),),))


@pytest.mark.parametrize("field_index", [6, 7, 8, 9])
@pytest.mark.parametrize("bad_value", [1, math.nan, math.inf, -math.inf])
def test_non_finite_float_metrics_raise_value_error(field_index, bad_value):
    tile = list(_entry())
    tile[field_index] = bad_value
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tuple(tile),),))


def test_non_positive_combined_sigma_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(combined_sigma=0.0)))


def test_negative_abs_error_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(abs_error=-0.5)))


def test_inverted_bounds_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(ix0=256, ix1=255)))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(iy0=256, iy1=255)))


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(_assessment(
            _entry(tx=1, ty=0, ix0=256, ix1=511),
            _entry(tx=0, ty=0),
        ))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(_assessment(_entry(), _entry()))
