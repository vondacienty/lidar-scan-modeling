"""Tests for :func:`summarize_tile_pyramid_assessment`."""

from __future__ import annotations

import math

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


def test_empty_levels_keep_level_index_and_zero_count_sum():
    assert summarize_tile_pyramid_assessment(((), ())) == (
        (0, None, None, None, None, 0),
        (1, None, None, None, None, 0),
    )


def test_empty_level_summary_field_types():
    summary = summarize_tile_pyramid_assessment(((),))[0]
    level, bias_min, bias_max, abs_error_mean, z_score_rms, count_sum = summary
    assert level == 0
    assert bias_min is None and bias_max is None
    assert abs_error_mean is None and z_score_rms is None
    assert isinstance(count_sum, int) and count_sum == 0


# ---------------------------------------------------------------------------
# single tile
# ---------------------------------------------------------------------------

def test_single_tile_level_summary():
    assessment = _assessment(_entry(bias=-1.25, abs_error=1.25,
                                    combined_sigma=2.5, z_score=-0.5,
                                    count_delta=7))
    assert summarize_tile_pyramid_assessment(assessment) == (
        (0, -1.25, -1.25, 1.25, 0.5, 7),
    )


def test_zero_metrics_are_positive_zero_floats():
    (summary,) = summarize_tile_pyramid_assessment(
        _assessment(_entry(bias=0.0, abs_error=0.0, combined_sigma=1.0,
                           z_score=-0.0, count_delta=0)))
    _, bias_min, bias_max, abs_error_mean, z_score_rms, count_sum = summary
    for value in (bias_min, bias_max, abs_error_mean, z_score_rms):
        assert isinstance(value, float)
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0
    assert count_sum == 0


# ---------------------------------------------------------------------------
# multiple tiles
# ---------------------------------------------------------------------------

def test_multiple_tiles_aggregate():
    assessment = _assessment(
        _entry(tx=0, bias=1.0, abs_error=1.0, combined_sigma=2.0,
               z_score=0.5, count_delta=2),
        _entry(tx=1, bias=3.0, abs_error=3.0, combined_sigma=4.0,
               z_score=-0.5, count_delta=-5),
    )
    assert summarize_tile_pyramid_assessment(assessment) == (
        (0, 1.0, 3.0, 2.0, 0.5, -3),
    )


def test_rms_and_mean_with_irrational_values():
    assessment = _assessment(
        _entry(tx=0, bias=-1.5, abs_error=1.5, combined_sigma=5.0,
               z_score=0.3, count_delta=0),
        _entry(tx=1, bias=2.5, abs_error=2.5, combined_sigma=5.0,
               z_score=-0.7, count_delta=0),
    )
    summary = summarize_tile_pyramid_assessment(assessment)[0]
    assert summary[0] == 0
    assert summary[1] == -1.5
    assert summary[2] == 2.5
    assert summary[3] == 2.0
    # sqrt((0.09 + 0.49) / 2) = sqrt(0.29) = 0.53851648... -> 0.538516
    assert summary[4] == pytest.approx(0.538516, abs=1e-9)
    assert summary[5] == 0


def test_metrics_quantized_to_six_decimal_places():
    assessment = _assessment(
        _entry(tx=0, bias=1.0, abs_error=1.0, z_score=0.123456789),
        _entry(tx=1, bias=2.0, abs_error=2.0, z_score=0.0),
    )
    _, _, _, abs_error_mean, z_score_rms, _ = (
        summarize_tile_pyramid_assessment(assessment)[0])
    assert abs_error_mean == 1.5
    # sqrt((0.123456789**2) / 2) rounded to six decimals
    assert z_score_rms == pytest.approx(0.087297, abs=1e-9)
    for value in (abs_error_mean, z_score_rms):
        assert value == round(value, 6)


def test_bias_min_max_choose_extremes_not_abs():
    assessment = _assessment(
        _entry(tx=0, bias=-4.0, abs_error=4.0),
        _entry(tx=1, bias=2.0, abs_error=2.0),
    )
    _, bias_min, bias_max, _, _, _ = summarize_tile_pyramid_assessment(
        assessment)[0]
    assert bias_min == -4.0
    assert bias_max == 2.0


def test_count_delta_sum_is_exact_int():
    assessment = _assessment(
        _entry(tx=0, count_delta=10 ** 30),
        _entry(tx=1, count_delta=-(10 ** 30) + 3),
        _entry(tx=2, count_delta=-3),
    )
    summary = summarize_tile_pyramid_assessment(assessment)[0]
    assert isinstance(summary[5], int)
    assert summary[5] == 0


def test_aggregation_independent_of_metric_assignment_to_coordinates():
    first = _assessment(
        _entry(tx=0, bias=-1.5, abs_error=1.5, z_score=0.3, count_delta=2),
        _entry(tx=1, bias=2.5, abs_error=2.5, z_score=-0.7, count_delta=-9),
    )
    second = _assessment(
        _entry(tx=0, bias=2.5, abs_error=2.5, z_score=-0.7, count_delta=-9),
        _entry(tx=1, bias=-1.5, abs_error=1.5, z_score=0.3, count_delta=2),
    )
    assert (summarize_tile_pyramid_assessment(first)
            == summarize_tile_pyramid_assessment(second))


def test_multiple_levels_summarized_independently():
    assessment = (
        (_entry(tx=0, ty=0, bias=1.0, abs_error=1.0, z_score=0.5,
                count_delta=4),),
        (),
        (_entry(tx=0, ty=0, ix1=511, iy1=511, bias=-2.0, abs_error=2.0,
                z_score=-1.0, count_delta=-1),
         _entry(tx=1, ty=0, ix1=511, iy1=511, bias=2.0, abs_error=2.0,
                z_score=1.0, count_delta=1)),
    )
    assert summarize_tile_pyramid_assessment(assessment) == (
        (0, 1.0, 1.0, 1.0, 0.5, 4),
        (1, None, None, None, None, 0),
        (2, -2.0, 2.0, 2.0, 1.0, 0),
    )


# ---------------------------------------------------------------------------
# end-to-end with assess_tile_pyramid_stats
# ---------------------------------------------------------------------------

def test_end_to_end_with_assess_tile_pyramid_stats():
    points = [(0.1, 0.2, 1.0, 1, 0.5), (300.0, 5.0, 4.0, 0, 1.0),
              (600.0, 600.0, -2.0, 0, 2.0)]
    estimate = build_tile_pyramid_stats(points, levels=2)
    reference = build_tile_pyramid_stats(
        [(0.4, 0.9, 0.5, 0, 0.7), (300.0, 5.0, 5.0, 0, 1.0),
         (600.0, 600.0, -1.0, 0, 1.5)], levels=2)
    assessment = assess_tile_pyramid_stats(estimate, reference)
    summaries = summarize_tile_pyramid_assessment(assessment)

    assert len(summaries) == len(assessment) == 2
    for level, (level_tiles, summary) in enumerate(zip(assessment,
                                                       summaries)):
        assert summary[0] == level
        if not level_tiles:
            assert summary == (level, None, None, None, None, 0)
            continue
        biases = [tile[6] for tile in level_tiles]
        abs_errors = [tile[7] for tile in level_tiles]
        z_scores = [tile[9] for tile in level_tiles]
        count_deltas = [tile[10] for tile in level_tiles]
        assert summary[1] == pytest.approx(min(biases))
        assert summary[2] == pytest.approx(max(biases))
        assert summary[3] == pytest.approx(sum(abs_errors) / len(abs_errors))
        assert summary[4] == pytest.approx(
            math.sqrt(sum(z * z for z in z_scores) / len(z_scores)))
        assert summary[5] == sum(count_deltas)


def test_input_not_modified():
    assessment = _assessment(
        _entry(tx=0, bias=1.0, count_delta=2),
        _entry(tx=1, bias=3.0, count_delta=-5),
    )
    snapshot = tuple(tuple(tile) for level in assessment for tile in level)
    summarize_tile_pyramid_assessment(assessment)
    current = tuple(tuple(tile) for level in assessment for tile in level)
    assert current == snapshot


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, [], [()], {}, "()", 0, False])
def test_non_tuple_assessment_raises_type_error(bad):
    with pytest.raises(TypeError):
        summarize_tile_pyramid_assessment(bad)


# ---------------------------------------------------------------------------
# ValueError
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(([_entry()],))


def test_tile_not_tuple_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(([list(_entry())],))


@pytest.mark.parametrize("length", [10, 12])
def test_tile_bad_length_raises_value_error(length):
    tile = _entry()[:length] if length < 11 else _entry() + (1,)
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tile,),))


@pytest.mark.parametrize("field_index", [0, 1, 2, 3, 4, 5, 10])
def test_bool_integer_field_raises_value_error(field_index):
    values = list(_entry())
    values[field_index] = True
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tuple(values),),))


@pytest.mark.parametrize("field_index", [0, 1, 2, 3, 4, 5, 10])
def test_non_integer_field_raises_value_error(field_index):
    values = list(_entry())
    values[field_index] = 1.5
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tuple(values),),))


@pytest.mark.parametrize("field_index", [6, 7, 8, 9])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"),
                                 float("-inf"), 1, 0])
def test_metric_not_finite_float_raises_value_error(field_index, bad):
    values = list(_entry())
    values[field_index] = bad
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(((tuple(values),),))


def test_non_positive_combined_sigma_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(combined_sigma=0.0)))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(combined_sigma=-1.0)))


def test_negative_abs_error_raises_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(abs_error=-0.000001)))


def test_negative_zero_abs_error_is_accepted():
    assert summarize_tile_pyramid_assessment(
        _assessment(_entry(bias=-0.0, abs_error=-0.0)))[0][3] == 0.0


def test_inverted_bounds_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(ix0=10, ix1=9)))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(
            _assessment(_entry(iy0=10, iy1=9)))


def test_unsorted_coordinates_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(_assessment(
            _entry(tx=1), _entry(tx=0)))


def test_duplicate_coordinates_raise_value_error():
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(_assessment(
            _entry(tx=0, ty=1), _entry(tx=0, ty=0)))
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(_assessment(
            _entry(tx=0), _entry(tx=0)))


def test_bad_second_level_still_raises_value_error():
    assessment = (
        (_entry(tx=0),),
        (_entry(tx=0), _entry(tx=0)),
    )
    with pytest.raises(ValueError):
        summarize_tile_pyramid_assessment(assessment)
