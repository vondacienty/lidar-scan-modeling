"""Tests for :func:`lidar_scan.assess_delta_summary`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import assess_delta_summary
from lidar_scan import tiles as tiles_module


def _window(key=(0, 0, 0, 10, 10), summary=(-1.0, 2.0, 5, 1)):
    level, ix_min, iy_min, ix_max, iy_max = key
    return (level, ix_min, iy_min, ix_max, iy_max, summary)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.assess_delta_summary is assess_delta_summary
    import lidar_scan
    assert "assess_delta_summary" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_double_empty_returns_empty_tuple():
    assert assess_delta_summary((), ()) == ()


def test_double_none_summary_gives_none_delta():
    estimate = (_window(summary=None),)
    reference = (_window(summary=None),)
    assert assess_delta_summary(estimate, reference) == (
        (0, 0, 0, 10, 10, None),)


def test_basic_float_differences():
    estimate = (_window(summary=(-1.5, 3.25, 7, 3)),)
    reference = (_window(summary=(0.5, 1.0, 2, 1)),)
    assert assess_delta_summary(estimate, reference) == (
        (0, 0, 0, 10, 10, (-2.0, 2.25, 5, 2)),)


def test_integer_extrema_difference_is_exact_int():
    estimate = (_window(summary=(-2, 4, 7, 2)),)
    reference = (_window(summary=(1, 3, 10, 5)),)
    result = assess_delta_summary(estimate, reference)
    delta = result[0][5]
    assert delta == (-3, 1, -3, -3)
    assert isinstance(delta[0], int) and isinstance(delta[1], int)
    assert isinstance(delta[2], int) and isinstance(delta[3], int)


def test_mixed_int_float_extremum_difference_is_quantized_float():
    estimate = (_window(summary=(1, 2.5, 0, 1)),)
    reference = (_window(summary=(0.25, 2, 0, 1)),)
    result = assess_delta_summary(estimate, reference)
    delta = result[0][5]
    assert delta[0] == pytest.approx(0.75)
    assert delta[1] == pytest.approx(0.5)
    assert isinstance(delta[0], float) and isinstance(delta[1], float)


def test_result_keys_are_echoed_in_order():
    estimate = (
        _window(key=(0, 0, 0, 10, 10), summary=(0.0, 1.0, 1, 1)),
        _window(key=(1, 5, 5, 9, 9), summary=(2.0, 3.0, 4, 2)),
    )
    reference = (
        _window(key=(0, 0, 0, 10, 10), summary=(0.0, 0.0, 0, 1)),
        _window(key=(1, 5, 5, 9, 9), summary=(1.0, 1.0, 1, 1)),
    )
    result = assess_delta_summary(estimate, reference)
    assert result == (
        (0, 0, 0, 10, 10, (0.0, 1.0, 1, 0)),
        (1, 5, 5, 9, 9, (1.0, 2.0, 3, 1)),
    )


def test_negative_zero_is_normalized():
    estimate = (_window(summary=(1.0, 1.0, 1, 1)),)
    reference = (_window(summary=(1.0, 1.0, 1, 1)),)
    result = assess_delta_summary(estimate, reference)
    dmin, dmax, _, _ = result[0][5]
    assert dmin == 0.0 and math.copysign(1.0, dmin) == 1.0
    assert dmax == 0.0 and math.copysign(1.0, dmax) == 1.0


def test_decimal_precision_half_even_quantization():
    estimate = (_window(summary=(1.0000005, 2.0, 1, 1)),)
    reference = (_window(summary=(0.0, 0.0, 0, 1)),)
    result = assess_delta_summary(estimate, reference)
    # Round-half-even at the seventh decimal rounds 1.0000005 -> 1.000000.
    assert result[0][5][0] == 1.0
    assert result[0][5][1] == 2.0


def test_inputs_are_not_modified():
    estimate = [_window(summary=(-1.5, 3.25, 7, 3))]
    reference = [_window(summary=(0.5, 1.0, 2, 1))]
    est_copy = tuple(estimate)
    ref_copy = tuple(reference)
    assess_delta_summary(tuple(estimate), tuple(reference))
    assert tuple(estimate) == est_copy
    assert tuple(reference) == ref_copy


# ---------------------------------------------------------------------------
# type errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, [], {}, "x", 1])
def test_non_tuple_argument_raises_type_error(value):
    good = (_window(),)
    with pytest.raises(TypeError):
        assess_delta_summary(value, good)
    with pytest.raises(TypeError):
        assess_delta_summary(good, value)


# ---------------------------------------------------------------------------
# value errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    (0, 0, 0, 10, 10),                            # not a 6-tuple
    (0, 0, 0, 10, 10, (1.0, 2.0, 1, 1), 9),       # 7-tuple
])
def test_bad_window_length_raises_value_error(bad):
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


@pytest.mark.parametrize("field_index", range(5))
def test_bool_key_field_raises_value_error(field_index):
    fields = [0, 0, 0, 10, 10]
    fields[field_index] = True
    bad = tuple(fields) + ((0.0, 1.0, 1, 1),)
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


@pytest.mark.parametrize("field_index", range(5))
def test_non_int_key_field_raises_value_error(field_index):
    fields = [0, 0, 0, 10, 10]
    fields[field_index] = 1.5
    bad = tuple(fields) + ((0.0, 1.0, 1, 1),)
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


def test_negative_level_raises_value_error():
    bad = (-1, 0, 0, 10, 10, (0.0, 1.0, 1, 1))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(key=(-1, 0, 0, 10, 10)),))


@pytest.mark.parametrize("key", [
    (0, 5, 0, 4, 10),   # ix_min > ix_max
    (0, 0, 5, 10, 4),   # iy_min > iy_max
])
def test_inverted_bounds_raise_value_error(key):
    with pytest.raises(ValueError):
        assess_delta_summary((_window(key=key),), (_window(key=key),))


def test_duplicate_keys_raise_value_error():
    windows = (_window(), _window())
    with pytest.raises(ValueError):
        assess_delta_summary(windows, windows)


def test_unsorted_keys_raise_value_error():
    windows = (
        _window(key=(1, 0, 0, 10, 10)),
        _window(key=(0, 0, 0, 10, 10)),
    )
    with pytest.raises(ValueError):
        assess_delta_summary(windows, windows)


def test_bad_summary_length_raises_value_error():
    bad = (0, 0, 0, 10, 10, (0.0, 1.0, 1))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


def test_summary_not_tuple_raises_value_error():
    bad = (0, 0, 0, 10, 10, [0.0, 1.0, 1, 1])
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


@pytest.mark.parametrize("extremum", [True, False, "1.0", None, float("nan"),
                                      float("inf"), float("-inf")])
def test_bad_extremum_raises_value_error(extremum):
    bad = (0, 0, 0, 10, 10, (extremum, 1.0, 1, 1))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


def test_integer_extrema_are_accepted():
    estimate = (_window(summary=(0, 1, 2, 1)),)
    reference = (_window(summary=(-1, 0, 0, 1)),)
    assert assess_delta_summary(estimate, reference)[0][5] == (1, 1, 2, 0)


def test_min_greater_than_max_raises_value_error():
    bad = (0, 0, 0, 10, 10, (2.0, 1.0, 1, 1))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


@pytest.mark.parametrize("count_value", [True, False, 1.0, "1"])
def test_bad_count_type_raises_value_error(count_value):
    bad = (0, 0, 0, 10, 10, (0.0, 1.0, count_value, 1))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


def test_non_positive_match_count_raises_value_error():
    bad = (0, 0, 0, 10, 10, (0.0, 1.0, 1, 0))
    with pytest.raises(ValueError):
        assess_delta_summary((bad,), (_window(),))


def test_different_lengths_raise_value_error():
    with pytest.raises(ValueError):
        assess_delta_summary((_window(),), ())


def test_different_key_sets_raise_value_error():
    estimate = (_window(key=(0, 0, 0, 10, 10)),)
    reference = (_window(key=(0, 0, 0, 11, 10)),)
    with pytest.raises(ValueError):
        assess_delta_summary(estimate, reference)


def test_none_status_mismatch_raises_value_error():
    estimate = (_window(summary=None),)
    reference = (_window(summary=(0.0, 1.0, 1, 1)),)
    with pytest.raises(ValueError):
        assess_delta_summary(estimate, reference)
    with pytest.raises(ValueError):
        assess_delta_summary(reference, estimate)
