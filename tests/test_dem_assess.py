"""Tests for :func:`lidar_scan.dem.assess_dem`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import assess_dem


def test_empty_pair_returns_empty_tuple():
    assert assess_dem([], []) == ()
    assert assess_dem(iter(()), iter(())) == ()


def test_reference_still_validated_when_estimate_empty():
    with pytest.raises(ValueError):
        assess_dem([], ((0, 0, 0.0, 1.0, 1), (0, 0, 0.0, 2.0, 1)))


def test_single_cell_metrics():
    result = assess_dem([(0, 0, 12.0, 3.0, 2)], [(0, 0, 9.0, 4.0, 5)])
    ix, iy, bias, abs_error, combined, z_score = result[0]
    assert (ix, iy) == (0, 0)
    assert bias == 3.0
    assert abs_error == 3.0
    assert combined == pytest.approx(5.0, abs=1e-6)
    assert z_score == pytest.approx(0.6, abs=1e-6)


def test_negative_bias_gives_negative_z_score():
    result = assess_dem([(0, 0, 9.0, 3.0, 2)], [(0, 0, 12.0, 4.0, 5)])
    _, _, bias, abs_error, _, z_score = result[0]
    assert bias == -3.0
    assert abs_error == 3.0
    assert z_score == pytest.approx(-0.6, abs=1e-6)


def test_zero_bias_normalized_to_positive_zero():
    result = assess_dem([(0, 0, 10.0, 2.0, 1)], [(0, 0, 10.0, 2.0, 7)])
    _, _, bias, abs_error, combined, z_score = result[0]
    assert bias == 0.0 and math.copysign(1.0, bias) == 1.0
    assert abs_error == 0.0
    assert combined == pytest.approx(math.sqrt(8.0), abs=1e-6)
    assert z_score == 0.0 and math.copysign(1.0, z_score) == 1.0


def test_output_sorted_by_ix_iy_regardless_of_input_order():
    estimate = [(2, 0, 1.0, 1.0, 1), (-1, 5, 1.0, 1.0, 1),
                (0, -3, 1.0, 1.0, 1), (0, 2, 1.0, 1.0, 1)]
    reference = [(0, 2, 0.0, 1.0, 1), (0, -3, 0.0, 1.0, 1),
                 (-1, 5, 0.0, 1.0, 1), (2, 0, 0.0, 1.0, 1)]
    result = assess_dem(reversed(estimate), reversed(reference))
    assert [(row[0], row[1]) for row in result] == [(-1, 5), (0, -3), (0, 2), (2, 0)]


def test_order_independent_values():
    estimate = [(0, 1, 12.0, 1.0, 3), (2, -1, 10.0, 2.0, 1),
                (0, 0, 5.5, 0.5, 2)]
    reference = [(0, 0, 5.0, 1.5, 9), (2, -1, 10.0, 2.0, 7),
                 (0, 1, 11.0, 1.0, 4)]
    assert assess_dem(estimate, reference) == assess_dem(
        list(reversed(estimate)), list(reversed(reference)))


def test_single_pass_iterables_accepted():
    def gen(data):
        yield from data

    estimate = [(0, 0, 2.0, 1.0, 1)]
    reference = [(0, 0, 1.0, 1.0, 1)]
    result = assess_dem(gen(estimate), gen(reference))
    assert result[0][2] == 1.0


def test_integer_values_accepted():
    result = assess_dem([(0, 0, 5, 3, 2)], [(0, 0, 1, 4, 1)])
    assert result == ((0, 0, 4, 4, 5.0, 0.8),)


def test_mismatched_index_sets_raise_value_error():
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1), (1, 0, 1.0, 1.0, 1)],
                   [(0, 0, 1.0, 1.0, 1)])
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1)],
                   [(0, 0, 1.0, 1.0, 1), (1, 0, 1.0, 1.0, 1)])
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1)],
                   [(0, 1, 1.0, 1.0, 1)])


def test_one_empty_one_nonempty_raises_value_error():
    with pytest.raises(ValueError):
        assess_dem([], [(0, 0, 1.0, 1.0, 1)])
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1)], [])


def test_duplicate_index_raises_value_error():
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1), (0, 0, 2.0, 1.0, 1)],
                   [(0, 0, 1.0, 1.0, 1)])
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 1.0, 1)],
                   [(0, 0, 1.0, 1.0, 1), (0, 0, 2.0, 1.0, 1)])


def test_nonpositive_sigma_raises_value_error():
    good = [(0, 0, 1.0, 1.0, 1)]
    with pytest.raises(ValueError):
        assess_dem([(0, 0, 1.0, 0.0, 1)], good)
    with pytest.raises(ValueError):
        assess_dem(good, [(0, 0, 1.0, -1.5, 1)])


def test_nonfinite_z_raises_value_error():
    good = [(0, 0, 1.0, 1.0, 1)]
    for bad in float("nan"), float("inf"), float("-inf"):
        with pytest.raises(ValueError):
            assess_dem([(0, 0, bad, 1.0, 1)], good)
        with pytest.raises(ValueError):
            assess_dem(good, [(0, 0, bad, 1.0, 1)])
        with pytest.raises(ValueError):
            assess_dem([(0, 0, 1.0, bad, 1)], good)


def test_non_iterable_raises_type_error():
    with pytest.raises(TypeError):
        assess_dem(42, [(0, 0, 1.0, 1.0, 1)])
    with pytest.raises(TypeError):
        assess_dem([(0, 0, 1.0, 1.0, 1)], None)


def test_wrong_element_type_raises_type_error():
    good = [(0, 0, 1.0, 1.0, 1)]
    with pytest.raises(TypeError):
        assess_dem([{0, 0, 1.0, 1.0, 1}], good)
    with pytest.raises(TypeError):
        assess_dem(good, ["abc"])


def test_wrong_field_count_raises_type_error():
    good = [(0, 0, 1.0, 1.0, 1)]
    with pytest.raises(TypeError):
        assess_dem([(0, 0, 1.0, 1.0)], good)
    with pytest.raises(TypeError):
        assess_dem(good, [(0, 0, 1.0, 1.0, 1, 9)])


@pytest.mark.parametrize("pos", [0, 1, 4])
def test_non_int_index_or_count_raises_type_error(pos):
    good = [(0, 0, 1.0, 1.0, 1)]
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = 1.5
    with pytest.raises(TypeError):
        assess_dem([tuple(cell)], good)
    cell[pos] = "1"
    with pytest.raises(TypeError):
        assess_dem(good, [tuple(cell)])


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
def test_bool_fields_rejected(pos):
    good = [(0, 0, 1.0, 1.0, 1)]
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = True
    with pytest.raises(TypeError):
        assess_dem([tuple(cell)], good)


@pytest.mark.parametrize("pos", [2, 3])
def test_non_numeric_z_or_sigma_raises_type_error(pos):
    good = [(0, 0, 1.0, 1.0, 1)]
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = 1 + 0j
    with pytest.raises(TypeError):
        assess_dem([tuple(cell)], good)
    cell[pos] = "1.0"
    with pytest.raises(TypeError):
        assess_dem(good, [tuple(cell)])


def test_inputs_not_modified():
    estimate = [[0, 0, 1.0, 1.0, 1], [1, 1, 2.0, 2.0, 3]]
    reference = [[0, 0, 0.0, 1.0, 2], [1, 1, 0.0, 2.0, 4]]
    estimate_snapshot = [list(cell) for cell in estimate]
    reference_snapshot = [list(cell) for cell in reference]
    assess_dem(estimate, reference)
    assert estimate == estimate_snapshot
    assert reference == reference_snapshot


def test_quantized_to_six_decimal_places():
    # bias 1/3 -> 0.333333; combined sigma from sigma=1 and sigma=1
    result = assess_dem([(0, 0, 1 / 3, 1.0, 1)], [(0, 0, 0.0, 1.0, 1)])
    _, _, bias, _, combined, z_score = result[0]
    assert bias == 0.333333
    assert combined == pytest.approx(math.sqrt(2.0), abs=5e-7)
    assert z_score == pytest.approx((1 / 3) / math.sqrt(2.0), abs=5e-7)
