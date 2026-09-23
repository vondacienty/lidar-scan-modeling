"""Tests for :func:`lidar_scan.dem.merge_dems`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import build_dem, merge_dems


def test_empty_outer_tuple_returns_empty_tuple():
    assert merge_dems(()) == ()


def test_empty_members_return_empty_tuple():
    assert merge_dems(((),)) == ()
    assert merge_dems(((), (), ())) == ()


def test_non_tuple_outer_raises_type_error():
    for bad in [[], None, iter(())]:
        with pytest.raises(TypeError):
            merge_dems(bad)


def test_non_tuple_member_raises_value_error_even_inside_tuple():
    with pytest.raises(ValueError):
        merge_dems((((() for _ in (0,))),))


def test_single_member_passthrough_round_trips_build_dem():
    dem = build_dem([(0.0, 0.0, 2.0, 0, 1.0), (0.5, 0.5, 4.0, 0, 1.0)])
    assert merge_dems((dem,)) == dem


def test_equal_sigmas_give_unweighted_mean():
    result = merge_dems((((0, 0, 2.0, 1.0, 3),), ((0, 0, 4.0, 1.0, 1),)))
    assert result == ((0, 0, 3.0, pytest.approx(1 / math.sqrt(2), abs=1e-6), 4),)


def test_inverse_variance_weighting():
    # sigma 0.5 -> w=4, sigma 1 -> w=1
    result = merge_dems((((0, 0, 5.5, 0.5, 2),), ((0, 0, 5.0, 1.0, 9),)))
    ix, iy, z, sigma, count = result[0]
    assert (ix, iy) == (0, 0)
    assert z == pytest.approx(5.4, abs=1e-6)
    assert sigma == pytest.approx(math.sqrt(0.2), abs=1e-6)
    assert count == 11


def test_missing_coordinates_still_emitted():
    result = merge_dems((
        ((0, 0, 2.0, 1.0, 1), (1, 0, 9.0, 3.0, 2)),
        ((0, 0, 4.0, 1.0, 1),),
    ))
    assert [(c[0], c[1]) for c in result] == [(0, 0), (1, 0)]
    only = result[1]
    assert (only[2], only[3], only[4]) == (9.0, 3.0, 2)


def test_disjoint_coordinates_form_union_sorted():
    result = merge_dems((
        ((2, 0, 1.0, 1.0, 1),),
        ((-1, 5, 1.0, 1.0, 1),),
        ((0, -3, 1.0, 1.0, 1), (0, 2, 1.0, 1.0, 1)),
    ))
    assert [(c[0], c[1]) for c in result] == [(-1, 5), (0, -3), (0, 2), (2, 0)]


def test_count_is_exact_integer_sum():
    result = merge_dems((
        ((0, 0, 1.0, 1.0, 10 ** 40),),
        ((0, 0, 2.0, 1.0, 10 ** 40),),
    ))
    assert result[0][4] == 2 * 10 ** 40


def test_result_independent_of_member_order():
    a = ((0, 0, 5.5, 0.5, 2), (0, 1, 12.0, 1.0, 3), (2, -1, 10.0, 2.0, 1))
    b = ((0, 0, 5.0, 1.5, 9), (0, 1, 11.0, 1.0, 4), (2, -1, 10.0, 2.0, 7))
    assert merge_dems((a, b)) == merge_dems((b, a))


def test_negative_zero_normalized():
    result = merge_dems((((0, 0, -0.0, 1.0, 1),),))
    z = result[0][2]
    assert z == 0.0 and math.copysign(1.0, z) == 1.0


def test_inputs_not_modified():
    a = ((0, 0, 2.0, 1.0, 3), (1, 1, 4.0, 1.0, 1))
    snapshot = tuple(cell for cell in a)
    merge_dems((a, a))
    assert a == snapshot


def test_non_tuple_member_raises_value_error():
    with pytest.raises(ValueError):
        merge_dems(([],))
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, 1.0, 1),), "not a tuple"))


def test_cell_must_be_tuple():
    with pytest.raises(ValueError):
        merge_dems((([0, 0, 1.0, 1.0, 1],),))


def test_wrong_field_count_raises_value_error():
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, 1.0),),))
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, 1.0, 1, 9),),))
    with pytest.raises(ValueError):
        merge_dems(((1,),))


@pytest.mark.parametrize("pos", [0, 1, 4])
def test_non_int_index_or_count_raises_value_error(pos):
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = 1.5
    with pytest.raises(ValueError):
        merge_dems(((tuple(cell),),))
    cell[pos] = "1"
    with pytest.raises(ValueError):
        merge_dems(((tuple(cell),),))


@pytest.mark.parametrize("pos", [0, 1, 2, 3, 4])
def test_bool_fields_rejected(pos):
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = True
    with pytest.raises(ValueError):
        merge_dems(((tuple(cell),),))


@pytest.mark.parametrize("pos", [2, 3])
def test_non_float_z_or_sigma_raises_value_error(pos):
    cell = [0, 0, 1.0, 1.0, 1]
    cell[pos] = 1
    with pytest.raises(ValueError):
        merge_dems(((tuple(cell),),))
    cell[pos] = "1.0"
    with pytest.raises(ValueError):
        merge_dems(((tuple(cell),),))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_z_or_sigma_raises_value_error(bad):
    with pytest.raises(ValueError):
        merge_dems((((0, 0, bad, 1.0, 1),),))
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, bad, 1),),))


def test_nonpositive_sigma_raises_value_error():
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, 0.0, 1),),))
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, -1.5, 1),),))


def test_unsorted_or_duplicate_cells_raise_value_error():
    with pytest.raises(ValueError):
        merge_dems((((1, 0, 1.0, 1.0, 1), (0, 0, 1.0, 1.0, 1)),))
    with pytest.raises(ValueError):
        merge_dems((((0, 0, 1.0, 1.0, 1), (0, 0, 2.0, 1.0, 1)),))
    with pytest.raises(ValueError):
        merge_dems((((0, 1, 1.0, 1.0, 1), (0, 0, 1.0, 1.0, 1)),))
