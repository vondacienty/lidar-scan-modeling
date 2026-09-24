"""Tests for :func:`lidar_scan.assess_tile_pyramid_deltas`."""

from __future__ import annotations

import math

from lidar_scan import assess_tile_pyramid_deltas
from lidar_scan import tiles as tiles_module


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255,
                  zmin=1.0, zmax=2.0, count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["zmin"], fields["zmax"],
            fields["count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.assess_tile_pyramid_deltas is assess_tile_pyramid_deltas
    import lidar_scan
    assert "assess_tile_pyramid_deltas" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_two_empty_pyramids_return_empty_tuple():
    assert assess_tile_pyramid_deltas((), ()) == ()


def test_empty_levels_are_preserved():
    assert assess_tile_pyramid_deltas(((), (), ()), ((), (), ())) == \
        ((), (), ())


def test_single_tile_deltas():
    estimate = ((_tile(zmin=1.5, zmax=3.0, count=10),),)
    reference = ((_tile(zmin=1.0, zmax=2.0, count=4),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result == (
        ((0, 0, 0, 0, 255, 255, 0.5, 1.0, 6),),
    )


def test_delta_signs():
    estimate = ((_tile(zmin=0.5, zmax=1.0, count=2),),)
    reference = ((_tile(zmin=2.0, zmax=4.0, count=9),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result[0][0][6] == -1.5
    assert result[0][0][7] == -3.0
    assert result[0][0][8] == -7


def test_count_delta_is_exact_int():
    result = assess_tile_pyramid_deltas(
        ((_tile(count=3),),), ((_tile(count=10),),))
    dcount = result[0][0][8]
    assert dcount == -7
    assert type(dcount) is int


def test_zero_deltas_normalized_to_positive_zero():
    result = assess_tile_pyramid_deltas(((_tile(),),), ((_tile(),),))
    tile = result[0][0]
    assert tile[6] == 0.0 and math.copysign(1.0, tile[6]) == 1.0
    assert tile[7] == 0.0 and math.copysign(1.0, tile[7]) == 1.0
    assert tile[8] == 0


def test_deltas_quantized_to_six_decimal_places():
    estimate = ((_tile(zmin=1 / 3, zmax=2 / 3),),)
    reference = ((_tile(zmin=0.0, zmax=0.0),),)
    tile = assess_tile_pyramid_deltas(estimate, reference)[0][0]
    assert tile[6] == 0.333333
    assert tile[7] == 0.666667


def test_multiple_levels_and_tile_order_preserved():
    estimate = (
        (
            _tile(tx=0, ty=0, zmin=2.0),
            _tile(tx=0, ty=1, iy0=256, iy1=511, zmin=5.0, zmax=6.0),
            _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=8.0, zmax=9.0),
        ),
        (_tile(ix1=511, iy1=511, zmin=9.0, zmax=9.5, count=2),),
    )
    reference = (
        (
            _tile(tx=0, ty=0, zmin=1.0),
            _tile(tx=0, ty=1, iy0=256, iy1=511, zmin=1.0, zmax=8.0),
            _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=8.5, zmax=9.5),
        ),
        (_tile(ix1=511, iy1=511, zmin=3.0, zmax=4.0, count=5),),
    )
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert [t[0:2] for t in result[0]] == [(0, 0), (0, 1), (1, 0)]
    assert [t[6] for t in result[0]] == [1.0, 4.0, -0.5]
    assert [t[7] for t in result[0]] == [0.0, -2.0, -0.5]
    assert result[1][0][6:9] == (6.0, 5.5, -3)


def test_result_repeats_are_identical_and_inputs_not_modified():
    estimate = ((_tile(tx=0, ty=0),
                 _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=3.0, zmax=3.5)),)
    reference = ((_tile(tx=0, ty=0, zmin=0.5),
                  _tile(tx=1, ty=0, ix0=256, ix1=511, zmin=1.0, zmax=4.0)),)
    est_snapshot = tuple(tuple(t) for t in estimate[0])
    ref_snapshot = tuple(tuple(t) for t in reference[0])
    first = assess_tile_pyramid_deltas(estimate, reference)
    second = assess_tile_pyramid_deltas(estimate, reference)
    assert first == second
    assert tuple(tuple(t) for t in estimate[0]) == est_snapshot
    assert tuple(tuple(t) for t in reference[0]) == ref_snapshot


def test_count_may_differ_when_coordinates_and_bounds_match():
    result = assess_tile_pyramid_deltas(
        ((_tile(count=1),),), ((_tile(count=999),),))
    assert result[0][0][8] == -998


# ---------------------------------------------------------------------------
# TypeError
# ---------------------------------------------------------------------------

def test_non_tuple_outer_inputs_raise_type_error():
    for bad in ([], None, 42, "x"):
        try:
            assess_tile_pyramid_deltas(bad, ())
        except TypeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected TypeError")
        try:
            assess_tile_pyramid_deltas((), bad)
        except TypeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected TypeError")


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_level_not_tuple_raises_value_error():
    try:
        assess_tile_pyramid_deltas(([_tile()],), ((_tile(),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_different_level_counts_raise_value_error():
    try:
        assess_tile_pyramid_deltas(((), ()), ((),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_tile_wrong_length_raises_value_error():
    short = (((0, 0, 0, 0, 255, 255, 1.0, 2.0),),)
    long = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 0),),)
    for bad in (short, long):
        try:
            assess_tile_pyramid_deltas(bad, ((_tile(),),))
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_tile_not_tuple_raises_value_error():
    try:
        assess_tile_pyramid_deltas((([0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],),),
                                   ((_tile(),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_non_int_indices_or_count_raise_value_error():
    for pos in (0, 1, 2, 3, 4, 5, 8):
        tile = list(_tile())
        tile[pos] = 1.5
        bad = ((tuple(tile),),)
        for args in ((bad, ((_tile(),),)), (((_tile(),),), bad)):
            try:
                assess_tile_pyramid_deltas(*args)
            except ValueError:
                pass
            else:  # pragma: no cover
                raise AssertionError(f"expected ValueError at pos {pos}")


def test_bool_indices_or_count_raise_value_error():
    for pos in (0, 1, 2, 3, 4, 5, 8):
        tile = list(_tile())
        tile[pos] = True
        bad = ((tuple(tile),),)
        try:
            assess_tile_pyramid_deltas(bad, ((_tile(),),))
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError at pos {pos}")


def test_non_float_or_nonfinite_z_bounds_raise_value_error():
    # int where a float is required
    tile = list(_tile())
    tile[6] = 1
    try:
        assess_tile_pyramid_deltas(((tuple(tile),),), ((_tile(),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    for bad_value in (float("nan"), float("inf"), float("-inf")):
        for pos in (6, 7):
            tile = list(_tile())
            tile[pos] = bad_value
            try:
                assess_tile_pyramid_deltas(((tuple(tile),),), ((_tile(),),))
            except ValueError:
                pass
            else:  # pragma: no cover
                raise AssertionError("expected ValueError")


def test_inverted_cell_bounds_raise_value_error():
    try:
        assess_tile_pyramid_deltas(
            ((_tile(ix0=256, ix1=200),),), ((_tile(),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
    try:
        assess_tile_pyramid_deltas(
            ((_tile(),),), ((_tile(iy0=256, iy1=200),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_inverted_z_bounds_raise_value_error():
    try:
        assess_tile_pyramid_deltas(
            ((_tile(zmin=2.0, zmax=1.0),),), ((_tile(),),))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_unsorted_or_duplicate_coordinates_raise_value_error():
    unsorted = ((
        _tile(tx=1, ty=0, ix0=256, ix1=511),
        _tile(tx=0, ty=0),
    ),)
    duplicate = ((
        _tile(tx=0, ty=0),
        _tile(tx=0, ty=0, ix0=256, ix1=511),
    ),)
    for bad in (unsorted, duplicate):
        try:
            assess_tile_pyramid_deltas(bad, bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# ValueError: coordinate / bounds agreement
# ---------------------------------------------------------------------------

def test_different_coordinate_sets_raise_value_error():
    estimate = ((_tile(tx=1, ty=0, ix0=256, ix1=511),),)
    reference = ((_tile(tx=0, ty=0),),)
    try:
        assess_tile_pyramid_deltas(estimate, reference)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_different_tile_counts_at_a_level_raise_value_error():
    estimate = ((_tile(), _tile(tx=1, ty=0, ix0=256, ix1=511)),)
    reference = ((_tile(),),)
    try:
        assess_tile_pyramid_deltas(estimate, reference)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_same_coordinate_mismatched_cell_bounds_raise_value_error():
    estimate = ((_tile(ix1=200),),)
    reference = ((_tile(ix1=255),),)
    try:
        assess_tile_pyramid_deltas(estimate, reference)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
