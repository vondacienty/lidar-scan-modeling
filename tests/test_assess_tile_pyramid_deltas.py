"""Tests for :func:`lidar_scan.assess_tile_pyramid_deltas`."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN

import pytest

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
    assert lidar_scan.assess_tile_pyramid_deltas is assess_tile_pyramid_deltas
    assert "assess_tile_pyramid_deltas" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basic results
# ---------------------------------------------------------------------------

def test_double_empty_returns_empty_tuple():
    assert assess_tile_pyramid_deltas((), ()) == ()


def test_empty_levels_are_preserved():
    result = assess_tile_pyramid_deltas(((), (), ()), ((), (), ()))
    assert result == ((), (), ())


def test_empty_level_alongside_populated_level():
    est = ((), (_tile(tx=0),))
    ref = ((), (_tile(tx=0, zmin=0.0, zmax=1.0, count=4),))
    result = assess_tile_pyramid_deltas(est, ref)
    assert result == ((), ((0, 0, 0, 0, 255, 255, 1.0, 1.0, -3),))


def test_single_tile_deltas():
    estimate = ((_tile(zmin=1.0, zmax=2.5, count=5),),)
    reference = ((_tile(zmin=0.5, zmax=3.0, count=2),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result == (((0, 0, 0, 0, 255, 255, 0.5, -0.5, 3),),)


def test_result_fields_and_ordering():
    estimate = ((
        _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=3, iy1=3, zmin=1.0, zmax=2.0, count=5),
        _tile(tx=0, ty=1, ix0=0, iy0=4, ix1=3, iy1=7, zmin=5.0, zmax=6.0, count=9),
        _tile(tx=1, ty=0, ix0=4, iy0=0, ix1=7, iy1=3, zmin=3.0, zmax=4.0, count=2),
    ),)
    reference = ((
        _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=3, iy1=3, zmin=0.0, zmax=1.0, count=4),
        _tile(tx=0, ty=1, ix0=0, iy0=4, ix1=3, iy1=7, zmin=2.5, zmax=3.5, count=9),
        _tile(tx=1, ty=0, ix0=4, iy0=0, ix1=7, iy1=3, zmin=3.0, zmax=5.0, count=10),
    ),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result == ((
        (0, 0, 0, 0, 3, 3, 1.0, 1.0, 1),
        (0, 1, 0, 4, 3, 7, 2.5, 2.5, 0),
        (1, 0, 4, 0, 7, 3, 0.0, -1.0, -8),
    ),)


def test_level_order_and_estimate_intra_level_order():
    estimate = (
        (_tile(tx=2, count=7),),
        (_tile(tx=1, count=7),),
        (_tile(tx=0, count=7),),
    )
    reference = (
        (_tile(tx=2, count=10),),
        (_tile(tx=1, count=10),),
        (_tile(tx=0, count=10),),
    )
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert [level[0][0] for level in result] == [2, 1, 0]
    assert all(level[0][8] == -3 for level in result)


def test_dcount_is_exact_int():
    estimate = ((_tile(count=10 ** 40),),)
    reference = ((_tile(count=1),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    dcount = result[0][0][8]
    assert dcount == 10 ** 40 - 1
    assert isinstance(dcount, int)
    assert isinstance(result[0][0][6], float)
    assert isinstance(result[0][0][7], float)


def test_inputs_not_modified_and_repeatable():
    import copy
    estimate = ((_tile(zmin=1.5, count=3),),)
    reference = ((_tile(zmin=0.25, count=8),),)
    est_snapshot = copy.deepcopy(estimate)
    ref_snapshot = copy.deepcopy(reference)
    first = assess_tile_pyramid_deltas(estimate, reference)
    second = assess_tile_pyramid_deltas(estimate, reference)
    assert first == second
    assert estimate == est_snapshot
    assert reference == ref_snapshot


# ---------------------------------------------------------------------------
# Decimal arithmetic and rounding
# ---------------------------------------------------------------------------

def _expected_delta(value):
    result = float(
        Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    )
    return 0.0 if result == 0.0 else result


def test_half_even_tie_rounding():
    # Exact half-way ties round to the even sixth-decimal digit.
    reference = ((_tile(zmin=0.0, zmax=0.0),),)
    estimate = ((_tile(zmin=0.0000005, zmax=0.0000015),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result[0][0][6] == _expected_delta(0.0000005)
    assert result[0][0][7] == _expected_delta(0.0000015)


def test_negative_zero_normalized():
    pyramid = ((_tile(zmin=1.0, zmax=2.0),),)
    result = assess_tile_pyramid_deltas(pyramid, pyramid)
    dzmin, dzmax = result[0][0][6], result[0][0][7]
    assert dzmin == 0.0 and dzmax == 0.0
    assert math.copysign(1.0, dzmin) == 1.0
    assert math.copysign(1.0, dzmax) == 1.0


def test_negative_signed_zero_inputs_normalized():
    estimate = ((_tile(zmin=-0.0, zmax=0.0),),)
    reference = ((_tile(zmin=0.0, zmax=-0.0),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result[0][0][6] == 0.0
    assert result[0][0][7] == 0.0
    assert math.copysign(1.0, result[0][0][6]) == 1.0
    assert math.copysign(1.0, result[0][0][7]) == 1.0


# ---------------------------------------------------------------------------
# TypeError: outer container
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [], [()], {}, "()", None, 1, 1.0, object(),
])
def test_estimate_must_be_outer_tuple(bad):
    with pytest.raises(TypeError):
        assess_tile_pyramid_deltas(bad, ())


@pytest.mark.parametrize("bad", [
    [], [()], {}, "()", None, 1, 1.0, object(),
])
def test_reference_must_be_outer_tuple(bad):
    with pytest.raises(TypeError):
        assess_tile_pyramid_deltas((), bad)


# ---------------------------------------------------------------------------
# ValueError: structure
# ---------------------------------------------------------------------------

def test_level_must_be_tuple():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(([],), ((),))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((),), ([],))


def test_tile_must_be_tuple():
    bad = (([0, 0, 0, 0, 255, 255, 1.0, 2.0, 1],),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


def test_tile_must_have_nine_fields():
    short = (((0, 0, 0, 0, 255, 255, 1.0, 2.0),),)
    long = (((0, 0, 0, 0, 255, 255, 1.0, 2.0, 1, 9),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(short, short)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(long, long)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5, 8])
def test_integer_fields_must_be_non_bool_int(position):
    fields = [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1]
    fields[position] = True
    bad = ((tuple(fields),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5, 8])
def test_integer_fields_reject_float(position):
    fields = [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1]
    fields[position] = 1.0
    bad = ((tuple(fields),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


@pytest.mark.parametrize("position", [6, 7])
def test_z_fields_must_be_float(position):
    fields = [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1]
    fields[position] = 1
    bad = ((tuple(fields),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


@pytest.mark.parametrize("position", [6, 7])
@pytest.mark.parametrize("nonfinite", [
    float("nan"), float("inf"), float("-inf"),
])
def test_z_fields_must_be_finite(position, nonfinite):
    fields = [0, 0, 0, 0, 255, 255, 1.0, 2.0, 1]
    fields[position] = nonfinite
    bad = ((tuple(fields),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


def test_tiles_must_be_strictly_sorted():
    estimate = ((
        _tile(tx=1, ix0=256, ix1=511),
        _tile(tx=0, ix0=0, ix1=255),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, estimate)


def test_tile_coordinates_must_be_unique():
    tile = _tile(tx=0)
    estimate = ((tile, tile),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, estimate)


def test_ix_bounds_must_be_ordered():
    bad = ((_tile(ix0=10, ix1=9),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


def test_iy_bounds_must_be_ordered():
    bad = ((_tile(iy0=10, iy1=9),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


def test_zmin_must_not_exceed_zmax():
    bad = ((_tile(zmin=2.0, zmax=1.0),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(bad, bad)


# ---------------------------------------------------------------------------
# ValueError: cross-input agreement
# ---------------------------------------------------------------------------

def test_level_counts_must_match():
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((),), ((), ()))
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(((), (), ()), ((), ()))


def test_coordinate_sets_must_match():
    estimate = ((_tile(tx=0, ix0=0, ix1=255),),)
    reference = ((_tile(tx=1, ix0=256, ix1=511),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, reference)


def test_coordinate_set_mismatch_among_multiple_tiles():
    estimate = ((
        _tile(tx=0, ix0=0, ix1=255),
        _tile(tx=1, ix0=256, ix1=511),
    ),)
    reference = ((
        _tile(tx=0, ix0=0, ix1=255),
        _tile(tx=2, ix0=512, ix1=767),
    ),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, reference)


def test_tile_counts_may_differ_but_coordinates_align():
    estimate = ((_tile(tx=0, count=5),),)
    reference = ((_tile(tx=0, count=99),),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert result[0][0][8] == -94


def test_shared_cell_bounds_must_agree():
    estimate = ((_tile(tx=0, ix0=0, iy0=0, ix1=255, iy1=255),),)
    reference = ((_tile(tx=0, ix0=0, iy0=0, ix1=256, iy1=255),),)
    with pytest.raises(ValueError):
        assess_tile_pyramid_deltas(estimate, reference)


def test_matching_is_by_coordinate_not_position():
    # Both strictly sorted by (tx, ty): equal sets imply identical alignment,
    # and a differing set is rejected rather than mis-paired.
    estimate = ((
        _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255, count=2),
        _tile(tx=0, ty=1, ix0=0, iy0=256, ix1=255, iy1=511, count=4),
    ),)
    reference = ((
        _tile(tx=0, ty=0, ix0=0, iy0=0, ix1=255, iy1=255, count=1),
        _tile(tx=0, ty=1, ix0=0, iy0=256, ix1=255, iy1=511, count=1),
    ),)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert [tile[8] for tile in result[0]] == [1, 3]


# ---------------------------------------------------------------------------
# integration with the pyramid builder
# ---------------------------------------------------------------------------

def test_matches_build_tile_pyramid_outputs():
    points_a = [
        (0.0, 0.0, 1.0, 1, 0.5),
        (300.0, 0.0, 4.0, 1, 0.5),
        (-10.0, 300.0, 7.0, 1, 0.5),
    ]
    points_b = [
        (0.0, 0.0, 1.5, 1, 0.5),
        (300.0, 0.0, 3.5, 1, 0.5),
        (-10.0, 300.0, 7.0, 1, 0.5),
    ]
    estimate = tiles_module.build_tile_pyramid(points_b, cell_size=1.0,
                                               tile_cells=256, levels=2)
    reference = tiles_module.build_tile_pyramid(points_a, cell_size=1.0,
                                                tile_cells=256, levels=2)
    result = assess_tile_pyramid_deltas(estimate, reference)
    assert len(result) == 2
    for est_level, ref_level, out_level in zip(estimate, reference, result):
        assert len(out_level) == len(est_level) == len(ref_level)
        for est_tile, ref_tile, out_tile in zip(est_level, ref_level, out_level):
            assert out_tile[:6] == est_tile[:6] == ref_tile[:6]
            assert out_tile[8] == est_tile[8] - ref_tile[8]
            assert out_tile[6] == pytest.approx(
                est_tile[6] - ref_tile[6], abs=1e-9)
            assert out_tile[7] == pytest.approx(
                est_tile[7] - ref_tile[7], abs=1e-9)
