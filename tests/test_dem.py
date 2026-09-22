"""Tests for :func:`lidar_scan.dem.assess_dem`."""

from __future__ import annotations

import math

import pytest

from lidar_scan import __all__ as root_all
from lidar_scan import assess_dem
from lidar_scan.cli import main


def gen(cells):
    """A single-pass generator with no ``len``."""
    return (cell for cell in cells)


def test_empty_pair_returns_empty_tuple():
    assert assess_dem((), ()) == ()
    assert assess_dem(gen(()), gen(())) == ()
    assert isinstance(assess_dem((), ()), tuple)


def test_single_cell_metrics():
    estimate = ((0, 0, 5.0, 1.0, 3),)
    reference = ((0, 0, 3.0, 0.75, 2),)
    # bias=2, abs=2, combined=sqrt(1 + 0.5625)=1.25, z=2/1.25=1.6
    assert assess_dem(estimate, reference) == ((0, 0, 2.0, 2.0, 1.25, 1.6),)


def test_negative_bias_and_abs_error():
    result = assess_dem(((0, 0, 3.0, 1.0, 1),), ((0, 0, 5.0, 1.0, 1),))
    # sqrt(2) quantized to six places is 1.414214
    assert result == ((0, 0, -2.0, 2.0, 1.414214, -1.414214),)


def test_output_sorted_by_index_regardless_of_input_order():
    estimate = ((2, 1, 1.0, 1.0, 1),
                (-1, -1, 1.0, 1.0, 1),
                (0, 0, 1.0, 1.0, 1),
                (2, 0, 1.0, 1.0, 1))
    reference = tuple((ix, iy, 0.0, 1.0, 1) for ix, iy, _, _, _ in reversed(estimate))
    result = assess_dem(gen(estimate), gen(reference))
    assert [row[:2] for row in result] == [(-1, -1), (0, 0), (2, 0), (2, 1)]
    for row in result:
        assert row[2:] == (1.0, 1.0, pytest.approx(math.sqrt(2)), pytest.approx(1 / math.sqrt(2)))


def test_list_elements_accepted():
    assert assess_dem([[0, 0, 5, 1, 3]], [[0, 0, 3, 0.75, 2]]) == \
        ((0, 0, 2.0, 2.0, 1.25, 1.6),)


def test_single_pass_iterables_consumed_once():
    # generators (no len, no indexing) must work directly
    result = assess_dem(gen([(0, 0, 1.0, 2.0, 7)]), gen([(0, 0, 1.0, 2.0, 5)]))
    assert result == ((0, 0, 0.0, 0.0, pytest.approx(math.sqrt(8)), 0.0),)


class _IterableWithoutLen:
    """Iterable wrapper that must not be probed with ``len``."""

    def __init__(self, cells):
        self._cells = cells
        self.len_called = False

    def __iter__(self):
        return iter(self._cells)

    def __len__(self):  # pragma: no cover - must never be invoked
        self.len_called = True
        raise AssertionError("assess_dem must not call len() on its inputs")


def test_no_len_and_input_not_modified():
    estimate_cells = [(1, 2, 5.0, 1.0, 3), (0, 0, 4.0, 1.0, 2)]
    reference_cells = [(0, 0, 3.0, 1.0, 2), (1, 2, 6.0, 1.0, 3)]
    estimate_copy = [tuple(cell) for cell in estimate_cells]
    reference_copy = [tuple(cell) for cell in reference_cells]

    est = _IterableWithoutLen(estimate_cells)
    ref = _IterableWithoutLen(reference_cells)
    assess_dem(est, ref)

    assert not est.len_called and not ref.len_called
    assert [tuple(cell) for cell in estimate_cells] == estimate_copy
    assert [tuple(cell) for cell in reference_cells] == reference_copy


def test_negative_zero_normalized_to_positive():
    # |bias| below half a micro in magnitude quantizes to zero; sign must be dropped
    result = assess_dem(((0, 0, -0.0000004, 1.0, 1),), ((0, 0, 0.0, 1.0, 1),))
    row = result[0]
    assert row[2] == 0.0 and math.copysign(1.0, row[2]) == 1.0
    assert row[3] == 0.0 and math.copysign(1.0, row[3]) == 1.0
    assert row[5] == 0.0 and math.copysign(1.0, row[5]) == 1.0


def test_quantize_six_places_round_half_even():
    # 7th digit 5 after an odd 6th digit -> rounds up; after an even digit -> stays
    up = assess_dem(((0, 0, 2.0000015, 1.0, 1),), ((0, 0, 0.0, 1.0, 1),))
    even = assess_dem(((0, 0, 2.0000025, 1.0, 1),), ((0, 0, 0.0, 1.0, 1),))
    assert up[0][2] == 2.000002
    assert even[0][2] == 2.000002


def test_duplicate_estimate_index_raises_value_error():
    with pytest.raises(ValueError):
        assess_dem(((0, 0, 1.0, 1.0, 1), (0, 0, 2.0, 1.0, 1)),
                   ((0, 0, 0.0, 1.0, 1),))


def test_duplicate_reference_index_raises_value_error():
    with pytest.raises(ValueError):
        assess_dem(((0, 0, 1.0, 1.0, 1),),
                   ((0, 0, 0.0, 1.0, 1), (0, 0, 3.0, 1.0, 1)))


def test_index_sets_must_match():
    with pytest.raises(ValueError):
        assess_dem(((0, 0, 1.0, 1.0, 1), (1, 0, 1.0, 1.0, 1)),
                   ((0, 0, 0.0, 1.0, 1),))
    with pytest.raises(ValueError):
        assess_dem(((0, 0, 1.0, 1.0, 1),), ())
    with pytest.raises(ValueError):
        assess_dem((), ((0, 0, 0.0, 1.0, 1),))


def test_non_iterable_raises_type_error():
    with pytest.raises(TypeError):
        assess_dem(42, ())
    with pytest.raises(TypeError):
        assess_dem((), 42)


def test_element_wrong_type_or_length_raises_type_error():
    good = (0, 0, 1.0, 1.0, 1)
    with pytest.raises(TypeError):
        assess_dem((good,), ((0, 0, 0.0, 1.0),))             # too short
    with pytest.raises(TypeError):
        assess_dem(((0, 0, 1.0, 1.0, 1, 9),), (good,))       # too long
    with pytest.raises(TypeError):
        assess_dem(({0, 0, 1, 1, 1},), (good,))              # not tuple/list
    with pytest.raises(TypeError):
        assess_dem(("not-a-cell",), (good,))


def test_field_type_errors_raise_type_error():
    good = (0, 0, 1.0, 1.0, 1)
    bad_cases = [
        (0.0, 0, 1.0, 1.0, 1),   # ix float
        (True, 0, 1.0, 1.0, 1),  # ix bool
        (0, False, 1.0, 1.0, 1), # iy bool
        (0, 0, "1.0", 1.0, 1),   # z str
        (0, 0, 1.0, True, 1),    # sigma bool
        (0, 0, 1.0, 1.0, 1.5),   # count float
        (0, 0, 1.0, 1.0, True),  # count bool
        (0, 0, 1.0, 1.0, None),  # count None
    ]
    for bad in bad_cases:
        with pytest.raises(TypeError):
            assess_dem((bad,), (good,))
        with pytest.raises(TypeError):
            assess_dem((good,), (bad,))


def test_non_finite_values_raise_value_error():
    good = (0, 0, 1.0, 1.0, 1)
    for bad_z in (float("nan"), float("inf"), float("-inf")):
        bad = (0, 0, bad_z, 1.0, 1)
        with pytest.raises(ValueError):
            assess_dem((bad,), (good,))
        with pytest.raises(ValueError):
            assess_dem((good,), (bad,))
    for bad_sigma in (float("nan"), float("inf"), float("-inf")):
        bad = (0, 0, 1.0, bad_sigma, 1)
        with pytest.raises(ValueError):
            assess_dem((bad,), (good,))


def test_non_positive_sigma_raises_value_error():
    good = (0, 0, 1.0, 1.0, 1)
    for sigma in (0, -1, -0.5):
        bad = (0, 0, 1.0, sigma, 1)
        with pytest.raises(ValueError):
            assess_dem((bad,), (good,))
        with pytest.raises(ValueError):
            assess_dem((good,), (bad,))


def test_exported_from_root_and_module():
    import lidar_scan
    from lidar_scan import dem
    assert "assess_dem" in root_all
    assert lidar_scan.assess_dem is assess_dem
    assert dem.assess_dem is assess_dem


def test_cli_version_and_help_unchanged(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage: lidar-scan-modeling" in out
    assert "version" in out
