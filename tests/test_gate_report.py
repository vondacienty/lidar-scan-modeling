"""Tests for :func:`lidar_scan.gate_report`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (accumulate_reconciliation,
                        finalize_reconciliation_state, gate_report)
from lidar_scan import tiles as tiles_module

WINDOWS = ((0, 0, 0, 9, 9), (0, 100, 100, 109, 109))

R_A = '[-2.000000,5.000000,1.414214,2.000000,5]'
R_B = '[1.000000,2.000000,1.581139,1.000000,1]'
REPORT = (
    '{"windows":[[0,0,0,9,9,' + R_A + '],'
    '[0,100,100,109,109,' + R_B + ']]}'
)


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=9, iy1=9,
                  err_zmin=1.0, err_zmax=2.0, err_count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["err_zmin"],
            fields["err_zmax"], fields["err_count"])


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.gate_report is gate_report
    import lidar_scan
    assert "gate_report" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_empty_windows_passes():
    assert gate_report('{"windows":[]}', (0, 0, 0, 1)) == (
        '{"windows":[],"passed":true}'
    )


def test_all_windows_pass():
    result = gate_report(REPORT, (5.0, 2.0, 2.0, 1))
    assert result == (
        '{"windows":[[0,0,0,9,9,[-2.000000,5.000000,1.414214,2.000000,5,true]],'
        '[0,100,100,109,109,[1.000000,2.000000,1.581139,1.000000,1,true]]],'
        '"passed":true}'
    )


def test_null_summary_never_passes():
    report = '{"windows":[[0,0,0,9,9,null]]}'
    result = gate_report(report, (100.0, 100.0, 100.0, 1))
    assert result == (
        '{"windows":[[0,0,0,9,9,null,false]],"passed":false}'
    )


def test_key_order_is_windows_then_passed():
    result = json.loads(gate_report('{"windows":[]}', (0, 0, 0, 1)))
    assert list(result) == ["windows", "passed"]


def test_integer_limits_are_accepted():
    result = gate_report(REPORT, (5, 2, 2, 1))
    assert result.endswith('"passed":true}')


@pytest.mark.parametrize("limits,expected", [
    ((4.999999, 2.0, 2.0, 1), False),   # |emin| = 2, emax = 5 > 4.999999
    ((5.0, 1.414213, 2.0, 1), False),   # rmse 1.414214 > limit
    ((5.0, 2.0, 1.999999, 1), False),   # amean 2.0 > limit
    ((5.0, 2.0, 2.0, 6), False),        # n 5 < min_n
    ((5.0, 1.414214, 2.0, 5), True),    # boundary inclusive
    ((5, 2, 2, 5), True),               # integer limits, inclusive
])
def test_limit_boundaries(limits, expected):
    result = gate_report(
        '{"windows":[[0,0,0,9,9,' + R_A + ']]}', limits
    )
    flag = "true" if expected else "false"
    assert result.endswith('"passed":' + flag + "}")
    assert (R_A[:-1] + "," + flag + "]") in result


def test_overall_passed_requires_every_window():
    # First window passes rmse 1.5, second window's rmse 1.581139 fails.
    result = gate_report(REPORT, (5.0, 1.5, 2.0, 1))
    assert result == (
        '{"windows":[[0,0,0,9,9,[-2.000000,5.000000,1.414214,2.000000,5,true]],'
        '[0,100,100,109,109,[1.000000,2.000000,1.581139,1.000000,1,false]]],'
        '"passed":false}'
    )


def test_zero_limits_pass_zero_values_with_negative_zero():
    report = (
        '{"windows":[[0,0,0,9,9,'
        '[0.000000,0.000000,0.000000,0.000000,1]]]}'
    )
    assert gate_report(report, (0.0, 0.0, 0.0, 1)) == (
        '{"windows":[[0,0,0,9,9,'
        '[0.000000,0.000000,0.000000,0.000000,1,true]]],"passed":true}'
    )


def test_round_trip_from_finalize():
    assessment = ((_tile(err_zmin=1.0, err_zmax=2.0, err_count=1),
                   _tile(tx=1, err_zmin=-3.0, err_zmax=5.0, err_count=-2)),)
    windows = ((0, 0, 0, 9, 9),)
    state = accumulate_reconciliation(None, assessment, windows)
    report = finalize_reconciliation_state(state, windows)
    result = gate_report(report, (5.0, 4.0, 2.0, 2))
    assert result == (
        '{"windows":[[0,0,0,9,9,'
        '[-3.000000,5.000000,3.122499,1.500000,2,true]]],"passed":true}'
    )


def test_output_has_no_trailing_newline_or_whitespace():
    result = gate_report(REPORT, (5.0, 2.0, 2.0, 1))
    assert not result.endswith("\n")
    assert " " not in result
    assert "NaN" not in result and "Infinity" not in result


def test_original_report_window_entries_are_preserved_byte_for_byte():
    result = gate_report(REPORT, (0.0, 0.0, 0.0, 99))
    # Five key fields and R remain exactly as supplied; only flags added.
    assert result == (
        '{"windows":[[0,0,0,9,9,[-2.000000,5.000000,1.414214,2.000000,5,false]],'
        '[0,100,100,109,109,[1.000000,2.000000,1.581139,1.000000,1,false]]],'
        '"passed":false}'
    )


def test_inputs_are_not_modified():
    limits = [5.0, 2.0, 2.0, 1]
    with pytest.raises(TypeError):
        gate_report(REPORT, limits)
    assert limits == [5.0, 2.0, 2.0, 1]
    snapshot = REPORT
    gate_report(REPORT, (5.0, 2.0, 2.0, 1))
    assert REPORT == snapshot


def test_repeated_calls_are_byte_identical():
    assert (gate_report(REPORT, (5.0, 2.0, 2.0, 1))
            == gate_report(REPORT, (5.0, 2.0, 2.0, 1)))


# ---------------------------------------------------------------------------
# argument validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_report", [None, 0, 1.5, b"", [], {}])
def test_report_wrong_type_raises_type_error(bad_report):
    with pytest.raises(TypeError):
        gate_report(bad_report, (0, 0, 0, 1))


@pytest.mark.parametrize("bad_limits", [None, [], {}, 0, 1.5, "(0,0,0,1)"])
def test_limits_wrong_type_raises_type_error(bad_limits):
    with pytest.raises(TypeError):
        gate_report('{"windows":[]}', bad_limits)


@pytest.mark.parametrize("bad_limits", [
    (),
    (0, 0, 0),
    (0, 0, 0, 1, 2),
])
def test_limits_wrong_length_raises_type_error(bad_limits):
    with pytest.raises(TypeError):
        gate_report('{"windows":[]}', bad_limits)


@pytest.mark.parametrize("bad_limits", [
    (True, 0, 0, 1),
    (0, False, 0, 1),
    (0, 0, 1j, 1),
    ("0", 0, 0, 1),
    (0, None, 0, 1),
])
def test_limit_numeric_fields_wrong_type_raise_type_error(bad_limits):
    with pytest.raises(TypeError):
        gate_report('{"windows":[]}', bad_limits)


@pytest.mark.parametrize("bad_min_n", [True, False, 1.0, 0.0, "1", None])
def test_min_n_wrong_type_raises_type_error(bad_min_n):
    with pytest.raises(TypeError):
        gate_report('{"windows":[]}', (0, 0, 0, bad_min_n))


# ---------------------------------------------------------------------------
# argument validation: values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_limits", [
    (float("nan"), 0, 0, 1),
    (float("inf"), 0, 0, 1),
    (0, float("-inf"), 0, 1),
    (-0.1, 0, 0, 1),
    (0, -1, 0, 1),
    (0, 0, -0.000001, 1),
])
def test_bad_numeric_limit_values_raise_value_error(bad_limits):
    with pytest.raises(ValueError):
        gate_report('{"windows":[]}', bad_limits)


@pytest.mark.parametrize("bad_min_n", [0, -1, -10])
def test_min_n_non_positive_raises_value_error(bad_min_n):
    with pytest.raises(ValueError):
        gate_report('{"windows":[]}', (0, 0, 0, bad_min_n))


@pytest.mark.parametrize("bad_report", [
    '{',
    'null',
    '[]',
    '{}',
    '{"levels":[]}',
    '{"windows":{}}',
    '{"windows":null}',
    '{"windows":[[0,0,0,9,9]]}',
    '{"windows":[[0,0,0,9,9,[1,2,3,4]]]}',
    '{"windows":[[0,0,0,9,9,{}]]}',
    '{"windows":[["0",0,0,9,9,null]]}',
    '{"windows":[[true,0,0,9,9,null]]}',
    '{"windows":[[-1,0,0,9,9,null]]}',
    '{"windows":[[0,9,0,0,9,null]]}',
    '{"windows":[[0,0,9,9,0,null]]}',
    '{"windows":[[0,0,0,9,9,[1,2,0,0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.0,2.000000,0.000000,0.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0.000000,0.5,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0.000000,0.000000,1.0]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0.000000,0.000000,true]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0.000000,0.000000,0]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,-0.000001,0.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,0.000000,-0.000001,1]]]}',
    '{"windows":[[0,0,0,9,9,[NaN,2.000000,0.000000,0.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,Infinity,0.000000,0.000000,1]]]}',
    '{ "windows":[]}',
    '{"windows" :[]}',
    "{'windows':[]}",
    '{"windows":[],"extra":1}',
])
def test_bad_report_raises_value_error(bad_report):
    with pytest.raises(ValueError):
        gate_report(bad_report, (0, 0, 0, 1))
