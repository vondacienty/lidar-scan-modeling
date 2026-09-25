"""Tests for :func:`lidar_scan.gate_report`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (accumulate_reconciliation,
                        finalize_reconciliation_state, gate_report)
from lidar_scan import tiles as tiles_module


WINDOWS = ((0, 0, 0, 9, 9), (0, 100, 100, 109, 109))

REPORT = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
    '[0,100,100,109,109,null]]}'
)


def _tile(**overrides):
    fields = dict(tx=0, ty=0, ix0=0, iy0=0, ix1=9, iy1=9,
                  err_zmin=1.0, err_zmax=2.0, err_count=1)
    fields.update(overrides)
    return (fields["tx"], fields["ty"], fields["ix0"], fields["iy0"],
            fields["ix1"], fields["iy1"], fields["err_zmin"],
            fields["err_zmax"], fields["err_count"])


def _report(assessment, windows=WINDOWS):
    state = accumulate_reconciliation(None, assessment, windows)
    return finalize_reconciliation_state(state, windows)


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
    assert gate_report('{"windows":[]}', (1, 1, 1, 1)) == (
        '{"windows":[],"passed":true}'
    )


def test_null_summary_window_fails():
    assert gate_report(REPORT, (5, 5, 5, 1)) == (
        '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],'
        'true],[0,100,100,109,109,null,false]],"passed":false}'
    )


def test_all_windows_must_pass():
    report = (
        '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,[0.000000,0.500000,0.250000,0.500000,3]]]}'
    )
    assert gate_report(report, (2, 2, 1, 1)) == (
        '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],'
        'true],[0,100,100,109,109,[0.000000,0.500000,0.250000,0.500000,3],'
        'true]],"passed":true}'
    )


def test_window_flags_are_independent():
    report = (
        '{"windows":[[0,0,0,9,9,[-3.000000,2.000000,1.581139,1.000000,1]],'
        '[0,100,100,109,109,[0.000000,0.500000,0.250000,0.500000,3]]]}'
    )
    result = json.loads(gate_report(report, (2, 2, 1, 1)))
    assert result["windows"][0][6] is False
    assert result["windows"][1][6] is True
    assert result["passed"] is False


def test_boundary_values_are_inclusive():
    report = (
        '{"windows":[[0,0,0,9,9,[-2.000000,2.000000,1.581139,1.000000,1]]]}'
    )
    assert gate_report(report, (2, 1.581139, 1.0, 1)).endswith(
        '"passed":true}')


@pytest.mark.parametrize("limits", [
    (1.999999, 5, 2, 1),          # abs(emin)/abs(emax) exceed max_abs
    (2, 1.581138, 2, 1),          # rmse exceeds
    (2, 5, 0.999999, 1),          # amean exceeds
    (2, 5, 2, 2),                 # n below min_n
])
def test_any_violation_fails_the_window(limits):
    report = (
        '{"windows":[[0,0,0,9,9,[-2.000000,2.000000,1.581139,1.000000,1]]]}'
    )
    result = json.loads(gate_report(report, limits))
    assert result["windows"][0][6] is False
    assert result["passed"] is False


def test_negative_zero_limits_and_metrics_pass_at_zero():
    report = (
        '{"windows":[[0,0,0,9,9,[0.000000,0.000000,0.000000,0.000000,1]]]}'
    )
    assert gate_report(report, (0.0, -0.0, -0.0, 1)) == (
        '{"windows":[[0,0,0,9,9,[0.000000,0.000000,0.000000,0.000000,1],'
        'true]],"passed":true}'
    )


def test_integer_limits_are_accepted():
    report = (
        '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.000000,1.000000,2]]]}'
    )
    result = json.loads(gate_report(report, (2, 1, 1, 2)))
    assert result["passed"] is True


def test_output_shape_and_key_order():
    text = gate_report(REPORT, (5, 5, 5, 1))
    assert not text.endswith("\n")
    assert '"windows"' in text and '"passed"' in text
    assert list(json.loads(text)) == ["windows", "passed"]
    window = json.loads(text)["windows"][0]
    assert len(window) == 7
    assert window[:5] == [0, 0, 0, 9, 9]
    assert window[5] == [-1.0, 2.0, 1.581139, 1.0, 1]
    assert window[6] is True


def test_accepts_finalize_output_end_to_end():
    assessment = ((_tile(err_zmin=-1.0, err_zmax=2.0, err_count=2),
                   _tile(tx=1, err_zmin=0.5, err_zmax=1.5, err_count=4)),)
    report = _report(assessment, ((0, 0, 0, 9, 9),))
    text = gate_report(report, (2, 2, 3, 2))
    assert json.loads(text)["passed"] is True


def test_inputs_are_not_modified():
    snapshot = REPORT
    gate_report(REPORT, (5, 5, 5, 1))
    assert REPORT == snapshot


def test_repeated_calls_are_byte_identical():
    assert gate_report(REPORT, (5, 5, 5, 1)) == gate_report(
        REPORT, (5, 5, 5, 1))


# ---------------------------------------------------------------------------
# argument validation: types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_report", [None, 0, 1.5, b"", [], {}])
def test_report_wrong_type_raises_type_error(bad_report):
    with pytest.raises(TypeError):
        gate_report(bad_report, (1, 1, 1, 1))


@pytest.mark.parametrize("bad_limits",
    [[1, 1, 1, 1], (1, 1, 1), (1, 1, 1, 1, 1)])
def test_limits_container_and_length_raise_type_error(bad_limits):
    with pytest.raises(TypeError):
        gate_report(REPORT, bad_limits)


@pytest.mark.parametrize("bad_limits", [
    (True, 1, 1, 1),
    (1, False, 1, 1),
    (1, 1, "1", 1),
    (1, 1, None, 1),
    (1, 1, 1, 1.0),
    (1, 1, 1, True),
    (1, 1, 1, 1.5),
])
def test_limit_field_types_raise_type_error(bad_limits):
    with pytest.raises(TypeError):
        gate_report(REPORT, bad_limits)


# ---------------------------------------------------------------------------
# argument validation: values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_limits", [
    (-1, 1, 1, 1),
    (1, -0.000001, 1, 1),
    (1, 1, -1, 1),
    (1, 1, 1, 0),
    (1, 1, 1, -1),
    (float("nan"), 1, 1, 1),
    (1, float("inf"), 1, 1),
    (1, 1, float("-inf"), 1),
])
def test_limit_values_raise_value_error(bad_limits):
    with pytest.raises(ValueError):
        gate_report(REPORT, bad_limits)


@pytest.mark.parametrize("bad_report", [
    '{',
    'null',
    '[]',
    '{}',
    '{"levels":[]}',
    '{"windows":{}}',
    '{"windows":null}',
    '{"windows":[[]]}',
    '{"windows":[[0,0,0,9,9]]}',
    '{"windows":[[0,0,0,9,9,[1,2,3,4]]]}',
    '{"windows":[[0,0,0,9,9,{}]]}',
    '{"windows":[["0",0,0,9,9,null]]}',
    '{"windows":[[true,0,0,9,9,null]]}',
    '{"windows":[[-1,0,0,9,9,null]]}',
    '{"windows":[[0,9,0,0,9,null]]}',
    '{"windows":[[0,0,9,9,0,null]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,1.000000,1.000000,0]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,-1.000000,1.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,1.000000,-1.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,1.000000,1.000000,true]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,2.000000,1.000000,1.0,1]]]}',
    '{"windows":[[0,0,0,9,9,[1,2,1,1,1]]]}',
    '{"windows":[[0,0,0,9,9,[NaN,2.000000,1.000000,1.000000,1]]]}',
    '{"windows":[[0,0,0,9,9,[1.000000,Infinity,1.000000,1.000000,1]]]}',
    '{ "windows":[]}',
    '{"windows" :[]}',
    "{'windows':[]}",
    '{"windows":[],"extra":1}',
    '{"passed":false,"windows":[]}',
    '{"windows":[[0,0,0,9,9,null,true]]}',
])
def test_bad_report_raises_value_error(bad_report):
    with pytest.raises(ValueError):
        gate_report(bad_report, (1, 1, 1, 1))
