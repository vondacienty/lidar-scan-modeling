"""Tests for the ``lidar-scan-modeling window-tiles`` command line entry."""

from __future__ import annotations

import subprocess
import sys


def _run_window_tiles(stdin_text: str):
    return subprocess.run(
        [sys.executable, "-m", "lidar_scan.cli", "window-tiles"],
        input=stdin_text, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# success
# ---------------------------------------------------------------------------

def test_success_writes_one_compact_json_line():
    result = _run_window_tiles(
        '{"points":[[0.0,0.0,1.0,0,1.0]],"windows":[[0,0,0,255,255]]}')
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        '{"windows":[[0,0,0,255,255,'
        '[[0,0,0,0,255,255,1.000000,1.000000,1]]]]}\n')


def test_empty_points_and_windows():
    result = _run_window_tiles('{"points":[],"windows":[]}')
    assert result.returncode == 0
    assert result.stdout == '{"windows":[]}\n'


# ---------------------------------------------------------------------------
# point field type errors -> TypeError, exit 2, empty stdout
# ---------------------------------------------------------------------------

def test_point_field_wrong_type_reports_type_error():
    for document in (
            '{"points":[["x",0,0,0,1]],"windows":[]}',
            '{"points":[[0,"x",0,0,1]],"windows":[]}',
            '{"points":[[0,0,"x",0,1]],"windows":[]}',
            '{"points":[[0,0,0,"x",1]],"windows":[]}',
            '{"points":[[0,0,0,0,"x"]],"windows":[]}',
            '{"points":[[true,0,0,0,1]],"windows":[]}',
            '{"points":[[null,0,0,0,1]],"windows":[]}',
            '{"points":[[[0],0,0,0,1]],"windows":[]}',
            '{"points":[[{"x":0},0,0,0,1]],"windows":[]}'):
        result = _run_window_tiles(document)
        assert result.returncode == 2, document
        assert result.stdout == "", document
        assert result.stderr.splitlines()[0] == "TypeError", document


def test_point_field_type_error_late_in_the_stream():
    result = _run_window_tiles(
        '{"points":[[0,0,0,0,1],[1,1,1,1,1],[2,"x",2,2,2]],'
        '"windows":[[0,0,0,255,255]]}')
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "TypeError"


# ---------------------------------------------------------------------------
# JSON / array structure errors -> ValueError, exit 2, empty stdout
# ---------------------------------------------------------------------------

def test_structure_errors_report_value_error():
    for document in (
            "",
            "not json",
            "{",
            "[1,2]",
            "42",
            '"text"',
            "{}",
            '{"points":[]}',
            '{"windows":[]}',
            '{"points":[],"windows":[],"extra":1}',
            '{"points":5,"windows":[]}',
            '{"points":[5],"windows":[]}',
            '{"points":[[0,0,0]],"windows":[]}',
            '{"points":[[0,0,0,0,1,2]],"windows":[]}',
            '{"points":[],"windows":5}',
            '{"points":[],"windows":[5]}',
            '{"points":[],"windows":[[0,0,0,1]]}',
            '{"points":[],"windows":[[0,0,0,1,1,2]]}'):
        result = _run_window_tiles(document)
        assert result.returncode == 2, document
        assert result.stdout == "", document
        assert result.stderr.splitlines()[0] == "ValueError", document


def test_invalid_utf8_reports_value_error():
    result = subprocess.run(
        [sys.executable, "-m", "lidar_scan.cli", "window-tiles"],
        input=b'{"points":[]\xff}', capture_output=True)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.splitlines()[0] == b"ValueError"


# ---------------------------------------------------------------------------
# field value errors -> ValueError
# ---------------------------------------------------------------------------

def test_field_value_errors_report_value_error():
    for document in (
            '{"points":[[0,0,0,0,0]],"windows":[]}',
            '{"points":[[0,0,0,0,-1]],"windows":[]}',
            '{"points":[],"windows":[],"cell_size":0}',
            '{"points":[],"windows":[],"cell_size":-1.5}',
            '{"points":[],"windows":[],"tile_cells":0}',
            '{"points":[],"windows":[],"levels":0}',
            '{"points":[],"windows":[[3,0,0,1,1]]}',
            '{"points":[],"windows":[[0,5,0,1,1]]}'):
        result = _run_window_tiles(document)
        assert result.returncode == 2, document
        assert result.stdout == "", document
        assert result.stderr.splitlines()[0] == "ValueError", document


def test_scalar_field_type_errors_report_type_error():
    for document in (
            '{"points":[],"windows":[],"cell_size":"1"}',
            '{"points":[],"windows":[],"cell_size":true}',
            '{"points":[],"windows":[],"tile_cells":1.5}',
            '{"points":[],"windows":[],"tile_cells":true}',
            '{"points":[],"windows":[],"levels":1.5}',
            '{"points":[],"windows":[],"levels":null}',
            '{"points":[],"windows":[[0,0,0,1,"1"]]}',
            '{"points":[],"windows":[["0",0,0,1,1]]}'):
        result = _run_window_tiles(document)
        assert result.returncode == 2, document
        assert result.stdout == "", document
        assert result.stderr.splitlines()[0] == "TypeError", document
