"""Tests for the ``lidar-scan-modeling publish-migration`` command line."""

from __future__ import annotations

import json
import subprocess
import sys


def _run_publish_migration(payload):
    stdin_text = json.dumps(payload)
    return subprocess.run(
        [sys.executable, "-m", "lidar_scan.cli", "publish-migration"],
        input=stdin_text, capture_output=True, text=True)


def test_missing_index_reports_unified_oserror(tmp_path):
    # A missing index raises FileNotFoundError, an OSError subclass; the
    # command must report it under the unified OSError heading.
    payload = {
        "journal_path": str(tmp_path / "journal.json"),
        "index_path": str(tmp_path / "missing-index.json"),
        "before": "before",
        "after": "after"}
    result = _run_publish_migration(payload)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "OSError"


def test_type_error_heading_unchanged(tmp_path):
    payload = {
        "journal_path": str(tmp_path / "journal.json"),
        "index_path": str(tmp_path / "index.json"),
        "before": 123,
        "after": "after"}
    result = _run_publish_migration(payload)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "TypeError"


def test_value_error_heading_unchanged():
    # Structural problem in the stdin document is reported as ValueError.
    result = _run_publish_migration({"journal_path": "j"})
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "ValueError"
