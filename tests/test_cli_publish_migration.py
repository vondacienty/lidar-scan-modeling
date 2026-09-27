"""Tests for the ``lidar-scan-modeling publish-migration`` CLI entry."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_publish_migration(document):
    return subprocess.run(
        [sys.executable, "-m", "lidar_scan.cli", "publish-migration"],
        input=json.dumps(document), capture_output=True, text=True)


def _empty_index():
    from lidar_scan import merge_recovery_indexes, update_recovery_index
    return merge_recovery_indexes((update_recovery_index(None, ()),))


# ---------------------------------------------------------------------------
# success
# ---------------------------------------------------------------------------

def test_zero_step_migration_succeeds(tmp_path):
    index_path = str(tmp_path / "index.json")
    index = _empty_index()
    with open(index_path, "w", encoding="utf-8") as stream:
        stream.write(index)
    document = {
        "journal_path": str(tmp_path / "journal.json"),
        "index_path": index_path,
        "before": index,
        "after": index,
    }
    result = _run_publish_migration(document)
    assert result.returncode == 0
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output["published"] is True
    assert output["receipt"] == [0, 0, 0]
    assert result.stdout.endswith("\n")


# ---------------------------------------------------------------------------
# OSError subclasses are reported with the single first line "OSError"
# ---------------------------------------------------------------------------

def test_missing_index_reports_oserror_not_subclass(tmp_path):
    document = {
        "journal_path": str(tmp_path / "journal.json"),
        "index_path": str(tmp_path / "missing-index.json"),
        "before": _empty_index(),
        "after": _empty_index(),
    }
    result = _run_publish_migration(document)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "OSError"
    # The concrete subclass name must not leak onto the first line.
    assert result.stderr.splitlines()[0] != "FileNotFoundError"


def test_value_error_keeps_its_own_name(tmp_path):
    index_path = str(tmp_path / "index.json")
    with open(index_path, "w", encoding="utf-8") as stream:
        stream.write("{}")
    document = {
        "journal_path": str(tmp_path / "journal.json"),
        "index_path": index_path,
        "before": _empty_index(),
        "after": _empty_index(),
    }
    result = _run_publish_migration(document)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "ValueError"


def test_type_error_keeps_its_own_name(tmp_path):
    index_path = str(tmp_path / "index.json")
    with open(index_path, "w", encoding="utf-8") as stream:
        stream.write(_empty_index())
    document = {
        "journal_path": 1,
        "index_path": index_path,
        "before": _empty_index(),
        "after": _empty_index(),
    }
    result = _run_publish_migration(document)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "TypeError"


def test_malformed_json_input_is_value_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "lidar_scan.cli", "publish-migration"],
        input="{not json", capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.splitlines()[0] == "ValueError"
    assert os.listdir(tmp_path) == []
