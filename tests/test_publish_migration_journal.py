"""Tests for :func:`lidar_scan.publish_migration_journal`."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lidar_scan import (build_recovery_history,
                        merge_checkout_logs, merge_recovery_indexes,
                        plan_checkout_recovery,
                        plan_recovery_index_migration,
                        publish_migration_journal,
                        resume_migration_journal, update_recovery_index)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_build_recovery_history as recovery_fixtures


def _completed_history(plan, units):
    states = recovery_fixtures._states(plan, units)
    return build_recovery_history(
        plan, tuple(recovery_fixtures._summary(plan, states[:i + 1])
                    for i in range(units)))


def _index(pairs):
    return merge_recovery_indexes((update_recovery_index(None, pairs),))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    ledger_alt = ledgers[1].replace('"b1"', '"b1a"')
    p1_alt = plan_checkout_recovery(
        log, checkpoint_one, ledgers[:1] + (ledger_alt,))

    h0 = _completed_history(p0, 1)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)

    before = _index(((p0, h0), (p12, h12)))
    after = _index(((p0, h0), (p1_alt, h1_alt)))

    return {"before": before, "after": after}


@pytest.fixture
def paths(tmp_path):
    return {
        "journal": str(tmp_path / "journal.json"),
        "index": str(tmp_path / "index.json"),
    }


def _write(path, text):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.publish_migration_journal is publish_migration_journal
    assert "publish_migration_journal" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_unfinished_document_shape(scene, paths):
    _write(paths["index"], scene["before"])
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"],
        max_steps=1)
    document = json.loads(text)
    assert list(document) == ["journal", "index", "published", "receipt"]
    assert document["published"] is False
    assert document["receipt"] is None
    assert document["index"] == json.loads(scene["before"])
    assert document["journal"]["complete"] is False
    assert list(document["journal"]) == [
        "plan", "states", "index", "complete", "receipt"]
    assert " " not in text and not text.endswith("\n")


def test_completed_document_shape_and_receipt(scene, paths):
    _write(paths["index"], scene["before"])
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    document = json.loads(text)
    assert list(document) == ["journal", "index", "published", "receipt"]
    assert document["published"] is True
    assert document["receipt"] == [2, 1, 3]
    assert document["index"] == json.loads(scene["after"])
    assert document["journal"]["complete"] is True
    assert " " not in text and not text.endswith("\n")


def test_embedded_documents_are_canonical(scene, paths):
    _write(paths["index"], scene["before"])
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    document = json.loads(text)
    canonical = json.dumps(document, ensure_ascii=False,
                           separators=(",", ":"))
    assert text == canonical


# ---------------------------------------------------------------------------
# progression and publication
# ---------------------------------------------------------------------------

def test_unfinished_call_leaves_index_and_writes_journal(scene, paths):
    _write(paths["index"], scene["before"])
    publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"],
        max_steps=1)
    with open(paths["index"], "rb") as stream:
        assert stream.read().decode("utf-8") == scene["before"]
    with open(paths["journal"], encoding="utf-8") as stream:
        journal = json.load(stream)
    assert len(journal["states"]) == 1
    assert journal["index"] == json.loads(scene["before"])


def test_completion_publishes_after_byte_for_byte(scene, paths):
    _write(paths["index"], scene["before"])
    publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"],
        max_steps=1)
    publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"],
        max_steps=1)
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    assert json.loads(text)["published"] is True
    with open(paths["index"], "rb") as stream:
        assert stream.read().decode("utf-8") == scene["after"]


def test_completed_reentry_is_byte_identical(scene, paths):
    _write(paths["index"], scene["before"])
    first = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    second = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    assert second == first
    third = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"],
        max_steps=0)
    assert third == first


def test_completed_journal_with_before_retries_publication(scene, paths):
    _write(paths["index"], scene["before"])
    journal = resume_migration_journal(
        paths["journal"], scene["before"], scene["after"])
    assert json.loads(journal)["complete"] is True
    # index still at before; publishing retries the atomic replacement
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["before"], scene["after"])
    assert json.loads(text)["published"] is True
    with open(paths["index"], "rb") as stream:
        assert stream.read().decode("utf-8") == scene["after"]


def test_index_after_with_incomplete_journal_rejected(scene, paths):
    _write(paths["index"], scene["before"])
    resume_migration_journal(
        paths["journal"], scene["before"], scene["after"], max_steps=1)
    _write(paths["index"], scene["after"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"], max_steps=0)
    # the journal advanced by the call is preserved
    with open(paths["journal"], encoding="utf-8") as stream:
        assert json.load(stream)["complete"] is False


def test_non_canonical_encoding_of_index_rejected(scene, paths):
    _write(paths["index"],
           json.dumps(json.loads(scene["before"]), indent=2))
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"])


# ---------------------------------------------------------------------------
# zero-step migration
# ---------------------------------------------------------------------------

def test_zero_step_migration_publishes_immediately(scene, paths):
    _write(paths["index"], scene["after"])
    text = publish_migration_journal(
        paths["journal"], paths["index"], scene["after"], scene["after"])
    document = json.loads(text)
    assert document["published"] is True
    assert document["receipt"] == [0, 0, 0]
    again = publish_migration_journal(
        paths["journal"], paths["index"], scene["after"], scene["after"])
    assert again == text


# ---------------------------------------------------------------------------
# index file error handling
# ---------------------------------------------------------------------------

def test_missing_index_raises_oserror(scene, paths):
    with pytest.raises(OSError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"])
    assert not os.path.exists(paths["journal"])


def test_index_bad_utf8_raises_value_error(scene, paths):
    with open(paths["index"], "wb") as stream:
        stream.write(b"\xff\xfe")
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"])


def test_index_unrelated_content_raises_value_error(scene, paths):
    _write(paths["index"], scene["after"] + " ")
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"])


def test_write_failure_keeps_old_index_and_advanced_journal(
        scene, tmp_path):
    journal_path = str(tmp_path / "journal.json")
    read_only = tmp_path / "readonly"
    read_only.mkdir()
    index_path = str(read_only / "index.json")
    _write(index_path, scene["before"])
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.geteuid() == 0:
            pytest.skip("root bypasses directory write permissions")
        with pytest.raises(OSError):
            publish_migration_journal(
                journal_path, index_path, scene["before"], scene["after"])
        with open(index_path, "rb") as stream:
            assert stream.read().decode("utf-8") == scene["before"]
        with open(journal_path, encoding="utf-8") as stream:
            assert json.load(stream)["complete"] is True
        # restoring write permission allows the retry to publish
        read_only.chmod(stat.S_IRWXU)
        text = publish_migration_journal(
            journal_path, index_path, scene["before"], scene["after"])
        assert json.loads(text)["published"] is True
        with open(index_path, "rb") as stream:
            assert stream.read().decode("utf-8") == scene["after"]
    finally:
        read_only.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# journal errors
# ---------------------------------------------------------------------------

def test_malformed_journal_raises_value_error(scene, paths):
    _write(paths["index"], scene["before"])
    _write(paths["journal"], "{not json")
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"])


def test_journal_bound_to_other_indexes_rejected(scene, tmp_path):
    first_journal = str(tmp_path / "journal.json")
    first_index = str(tmp_path / "index.json")
    _write(first_index, scene["before"])
    publish_migration_journal(
        first_journal, first_index, scene["before"], scene["after"],
        max_steps=1)
    # reuse the journal with before/after swapped
    swapped_index = str(tmp_path / "swapped.json")
    _write(swapped_index, scene["after"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            first_journal, swapped_index, scene["after"],
            scene["before"], max_steps=0)


# ---------------------------------------------------------------------------
# type and value validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_string_arguments_wrong_type(scene, paths, bad):
    jp, ip, before, after = (
        paths["journal"], paths["index"], scene["before"], scene["after"])
    with pytest.raises(TypeError):
        publish_migration_journal(bad, ip, before, after)
    with pytest.raises(TypeError):
        publish_migration_journal(jp, bad, before, after)
    with pytest.raises(TypeError):
        publish_migration_journal(jp, ip, bad, after)
    with pytest.raises(TypeError):
        publish_migration_journal(jp, ip, before, bad)


@pytest.mark.parametrize("bad", [1.0, True, False, "x", [], 1.5])
def test_max_steps_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["before"])
    with pytest.raises(TypeError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"], max_steps=bad)


def test_negative_max_steps_rejected(scene, paths):
    _write(paths["index"], scene["before"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"], max_steps=-1)


def test_empty_paths_rejected(scene, tmp_path):
    index_path = str(tmp_path / "index.json")
    _write(index_path, scene["before"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            "", index_path, scene["before"], scene["after"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            str(tmp_path / "journal.json"), "", scene["before"],
            scene["after"])


def test_equal_paths_rejected(scene, paths):
    _write(paths["index"], scene["before"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["index"], paths["index"], scene["before"],
            scene["after"])


def test_malformed_indexes_rejected(scene, paths):
    _write(paths["index"], scene["before"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], "{", scene["after"])
    with pytest.raises(ValueError):
        publish_migration_journal(
            paths["journal"], paths["index"], scene["before"],
            scene["after"] + " ")
