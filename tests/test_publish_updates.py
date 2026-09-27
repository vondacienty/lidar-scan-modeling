"""Tests for :func:`lidar_scan.publish_updates`."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lidar_scan import (build_recovery_history,
                        merge_checkout_logs, merge_recovery_indexes,
                        plan_checkout_recovery, publish_updates,
                        update_recovery_index)
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


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    h0 = _completed_history(p0, 1)
    h12 = _completed_history(p12, 2)

    states = recovery_fixtures._states(p12, 2)
    partial12 = build_recovery_history(
        p12, (recovery_fixtures._summary(p12, states[:1]),))

    empty = _index(())
    after0 = _index(((p0, h0),))
    after1 = _index(((p0, h0), (p12, h12)))

    return {"log": log, "p0": p0, "p12": p12, "h0": h0, "h12": h12,
            "partial12": partial12, "empty": empty,
            "after0": after0, "after1": after1,
            "batch0": (("b0", ((p0, h0),)),),
            "batch1": (("b1", ((p12, h12),)),)}


@pytest.fixture
def paths(tmp_path):
    return {"state": str(tmp_path / "state.json"),
            "index": str(tmp_path / "index.json")}


def _write(path, text):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text)


def _read(path):
    with open(path, "rb") as stream:
        return stream.read().decode("utf-8")


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.publish_updates is publish_updates
    assert "publish_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# document shape
# ---------------------------------------------------------------------------

def test_empty_batches_document_shape(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"], ())
    document = json.loads(text)
    assert list(document) == ["batches", "index", "complete"]
    assert document["batches"] == []
    assert document["index"] == json.loads(scene["empty"])
    assert document["complete"] is True
    assert _read(paths["state"]) == text
    assert " " not in text and not text.endswith("\n")


def test_completed_document_shape_and_receipt(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(
        paths["state"], paths["index"], scene["batch0"] + scene["batch1"])
    document = json.loads(text)
    assert list(document) == ["batches", "index", "complete"]
    assert document["complete"] is True
    assert [row[0] for row in document["batches"]] == ["b0", "b1"]
    row0, row1 = document["batches"]
    assert row0[3:7] == [1, 1, True, [0, 1, 1]]
    assert row1[3:7] == [2, 2, True, [0, 2, 2]]
    assert document["index"] == json.loads(scene["after1"])
    assert _read(paths["state"]) == text
    assert " " not in text and not text.endswith("\n")


def test_batch_row_is_seven_fields_and_chains(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(
        paths["state"], paths["index"], scene["batch0"] + scene["batch1"])
    rows = json.loads(text)["batches"]
    for row in rows:
        assert len(row) == 7
    assert rows[0][1] == json.loads(scene["empty"])
    assert rows[0][2] == json.loads(scene["after0"])
    assert rows[1][1] == json.loads(scene["after0"])
    assert rows[1][2] == json.loads(scene["after1"])


def test_embedded_documents_are_canonical(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(
        paths["state"], paths["index"], scene["batch0"])
    document = json.loads(text)
    assert text == _canonical(document)


def test_state_initialized_when_missing(scene, paths):
    _write(paths["index"], scene["empty"])
    assert not os.path.exists(paths["state"])
    publish_updates(paths["state"], paths["index"], scene["batch0"])
    assert os.path.exists(paths["state"])


# ---------------------------------------------------------------------------
# index publication
# ---------------------------------------------------------------------------

def test_index_replaced_byte_for_byte_at_completion(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], scene["batch0"])
    assert _read(paths["index"]) == scene["after0"]
    publish_updates(
        paths["state"], paths["index"], scene["batch0"] + scene["batch1"])
    assert _read(paths["index"]) == scene["after1"]


def test_unfinished_batch_leaves_index_at_before(scene, paths):
    _write(paths["index"], scene["after0"])
    publish_updates(
        paths["state"], paths["index"], scene["batch1"], limit=1)
    assert _read(paths["index"]) == scene["after0"]


# ---------------------------------------------------------------------------
# budgeted confirmation and re-entry
# ---------------------------------------------------------------------------

def test_limit_pauses_then_resumes(scene, paths):
    _write(paths["index"], scene["empty"])
    first = publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"], limit=2)
    document = json.loads(first)
    assert document["complete"] is False
    assert document["batches"][0][5] is True
    assert document["batches"][1][3:7] == [1, 2, False, None]
    assert _read(paths["index"]) == scene["after0"]

    second = publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"], limit=1)
    document = json.loads(second)
    assert document["complete"] is True
    assert document["batches"][1][3:7] == [2, 2, True, [0, 2, 2]]
    assert _read(paths["index"]) == scene["after1"]


def test_limit_zero_on_completed_is_byte_identical(scene, paths):
    _write(paths["index"], scene["empty"])
    batches = scene["batch0"] + scene["batch1"]
    first = publish_updates(paths["state"], paths["index"], batches)
    second = publish_updates(paths["state"], paths["index"], batches,
                             limit=0)
    assert second == first


def test_appending_a_new_batch_keeps_published_prefix(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], scene["batch0"])
    text = publish_updates(
        paths["state"], paths["index"], scene["batch0"] + scene["batch1"])
    assert json.loads(text)["complete"] is True


def test_records_must_be_input_prefix(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], scene["batch0"])
    # the same index cannot be driven by a different leading batch id
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], scene["batch1"])


def test_input_cannot_drop_a_recorded_batch(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"], limit=2)
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], scene["batch0"])


def test_recorded_active_batch_after_must_match_items(scene, paths):
    # b0 published, b1 in progress but its recorded after is wrong: on
    # re-entry the after is recomputed from the items and must agree.
    _write(paths["index"], scene["after0"])
    _write(paths["state"], _canonical({
        "batches": [
            ["b0", json.loads(scene["empty"]), json.loads(scene["after0"]),
             1, 1, True, [0, 1, 1]],
            ["b1", json.loads(scene["after0"]),
             json.loads(scene["after0"]), 1, 2, False, None]],
        "index": json.loads(scene["after0"]),
        "complete": False}))
    with pytest.raises(ValueError):
        publish_updates(
            paths["state"], paths["index"],
            scene["batch0"] + scene["batch1"])


# ---------------------------------------------------------------------------
# crash windows
# ---------------------------------------------------------------------------

def _ready_state(scene, disk_target):
    """State with b0 published and b1 fully confirmed but not marked so."""
    return _canonical({
        "batches": [
            ["b0", json.loads(scene["empty"]), json.loads(scene["after0"]),
             1, 1, True, [0, 1, 1]],
            ["b1", json.loads(scene["after0"]), json.loads(scene["after1"]),
             2, 2, False, None]],
        "index": json.loads(scene["after0"]),
        "complete": False})


def test_ready_state_with_before_index_publishes(scene, paths):
    _write(paths["index"], scene["after0"])
    _write(paths["state"], _ready_state(scene, scene["after0"]))
    text = publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"], limit=0)
    document = json.loads(text)
    assert document["complete"] is True
    assert document["batches"][1][5] is True
    assert _read(paths["index"]) == scene["after1"]


def test_ready_state_with_after_index_does_not_replace_again(scene, paths):
    _write(paths["index"], scene["after1"])
    _write(paths["state"], _ready_state(scene, scene["after1"]))
    text = publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"])
    document = json.loads(text)
    assert document["complete"] is True
    assert document["batches"][1][6] == [0, 2, 2]
    assert _read(paths["index"]) == scene["after1"]


def test_in_progress_state_resumes_at_confirmed_prefix(scene, paths):
    _write(paths["index"], scene["after0"])
    in_progress = _canonical({
        "batches": [
            ["b0", json.loads(scene["empty"]), json.loads(scene["after0"]),
             1, 1, True, [0, 1, 1]],
            ["b1", json.loads(scene["after0"]), json.loads(scene["after1"]),
             1, 2, False, None]],
        "index": json.loads(scene["after0"]),
        "complete": False})
    _write(paths["state"], in_progress)
    text = publish_updates(
        paths["state"], paths["index"],
        scene["batch0"] + scene["batch1"])
    document = json.loads(text)
    assert document["complete"] is True
    assert document["batches"][1][3] == 2
    assert _read(paths["index"]) == scene["after1"]


def test_completed_final_state_requires_matching_disk(scene, paths):
    _write(paths["index"], scene["after0"])
    finished = _canonical({
        "batches": [
            ["b0", json.loads(scene["empty"]), json.loads(scene["after0"]),
             1, 1, True, [0, 1, 1]]],
        "index": json.loads(scene["after0"]),
        "complete": True})
    _write(paths["state"], finished)
    # disk agrees with the terminal after, so re-entry is fine
    assert publish_updates(
        paths["state"], paths["index"], scene["batch0"]) == finished
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], scene["batch0"])


# ---------------------------------------------------------------------------
# replace-unfinished-last-entry batch
# ---------------------------------------------------------------------------

def test_batch_completing_an_unfinished_entry(scene, paths):
    partial = _index(((scene["p0"], scene["h0"]),
                      (scene["p12"], scene["partial12"])))
    _write(paths["index"], partial)
    text = publish_updates(
        paths["state"], paths["index"],
        (("b1", ((scene["p12"], scene["h12"]),)),))
    document = json.loads(text)
    assert document["complete"] is True
    assert document["batches"][0][3:7] == [1, 1, True, [0, 1, 1]]
    assert _read(paths["index"]) == scene["after1"]


# ---------------------------------------------------------------------------
# write failure preserves valid files
# ---------------------------------------------------------------------------

def test_index_write_failure_keeps_before_and_is_retryable(scene, tmp_path):
    read_only = tmp_path / "readonly"
    read_only.mkdir()
    index_path = str(read_only / "index.json")
    state_path = str(tmp_path / "state.json")
    _write(index_path, scene["empty"])
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.geteuid() == 0:
            pytest.skip("root bypasses directory write permissions")
        with pytest.raises(OSError):
            publish_updates(state_path, index_path, scene["batch0"])
        assert _read(index_path) == scene["empty"]
        state = json.loads(_read(state_path))
        assert state["batches"][0][5] is False
        read_only.chmod(stat.S_IRWXU)
        text = publish_updates(state_path, index_path, scene["batch0"])
        assert json.loads(text)["complete"] is True
        assert _read(index_path) == scene["after0"]
    finally:
        read_only.chmod(stat.S_IRWXU)


def test_state_write_failure_leaves_index_untouched(scene, tmp_path):
    read_only = tmp_path / "readonly"
    read_only.mkdir()
    state_path = str(read_only / "state.json")
    index_path = str(tmp_path / "index.json")
    _write(index_path, scene["empty"])
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.geteuid() == 0:
            pytest.skip("root bypasses directory write permissions")
        with pytest.raises(OSError):
            publish_updates(state_path, index_path, scene["batch0"])
        assert _read(index_path) == scene["empty"]
        assert not os.path.exists(state_path)
    finally:
        read_only.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_path_arguments_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["empty"])
    with pytest.raises(TypeError):
        publish_updates(bad, paths["index"], ())
    with pytest.raises(TypeError):
        publish_updates(paths["state"], bad, ())


@pytest.mark.parametrize("bad", [None, [], "x", 1])
def test_batches_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["empty"])
    with pytest.raises(TypeError):
        publish_updates(paths["state"], paths["index"], bad)


def test_batch_shape_and_member_types(scene, paths):
    _write(paths["index"], scene["empty"])
    for bad_batches in (
            (("b0",),),
            (("b0", (), 1),),
            (["b0", ()],),
            ((1, ()),),
            (("b0", []),),
            (("b0", ((1, scene["h0"]),)),),
            (("b0", ((scene["p0"], 1),)),)):
        with pytest.raises(TypeError):
            publish_updates(paths["state"], paths["index"], bad_batches)


@pytest.mark.parametrize("bad", [1.0, True, False, "x", []])
def test_limit_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["empty"])
    with pytest.raises(TypeError):
        publish_updates(paths["state"], paths["index"], (), bad)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_invalid_batch_ids_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    for bad_id in ("", "b/0", "b 0", "b:0", "x/y"):
        with pytest.raises(ValueError):
            publish_updates(paths["state"], paths["index"],
                            ((bad_id, ()),))


def test_duplicate_batch_ids_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(
            paths["state"], paths["index"],
            (("b0", ()), ("b0", ())))


def test_items_contract_violations_raise_value_error(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        (("b0", (("not json", scene["h0"]),)),))


def test_negative_limit_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], (), -1)


def test_empty_and_equal_paths_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates("", paths["index"], ())
    with pytest.raises(ValueError):
        publish_updates(paths["state"], "", ())
    with pytest.raises(ValueError):
        publish_updates(paths["index"], paths["index"], ())


def test_missing_index_raises_oserror(scene, paths):
    with pytest.raises(OSError):
        publish_updates(paths["state"], paths["index"], ())
    assert not os.path.exists(paths["state"])


def test_index_bad_utf8_raises_value_error(scene, paths):
    with open(paths["index"], "wb") as stream:
        stream.write(b"\xff\xfe")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_index_malformed_raises_value_error(scene, paths):
    _write(paths["index"], "{")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_state_bad_utf8_raises_value_error(scene, paths):
    _write(paths["index"], scene["empty"])
    with open(paths["state"], "wb") as stream:
        stream.write(b"\xff\xfe")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_state_malformed_raises_value_error(scene, paths):
    _write(paths["index"], scene["empty"])
    _write(paths["state"], "{")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_non_canonical_state_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    _write(paths["state"], json.dumps(
        {"batches": [], "index": json.loads(scene["empty"]),
         "complete": True}, indent=2))
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_state_index_must_match_disk_when_empty(scene, paths):
    _write(paths["index"], scene["empty"])
    _write(paths["state"], _canonical({
        "batches": [], "index": json.loads(scene["after0"]),
        "complete": True}))
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())
