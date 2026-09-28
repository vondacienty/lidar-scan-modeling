"""Tests for :func:`lidar_scan.publish_updates`."""

from __future__ import annotations

import json
import os

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
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


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    plan_zero = plan_checkout_recovery(log, None, ledgers[:1])
    plan_twelve = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    history_zero = _completed_history(plan_zero, 1)
    history_twelve = _completed_history(plan_twelve, 2)

    empty = update_recovery_index(None, ())
    after_zero = update_recovery_index(empty, ((plan_zero, history_zero),))
    after_all = update_recovery_index(
        after_zero, ((plan_twelve, history_twelve),))

    return {
        "empty": empty,
        "plan_zero": plan_zero,
        "plan_twelve": plan_twelve,
        "history_zero": history_zero,
        "history_twelve": history_twelve,
        "after_zero": after_zero,
        "after_all": after_all,
        "two_batches": (("b0", ((plan_zero, history_zero),)),
                        ("b1", ((plan_twelve, history_twelve),))),
        "one_multi_batch": (
            ("b0", ((plan_zero, history_zero),
                    (plan_twelve, history_twelve))),),
    }


@pytest.fixture
def paths(tmp_path):
    return {
        "state": str(tmp_path / "state.json"),
        "index": str(tmp_path / "index.json"),
    }


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

def test_completed_document_shape(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"])
    document = json.loads(text)
    assert list(document) == ["batches", "index", "complete"]
    assert document["complete"] is True
    assert len(document["batches"]) == 2
    for record in document["batches"]:
        assert len(record) == 7
        assert record[3] == record[4]
        assert record[5] is True
        assert isinstance(record[6], list) and len(record[6]) == 3
    assert document["batches"][0][0] == "b0"
    assert document["batches"][1][0] == "b1"
    assert document["batches"][0][6] == [0, 1, 1]
    assert document["batches"][1][6] == [0, 2, 2]
    assert _read(paths["index"]) == scene["after_all"]
    assert _canonical(document) == text
    assert " " not in text and not text.endswith("\n")


def test_index_field_embeds_published_index(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"])
    document = json.loads(text)
    assert _canonical(document["index"]) == scene["after_all"]


def test_multi_item_batch_receipt_counts_all_audit_rows(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["one_multi_batch"])
    record = json.loads(text)["batches"][0]
    assert (record[3], record[4], record[5]) == (2, 2, True)
    assert record[6] == [0, 3, 3]
    assert _read(paths["index"]) == scene["after_all"]


def test_empty_batches_publish_immediately_with_zero_step_receipt(
        scene, paths):
    _write(paths["index"], scene["empty"])
    batches = (("leading", ()), scene["two_batches"][0], ("trailing", ()))
    text = publish_updates(paths["state"], paths["index"], batches)
    document = json.loads(text)
    assert document["complete"] is True
    assert [record[5] for record in document["batches"]] == \
        [True, True, True]
    assert document["batches"][0][6] == [0, 0, 0]
    assert document["batches"][2][6] == [0, 0, 0]
    assert _read(paths["index"]) == scene["after_zero"]


# ---------------------------------------------------------------------------
# budgets and re-entry
# ---------------------------------------------------------------------------

def test_zero_limit_commits_initial_state_without_publishing(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"], limit=0)
    document = json.loads(text)
    assert document["complete"] is False
    assert len(document["batches"]) == 1
    record = document["batches"][0]
    assert (record[3], record[4], record[5]) == (0, 1, False)
    assert record[6] is None
    assert _canonical(document["index"]) == scene["empty"]
    assert _read(paths["index"]) == scene["empty"]


def test_budget_stops_between_batches_and_resumes(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"], limit=1)
    document = json.loads(text)
    assert document["complete"] is False
    assert [record[5] for record in document["batches"]] == [True, False]
    assert (document["batches"][1][3], document["batches"][1][4]) == (0, 1)
    assert _read(paths["index"]) == scene["after_zero"]
    assert _canonical(document["index"]) == scene["after_zero"]

    final = publish_updates(paths["state"], paths["index"],
                            scene["two_batches"])
    assert json.loads(final)["complete"] is True
    assert _read(paths["index"]) == scene["after_all"]


def test_mid_batch_interruption_resumes_without_republishing(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["one_multi_batch"], limit=1)
    record = json.loads(text)["batches"][0]
    assert (record[3], record[4], record[5]) == (1, 2, False)
    assert _read(paths["index"]) == scene["empty"]

    final = publish_updates(paths["state"], paths["index"],
                            scene["one_multi_batch"])
    record = json.loads(final)["batches"][0]
    assert (record[3], record[4], record[5]) == (2, 2, True)
    assert record[6] == [0, 3, 3]
    assert _read(paths["index"]) == scene["after_all"]


def test_zero_new_budget_reentry_leaves_state_unchanged(scene, paths):
    _write(paths["index"], scene["empty"])
    first = publish_updates(paths["state"], paths["index"],
                            scene["two_batches"], limit=1)
    second = publish_updates(paths["state"], paths["index"],
                             scene["two_batches"], limit=0)
    assert second == first
    assert _read(paths["index"]) == scene["after_zero"]


def test_completed_reentry_is_byte_identical(scene, paths):
    _write(paths["index"], scene["empty"])
    first = publish_updates(paths["state"], paths["index"],
                            scene["two_batches"])
    second = publish_updates(paths["state"], paths["index"],
                             scene["two_batches"])
    third = publish_updates(paths["state"], paths["index"],
                            scene["two_batches"], limit=0)
    assert second == first
    assert third == first


def test_batches_may_only_be_appended(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"],
                    scene["two_batches"][:1])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"])
    assert json.loads(text)["complete"] is True
    assert _read(paths["index"]) == scene["after_all"]


# ---------------------------------------------------------------------------
# crash window: index replaced, published state not written
# ---------------------------------------------------------------------------

def test_reentry_backfills_replaced_index_without_redoing(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"],
                    scene["two_batches"], limit=1)
    crash_records = [
        {"id": "b0", "before": scene["empty"], "after": scene["after_zero"],
         "confirmed": 1, "total": 1, "published": True,
         "receipt": (0, 1, 1)},
        {"id": "b1", "before": scene["after_zero"],
         "after": scene["after_all"], "confirmed": 1, "total": 1,
         "published": False, "receipt": None}]
    _write(paths["state"],
           tiles_module._format_publish_updates_state(
               crash_records, scene["after_zero"], False))
    _write(paths["index"], scene["after_all"])

    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"])
    document = json.loads(text)
    assert document["complete"] is True
    assert [record[5] for record in document["batches"]] == [True, True]
    assert _read(paths["index"]) == scene["after_all"]
    assert publish_updates(paths["state"], paths["index"],
                           scene["two_batches"]) == text


# ---------------------------------------------------------------------------
# state and index consistency
# ---------------------------------------------------------------------------

def test_divergent_index_file_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"],
                    scene["two_batches"], limit=1)
    _write(paths["index"], scene["empty"])  # roll the index back
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])


def test_state_batches_must_be_a_prefix_of_the_inputs(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"],
                    scene["two_batches"], limit=1)
    reordered = (("other", ((scene["plan_zero"], scene["history_zero"]),)),
                 scene["two_batches"][1])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], reordered)


def test_state_may_not_have_more_batches_than_inputs(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], scene["two_batches"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"][:1])


def test_changed_inputs_are_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"],
                    scene["two_batches"], limit=1)
    # The unfinished second batch is presented with different items.
    changed = (scene["two_batches"][0],
               ("b1", ((scene["plan_zero"], scene["history_zero"]),)))
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], changed)


@pytest.mark.parametrize("tampered", [
    "{not json",
    '{"batches":[],"index":{},"complete":false}',
])
def test_malformed_state_rejected(scene, paths, tampered):
    _write(paths["index"], scene["empty"])
    _write(paths["state"], tampered)
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])


def test_non_canonical_state_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"], limit=0)
    _write(paths["state"], json.dumps(json.loads(text), indent=2))
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"], limit=0)


# ---------------------------------------------------------------------------
# planning failures happen before any file is written
# ---------------------------------------------------------------------------

def test_invalid_later_batch_fails_before_any_publication(scene, paths):
    _write(paths["index"], scene["empty"])
    bad = (scene["two_batches"][0],
           ("b1", (("{", scene["history_twelve"]),)))
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], bad)
    assert _read(paths["index"]) == scene["empty"]
    assert not os.path.exists(paths["state"])


def test_non_matching_plan_history_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    # A history from one plan does not belong to a different plan.
    broken = (("b0", ((scene["plan_zero"], scene["history_twelve"]),)),)
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], broken)
    assert not os.path.exists(paths["state"])


# ---------------------------------------------------------------------------
# type and value validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_string_arguments_wrong_type(scene, paths, bad):
    state, index, batches = (
        paths["state"], paths["index"], scene["two_batches"])
    with pytest.raises(TypeError):
        publish_updates(bad, index, batches)
    with pytest.raises(TypeError):
        publish_updates(state, bad, batches)


@pytest.mark.parametrize("bad", [[], None, 1, "x"])
def test_batches_wrong_type(scene, paths, bad):
    with pytest.raises(TypeError):
        publish_updates(paths["state"], paths["index"], bad)


@pytest.mark.parametrize("bad", [
    ("b0",), ["b0", ()], (1, ()), (None, ()), ("b0", []),
    ("b0", ((1, "x"),)), ("b0", (("x", 1),)),
])
def test_batch_or_items_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["empty"])
    with pytest.raises(TypeError):
        publish_updates(paths["state"], paths["index"], (bad,))


@pytest.mark.parametrize("bad", [1.0, True, False, "x", 1.5])
def test_limit_wrong_type(scene, paths, bad):
    _write(paths["index"], scene["empty"])
    with pytest.raises(TypeError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"], limit=bad)


def test_negative_limit_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"], limit=-1)


@pytest.mark.parametrize("bad_id", ["", "bad id", "a/b", "x#y", "a:b"])
def test_invalid_batch_ids_rejected(scene, paths, bad_id):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        ((bad_id, ()),))


def test_duplicate_batch_ids_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        (("same", ()), ("same", ())))


def test_empty_paths_and_equal_paths_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    with pytest.raises(ValueError):
        publish_updates("", paths["index"], scene["two_batches"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], "", scene["two_batches"])
    with pytest.raises(ValueError):
        publish_updates(paths["index"], paths["index"],
                        scene["two_batches"])


def test_empty_batches_publish_completed_noop_state(scene, paths):
    _write(paths["index"], scene["empty"])
    text = publish_updates(paths["state"], paths["index"], ())
    document = json.loads(text)
    assert list(document) == ["batches", "index", "complete"]
    assert document["batches"] == []
    assert document["complete"] is True
    assert _canonical(document) == text
    assert " " not in text and not text.endswith("\n")
    # The no-op never replaces the index file.
    assert _read(paths["index"]) == scene["empty"]


def test_empty_batches_embed_the_current_index(scene, paths):
    _write(paths["index"], scene["after_zero"])
    text = publish_updates(paths["state"], paths["index"], ())
    assert _canonical(json.loads(text)["index"]) == scene["after_zero"]
    assert _read(paths["index"]) == scene["after_zero"]


def test_empty_batches_reentry_is_byte_identical(scene, paths):
    _write(paths["index"], scene["empty"])
    first = publish_updates(paths["state"], paths["index"], ())
    second = publish_updates(paths["state"], paths["index"], ())
    third = publish_updates(paths["state"], paths["index"], (), limit=0)
    assert second == first
    assert third == first
    assert _read(paths["index"]) == scene["empty"]


def test_empty_state_allows_later_batches_to_be_appended(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], ())
    text = publish_updates(paths["state"], paths["index"],
                           scene["two_batches"])
    assert json.loads(text)["complete"] is True
    assert _read(paths["index"]) == scene["after_all"]
    # Re-entering the completed publication stays byte identical.
    assert publish_updates(paths["state"], paths["index"],
                           scene["two_batches"]) == text


def test_empty_batches_against_non_empty_state_rejected(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], scene["two_batches"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_empty_batches_reject_a_diverged_index(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], ())
    _write(paths["index"], scene["after_zero"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())


def test_appending_to_empty_state_rejects_a_diverged_index(scene, paths):
    _write(paths["index"], scene["empty"])
    publish_updates(paths["state"], paths["index"], ())
    _write(paths["index"], scene["after_zero"])
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])


def test_empty_batches_validate_index_before_writing_state(scene, paths):
    _write(paths["index"], "{not json")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"], ())
    assert not os.path.exists(paths["state"])


# ---------------------------------------------------------------------------
# file errors
# ---------------------------------------------------------------------------

def test_missing_index_raises_oserror(scene, paths):
    with pytest.raises(OSError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])
    assert not os.path.exists(paths["state"])


def test_bad_utf8_index_raises_value_error(scene, paths):
    with open(paths["index"], "wb") as stream:
        stream.write(b"\xff\xfe")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])


def test_bad_utf8_state_raises_value_error(scene, paths):
    _write(paths["index"], scene["empty"])
    with open(paths["state"], "wb") as stream:
        stream.write(b"\xff\xfe")
    with pytest.raises(ValueError):
        publish_updates(paths["state"], paths["index"],
                        scene["two_batches"])
