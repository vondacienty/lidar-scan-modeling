"""Tests for :func:`lidar_scan.publish_tile_update_plan`."""

from __future__ import annotations

import json
import os
import re

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        execute_tile_update_plan, publish_tile_update_plan)
from lidar_scan import tiles as tiles_module
import lidar_scan
import test_commit_tile_update_plan as commit_fixtures

# Reuse the prepared pyramids, plans and execution builder of the commit
# tests so the on-disk behaviour is exercised against the same scenes.
_PLAN_TWO = commit_fixtures._PLAN_TWO
_PLAN_TWO_LEVELS = commit_fixtures._PLAN_TWO_LEVELS
_PLAN_EMPTY = commit_fixtures._PLAN_EMPTY
_base_pyramid = commit_fixtures._base_pyramid
_new_pyramid = commit_fixtures._new_pyramid
_execution = commit_fixtures._execution


def _array_text(pyramid):
    text = tiles_module.encode_tile_pyramid(pyramid)
    return text[len('{"levels":'):-1]


def _pyramid_of(document):
    return tuple(tuple(tuple(tile) for tile in level)
                 for level in json.loads(document)["pyramid"])


def _write(path, text):
    with open(path, "wb" if isinstance(text, bytes) else "w",
              encoding=None if isinstance(text, bytes) else "utf-8") as stream:
        stream.write(text)


def _read(path):
    with open(path, "rb") as stream:
        return stream.read().decode("utf-8")


@pytest.fixture
def paths(tmp_path):
    return {
        "state": str(tmp_path / "state.json"),
        "pyramid": str(tmp_path / "pyramid.json"),
    }


@pytest.fixture
def scene():
    base = _base_pyramid()
    execution = _execution()
    return {
        "base": base,
        "execution": execution,
        "base_text": _array_text(base),
        "one_shot": commit_tile_update_plan(base, execution),
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.publish_tile_update_plan is publish_tile_update_plan
    assert "publish_tile_update_plan" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# fresh publications
# ---------------------------------------------------------------------------

def test_fresh_zero_step_initializes_state_and_keeps_pyramid(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    scene["base"], scene["execution"],
                                    max_tasks=0)
    assert _read(paths["state"]) == text
    assert _read(paths["pyramid"]) == scene["base_text"]
    document = json.loads(text)
    assert list(document) == ["plan", "base", "committed", "receipts",
                              "pyramid", "complete"]
    assert document["committed"] == 0
    assert document["receipts"] == []
    assert document["complete"] is False
    assert text == commit_tile_update_plan(
        scene["base"], scene["execution"], max_tasks=0)


def test_fresh_complete_run_writes_state_and_final_pyramid(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    scene["base"], scene["execution"])
    assert text == scene["one_shot"]
    assert _read(paths["state"]) == scene["one_shot"]
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(text))
    assert not text.endswith("\n")
    assert " " not in text and "\t" not in text and "\n" not in text


def test_pyramid_file_is_a_bare_canonical_layer_array(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    publish_tile_update_plan(paths["state"], paths["pyramid"],
                             scene["base"], scene["execution"])
    raw = _read(paths["pyramid"])
    assert raw[0] == "[" and raw[-1] == "]"
    assert "levels" not in raw
    assert not raw.startswith("﻿")
    for number in re.findall(r"-?\d+\.\d+", raw):
        assert re.fullmatch(r"-?\d+\.\d{6}", number)
    assert "-0.000000" not in raw


def test_max_tasks_chunks_the_publication(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    first = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                     scene["base"], scene["execution"],
                                     max_tasks=1)
    assert json.loads(first)["committed"] == 1
    assert _read(paths["state"]) == first
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(first))
    second = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                      scene["base"], scene["execution"])
    assert second == scene["one_shot"]
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(second))


def test_empty_completed_plan(paths):
    base = _base_pyramid()
    execution = execute_tile_update_plan(_PLAN_EMPTY, ())
    _write(paths["pyramid"], _array_text(base))
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    base, execution)
    document = json.loads(text)
    assert document["committed"] == 0
    assert document["complete"] is True
    assert _read(paths["pyramid"]) == _array_text(base)
    # Idempotent re-entry.
    assert publish_tile_update_plan(
        paths["state"], paths["pyramid"], base, execution) == text


# ---------------------------------------------------------------------------
# idempotency and re-entry
# ---------------------------------------------------------------------------

def test_zero_step_reentry_is_byte_identical(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    first = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                     scene["base"], scene["execution"],
                                     max_tasks=1)
    pyramid_before = _read(paths["pyramid"])
    again = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                     scene["base"], scene["execution"],
                                     max_tasks=0)
    assert again == first
    assert _read(paths["state"]) == first
    assert _read(paths["pyramid"]) == pyramid_before


def test_completed_reentry_is_byte_identical(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    completed = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                         scene["base"], scene["execution"])
    pyramid_before = _read(paths["pyramid"])
    assert publish_tile_update_plan(
        paths["state"], paths["pyramid"],
        scene["base"], scene["execution"]) == completed
    assert publish_tile_update_plan(
        paths["state"], paths["pyramid"],
        scene["base"], scene["execution"], max_tasks=0) == completed
    assert _read(paths["pyramid"]) == pyramid_before


def test_chunked_publication_matches_one_shot(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    pyramid = _new_pyramid()
    execution = execute_tile_update_plan(
        _PLAN_TWO, (("b0", pyramid),), max_tasks=0)
    state = None
    for _ in range(2):
        execution = execute_tile_update_plan(
            _PLAN_TWO, (("b0", pyramid),), state=execution, max_tasks=1)
        state = publish_tile_update_plan(
            paths["state"], paths["pyramid"],
            scene["base"], execution, max_tasks=1)
    assert state == scene["one_shot"]


# ---------------------------------------------------------------------------
# crash recovery: S leads P
# ---------------------------------------------------------------------------

def test_state_leading_pyramid_is_reconciled_for_free(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    # Simulate a crash after S reached committed=1 but before P was
    # replaced: S holds the prefix, P is still the base pyramid.
    _write(paths["state"], first)
    _write(paths["pyramid"], scene["base_text"])
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    scene["base"], scene["execution"],
                                    max_tasks=0)
    assert text == first
    assert _read(paths["state"]) == first
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(first))


def test_lagging_pyramid_is_backfilled_before_spending_budget(paths, scene):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    completed = scene["one_shot"]
    # S is fully committed, P is one prefix behind; max_tasks=0 must
    # still reconcile P and leave S at completion.
    _write(paths["state"], completed)
    _write(paths["pyramid"], _array_text(_pyramid_of(first)))
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    scene["base"], scene["execution"],
                                    max_tasks=0)
    assert text == completed
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(completed))


def test_pyramid_matching_base_with_advanced_state_is_a_lag(paths, scene):
    completed = scene["one_shot"]
    _write(paths["state"], completed)
    _write(paths["pyramid"], scene["base_text"])
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    scene["base"], scene["execution"])
    assert text == completed
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(completed))


# ---------------------------------------------------------------------------
# TypeError validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_state_path_wrong_type(paths, scene, bad):
    with pytest.raises(TypeError):
        publish_tile_update_plan(bad, paths["pyramid"],
                                 scene["base"], scene["execution"])


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_pyramid_path_wrong_type(paths, scene, bad):
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], bad,
                                 scene["base"], scene["execution"])


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_base_wrong_type(paths, scene, bad):
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 bad, scene["execution"])


@pytest.mark.parametrize("bad", [None, 1, [], b"x", {}])
def test_execution_wrong_type(paths, scene, bad):
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (), []])
def test_max_tasks_wrong_type(paths, scene, bad):
    with pytest.raises(TypeError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"],
                                 max_tasks=bad)


def test_base_internal_structure_is_value_error(paths, scene):
    _write(paths["pyramid"], scene["base_text"])
    execution = scene["execution"]
    for bad_base in (([()],),
                     (((0, 0, 0, 0, 255, 255, 1, 2.0, 1),),),
                     (((0, 0, 0, 0, 255, 255, 1.0, 2, 1),),),
                     (((True, 0, 0, 0, 255, 255, 1.0, 2.0, 1),),),
                     ()):
        with pytest.raises(ValueError):
            publish_tile_update_plan(paths["state"], paths["pyramid"],
                                     bad_base, execution)


# ---------------------------------------------------------------------------
# ValueError / OSError validation
# ---------------------------------------------------------------------------

def test_empty_and_equal_paths_rejected(paths, scene):
    with pytest.raises(ValueError):
        publish_tile_update_plan("", paths["pyramid"],
                                 scene["base"], scene["execution"])
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], "",
                                 scene["base"], scene["execution"])
    with pytest.raises(ValueError):
        publish_tile_update_plan("same", "same",
                                 scene["base"], scene["execution"])


def test_negative_max_tasks_rejected(paths, scene):
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"],
                                 max_tasks=-1)


def test_missing_pyramid_raises_oserror(paths, scene):
    with pytest.raises(OSError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])
    assert not os.path.exists(paths["state"])


@pytest.mark.parametrize("content", [
    b"\xff\xfe[]",
    b"",
    b"{not json",
    b"null",
    b"{}",
    b'{"levels":[]}',
    b" []",
    b"[]\n",
])
def test_malformed_pyramid_file_rejected(paths, scene, content):
    _write(paths["pyramid"], content)
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])
    assert not os.path.exists(paths["state"])


def test_pyramid_not_base_when_state_missing_rejected(paths, scene):
    other = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    _write(paths["pyramid"], _array_text(other))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


def test_pyramid_prefix_without_state_rejected(paths, scene):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    _write(paths["pyramid"], _array_text(_pyramid_of(first)))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


def test_pyramid_ahead_of_state_rejected(paths, scene):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    _write(paths["state"], first)
    _write(paths["pyramid"], _array_text(_pyramid_of(scene["one_shot"])))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"],
                                 max_tasks=0)
    # The ahead-of-state pyramid is left untouched.
    assert _read(paths["pyramid"]) == _array_text(
        _pyramid_of(scene["one_shot"]))


def test_unrelated_pyramid_rejected_against_state(paths, scene):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    _write(paths["state"], first)
    other = build_tile_pyramid(((10.0, 10.0, 5.0, 1, 1.0),), levels=1)
    _write(paths["pyramid"], _array_text(other))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


@pytest.mark.parametrize("content", [
    b"\xff\xfe[]",
    "{not json",
    "null",
    "[]",
    ('{"plan":{},"base":[],"committed":0,"receipts":[],'
     '"pyramid":[],"complete":false}'),
])
def test_malformed_existing_state_rejected(paths, scene, content):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    _write(paths["state"], content.encode() if isinstance(content, str)
           else content)
    _write(paths["pyramid"], _array_text(_pyramid_of(first)))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


def test_state_bound_to_other_base_rejected(paths, scene):
    other_base = build_tile_pyramid(
        ((999.0, 999.0, 1.0, 1, 1.0),), levels=1)
    other_state = commit_tile_update_plan(other_base, scene["execution"])
    _write(paths["state"], other_state)
    _write(paths["pyramid"], _array_text(other_base))
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


def test_non_canonical_state_rejected(paths, scene):
    first = commit_tile_update_plan(scene["base"], scene["execution"],
                                    max_tasks=1)
    _write(paths["state"], first + " ")
    _write(paths["pyramid"], scene["base_text"])
    with pytest.raises(ValueError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])


def test_failed_state_write_leaves_pyramid_in_place(paths, scene,
                                                    monkeypatch):
    _write(paths["pyramid"], scene["base_text"])
    real_write = tiles_module._atomic_write_json
    calls = []

    def failing_write(path, text, prefix=".publish-updates-"):
        calls.append(path)
        if path == paths["state"]:
            raise OSError("simulated state write failure")
        return real_write(path, text, prefix=prefix)

    monkeypatch.setattr(tiles_module, "_atomic_write_json", failing_write)
    with pytest.raises(OSError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"])
    assert calls and calls[0] == paths["state"]
    assert _read(paths["pyramid"]) == scene["base_text"]


def test_failed_pyramid_write_keeps_previous_pyramid(paths, scene,
                                                     monkeypatch):
    _write(paths["pyramid"], scene["base_text"])
    real_write = tiles_module._atomic_write_json

    def failing_write(path, text, prefix=".publish-updates-"):
        if path == paths["pyramid"]:
            raise OSError("simulated pyramid write failure")
        return real_write(path, text, prefix=prefix)

    monkeypatch.setattr(tiles_module, "_atomic_write_json", failing_write)
    with pytest.raises(OSError):
        publish_tile_update_plan(paths["state"], paths["pyramid"],
                                 scene["base"], scene["execution"],
                                 max_tasks=1)
    # P retains its previous content; S has already advanced and drives
    # the recovery on the next (successful) call.
    assert _read(paths["pyramid"]) == scene["base_text"]
    monkeypatch.undo()
    recovered = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                         scene["base"], scene["execution"])
    assert recovered == scene["one_shot"]
    assert _read(paths["pyramid"]) == _array_text(_pyramid_of(recovered))


# ---------------------------------------------------------------------------
# multi-level pyramids
# ---------------------------------------------------------------------------

def test_multi_level_backfill(paths):
    new_pyramid = build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),), levels=2)
    base = build_tile_pyramid(
        ((10.0, 10.0, 1.0, 1, 1.0),
         (100000.0, 100000.0, 7.0, 1, 1.0)), levels=2)
    execution = _execution(_PLAN_TWO_LEVELS, new_pyramid)
    _write(paths["pyramid"], _array_text(base))
    text = publish_tile_update_plan(paths["state"], paths["pyramid"],
                                    base, execution)
    document = json.loads(text)
    assert document["complete"] is True
    assert _read(paths["pyramid"]) == _array_text(
        tuple(tuple(tuple(tile) for tile in level)
              for level in document["pyramid"]))
