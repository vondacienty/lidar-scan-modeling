"""Tests for :func:`lidar_scan.plan_tile_update_generation_recovery`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import plan_tile_update_generation_recovery
from lidar_scan import coordinate_tile_update_generations
from lidar_scan import tiles as tiles_module
import lidar_scan

from test_coordinate_tile_update_generations import (_all_paths, _generation,
                                                     _top_path)
from test_coordinate_tile_updates import (_make_job, _read, _write)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.plan_tile_update_generation_recovery is \
        plan_tile_update_generation_recovery
    assert "plan_tile_update_generation_recovery" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation (the shared generations contract)
# ---------------------------------------------------------------------------

def test_path_contract(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    with pytest.raises(TypeError):
        plan_tile_update_generation_recovery(1, generations)
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery("", generations)


def test_generations_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), [])


def test_generation_shape(tmp_path):
    generation = _generation(tmp_path, "g1")
    with pytest.raises(TypeError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation[:2],))
    with pytest.raises(TypeError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation + (generation,),))


def test_generation_ids_must_strictly_increase(tmp_path):
    g1 = _generation(tmp_path, "g2")
    g2 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), (g1, g2))


# ---------------------------------------------------------------------------
# plan output
# ---------------------------------------------------------------------------

def test_empty_generations_plan_is_completed(tmp_path):
    result = plan_tile_update_generation_recovery(_top_path(tmp_path), ())
    assert result == '{"generations":[],"next":null,"complete":true}'


def test_unstarted_generations_plan_zero_confirmed_jobs(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    result = plan_tile_update_generation_recovery(_top_path(tmp_path),
                                                  generations)
    document = json.loads(result)
    assert list(document) == ["generations", "next", "complete"]
    assert document == {
        "generations": [
            ["g1", False,
             [["alpha", 0, 3], ["beta", 0, 2]], False],
            ["g2", False,
             [["alpha", 0, 3], ["beta", 0, 2]], False],
        ],
        "next": ["g1", "alpha"],
        "complete": False,
    }


def test_partial_publication_plans_first_open_job(tmp_path):
    # g1's alpha commits two of its three tasks under a budget of two;
    # beta is untouched and g2 never starts.
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)

    result = plan_tile_update_generation_recovery(path, generations)
    assert json.loads(result) == {
        "generations": [
            ["g1", True,
             [["alpha", 2, 3], ["beta", 0, 2]], False],
            ["g2", False,
             [["alpha", 0, 3], ["beta", 0, 2]], False],
        ],
        "next": ["g1", "alpha"],
        "complete": False,
    }

    # Once alpha is finished, the pointer moves on to beta in the same
    # generation; completing g1 moves the pointer into g2.
    coordinate_tile_update_generations(path, generations, limit=3)
    document = json.loads(plan_tile_update_generation_recovery(path,
                                                               generations))
    assert document["generations"][0][2] == \
        [["alpha", 3, 3], ["beta", 2, 2]]
    assert document["generations"][0][3] is True
    assert document["next"] == ["g2", "alpha"]


def test_completed_generations_plan_null_next(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)

    result = plan_tile_update_generation_recovery(path, generations)
    assert result == (
        '{"generations":['
        '["g1",true,[["alpha",3,3],["beta",2,2]],true],'
        '["g2",true,[["alpha",3,3],["beta",2,2]],true]'
        '],"next":null,"complete":true}')


def test_next_is_null_when_only_generation_level_reconciliation_remains():
    # A consistent snapshot can never be in this state, but the planner's
    # rule is explicit: an incomplete generation whose every job is at
    # its total needs no job pointer.
    rows = [{
        "id": "g1", "started": True,
        "jobs": (("alpha", 3, 3), ("beta", 2, 2)),
        "job_count": 2, "committed": 5, "complete": False,
    }]
    result = tiles_module._format_generation_recovery_plan(rows)
    assert json.loads(result)["next"] is None


def test_output_is_compact_canonical_json(tmp_path):
    generations = (_generation(tmp_path, "g一代"),)
    coordinate_tile_update_generations(_top_path(tmp_path), generations)
    result = plan_tile_update_generation_recovery(_top_path(tmp_path),
                                                  generations)
    raw = result.encode("utf-8")
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)
    assert "一代" in result
    assert "\\u" not in result


def test_plan_is_strictly_read_only(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    all_paths = [path] + _all_paths(generations)
    before = {candidate: _read(candidate) for candidate in all_paths
              if os.path.exists(candidate)}
    plan_tile_update_generation_recovery(path, generations)
    for candidate, content in before.items():
        assert _read(candidate) == content
    # The unstarted generation gains neither a manifest nor a journal or
    # state file (its empty locks pre-exist from the coordinator, which
    # opens every L up front).
    assert not os.path.exists(generations[1][1])
    for job in generations[1][2]:
        assert not os.path.exists(job[1][1])
        assert not os.path.exists(job[1][2])


def test_unstarted_plan_creates_no_file_at_all(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    plan_tile_update_generation_recovery(path, generations)
    assert not os.path.exists(path)
    for _id, manifest, jobs in generations:
        assert not os.path.exists(manifest)
        for job in jobs:
            for candidate in job[1][:3]:
                assert not os.path.exists(candidate)


# ---------------------------------------------------------------------------
# on-disk consistency (contract shared with the audit)
# ---------------------------------------------------------------------------

def test_missing_state_file_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(generations[0][2][0][1][2])
    with pytest.raises(OSError):
        plan_tile_update_generation_recovery(path, generations)


def test_broken_journal_chain_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    journal = generations[0][2][0][1][1]
    document = json.loads(_read(journal).decode("utf-8"))
    document["records"][0][0] = 5
    _write(journal, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(path, generations)


def test_orphan_journal_without_manifest_is_a_binding_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    _write(generations[0][2][0][1][1], '{"records":[]}')
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), generations)


def test_orphan_state_without_manifest_or_lock_is_a_binding_error(tmp_path):
    directory = tmp_path / "only"
    job = _make_job(directory, "solo")
    generation = ("g1", str(directory / "manifest.json"), (job,))
    _write(job[1][2], "{}")
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation,))


# ---------------------------------------------------------------------------
# read-only lock boundary
# ---------------------------------------------------------------------------

def test_started_generation_with_missing_lock_is_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    for job in generations[0][2]:
        os.unlink(job[1][0])
    with pytest.raises(OSError):
        plan_tile_update_generation_recovery(path, generations)
    # The missing locks are never recreated.
    for job in generations[0][2]:
        assert not os.path.exists(job[1][0])


def test_lock_contention_raises_without_writing(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    held = generations[0][2][1][1][0]
    open(held, "wb").close()
    holder = open(held, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            plan_tile_update_generation_recovery(path, generations)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    assert not os.path.exists(path)
    for _id, manifest, jobs in generations:
        assert not os.path.exists(manifest)
        for job in jobs:
            assert not os.path.exists(job[1][1])
            assert not os.path.exists(job[1][2])


def _one_job_generation(root, gen_id="g1", job_id="solo"):
    directory = root / gen_id
    job = _make_job(directory, job_id)
    return (gen_id, str(directory / "manifest.json"), (job,)), job


def test_lock_appearing_during_read_is_concurrent_start(tmp_path,
                                                        monkeypatch):
    generation, job = _one_job_generation(tmp_path)
    lock_path = job[1][0]
    original_exists = os.path.exists
    calls = {"exists": 0}

    def watching_exists(candidate):
        if candidate == lock_path:
            calls["exists"] += 1
            # The boundary opens L directly; the read-side retest is the
            # first existence test and finds a lock just created by
            # another process.
            if calls["exists"] == 1:
                open(lock_path, "wb").close()
        return original_exists(candidate)

    monkeypatch.setattr(tiles_module.os.path, "exists", watching_exists)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation,))


def test_journal_appearing_during_read_is_concurrent_start(tmp_path,
                                                           monkeypatch):
    generation, job = _one_job_generation(tmp_path)
    journal_path = job[1][1]
    original_exists = os.path.exists
    calls = {"exists": 0}

    def watching_exists(candidate):
        if candidate == journal_path:
            calls["exists"] += 1
            if calls["exists"] == 2:
                _write(journal_path, '{"records":[]}')
        return original_exists(candidate)

    monkeypatch.setattr(tiles_module.os.path, "exists", watching_exists)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation,))


def test_state_appearing_during_read_is_concurrent_start(tmp_path,
                                                         monkeypatch):
    generation, job = _one_job_generation(tmp_path)
    state_path = job[1][2]
    original_exists = os.path.exists
    calls = {"exists": 0}

    def watching_exists(candidate):
        if candidate == state_path:
            calls["exists"] += 1
            if calls["exists"] == 2:
                _write(state_path, "{}")
        return original_exists(candidate)

    monkeypatch.setattr(tiles_module.os.path, "exists", watching_exists)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation,))


def test_manifest_appearing_during_read_is_concurrent_start(tmp_path,
                                                            monkeypatch):
    generation, job = _one_job_generation(tmp_path)
    manifest_path = generation[1]
    original_read = tiles_module._read_utf8_file

    def manifest_appears(candidate, label):
        if candidate == manifest_path:
            _write(manifest_path, "{}")
        return original_read(candidate, label)

    monkeypatch.setattr(tiles_module, "_read_utf8_file", manifest_appears)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation,))
