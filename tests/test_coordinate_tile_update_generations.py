"""Tests for :func:`lidar_scan.coordinate_tile_update_generations`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import coordinate_tile_update_generations
from lidar_scan import tiles as tiles_module
import lidar_scan

from test_coordinate_tile_updates import (_PLAN_THREE, _PLAN_TWO, _make_job,
                                         _read, _write)


def _generation(root, gen_id, plan_two=_PLAN_TWO):
    directory = root / gen_id
    directory.mkdir(parents=True, exist_ok=True)
    jobs = (_make_job(directory, "alpha", _PLAN_THREE),
            _make_job(directory, "beta", plan_two))
    return (gen_id, str(directory / "manifest.json"), jobs)


def _top_path(root):
    return str(root / "generations.json")


def _all_paths(generations):
    paths = []
    for _id, manifest, jobs in generations:
        paths.append(manifest)
        paths.extend(path for job in jobs for path in job[1])
    return paths


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.coordinate_tile_update_generations is \
        coordinate_tile_update_generations
    assert "coordinate_tile_update_generations" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_path_contract(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(1, generations)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations("", generations)


def test_generations_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_top_path(tmp_path), [])


def test_generation_shape(tmp_path):
    generation = _generation(tmp_path, "g1")
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_top_path(tmp_path),
                                           (generation[:2],))
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_top_path(tmp_path),
                                           (generation + (generation,),))


def test_generation_id_contract(tmp_path):
    generation = _generation(tmp_path, "g1")
    bad = (1, generation[1], generation[2])
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_top_path(tmp_path), (bad,))
    empty = ("", generation[1], generation[2])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(_top_path(tmp_path), (empty,))
    duplicate = ("g1", generation[1], generation[2])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(_top_path(tmp_path),
                                           (generation, duplicate))


def test_generation_ids_must_strictly_increase(tmp_path):
    g1 = _generation(tmp_path, "g2")
    g2 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(_top_path(tmp_path), (g1, g2))


def test_all_paths_globally_pairwise_distinct(tmp_path):
    g1 = _generation(tmp_path, "g1")
    # The generations manifest path may not equal a generation manifest.
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(g1[1], (g1,))
    # ... nor a job L/A/S/P path.
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(g1[2][0][1][1], (g1,))
    # A generation manifest may not collide with another generation's
    # job paths.
    g2 = _generation(tmp_path, "g2")
    g2_bad = ("g2", g1[2][0][1][2], g2[2])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(_top_path(tmp_path),
                                           (g1, g2_bad))


def test_limit_and_verify_contract(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(path, generations, limit=1.0)
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(path, generations, limit=True)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, limit=-1)
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(path, generations, verify=0)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, limit=1,
                                           verify=True)


# ---------------------------------------------------------------------------
# publish flow
# ---------------------------------------------------------------------------

def test_fresh_publication_initializes_canonical_manifest(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)

    result = coordinate_tile_update_generations(path, generations)

    assert result == _read(path).decode("utf-8")
    raw = _read(path)
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)
    document = json.loads(result)
    assert list(document) == ["generations", "complete"]
    assert document["complete"] is True
    records = document["generations"]
    assert [record[0] for record in records] == ["g1", "g2"]
    for record, generation in zip(records, generations):
        assert record[1] == generation[1]
        assert record[2] == json.loads(_read(generation[1]).decode())
        assert record[2]["complete"] is True


def test_shared_budget_holds_later_generation_at_genesis(tmp_path):
    # g1's alpha needs three tasks; with a total budget of two it commits
    # two and g1 stays incomplete, so g2 never starts and registers no
    # publication at all.
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)

    result = coordinate_tile_update_generations(path, generations, limit=2)
    document = json.loads(result)
    assert document["complete"] is False
    # g1 holds a partial generation manifest, embedded as its state even
    # though it is incomplete; g2 never starts and stays null.
    sub = json.loads(_read(generations[0][1]).decode())
    assert sub["complete"] is False
    assert document["generations"][0][2] == sub
    assert document["generations"][1][2] is None
    assert not os.path.exists(generations[1][1])
    assert not os.path.exists(generations[1][2][0][1][1])

    # Resuming without a budget completes both generations in order.
    completed = coordinate_tile_update_generations(path, generations)
    document = json.loads(completed)
    assert document["complete"] is True
    assert all(record[2]["complete"] is True
               for record in document["generations"])


def test_later_generation_starts_only_when_previous_completes(tmp_path):
    # g1 has two two-task jobs; budget 3 leaves exactly one g1 task
    # unfinished, so g2 must not start even though one task of budget
    # would remain.
    g1 = _generation(tmp_path, "g1")
    g2 = _generation(tmp_path, "g2")
    coordinate_tile_update_generations(_top_path(tmp_path), (g1, g2),
                                       limit=3)
    assert not os.path.exists(g2[1])
    assert not os.path.exists(g2[2][0][1][1])
    assert os.path.exists(g1[1])
    assert json.loads(_read(g1[1]).decode())["complete"] is False


def test_re_entered_completion_is_byte_identical(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    completed = coordinate_tile_update_generations(path, generations)
    all_paths = [path] + _all_paths(generations)
    before = {candidate: _read(candidate) for candidate in all_paths}

    again = coordinate_tile_update_generations(path, generations)
    assert again == completed
    for candidate in all_paths:
        assert _read(candidate) == before[candidate]


def test_empty_generations_writes_empty_completed_manifest(tmp_path):
    path = _top_path(tmp_path)
    result = coordinate_tile_update_generations(path, ())
    assert result == '{"generations":[],"complete":true}'
    assert coordinate_tile_update_generations(path, (), verify=True) == \
        (0, 0, True)


# ---------------------------------------------------------------------------
# recovery and preflight
# ---------------------------------------------------------------------------

def test_lagging_top_manifest_is_backfilled(tmp_path):
    from lidar_scan import coordinate_tile_updates
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    # Finish g1 through the plural coordinator, leaving the top manifest
    # behind; verify refuses the trailing document and re-entry backfills.
    coordinate_tile_updates(generations[0][1], generations[0][2])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, verify=True)
    result = coordinate_tile_update_generations(path, generations)
    document = json.loads(result)
    assert document["complete"] is True
    assert coordinate_tile_update_generations(
        path, generations, verify=True) == (2, 2, True)


def test_tampered_generation_manifest_rejected_with_zero_writes(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    manifest = generations[0][1]
    document = json.loads(_read(manifest).decode())
    job_paths = document["jobs"][0][1]
    forged = ('{"jobs":[["alpha",['
              + ",".join(json.dumps(candidate) for candidate in job_paths)
              + '],99,true]],"complete":true}')
    snapshot = {candidate: _read(candidate)
                for candidate in [path, manifest, generations[1][1]]}
    _write(manifest, forged)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations)
    assert _read(path) == snapshot[path]
    assert _read(generations[1][1]) == snapshot[generations[1][1]]


def test_broken_generation_chain_rejected_before_any_write(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    # Tamper with g1 alpha's journal; g2 must never be rewritten.
    journal = generations[0][2][0][1][1]
    text = _read(journal).decode("utf-8")
    document = json.loads(text)
    document["records"][0][0] = 5
    _write(journal, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations)


def test_later_generation_registered_without_earlier_is_rejected(tmp_path):
    g1 = _generation(tmp_path, "g1")
    g2 = _generation(tmp_path, "g2")
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, (g1, g2))
    os.unlink(g1[1])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, (g1, g2))


# ---------------------------------------------------------------------------
# verify flow
# ---------------------------------------------------------------------------

def test_verify_returns_counts(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    assert coordinate_tile_update_generations(
        path, generations, verify=True) == (2, 0, False)
    coordinate_tile_update_generations(path, generations)
    assert coordinate_tile_update_generations(
        path, generations, verify=True) == (2, 2, True)


def test_verify_is_strictly_read_only(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    all_paths = [path] + _all_paths(generations)
    before = {candidate: _read(candidate) for candidate in all_paths}
    assert coordinate_tile_update_generations(
        path, generations, verify=True) == (1, 1, True)
    for candidate in all_paths:
        assert _read(candidate) == before[candidate]


def test_verify_requires_generations_manifest(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    with pytest.raises(OSError):
        coordinate_tile_update_generations(
            _top_path(tmp_path), generations, verify=True)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_writes_nothing(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    held = generations[0][2][1][1][0]
    open(held, "wb").close()
    holder = open(held, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_update_generations(path, generations)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    assert not os.path.exists(path)
    for _id, manifest, jobs in generations:
        assert not os.path.exists(manifest)
        for job in jobs:
            assert not os.path.exists(job[1][1])
            assert not os.path.exists(job[1][2])
