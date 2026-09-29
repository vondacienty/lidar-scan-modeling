"""Tests for :func:`lidar_scan.coordinate_tile_update_generations`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        coordinate_tile_update,
                        coordinate_tile_update_generations,
                        coordinate_tile_updates)
from lidar_scan import tiles as tiles_module
import lidar_scan

_PLAN_TWO = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,2,null]]],'
    '[0,512,512,767,767,[["b0",1,2,null]]]],"complete":false}')

_PLAN_THREE = (
    '{"tasks":[[0,0,0,255,255,[["b0",1,3,null]]],'
    '[0,512,512,767,767,[["b0",1,3,null]]],'
    '[0,1024,1024,1279,1279,[["b0",1,3,null]]]],"complete":false}')


def _new_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 5.0, 1, 1.0),
         (600.0, 600.0, 9.0, 1, 1.0),
         (1100.0, 1100.0, 4.0, 1, 1.0)), levels=1)


def _base_pyramid():
    return build_tile_pyramid(
        ((10.0, 10.0, 1.0, 1, 1.0),
         (600.0, 600.0, 2.0, 1, 1.0),
         (1100.0, 1100.0, 3.0, 1, 1.0)), levels=1)


def _write(path, text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(text)


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


def _job_paths(root, job_id):
    directory = root / job_id
    directory.mkdir(parents=True, exist_ok=True)
    base = str(directory)
    return (os.path.join(base, "lock"),
            os.path.join(base, "journal.json"),
            os.path.join(base, "state.json"),
            os.path.join(base, "pyramid.json"))


def _make_job(root, job_id, plan=_PLAN_TWO):
    from lidar_scan import execute_tile_update_plan
    paths = _job_paths(root, job_id)
    base = _base_pyramid()
    pyramid = _new_pyramid()
    sources = sorted({source[0]
                      for task in json.loads(plan)["tasks"]
                      for source in task[5]})
    updates = tuple((source_id, pyramid) for source_id in sources)
    execution = execute_tile_update_plan(plan, updates)
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    node = tiles_module._parse_json_node(
        zero_state, tiles_module._skip_json_ws(zero_state, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    _write(paths[3], zero_state[start:end])
    return (job_id, paths, base, execution)


def _generation(root, generation_id, job_id, plan=_PLAN_TWO):
    directory = root / generation_id
    directory.mkdir(parents=True, exist_ok=True)
    job = _make_job(directory, job_id, plan)
    manifest = str(directory / "manifest.json")
    return (generation_id, manifest, (job,)), manifest, job


def _generations_path(root):
    return str(root / "generations.json")


def _two_generations(root, first_plan=_PLAN_THREE, second_plan=_PLAN_TWO):
    first, manifest0, job0 = _generation(root, "g0", "alpha", first_plan)
    second, manifest1, job1 = _generation(root, "g1", "beta", second_plan)
    return (first, second), (manifest0, manifest1), (job0, job1)


def _parse(text):
    return json.loads(text)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert (tiles_module.coordinate_tile_update_generations
            is coordinate_tile_update_generations)
    assert "coordinate_tile_update_generations" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_path_must_be_non_empty_str(tmp_path):
    generations, _manifests, _jobs = _two_generations(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(1, generations)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations("", generations)


def test_generations_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_generations_path(tmp_path), [])


def test_generation_item_shape(tmp_path):
    first, _manifest, _job = _generation(tmp_path, "g0", "alpha")
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_generations_path(tmp_path),
                                           (first[0],))
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(_generations_path(tmp_path),
                                           (first + (first,),))


def test_generation_id_contract(tmp_path):
    first, manifest, _job = _generation(tmp_path, "g0", "alpha")
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), ((1, manifest, ()),))
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (("", manifest, ()),))


def test_generation_ids_must_strictly_increase(tmp_path):
    first, manifest0, _j0 = _generation(tmp_path, "g1", "alpha")
    second, manifest1, _j1 = _generation(tmp_path, "g0", "beta")
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (first, second))
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (first, first))


def test_generation_manifest_contract(tmp_path):
    first, _manifest, _job = _generation(tmp_path, "g0", "alpha")
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (("g0", 1, ()),))
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (("g0", "", ()),))


def test_generation_jobs_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (("g0", "/tmp/m.json", []),))


def test_paths_must_be_globally_distinct(tmp_path):
    generations, manifests, _jobs = _two_generations(tmp_path)
    # The generations document itself may not equal a sub-manifest path.
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(manifests[0], generations)
    # Two generations may not share a sub-manifest path.
    first = generations[0]
    shared_second = ("g1", manifests[0], generations[1][2])
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (first, shared_second))
    # A sub-manifest may not equal one of its job paths.
    job = generations[0][2][0]
    colliding = ("g0", job[1][1], (job,))
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), (colliding,))


def test_paths_must_be_distinct_across_generations(tmp_path):
    # The same L/A/S/P path may not be reused by a later generation.
    first, manifest0, job0 = _generation(tmp_path, "g0", "alpha")
    second_dir = tmp_path / "g1"
    second_dir.mkdir(parents=True, exist_ok=True)
    manifest1 = str(second_dir / "manifest.json")
    reused_job = (job0[0],) + (job0[1],) + job0[2:]
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path),
            (first, ("g1", manifest1, (reused_job,))))


def test_limit_and_verify_contract(tmp_path):
    generations, _manifests, _jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
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

def test_fresh_publication_writes_canonical_document(tmp_path):
    generations, _manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)

    result = coordinate_tile_update_generations(path, generations)

    assert result == _read(path).decode("utf-8")
    raw = _read(path)
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)
    document = _parse(result)
    assert list(document) == ["generations", "complete"]
    assert document["complete"] is True
    records = document["generations"]
    assert [record[0] for record in records] == ["g0", "g1"]
    for record, generation in zip(records, generations):
        assert record[1] == generation[1]
        assert isinstance(record[2], dict)
        assert list(record[2]) == ["jobs", "complete"]
        assert record[2]["complete"] is True


def test_later_generation_does_not_start_before_predecessor_completes(
        tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)

    # g0 needs three tasks; two leave it incomplete so g1 never starts.
    result = coordinate_tile_update_generations(path, generations, limit=2)
    document = _parse(result)

    assert document["complete"] is False
    assert document["generations"][0][2]["complete"] is False
    assert document["generations"][1][2] is None
    assert not os.path.exists(manifests[1])
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 0, False)


def test_shared_budget_serializes_generations(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)

    # Two tasks leave g0 at 2/3 incomplete; g1 untouched.
    coordinate_tile_update_generations(path, generations, limit=2)
    # One more task completes g0; g1 may then register genesis but the
    # budget is spent, so it commits nothing.
    document = _parse(
        coordinate_tile_update_generations(path, generations, limit=1))
    assert document["generations"][0][2]["complete"] is True
    assert document["generations"][1][2] is not None
    assert document["generations"][1][2]["complete"] is False
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 1, False)

    # Resuming without a budget completes g1.
    document = _parse(coordinate_tile_update_generations(path, generations))
    assert document["complete"] is True
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 2, True)


def test_re_entered_completion_is_byte_identical(tmp_path):
    generations, _manifests, _jobs = _two_generations(
        tmp_path, _PLAN_TWO, _PLAN_THREE)
    path = _generations_path(tmp_path)
    completed = coordinate_tile_update_generations(path, generations)
    all_paths = [path]
    for _id, manifest, gen_jobs in generations:
        all_paths.append(manifest)
        for job in gen_jobs:
            all_paths.extend(job[1])
    before = {target: _read(target) for target in all_paths}

    again = coordinate_tile_update_generations(path, generations)
    assert again == completed
    for target in all_paths:
        assert _read(target) == before[target]


def test_empty_generations_writes_empty_completed_document(tmp_path):
    path = _generations_path(tmp_path)
    result = coordinate_tile_update_generations(path, ())
    assert result == '{"generations":[],"complete":true}'
    assert coordinate_tile_update_generations(path, (), verify=True) \
        == (0, 0, True)


def test_lagging_document_is_backfilled(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    partial = coordinate_tile_update_generations(path, generations, limit=2)
    assert _parse(partial)["complete"] is False

    # Advance the first generation's jobs through lower-level
    # coordinators, leaving the generations document behind, then resume.
    for _id, _manifest, gen_jobs in generations[:1]:
        for _job_id, job_paths, base, execution in gen_jobs:
            coordinate_tile_update(job_paths, base, execution)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, verify=True)

    result = coordinate_tile_update_generations(path, generations)
    assert _parse(result)["complete"] is True
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 2, True)


# ---------------------------------------------------------------------------
# rejection and zero-write guarantees
# ---------------------------------------------------------------------------

def test_document_leading_a_sub_manifest_is_rejected_without_writes(
        tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)

    # Forge the recorded g0 terminal sequence ahead of the on-disk
    # sub-manifest (g0's alpha job actually sits at n == 1).
    paths = jobs[0][1]
    leading = ('{"jobs":[["alpha",['
               + ",".join(json.dumps(p) for p in paths)
               + '],9,true]],"complete":true}')
    document = _read(path).decode("utf-8")
    on_disk = _read(manifests[0]).decode("utf-8")
    forged_document = document.replace(on_disk, leading)
    assert forged_document != document
    _write(path, forged_document)

    all_paths = [path] + list(manifests)
    for job in jobs:
        all_paths.extend(job[1])
    before = {target: _read(target) for target in all_paths}
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, verify=True)
    for target in all_paths:
        assert _read(target) == before[target]


def test_tampered_sub_manifest_on_disk_is_rejected(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    _write(manifests[0], _read(manifests[0]).decode("utf-8") + "  ")
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, verify=True)


def test_misbound_sub_manifest_path_is_rejected(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    # Rewrite the generations document binding g0 to g1's manifest.
    document = _read(path).decode("utf-8")
    rebound = document.replace(json.dumps(manifests[0]),
                               json.dumps(manifests[1]), 1)
    _write(path, rebound)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations)


def test_recorded_sub_manifest_missing_is_os_error(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(manifests[0])
    with pytest.raises(OSError):
        coordinate_tile_update_generations(path, generations)
    with pytest.raises(OSError):
        coordinate_tile_update_generations(path, generations, verify=True)


# ---------------------------------------------------------------------------
# verify flow
# ---------------------------------------------------------------------------

def test_verify_returns_counts(tmp_path):
    generations, _manifests, _jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 0, False)
    coordinate_tile_update_generations(path, generations)
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 2, True)


def test_verify_is_strictly_read_only(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    all_paths = [path] + list(manifests)
    for job in jobs:
        all_paths.extend(job[1])
    before = {target: _read(target) for target in all_paths}
    assert coordinate_tile_update_generations(path, generations,
                                              verify=True) == (2, 2, True)
    for target in all_paths:
        assert _read(target) == before[target]


def test_verify_requires_generations_document(tmp_path):
    generations, _manifests, _jobs = _two_generations(tmp_path)
    with pytest.raises(OSError):
        coordinate_tile_update_generations(
            _generations_path(tmp_path), generations, verify=True)


def test_verify_requires_recorded_job_files(tmp_path):
    generations, _manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(jobs[0][1][2])  # S of the first generation's job
    with pytest.raises(OSError):
        coordinate_tile_update_generations(path, generations, verify=True)


def test_verify_rejects_generation_started_before_predecessor(tmp_path):
    generations, manifests, jobs = _two_generations(
        tmp_path, _PLAN_TWO, _PLAN_TWO)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    # Hand-forge a document whose g0 is incomplete but whose g1 already
    # carries a sub-manifest: an illegal ordering.
    g0 = ('{"jobs":[["alpha",['
          + ",".join(json.dumps(p) for p in jobs[0][1])
          + '],1,false]],"complete":false}')
    g1 = _read(manifests[1]).decode("utf-8")
    forged = ('{"generations":[["g0",' + json.dumps(manifests[0]) + ","
              + g0 + '],["g1",' + json.dumps(manifests[1]) + "," + g1
              + ']],"complete":false}')
    _write(path, forged)
    with pytest.raises(ValueError):
        coordinate_tile_update_generations(path, generations, verify=True)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_writes_nothing(tmp_path):
    generations, manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    for job in jobs:
        open(job[1][0], "wb").close()
    held_path = jobs[0][1][0]
    holder = open(held_path, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_update_generations(path, generations)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    assert not os.path.exists(path)
    for manifest in manifests:
        assert not os.path.exists(manifest)
    for job in jobs:
        assert not os.path.exists(job[1][1])  # A
        assert not os.path.exists(job[1][2])  # S


def test_lock_contention_blocks_verify_too(tmp_path):
    generations, _manifests, jobs = _two_generations(tmp_path)
    path = _generations_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    holder = open(jobs[0][1][0], "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_update_generations(path, generations,
                                               verify=True)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
