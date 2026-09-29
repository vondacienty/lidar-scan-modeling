"""Tests for :func:`lidar_scan.coordinate_tile_updates`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        coordinate_tile_update, coordinate_tile_updates)
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
    # Seed P with the embedded base pyramid.
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    node = tiles_module._parse_json_node(
        zero_state, tiles_module._skip_json_ws(zero_state, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    _write(paths[3], zero_state[start:end])
    return (job_id, paths, base, execution)


def _manifest_path(root):
    return str(root / "manifest.json")


def _state_committed(paths):
    with open(paths[2], "rb") as stream:
        node = tiles_module._parse_json_node(
            stream.read().decode("utf-8"), 0)
    return node[0]["committed"][0]


def _parse_manifest(text):
    return json.loads(text)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.coordinate_tile_updates is coordinate_tile_updates
    assert "coordinate_tile_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_manifest_must_be_non_empty_str(tmp_path):
    job = _make_job(tmp_path, "a")
    with pytest.raises(TypeError):
        coordinate_tile_updates(1, (job,))
    with pytest.raises(ValueError):
        coordinate_tile_updates("", (job,))


def test_jobs_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path), [])


def test_job_item_shape(tmp_path):
    job = _make_job(tmp_path, "a")
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path), (["a"],))
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                (job[:3],))
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                (job + (job,),))


def test_job_id_contract(tmp_path):
    job = _make_job(tmp_path, "a")
    bad_id = (1, job[1], job[2], job[3])
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path), (bad_id,))
    empty_id = ("", job[1], job[2], job[3])
    with pytest.raises(ValueError):
        coordinate_tile_updates(_manifest_path(tmp_path), (empty_id,))
    with pytest.raises(ValueError):
        coordinate_tile_updates(_manifest_path(tmp_path), (job, job))


def test_job_paths_contract(tmp_path):
    job = _make_job(tmp_path, "a")
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                ((job[0], list(job[1]), job[2], job[3]),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                ((job[0], job[1][:3], job[2], job[3]),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(
            _manifest_path(tmp_path),
            ((job[0], (1, job[1][1], job[1][2], job[1][3]),
              job[2], job[3]),))
    with pytest.raises(ValueError):
        coordinate_tile_updates(
            _manifest_path(tmp_path),
            ((job[0], ("",) + job[1][1:], job[2], job[3]),))


def test_paths_must_be_pairwise_distinct_across_jobs(tmp_path):
    a = _make_job(tmp_path, "a")
    b = _make_job(tmp_path, "b")
    shared_lock = (b[0], (a[1][0],) + b[1][1:], b[2], b[3])
    with pytest.raises(ValueError):
        coordinate_tile_updates(_manifest_path(tmp_path), (a, shared_lock))


def test_manifest_path_must_differ_from_every_job_path(tmp_path):
    job = _make_job(tmp_path, "a")
    for position in range(4):
        with pytest.raises(ValueError):
            coordinate_tile_updates(job[1][position], (job,))


def test_base_and_execution_types(tmp_path):
    job = _make_job(tmp_path, "a")
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                ((job[0], job[1], [], job[3]),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(_manifest_path(tmp_path),
                                ((job[0], job[1], job[2], 1),))


def test_limit_and_verify_contract(tmp_path):
    job = _make_job(tmp_path, "a")
    manifest = _manifest_path(tmp_path)
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (job,), limit=1.0)
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (job,), limit=True)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, (job,), limit=-1)
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (job,), verify=0)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, (job,), limit=1, verify=True)


# ---------------------------------------------------------------------------
# publish flow
# ---------------------------------------------------------------------------

def test_fresh_publication_initializes_canonical_manifest(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)

    result = coordinate_tile_updates(manifest, jobs)

    assert result == _read(manifest).decode("utf-8")
    raw = _read(manifest)
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)
    document = _parse_manifest(result)
    assert list(document) == ["jobs", "complete"]
    assert document["complete"] is True
    records = document["jobs"]
    # Records are sorted by id regardless of the input jobs order.
    assert [record[0] for record in records] == ["alpha", "beta"]
    by_id = {record[0]: record for record in records}
    for job in jobs:
        job_id, paths, _base, _execution = job
        record = by_id[job_id]
        assert record[1] == list(paths)
        assert record[3] is True
    assert by_id["alpha"][2] == 1
    assert by_id["beta"][2] == 1


def test_shared_budget_is_consumed_in_jobs_order(tmp_path):
    # alpha needs three tasks; with a total budget of two it commits two
    # and beta (processed second) only registers its genesis record.
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)

    result = coordinate_tile_updates(manifest, jobs, limit=2)

    document = _parse_manifest(result)
    by_id = {record[0]: record for record in document["jobs"]}
    # n is the journal sequence: genesis 0 plus one advancing record.
    assert by_id["alpha"][2] == 1
    assert by_id["alpha"][3] is False
    assert by_id["beta"][2] == 0
    assert by_id["beta"][3] is False
    assert document["complete"] is False
    assert _state_committed(jobs[0][1]) == 2
    assert _state_committed(jobs[1][1]) == 0

    # Resuming without a budget completes both jobs.
    completed = coordinate_tile_updates(manifest, jobs)
    document = _parse_manifest(completed)
    assert document["complete"] is True
    assert {record[0]: record[2] for record in document["jobs"]} == {
        "alpha": 2, "beta": 1}
    assert _state_committed(jobs[0][1]) == 3
    assert _state_committed(jobs[1][1]) == 2


def test_processing_order_is_by_id_regardless_of_jobs_order(tmp_path):
    # beta is listed first, but jobs are processed in ascending id order
    # so alpha spends the whole budget and beta gets none.
    jobs = (_make_job(tmp_path, "beta", _PLAN_THREE),
            _make_job(tmp_path, "alpha", _PLAN_THREE))
    document = _parse_manifest(
        coordinate_tile_updates(_manifest_path(tmp_path), jobs, limit=2))
    by_id = {record[0]: record for record in document["jobs"]}
    assert by_id["alpha"][2] == 1
    assert by_id["beta"][2] == 0
    assert _state_committed(jobs[1][1]) == 2
    assert _state_committed(jobs[0][1]) == 0
    assert [record[0] for record in document["jobs"]] == ["alpha", "beta"]


def test_re_entered_completion_is_byte_identical(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),
            _make_job(tmp_path, "beta", _PLAN_THREE))
    manifest = _manifest_path(tmp_path)
    completed = coordinate_tile_updates(manifest, jobs)
    all_paths = [manifest] + [path for job in jobs for path in job[1]]
    before = {path: _read(path) for path in all_paths}

    again = coordinate_tile_updates(manifest, jobs)
    assert again == completed
    for path in all_paths:
        assert _read(path) == before[path]


def test_empty_jobs_writes_empty_completed_manifest(tmp_path):
    manifest = _manifest_path(tmp_path)
    result = coordinate_tile_updates(manifest, ())
    assert result == '{"jobs":[],"complete":true}'
    assert coordinate_tile_updates(manifest, (), verify=True) == (0, 0, True)


def test_missing_pyramid_is_os_error(tmp_path):
    job = _make_job(tmp_path, "alpha")
    os.unlink(job[1][3])
    with pytest.raises(OSError):
        coordinate_tile_updates(_manifest_path(tmp_path), (job,))


# ---------------------------------------------------------------------------
# manifest recovery
# ---------------------------------------------------------------------------

def test_lagging_manifest_is_backfilled(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)
    partial = coordinate_tile_updates(manifest, jobs, limit=1)
    assert _parse_manifest(partial)["complete"] is False

    # Advance every job through the singular coordinator, leaving the
    # manifest behind, then verify refuses the trailing manifest and
    # re-entering the plural coordinator backfills it.
    for _id, paths, base, execution in jobs:
        coordinate_tile_update(paths, base, execution)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, verify=True)

    result = coordinate_tile_updates(manifest, jobs)
    document = _parse_manifest(result)
    assert document["complete"] is True
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 2, True)


def test_manifest_leading_a_journal_is_rejected(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    # Forge a structurally canonical manifest whose n is ahead.
    paths = jobs[0][1]
    forged = ('{"jobs":[["alpha",['
              + ",".join(json.dumps(p) for p in paths)
              + '],5,false]],"complete":false}')
    _write(manifest, forged)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs)


def test_manifest_with_tampered_paths_is_rejected(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    document = _parse_manifest(_read(manifest).decode("utf-8"))
    paths = list(document["jobs"][0][1])
    paths[2] = os.path.join(str(tmp_path), "alpha", "elsewhere.json")
    forged = ('{"jobs":[["alpha",['
              + ",".join(json.dumps(p) for p in paths)
              + '],1,true]],"complete":true}')
    _write(manifest, forged)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs)


def test_manifest_with_tampered_complete_flag_is_rejected(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs, limit=1)
    document = _parse_manifest(_read(manifest).decode("utf-8"))
    paths = document["jobs"][0][1]
    forged = ('{"jobs":[["alpha",['
              + ",".join(json.dumps(p) for p in paths)
              + '],1,true]],"complete":false}')
    _write(manifest, forged)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs)


def test_preflight_failure_writes_nothing(tmp_path):
    # beta's journal is corrupted while alpha could still spend the
    # budget; the preflight under both locks must abort before alpha or
    # the manifest is touched.
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs, limit=1)
    forged_journal = '{"records":[[5,' + \
        commit_tile_update_plan(jobs[1][2], jobs[1][3], max_tasks=0) + ']]}'
    _write(jobs[1][1][1], forged_journal)
    snapshot = {path: _read(path)
                for path in [manifest] + [p for job in jobs for p in job[1]]}
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, limit=5)
    for path, data in snapshot.items():
        assert _read(path) == data


# ---------------------------------------------------------------------------
# verify flow
# ---------------------------------------------------------------------------

def test_verify_returns_counts(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_THREE),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs, limit=1)
    # alpha committed one (incomplete); beta only has genesis, incomplete.
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 0, False)

    coordinate_tile_updates(manifest, jobs)
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 2, True)


def test_verify_is_strictly_read_only(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    all_paths = [manifest] + [path for job in jobs for path in job[1]]
    before = {path: _read(path) for path in all_paths}
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (1, 1, True)
    for path in all_paths:
        assert _read(path) == before[path]


def test_verify_requires_manifest(tmp_path):
    job = _make_job(tmp_path, "alpha")
    with pytest.raises(OSError):
        coordinate_tile_updates(_manifest_path(tmp_path), (job,),
                                verify=True)


def test_verify_requires_every_job_file(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    os.unlink(jobs[1][1][2])
    with pytest.raises(OSError):
        coordinate_tile_updates(manifest, jobs, verify=True)


def test_verify_rejects_tampered_state(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    _id, paths, base, execution = jobs[0]
    _write(paths[2], commit_tile_update_plan(base, execution, max_tasks=0))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, verify=True)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_releases_acquired_locks_and_writes_nothing(
        tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),
            _make_job(tmp_path, "beta", _PLAN_TWO))
    manifest = _manifest_path(tmp_path)
    # The coordinator creates lock files on first run; pre-create them so
    # the external holder can flock the one that sorts second (beta).
    for job in jobs:
        open(job[1][0], "wb").close()
    held_path = jobs[1][1][0]
    holder = open(held_path, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_updates(manifest, jobs)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    # No payload file was written: the manifest, journals and states are
    # all absent (only the pre-seeded pyramids and the empty locks exist).
    assert not os.path.exists(manifest)
    for job in jobs:
        assert not os.path.exists(job[1][1])  # A
        assert not os.path.exists(job[1][2])  # S
        assert os.path.exists(job[1][3])      # P was seeded beforehand
    # alpha's acquired lock was released again.
    probe = open(jobs[0][1][0], "rb")
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        probe.close()


def test_lock_contention_blocks_verify_too(tmp_path):
    jobs = (_make_job(tmp_path, "alpha", _PLAN_TWO),)
    manifest = _manifest_path(tmp_path)
    coordinate_tile_updates(manifest, jobs)
    holder = open(jobs[0][1][0], "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_updates(manifest, jobs, verify=True)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
