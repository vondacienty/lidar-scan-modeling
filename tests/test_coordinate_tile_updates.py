"""Tests for :func:`lidar_scan.coordinate_tile_updates`."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

from lidar_scan import (build_tile_pyramid, commit_tile_update_plan,
                        coordinate_tile_update, coordinate_tile_updates,
                        execute_tile_update_plan)
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


def _execution(plan=_PLAN_TWO, pyramid=None, **kwargs):
    if pyramid is None:
        pyramid = _new_pyramid()
    sources = sorted({source[0]
                      for task in json.loads(plan)["tasks"]
                      for source in task[5]})
    updates = tuple((source_id, pyramid) for source_id in sources)
    return execute_tile_update_plan(plan, updates, **kwargs)


def _write(path, text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(text)


def _read(path):
    with open(path, "rb") as stream:
        return stream.read()


def _job_paths(tmp_path, tag):
    base = str(tmp_path / tag)
    return (base + "-L", base + "-A", base + "-S", base + "-P")


def _seed_base_pyramid(paths, base, execution):
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    node = tiles_module._parse_json_node(
        zero_state, tiles_module._skip_json_ws(zero_state, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    _write(paths[3], zero_state[start:end])


def _two_jobs(tmp_path, base, execution):
    paths_a = _job_paths(tmp_path, "a")
    paths_b = _job_paths(tmp_path, "b")
    _seed_base_pyramid(paths_a, base, execution)
    _seed_base_pyramid(paths_b, base, execution)
    manifest = str(tmp_path / "manifest.json")
    jobs = (("b", paths_b, base, execution),
            ("a", paths_a, base, execution))
    return manifest, jobs, paths_a, paths_b


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.coordinate_tile_updates is coordinate_tile_updates
    assert "coordinate_tile_updates" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_manifest_must_be_a_non_empty_str(tmp_path):
    manifest, jobs, _pa, _pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_updates(1, jobs)
    with pytest.raises(ValueError):
        coordinate_tile_updates("", jobs)


def test_jobs_must_be_a_tuple_of_four_item_tuples(tmp_path):
    manifest, jobs, _pa, _pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, list(jobs))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, tuple(list(job) for job in jobs))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (jobs[0][:3],))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (jobs[0] + (1,),))


def test_job_ids_must_be_unique_non_empty_strs(tmp_path):
    manifest, jobs, _pa, _pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    # A non-str id is a TypeError.
    with pytest.raises(TypeError):
        coordinate_tile_updates(
            manifest, ((1, jobs[0][1], jobs[0][2], jobs[0][3]),))
    dup = (("a", jobs[0][1], jobs[0][2], jobs[0][3]),
           ("a", jobs[1][1], jobs[1][2], jobs[1][3]))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, dup)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, (("",) + jobs[0][1:],))


def test_paths_and_base_and_execution_keep_single_job_contract(tmp_path):
    manifest, jobs, pa, pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    base = jobs[0][2]
    execution = jobs[0][3]
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (("a", list(pa), base, execution),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (("a", pa[:3], base, execution),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (("a", pa, [], execution),))
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, (("a", pa, base, 1),))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, (("a", ("",) + pa[1:], base,
                                            execution),))
    repeated = (pa[0], pa[1], pa[2], pa[2])
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, (("a", repeated, base,
                                            execution),))


def test_all_paths_across_jobs_must_be_pairwise_distinct(tmp_path):
    manifest, jobs, pa, pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    base = jobs[0][2]
    execution = jobs[0][3]
    # Job a's A collides with job b's A.
    collision = (pa[0], pb[1], pa[2], pa[3])
    with pytest.raises(ValueError):
        coordinate_tile_updates(
            manifest, (("a", collision, base, execution),
                       ("b", pb, base, execution)))
    # The manifest path may not coincide with a job path.
    with pytest.raises(ValueError):
        coordinate_tile_updates(pa[1], (("a", pa, base, execution),))


def test_limit_and_verify_contract(tmp_path):
    manifest, jobs, _pa, _pb = _two_jobs(
        tmp_path, _base_pyramid(), _execution())
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, jobs, limit=1.0)
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, jobs, limit=True)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, limit=-1)
    with pytest.raises(TypeError):
        coordinate_tile_updates(manifest, jobs, verify=0)
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, limit=1, verify=True)


# ---------------------------------------------------------------------------
# publish flow and the shared budget
# ---------------------------------------------------------------------------

def test_fresh_publication_initializes_manifest_and_every_job(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)

    result = coordinate_tile_updates(manifest, jobs)

    assert _read(manifest).decode("utf-8") == result
    document = json.loads(result)
    assert list(document) == ["jobs", "complete"]
    assert document["complete"] is True
    assert list(document["jobs"]) == ["a", "b"]
    for job_id, paths in (("a", pa), ("b", pb)):
        entry = document["jobs"][job_id]
        assert entry[0] == job_id
        assert entry[1] == list(paths)
        assert entry[2] == 1  # genesis plus the completing publication
        assert entry[3] is True
        # The single-job files were published through as before.
        state = _read(paths[2]).decode("utf-8")
        assert json.loads(state)["committed"] == 2
        node = tiles_module._parse_json_node(
            state, tiles_module._skip_json_ws(state, 0))
        start, end = node[0]["pyramid"][1], node[0]["pyramid"][2]
        assert _read(paths[3]).decode("utf-8") == state[start:end]
    # Compact canonical UTF-8: no BOM, whitespace or trailing newline.
    raw = _read(manifest)
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)


def test_manifest_pyramid_files_match_their_state(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs)
    for paths in (pa, pb):
        state = _read(paths[2]).decode("utf-8")
        node = tiles_module._parse_json_node(
            state, tiles_module._skip_json_ws(state, 0))
        start, end = node[0]["pyramid"][1], node[0]["pyramid"][2]
        assert _read(paths[3]).decode("utf-8") == state[start:end]


def test_shared_budget_is_spent_in_id_order(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    pa = _job_paths(tmp_path, "a")
    pb = _job_paths(tmp_path, "b")
    _seed_base_pyramid(pa, base, execution)
    _seed_base_pyramid(pb, base, execution)
    manifest = str(tmp_path / "manifest.json")
    jobs = (("b", pb, base, execution), ("a", pa, base, execution))

    first = json.loads(coordinate_tile_updates(manifest, jobs, limit=1))
    assert json.loads(_read(pa[2]))["committed"] == 1
    assert json.loads(_read(pb[2]))["committed"] == 0
    assert first["jobs"]["a"][2:] == [1, False]
    assert first["jobs"]["b"][2:] == [0, False]
    assert first["complete"] is False

    second = json.loads(coordinate_tile_updates(manifest, jobs, limit=3))
    # a needs two more and completes; the remaining one advances b.
    assert json.loads(_read(pa[2]))["committed"] == 3
    assert json.loads(_read(pb[2]))["committed"] == 1
    assert second["jobs"]["a"][2:] == [2, True]
    assert second["jobs"]["b"][2:] == [1, False]
    assert second["complete"] is False

    finished = json.loads(coordinate_tile_updates(manifest, jobs))
    assert finished["complete"] is True
    assert finished["jobs"]["b"][2:] == [2, True]


def test_re_entered_completed_coordination_is_byte_identical(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    completed = coordinate_tile_updates(manifest, jobs)

    paths = [manifest, pa[1], pa[2], pa[3], pb[1], pb[2], pb[3]]
    before = tuple(_read(path) for path in paths)
    again = coordinate_tile_updates(manifest, jobs)
    after = tuple(_read(path) for path in paths)
    assert again == completed
    assert before == after


def test_lagging_manifest_entry_is_backfilled(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs, limit=2)

    # Tamper the manifest to lag job a's observed terminal record; a
    # re-entry catches the entry up rather than rejecting it.
    document = json.loads(_read(manifest))
    document["jobs"]["a"][2] = 0
    document["jobs"]["a"][3] = False
    _write(manifest, json.dumps(document, separators=(",", ":")))
    result = json.loads(coordinate_tile_updates(manifest, jobs))
    assert result["jobs"]["a"][2:] == [1, True]
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 2, True)


def test_lagging_pyramid_is_backfilled(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    first = coordinate_tile_updates(manifest, jobs, limit=2)
    first_a_state = _read(pa[2]).decode("utf-8")
    # Crash before a's P replacement: P is back at the base.
    node = tiles_module._parse_json_node(
        first_a_state, tiles_module._skip_json_ws(first_a_state, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    _write(pa[3], first_a_state[start:end])
    journal_before = _read(pa[1])

    coordinate_tile_updates(manifest, jobs, limit=0)
    node = tiles_module._parse_json_node(
        first_a_state, tiles_module._skip_json_ws(first_a_state, 0))
    start, end = node[0]["pyramid"][1], node[0]["pyramid"][2]
    assert _read(pa[3]).decode("utf-8") == first_a_state[start:end]
    assert _read(pa[1]) == journal_before


# ---------------------------------------------------------------------------
# verify flow
# ---------------------------------------------------------------------------

def test_verify_returns_task_complete_and_all_complete(tmp_path):
    base = _base_pyramid()
    execution = _execution(_PLAN_THREE)
    pa = _job_paths(tmp_path, "a")
    pb = _job_paths(tmp_path, "b")
    _seed_base_pyramid(pa, base, execution)
    _seed_base_pyramid(pb, base, execution)
    manifest = str(tmp_path / "manifest.json")
    jobs = (("a", pa, base, execution), ("b", pb, base, execution))

    coordinate_tile_updates(manifest, jobs, limit=1)
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 0, False)

    coordinate_tile_updates(manifest, jobs)
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 2, True)


def test_verify_is_strictly_read_only(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs)

    paths = [manifest, pa[0], pa[1], pa[2], pa[3], pb[0], pb[1], pb[2],
             pb[3]]
    before = tuple(_read(path) for path in paths)
    assert coordinate_tile_updates(manifest, jobs, verify=True) == \
        (2, 2, True)
    after = tuple(_read(path) for path in paths)
    assert before == after


def test_verify_requires_the_manifest_and_every_job_file(tmp_path):
    base = _base_pyramid()
    execution = _execution()

    def setup(tag):
        pa = _job_paths(tmp_path, tag + "-a")
        pb = _job_paths(tmp_path, tag + "-b")
        _seed_base_pyramid(pa, base, execution)
        _seed_base_pyramid(pb, base, execution)
        manifest = str(tmp_path / (tag + "-manifest.json"))
        jobs = (("a", pa, base, execution), ("b", pb, base, execution))
        return manifest, jobs, pa, pb

    # Nothing published yet: the manifest (and the job journals) are
    # missing, so verifying raises OSError.
    manifest, jobs, _pa, _pb = setup("fresh")
    with pytest.raises(OSError):
        coordinate_tile_updates(manifest, jobs, verify=True)

    # Every required file of a published coordination is mandatory.
    for tag, removed in (("m", "manifest"),
                         ("aA", "journal-a"),
                         ("aS", "state-a"),
                         ("aP", "pyramid-a")):
        manifest, jobs, pa, pb = setup(tag)
        coordinate_tile_updates(manifest, jobs)
        target = {
            "manifest": manifest,
            "journal-a": pa[1],
            "state-a": pa[2],
            "pyramid-a": pa[3],
        }[removed]
        os.unlink(target)
        with pytest.raises(OSError):
            coordinate_tile_updates(manifest, jobs, verify=True)


def test_verify_rejects_manifest_tampering(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs)

    good = _read(manifest).decode("utf-8")

    # The manifest may not lead the job journals.
    document = json.loads(good)
    document["jobs"]["a"][2] = 99
    _write(manifest, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, verify=True)

    # A complete flag that disagrees with the terminal state.
    document = json.loads(good)
    document["jobs"]["b"][3] = False
    _write(manifest, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, verify=True)

    # A job present in the manifest but absent from the jobs tuple.
    document = json.loads(good)
    document["jobs"]["ghost"] = ["ghost", list(pb), 1, True]
    _write(manifest, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs, verify=True)


def test_publish_rejects_manifest_leading_a_journal(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs)
    document = json.loads(_read(manifest))
    document["jobs"]["a"][2] = 5
    _write(manifest, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs)


def test_publish_rejects_manifest_recording_unknown_job(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs)
    document = json.loads(_read(manifest))
    document["jobs"]["ghost"] = ["ghost", list(pb), 1, True]
    _write(manifest, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        coordinate_tile_updates(manifest, jobs)


def test_job_state_missing_with_journal_is_os_error(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs, limit=2)
    os.unlink(pa[2])
    with pytest.raises(OSError):
        coordinate_tile_updates(manifest, jobs)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_releases_acquired_locks_and_writes_nothing(
        tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs, limit=2)

    paths = [manifest, pa[1], pa[2], pa[3], pb[1], pb[2], pb[3]]
    before = tuple(_read(path) for path in paths)

    holder = open(pb[0], "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            coordinate_tile_updates(manifest, jobs, verify=True)
        with pytest.raises(BlockingIOError):
            coordinate_tile_updates(manifest, jobs)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    after = tuple(_read(path) for path in paths)
    assert before == after


def test_lock_contention_across_processes_is_non_blocking(tmp_path):
    base = _base_pyramid()
    execution = _execution()
    manifest, jobs, pa, pb = _two_jobs(tmp_path, base, execution)
    coordinate_tile_updates(manifest, jobs, limit=2)

    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,sys,time\n"
         "f=open(sys.argv[1],'rb')\n"
         "fcntl.flock(f, fcntl.LOCK_EX)\n"
         "time.sleep(5)\n", pa[0]],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        started = time.monotonic()
        with pytest.raises(BlockingIOError):
            coordinate_tile_updates(manifest, jobs)
        assert time.monotonic() - started < 2.0
    finally:
        holder.terminate()
        holder.wait()
