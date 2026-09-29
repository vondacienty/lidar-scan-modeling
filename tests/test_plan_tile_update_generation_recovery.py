"""Tests for :func:`lidar_scan.plan_tile_update_generation_recovery`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import commit_tile_update_plan
from lidar_scan import coordinate_tile_update_generations
from lidar_scan import coordinate_tile_updates
from lidar_scan import execute_tile_update_plan
from lidar_scan import plan_tile_update_generation_recovery
from lidar_scan import tiles as tiles_module
import lidar_scan

from test_coordinate_tile_update_generations import (_all_paths, _generation,
                                                     _top_path)
from test_coordinate_tile_updates import (_base_pyramid, _job_paths, _read,
                                          _write)


def _zero_task_generation(root, gen_id):
    # A generation whose single job plans zero tasks: it can only ever
    # need generation-level reconciliation.
    directory = root / gen_id
    directory.mkdir(parents=True, exist_ok=True)
    paths = _job_paths(directory, "alpha")
    base = _base_pyramid()
    execution = execute_tile_update_plan('{"tasks":[],"complete":true}', ())
    zero_state = commit_tile_update_plan(base, execution, max_tasks=0)
    node = tiles_module._parse_json_node(
        zero_state, tiles_module._skip_json_ws(zero_state, 0))
    start, end = node[0]["base"][1], node[0]["base"][2]
    _write(paths[3], zero_state[start:end])
    return (gen_id, str(directory / "manifest.json"),
            (("alpha", paths, base, execution),))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.plan_tile_update_generation_recovery is \
        plan_tile_update_generation_recovery
    assert "plan_tile_update_generation_recovery" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
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


def test_generation_id_contract(tmp_path):
    generation = _generation(tmp_path, "g1")
    bad = (1, generation[1], generation[2])
    with pytest.raises(TypeError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), (bad,))
    empty = ("", generation[1], generation[2])
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), (empty,))
    duplicate = ("g1", generation[1], generation[2])
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (generation, duplicate))


def test_generation_ids_must_strictly_increase(tmp_path):
    g1 = _generation(tmp_path, "g2")
    g2 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path), (g1, g2))


def test_all_paths_globally_pairwise_distinct(tmp_path):
    g1 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(g1[1], (g1,))
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(g1[2][0][1][1], (g1,))
    g2 = _generation(tmp_path, "g2")
    g2_bad = ("g2", g1[2][0][1][2], g2[2])
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             (g1, g2_bad))


# ---------------------------------------------------------------------------
# plan output
# ---------------------------------------------------------------------------

def test_empty_generations_plan_empty_completed_document(tmp_path):
    result = plan_tile_update_generation_recovery(_top_path(tmp_path), ())
    assert result == '{"generations":[],"next":null,"complete":true}'


def test_unstarted_generations_plan_every_job(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    result = plan_tile_update_generation_recovery(_top_path(tmp_path),
                                                  generations)
    assert result == ('{"generations":['
                      '["g1",false,[["alpha",0,3],["beta",0,2]],false],'
                      '["g2",false,[["alpha",0,3],["beta",0,2]],false]],'
                      '"next":["g1","alpha"],"complete":false}')


def test_unstarted_plan_writes_no_file_at_all(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    plan_tile_update_generation_recovery(path, generations)
    assert not os.path.exists(path)
    for _id, manifest, jobs in generations:
        assert not os.path.exists(manifest)
        for job in jobs:
            # Only the seeded pyramid may exist; the plan creates
            # nothing, not even a lock file.
            for candidate in job[1][:3]:
                assert not os.path.exists(candidate)


def test_completed_generations_plan_next_null(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)

    result = plan_tile_update_generation_recovery(path, generations)
    assert result == ('{"generations":['
                      '["g1",true,[["alpha",3,3],["beta",2,2]],true],'
                      '["g2",true,[["alpha",3,3],["beta",2,2]],true]],'
                      '"next":null,"complete":true}')
    document = json.loads(result)
    assert list(document) == ["generations", "next", "complete"]


def test_partial_publication_plans_the_next_job(tmp_path):
    # g1's alpha commits two of its three tasks under a budget of two;
    # g2 never starts.
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)

    result = plan_tile_update_generation_recovery(path, generations)
    assert result == ('{"generations":['
                      '["g1",true,[["alpha",2,3],["beta",0,2]],false],'
                      '["g2",false,[["alpha",0,3],["beta",0,2]],false]],'
                      '"next":["g1","alpha"],"complete":false}')

    # Resuming to completion empties the plan.
    coordinate_tile_update_generations(path, generations)
    assert plan_tile_update_generation_recovery(path, generations) == (
        '{"generations":['
        '["g1",true,[["alpha",3,3],["beta",2,2]],true],'
        '["g2",true,[["alpha",3,3],["beta",2,2]],true]],'
        '"next":null,"complete":true}')


def test_next_skips_complete_generations(tmp_path):
    # A budget of seven completes g1 (five tasks) and commits two of
    # g2 alpha's three tasks.
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=7)

    result = plan_tile_update_generation_recovery(path, generations)
    assert result == ('{"generations":['
                      '["g1",true,[["alpha",3,3],["beta",2,2]],true],'
                      '["g2",true,[["alpha",2,3],["beta",0,2]],false]],'
                      '"next":["g2","alpha"],"complete":false}')


def test_next_null_when_only_generation_reconciliation(tmp_path):
    # A zero-task generation is incomplete until it starts, yet no job
    # has confirmed < total: only generation-level reconciliation
    # remains, so next is null.
    generation = _zero_task_generation(tmp_path, "g1")
    path = _top_path(tmp_path)
    result = plan_tile_update_generation_recovery(path, (generation,))
    assert result == ('{"generations":[["g1",false,[["alpha",0,0]],false]],'
                      '"next":null,"complete":false}')

    coordinate_tile_update_generations(path, (generation,))
    assert plan_tile_update_generation_recovery(path, (generation,)) == (
        '{"generations":[["g1",true,[["alpha",0,0]],true]],'
        '"next":null,"complete":true}')


def test_jobs_reported_in_id_order(tmp_path):
    generation = _generation(tmp_path, "g1")
    generation = (generation[0], generation[1],
                  (generation[2][1], generation[2][0]))
    result = plan_tile_update_generation_recovery(_top_path(tmp_path),
                                                  (generation,))
    assert result == ('{"generations":[["g1",false,[["alpha",0,3],'
                      '["beta",0,2]],false]],"next":["g1","alpha"],'
                      '"complete":false}')


def test_output_is_compact_canonical_json(tmp_path):
    generations = (_generation(tmp_path, "gé"),)
    coordinate_tile_update_generations(_top_path(tmp_path), generations)
    result = plan_tile_update_generation_recovery(_top_path(tmp_path),
                                                  generations)
    raw = result.encode("utf-8")
    assert not raw.endswith(b"\n")
    assert all(ch not in b" \t\r\n" for ch in raw)
    # Non-ASCII ids are not escaped.
    assert "gé" in result
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
    assert not os.path.exists(generations[1][1])


# ---------------------------------------------------------------------------
# on-disk consistency
# ---------------------------------------------------------------------------

def test_missing_state_file_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(generations[0][2][0][1][2])
    with pytest.raises(OSError):
        plan_tile_update_generation_recovery(path, generations)


def test_broken_journal_chain_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    journal = generations[0][2][0][1][1]
    document = json.loads(_read(journal).decode("utf-8"))
    document["records"][0][0] = 5
    _write(journal, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(path, generations)


def test_tampered_generation_manifest_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    manifest = generations[0][1]
    document = json.loads(_read(manifest).decode())
    job_paths = document["jobs"][0][1]
    forged = ('{"jobs":[["alpha",['
              + ",".join(json.dumps(candidate) for candidate in job_paths)
              + '],99,true]],"complete":true}')
    _write(manifest, forged)
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(path, generations)


def test_non_canonical_generation_manifest_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    manifest = generations[0][1]
    document = json.loads(_read(manifest).decode())
    _write(manifest, json.dumps(document, indent=1))
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(path, generations)


def test_orphan_journal_without_manifest_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    _write(generations[0][2][0][1][1], '{"records":[]}')
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             generations)


def test_later_generation_registered_before_earlier_completes(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    # Complete g2 directly while g1 is still incomplete.
    coordinate_tile_updates(generations[1][1], generations[1][2])
    with pytest.raises(ValueError):
        plan_tile_update_generation_recovery(path, generations)


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

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


def test_started_generation_missing_lock_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(generations[0][2][0][1][0])
    with pytest.raises(OSError) as excinfo:
        plan_tile_update_generation_recovery(path, generations)
    assert not isinstance(excinfo.value, BlockingIOError)


def test_unstarted_generation_state_without_lock_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    _write(generations[0][2][0][1][2], "{}")
    with pytest.raises(OSError) as excinfo:
        plan_tile_update_generation_recovery(_top_path(tmp_path),
                                             generations)
    assert not isinstance(excinfo.value, BlockingIOError)


def test_lock_appearing_during_read_is_blocking(tmp_path, monkeypatch):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    lock_path = generations[0][2][0][1][0]
    original = tiles_module._read_utf8_file

    def concurrent_start(candidate, label):
        if candidate == generations[0][1] \
                and not os.path.exists(lock_path):
            _write(lock_path, "")
        return original(candidate, label)

    monkeypatch.setattr(tiles_module, "_read_utf8_file", concurrent_start)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(path, generations)


def test_unstarted_evidence_appearing_during_read_is_blocking(
        tmp_path, monkeypatch):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    lock_path = generations[0][2][1][1][0]
    original = tiles_module._read_utf8_file

    def concurrent_start(candidate, label):
        if candidate == generations[1][1] \
                and not os.path.exists(lock_path):
            _write(lock_path, "")
        return original(candidate, label)

    monkeypatch.setattr(tiles_module, "_read_utf8_file", concurrent_start)
    with pytest.raises(BlockingIOError):
        plan_tile_update_generation_recovery(path, generations)
