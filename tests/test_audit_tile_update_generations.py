"""Tests for :func:`lidar_scan.audit_tile_update_generations`."""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from lidar_scan import audit_tile_update_generations
from lidar_scan import coordinate_tile_update_generations
from lidar_scan import tiles as tiles_module
import lidar_scan

from test_coordinate_tile_update_generations import _all_paths, _generation
from test_coordinate_tile_updates import _read, _write


def _top_path(root):
    return str(root / "generations.json")


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    assert tiles_module.audit_tile_update_generations is \
        audit_tile_update_generations
    assert "audit_tile_update_generations" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_path_contract(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    with pytest.raises(TypeError):
        audit_tile_update_generations(1, generations)
    with pytest.raises(ValueError):
        audit_tile_update_generations("", generations)


def test_generations_must_be_a_tuple(tmp_path):
    with pytest.raises(TypeError):
        audit_tile_update_generations(_top_path(tmp_path), [])


def test_generation_shape(tmp_path):
    generation = _generation(tmp_path, "g1")
    with pytest.raises(TypeError):
        audit_tile_update_generations(_top_path(tmp_path),
                                      (generation[:2],))
    with pytest.raises(TypeError):
        audit_tile_update_generations(_top_path(tmp_path),
                                      (generation + (generation,),))


def test_generation_id_contract(tmp_path):
    generation = _generation(tmp_path, "g1")
    bad = (1, generation[1], generation[2])
    with pytest.raises(TypeError):
        audit_tile_update_generations(_top_path(tmp_path), (bad,))
    empty = ("", generation[1], generation[2])
    with pytest.raises(ValueError):
        audit_tile_update_generations(_top_path(tmp_path), (empty,))
    duplicate = ("g1", generation[1], generation[2])
    with pytest.raises(ValueError):
        audit_tile_update_generations(_top_path(tmp_path),
                                      (generation, duplicate))


def test_generation_ids_must_strictly_increase(tmp_path):
    g1 = _generation(tmp_path, "g2")
    g2 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        audit_tile_update_generations(_top_path(tmp_path), (g1, g2))


def test_all_paths_globally_pairwise_distinct(tmp_path):
    g1 = _generation(tmp_path, "g1")
    with pytest.raises(ValueError):
        audit_tile_update_generations(g1[1], (g1,))
    with pytest.raises(ValueError):
        audit_tile_update_generations(g1[2][0][1][1], (g1,))
    g2 = _generation(tmp_path, "g2")
    g2_bad = ("g2", g1[2][0][1][2], g2[2])
    with pytest.raises(ValueError):
        audit_tile_update_generations(_top_path(tmp_path), (g1, g2_bad))


# ---------------------------------------------------------------------------
# audit output
# ---------------------------------------------------------------------------

def test_fresh_tree_reports_every_generation_unstarted(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    result = audit_tile_update_generations(_top_path(tmp_path), generations)
    assert result == ('{"generations":[["g1",false,0,0,false],'
                      '["g2",false,0,0,false]],"complete":false}')


def test_empty_generations_reports_empty_completed_document(tmp_path):
    result = audit_tile_update_generations(_top_path(tmp_path), ())
    assert result == '{"generations":[],"complete":true}'


def test_partial_run_reports_started_and_unstarted_generations(tmp_path):
    # g1's alpha needs three tasks; a budget of two commits two, so g1 is
    # started but incomplete and g2 has never started.
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)

    result = audit_tile_update_generations(path, generations)
    assert result == ('{"generations":[["g1",true,2,2,false],'
                      '["g2",false,0,0,false]],"complete":false}')


def test_completed_run_reports_every_generation_complete(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)

    result = audit_tile_update_generations(path, generations)
    assert result == ('{"generations":[["g1",true,2,5,true],'
                      '["g2",true,2,5,true]],"complete":true}')
    document = json.loads(result)
    assert list(document) == ["generations", "complete"]
    for record in document["generations"]:
        assert len(record) == 5


def test_output_is_compact_and_canonical(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    result = audit_tile_update_generations(path, generations)
    assert not result.endswith("\n")
    assert all(ch not in " \t\r" for ch in result)


def test_non_ascii_ids_are_not_escaped(tmp_path):
    generations = (_generation(tmp_path, "gén"),)
    result = audit_tile_update_generations(_top_path(tmp_path), generations)
    assert result == '{"generations":[["gén",false,0,0,false]],' \
                     '"complete":false}'


def test_audit_is_strictly_read_only(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations, limit=2)
    payload_paths = [path] + _all_paths(generations)
    before = {candidate: _read(candidate) for candidate in payload_paths
              if os.path.exists(candidate)}

    audit_tile_update_generations(path, generations)

    for candidate, content in before.items():
        assert _read(candidate) == content


def test_audit_ignores_the_top_manifest(tmp_path):
    # The audit reports the sub-manifests and A/S/P bindings; the top
    # generations manifest is not consulted, so even a corrupt top
    # document does not change the report.
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    _write(path, "{corrupt")
    assert audit_tile_update_generations(path, generations) == \
        '{"generations":[["g1",true,2,5,true]],"complete":true}'


# ---------------------------------------------------------------------------
# integrity failures
# ---------------------------------------------------------------------------

def test_orphan_journal_without_manifest_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    _write(generations[0][2][0][1][1], "{}")
    with pytest.raises(ValueError):
        audit_tile_update_generations(_top_path(tmp_path), generations)


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
        audit_tile_update_generations(path, generations)


def test_broken_journal_chain_is_rejected(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    journal = generations[0][2][0][1][1]
    document = json.loads(_read(journal).decode())
    document["records"][0][0] = 5
    _write(journal, json.dumps(document, separators=(",", ":")))
    with pytest.raises(ValueError):
        audit_tile_update_generations(path, generations)


def test_missing_pyramid_of_started_generation_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(generations[0][2][0][1][3])
    with pytest.raises(OSError):
        audit_tile_update_generations(path, generations)


def test_missing_journal_of_started_generation_is_an_os_error(tmp_path):
    generations = (_generation(tmp_path, "g1"),)
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, generations)
    os.unlink(generations[0][2][0][1][1])
    with pytest.raises(OSError):
        audit_tile_update_generations(path, generations)


def test_later_generation_registered_without_earlier_is_rejected(tmp_path):
    g1 = _generation(tmp_path, "g1")
    g2 = _generation(tmp_path, "g2")
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, (g1, g2))
    os.unlink(g1[1])
    with pytest.raises(ValueError):
        audit_tile_update_generations(path, (g1, g2))


def test_later_generation_registered_before_earlier_completes(tmp_path):
    # Complete g1 only partially, then start g2 behind the coordinator's
    # back: the audit rejects the broken registration chain.
    from lidar_scan import coordinate_tile_updates
    g1 = _generation(tmp_path, "g1")
    g2 = _generation(tmp_path, "g2")
    path = _top_path(tmp_path)
    coordinate_tile_update_generations(path, (g1, g2), limit=2)
    coordinate_tile_updates(g2[1], g2[2])
    with pytest.raises(ValueError):
        audit_tile_update_generations(path, (g1, g2))


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_lock_contention_raises_blocking_io_error(tmp_path):
    generations = (_generation(tmp_path, "g1"), _generation(tmp_path, "g2"))
    path = _top_path(tmp_path)
    held = generations[0][2][1][1][0]
    open(held, "wb").close()
    holder = open(held, "rb")
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            audit_tile_update_generations(path, generations)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    # The audit wrote nothing: no manifest, journal or state file exists.
    assert not os.path.exists(path)
    for _id, manifest, jobs in generations:
        assert not os.path.exists(manifest)
        for job in jobs:
            assert not os.path.exists(job[1][1])
            assert not os.path.exists(job[1][2])
