"""Tests for :func:`lidar_scan.plan_recovery_index_migration`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (build_recovery_history, merge_checkout_logs,
                        merge_recovery_indexes, plan_checkout_recovery,
                        plan_recovery_index_migration, update_recovery_index)
from lidar_scan import tiles as tiles_module
import test_build_recovery_history as recovery_fixtures


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def _states(plan, count):
    return recovery_fixtures._states(plan, count)


def _summary(plan, states):
    return recovery_fixtures._summary(plan, states)


def _completed_history(plan, units):
    states = _states(plan, units)
    return build_recovery_history(
        plan, tuple(_summary(plan, states[:i + 1])
                    for i in range(units)))


def _index(pairs):
    return merge_recovery_indexes((update_recovery_index(None, pairs),))


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    checkpoint_one = merge_checkout_logs(log, ledgers[:1])

    # p0 applies b0 from snaps[1]; p01 applies b0,b1 from snaps[1].
    p0 = plan_checkout_recovery(log, None, ledgers[:1])
    p01 = plan_checkout_recovery(log, None, ledgers[:2])
    # From checkpoint b0: b1 alone, b1,b2, or the relabelled batch b1a
    # (same commit, same endpoints, different batch id).
    p1 = plan_checkout_recovery(log, checkpoint_one, ledgers[:2])
    p12 = plan_checkout_recovery(log, checkpoint_one, ledgers[:3])
    ledger_alt = ledgers[1].replace('"b1"', '"b1a"')
    p1_alt = plan_checkout_recovery(
        log, checkpoint_one, ledgers[:1] + (ledger_alt,))

    h0 = _completed_history(p0, 1)
    h01 = _completed_history(p01, 2)
    h1 = _completed_history(p1, 1)
    h12 = _completed_history(p12, 2)
    h1_alt = _completed_history(p1_alt, 1)

    states = _states(p01, 1)
    h01_partial = build_recovery_history(
        p01, (_summary(p01, states),))
    h01_started = build_recovery_history(p01, ())

    return {
        "snaps": snaps,
        "empty": merge_recovery_indexes(()),
        "p0_done": _index(((p0, h0),)),
        "p01_done": _index(((p01, h01),)),
        "p01_partial": _index(((p01, h01_partial),)),
        "p01_started": _index(((p01, h01_started),)),
        "p1_done": _index(((p1, h1),)),
        "p0_then_p1": _index(((p0, h0), (p1, h1))),
        "p0_then_p12": _index(((p0, h0), (p12, h12))),
        "p0_then_alt": _index(((p0, h0), (p1_alt, h1_alt))),
        "partial_history": json.loads(h01_partial),
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.plan_recovery_index_migration \
        is plan_recovery_index_migration
    assert "plan_recovery_index_migration" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty sides
# ---------------------------------------------------------------------------

def test_both_empty(scene):
    text = plan_recovery_index_migration(scene["empty"], scene["empty"])
    assert text == ('{"common":[0,0,null],"rollback":[],"pending":[],'
                    '"snapshot":null,"resume":null,"complete":true}')
    assert list(json.loads(text)) == [
        "common", "rollback", "pending", "snapshot", "resume", "complete"]
    assert not text.endswith("\n") and " " not in text


def test_empty_before_uses_first_after_start(scene):
    text = plan_recovery_index_migration(
        scene["empty"], scene["p0_done"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document == {
        "common": [0, 0, json.loads(snaps[1])],
        "rollback": [],
        "pending": [["b0", "applied"]],
        "snapshot": json.loads(snaps[2]),
        "resume": None,
        "complete": True,
    }


def test_empty_after_uses_first_before_start(scene):
    text = plan_recovery_index_migration(
        scene["p0_done"], scene["empty"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document == {
        "common": [0, 0, json.loads(snaps[1])],
        "rollback": [["b0", "applied"]],
        "pending": [],
        "snapshot": None,
        "resume": None,
        "complete": True,
    }


def test_empty_before_into_unfinished_after_keeps_resume(scene):
    text = plan_recovery_index_migration(
        scene["empty"], scene["p01_partial"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["pending"] == [["b0", "applied"]]
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["resume"] == scene["partial_history"]["resume"]
    assert document["complete"] is False


# ---------------------------------------------------------------------------
# identical indexes
# ---------------------------------------------------------------------------

def test_identical_indexes_share_every_entry(scene):
    text = plan_recovery_index_migration(
        scene["p0_then_p1"], scene["p0_then_p1"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [2, 0, json.loads(snaps[3])]
    assert document["rollback"] == []
    assert document["pending"] == []
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] is None
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# same-plan entry compared by confirmed units and audit segments
# ---------------------------------------------------------------------------

def test_same_plan_unit_prefix_extends_forward(scene):
    text = plan_recovery_index_migration(
        scene["p01_partial"], scene["p01_done"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [0, 1, json.loads(snaps[2])]
    assert document["rollback"] == []
    assert document["pending"] == [["b1", "applied"]]
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] is None
    assert document["complete"] is True


def test_same_plan_unit_prefix_rolls_back(scene):
    text = plan_recovery_index_migration(
        scene["p01_done"], scene["p01_partial"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [0, 1, json.loads(snaps[2])]
    assert document["rollback"] == [["b1", "applied"]]
    assert document["pending"] == []
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["resume"] == scene["partial_history"]["resume"]
    assert document["complete"] is False


def test_same_plan_with_no_confirmed_units(scene):
    text = plan_recovery_index_migration(
        scene["p01_started"], scene["p01_partial"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["rollback"] == []
    assert document["pending"] == [["b0", "applied"]]
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["resume"] == scene["partial_history"]["resume"]
    assert document["complete"] is False


# ---------------------------------------------------------------------------
# divergence past wholly common entries
# ---------------------------------------------------------------------------

def test_diverged_entries_with_disjoint_batch_ids(scene):
    text = plan_recovery_index_migration(
        scene["p0_then_p1"], scene["p0_then_alt"])
    document = json.loads(text)
    snaps = scene["snaps"]
    assert document["common"] == [1, 0, json.loads(snaps[2])]
    assert document["rollback"] == [["b1", "applied"]]
    assert document["pending"] == [["b1a", "applied"]]
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] is None
    assert document["complete"] is True


def test_rollback_rows_are_reverse_order(scene):
    text = plan_recovery_index_migration(
        scene["p0_then_p12"], scene["p0_done"])
    document = json.loads(text)
    assert document["common"][0] == 1
    assert document["rollback"] == [["b2", "applied"], ["b1", "applied"]]
    assert document["pending"] == []


# ---------------------------------------------------------------------------
# forks and incompatible starts
# ---------------------------------------------------------------------------

def test_intersecting_batch_ids_is_a_fork(scene):
    # before runs b0,b1 as one entry; after runs b0 then b1a: the tails
    # past the boundary both still carry b0.
    with pytest.raises(ValueError):
        plan_recovery_index_migration(
            scene["p01_done"], scene["p0_then_alt"])


def test_different_first_starts_rejected(scene):
    with pytest.raises(ValueError):
        plan_recovery_index_migration(
            scene["p0_done"], scene["p1_done"])


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    index = scene["p0_done"]
    for value in (None, 1, [], b"x", {}):
        with pytest.raises(TypeError):
            plan_recovery_index_migration(value, index)
        with pytest.raises(TypeError):
            plan_recovery_index_migration(index, value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_malformed_and_non_canonical_indexes_rejected(scene):
    index = scene["p0_done"]
    with pytest.raises(ValueError):
        plan_recovery_index_migration("{", index)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(index + " ", index)
    pretty = json.dumps(json.loads(index), indent=2)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(pretty, index)


def test_tampered_derived_fields_rejected(scene):
    index = scene["p01_done"]
    tampered = json.loads(index)
    tampered["audit"] = []
    with pytest.raises(ValueError):
        plan_recovery_index_migration(_canonical(tampered), index)
    tampered = json.loads(index)
    tampered["complete"] = False
    with pytest.raises(ValueError):
        plan_recovery_index_migration(index, _canonical(tampered))
    tampered = json.loads(index)
    tampered["snapshot"] = None
    with pytest.raises(ValueError):
        plan_recovery_index_migration(index, _canonical(tampered))


def test_broken_chain_rejected(scene):
    document = json.loads(scene["p0_then_p12"])
    document["entries"].append(document["entries"][0])
    with pytest.raises(ValueError):
        plan_recovery_index_migration(
            scene["p0_done"], _canonical(document))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeatable_and_inputs_unchanged(scene):
    before = scene["p0_then_p1"]
    after = scene["p0_then_alt"]
    saved_before, saved_after = before, after
    one = plan_recovery_index_migration(before, after)
    two = plan_recovery_index_migration(before, after)
    assert one == two
    assert before == saved_before
    assert after == saved_after
