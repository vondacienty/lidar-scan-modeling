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


@pytest.fixture
def scene():
    log, snaps = recovery_fixtures._chain()
    ledgers = recovery_fixtures._ledgers(log, snaps, 3)
    plan_one = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    plan_two = plan_checkout_recovery(log, checkpoint, ledgers[:3])

    states_one = recovery_fixtures._states(plan_one, 2)
    history_one_started = build_recovery_history(plan_one, ())
    history_one_partial = build_recovery_history(
        plan_one,
        (recovery_fixtures._summary(plan_one, states_one[:1]),))
    history_one_complete = build_recovery_history(
        plan_one,
        (recovery_fixtures._summary(plan_one, states_one[:1]),
         recovery_fixtures._summary(plan_one, states_one)))
    history_two_started = build_recovery_history(plan_two, ())
    states_two = recovery_fixtures._states(plan_two, 1)
    history_two_complete = build_recovery_history(
        plan_two,
        (recovery_fixtures._summary(plan_two, states_two),))
    plan_missing = plan_checkout_recovery(log, checkpoint, ())
    history_missing_partial = build_recovery_history(
        plan_missing,
        (recovery_fixtures._summary(
            plan_missing, recovery_fixtures._states(plan_missing, 1)),))
    history_missing_started = build_recovery_history(plan_missing, ())

    def _index(*pairs):
        return merge_recovery_indexes(
            (update_recovery_index(None, tuple(pairs)),))

    empty = merge_recovery_indexes(())
    return {
        "log": log, "snaps": snaps, "plan_one": plan_one,
        "plan_two": plan_two, "plan_missing": plan_missing,
        "h1_started": history_one_started,
        "h1_partial": history_one_partial,
        "h1_complete": history_one_complete,
        "h2_started": history_two_started,
        "h2_complete": history_two_complete,
        "hm_started": history_missing_started,
        "hm_partial": history_missing_partial,
        "index": _index, "empty": empty}


def _result(before, after):
    return json.loads(plan_recovery_index_migration(before, after))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.plan_recovery_index_migration \
        is plan_recovery_index_migration
    assert "plan_recovery_index_migration" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# empty indexes
# ---------------------------------------------------------------------------

def test_both_empty():
    empty = merge_recovery_indexes(())
    text = plan_recovery_index_migration(empty, empty)
    assert text == (
        '{"common":[0,0,null],"rollback":[],"pending":[],'
        '"snapshot":null,"resume":null,"complete":true}')
    document = json.loads(text)
    assert list(document) == ["common", "rollback", "pending", "snapshot",
                              "resume", "complete"]


def test_only_after_nonempty_uses_first_start(scene):
    index = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = _result(scene["empty"], index)
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["rollback"] == []
    assert document["pending"] == [["b0", "applied"], ["b1", "applied"]]
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["resume"] is None
    assert document["complete"] is True


def test_only_before_nonempty_rolls_back_everything_reversed(scene):
    index = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = _result(index, scene["empty"])
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["rollback"] == [["b1", "applied"], ["b0", "applied"]]
    assert document["pending"] == []
    assert document["snapshot"] is None
    assert document["resume"] is None
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# common entries
# ---------------------------------------------------------------------------

def test_identical_complete_index(scene):
    index = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = _result(index, index)
    assert document["common"] == [
        1, 0, json.loads(index)["snapshot"]]
    assert document["rollback"] == []
    assert document["pending"] == []
    assert document["snapshot"] == json.loads(index)["snapshot"]
    assert document["resume"] is None
    assert document["complete"] is True


def test_common_entries_then_divergence(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                            (scene["plan_two"], scene["h2_complete"]))
    after = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                           (scene["plan_two"], scene["h2_started"]))
    document = _result(before, after)
    snaps = scene["snaps"]
    assert document["common"] == [1, 0, json.loads(snaps[3])]
    assert document["rollback"] == [["b2", "applied"]]
    assert document["pending"] == []
    assert document["snapshot"] == json.loads(snaps[3])
    assert document["complete"] is False
    assert document["resume"] == json.loads(after)["resume"][1]


def test_common_entries_then_forward_progress(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                            (scene["plan_two"], scene["h2_started"]))
    after = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                           (scene["plan_two"], scene["h2_complete"]))
    document = _result(before, after)
    assert document["common"][:2] == [1, 0]
    assert document["rollback"] == []
    assert document["pending"] == [["b2", "applied"]]
    assert document["complete"] is True
    assert document["resume"] is None


# ---------------------------------------------------------------------------
# partial entry: common confirmed units
# ---------------------------------------------------------------------------

def test_partial_entry_rolls_back_tail_units(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    after = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    snaps = scene["snaps"]
    document = _result(before, after)
    assert document["common"] == [0, 1, json.loads(snaps[2])]
    assert document["rollback"] == [["b1", "applied"]]
    assert document["pending"] == []
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["resume"] == json.loads(after)["resume"][1]
    assert document["complete"] is False


def test_partial_entry_applies_tail_units(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    after = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = _result(before, after)
    assert document["common"][:2] == [0, 1]
    assert document["rollback"] == []
    assert document["pending"] == [["b1", "applied"]]
    assert document["resume"] is None
    assert document["complete"] is True


def test_partial_unit_prefix_then_later_entries_rollback(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                            (scene["plan_two"], scene["h2_complete"]))
    after = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    document = _result(before, after)
    assert document["common"][:2] == [0, 1]
    # later complete entry rows come first in the reverse audit, then the
    # partial entry's tail rows
    assert document["rollback"] == [["b2", "applied"], ["b1", "applied"]]
    assert document["pending"] == []
    assert document["complete"] is False


def test_partial_unit_prefix_then_later_entries_pending(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    after = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                           (scene["plan_two"], scene["h2_complete"]))
    document = _result(before, after)
    assert document["common"][:2] == [0, 1]
    assert document["rollback"] == []
    assert document["pending"] == [["b1", "applied"], ["b2", "applied"]]
    assert document["complete"] is True


def test_started_entry_with_no_confirmed_units_shares_no_units(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    after = scene["index"]((scene["plan_one"], scene["h1_started"]))
    document = _result(before, after)
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["rollback"] == [["b0", "applied"]]
    assert document["pending"] == []
    assert document["complete"] is False
    assert document["resume"] == json.loads(after)["resume"][1]


# ---------------------------------------------------------------------------
# differing plans after the entry prefix
# ---------------------------------------------------------------------------

def test_differing_plans_disjoint_audit_is_not_a_fork(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    after = scene["index"]((scene["plan_missing"], scene["hm_started"]))
    document = _result(before, after)
    snaps = scene["snaps"]
    assert document["common"] == [0, 0, json.loads(snaps[1])]
    assert document["rollback"] == [["b1", "applied"], ["b0", "applied"]]
    assert document["pending"] == []
    assert document["complete"] is False
    assert document["resume"] == json.loads(after)["resume"][1]


def test_differing_plans_intersecting_audit_forks(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    after = scene["index"]((scene["plan_missing"], scene["hm_partial"]))
    with pytest.raises(ValueError):
        plan_recovery_index_migration(before, after)


def test_different_first_entries_different_starts_fork(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    after = scene["index"]((scene["plan_two"], scene["h2_started"]))
    with pytest.raises(ValueError):
        plan_recovery_index_migration(before, after)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_non_str_arguments_raise_type_error(scene):
    empty = scene["empty"]
    for bad in (None, 1, (), bytes(empty, "ascii")):
        with pytest.raises(TypeError):
            plan_recovery_index_migration(bad, empty)
        with pytest.raises(TypeError):
            plan_recovery_index_migration(empty, bad)


def test_non_canonical_index_raises_value_error(scene):
    empty = scene["empty"]
    with pytest.raises(ValueError):
        plan_recovery_index_migration(empty + " ", empty)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(empty, "not json")


def test_derived_fields_are_rechecked(scene):
    index = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = json.loads(index)
    document["complete"] = False
    tampered = _canonical(document)
    with pytest.raises(ValueError):
        plan_recovery_index_migration(tampered, scene["empty"])
    with pytest.raises(ValueError):
        plan_recovery_index_migration(scene["empty"], tampered)


def test_duplicate_batch_id_in_an_index_is_rejected(scene):
    index = scene["index"]((scene["plan_one"], scene["h1_complete"]))
    document = json.loads(index)
    document["audit"].append([0, "b0", "applied"])
    with pytest.raises(ValueError):
        plan_recovery_index_migration(_canonical(document), scene["empty"])


# ---------------------------------------------------------------------------
# encoding, immutability and determinism
# ---------------------------------------------------------------------------

def test_canonical_compact_encoding(scene):
    pairs = [
        (scene["empty"], scene["empty"]),
        (scene["empty"],
         scene["index"]((scene["plan_one"], scene["h1_complete"]))),
        (scene["index"]((scene["plan_one"], scene["h1_complete"]),
                        (scene["plan_two"], scene["h2_complete"])),
         scene["index"]((scene["plan_one"], scene["h1_partial"]))),
    ]
    for before, after in pairs:
        text = plan_recovery_index_migration(before, after)
        assert text == _canonical(json.loads(text))
        assert "\n" not in text and not text.endswith("\n")
        assert list(json.loads(text)) == [
            "common", "rollback", "pending", "snapshot", "resume",
            "complete"]


def test_inputs_unchanged_and_repeated_calls_byte_identical(scene):
    before = scene["index"]((scene["plan_one"], scene["h1_complete"]),
                            (scene["plan_two"], scene["h2_complete"]))
    after = scene["index"]((scene["plan_one"], scene["h1_partial"]))
    before_copy, after_copy = before, after
    first = plan_recovery_index_migration(before, after)
    assert before == before_copy and after == after_copy
    second = plan_recovery_index_migration(before, after)
    assert first == second
