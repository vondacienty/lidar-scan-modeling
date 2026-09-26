"""Tests for :func:`lidar_scan.execute_checkout_recovery`."""

from __future__ import annotations

import json

import pytest

from lidar_scan import (audit_delivery_changes, build_delivery_manifest,
                        build_delivery_plan, build_delivery_receipt,
                        build_delivery_snapshot, checkout_log,
                        execute_checkout_recovery, merge_checkout_logs,
                        merge_delivery_manifests, merge_delivery_receipts,
                        plan_checkout_recovery, plan_delivery_checkout,
                        plan_delivery_updates, delivery_log)
from lidar_scan import tiles as tiles_module

PASSING_GATE = (
    '{"windows":[[0,0,0,9,9,[-1.000000,2.000000,1.581139,1.000000,1],true]],'
    '"passed":true}'
)

EMPTY_CHANGES = '{"changes":[],"releasable":true}'
EMPTY_RECEIPTS = '{"products":[],"ready":true}'


def _empty_snapshot():
    return build_delivery_snapshot(EMPTY_CHANGES, EMPTY_RECEIPTS)


def _ready_p(batches):
    """Product ``p`` with the given (batch, version) pairs, all passing."""
    items = tuple((batch, "p", version, PASSING_GATE)
                  for batch, version in batches)
    changes = merge_delivery_manifests(
        tuple(build_delivery_manifest((item,)) for item in items))
    plan = build_delivery_plan(audit_delivery_changes(changes))
    receipts = merge_delivery_receipts(
        (build_delivery_receipt(plan, (("p", "succeeded", ""),)),))
    return build_delivery_snapshot(changes, receipts)


def _chain():
    """A log of four commits growing product ``p`` batch by batch."""
    snap0 = _empty_snapshot()
    snaps = [snap0] + [
        _ready_p(tuple((batch, str(batch + 1)) for batch in range(k + 1)))
        for k in range(4)]
    plans = [plan_delivery_updates(snaps[k], snaps[k + 1], (("p", k, k),))
             for k in range(4)]
    entries = tuple(
        ("c%d" % k, None if k == 0 else "c%d" % (k - 1), snaps[k],
         snaps[k + 1], plans[k])
        for k in range(4))
    return delivery_log(entries), snaps


def _ledgers(log, snaps, count):
    """One batch per ledger; ledger ``k`` applies c_k -> c_{k+1}."""
    checkout_plans = [plan_delivery_checkout(log, "c%d" % k, "c%d" % (k + 1))
                      for k in range(3)]
    return tuple(
        checkout_log(log, (("b%d" % k, snaps[k + 1],
                            (checkout_plans[k],)),))
        for k in range(count))


@pytest.fixture
def scene():
    log, snaps = _chain()
    ledgers = _ledgers(log, snaps, 3)
    return log, snaps, ledgers


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_function_exported_from_module_and_package():
    import lidar_scan
    assert tiles_module.execute_checkout_recovery is execute_checkout_recovery
    assert "execute_checkout_recovery" in lidar_scan.__all__


# ---------------------------------------------------------------------------
# pending direction
# ---------------------------------------------------------------------------

def test_pending_full_confirmation(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    text = execute_checkout_recovery(plan)
    document = json.loads(text)
    assert list(document) == ["common", "direction", "confirmed", "records",
                              "snapshot", "complete"]
    assert document["common"] == 0
    assert document["direction"] == "pending"
    assert document["confirmed"] == 2
    assert document["complete"] is True
    assert document["snapshot"] == json.loads(snaps[3])
    first, second = document["records"]
    assert first == ["b0", "b0", 1, json.loads(snaps[1]),
                     json.loads(snaps[2]), [["b0", "applied"]]]
    assert second[0] == "b1" and second[1] == "b1" and second[2] == 1
    assert second[3] == json.loads(snaps[2])
    assert second[4] == json.loads(snaps[3])
    assert second[5] == [["b1", "applied"]]
    assert text == json.dumps(document, ensure_ascii=False,
                              separators=(",", ":"))
    assert "\n" not in text and not text.endswith("\n")


def test_pending_common_prefix_echoed(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:1])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:2])
    assert json.loads(plan)["common"] == 1
    document = json.loads(execute_checkout_recovery(plan))
    assert document["common"] == 1
    assert document["confirmed"] == 1
    assert document["records"][0][0] == "b1"


def test_zero_confirmations_uses_first_before(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    document = json.loads(execute_checkout_recovery(plan, None, 0))
    assert document["confirmed"] == 0
    assert document["records"] == []
    assert document["snapshot"] == json.loads(snaps[1])
    assert document["complete"] is False


def test_stepwise_confirmation_through_state(scene):
    log, snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 0)
    state = execute_checkout_recovery(plan, state, 1)
    document = json.loads(state)
    assert document["confirmed"] == 1
    assert document["snapshot"] == json.loads(snaps[2])
    assert document["complete"] is False
    finished = execute_checkout_recovery(plan, state, 1)
    assert json.loads(finished)["confirmed"] == 2
    assert json.loads(finished)["complete"] is True
    # overshooting the remainder still finishes
    assert execute_checkout_recovery(plan, state, 9) == finished
    # and no limit finishes from a partial state
    assert execute_checkout_recovery(plan, state) == finished


def test_completed_state_reentry_is_identical(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    finished = execute_checkout_recovery(plan)
    assert execute_checkout_recovery(plan, finished) == finished
    assert execute_checkout_recovery(plan, finished, 0) == finished
    assert execute_checkout_recovery(plan, finished, 3) == finished


# ---------------------------------------------------------------------------
# missing direction
# ---------------------------------------------------------------------------

def test_missing_direction_confirms_missing_rows(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    plan_document = json.loads(plan)
    assert plan_document["common"] == 1
    assert plan_document["pending"] == []
    assert len(plan_document["missing"]) == 2
    zero = json.loads(execute_checkout_recovery(plan, None, 0))
    assert zero["direction"] == "missing"
    assert zero["confirmed"] == 0
    assert zero["snapshot"] == plan_document["missing"][0][0][3]
    assert zero["complete"] is False
    one = json.loads(execute_checkout_recovery(plan, None, 1))
    assert one["confirmed"] == 1 and len(one["records"]) == 1
    assert one["records"][0][0] == "b1"
    assert one["complete"] is False
    done = json.loads(execute_checkout_recovery(plan))
    assert done["confirmed"] == 2
    assert done["snapshot"] == plan_document["snapshot"]
    assert done["complete"] is True
    assert execute_checkout_recovery(plan, json.dumps(done,
                                                      ensure_ascii=False,
                                                      separators=(",", ":"))) \
        == json.dumps(done, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# none direction
# ---------------------------------------------------------------------------

def test_none_direction_is_complete_without_records(scene):
    log, _snaps, ledgers = scene
    checkpoint = merge_checkout_logs(log, ledgers[:2])
    plan = plan_checkout_recovery(log, checkpoint, ledgers[:2])
    plan_document = json.loads(plan)
    assert plan_document["missing"] == []
    assert plan_document["pending"] == []
    text = execute_checkout_recovery(plan)
    document = json.loads(text)
    assert document["direction"] == "none"
    assert document["confirmed"] == 0
    assert document["records"] == []
    assert document["snapshot"] == plan_document["snapshot"]
    assert document["complete"] is True
    assert execute_checkout_recovery(plan, text) == text


def test_both_empty_with_null_snapshot(scene):
    log, _snaps, _ledgers = scene
    plan = plan_checkout_recovery(log, None, ())
    document = json.loads(execute_checkout_recovery(plan))
    assert document["direction"] == "none"
    assert document["snapshot"] is None
    assert document["complete"] is True


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------

def test_type_errors(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for value in (1, None, b"x", [], {}):
        with pytest.raises(TypeError):
            execute_checkout_recovery(value)
    with pytest.raises(TypeError):
        execute_checkout_recovery(plan, 1)
    for value in (True, False, 1.0, "1", [], ()):
        with pytest.raises(TypeError):
            execute_checkout_recovery(plan, None, value)


# ---------------------------------------------------------------------------
# value validation
# ---------------------------------------------------------------------------

def test_negative_max_units_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    with pytest.raises(ValueError):
        execute_checkout_recovery(plan, None, -1)


def test_malformed_plan_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    for bad in ("", "not json", "{", "[]", "null",
                '{"common":0,"missing":[],"pending":[],'
                '"snapshot":null} extra',
                " " + plan,
                json.dumps(json.loads(plan), indent=2)):
        with pytest.raises(ValueError):
            execute_checkout_recovery(bad)


def test_forked_plan_with_both_arms_non_empty_rejected(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    pending_rows = pending_plan[
        pending_plan.index('"pending":[') + 11:
        pending_plan.index('],"snapshot"')]
    missing_rows = missing_plan[
        missing_plan.index('"missing":[') + 11:
        missing_plan.index('],"pending"')]
    snapshot_raw = pending_plan[
        pending_plan.index('"snapshot":') + 11:-1]
    forked = ('{"common":0,"missing":[' + missing_rows + '],"pending":['
              + pending_rows + '],"snapshot":' + snapshot_raw + "}")
    json.loads(forked)
    with pytest.raises(ValueError):
        execute_checkout_recovery(forked)


def test_state_must_belong_to_the_plan(scene):
    log, _snaps, ledgers = scene
    pending_plan = plan_checkout_recovery(log, None, ledgers[:2])
    checkpoint = merge_checkout_logs(log, ledgers[:3])
    missing_plan = plan_checkout_recovery(log, checkpoint, ledgers[:1])
    missing_state = execute_checkout_recovery(missing_plan, None, 0)
    with pytest.raises(ValueError):
        execute_checkout_recovery(pending_plan, missing_state)


def test_state_must_be_a_confirmation_prefix(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 1)
    tampered = state.replace('"b0"', '"b1"', 1)
    assert tampered != state
    with pytest.raises(ValueError):
        execute_checkout_recovery(plan, tampered)


def test_malformed_or_non_canonical_state_rejected(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    state = execute_checkout_recovery(plan, None, 0)
    with pytest.raises(ValueError):
        execute_checkout_recovery(plan, "{")
    with pytest.raises(ValueError):
        execute_checkout_recovery(plan, json.dumps(json.loads(state),
                                                   indent=2))
    with pytest.raises(ValueError):
        execute_checkout_recovery(plan, state + " ")


def test_repeatable_and_inputs_unchanged(scene):
    log, _snaps, ledgers = scene
    plan = plan_checkout_recovery(log, None, ledgers[:2])
    first = execute_checkout_recovery(plan, None, 1)
    second = execute_checkout_recovery(plan, None, 1)
    assert first == second
    assert plan == plan_checkout_recovery(log, None, ledgers[:2])
