"""
training-smoketests/smoke_client_dp_preflight.py

Regression tests for the DP early validation added to federated/client.py:
  - config rules: "functorch" only, effective == dp-logical == LOGICAL_BATCH_SIZE
  - early budget + exact round-progress checks, which must reject BEFORE the
    gradient gate, the model or any DataLoader is touched.

No model, data or GPU is needed: the calibration file is replaced by an
in-memory dict, and the three expensive entry points
(_build_model, get_dataloaders, ensure_per_sample_gradients_verified) are
replaced by recorders. A test passes the early checks exactly when the
FIRST expensive call (the gradient gate) is reached.

Run with pytest, or `python smoke_client_dp_preflight.py`.

Merge note for smoke_client.py: C11 ("hooks" mode) now fails at CONFIG
validation, so it must expect ValueError (not DPGradientCheckError). The
gradient gate itself is still covered by the dp_checks tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federated import client as c  # adjust if the package layout differs

ROUNDS, LOCAL_EPOCHS, NUM_BATCHES = 10, 3, 6  # IDRiD-like: 330 images at 64
PLANNED = LOCAL_EPOCHS * NUM_BATCHES          # 18 steps per round
TOTAL = ROUNDS * PLANNED                      # 180 calibrated steps
CLIENT, PARTITION = "IDRiD", 1


class _Reached(Exception):
    """Raised by the patched gradient gate: the early checks passed."""


def _calibration():
    return {
        "accountant": "rdp", "target_delta": 1e-5, "logical_batch_size": c.LOGICAL_BATCH_SIZE,
        "epochs": ROUNDS * LOCAL_EPOCHS, "target_epsilon": 5.0,
        "clients": {CLIENT: {
            "noise_multiplier": 1.1, "sample_rate": 1.0 / NUM_BATCHES,
            "num_batches": NUM_BATCHES, "steps": TOTAL, "dataset_size": 330,
        }},
    }


def _dp_cfg(**over):
    logical = c.LOGICAL_BATCH_SIZE
    cfg = {
        "lora-rank": 8, "lora-alpha": 16, "img-size": 224, "num-classes": 5,
        "manifest-root": "unused", "num-workers": 0, "run-seed": 7,
        "experiment-id": "exp4", "algorithm": "fedprox", "dp": True,
        "num-rounds": ROUNDS, "local-epochs": LOCAL_EPOCHS,
        "physical-batch-size": 32, "accumulation-steps": 2, "effective-batch-size": logical,
        "label-smoothing": 0.1, "weight-decay": 0.01, "round": 1,
        "lr-lora": 1e-4, "lr-head": 1e-3, "proximal-mu": 0.01,
        "calibration-path": "unused", "max-grad-norm": 1.0, "delta": 1e-5,
        "grad-sample-mode": "functorch", "dp-logical-batch-size": logical,
        "accountant": "rdp", "target-epsilon": 5.0,
    }
    cfg.update(over)
    return cfg


def _state(steps, mechanism="rdp"):
    return {"history": [[1.1, 1.0 / NUM_BATCHES, steps]] if steps else [], "mechanism": mechanism}


def _run_round(cfg, state):
    """Call train_client_round with the expensive parts replaced. Returns
    (outcome, calls): outcome is 'reached' when the gradient gate was
    reached (early checks passed), otherwise the exception raised."""
    calls = []

    def recorder(name, exc=None):
        def _fn(*args, **kwargs):
            calls.append(name)
            if exc is not None:
                raise exc
            raise AssertionError(f"{name} must not be reached")
        return _fn

    states = {str(PARTITION): state} if state is not None else None
    with mock.patch.object(c, "load_calibration", return_value=_calibration()), \
         mock.patch.object(c, "_manifest_path", return_value=Path("unused.csv")), \
         mock.patch.object(c, "_build_model", recorder("model")), \
         mock.patch.object(c, "get_dataloaders", recorder("loader")), \
         mock.patch.object(c, "ensure_per_sample_gradients_verified", recorder("gate", _Reached())):
        try:
            c.train_client_round(partition_id=PARTITION, cfg=cfg, global_state={}, accountant_states=states)
        except _Reached:
            return "reached", calls
        except Exception as exc:  # noqa: BLE001 - the test inspects the type
            return exc, calls
    return "completed", calls


def _expect(cfg, state, exc_type, *, mention=()):
    outcome, calls = _run_round(cfg, state)
    assert isinstance(outcome, exc_type), f"expected {exc_type.__name__}, got {outcome!r}"
    assert calls == [], f"expensive work ran before the rejection: {calls}"
    for text in mention:
        assert text in str(outcome), f"{text!r} missing from message: {outcome}"


def _expect_passes(cfg, state):
    outcome, calls = _run_round(cfg, state)
    assert outcome == "reached", f"early checks should pass, got {outcome!r}"
    assert calls == ["gate"], f"only the gate may have been reached, got {calls}"


# ---- config validation -----------------------------------------------------

def test_P0_shared_constant_is_importable():
    assert c.LOGICAL_BATCH_SIZE is not None, "fix the LOGICAL_BATCH_SIZE import at the top of client.py"


def test_P1_valid_dp_config_passes_validation():
    c._validate_cfg(_dp_cfg(), for_train=True)


def test_P2_hooks_mode_fails_config_validation_with_valueerror():
    for mode in ("hooks", "ew"):
        try:
            c._validate_cfg(_dp_cfg(**{"grad-sample-mode": mode}), for_train=True)
        except ValueError as exc:
            assert "functorch" in str(exc)
        else:
            raise AssertionError(f"{mode!r} must be rejected")
    _expect(_dp_cfg(**{"grad-sample-mode": "hooks"}), None, ValueError, mention=("functorch",))


def test_P3_effective_must_equal_dp_logical_batch():
    cfg = _dp_cfg(**{"effective-batch-size": c.LOGICAL_BATCH_SIZE // 2})
    _expect(cfg, None, ValueError, mention=("effective-batch-size", "dp-logical-batch-size"))


def test_P4_dp_logical_batch_must_equal_the_shared_constant():
    other = c.LOGICAL_BATCH_SIZE // 2  # effective set equal, so only this rule can fire
    cfg = _dp_cfg(**{"dp-logical-batch-size": other, "effective-batch-size": other})
    _expect(cfg, None, ValueError, mention=("LOGICAL_BATCH_SIZE",))


def test_P5_non_dp_batch_rules_unchanged():
    base = {k: v for k, v in _dp_cfg().items() if k not in (
        "calibration-path", "max-grad-norm", "delta", "grad-sample-mode",
        "dp-logical-batch-size", "accountant", "target-epsilon")}
    base["dp"] = False
    c._validate_cfg(base, for_train=True)                       # 32 * 2 == 64 passes
    bad = {**base, "effective-batch-size": 128}
    try:
        c._validate_cfg(bad, for_train=True)
    except ValueError:
        pass
    else:
        raise AssertionError("non-DP effective != physical * accumulation must still fail")


# ---- early budget and round progress ---------------------------------------

def test_P6_valid_rounds_pass_the_early_checks():
    _expect_passes(_dp_cfg(round=1), None)                              # first round, no state
    _expect_passes(_dp_cfg(round=2), _state(PLANNED))                   # state carried forward
    _expect_passes(_dp_cfg(round=ROUNDS), _state(TOTAL - PLANNED))      # last round fits exactly


def test_P7_replayed_round_rejected_before_any_expensive_work():
    # Round 2 again, state already advanced to 2 rounds: 36 + 18 = 54 <= 180 so the
    # budget check alone would pass it -- the exact-progress check must catch it.
    _expect(_dp_cfg(round=2), _state(2 * PLANNED), c.PrivacyBudgetError,
            mention=(CLIENT, "round 2", "36", "18"))


def test_P8_overspent_state_still_rejected():
    _expect(_dp_cfg(round=ROUNDS), _state(TOTAL - PLANNED + 8), c.PrivacyBudgetError,
            mention=(CLIENT, f"round {ROUNDS}", str(TOTAL)))
    _expect(_dp_cfg(round=2), _state(10 * TOTAL), c.PrivacyBudgetError)


def test_P9_lost_or_stale_state_rejected():
    _expect(_dp_cfg(round=3), None, c.PrivacyBudgetError, mention=("0", str(2 * PLANNED)))  # lost
    _expect(_dp_cfg(round=1), _state(PLANNED), c.PrivacyBudgetError)                        # stale


def test_P10_malformed_state_is_never_read_as_zero_steps():
    malformed = [
        {"mechanism": "rdp"},                                         # no history
        {"history": "nope", "mechanism": "rdp"},
        {"history": [[1.1, 0.1]], "mechanism": "rdp"},                # entry too short
        {"history": [[1.1, 0.1, 18.0]], "mechanism": "rdp"},          # float steps
        {"history": [[1.1, 0.1, True]], "mechanism": "rdp"},          # bool steps
        {"history": [[1.1, 0.1, -3]], "mechanism": "rdp"},            # negative steps
        {"history": [[float("nan"), 0.1, 18]], "mechanism": "rdp"},   # non-finite noise
        {},                                                           # empty mapping
        ["not", "a", "mapping"],
    ]
    for state in malformed:
        _expect(_dp_cfg(round=2), state, ValueError)
        _expect(_dp_cfg(round=1), state, ValueError)


def test_P11_wrong_mechanism_rejected_early():
    _expect(_dp_cfg(round=2), _state(PLANNED, mechanism="prv"), ValueError, mention=("prv", "rdp"))


def test_P12_backstop_helper_arithmetic():
    cal, entry = _calibration(), _calibration()["clients"][CLIENT]
    for rnd in range(1, ROUNDS + 1):
        state = _state((rnd - 1) * PLANNED) if rnd > 1 else None
        assert c._check_dp_round_progress(
            client=CLIENT, true_round=rnd, cfg=_dp_cfg(round=rnd), calibration=cal,
            entry=entry, own_state=state) == (rnd - 1) * PLANNED


def test_P13_round_must_be_within_the_run():
    for rnd in (0, ROUNDS + 1):
        _expect(_dp_cfg(round=rnd), None, ValueError, mention=("round",))


def test_P14_history_entries_must_match_the_calibration():
    wrong_sigma = {"history": [[1.2, 1.0 / NUM_BATCHES, PLANNED]], "mechanism": "rdp"}
    wrong_rate = {"history": [[1.1, 0.5, PLANNED]], "mechanism": "rdp"}
    for state in (wrong_sigma, wrong_rate):
        _expect(_dp_cfg(round=2), state, (ValueError, c.PrivacyBudgetError))
    _expect_passes(_dp_cfg(round=2), _state(PLANNED))


def _run_states(cfg, states):
    calls = []

    def rec(name, exc=None):
        def _fn(*args, **kwargs):
            calls.append(name)
            if exc is not None:
                raise exc
            raise AssertionError(f"{name} must not be reached")
        return _fn

    with mock.patch.object(c, "load_calibration", return_value=_calibration()), \
         mock.patch.object(c, "_manifest_path", return_value=Path("unused.csv")), \
         mock.patch.object(c, "_build_model", rec("model")), \
         mock.patch.object(c, "get_dataloaders", rec("loader")), \
         mock.patch.object(c, "ensure_per_sample_gradients_verified", rec("gate", _Reached())):
        try:
            c.train_client_round(partition_id=PARTITION, cfg=cfg, global_state={}, accountant_states=states)
        except _Reached:
            return "reached", calls
        except Exception as exc:  # noqa: BLE001
            return exc, calls
    return "completed", calls


def test_P15_round_one_accepts_only_an_absent_or_empty_state_collection():
    rejected = ({"0": _state(PLANNED)}, {str(PARTITION): _state(0)}, {str(PARTITION): _state(PLANNED)})
    for states in rejected:
        outcome, calls = _run_states(_dp_cfg(round=1), states)
        assert isinstance(outcome, c.PrivacyBudgetError), outcome
        assert calls == []
    for absent in (None, {}):
        outcome, calls = _run_states(_dp_cfg(round=1), absent)
        assert outcome == "reached" and calls == ["gate"], (outcome, calls)


def test_P16_later_rounds_require_this_clients_own_entry():
    for states in ({"0": _state(PLANNED)}, {}, None):
        outcome, calls = _run_states(_dp_cfg(round=2), states)
        assert isinstance(outcome, (ValueError, c.PrivacyBudgetError)), outcome
        assert calls == []


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)
