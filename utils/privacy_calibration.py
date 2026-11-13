"""
utils/privacy_calibration.py

Per-client noise-multiplier calibration, per training-phase-decisions.md
Section 4 and the discussion chat's patch spec (schema v1 -> v2).

Scope, deliberately narrow: this file answers ONE question -- "what
noise_multiplier does client X need to hit target epsilon E, given its
own dataset size" -- and persists the answer. It does NOT construct a
PrivacyEngine, a DPOptimizer, or an accountant; that happens in
federated/client.py (not yet written), using this file's output as one
input to its own PrivacyEngine(accountant=...).make_private(...) call.

PATCHED (this version) -- two real, confirmed bugs fixed:

1. sample_rate was computed as logical_batch_size / dataset_size (and
   before that, from a parameter literally named physical_batch_size --
   an even earlier, worse version of the same bug). Both are wrong.
   Opacus's DP loader is built at the LOGICAL batch size with
   drop_last=False, which means num_batches = ceil(N / B), and the
   actual Poisson sampling rate the accounting is based on is
   q = 1 / num_batches -- NOT B / N directly. These differ whenever N
   isn't an exact multiple of B, and the gap is NOT negligible for a
   small client: IDRiD (N=330, B=64) gets num_batches=6 and
   q=0.16667, versus the wrong B/N=0.19394 -- a real, meaningful
   difference, not a rounding nicety. The gap shrinks toward
   negligible for large N (EyePACS, N=70960: q=1/1109=0.0009019 either
   way, to 4 significant figures) -- which is exactly why this bug was
   easy to miss: it looks fine on the large client and wrong on the
   small one. Fixed: num_batches_for()/sample_rate_for() below compute
   both correctly, and are the ONLY place sample_rate is ever derived
   anywhere in this file. get_noise_multiplier() is now called with
   steps= (an exact integer, epochs * num_batches) instead of epochs=,
   since Opacus's internal epochs-to-steps conversion
   (int(epochs / sample_rate)) can silently truncate a step when
   sample_rate isn't an exact reciprocal of an integer.

2. accountant had a default value ("rdp"), justified by a FALSE claim
   that "rdp" is Opacus's own default. It isn't, uniformly: Opacus's
   PrivacyEngine defaults to "prv", while get_noise_multiplier()
   defaults to "rdp" -- two DIFFERENT mechanisms, each the default of a
   DIFFERENT Opacus API. A caller relying on either function's own
   default, independently, could silently end up calibrating with one
   mechanism and running with another -- PRV can report a tighter
   epsilon than RDP for the same actual noise, so mixing them would
   make a reported epsilon meaningless without anyone necessarily
   noticing. Fixed: accountant is now a REQUIRED keyword, everywhere,
   with no default and no module-level constant -- every caller must
   say explicitly which mechanism it means. The project intends "rdp"
   unless the discussion chat decides otherwise, but that's a project
   decision for callers to pass in, not something this file should
   assume on their behalf. The accountant string a calibration file was
   generated with is recorded in that file (save_calibration's
   `accountant` field) specifically so federated/client.py can build
   its PrivacyEngine from that SAME recorded string
   (PrivacyEngine(accountant=calibration["accountant"])) rather than
   choosing one independently -- structurally preventing the mismatch,
   not just documenting the risk.

   RDP (Renyi DP) is the project's intended accountant for both
   calibration and runtime, pending a different decision from the
   discussion chat. PRV can report a tighter (smaller) epsilon than RDP
   for identical noise/sample_rate/steps -- using different mechanisms
   for calibration vs. runtime would make the reported epsilon
   internally inconsistent, independent of whether either number is
   individually "correct" for its own mechanism.

Also fixed/added, per the same patch:
- logical_batch_size is required everywhere (no default), sourced from
  the project's base config (key dp_logical_batch_size) -- the same
  value preprocessing/dataset.py's DP loader reads, so the two can
  never silently diverge as long as both read the same config key.
  check_loader_matches_calibration() below is the runtime-side half of
  that guarantee: it compares an ACTUAL Opacus DP loader against a
  calibration entry before training starts, rather than trusting that
  whatever built the loader used the same numbers calibration assumed.
- Every public entry point now validates its own inputs explicitly
  (real ints where ints are expected -- bool is a subclass of int in
  Python and is explicitly rejected, not silently accepted -- finite
  numbers, correct ranges) and raises ValueError/CalibrationError with
  a specific message, never asserts.
- load_calibration() now validates the loaded file's CONTENT, not just
  its schema_version -- exact top-level and per-client key sets, types,
  ranges, and the same cross-checks (num_batches == ceil(N/B),
  steps == epochs*num_batches, sample_rate == 1/num_batches) a freshly
  computed calibration already satisfies by construction. A hand-edited
  or corrupted calibration file fails loudly, with every problem listed
  at once, rather than being silently trusted.
- save_calibration()'s per-client field is now "clients" (a richer
  per-client dict: noise_multiplier, sample_rate, num_batches, steps,
  dataset_size), replacing the old "client_noise_multipliers" field
  (a bare client -> noise_multiplier mapping) -- the old key must not
  exist in a v2 file. metadata.opacus_version is always set from the
  actually-installed Opacus version at save time, overriding any value
  a caller might (accidentally or not) supply -- a stored version
  string that could lie about what actually produced the file would
  defeat the point of recording it.
- save_calibration()'s atomic JSON write now flushes and fsyncs before
  the rename, same durability reasoning as checkpoint.py's atomic
  write (NFS can hold data in client-side cache past a write() return).

Explicitly OUT of scope (federated/client.py's job, not this file's):
constructing PrivacyEngine or any DPOptimizer; verifying the loaded
calibration matches the experiment's actual config at startup beyond
what check_loader_matches_calibration() checks about the loader itself;
the per-sample-gradient gate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationError",
    "num_batches_for",
    "sample_rate_for",
    "compute_noise_multiplier",
    "calibrate_client_noise_multipliers",
    "calibrate_and_save_all",
    "save_calibration",
    "load_calibration",
    "check_loader_matches_calibration",
]

TARGET_DELTA = 1e-5  # locked globally, Section 4
FL_ROUNDS = 10
LOCAL_EPOCHS_PER_ROUND = 3
GLOBAL_TRAINING_EPOCHS = FL_ROUNDS * LOCAL_EPOCHS_PER_ROUND  # 30 -- fixed, same for every client
TARGET_EPSILONS: Tuple[float, ...] = (50.0, 10.0, 5.0, 1.0, 0.5)  # Experiments 4-8, locked order

SCHEMA_VERSION = 2

_TOP_LEVEL_KEYS = {
    "schema_version", "target_epsilon", "target_delta", "accountant",
    "logical_batch_size", "epochs", "epsilon_tolerance", "clients", "metadata",
}
_CLIENT_KEYS = {"noise_multiplier", "sample_rate", "num_batches", "steps", "dataset_size"}


class CalibrationError(RuntimeError):
    """Raised when Opacus's calibration API can't be found or used as
    expected, when a saved calibration file fails content validation, or
    when a real DP loader doesn't match its calibration entry. Loud
    rather than silently falling back to a guessed replacement -- a
    wrong noise_multiplier corrupts the privacy guarantee of whichever
    experiment uses it, invisibly.
    """


def _get_noise_multiplier_fn():
    """Import opacus.accountants.utils.get_noise_multiplier lazily, with
    an actionable error if it's moved. Imported lazily so this module
    stays importable for syntax-level editing in an environment without
    Opacus, same pattern used throughout this project.
    """
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return get_noise_multiplier
    except ImportError as exc:
        raise CalibrationError(
            "Could not import get_noise_multiplier from "
            "opacus.accountants.utils. If this import path has moved in "
            "the installed Opacus version, verify the new location "
            "against installed source before proceeding -- do not guess "
            "a replacement path; a silently wrong calibration function "
            "would corrupt every DP experiment's privacy guarantee "
            "invisibly."
        ) from exc


def _is_strict_int(value: Any) -> bool:
    """True only for a real int -- excludes bool (a subclass of int in
    Python: isinstance(True, int) is True) and float.
    """
    return type(value) is int


def _is_number(value: Any) -> bool:
    """True for int or float, explicitly excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(label: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return value


def _require_positive_int(label: str, value: Any) -> int:
    if not _is_strict_int(value) or not value > 0:
        raise ValueError(f"{label} must be an int > 0 (not bool/float), got {value!r}")
    return value


def _require_accountant(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"accountant must be a non-empty string, got {value!r}")
    return value


def num_batches_for(dataset_size: int, logical_batch_size: int) -> int:
    """ceil(dataset_size / logical_batch_size) -- the number of batches
    Opacus's DP loader (built with drop_last=False) actually produces
    for one epoch over this many examples. The single source of truth
    for this calculation -- every other function in this file that
    needs a batch count calls this rather than recomputing it inline.
    """
    _require_positive_int("dataset_size", dataset_size)
    _require_positive_int("logical_batch_size", logical_batch_size)
    return math.ceil(dataset_size / logical_batch_size)


def sample_rate_for(dataset_size: int, logical_batch_size: int) -> float:
    """1 / num_batches_for(...) -- the actual Poisson sampling rate
    Opacus's privacy accounting is based on. NOT logical_batch_size /
    dataset_size -- see this module's docstring (fix 1) for why those
    differ, and by how much, for a small client.
    """
    return 1.0 / num_batches_for(dataset_size, logical_batch_size)


def compute_noise_multiplier(
    *,
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    accountant: str,
    target_delta: float = TARGET_DELTA,
    epsilon_tolerance: float = 0.01,
) -> float:
    """One client's noise_multiplier for one target epsilon, at an
    EXACT step count. Thin wrapper around Opacus's own calibration
    search -- this file does not reimplement or approximate that search
    itself.

    Args:
        target_epsilon: the epsilon this client should land at after
            `steps` real optimizer steps at `sample_rate`.
        sample_rate: from sample_rate_for() -- 1/num_batches, not
            batch_size/dataset_size.
        steps: the EXACT total step count (epochs * num_batches, an
            integer multiplication) -- passed as steps=, never epochs=,
            since Opacus's internal epochs-to-steps conversion can
            silently truncate a step when sample_rate isn't an exact
            reciprocal of an integer (module docstring, fix 1).
        accountant: REQUIRED, no default (module docstring, fix 2) --
            "rdp" is the project's intent, but this file will not
            assume it on a caller's behalf.
        target_delta: locked at 1e-5 globally.
        epsilon_tolerance: precision of Opacus's internal search.

    Raises:
        ValueError: any input fails validation.
        CalibrationError: Opacus's own search fails (target_epsilon is
            unachievable for this sample_rate/steps/delta combination --
            not enough steps, or too small a sample_rate, for any noise
            level to hit such a tight budget) -- converted from
            Opacus's internal ValueError, with the attempted values
            included, rather than letting a context-free error surface.
    """
    _require_number("target_epsilon", target_epsilon)
    if not target_epsilon > 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon!r}")
    _require_number("target_delta", target_delta)
    if not (0 < target_delta < 1):
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta!r}")
    _require_number("sample_rate", sample_rate)
    if not (0 < sample_rate <= 1):
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate!r}")
    _require_positive_int("steps", steps)
    _require_accountant(accountant)
    _require_number("epsilon_tolerance", epsilon_tolerance)
    if not epsilon_tolerance > 0:
        raise ValueError(f"epsilon_tolerance must be > 0, got {epsilon_tolerance!r}")

    get_noise_multiplier = _get_noise_multiplier_fn()
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=accountant,
            epsilon_tolerance=epsilon_tolerance,
        )
    except ValueError as exc:
        raise CalibrationError(
            f"Opacus could not find a noise_multiplier achieving "
            f"target_epsilon={target_epsilon} (target_delta={target_delta}, "
            f"sample_rate={sample_rate!r}, steps={steps}, "
            f"accountant={accountant!r}). This usually means the "
            f"requested epsilon is too tight for this sample_rate/steps "
            f"combination -- not enough steps, or too small a "
            f"sample_rate, for any noise level to hit such a tight "
            f"budget. Consider more epochs or a looser target_epsilon. "
            f"Original error: {exc}"
        ) from exc

    logger.debug(
        "compute_noise_multiplier: target_epsilon=%s sample_rate=%.6f "
        "steps=%d accountant=%s -> noise_multiplier=%.6f",
        target_epsilon, sample_rate, steps, accountant, noise_multiplier,
    )
    return noise_multiplier


def calibrate_client_noise_multipliers(
    *,
    target_epsilon: float,
    client_train_sizes: Dict[str, int],
    accountant: str,
    logical_batch_size: int,
    epochs: int = GLOBAL_TRAINING_EPOCHS,
    target_delta: float = TARGET_DELTA,
    epsilon_tolerance: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    """Per-client calibration for ONE shared target_epsilon -- the
    actual function federated experiments 4-8 each call once, at their
    true start (see checkpoint.py's run_config/expected_config for how
    a resume avoids recalibrating).

    client_train_sizes: TRAIN-split sizes only (not val/test) -- e.g.
        {"aptos": 2930, "idrid": 330, "messidor2": 1396,
         "eyepacs": 70960}. This file doesn't compute these itself;
        supplying them is the caller's job.

    Returns a dict keyed by client_id, each value EXACTLY:
        {"noise_multiplier": float, "sample_rate": float,
         "num_batches": int, "steps": int, "dataset_size": int}
    """
    _require_number("target_epsilon", target_epsilon)
    if not target_epsilon > 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon!r}")
    _require_accountant(accountant)
    _require_positive_int("logical_batch_size", logical_batch_size)
    _require_positive_int("epochs", epochs)
    _require_number("target_delta", target_delta)
    if not (0 < target_delta < 1):
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta!r}")
    _require_number("epsilon_tolerance", epsilon_tolerance)
    if not epsilon_tolerance > 0:
        raise ValueError(f"epsilon_tolerance must be > 0, got {epsilon_tolerance!r}")
    if not client_train_sizes:
        raise ValueError("client_train_sizes must be non-empty.")

    results: Dict[str, Dict[str, Any]] = {}
    for client_id, dataset_size in client_train_sizes.items():
        _require_positive_int(f"client_train_sizes[{client_id!r}]", dataset_size)

        num_batches = num_batches_for(dataset_size, logical_batch_size)
        sample_rate = sample_rate_for(dataset_size, logical_batch_size)
        steps = epochs * num_batches

        # Defensive, not decorative: these hold by construction (that's
        # literally how num_batches/sample_rate/steps were just
        # computed), but checking explicitly catches a future refactor
        # that silently breaks the relationship, rather than letting a
        # wrong pairing reach get_noise_multiplier unnoticed.
        if steps != epochs * num_batches:
            raise ValueError(
                f"Internal consistency check failed for client "
                f"{client_id!r}: steps ({steps}) != epochs*num_batches "
                f"({epochs * num_batches}). Should be impossible by "
                f"construction -- investigate before proceeding."
            )
        if sample_rate != 1.0 / num_batches:
            raise ValueError(
                f"Internal consistency check failed for client "
                f"{client_id!r}: sample_rate ({sample_rate!r}) != "
                f"1/num_batches ({1.0 / num_batches!r}). Should be "
                f"impossible by construction -- investigate before "
                f"proceeding."
            )

        noise_multiplier = compute_noise_multiplier(
            target_epsilon=target_epsilon,
            sample_rate=sample_rate,
            steps=steps,
            accountant=accountant,
            target_delta=target_delta,
            epsilon_tolerance=epsilon_tolerance,
        )

        results[client_id] = {
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "num_batches": num_batches,
            "steps": steps,
            "dataset_size": dataset_size,
        }
        logger.info(
            "calibrate_client_noise_multipliers: client=%s dataset_size=%d "
            "num_batches=%d sample_rate=%.6f steps=%d -> "
            "noise_multiplier=%.4f (target_epsilon=%s, accountant=%s)",
            client_id, dataset_size, num_batches, sample_rate, steps,
            noise_multiplier, target_epsilon, accountant,
        )
    return results


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            # Destination may be an NFS volume (same reasoning as
            # checkpoint.py's atomic write) -- flush and fsync before
            # the rename so a kill shortly after a "completed" save
            # can't lose data that was never actually pushed to the
            # NFS server.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(destination))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def save_calibration(
    path: str,
    *,
    target_epsilon: float,
    clients: Dict[str, Dict[str, Any]],
    accountant: str,
    logical_batch_size: int,
    epochs: int,
    target_delta: float,
    epsilon_tolerance: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist one experiment's calibration result as JSON. Same atomic
    temp-file-then-os.replace() pattern as checkpoint.py.

    Top-level JSON keys, exactly: schema_version, target_epsilon,
    target_delta, accountant, logical_batch_size, epochs,
    epsilon_tolerance, clients, metadata. The per-client field is
    "clients" (keyed by client_id, each a full calibrate_client_
    noise_multipliers()-shaped dict) -- the old "client_noise_
    multipliers" key (a bare client -> noise_multiplier mapping) does
    not exist in a v2 file.

    metadata: the caller's dict is merged first, then "opacus_version"
    is set LAST from the actually-installed Opacus version at save
    time ("not installed" if the package isn't present) -- this
    overrides any opacus_version the caller might have supplied. A
    stored version string that could be spoofed by the caller would
    defeat the point of recording it for provenance.

    Every DP experiment (4-8) should call this exactly once per target
    epsilon, at that experiment's true start, and load_calibration() on
    every subsequent resume rather than recalibrating.
    """
    _require_number("target_epsilon", target_epsilon)
    _require_accountant(accountant)
    _require_positive_int("logical_batch_size", logical_batch_size)
    _require_positive_int("epochs", epochs)
    _require_number("target_delta", target_delta)
    _require_number("epsilon_tolerance", epsilon_tolerance)
    if not clients:
        raise ValueError("clients must be non-empty.")

    try:
        import importlib.metadata
        opacus_version = importlib.metadata.version("opacus")
    except Exception:
        opacus_version = "not installed"

    merged_metadata = dict(metadata or {})
    merged_metadata["opacus_version"] = opacus_version  # set LAST -- overrides any caller value

    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_epsilon": target_epsilon,
        "target_delta": target_delta,
        "accountant": accountant,
        "logical_batch_size": logical_batch_size,
        "epochs": epochs,
        "epsilon_tolerance": epsilon_tolerance,
        "clients": clients,
        "metadata": merged_metadata,
    }
    _atomic_write_json(path, payload)
    logger.info(
        "save_calibration: wrote %s (target_epsilon=%s, accountant=%s, %d clients)",
        path, target_epsilon, accountant, len(clients),
    )


def load_calibration(path: str) -> Dict[str, Any]:
    """Load and FULLY validate a calibration file written by
    save_calibration() -- not just its schema_version.

    Raises CalibrationError listing EVERY problem found (not just the
    first) if any of the following don't hold: schema_version == 2;
    exact top-level key set; exact per-client key set (the five fields
    listed in calibrate_client_noise_multipliers()'s docstring); every
    numeric field is the right type (ints are real ints -- bool and
    float are both explicitly rejected where an int is expected) and
    finite; target_epsilon > 0; 0 < target_delta < 1;
    epsilon_tolerance > 0; accountant is a non-empty string; and, per
    client: noise_multiplier > 0; sample_rate == 1/num_batches (within
    1e-15); steps == epochs*num_batches; num_batches ==
    ceil(dataset_size/logical_batch_size). A hand-edited or corrupted
    file fails loudly and completely, rather than being silently
    trusted for whichever fields happen to look reasonable.

    Returns the full validated payload dict.
    """
    with open(path, "r") as f:
        payload = json.load(f)

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError(
            f"Calibration file at {path} has schema_version="
            f"{payload.get('schema_version')!r}, but this code expects "
            f"{SCHEMA_VERSION}. A v1 file (bare client -> noise_multiplier "
            f"mapping under 'client_noise_multipliers', no accountant/"
            f"logical_batch_size/epochs/epsilon_tolerance fields) is NOT "
            f"compatible with this v2 loader -- do not force-load it; "
            f"regenerate with the current code."
        )

    problems: List[str] = []

    actual_top_keys = set(payload.keys())
    if actual_top_keys != _TOP_LEVEL_KEYS:
        missing = _TOP_LEVEL_KEYS - actual_top_keys
        unexpected = actual_top_keys - _TOP_LEVEL_KEYS
        if missing:
            problems.append(f"missing top-level key(s): {sorted(missing)}")
        if unexpected:
            problems.append(f"unexpected top-level key(s): {sorted(unexpected)}")

    target_epsilon = payload.get("target_epsilon")
    if not _is_number(target_epsilon) or not math.isfinite(target_epsilon):
        problems.append(f"target_epsilon: expected a finite number, got {target_epsilon!r}")
    elif not target_epsilon > 0:
        problems.append(f"target_epsilon: must be > 0, got {target_epsilon!r}")

    target_delta = payload.get("target_delta")
    if not _is_number(target_delta) or not math.isfinite(target_delta):
        problems.append(f"target_delta: expected a finite number, got {target_delta!r}")
    elif not (0 < target_delta < 1):
        problems.append(f"target_delta: must be in (0, 1), got {target_delta!r}")

    epsilon_tolerance = payload.get("epsilon_tolerance")
    if not _is_number(epsilon_tolerance) or not math.isfinite(epsilon_tolerance):
        problems.append(f"epsilon_tolerance: expected a finite number, got {epsilon_tolerance!r}")
    elif not epsilon_tolerance > 0:
        problems.append(f"epsilon_tolerance: must be > 0, got {epsilon_tolerance!r}")

    accountant = payload.get("accountant")
    if not isinstance(accountant, str) or not accountant:
        problems.append(f"accountant: expected a non-empty string, got {accountant!r}")

    logical_batch_size = payload.get("logical_batch_size")
    if not _is_strict_int(logical_batch_size) or not logical_batch_size > 0:
        problems.append(f"logical_batch_size: expected an int > 0, got {logical_batch_size!r}")

    epochs = payload.get("epochs")
    if not _is_strict_int(epochs) or not epochs > 0:
        problems.append(f"epochs: expected an int > 0, got {epochs!r}")

    can_cross_check = (
        _is_strict_int(logical_batch_size) and logical_batch_size > 0
        and _is_strict_int(epochs) and epochs > 0
    )

    clients = payload.get("clients")
    if not isinstance(clients, dict) or not clients:
        problems.append(f"clients: expected a non-empty dict, got {type(clients).__name__}")
    else:
        for client_id, entry in clients.items():
            if not isinstance(entry, dict):
                problems.append(f"clients[{client_id!r}]: expected a dict, got {type(entry).__name__}")
                continue
            actual_client_keys = set(entry.keys())
            if actual_client_keys != _CLIENT_KEYS:
                c_missing = _CLIENT_KEYS - actual_client_keys
                c_unexpected = actual_client_keys - _CLIENT_KEYS
                if c_missing:
                    problems.append(f"clients[{client_id!r}]: missing key(s) {sorted(c_missing)}")
                if c_unexpected:
                    problems.append(f"clients[{client_id!r}]: unexpected key(s) {sorted(c_unexpected)}")
                continue  # remaining checks need all five keys present

            noise_multiplier = entry["noise_multiplier"]
            sample_rate = entry["sample_rate"]
            num_batches = entry["num_batches"]
            steps = entry["steps"]
            dataset_size = entry["dataset_size"]

            nm_ok = _is_number(noise_multiplier) and math.isfinite(noise_multiplier) and noise_multiplier > 0
            if not nm_ok:
                problems.append(
                    f"clients[{client_id!r}].noise_multiplier: expected a "
                    f"finite number > 0, got {noise_multiplier!r}"
                )
            sr_ok = _is_number(sample_rate) and math.isfinite(sample_rate) and 0 < sample_rate <= 1
            if not sr_ok:
                problems.append(
                    f"clients[{client_id!r}].sample_rate: expected a "
                    f"finite number in (0, 1], got {sample_rate!r}"
                )
            nb_ok = _is_strict_int(num_batches) and num_batches > 0
            if not nb_ok:
                problems.append(
                    f"clients[{client_id!r}].num_batches: expected an int "
                    f"> 0, got {num_batches!r}"
                )
            st_ok = _is_strict_int(steps) and steps > 0
            if not st_ok:
                problems.append(
                    f"clients[{client_id!r}].steps: expected an int > 0, "
                    f"got {steps!r}"
                )
            ds_ok = _is_strict_int(dataset_size) and dataset_size > 0
            if not ds_ok:
                problems.append(
                    f"clients[{client_id!r}].dataset_size: expected an "
                    f"int > 0, got {dataset_size!r}"
                )

            if can_cross_check and nb_ok and ds_ok:
                expected_num_batches = num_batches_for(dataset_size, logical_batch_size)
                if num_batches != expected_num_batches:
                    problems.append(
                        f"clients[{client_id!r}].num_batches ({num_batches}) "
                        f"!= ceil(dataset_size/logical_batch_size) "
                        f"({expected_num_batches})"
                    )
                if st_ok:
                    expected_steps = epochs * num_batches
                    if steps != expected_steps:
                        problems.append(
                            f"clients[{client_id!r}].steps ({steps}) != "
                            f"epochs*num_batches ({expected_steps})"
                        )
                if sr_ok:
                    expected_sample_rate = 1.0 / num_batches
                    if abs(sample_rate - expected_sample_rate) > 1e-15:
                        problems.append(
                            f"clients[{client_id!r}].sample_rate "
                            f"({sample_rate!r}) != 1/num_batches "
                            f"({expected_sample_rate!r}) within 1e-15"
                        )

    if problems:
        raise CalibrationError(
            f"Calibration file at {path} failed content validation with "
            f"{len(problems)} problem(s):\n  " + "\n  ".join(problems)
        )

    return payload


def check_loader_matches_calibration(dp_loader, client_entry: Dict[str, Any]) -> None:
    """Verify an ACTUAL Opacus DP loader matches a calibration entry
    before training starts, rather than trusting that whatever built
    the loader used the same numbers calibration assumed.

    Checks three things, all of which must hold:
      - len(dp_loader.dataset) == client_entry["dataset_size"]. This
        matters on its own, separately from the other two: two
        different clients can share the same ceil(N/B)/sample_rate
        pair despite having different N (e.g. N=320 and N=330 can round
        to the same num_batches at B=64) -- matching sample_rate/
        num_batches alone would NOT catch a loader accidentally built
        from the wrong client's data.
      - len(dp_loader) == client_entry["num_batches"].
      - abs(dp_loader.sample_rate - client_entry["sample_rate"]) < 1e-12.

    Raises CalibrationError listing every mismatch found (not just the
    first) if any of the three don't hold -- training with a mismatched
    loader would silently invalidate the calibrated noise_multiplier's
    privacy guarantee, with no other symptom.
    """
    problems: List[str] = []

    actual_dataset_size = len(dp_loader.dataset)
    if actual_dataset_size != client_entry["dataset_size"]:
        problems.append(
            f"dataset size mismatch: loader has {actual_dataset_size}, "
            f"calibration expects {client_entry['dataset_size']}"
        )

    actual_num_batches = len(dp_loader)
    if actual_num_batches != client_entry["num_batches"]:
        problems.append(
            f"num_batches mismatch: loader has {actual_num_batches}, "
            f"calibration expects {client_entry['num_batches']}"
        )

    actual_sample_rate = dp_loader.sample_rate
    diff = abs(actual_sample_rate - client_entry["sample_rate"])
    if diff >= 1e-12:
        problems.append(
            f"sample_rate mismatch: loader has {actual_sample_rate!r}, "
            f"calibration expects {client_entry['sample_rate']!r} "
            f"(diff {diff!r})"
        )

    if problems:
        raise CalibrationError(
            "DP loader does not match its calibration entry -- training "
            "with it would silently invalidate the calibrated "
            "noise_multiplier's privacy guarantee. Problems:\n  "
            + "\n  ".join(problems)
        )


def calibrate_and_save_all(
    output_dir: str,
    *,
    client_train_sizes: Dict[str, int],
    accountant: str,
    logical_batch_size: int,
    epochs: int = GLOBAL_TRAINING_EPOCHS,
    target_delta: float = TARGET_DELTA,
    epsilon_tolerance: float = 0.01,
) -> Dict[float, Dict[str, Dict[str, Any]]]:
    """Convenience sweep over all 5 locked target epsilons
    (TARGET_EPSILONS), writing one calibration file per epsilon into
    output_dir. Meant as a standalone pre-experiment sanity pass -- not
    the call path an actual experiment run should use (that's
    load_calibration() on the one already-computed file for whichever
    target_epsilon that specific experiment is using).
    """
    results: Dict[float, Dict[str, Dict[str, Any]]] = {}
    table_rows = []
    for target_epsilon in TARGET_EPSILONS:
        clients = calibrate_client_noise_multipliers(
            target_epsilon=target_epsilon,
            client_train_sizes=client_train_sizes,
            accountant=accountant,
            logical_batch_size=logical_batch_size,
            epochs=epochs,
            target_delta=target_delta,
            epsilon_tolerance=epsilon_tolerance,
        )
        results[target_epsilon] = clients
        save_calibration(
            str(Path(output_dir) / f"epsilon_{target_epsilon}.json"),
            target_epsilon=target_epsilon,
            clients=clients,
            accountant=accountant,
            logical_batch_size=logical_batch_size,
            epochs=epochs,
            target_delta=target_delta,
            epsilon_tolerance=epsilon_tolerance,
        )
        for client_id, entry in clients.items():
            table_rows.append((target_epsilon, client_id, entry["noise_multiplier"]))

    logger.info(
        "calibrate_and_save_all: noise-multiplier table (target_epsilon, "
        "client, noise_multiplier), %d values:\n%s",
        len(table_rows),
        "\n".join(f"  {eps}\t{cid}\t{nm:.4f}" for eps, cid, nm in table_rows),
    )
    return results


def _achieved_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, accountant: str, target_delta: float
) -> float:
    """Smoke-test-only helper, NOT part of the public interface --
    computes the actual epsilon a calibrated noise_multiplier achieves,
    independent of get_noise_multiplier()'s own internal search, by
    building a fresh accountant and looping its public .step() method
    the full `steps` times (slower than manipulating internal history
    directly, but unambiguously the documented, correct way to use an
    accountant -- worth the extra confidence for a verification step).

    NOTE: opacus.accountants.create's exact import path/signature is
    NOT independently verified against the installed Opacus version in
    this environment -- re-confirm when this smoke test is actually
    run.
    """
    from opacus.accountants import create_accountant
    acc = create_accountant(mechanism=accountant)
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acc.get_epsilon(delta=target_delta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Real train-split sizes (preprocessing phase), not placeholders.
    real_train_sizes = {
        "aptos": 2930,
        "idrid": 330,
        "messidor2": 1396,
        "eyepacs": 70960,
    }
    _target_epsilon = 5.0
    _accountant = "rdp"
    _logical_batch_size = 64
    _epsilon_tolerance = 0.01

    result_1 = calibrate_client_noise_multipliers(
        target_epsilon=_target_epsilon,
        client_train_sizes=real_train_sizes,
        accountant=_accountant,
        logical_batch_size=_logical_batch_size,
        epsilon_tolerance=_epsilon_tolerance,
    )
    result_2 = calibrate_client_noise_multipliers(
        target_epsilon=_target_epsilon,
        client_train_sizes=real_train_sizes,
        accountant=_accountant,
        logical_batch_size=_logical_batch_size,
        epsilon_tolerance=_epsilon_tolerance,
    )
    assert result_1 == result_2, (
        f"Non-deterministic calibration result across two identical "
        f"calls: {result_1} vs {result_2}. The resume-safety design "
        f"this project relies on (calibrate once, persist, reload on "
        f"resume rather than recompute) assumes determinism -- if this "
        f"fails, that assumption needs re-examining."
    )

    for client_id, entry in result_1.items():
        achieved = _achieved_epsilon(
            entry["noise_multiplier"], entry["sample_rate"], entry["steps"],
            _accountant, TARGET_DELTA,
        )
        assert abs(achieved - _target_epsilon) <= _epsilon_tolerance, (
            f"client={client_id}: achieved epsilon {achieved:.4f} is more "
            f"than epsilon_tolerance ({_epsilon_tolerance}) away from "
            f"target {_target_epsilon} -- using the REAL sample_rate "
            f"({entry['sample_rate']:.6f}) and steps ({entry['steps']}) "
            f"this client actually calibrated with."
        )
        logger.info(
            "Smoke test: client=%s target_epsilon=%s achieved_epsilon=%.4f "
            "noise_multiplier=%.4f sample_rate=%.6f steps=%d",
            client_id, _target_epsilon, achieved, entry["noise_multiplier"],
            entry["sample_rate"], entry["steps"],
        )

    logger.info("Smoke test passed: %s", result_1)