"""
configs/loader.py: resolve, validate and render the experiment configs.

base.yaml holds everything identical across experiments; exp{N}.yaml holds the
per-experiment overrides only (algorithm, dp, target-epsilon, proximal-mu;
experiment-id and calibration-path are derived). Keys are flat and hyphenated,
exactly as federated.server reads them.

  load_experiment_config(n)      resolved flat dict (experiments 1..8)
  to_centralized_kwargs(cfg)     Experiment 1 -> run_centralized_training kwargs
  check_comparability()          experiments 2..8 differ only where allowed
  render_run_config_toml(cfg)    the TOML that `flwr run --run-config` takes
  main()                         CLI: python configs/loader.py N --client-cpus C ...

Refused, never defaulted: a null step-zero value (lora-rank, lora-alpha,
max-grad-norm), and missing pod inputs (manifest-root, checkpoint-root, which
come from the CLI or DR_MANIFEST_ROOT / DR_CHECKPOINT_ROOT; the Ray client CPU
count, CLI or DR_CLIENT_CPUS). allow_unset=True (tests only) skips those
refusals and the server/client validation.

Module level imports only the stdlib and yaml; federated.server, tomllib/tomli
are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

__all__ = [
    "CONFIG_DIR", "RUN_HOOK", "CLIENTS", "STEP_ZERO_KEYS", "POD_KEYS", "OVERRIDE_WHITELIST", "EXP_FILE_ALLOWED",
    "DP_ONLY_KEYS", "FL_ONLY_KEYS", "CENTRALIZED_ONLY_KEYS", "EXPECTED", "ConfigError", "ComparabilityError",
    "experiment_id_for", "load_experiment_config", "to_centralized_kwargs", "check_comparability",
    "render_run_config_toml", "preflight_pyproject", "flwr_run_command", "federation_setup_text", "main",
]

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
RUN_HOOK = "utils.experiment_tracker:make_hook"
CLIENTS = ("APTOS", "IDRiD", "Messidor-2", "EyePACS")  # same order as the server's partitions

STEP_ZERO_KEYS = ("lora-rank", "lora-alpha", "max-grad-norm")
POD_KEYS = ("manifest-root", "checkpoint-root")
_POD_ENV = {"manifest-root": "DR_MANIFEST_ROOT", "checkpoint-root": "DR_CHECKPOINT_ROOT"}
CPU_ENV = "DR_CLIENT_CPUS"

OVERRIDE_WHITELIST = frozenset({
    "manifest-root", "checkpoint-root", "num-workers", "run-seed", "num-rounds", "wandb-cm-every", "diagnostic"})
EXP_FILE_ALLOWED = frozenset({
    "algorithm", "dp", "target-epsilon", "proximal-mu", "experiment-id", "calibration-path"})

FL_ONLY_KEYS = frozenset({
    "num-rounds", "local-epochs", "warmup-rounds", "warmup-start-factor", "lr-min-factor", "timeout",
    "epsilon-tolerance"})
DP_ONLY_KEYS = frozenset({
    "calibration-path", "target-epsilon", "accountant", "delta", "max-grad-norm", "grad-sample-mode",
    "dp-logical-batch-size"})
CENTRALIZED_ONLY_KEYS = frozenset({
    "centralized-max-epochs", "centralized-patience", "centralized-warmup-epochs",
    "centralized-warmup-start-factor"})
_SHARED_KEYS = frozenset({
    "run-seed", "img-size", "num-classes", "pretrained", "physical-batch-size", "accumulation-steps",
    "effective-batch-size", "label-smoothing", "weight-decay", "num-workers", "lr-lora-peak", "lr-head-peak",
    "lora-rank", "lora-alpha", "wandb-cm-every", "diagnostic", "dp"})
BASE_ALLOWED = _SHARED_KEYS | FL_ONLY_KEYS | CENTRALIZED_ONLY_KEYS | (DP_ONLY_KEYS - {"target-epsilon", "calibration-path"})

# (algorithm, dp, target epsilon) per experiment; used by check_comparability.
EXPECTED: Dict[int, Tuple[str, bool, Optional[float]]] = {
    1: ("centralized", False, None), 2: ("fedavg", False, None), 3: ("fedprox", False, None),
    4: ("fedprox", True, 50.0), 5: ("fedprox", True, 10.0), 6: ("fedprox", True, 5.0),
    7: ("fedprox", True, 1.0), 8: ("fedprox", True, 0.5),
}
_COMPARABLE_DIFF = frozenset({"experiment-id", "algorithm", "dp", "proximal-mu"}) | DP_ONLY_KEYS
_DP_PER_EXPERIMENT = frozenset({"experiment-id", "target-epsilon", "calibration-path"})


class ConfigError(ValueError):
    """A config problem; the message names the key."""


class ComparabilityError(ConfigError):
    """Experiments 2..8 differ somewhere they must not."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: Any, deep: bool = False) -> Dict[Any, Any]:
        seen = set()
        for key_node, _v in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ConfigError(f"duplicate key {key!r} in YAML (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ConfigError(f"{path.name}: expected a flat mapping of string keys")
    return data


def _check_keys(data: Mapping[str, Any], allowed: frozenset, where: str, *, what: str) -> None:
    bad = sorted(set(data) - allowed)
    if bad:
        raise ConfigError(f"{where}: {what} not allowed here: {bad}; allowed: {sorted(allowed)}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def experiment_id_for(n: int, algorithm: str, dp: bool, target_epsilon: Optional[float] = None) -> str:
    """exp{N}-{algorithm}[-eps{value}], e.g. exp4-fedprox-eps50, exp8-fedprox-eps0.5."""
    name = f"exp{n}-{algorithm}"
    if dp:
        if not _is_num(target_epsilon) or target_epsilon <= 0:
            raise ConfigError(f"a DP experiment needs a positive 'target-epsilon', got {target_epsilon!r}")
        name += f"-eps{float(target_epsilon):g}"
    return name


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/\\") or root, *parts])


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def load_experiment_config(
    n: int, *, allow_unset: bool = False, overrides: Optional[Mapping[str, Any]] = None, dry_run: bool = False,
    config_dir: Any = None, env: Optional[Mapping[str, str]] = None,
    manifest_root: Optional[str] = None, checkpoint_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge base.yaml + exp{n}.yaml + overrides into the flat config of one experiment.

    overrides: only OVERRIDE_WHITELIST keys; num-rounds only with dry_run=True,
    which isolates the run (experiment-id gets a '-dryrun' suffix, so its
    checkpoints and calibration never collide with the real run) and sets
    diagnostic. manifest_root / checkpoint_root (or the DR_* environment
    variables) are required unless allow_unset.
    """
    if not _is_int(n) or not 1 <= n <= 8:
        raise ConfigError(f"experiment number must be 1..8, got {n!r}")
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    base = _read_yaml(cdir / "base.yaml")
    exp = _read_yaml(cdir / f"exp{n}.yaml")
    _check_keys(base, BASE_ALLOWED, "base.yaml", what="keys")
    _check_keys(exp, EXP_FILE_ALLOWED, f"exp{n}.yaml", what="overrides")
    cfg: Dict[str, Any] = {**base, **exp}

    ov = dict(overrides or {})
    if manifest_root is not None:
        ov["manifest-root"] = manifest_root
    if checkpoint_root is not None:
        ov["checkpoint-root"] = checkpoint_root
    _check_keys(ov, OVERRIDE_WHITELIST, "overrides", what="override keys")
    if "num-rounds" in ov and not dry_run:
        raise ConfigError("'num-rounds' may only be overridden for a dry run (dry_run=True / --dry-run)")
    cfg.update(ov)

    environ = os.environ if env is None else env
    for key in POD_KEYS:
        if cfg.get(key) is None:
            val = str(environ.get(_POD_ENV[key], "") or "").strip()
            cfg[key] = val or None

    algorithm, dp = cfg.get("algorithm"), cfg.get("dp", False)
    if algorithm not in ("centralized", "fedavg", "fedprox"):
        raise ConfigError(f"exp{n}.yaml: 'algorithm' must be centralized, fedavg or fedprox, got {algorithm!r}")
    if (n == 1) != (algorithm == "centralized"):
        raise ConfigError(f"exp{n}: algorithm {algorithm!r} does not fit this experiment "
                          f"(only experiment 1 is centralized)")
    if not isinstance(dp, bool):
        raise ConfigError(f"'dp' must be true or false, got {dp!r}")
    if n == 1 and dp:
        raise ConfigError("experiment 1 is non-DP")
    if dp and "target-epsilon" not in cfg:
        raise ConfigError(f"exp{n}: dp is true but 'target-epsilon' is missing")
    if not dp and "target-epsilon" in cfg:
        raise ConfigError(f"exp{n}: 'target-epsilon' is set but dp is false")
    if algorithm == "fedprox" and "proximal-mu" not in cfg:
        raise ConfigError(f"exp{n}: fedprox needs 'proximal-mu'")
    if algorithm != "fedprox" and "proximal-mu" in cfg:
        raise ConfigError(f"exp{n}: only fedprox takes 'proximal-mu'")
    if "target-epsilon" in cfg:
        if not _is_num(cfg["target-epsilon"]) or cfg["target-epsilon"] <= 0:
            raise ConfigError(f"'target-epsilon' must be a positive number, got {cfg['target-epsilon']!r}")
        cfg["target-epsilon"] = float(cfg["target-epsilon"])

    derived = experiment_id_for(n, algorithm, dp, cfg.get("target-epsilon"))
    if "experiment-id" in cfg and cfg["experiment-id"] != derived:
        raise ConfigError(f"exp{n}.yaml: experiment-id {cfg['experiment-id']!r} must be {derived!r}")
    cfg["experiment-id"] = derived + ("-dryrun" if dry_run else "")
    if dry_run:
        cfg["diagnostic"] = True

    if dp and "calibration-path" not in cfg:
        root = cfg.get("checkpoint-root")
        cfg["calibration-path"] = _join(root, cfg["experiment-id"], "calibration.json") if root else None

    if not allow_unset:
        need = STEP_ZERO_KEYS if dp else tuple(k for k in STEP_ZERO_KEYS if k != "max-grad-norm")
        unset = [k for k in need if cfg.get(k) is None]
        if unset:
            raise ConfigError(
                f"step-zero values are still null in base.yaml: {unset}; run the diagnostics and fill them in "
                f"(no experiment renders until then)")

    drop = set(CENTRALIZED_ONLY_KEYS) if n != 1 else set(FL_ONLY_KEYS) | {"proximal-mu"}
    if n == 1 or not dp:
        drop |= DP_ONLY_KEYS
    cfg = {k: v for k, v in cfg.items() if k not in drop}
    if n != 1:
        cfg["run-hook"] = RUN_HOOK

    if not allow_unset:
        missing = [k for k, v in cfg.items() if v is None]
        if missing:
            hints = {"manifest-root": "--manifest-root or DR_MANIFEST_ROOT",
                     "checkpoint-root": "--checkpoint-root or DR_CHECKPOINT_ROOT"}
            raise ConfigError("required values are unset: " + ", ".join(
                f"{k} ({hints[k]})" if k in hints else k for k in missing))
        if not _is_int(cfg.get("num-workers")) or cfg["num-workers"] < 1:
            raise ConfigError(f"'num-workers' must be an int >= 1, got {cfg.get('num-workers')!r}")
        if n == 1:
            _validate_centralized(cfg)
        else:
            _validate_server(cfg)

    first = ("experiment-id", "algorithm", "dp")
    return {**{k: cfg[k] for k in first}, **{k: cfg[k] for k in sorted(cfg) if k not in first}}


def _validate_server(cfg: Mapping[str, Any]) -> None:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from federated.server import validate_server_config  # lazy: pulls in torch and flwr
    try:
        validate_server_config(dict(cfg))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _validate_centralized(cfg: Mapping[str, Any]) -> None:
    problems: List[str] = []
    for k in ("lora-rank", "lora-alpha", "physical-batch-size", "accumulation-steps", "effective-batch-size",
              "centralized-max-epochs", "centralized-patience"):
        if not _is_int(cfg.get(k)) or cfg[k] <= 0:
            problems.append(f"{k!r}: expected an int > 0, got {cfg.get(k)!r}")
    if _is_int(cfg.get("physical-batch-size")) and _is_int(cfg.get("accumulation-steps")) \
            and _is_int(cfg.get("effective-batch-size")) \
            and cfg["physical-batch-size"] * cfg["accumulation-steps"] != cfg["effective-batch-size"]:
        problems.append("physical-batch-size * accumulation-steps must equal effective-batch-size")
    if cfg.get("img-size") != 224 or cfg.get("num-classes") != 5:
        problems.append("img-size must be 224 and num-classes 5")
    if not _is_int(cfg.get("run-seed")) or not 0 <= cfg["run-seed"] < 2 ** 32:
        problems.append(f"'run-seed': expected an int in [0, 2**32), got {cfg.get('run-seed')!r}")
    for k in ("lr-lora-peak", "lr-head-peak", "weight-decay", "centralized-warmup-epochs"):
        if not _is_num(cfg.get(k)) or cfg[k] < 0:
            problems.append(f"{k!r}: expected a number >= 0, got {cfg.get(k)!r}")
    if not _is_num(cfg.get("label-smoothing")) or not 0 <= cfg["label-smoothing"] < 1:
        problems.append(f"'label-smoothing': expected a number in [0, 1), got {cfg.get('label-smoothing')!r}")
    wsf = cfg.get("centralized-warmup-start-factor")
    if not _is_num(wsf) or not 0 < wsf <= 1:
        problems.append(f"'centralized-warmup-start-factor': expected a number in (0, 1], got {wsf!r}")
    if problems:
        raise ConfigError("invalid centralized config:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def to_centralized_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for training.train_centralized.run_centralized_training
    (all four clients' manifests pooled). wandb_run_id and on_epoch_end are not
    included: the caller supplies them from utils.experiment_tracker."""
    if cfg.get("algorithm") != "centralized":
        raise ConfigError(f"to_centralized_kwargs needs the Experiment 1 config, got algorithm {cfg.get('algorithm')!r}")
    for k in ("manifest-root", "checkpoint-root", "lora-rank", "lora-alpha"):
        if cfg.get(k) is None:
            raise ConfigError(f"required value is unset: {k}")
    return {
        "client_manifest_paths": {name: Path(_join(str(cfg["manifest-root"]), f"{name}_manifest.csv"))
                                  for name in CLIENTS},
        "lora_rank": int(cfg["lora-rank"]),
        "lora_alpha": int(cfg["lora-alpha"]),
        "checkpoint_dir": _join(str(cfg["checkpoint-root"]), str(cfg["experiment-id"])),
        "img_size": int(cfg["img-size"]),
        "num_classes": int(cfg["num-classes"]),
        "physical_batch_size": int(cfg["physical-batch-size"]),
        "accumulation_steps": int(cfg["accumulation-steps"]),
        "effective_batch_size": int(cfg["effective-batch-size"]),
        "num_workers": int(cfg["num-workers"]),
        "lr_lora": float(cfg["lr-lora-peak"]),
        "lr_head": float(cfg["lr-head-peak"]),
        "weight_decay": float(cfg["weight-decay"]),
        "warmup_epochs": float(cfg["centralized-warmup-epochs"]),
        "warmup_start_factor": float(cfg["centralized-warmup-start-factor"]),
        "max_epochs": int(cfg["centralized-max-epochs"]),
        "patience": int(cfg["centralized-patience"]),
        "label_smoothing": float(cfg["label-smoothing"]),
        "seed": int(cfg["run-seed"]),
    }


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def check_comparability(configs: Optional[Mapping[int, Mapping[str, Any]]] = None, **load_kwargs: Any) -> None:
    """Experiments 2..8 must be identical except experiment-id, algorithm, dp,
    proximal-mu and the DP-only keys; DP experiments must also agree on every
    DP key except target-epsilon and calibration-path, and all FedProx
    experiments on proximal-mu. Raises ComparabilityError listing every
    difference. With configs=None the experiments are loaded (allow_unset
    defaults to True here: nothing depends on pod or step-zero values)."""
    if configs is None:
        load_kwargs.setdefault("allow_unset", True)
        configs = {n: load_experiment_config(n, **load_kwargs) for n in range(2, 9)}
    problems: List[str] = []
    if set(configs) != set(range(2, 9)):
        raise ComparabilityError(f"need configs for experiments 2..8, got {sorted(configs)}")

    for n in range(2, 9):
        c = configs[n]
        alg, dp, eps = EXPECTED[n]
        if c.get("algorithm") != alg or c.get("dp") is not dp:
            problems.append(f"exp{n}: expected algorithm={alg!r} dp={dp}, got {c.get('algorithm')!r} {c.get('dp')!r}")
        if dp and c.get("target-epsilon") != eps:
            problems.append(f"exp{n}: expected target-epsilon {eps}, got {c.get('target-epsilon')!r}")
        if not dp and "target-epsilon" in c:
            problems.append(f"exp{n}: non-DP config carries target-epsilon")
        try:
            want_id = experiment_id_for(n, alg, dp, eps)
        except ConfigError as exc:
            problems.append(str(exc))
            continue
        if c.get("experiment-id") not in (want_id, want_id + "-dryrun"):
            problems.append(f"exp{n}: experiment-id {c.get('experiment-id')!r}, expected {want_id!r}")
        if (alg == "fedprox") != ("proximal-mu" in c):
            problems.append(f"exp{n}: proximal-mu must be present exactly for fedprox")
        root = c.get("checkpoint-root")
        if dp and root and c.get("calibration-path") != _join(str(root), str(c.get("experiment-id")), "calibration.json"):
            problems.append(f"exp{n}: calibration-path {c.get('calibration-path')!r} is not "
                            f"{{checkpoint-root}}/{{experiment-id}}/calibration.json")

    ref = {k: v for k, v in configs[2].items() if k not in _COMPARABLE_DIFF}
    for n in range(3, 9):
        other = {k: v for k, v in configs[n].items() if k not in _COMPARABLE_DIFF}
        for key in sorted(set(ref) | set(other)):
            if ref.get(key, "<absent>") != other.get(key, "<absent>"):
                problems.append(f"exp2 vs exp{n}: {key!r}: {ref.get(key, '<absent>')!r} != {other.get(key, '<absent>')!r}")
    dp_ns = [n for n in range(2, 9) if EXPECTED[n][1]]
    for n in dp_ns[1:]:
        for key in sorted((set(configs[dp_ns[0]]) | set(configs[n])) - _DP_PER_EXPERIMENT):
            a, b = configs[dp_ns[0]].get(key, "<absent>"), configs[n].get(key, "<absent>")
            if a != b:
                problems.append(f"exp{dp_ns[0]} vs exp{n} (both DP): {key!r}: {a!r} != {b!r}")
    mus = {configs[n].get("proximal-mu") for n in range(2, 9) if EXPECTED[n][0] == "fedprox"}
    if len(mus) != 1:
        problems.append(f"FedProx experiments disagree on proximal-mu: {sorted(map(repr, mus))}")
    if problems:
        raise ComparabilityError("experiments 2..8 are not comparable:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# flwr run rendering
# ---------------------------------------------------------------------------

def _toml_value(key: str, v: Any) -> str:
    if v is None:
        raise ConfigError(f"cannot render null for {key!r} into TOML")
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ConfigError(f"cannot render non-finite {key!r}={v!r} into TOML")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # a TOML basic string: quotes, backslashes and non-ASCII escaped
    raise ConfigError(f"{key!r}: unsupported value type {type(v).__name__} for a flwr run-config")


def render_run_config_toml(cfg: Mapping[str, Any]) -> str:
    """Top-level key = value TOML for `flwr run --run-config <file>`; scalars
    only, strings quoted, never a null."""
    lines = [f"# generated by configs/loader.py for {cfg.get('experiment-id')}; do not edit"]
    if str(cfg.get("experiment-id", "")).endswith("-dryrun"):
        lines.append("# DRY RUN: isolated checkpoints, not comparable with the real experiments")
    for key in sorted(cfg):
        if not key.replace("-", "").replace("_", "").isalnum():
            raise ConfigError(f"key {key!r} is not a bare TOML key")
        lines.append(f"{key} = {_toml_value(key, cfg[key])}")
    return "\n".join(lines) + "\n"


def _read_toml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def preflight_pyproject(cfg: Mapping[str, Any], pyproject: Any = None) -> List[str]:
    """What flwr would do with this run-config. Raises ConfigError if a key is
    not declared in [tool.flwr.app.config] (flwr rejects unknown keys); returns
    warnings (the merged config failing validate_server_config, or no TOML
    reader being available)."""
    path = Path(pyproject) if pyproject is not None else REPO_ROOT / "pyproject.toml"
    data = _read_toml(path)
    if data is None:
        return ["tomllib/tomli not available: pyproject pre-flight skipped"]
    declared = data.get("tool", {}).get("flwr", {}).get("app", {}).get("config")
    if not isinstance(declared, dict):
        raise ConfigError(f"{path}: no [tool.flwr.app.config] table")
    undeclared = sorted(set(cfg) - set(declared))
    if undeclared:
        raise ConfigError(f"keys not declared in {path.name} [tool.flwr.app.config] (flwr would reject them): {undeclared}")
    warnings: List[str] = []
    fused = {**declared, **dict(cfg)}  # what the ServerApp will receive
    try:
        _validate_server(fused)
    except ConfigError as exc:
        warnings.append(f"the config flwr will deliver (pyproject defaults + overrides) fails validate_server_config:\n{exc}")
    return warnings


def flwr_run_command(toml_path: Any) -> str:
    """The launch command (from the repo root, after `source /data/env.sh`)."""
    return f"flwr run . --run-config {shlex.quote(str(toml_path))} --stream"


def federation_setup_text(client_cpus: int) -> str:
    """The 4-supernode, 1-GPU-per-client simulation setup (two equivalent
    options; the pod dry run decides which one sticks)."""
    return "\n".join([
        "Federation setup, once per SuperLink (then `pkill -f flower-superlink` after any env change):",
        "  option 1: add to $FLWR_HOME/config.toml",
        "    [superlink.local.options]",
        "    num-supernodes = 4",
        "    [superlink.local.options.backend.client-resources]",
        f"    num-cpus = {client_cpus}",
        "    num-gpus = 1.0",
        "    [superlink.local.options.backend.init-args]",
        "    num-gpus = 1",
        "  option 2:",
        f"    flwr federation simulation-config --num-supernodes 4 --client-resources-num-cpus {client_cpus} "
        "--client-resources-num-gpus 1.0",
        "    flwr federation ls",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {v}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python configs/loader.py", description="Resolve an experiment config and render its flwr run-config.")
    p.add_argument("experiment", type=int, choices=range(1, 9), metavar="N", help="experiment number 1..8")
    p.add_argument("--manifest-root", help="folder with {client}_manifest.csv (or DR_MANIFEST_ROOT)")
    p.add_argument("--checkpoint-root", help="checkpoint folder (or DR_CHECKPOINT_ROOT)")
    p.add_argument("--client-cpus", type=_positive_int, help="Ray CPUs per client (or DR_CLIENT_CPUS); experiments 2-8")
    p.add_argument("--num-workers", type=_positive_int)
    p.add_argument("--run-seed", type=int)
    p.add_argument("--wandb-cm-every", type=int)
    p.add_argument("--diagnostic", action="store_true", default=None, help="tag the wandb run 'diagnostic'")
    p.add_argument("--num-rounds", type=_positive_int, help="dry runs only; requires --dry-run")
    p.add_argument("--dry-run", action="store_true", help="isolated run: experiment-id gets '-dryrun'")
    p.add_argument("--out", help="where to write the run-config TOML (default {checkpoint-root}/_run-configs/)")
    p.add_argument("--show", action="store_true", help="also print the rendered TOML")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    flag_map = {"manifest-root": args.manifest_root, "checkpoint-root": args.checkpoint_root,
                "num-workers": args.num_workers, "run-seed": args.run_seed, "num-rounds": args.num_rounds,
                "wandb-cm-every": args.wandb_cm_every, "diagnostic": args.diagnostic}
    overrides = {k: v for k, v in flag_map.items() if v is not None}
    try:
        cfg = load_experiment_config(args.experiment, overrides=overrides, dry_run=args.dry_run)
        if args.experiment == 1:
            kwargs = to_centralized_kwargs(cfg)
            print(json.dumps(cfg, indent=2))
            print("\nrun_centralized_training kwargs (wandb_run_id and on_epoch_end come from utils.experiment_tracker):")
            print(json.dumps({k: (str(v) if isinstance(v, Path) else {a: str(b) for a, b in v.items()}
                                  if isinstance(v, dict) else v) for k, v in kwargs.items()}, indent=2))
            return 0
        cpus = args.client_cpus
        if cpus is None:
            raw = os.environ.get(CPU_ENV, "").strip()
            cpus = _positive_int(raw) if raw else None
        if cpus is None:
            raise ConfigError(f"the Ray client CPU count is required for experiments 2-8: --client-cpus N or {CPU_ENV}")
        warnings = preflight_pyproject(cfg)
        out = Path(args.out) if args.out else Path(_join(cfg["checkpoint-root"], "_run-configs", cfg["experiment-id"] + ".toml"))
        text = render_run_config_toml(cfg)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot write {out}: {exc}; pass --out") from exc
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"DRY RUN: {cfg['experiment-id']}, num-rounds {cfg['num-rounds']}; checkpoints are isolated, results not comparable\n")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"rendered {cfg['experiment-id']} -> {out}\n")
    if args.show:
        print(text)
    print(federation_setup_text(cpus))
    print("\nRun from the repo root after `source /data/env.sh`:")
    print(flwr_run_command(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
