"""
notebooks/runner/run_experiment1.py: driver for Experiment 1 (centralized baseline).

    python notebooks/runner/run_experiment1.py --manifest-root P --checkpoint-root P [--config-dir D]
    python notebooks/runner/run_experiment1.py ... --dry-run [--max-epochs N]

Config via configs.loader.load_experiment_config(1) (refuses null step-zero values) and
loader.to_centralized_kwargs; W&B via utils.experiment_tracker.start_centralized and
make_epoch_callback; training via train_centralized.run_centralized_training, which
resumes from latest_centralized.pt and returns at once when the run is already finished.
The test set is never touched here.

--dry-run uses the loader's dry-run naming (exp1-centralized-dryrun, diagnostic tag) so the
real checkpoints are never touched; --max-epochs (default 2) is accepted only with it.

Exit codes: 0 finished (also when it was already finished), 1 failure, 2 config or usage
error (including a null step-zero value), 130 interrupted.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import loader  # noqa: E402  (stdlib + yaml only)

DEFAULT_DRY_RUN_EPOCHS = 2


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run (or resume) Experiment 1, the centralized baseline.")
    p.add_argument("--manifest-root", default=os.environ.get("DR_MANIFEST_ROOT"), help="folder with {client}_manifest.csv")
    p.add_argument("--checkpoint-root", default=os.environ.get("DR_CHECKPOINT_ROOT"))
    p.add_argument("--config-dir", default=None, help="alternative configs folder (testing)")
    p.add_argument("--dry-run", action="store_true", help="isolated run named exp1-centralized-dryrun")
    p.add_argument("--max-epochs", type=int, default=None, help=f"dry runs only (default {DEFAULT_DRY_RUN_EPOCHS})")
    args = p.parse_args(argv)
    for name in ("manifest_root", "checkpoint_root"):
        if not getattr(args, name):
            p.error(f"--{name.replace('_', '-')} is required (or DR_{name.upper()})")
    if args.max_epochs is not None:
        if not args.dry_run:
            p.error("--max-epochs is accepted only with --dry-run")
        if args.max_epochs < 1:
            p.error("--max-epochs must be >= 1")
    if args.dry_run and args.max_epochs is None:
        args.max_epochs = DEFAULT_DRY_RUN_EPOCHS
    return args


def build_run(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(cfg, run_centralized_training kwargs) for this invocation."""
    cfg = loader.load_experiment_config(
        1, manifest_root=args.manifest_root, checkpoint_root=args.checkpoint_root, env={},
        config_dir=args.config_dir, dry_run=args.dry_run)
    kwargs = loader.to_centralized_kwargs(cfg)
    if args.dry_run:
        kwargs["max_epochs"] = args.max_epochs
    return cfg, kwargs


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Train (or resume, or return at once if finished); tracker.finish on every path."""
    cfg, kwargs = build_run(args)
    from training.train_centralized import run_centralized_training  # lazy: pulls in torch
    from utils.experiment_tracker import make_epoch_callback, start_centralized

    tracker, run_id = start_centralized(cfg, kwargs["checkpoint_dir"])
    callback = make_epoch_callback(tracker)
    try:
        result = run_centralized_training(**kwargs, wandb_run_id=run_id, on_epoch_end=callback)
    except BaseException as exc:
        tracker.finish(None, exc)
        raise
    tracker.finish(result, None)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except loader.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        print("Experiment 1 FAILED (see the traceback above).", file=sys.stderr)
        return 1
    print(f"Experiment 1 finished: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
