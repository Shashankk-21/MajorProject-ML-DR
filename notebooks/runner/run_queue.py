"""
notebooks/runner/run_queue.py: run the eight experiments in order, with automatic resume.

    python notebooks/runner/run_queue.py --manifest-root P --checkpoint-root P --client-cpus N
        [--experiments 1 2 3 ...] [--dry-run] [--max-attempts 5] [--sleep-between 60]
        [--restart-superlink] [--continue-on-failure] [--skip-lock-check] [--status] [--config-dir D]

DR_MANIFEST_ROOT, DR_CHECKPOINT_ROOT and DR_CLIENT_CPUS work in place of the flags
(--client-cpus is only needed when an FL experiment, 2-8, is in the queue).

Experiment 1 runs through notebooks/runner/run_experiment1.py. Experiments 2-8 render their
run-config with configs.loader into {checkpoint-root}/_run-configs/{id}.toml and run, from the
repo root, `flwr run . --run-config <toml> --stream`. Resume is automatic: every attempt of an
experiment is the SAME command, which resumes from latest.pt.

Status comes from the filesystem, never from memory: FL done = done.json, in progress =
latest.pt; Experiment 1 in progress = latest_centralized.pt (whether it is finished is decided
by running its driver, which returns at once when it is).

This script NEVER runs the evaluation: the test set is touched once, by hand, with
`python -m evaluation.evaluate`.

Exit codes: 0 everything done (or skipped as done); 2 preflight refusal; 3 something failed
(takes precedence over 4); 4 something was blocked (config cannot be rendered); 130 interrupted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import loader  # noqa: E402  (stdlib + yaml only)

EXIT_OK, EXIT_REFUSED, EXIT_FAILED, EXIT_BLOCKED, EXIT_INTERRUPTED = 0, 2, 3, 4, 130
DRY_RUN_ROUNDS = 2
DRY_RUN_EPOCHS = 2
GRACE_SECONDS = 10.0
LOCK_SCRIPT = REPO_ROOT / "tools" / "verify_locks.py"
EXP1_DRIVER = HERE / "run_experiment1.py"
_ROUND_RE = re.compile(r"^round_(\d+)\.pt$")

_STOP = threading.Event()  # set by SIGINT / SIGTERM
_CHILD_LOCK = threading.Lock()
_CHILD: Optional["subprocess.Popen[str]"] = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def experiment_id(n: int, dry_run: bool = False) -> str:
    """exp{N}-{algorithm}[-eps{value}] (configs.loader naming), plus '-dryrun' for dry runs."""
    if n not in loader.EXPECTED:
        raise ValueError(f"experiment number must be 1..8, got {n!r}")
    algorithm, dp, eps = loader.EXPECTED[n]
    return loader.experiment_id_for(n, algorithm, dp, eps) + ("-dryrun" if dry_run else "")


def _highest_round(folder: Path) -> Optional[int]:
    rounds = [int(m.group(1)) for p in folder.glob("round_*.pt") for m in [_ROUND_RE.match(p.name)] if m]
    return max(rounds) if rounds else None


def experiment_state(checkpoint_root: Any, n: int, num_rounds: Optional[int] = None, *, dry_run: bool = False,
                     blocked_reason: Optional[str] = None) -> Dict[str, str]:
    """{"state": not-started | in-progress | done | blocked, "detail": str}, from the filesystem.

    FL: done = done.json; in progress = latest.pt (detail: the highest round_NNN.pt, the last
    completed round); done.json without latest.pt is blocked (the server refuses such a folder).
    Experiment 1: in progress = latest_centralized.pt; it is never reported done here (the driver
    decides). blocked_reason (the loader's refusal) makes the state blocked."""
    if blocked_reason is not None:
        return {"state": "blocked", "detail": blocked_reason}
    folder = Path(checkpoint_root) / experiment_id(n, dry_run)
    if n == 1:
        if (folder / "latest_centralized.pt").is_file():
            return {"state": "in-progress", "detail": "latest_centralized.pt present (finished or not is decided by running the driver)"}
        return {"state": "not-started", "detail": "no latest_centralized.pt"}
    done, latest = (folder / "done.json").is_file(), (folder / "latest.pt").is_file()
    rnd = _highest_round(folder) if folder.is_dir() else None
    total = f" of {num_rounds}" if num_rounds else ""
    if done and not latest:
        return {"state": "blocked", "detail": "done.json exists but latest.pt is missing (inconsistent folder; inspect it)"}
    if done:
        return {"state": "done", "detail": f"done.json present (last round {rnd}{total})" if rnd else "done.json present"}
    if latest:
        return {"state": "in-progress", "detail": f"latest.pt present, last completed round {rnd}{total}" if rnd else "latest.pt present"}
    if rnd:
        return {"state": "not-started", "detail": f"round files up to {rnd} but no latest.pt (the first round never completed)"}
    return {"state": "not-started", "detail": "no checkpoints"}


def plan_queue(states: Mapping[int, Mapping[str, Any]], experiments: Iterable[int],
               blocked: Optional[Iterable[int]] = None) -> List[int]:
    """The experiments to run, in the given order (duplicates dropped): done and blocked ones are
    skipped. `blocked` may be a mapping or any iterable of experiment numbers; an experiment with no
    entry in `states` counts as not started."""
    blocked_set = set(blocked or ())
    plan: List[int] = []
    for n in experiments:
        if n in plan or n in blocked_set:
            continue
        state = (states.get(n) or {}).get("state", "not-started")
        if state in ("done", "blocked"):
            continue
        plan.append(n)
    return plan


def build_flwr_command(toml_path: Any) -> List[str]:
    """argv of `flwr run . --run-config <toml> --stream` (the loader's own command text)."""
    return shlex.split(loader.flwr_run_command(toml_path))


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one place that spawns processes
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stop_child(child: "subprocess.Popen[str]", signum: int) -> None:
    if child.poll() is not None:
        return
    try:
        child.send_signal(signum)
    except Exception:
        try:
            child.terminate()
        except Exception:
            pass
    timer = threading.Timer(GRACE_SECONDS, lambda: child.poll() is None and child.kill())
    timer.daemon = True
    timer.start()


def _on_signal(signum: int, _frame: Any) -> None:
    _STOP.set()
    with _CHILD_LOCK:
        child = _CHILD
    if child is not None:
        _stop_child(child, signum)


def run_cmd(argv: Sequence[str], log_path: Any, env: Optional[Mapping[str, str]] = None) -> int:
    """Run argv from the repo root, tee stdout and stderr (merged) to log_path and the console,
    return the exit code (127 if the program cannot be started). The ONLY function here that
    starts a process, so tests can replace it."""
    global _CHILD
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"# {_utc_stamp()} $ {' '.join(shlex.quote(str(a)) for a in argv)}\n")
        log.flush()
        try:
            proc = subprocess.Popen([str(a) for a in argv], cwd=str(REPO_ROOT), env=None if env is None else dict(env),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    encoding="utf-8", errors="replace")
        except OSError as exc:
            msg = f"could not start {argv[0]!r}: {exc}\n"
            sys.stdout.write(msg)
            log.write(msg)
            return 127
        with _CHILD_LOCK:
            _CHILD = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return proc.wait()
        finally:
            with _CHILD_LOCK:
                _CHILD = None


def _sleep(seconds: float) -> None:
    """Sleep in short steps so an interrupt is noticed."""
    end = time.monotonic() + seconds
    while not _STOP.is_set() and time.monotonic() < end:
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))


def cuda_available() -> bool:
    try:
        import torch  # lazy
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Arguments and preflight
# ---------------------------------------------------------------------------

def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {v}")
    return v


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python notebooks/runner/run_queue.py", description="Run the experiment queue.")
    p.add_argument("--manifest-root", default=os.environ.get("DR_MANIFEST_ROOT"))
    p.add_argument("--checkpoint-root", default=os.environ.get("DR_CHECKPOINT_ROOT"))
    p.add_argument("--client-cpus", type=_positive_int, default=None, help="Ray CPUs per client (or DR_CLIENT_CPUS); FL runs")
    p.add_argument("--experiments", type=int, nargs="+", choices=range(1, 9), metavar="N", default=list(range(1, 9)))
    p.add_argument("--dry-run", action="store_true", help="isolated -dryrun ids: FL 2 rounds, Experiment 1 2 epochs")
    p.add_argument("--max-attempts", type=_positive_int, default=5)
    p.add_argument("--sleep-between", type=float, default=60.0, help="seconds between attempts")
    p.add_argument("--restart-superlink", action="store_true", help="pkill -f flower-superlink once before the first FL attempt")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("--skip-lock-check", action="store_true")
    p.add_argument("--status", action="store_true", help="print the state of all eight experiments and exit")
    p.add_argument("--config-dir", default=None, help="alternative configs folder (testing)")
    args = p.parse_args(argv)
    if args.sleep_between < 0:
        p.error("--sleep-between must be >= 0")
    if args.client_cpus is None:
        raw = os.environ.get("DR_CLIENT_CPUS", "").strip()
        if raw:
            try:
                args.client_cpus = _positive_int(raw)
            except argparse.ArgumentTypeError as exc:
                p.error(f"DR_CLIENT_CPUS: {exc}")
    args.experiments = list(dict.fromkeys(args.experiments))
    return args


def _logs_dir(args: argparse.Namespace) -> Path:
    return Path(args.checkpoint_root) / "_logs"


def preflight(args: argparse.Namespace, env: Mapping[str, str]) -> List[str]:
    """Reasons to refuse (empty list = go). Warnings are printed here."""
    problems: List[str] = []
    fl = [n for n in args.experiments if n >= 2]
    if not args.manifest_root:
        problems.append("missing --manifest-root (or DR_MANIFEST_ROOT)")
    if not args.checkpoint_root:
        problems.append("missing --checkpoint-root (or DR_CHECKPOINT_ROOT)")
    if fl and not args.client_cpus:
        problems.append("missing --client-cpus (or DR_CLIENT_CPUS); FL experiments need the Ray client CPU count")
    if problems:
        return problems
    if not args.dry_run and not cuda_available():
        problems.append("no CUDA GPU is available to this process (a real run needs one; dry runs skip this check)")
    if not args.skip_lock_check:
        if not LOCK_SCRIPT.is_file():
            problems.append(f"{LOCK_SCRIPT} does not exist, so the locked files cannot be verified; "
                            f"create it or pass --skip-lock-check")
        else:
            log = _logs_dir(args) / f"verify_locks-{_utc_stamp()}.log"
            code = run_cmd([sys.executable, str(LOCK_SCRIPT)], log, env)
            if code != 0:
                problems.append(f"tools/verify_locks.py exited {code}: locked files differ from their recorded hashes (log: {log})")
    if fl:
        home = Path(env.get("FLWR_HOME") or (Path.home() / ".flwr"))
        cfg_path = home / "config.toml"
        try:
            ok = re.search(r"^\s*num-supernodes\s*=\s*4\b", cfg_path.read_text(encoding="utf-8"), re.MULTILINE) is not None
        except OSError:
            ok = False
        if not ok:
            print("!" * 78, file=sys.stderr)
            print(f"WARNING: {cfg_path} does not contain 'num-supernodes = 4'. flwr's default simulation has 2 supernodes "
                  f"and 0 GPUs, which would hang the 4-client server.", file=sys.stderr)
            print(loader.federation_setup_text(args.client_cpus), file=sys.stderr)
            print("!" * 78, file=sys.stderr)
    return problems


# ---------------------------------------------------------------------------
# Config rendering, states, attempts
# ---------------------------------------------------------------------------

def load_config(n: int, args: argparse.Namespace, *, allow_unset: bool = False) -> Dict[str, Any]:
    overrides = {"num-rounds": DRY_RUN_ROUNDS} if (args.dry_run and n >= 2) else None
    return loader.load_experiment_config(
        n, allow_unset=allow_unset, overrides=overrides, dry_run=args.dry_run, config_dir=args.config_dir, env={},
        manifest_root=args.manifest_root, checkpoint_root=args.checkpoint_root)


def find_blocked(args: argparse.Namespace, experiments: Iterable[int]) -> Dict[int, str]:
    """{n: reason} for experiments whose config cannot be rendered (null step-zero values, ...)."""
    blocked: Dict[int, str] = {}
    for n in experiments:
        try:
            cfg = load_config(n, args)
            if n >= 2:
                loader.preflight_pyproject(cfg)  # raises ConfigError for keys flwr would reject
        except loader.ConfigError as exc:
            blocked[n] = str(exc)
    return blocked


def render_toml(cfg: Mapping[str, Any], checkpoint_root: Any) -> Path:
    """Write the run-config to {checkpoint-root}/_run-configs/{id}.toml (the loader's own location)."""
    path = Path(checkpoint_root) / "_run-configs" / f"{cfg['experiment-id']}.toml"
    text = loader.render_run_config_toml(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")
    return path


def exp1_command(args: argparse.Namespace) -> List[str]:
    argv = [sys.executable, str(EXP1_DRIVER), "--manifest-root", str(args.manifest_root),
            "--checkpoint-root", str(args.checkpoint_root)]
    if args.config_dir:
        argv += ["--config-dir", str(args.config_dir)]
    if args.dry_run:
        argv += ["--dry-run", "--max-epochs", str(DRY_RUN_EPOCHS)]
    return argv


def num_rounds_of(n: int, args: argparse.Namespace) -> Optional[int]:
    if n == 1:
        return None
    try:
        return int(load_config(n, args, allow_unset=True)["num-rounds"])
    except Exception:
        return None


def attempt(n: int, k: int, args: argparse.Namespace, env: Mapping[str, str], session: Dict[str, bool]) -> bool:
    """One attempt of experiment n (the same command every time). True when it finished."""
    eid = experiment_id(n, args.dry_run)
    log = _logs_dir(args) / f"{eid}-attempt{k}-{_utc_stamp()}.log"
    if n == 1:
        return run_cmd(exp1_command(args), log, env) == 0
    cfg = load_config(n, args)
    toml_path = render_toml(cfg, args.checkpoint_root)
    if args.restart_superlink and not session.get("superlink_restarted"):
        session["superlink_restarted"] = True
        code = run_cmd(["pkill", "-f", "flower-superlink"], _logs_dir(args) / f"pkill-superlink-{_utc_stamp()}.log", env)
        if code not in (0, 1):  # pkill: 1 = no process matched
            print(f"WARNING: pkill -f flower-superlink exited {code}", file=sys.stderr)
    code = run_cmd(build_flwr_command(toml_path), log, env)
    state = experiment_state(args.checkpoint_root, n, cfg["num-rounds"], dry_run=args.dry_run)
    if code == 0 and state["state"] == "done":
        return True
    reason = f"flwr exited {code}" if code != 0 else "flwr exited 0 but done.json is missing"
    print(f"{eid}: attempt {k} did not finish ({reason}; {state['detail']})", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Status and the queue
# ---------------------------------------------------------------------------

def collect_states(args: argparse.Namespace, blocked: Mapping[int, str]) -> Dict[int, Dict[str, str]]:
    return {n: experiment_state(args.checkpoint_root, n, num_rounds_of(n, args), dry_run=args.dry_run,
                                blocked_reason=blocked.get(n)) for n in range(1, 9)}


def print_status(states: Mapping[int, Mapping[str, str]], dry_run: bool) -> None:
    rows = [[n, experiment_id(n, dry_run), states[n]["state"], states[n]["detail"]] for n in sorted(states)]
    print(format_table(["#", "experiment", "state", "detail"], rows))


def status_main(args: argparse.Namespace) -> int:
    if not args.checkpoint_root:
        print("error: missing --checkpoint-root (or DR_CHECKPOINT_ROOT)", file=sys.stderr)
        return EXIT_REFUSED
    blocked: Dict[int, str] = {}
    if args.manifest_root:
        try:
            blocked = find_blocked(args, range(1, 9))
        except ImportError as exc:
            print(f"note: config validation skipped ({exc})", file=sys.stderr)
    else:
        print("note: no --manifest-root, so blocked configs cannot be detected", file=sys.stderr)
    print_status(collect_states(args, blocked), args.dry_run)
    return EXIT_OK


def run_queue(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    try:
        blocked = find_blocked(args, args.experiments)
    except ImportError as exc:
        print(f"error: the project's Python environment cannot validate the configs ({exc})", file=sys.stderr)
        return EXIT_REFUSED
    for n, reason in blocked.items():
        print(f"BLOCKED {experiment_id(n, args.dry_run)}: {reason}", file=sys.stderr)
    states = collect_states(args, blocked)
    plan = plan_queue(states, args.experiments, blocked)
    results: Dict[int, Dict[str, Any]] = {}
    for n in args.experiments:
        if n in blocked:
            results[n] = {"result": "blocked", "attempts": 0}
        elif n not in plan:
            results[n] = {"result": "skipped (done)", "attempts": 0}
        else:
            results[n] = {"result": "not run", "attempts": 0}
    session: Dict[str, bool] = {}
    failed = False
    interrupted = False
    session_start = time.monotonic()
    for n in plan:
        if interrupted or (failed and not args.continue_on_failure):
            break
        eid = experiment_id(n, args.dry_run)
        print(f"\n=== {eid} ({states[n]['state']}: {states[n]['detail']}) ===", flush=True)
        finished = False
        for k in range(1, args.max_attempts + 1):
            if _STOP.is_set():
                interrupted = True
                break
            results[n]["attempts"] = k
            print(f"--- {eid}: attempt {k} of {args.max_attempts} ---", flush=True)
            try:
                finished = attempt(n, k, args, env, session)
            except loader.ConfigError as exc:  # the config became unrenderable after planning
                print(f"{eid}: attempt {k} cannot start: {exc}", file=sys.stderr)
                finished = False
            if _STOP.is_set():
                interrupted = True  # never counted as a failure
                break
            if finished:
                break
            if k < args.max_attempts:
                print(f"{eid}: retrying in {args.sleep_between:g}s (same command, resumes from latest.pt)", flush=True)
                _sleep(args.sleep_between)
        if interrupted:
            results[n]["result"] = "interrupted"
            break
        if finished:
            results[n]["result"] = "done"
        else:
            results[n]["result"] = "FAILED"
            failed = True
            print(f"{eid}: FAILED after {args.max_attempts} attempts", file=sys.stderr)

    rows = [[n, experiment_id(n, args.dry_run), results[n]["result"], results[n]["attempts"]] for n in args.experiments]
    print("\nSummary:")
    print(format_table(["#", "experiment", "result", "attempts"], rows))
    print(f"\nElapsed {int(time.monotonic() - session_start)}s. Logs: {_logs_dir(args)}")
    print("REMINDER: this runner never runs the evaluation. When all experiments are done, run "
          "`python -m evaluation.evaluate` by hand, once per experiment (the test set is touched once).")
    if interrupted:
        print("Interrupted: nothing was marked failed; run the same command again to resume.", file=sys.stderr)
        return EXIT_INTERRUPTED
    if failed:
        return EXIT_FAILED
    if blocked:
        print(f"{len(blocked)} experiment(s) were blocked (see above).", file=sys.stderr)
        return EXIT_BLOCKED
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.status:
        return status_main(args)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    problems = preflight(args, env)
    if problems:
        print("Refusing to start:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        return EXIT_REFUSED
    _STOP.clear()
    previous = {}
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            try:
                previous[name] = signal.signal(getattr(signal, name), _on_signal)
            except ValueError:  # not the main thread
                pass
    try:
        return run_queue(args, env)
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


if __name__ == "__main__":
    sys.exit(main())
