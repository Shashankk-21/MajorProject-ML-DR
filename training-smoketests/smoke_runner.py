import contextlib
import io
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "notebooks" / "runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_experiment1 as r1
import run_queue as rq
from configs import loader as ld

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

TESTS = []
STEP = {"lora-rank": "2", "lora-alpha": "4", "max-grad-norm": "1.0", "num-workers": "1"}
for _k in ("DR_MANIFEST_ROOT", "DR_CHECKPOINT_ROOT", "DR_CLIENT_CPUS", "FLWR_HOME"):
    os.environ.pop(_k, None)


class Skip(Exception):
    pass


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        if contains is not None and contains not in str(e):
            raise RuntimeError(f"error did not mention {contains!r}: {str(e)[:300]}")
        return
    raise RuntimeError(f"expected {exc}, nothing raised")


_N = [0]
_TMP = []


def tmp():
    if not _TMP:
        _TMP.append(Path(tempfile.mkdtemp(prefix="runner_smoke_")))
    return _TMP[0]


def fresh(prefix):
    _N[0] += 1
    d = tmp() / f"{prefix}{_N[0]}"
    d.mkdir()
    return d


def config_dir(**subs):
    d = fresh("cfg")
    for f in (ROOT / "configs").glob("*.yaml"):
        shutil.copy(f, d)
    text = (d / "base.yaml").read_text(encoding="utf-8")
    for k, v in {**STEP, **subs}.items():
        text, n = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", text, flags=re.M)
        assert n == 1, k
    (d / "base.yaml").write_text(text, encoding="utf-8")
    return d


def capture(fn, *a, **k):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **k)
    return rc, out.getvalue(), err.getvalue()


class Harness:
    """Replaces run_cmd, the sleeper, cuda detection and the lock script for one scenario."""

    def __init__(self, script):
        self.script, self.calls, self.sleeps, self.logs = script, [], [], []
        self.saved = {}

    def __call__(self, argv, log_path, env):
        self.calls.append(list(argv))
        self.logs.append(str(log_path))
        return self.script(self, list(argv))

    def __enter__(self):
        for name in ("run_cmd", "_sleep", "cuda_available", "LOCK_SCRIPT"):
            self.saved[name] = getattr(rq, name)
        rq.run_cmd = self
        rq._sleep = lambda s: self.sleeps.append(s)
        rq.cuda_available = lambda: True
        rq._STOP.clear()
        return self

    def __exit__(self, *exc):
        for name, v in self.saved.items():
            setattr(rq, name, v)
        rq._STOP.clear()

    def flwr_calls(self):
        return [c for c in self.calls if c and c[0] == "flwr"]

    def ids(self):
        return [Path(c[4]).stem for c in self.flwr_calls()]


def complete(ck, argv, rounds=2, done=True):
    d = Path(ck) / Path(argv[4]).stem
    d.mkdir(parents=True, exist_ok=True)
    for r in range(1, rounds + 1):
        (d / f"round_{r:03d}.pt").write_bytes(b"x")
    (d / "latest.pt").write_bytes(b"x")
    (d / "best.pt").write_bytes(b"x")
    if done:
        (d / "done.json").write_text("{}", encoding="utf-8")


def argv_for(experiments, ck, cfg, *extra):
    return ["--manifest-root", str(tmp() / "manifests"), "--checkpoint-root", str(ck), "--client-cpus", "4", "--experiments",
            *map(str, experiments), "--config-dir", str(cfg), "--skip-lock-check", *extra]


# ------------------------------------------------------------------ pure helpers
@test("Q1 experiment ids and dry-run ids follow the loader naming; the flwr command is exactly flwr run . --run-config <toml> --stream")
def _():
    assert [rq.experiment_id(n) for n in range(1, 9)] == [
        "exp1-centralized", "exp2-fedavg", "exp3-fedprox", "exp4-fedprox-eps50", "exp5-fedprox-eps10", "exp6-fedprox-eps5",
        "exp7-fedprox-eps1", "exp8-fedprox-eps0.5"]
    assert rq.experiment_id(2, True) == "exp2-fedavg-dryrun" and rq.experiment_id(8, True) == "exp8-fedprox-eps0.5-dryrun"
    assert rq.build_flwr_command(Path("/c/_run-configs/x.toml")) == ["flwr", "run", ".", "--run-config", str(Path("/c/_run-configs/x.toml")), "--stream"]


@test("Q2 experiment_state is derived from files: not-started, in-progress (with the last completed round), done, blocked; Experiment 1 is never 'done' from files")
def _():
    ck = fresh("ck")
    s = rq.experiment_state(ck, 2, 10)
    assert s["state"] == "not-started"
    d = ck / "exp2-fedavg"
    d.mkdir()
    for r in (1, 2, 3):
        (d / f"round_{r:03d}.pt").write_bytes(b"x")
    (d / "latest.pt").write_bytes(b"x")
    s = rq.experiment_state(ck, 2, 10)
    assert s["state"] == "in-progress" and "3" in s["detail"] and "10" in s["detail"], s
    (d / "done.json").write_text("{}")
    assert rq.experiment_state(ck, 2, 10)["state"] == "done"
    (d / "latest.pt").unlink()
    assert rq.experiment_state(ck, 2, 10)["state"] == "blocked"  # the server refuses done.json without latest.pt
    assert rq.experiment_state(ck, 3, 10, blocked_reason="max-grad-norm is null")["state"] == "blocked"
    assert "max-grad-norm" in rq.experiment_state(ck, 3, 10, blocked_reason="max-grad-norm is null")["detail"]
    e1 = ck / "exp1-centralized"
    e1.mkdir()
    assert rq.experiment_state(ck, 1)["state"] == "not-started"
    (e1 / "latest_centralized.pt").write_bytes(b"x")
    (e1 / "best_centralized.pt").write_bytes(b"x")
    assert rq.experiment_state(ck, 1)["state"] == "in-progress"
    dry = ck / "exp2-fedavg-dryrun"
    dry.mkdir()
    (dry / "latest.pt").write_bytes(b"x")
    (dry / "done.json").write_text("{}")
    assert rq.experiment_state(ck, 2, 2, dry_run=True)["state"] == "done" and rq.experiment_state(ck, 3, 2, dry_run=True)["state"] == "not-started"


@test("Q3 plan_queue keeps the given order, drops duplicates, skips done and blocked experiments")
def _():
    st = {n: {"state": "not-started", "detail": ""} for n in range(1, 9)}
    st[2]["state"] = "done"
    st[4]["state"] = "blocked"
    st[3]["state"] = "in-progress"
    assert rq.plan_queue(st, [3, 1, 2, 3, 4, 5], {4: "why"}) == [3, 1, 5]
    assert rq.plan_queue(st, [], {}) == []
    assert rq.plan_queue(st, [8, 7], {}) == [8, 7]


# ------------------------------------------------------------------ the queue, with a fake run_cmd
@test("Q4 retries the SAME command: a failed first attempt, then success; sleeps once between attempts; logs and run-config TOML are written per attempt")
def _():
    ck, cfg = fresh("ck"), config_dir()
    state = {"n": {}}

    def script(h, argv):
        eid = Path(argv[4]).stem
        state["n"][eid] = state["n"].get(eid, 0) + 1
        assert Path(argv[4]).is_file()  # the TOML exists before flwr is called
        if eid.startswith("exp2") and state["n"][eid] == 1:
            return 1
        complete(ck, argv)
        return 0

    with Harness(script) as h:
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--sleep-between", "7"))
    assert rc == 0, (out, err)
    assert h.ids() == ["exp2-fedavg-dryrun", "exp2-fedavg-dryrun", "exp3-fedprox-dryrun"] and h.sleeps == [7]
    assert any("-attempt1-" in l for l in h.logs) and any("-attempt2-" in l for l in h.logs) and all(Path(l).parent.name == "_logs" for l in h.logs)
    toml = ck / "_run-configs" / "exp2-fedavg-dryrun.toml"
    assert toml.is_file()
    if tomllib is not None:
        cfgd = tomllib.loads(toml.read_text(encoding="utf-8"))
        assert cfgd["num-rounds"] == 2 and cfgd["experiment-id"] == "exp2-fedavg-dryrun" and cfgd["diagnostic"] is True


@test("Q5 failure rules: flwr exit 0 without done.json is a failure; exhausted attempts stop the queue (exit 3) unless --continue-on-failure; an already finished experiment is skipped")
def _():
    ck, cfg = fresh("ck"), config_dir()
    with Harness(lambda h, a: 0) as h:  # flwr "succeeds" but never writes done.json
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--max-attempts", "3"))
    assert rc == 3 and h.ids() == ["exp2-fedavg-dryrun"] * 3 and len(h.sleeps) == 2 and "FAILED" in out
    ck = fresh("ck")

    def script(h, argv):
        if "exp2" in argv[4]:
            return 1
        complete(ck, argv)
        return 0

    with Harness(script) as h:
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--max-attempts", "2", "--continue-on-failure"))
    assert rc == 3 and h.ids() == ["exp2-fedavg-dryrun"] * 2 + ["exp3-fedprox-dryrun"]
    ck = fresh("ck")
    complete(ck, ["", "", "", "", "exp2-fedavg-dryrun.toml"], rounds=2, done=True)
    with Harness(lambda h, a: (complete(ck, a), 0)[1]) as h:
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run"))
    assert rc == 0 and h.ids() == ["exp3-fedprox-dryrun"]
    ck = fresh("ck")

    def spurious(h, argv):  # the CLI exits non-zero although the run finished: next attempt is a no-op that succeeds
        complete(ck, argv)
        return 0 if len(h.flwr_calls()) > 1 else 1

    with Harness(spurious) as h:
        rc, out, err = capture(rq.main, argv_for([2], ck, cfg, "--dry-run", "--max-attempts", "3"))
    assert rc == 0 and len(h.flwr_calls()) == 2


@test("Q6 blocked experiments (null step-zero values) are skipped with the loader's reason: exit 4; a failure takes precedence (exit 3)")
def _():
    ck = fresh("ck")
    cfg = config_dir(**{"max-grad-norm": "null"})  # non-DP experiments render, DP ones are blocked

    def ok(h, argv):
        complete(ck, argv)
        return 0

    with Harness(ok) as h:
        rc, out, err = capture(rq.main, argv_for([2, 4, 5], ck, cfg, "--dry-run"))
    assert rc == 4 and h.ids() == ["exp2-fedavg-dryrun"] and "max-grad-norm" in out + err
    with Harness(lambda h, a: 1) as h:
        rc, out, err = capture(rq.main, argv_for([2, 4], fresh("ck"), cfg, "--dry-run", "--max-attempts", "1"))
    assert rc == 3
    cfg2 = config_dir(**{"lora-rank": "null"})
    with Harness(ok) as h:
        rc, out, err = capture(rq.main, argv_for([1, 2], fresh("ck"), cfg2, "--dry-run"))
    assert rc == 4 and h.calls == [] and "lora-rank" in out + err


@test("Q7 an interrupt is not a failure: no retry, no sleep, exit 130, nothing marked FAILED")
def _():
    ck, cfg = fresh("ck"), config_dir()

    def script(h, argv):
        rq._STOP.set()
        return 130

    with Harness(script) as h:
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--max-attempts", "5"))
    assert rc == 130 and len(h.flwr_calls()) == 1 and h.sleeps == [] and "FAILED" not in out


@test("Q8 --restart-superlink: pkill -f flower-superlink exactly once, before the first FL attempt of the session, never for Experiment 1 alone; exit 1 from pkill is fine")
def _():
    ck, cfg = fresh("ck"), config_dir()

    def script(h, argv):
        if argv[0] == "pkill":
            return 1
        if argv[0] == "flwr":
            complete(ck, argv)
        return 0

    with Harness(script) as h:
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--restart-superlink"))
    pk = [i for i, c in enumerate(h.calls) if c[0] == "pkill"]
    assert rc == 0 and len(pk) == 1 and h.calls[pk[0]] == ["pkill", "-f", "flower-superlink"] and h.calls[pk[0] + 1][0] == "flwr"
    with Harness(script) as h:
        rc, out, err = capture(rq.main, argv_for([1], fresh("ck"), cfg, "--dry-run", "--restart-superlink"))
    assert not [c for c in h.calls if c[0] == "pkill"]


@test("Q9 Experiment 1 runs through its driver (same command each attempt, dry-run flags passed), exit 0 means finished, failures retry then stop")
def _():
    ck, cfg = fresh("ck"), config_dir()
    with Harness(lambda h, a: 0) as h:
        rc, out, err = capture(rq.main, argv_for([1], ck, cfg, "--dry-run"))
    assert rc == 0 and len(h.calls) == 1
    c = h.calls[0]
    assert c[0] == sys.executable and c[1].endswith("run_experiment1.py") and "--dry-run" in c and c[c.index("--max-epochs") + 1] == "2"
    assert c[c.index("--checkpoint-root") + 1] == str(ck) and c[c.index("--manifest-root") + 1] == str(tmp() / "manifests")
    with Harness(lambda h, a: 1) as h:
        rc, out, err = capture(rq.main, argv_for([1, 2], fresh("ck"), cfg, "--dry-run", "--max-attempts", "2"))
    assert rc == 3 and len(h.calls) == 2 and not h.flwr_calls()
    with Harness(lambda h, a: 0) as h:  # a real (non dry) Experiment 1 run has no dry-run flags
        rc, out, err = capture(rq.main, argv_for([1], fresh("ck"), cfg))
    assert "--dry-run" not in h.calls[0] and "--max-epochs" not in h.calls[0]


@test("Q10 refusals (exit 2): missing pod inputs, missing client CPUs for FL, no GPU on a real run, lock script missing or failing; dry runs skip the GPU check")
def _():
    ck, cfg = fresh("ck"), config_dir()
    base = ["--experiments", "2", "--config-dir", str(cfg), "--skip-lock-check"]
    with Harness(lambda h, a: 0):
        assert capture(rq.main, ["--checkpoint-root", str(ck), "--client-cpus", "4", *base])[0] == 2
        assert capture(rq.main, ["--manifest-root", str(tmp()), "--client-cpus", "4", *base])[0] == 2
        rc, out, err = capture(rq.main, ["--manifest-root", str(tmp()), "--checkpoint-root", str(ck), *base])
        assert rc == 2 and "cpu" in (out + err).lower()
        rq.cuda_available = lambda: False
        full = ["--manifest-root", str(tmp()), "--checkpoint-root", str(ck), "--client-cpus", "4", *base]
        rc, out, err = capture(rq.main, full)
        assert rc == 2 and "gpu" in (out + err).lower()
        rc, out, err = capture(rq.main, [*full, "--dry-run"])
        assert rc == 0 or rc == 3  # allowed to start; the fake never writes done.json
    with Harness(lambda h, a: 0) as h:
        no_skip = [a for a in full if a != "--skip-lock-check"]
        rq.LOCK_SCRIPT = tmp() / "tools" / "does_not_exist.py"
        rc, out, err = capture(rq.main, no_skip)
        assert rc == 2 and "verify_locks" in (out + err) or "lock" in (out + err).lower()
        script = tmp() / "verify_ok.py"
        script.write_text("")
        rq.LOCK_SCRIPT = script
        rc, out, err = capture(rq.main, no_skip)
        assert any(c[:2] == [sys.executable, str(script)] for c in h.calls)  # the lock script was actually run
    codes = {"n": 1}
    with Harness(lambda h, a: codes["n"] if a[:2] == [sys.executable, str(script)] else 0) as h:
        rq.LOCK_SCRIPT = script
        rc, out, err = capture(rq.main, no_skip)
        assert rc == 2 and not h.flwr_calls()


@test("Q11 --status prints all eight experiments from the files and exits 0 without running anything or needing a GPU")
def _():
    ck, cfg = fresh("ck"), config_dir()
    d = ck / "exp2-fedavg"
    d.mkdir()
    (d / "round_001.pt").write_bytes(b"x")
    (d / "latest.pt").write_bytes(b"x")
    with Harness(lambda h, a: 99) as h:
        rq.cuda_available = lambda: False
        rc, out, err = capture(rq.main, ["--status", "--manifest-root", str(tmp()), "--checkpoint-root", str(ck), "--config-dir", str(cfg)])
        assert rc == 0 and h.calls == []
        for n in range(1, 9):
            assert rq.experiment_id(n) in out, n
        assert "in-progress" in out and "not-started" in out and "exp2-fedavg" in out
        rc, out, err = capture(rq.main, ["--status", "--checkpoint-root", str(ck)])
        assert rc == 0 and h.calls == []


@test("Q12 run_cmd for real: tees stdout and stderr to the log and the console, returns the exit code, 127 when the program does not exist, a stop request terminates the child quickly")
def _():
    log = fresh("log") / "a.log"
    code, out, err = None, None, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rq.run_cmd([sys.executable, "-c", "import sys; print('to-out'); print('to-err', file=sys.stderr); sys.exit(3)"], log, dict(os.environ))
    text = log.read_text(encoding="utf-8")
    assert code == 3 and "to-out" in text and "to-err" in text and "to-out" in buf.getvalue()
    assert rq.run_cmd(["definitely-not-a-program-xyz"], fresh("log") / "b.log", None) == 127
    box = {}

    def child():
        with contextlib.redirect_stdout(io.StringIO()):
            box["code"] = rq.run_cmd([sys.executable, "-c", "import time; time.sleep(60)"], fresh("log") / "c.log", dict(os.environ))

    t = threading.Thread(target=child)
    t.start()
    time.sleep(1.5)
    started = time.time()
    rq._on_signal(signal.SIGTERM, None)
    t.join(25)
    assert not t.is_alive() and time.time() - started < 20 and box["code"] != 0 and rq._STOP.is_set()
    rq._STOP.clear()


# ------------------------------------------------------------------ run_experiment1
@test("E1 driver arguments: both roots required, --max-epochs only with --dry-run (default 2), at least 1")
def _():
    base = ["--manifest-root", "m", "--checkpoint-root", "c"]
    a = r1.parse_args(base)
    assert a.dry_run is False and a.max_epochs is None and a.config_dir is None
    assert r1.parse_args([*base, "--dry-run"]).max_epochs == 2 and r1.parse_args([*base, "--dry-run", "--max-epochs", "3"]).max_epochs == 3
    for bad in ([*base, "--max-epochs", "3"], [*base, "--dry-run", "--max-epochs", "0"], ["--checkpoint-root", "c"], ["--manifest-root", "m"]):
        raises(SystemExit, lambda bad=bad: capture(r1.parse_args, bad))


@test("E2 driver build_run: the loader's Experiment 1 kwargs; dry runs get the -dryrun folder and max_epochs; null step-zero values exit with code 2")
def _():
    cfg = config_dir()
    args = r1.parse_args(["--manifest-root", "/m", "--checkpoint-root", "/c", "--config-dir", str(cfg)])
    c, kw = r1.build_run(args)
    assert c["experiment-id"] == "exp1-centralized" and kw["max_epochs"] == 20 and kw["lora_rank"] == 2
    assert str(kw["checkpoint_dir"]).replace("\\", "/").endswith("/c/exp1-centralized")
    d = r1.parse_args(["--manifest-root", "/m", "--checkpoint-root", "/c", "--config-dir", str(cfg), "--dry-run", "--max-epochs", "1"])
    c2, kw2 = r1.build_run(d)
    assert c2["experiment-id"] == "exp1-centralized-dryrun" and kw2["max_epochs"] == 1
    assert str(kw2["checkpoint_dir"]).replace("\\", "/").endswith("/c/exp1-centralized-dryrun")
    nul = config_dir(**{"lora-rank": "null"})
    rc, out, err = capture(r1.main, ["--manifest-root", "/m", "--checkpoint-root", "/c", "--config-dir", str(nul)])
    assert rc == 2 and "lora-rank" in err


@test("E3 driver run(): tracker started with the config, W&B id and callback passed to the trainer, finish(result, None) on success, finish(None, error) then re-raise on failure; exit codes 0, 1, 130")
def _():
    try:
        import training.train_centralized as tc
        import utils.experiment_tracker as et
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"project stack not importable here ({type(exc).__name__})")
    cfg = config_dir()
    argv = ["--manifest-root", "/m", "--checkpoint-root", str(fresh("ck")), "--config-dir", str(cfg)]
    seen = {}

    class T:
        def finish(self, summary=None, error=None):
            seen["finish"] = (summary, error)

    orig = (et.start_centralized, et.make_epoch_callback, tc.run_centralized_training)
    et.start_centralized = lambda c, d: (seen.update(cfg=c, dir=str(d)) or (T(), "run-xyz"))
    et.make_epoch_callback = lambda tr: "CB"
    try:
        tc.run_centralized_training = lambda **kw: (seen.update(kw=kw) or {"final_epoch": 3})
        rc, out, err = capture(r1.main, argv)
        assert rc == 0 and seen["finish"] == ({"final_epoch": 3}, None) and seen["cfg"]["experiment-id"] == "exp1-centralized"
        assert seen["kw"]["wandb_run_id"] == "run-xyz" and seen["kw"]["on_epoch_end"] == "CB" and seen["kw"]["lora_rank"] == 2
        assert str(seen["dir"]).replace("\\", "/").endswith("exp1-centralized") and "finished" in out
        for exc, want in ((RuntimeError("boom"), 1), (KeyboardInterrupt(), 130)):
            def boom(**kw):
                raise exc

            tc.run_centralized_training = boom
            rc, out, err = capture(r1.main, argv)
            assert rc == want and seen["finish"][0] is None and seen["finish"][1] is exc
    finally:
        et.start_centralized, et.make_epoch_callback, tc.run_centralized_training = orig


def main():
    results = []
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, fn in TESTS:
        if only and not any(name.startswith(o + " ") for o in only):
            continue
        try:
            fn()
            results.append((name, "PASS", ""))
        except Skip as e:
            results.append((name, "SKIP", str(e)))
        except BaseException as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:300]}\n      " + "\n      ".join(tb[-4:])))
    print()
    for n, s, msg in results:
        print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
    print(f"\n{sum(s == 'PASS' for _n, s, _m in results)}/{len(results)} passed, {sum(s == 'SKIP' for _n, s, _m in results)} skipped")
    raise SystemExit(any(s == "FAIL" for _n, s, _m in results))


if __name__ == "__main__":
    main()
