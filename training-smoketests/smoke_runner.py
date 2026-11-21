import contextlib
import io
import json
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
    """Replaces run_cmd, guards, sleeper, cuda detection and the lock script, and restores every
    run_queue global it (or a test running under it) changes: functions, polling intervals and
    caps, the interrupt state and FLWR_HOME."""

    RESTORED = (
        "run_cmd", "_sleep", "cuda_available", "LOCK_SCRIPT", "flwr_ls", "stop_flwr_run",
        "acquire_runner_lock", "release_runner_lock",
        "ACTIVE_WAIT_POLL_SECONDS", "INTERRUPT_STOP_POLL_SECONDS", "INTERRUPT_STOP_CAP_SECONDS", "GRACE_SECONDS",
        "_CHILD", "_FL_ATTEMPT", "_FL_INTERRUPTED", "_FL_RUN_ID",
    )

    def __init__(self, script):
        self.script, self.calls, self.sleeps, self.logs = script, [], [], []
        self.saved = {}
        self.stopped = []

    def __call__(self, argv, log_path, env):
        if len(argv) == 2 and str(argv[1]) == str(getattr(self, "default_lock", "")):
            return 0
        self.calls.append(list(argv))
        self.logs.append(str(log_path))
        return self.script(self, list(argv))

    def __enter__(self):
        flwr_home = tmp() / "flwr"
        (flwr_home).mkdir(exist_ok=True)
        (flwr_home / "config.toml").write_text(
            "[superlink.local.options]\n"
            "num-supernodes = 4\n"
            "[superlink.local.options.backend.client-resources]\n"
            "num-cpus = 16\n"
            "num-gpus = 1.0\n"
            "[superlink.local.options.backend.init-args]\n"
            "num-gpus = 1\n",
            encoding="utf-8",
        )
        self.saved_flwr_home = os.environ.get("FLWR_HOME")
        os.environ["FLWR_HOME"] = str(flwr_home)
        for name in self.RESTORED:
            self.saved[name] = getattr(rq, name)
        rq.run_cmd = self
        self.default_lock = tmp() / "verify_default.py"
        self.default_lock.write_text("", encoding="utf-8")
        rq.LOCK_SCRIPT = self.default_lock
        rq._sleep = lambda s: self.sleeps.append(s)
        rq.cuda_available = lambda: True
        rq.flwr_ls = lambda timeout=60: []
        rq.stop_flwr_run = lambda run_id, timeout=60: self.stopped.append(str(run_id)) or True  # never spawns flwr
        rq.acquire_runner_lock = lambda checkpoint_root, argv: object()
        rq.release_runner_lock = lambda handle: None
        rq.ACTIVE_WAIT_POLL_SECONDS = 0
        rq.INTERRUPT_STOP_POLL_SECONDS = 0
        rq.INTERRUPT_STOP_CAP_SECONDS = 1.0
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_INTERRUPTED, rq._FL_RUN_ID = None, False, False, None
        rq._STOP.clear()
        return self

    def __exit__(self, *exc):
        for name, v in self.saved.items():
            setattr(rq, name, v)
        if self.saved_flwr_home is None:
            os.environ.pop("FLWR_HOME", None)
        else:
            os.environ["FLWR_HOME"] = self.saved_flwr_home
        rq._STOP.clear()

    def flwr_calls(self):
        return [c for c in self.calls if c and c[0] == "flwr"]

    def ids(self):
        return [Path(c[4]).stem for c in self.flwr_calls()]


class FakeChild:
    """Stands in for the streaming `flwr run ... --stream` process during an interrupt."""

    def __init__(self, events=None):
        self.events, self.code = events if events is not None else [], None

    def poll(self):
        return self.code

    def send_signal(self, sig):
        raise RuntimeError("an interrupt must never be forwarded to the streaming CLI")

    def terminate(self):
        self.events.append(("terminate",))
        self.code = -15

    def kill(self):
        self.events.append(("kill",))
        self.code = -9


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
    return ["--manifest-root", str(tmp() / "manifests"), "--checkpoint-root", str(ck), "--client-cpus", "16", "--experiments",
            *map(str, experiments), "--config-dir", str(cfg), *extra]


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


@test("Q10 refusals (exit 2): missing pod inputs, missing client CPUs for FL, no CUDA GPU (real runs AND --dry-run, which also trains on the GPU), lock script missing or failing")
def _():
    ck, cfg = fresh("ck"), config_dir()
    base = ["--experiments", "2", "--config-dir", str(cfg)]
    with Harness(lambda h, a: 0) as h:
        assert capture(rq.main, ["--checkpoint-root", str(ck), "--client-cpus", "4", *base])[0] == 2
        assert capture(rq.main, ["--manifest-root", str(tmp()), "--client-cpus", "4", *base])[0] == 2
        rc, out, err = capture(rq.main, ["--manifest-root", str(tmp()), "--checkpoint-root", str(ck), *base])
        assert rc == 2 and "cpu" in (out + err).lower()
        rq.cuda_available = lambda: False
        full = ["--manifest-root", str(tmp()), "--checkpoint-root", str(ck), "--client-cpus", "16", *base]
        rc, out, err = capture(rq.main, full)
        assert rc == 2 and "gpu" in (out + err).lower()
        rc, out, err = capture(rq.main, [*full, "--dry-run"])  # a dry run is real GPU training: refused too
        assert rc == 2 and "cuda" in (out + err).lower() and "--dry-run" in (out + err), (out, err)
        exp1_dry = ["--manifest-root", str(tmp()), "--checkpoint-root", str(ck), "--experiments", "1",
                    "--config-dir", str(cfg), "--dry-run"]
        rc, out, err = capture(rq.main, exp1_dry)  # Experiment 1 dry runs train on the GPU as well
        assert rc == 2 and "cuda" in (out + err).lower(), (out, err)
        assert h.calls == [], h.calls  # nothing was started by any refused invocation
    with Harness(lambda h, a: 0) as h:
        rq.LOCK_SCRIPT = tmp() / "tools" / "does_not_exist.py"
        rc, out, err = capture(rq.main, full)
        assert rc == 2 and "verify_locks" in (out + err) or "lock" in (out + err).lower()
        script = tmp() / "verify_ok.py"
        script.write_text("")
        rq.LOCK_SCRIPT = script
        rc, out, err = capture(rq.main, [*full, "--dry-run"])
        assert any(c[:2] == [sys.executable, str(script)] for c in h.calls)  # the lock script was actually run
    codes = {"n": 1}
    with Harness(lambda h, a: codes["n"] if a[:2] == [sys.executable, str(script)] else 0) as h:
        rq.LOCK_SCRIPT = script
        rc, out, err = capture(rq.main, [*full, "--dry-run"])
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

@test("R16 parse_flwr_ls accepts verified empty, running, starting, pending and finished:completed statuses")
def _():
    assert rq.parse_flwr_ls('{"success": true, "runs": []}') == []
    for status in ("running", "starting", "pending"):
        active = rq.parse_flwr_ls(json.dumps({
            "success": True,
            "runs": [{"run-id": "123", "status": status}],
        }))
        assert len(active) == 1 and active[0]["run_id"] == "123"
    integer_active = rq.parse_flwr_ls(json.dumps({
        "success": True,
        "runs": [{"run-id": 12345, "status": "running"}],
    }))
    assert len(integer_active) == 1 and integer_active[0]["run_id"] == "12345"
    assert rq.parse_flwr_ls('{"success": true, "runs": [{"run-id": "456", "status": "finished:completed"}]}') == []
    mixed = rq.parse_flwr_ls(json.dumps({
        "success": True,
        "runs": [{"run-id": "456", "status": "finished:completed"}, {"run-id": "789", "status": "running"}],
    }))
    assert [r["run_id"] for r in mixed] == ["789"]  # finished runs never count as active


@test("R17 parse_flwr_ls rejects unknown status, malformed JSON, success false, missing runs and shape errors")
def _():
    raises(rq.GuardError, lambda: rq.parse_flwr_ls("{not-json"))
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": false, "runs": []}'))
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": true}'))
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": true, "runs": [{}]}'))
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": true, "runs": [{"status": "queued"}]}'))
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": true, "runs": [42]}'))
    # an ACTIVE run must carry a decimal run id: a non-decimal one is refused, never guessed at
    raises(rq.GuardError, lambda: rq.parse_flwr_ls('{"success": true, "runs": [{"run-id": "run-123", "status": "running"}]}'),
           contains="invalid run-id")


@test("R18 active-run guard refuses the first FL attempt and names the run plus stop command")
def _():
    ck, cfg = fresh("ck"), config_dir()
    with Harness(lambda h, a: 0) as h:
        rq.flwr_ls = lambda timeout=60: [{
            "run_id": "12345",
            "raw": {"run-id": 12345, "status": "running"},
            "status": "running",
        }]
        rc, out, err = capture(rq.main, argv_for([2], ck, cfg, "--dry-run"))
    assert rc == 2 and h.flwr_calls() == [] and "12345" in err and "flwr stop 12345" in err
    summary = out[out.index("Summary:"):]
    assert "refused" in summary and "FAILED" not in summary and "FAILED" not in err, (out, err)


@test("R19 a failed stream that leaves a Flower run active: the runner waits, and the retry cannot launch until that run is inactive")
def _():
    ck, cfg = fresh("ck"), config_dir()
    with Harness(lambda h, a: 0):
        polls = {"n": 0}

        def ls_once(timeout=60):
            polls["n"] += 1
            return [{"run_id": "123", "raw": {"run-id": "123"}, "status": "running"}] if polls["n"] == 1 else []

        rq.flwr_ls = ls_once
        assert rq.wait_for_no_active_runs(1.0) is True and polls["n"] == 2

    events, flower = [], {"active": False, "active_polls": 0, "active_at_launch": []}
    ACTIVE_POLLS = 3  # the stream died, but the run stays active for three polls

    def ls(timeout=60):
        if flower["active"] and flower["active_polls"] >= ACTIVE_POLLS:
            flower["active"] = False  # the run finally becomes inactive
        if flower["active"]:
            flower["active_polls"] += 1
        events.append("active" if flower["active"] else "clear")
        return [{"run_id": "123", "raw": {"run-id": "123", "status": "running"}, "status": "running"}] if flower["active"] else []

    def retry_script(h, argv):
        events.append("launch")
        flower["active_at_launch"].append(flower["active"])
        if len(flower["active_at_launch"]) == 1:
            flower["active"] = True  # the stream fails, but its Flower run is still running
            return 1
        complete(ck, argv)
        return 0

    with Harness(retry_script) as h:
        rq.flwr_ls = ls
        rc, out, err = capture(rq.main, argv_for([2], ck, cfg, "--dry-run", "--active-wait-minutes", "1"))
        assert rq.ACTIVE_WAIT_POLL_SECONDS == 0  # this test's own polling interval is in force here
    assert rc == 0 and len(h.flwr_calls()) == 2, (rc, out, err)
    assert flower["active_polls"] == ACTIVE_POLLS and events.count("active") == ACTIVE_POLLS, events  # it really waited
    assert flower["active_at_launch"] == [False, False], flower  # no launch while a run was active
    launches = [i for i, e in enumerate(events) if e == "launch"]
    assert len(launches) == 2 and events.index("active") > launches[0], events
    assert max(i for i, e in enumerate(events) if e == "active") < launches[1], events  # retry only after the last active poll
    assert events[launches[1] - 1] == "clear", events  # and the retry's own guard saw the run gone
    assert h.stopped == []  # waiting never stops anything


@test("R20 active-run wait timeout is reported as its own FAILED result, never stops a run, and never relaunches")
def _():
    ck, cfg = fresh("ck"), config_dir()
    with Harness(lambda h, a: 0) as h:
        rq.flwr_ls = lambda timeout=60: [{"run_id": "999", "raw": {"run-id": "999"}, "status": "running"}]
        assert rq.wait_for_no_active_runs(0) is False
    assert h.stopped == []

    state = {"launched": False}

    def script(h, argv):
        state["launched"] = True
        return 1  # the stream fails and its run lingers

    def ls(timeout=60):
        if not state["launched"]:
            return []
        return [{"run_id": "999", "raw": {"run-id": "999"}, "status": "running"}]

    with Harness(script) as h:
        rq.flwr_ls = ls
        rc, out, err = capture(rq.main, argv_for([2, 3], ck, cfg, "--dry-run", "--active-wait-minutes", "0"))
    summary = out[out.index("Summary:"):]
    assert rc == 3 and len(h.flwr_calls()) == 1 and h.stopped == [], (rc, out, err)
    assert "FAILED (active Flower run wait timed out)" in summary, summary  # not overwritten by a generic FAILED
    assert "FAILED after" not in err and "still active: 999" in err, err
    assert "exp3" in summary and "not run" in summary  # the queue stopped there


@test("R21 second runner lock is refused and a stale lock left by a killed process is acquirable")
def _():
    try:
        import fcntl
    except ImportError:
        raise Skip("fcntl unavailable")
    ck = fresh("lock")
    first = rq.acquire_runner_lock(ck, ["--dry-run"])
    try:
        raises(rq.RunnerLockHeld, lambda: rq.acquire_runner_lock(ck, ["--dry-run"]))
    finally:
        rq.release_runner_lock(first)
    stale = rq.acquire_runner_lock(ck, ["--dry-run"])
    rq.release_runner_lock(stale)


@test("R22 interrupt during FL stops only this queue's run, polls until that run is inactive, then terminates the CLI; never forwards SIGINT")
def _():
    events = []
    child = FakeChild(events)
    mine = {"run_id": "111", "raw": {"run-id": "111"}, "status": "running"}
    other = {"run_id": "222", "raw": {"run-id": "222"}, "status": "running"}  # someone else's run
    flower = {"mine_active": True, "stopped": False, "polls_after_stop": 0}

    def ls(timeout=60):
        if flower["stopped"]:
            flower["polls_after_stop"] += 1
            if flower["polls_after_stop"] >= 3:
                flower["mine_active"] = False
        events.append(("ls", flower["mine_active"]))
        return ([mine] if flower["mine_active"] else []) + [other]

    def stop(run_id, timeout=60):
        events.append(("stop", run_id))
        flower["stopped"] = True
        return True

    with Harness(lambda h, a: 0):
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, "111"
        rq.flwr_ls, rq.stop_flwr_run = ls, stop
        rc, out, err = capture(rq._on_signal, signal.SIGINT, None)
        assert rq._STOP.is_set()
    assert [e[1] for e in events if e[0] == "stop"] == ["111"], events  # never the unrelated run 222
    assert flower["polls_after_stop"] >= 3 and ("ls", True) in events, events
    t = events.index(("terminate",))
    assert [e for e in events[:t] if e[0] == "ls"][-1] == ("ls", False), events  # verified inactive BEFORE terminating
    assert ("kill",) not in events and ("terminate",) in events
    assert rq._CHILD is None and rq._FL_ATTEMPT is False and rq._FL_RUN_ID is None  # Harness restored the interrupt state
    assert rq.INTERRUPT_STOP_CAP_SECONDS == 300.0 and rq.ACTIVE_WAIT_POLL_SECONDS == 30.0 and rq.INTERRUPT_STOP_POLL_SECONDS == 30.0


@test("R23 flwr ls failure is fail-closed and a malformed response refuses launch")
def _():
    ck, cfg = fresh("ck"), config_dir()
    for failure in (
        rq.GuardError("flwr ls exited 1"),
        rq.GuardError("flwr ls returned malformed JSON"),
    ):
        with Harness(lambda h, a: 0) as h:
            def fail(timeout=60, failure=failure):
                raise failure
            rq.flwr_ls = fail
            rc, out, err = capture(rq.main, argv_for([2], ck, cfg, "--dry-run"))
        assert rc == 2 and h.flwr_calls() == []


@test("R24 hard Flower config gate refuses missing or incorrect required values")
def _():
    path = fresh("flower") / "config.toml"
    valid = (
        "[superlink.local.options]\nnum-supernodes = 4\n"
        "[superlink.local.options.backend.client-resources]\nnum-cpus = 16\nnum-gpus = 1.0\n"
        "[superlink.local.options.backend.init-args]\nnum-gpus = 1\n"
    )
    path.write_text(valid, encoding="utf-8")
    assert rq.validate_flower_config(path) == []
    for bad in (
        valid.replace("num-supernodes = 4", "num-supernodes = 2"),
        valid.replace("num-cpus = 16", "num-cpus = 8"),
        valid.replace("num-gpus = 1.0", "num-gpus = 0.5"),
        valid.replace("num-gpus = 1\n", "num-gpus = 2\n"),
    ):
        path.write_text(bad, encoding="utf-8")
        assert rq.validate_flower_config(path)


@test("R25 no --skip-lock-check bypass and queue lock is acquired before preflight")
def _():
    raises(SystemExit, lambda: rq.parse_args(["--skip-lock-check"]))
    ck, cfg = fresh("ck"), config_dir()
    seen = {"n": 0}
    def acquire(root, argv):
        seen["n"] += 1
        return object()
    with Harness(lambda h, a: 0) as h:
        rq.acquire_runner_lock = acquire
        rq.LOCK_SCRIPT = tmp() / "missing-lock.py"
        rc, out, err = capture(rq.main, argv_for([2], ck, cfg, "--dry-run"))
    assert rc == 2 and seen["n"] == 1


@test("R26 interrupt never terminates the streaming CLI while its run is active or cannot be identified, and never stops an unrelated run")
def _():
    mine = {"run_id": "111", "raw": {"run-id": "111"}, "status": "running"}
    other = {"run_id": "222", "raw": {"run-id": "222"}, "status": "running"}

    def broken(timeout=60):
        raise rq.GuardError("flwr ls exited 1")

    # (a) our run ignores flwr stop and stays active past the cap: the CLI is left alone
    events = []
    child = FakeChild(events)
    with Harness(lambda h, a: 0) as h:
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, "111"
        rq.flwr_ls = lambda timeout=60: [mine, other]
        rq.INTERRUPT_STOP_CAP_SECONDS = 0
        rc, out, err = capture(rq._on_signal, signal.SIGTERM, None)
        assert rq._STOP.is_set()  # the queue is still shutting down, it just does not kill the CLI
    assert events == [] and child.poll() is None, events  # neither terminate nor kill
    assert h.stopped == ["111"] and "flwr stop 111" in err

    # (b) several active runs and no announced id: nothing can be attributed to this queue
    events = []
    child = FakeChild(events)
    with Harness(lambda h, a: 0) as h:
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, None
        rq.flwr_ls = lambda timeout=60: [mine, other]
        rc, out, err = capture(rq._on_signal, signal.SIGINT, None)
    assert events == [] and h.stopped == [] and "cannot tell" in err, (events, h.stopped, err)

    # (c) flwr ls itself fails and no id was announced: fail closed
    events = []
    child = FakeChild(events)
    with Harness(lambda h, a: 0) as h:
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, None
        rq.flwr_ls = broken
        rc, out, err = capture(rq._on_signal, signal.SIGINT, None)
    assert events == [] and h.stopped == [], (events, h.stopped)

    # (d) the announced id is known but flwr ls fails: stop that one run, never terminate unverified
    events = []
    child = FakeChild(events)
    with Harness(lambda h, a: 0) as h:
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, "111"
        rq.flwr_ls = broken
        rq.INTERRUPT_STOP_CAP_SECONDS = 0
        rc, out, err = capture(rq._on_signal, signal.SIGINT, None)
    assert events == [] and h.stopped == ["111"], (events, h.stopped)

    # (e) the id is unknown, exactly one run is active: that one is ours, stopped and verified before terminating
    events = []
    child = FakeChild(events)
    seen = {"stopped": False}

    def ls_one(timeout=60):
        return [] if seen["stopped"] else [mine]

    with Harness(lambda h, a: 0) as h:
        rq._CHILD, rq._FL_ATTEMPT, rq._FL_RUN_ID = child, True, None
        rq.flwr_ls = ls_one
        rq.stop_flwr_run = lambda run_id, timeout=60: seen.update(stopped=True) or h.stopped.append(run_id) or True
        rc, out, err = capture(rq._on_signal, signal.SIGINT, None)
    assert h.stopped == ["111"] and events == [("terminate",)], (h.stopped, events)


@test("R27 run_cmd records the run id announced by this queue's own streaming CLI, and only during a Flower attempt")
def _():
    announce = [sys.executable, "-c", "print('noise'); print('>> Successfully started run 1859953118041441032'); print('Successfully started run 7')"]
    with Harness(lambda h, a: 0) as h:
        real_run_cmd = h.saved["run_cmd"]
        with contextlib.redirect_stdout(io.StringIO()):
            rq._FL_ATTEMPT, rq._FL_RUN_ID = True, None
            assert real_run_cmd(announce, fresh("log") / "a.log", dict(os.environ)) == 0
            assert rq._FL_RUN_ID == "1859953118041441032"  # the first announcement wins
            rq._FL_ATTEMPT, rq._FL_RUN_ID = False, None
            assert real_run_cmd(announce, fresh("log") / "b.log", dict(os.environ)) == 0
            assert rq._FL_RUN_ID is None  # verify_locks / pkill output is never mistaken for a run


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