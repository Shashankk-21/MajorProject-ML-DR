import ast
import contextlib
import copy
import inspect
import io
import re
import shutil
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import loader as ld
import federated.server as sv

try:
    import tomllib # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

TMP = Path(tempfile.mkdtemp(prefix="configs_smoke_"))
REAL = ROOT / "configs"
RESULTS = []
STEP = {"lora-rank": "8", "lora-alpha": "16", "max-grad-norm": "1.0"}
KW = dict(manifest_root="/data/manifests", checkpoint_root="/data/ck", env={})
HOOK = "utils.experiment_tracker:make_hook"
EXPECT = {
    1: ("exp1-centralized", "centralized", False, None), 2: ("exp2-fedavg", "fedavg", False, None),
    3: ("exp3-fedprox", "fedprox", False, None), 4: ("exp4-fedprox-eps50", "fedprox", True, 50.0),
    5: ("exp5-fedprox-eps10", "fedprox", True, 10.0), 6: ("exp6-fedprox-eps5", "fedprox", True, 5.0),
    7: ("exp7-fedprox-eps1", "fedprox", True, 1.0), 8: ("exp8-fedprox-eps0.5", "fedprox", True, 0.5),
}


class Skip(Exception):
    pass


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, "PASS", ""))
        except Skip as e:
            RESULTS.append((name, "SKIP", str(e)))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:400]}"))
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


_COUNT = [0]


def workdir(step=None, edit=None):
    """Temp copy of the real configs/*.yaml with the step-zero values filled (or set as given)."""
    _COUNT[0] += 1
    d = TMP / f"cfg{_COUNT[0]}"
    d.mkdir()
    for f in REAL.glob("*.yaml"):
        shutil.copy(f, d)
    text = (d / "base.yaml").read_text(encoding="utf-8")
    for k, v in (STEP if step is None else step).items():
        text, n = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", text, flags=re.M)
        assert n == 1, k
    (d / "base.yaml").write_text(text, encoding="utf-8")
    for fname, new_text in (edit or {}).items():
        (d / fname).write_text(new_text, encoding="utf-8")
    return d


GOOD = workdir()


def load(n, d=None, **k):
    return ld.load_experiment_config(n, config_dir=d or GOOD, **{**KW, **k})


def all_cfgs(d=None):
    return {n: load(n, d) for n in range(1, 9)}


@check("K1 the 8 experiments: ids, algorithm, dp, epsilon, mu, derived calibration path, hook, key families")
def _():
    for n, (eid, alg, dp, eps) in EXPECT.items():
        c = load(n)
        assert c["experiment-id"] == eid and c["algorithm"] == alg and c["dp"] is dp, n
        assert list(c)[:3] == ["experiment-id", "algorithm", "dp"]
        if dp:
            assert c["target-epsilon"] == eps and type(c["target-epsilon"]) is float
            assert c["calibration-path"] == f"/data/ck/{eid}/calibration.json"
            assert ld.DP_ONLY_KEYS <= set(c)
        else:
            assert not (ld.DP_ONLY_KEYS & set(c)), n
        assert (c.get("proximal-mu") == 0.01) if alg == "fedprox" else ("proximal-mu" not in c), n
        if n == 1:
            assert "run-hook" not in c and not (ld.FL_ONLY_KEYS & set(c)) and ld.CENTRALIZED_ONLY_KEYS <= set(c)
        else:
            assert c["run-hook"] == HOOK and not (ld.CENTRALIZED_ONLY_KEYS & set(c))
            assert ld.FL_ONLY_KEYS <= set(c)
        assert all(v is not None for v in c.values()), n


@check("K2 locked decisions are pinned as values (seed, workers, batches, LRs, schedule, DP constants)")
def _():
    for n, c in all_cfgs().items():
        assert c["run-seed"] == 42 and c["num-workers"] == 8 and c["img-size"] == 224 and c["num-classes"] == 5
        assert c["pretrained"] is True and c["effective-batch-size"] == 64
        assert c["physical-batch-size"] * c["accumulation-steps"] == c["effective-batch-size"]
        assert c["label-smoothing"] == 0.1 and c["weight-decay"] == 0.01
        assert c["lr-lora-peak"] == 0.0001 and c["lr-head-peak"] == 0.001
        assert c["lora-rank"] == 8 and c["lora-alpha"] == 16  # the test's step-zero stand-ins
        if n > 1:
            assert c["num-rounds"] == 10 and c["local-epochs"] == 3
            assert c["warmup-rounds"] == 1 and c["warmup-start-factor"] == 0.5 and c["lr-min-factor"] == 0.1
        if c["dp"]:
            assert c["accountant"] == "rdp" and c["delta"] == 1e-5 and c["grad-sample-mode"] == "functorch"
            assert c["dp-logical-batch-size"] == 64 == c["effective-batch-size"]
    c1 = load(1)
    assert (c1["centralized-max-epochs"], c1["centralized-patience"]) == (20, 5)
    assert c1["centralized-warmup-epochs"] == 1.0 and c1["centralized-warmup-start-factor"] == 0.001


@check("K3 null step-zero values and missing pod inputs are refused, never defaulted")
def _():
    for key in ("lora-rank", "lora-alpha"):
        d = workdir(step={**STEP, key: "null"})
        for n in range(1, 9):
            raises(ld.ConfigError, lambda n=n, d=d: load(n, d), contains=key)
    d = workdir(step={**STEP, "max-grad-norm": "null"})
    for n in (1, 2, 3):  # non-DP experiments do not use max-grad-norm
        assert "max-grad-norm" not in load(n, d)
    for n in range(4, 9):
        raises(ld.ConfigError, lambda n=n: load(n, d), contains="max-grad-norm")
    raises(ld.ConfigError, lambda: ld.load_experiment_config(4, config_dir=GOOD, env={}), contains="manifest-root")
    raises(ld.ConfigError, lambda: ld.load_experiment_config(4, config_dir=GOOD, manifest_root="/m", env={}),
           contains="checkpoint-root")
    env = {"DR_MANIFEST_ROOT": "/env/m", "DR_CHECKPOINT_ROOT": "/env/c"}
    c = ld.load_experiment_config(4, config_dir=GOOD, env=env)
    assert c["manifest-root"] == "/env/m" and c["calibration-path"].startswith("/env/c/exp4-fedprox-eps50/")
    c = ld.load_experiment_config(4, config_dir=GOOD, env=env, manifest_root="/cli/m")
    assert c["manifest-root"] == "/cli/m"  # CLI beats the environment
    raises(ld.ConfigError, lambda: load(4, overrides={"num-workers": 0}), contains="num-workers")
    c = ld.load_experiment_config(4, config_dir=workdir(step={"lora-rank": "null", "lora-alpha": "null", "max-grad-norm": "null"}),
                                  env={}, allow_unset=True)
    assert c["lora-rank"] is None and c["manifest-root"] is None


@check("K4 override whitelist: only the seven keys; num-rounds only for an isolated dry run; unknown keys in yaml refused")
def _():
    for key, val in (("algorithm", "fedavg"), ("dp", True), ("lora-rank", 4), ("lora-alpha", 8), ("max-grad-norm", 2.0),
                     ("target-epsilon", 1.0), ("proximal-mu", 0.5), ("lr-lora-peak", 0.1), ("local-epochs", 1),
                     ("experiment-id", "x"), ("accountant", "prv"), ("delta", 0.1), ("run-hook", "a:b")):
        raises(ld.ConfigError, lambda key=key, val=val: load(4, overrides={key: val}), contains=key)
    c = ld.load_experiment_config(4, config_dir=GOOD, env={}, overrides={
        "num-workers": 4, "run-seed": 7, "wandb-cm-every": 3, "diagnostic": True,
        "manifest-root": "/o/m", "checkpoint-root": "/o/c"})
    assert (c["num-workers"], c["run-seed"], c["wandb-cm-every"], c["diagnostic"]) == (4, 7, 3, True)
    assert c["manifest-root"] == "/o/m" and c["calibration-path"].startswith("/o/c/")
    raises(ld.ConfigError, lambda: load(4, overrides={"num-rounds": 2}), contains="dry run")
    c = load(4, overrides={"num-rounds": 2}, dry_run=True)
    assert c["num-rounds"] == 2 and c["experiment-id"] == "exp4-fedprox-eps50-dryrun" and c["diagnostic"] is True
    assert c["calibration-path"] == "/data/ck/exp4-fedprox-eps50-dryrun/calibration.json"
    assert load(4)["diagnostic"] is False and load(4)["experiment-id"] == "exp4-fedprox-eps50"
    exp4 = (REAL / "exp4.yaml").read_text(encoding="utf-8")
    raises(ld.ConfigError, lambda: load(4, workdir(edit={"exp4.yaml": exp4 + "lora-rank: 4\n"})), contains="lora-rank")
    raises(ld.ConfigError, lambda: load(4, workdir(edit={"exp4.yaml": exp4 + "dp: false\ndp: true\n"})), contains="duplicate")
    drift = workdir(edit={"exp4.yaml": exp4.replace("50", "10")})
    assert load(4, drift)["experiment-id"] == "exp4-fedprox-eps10"  # the id is derived, so loading alone cannot notice
    raises(ld.ComparabilityError, lambda: ld.check_comparability(config_dir=drift, **KW), contains="target-epsilon")
    raises(ld.ConfigError, lambda: load(2, workdir(edit={"exp2.yaml": "algorithm: fedavg\nproximal-mu: 0.01\n"})))
    raises(ld.ConfigError, lambda: load(3, workdir(edit={"exp3.yaml": "algorithm: fedprox\n"})), contains="proximal-mu")
    raises(ld.ConfigError, lambda: load(1, workdir(edit={"exp1.yaml": "algorithm: fedavg\n"})))
    raises(ld.ConfigError, lambda: load(5, workdir(edit={"exp5.yaml": "algorithm: fedprox\ndp: false\ntarget-epsilon: 10\nproximal-mu: 0.01\n"})))
    base = (workdir() / "base.yaml").read_text(encoding="utf-8")
    raises(ld.ConfigError, lambda: load(2, workdir(edit={"base.yaml": base + "surprise-key: 1\n"})), contains="surprise-key")
    raises(ld.ConfigError, lambda: load(0))
    raises(ld.ConfigError, lambda: load(9))


@check("K5 check_comparability passes on the real files and fails loudly on every kind of drift")
def _():
    ld.check_comparability(config_dir=GOOD, **KW)
    good = {n: load(n) for n in range(2, 9)}

    def mutated(n, **changes):
        c = copy.deepcopy(good)
        for k, v in changes.items():
            if v == "<drop>":
                c[n].pop(k)
            else:
                c[n][k] = v
        return c

    for n, changes, word in (
            (5, {"lr-lora-peak": 0.5}, "lr-lora-peak"), (3, {"label-smoothing": 0.2}, "label-smoothing"),
            (7, {"proximal-mu": 0.5}, "proximal-mu"), (6, {"target-epsilon": 4.0}, "target-epsilon"),
            (6, {"accountant": "prv"}, "accountant"), (8, {"delta": 1e-6}, "delta"),
            (2, {"algorithm": "fedprox"}, "algorithm"), (4, {"experiment-id": "exp4-fedprox-eps5"}, "experiment-id"),
            (4, {"calibration-path": "/elsewhere/c.json"}, "calibration-path"), (3, {"extra-key": 1}, "extra-key"),
            (5, {"num-rounds": "<drop>"}, "num-rounds"), (2, {"dp": True}, "dp"),
            (3, {"manifest-root": "/other"}, "manifest-root"), (8, {"run-seed": 1}, "run-seed")):
        raises(ld.ComparabilityError, lambda n=n, ch=changes: ld.check_comparability(mutated(n, **ch)), contains=word)
    raises(ld.ComparabilityError, lambda: ld.check_comparability({n: good[n] for n in range(2, 8)}))


@check("K6 every exp2..8 config passes the REAL server validator, alone and merged with the pyproject defaults flwr adds")
def _():
    for n in range(2, 9):
        c = load(n)
        out = sv.validate_server_config(dict(c))
        assert out["experiment-id"] == c["experiment-id"]
        assert ld.preflight_pyproject(c) == [], (n, ld.preflight_pyproject(c))
    raises(ld.ConfigError, lambda: ld.preflight_pyproject({**load(4), "bogus-key": 1}), contains="bogus-key")
    for bad in ({"proximal-mu": 0.01, "algorithm": "fedavg"}, {"proximal-mu": True, "algorithm": "fedavg"}):
        raises(ValueError, lambda bad=bad: sv.validate_server_config({**load(2), **bad}))
    raises(ValueError, lambda: sv.validate_server_config({**load(3), "proximal-mu": 0.0}))
    assert "proximal-mu" in sv.validate_server_config({**load(2), "proximal-mu": 0.0})  # tolerated, never sent


@check("K7 pyproject.toml: only flwr as dependency, no federations table, components, fab-exclude, every server key declared, bare run fails")
def _():
    if tomllib is None:
        raise Skip("no tomllib/tomli")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == ["flwr==1.37.0"]
    flwr_tbl = data["tool"]["flwr"]
    assert "federations" not in flwr_tbl
    assert flwr_tbl["app"]["components"] == {"serverapp": "federated.server:app", "clientapp": "federated.client:app"}
    excl = flwr_tbl["app"]["fab-exclude"]
    assert "training-smoketests/**" in excl and "notebooks/**" in excl
    declared = flwr_tbl["app"]["config"]
    assert all(isinstance(v, (bool, int, float, str)) for v in declared.values())
    used = set().union(*[set(load(n)) for n in range(2, 9)])
    assert used <= set(declared), sorted(used - set(declared))
    server_keys = set(sv._REQUIRED) | set(sv._DP_REQUIRED) | set(sv._DEFAULTS)
    server_keys |= {"run-hook", "wandb-cm-every", "diagnostic", "proximal-mu", "calibration-path", "algorithm", "dp"}
    assert server_keys <= set(declared), sorted(server_keys - set(declared))
    assert declared["run-hook"] == "" and declared["dp"] is False
    raises(ValueError, lambda: sv.validate_server_config(dict(declared)))  # a bare `flwr run .` cannot run
    for key in ("lora-rank", "lora-alpha", "num-rounds", "local-epochs", "img-size"):
        assert declared[key] == 0, key  # placeholders are invalid on purpose


@check("K8 rendered run-config TOML parses back to identical values and types; null, nan, inf, lists and odd keys refused")
def _():
    if tomllib is None:
        raise Skip("no tomllib/tomli")
    for n in range(2, 9):
        c = load(n)
        text = ld.render_run_config_toml(c)
        back = tomllib.loads(text)
        assert back == c and all(type(back[k]) is type(c[k]) for k in c), n
        assert not any(isinstance(v, dict) for v in back.values())
        assert text.splitlines()[0].startswith("#") and "dryrun" not in text
    assert "DRY RUN" in ld.render_run_config_toml(load(4, overrides={"num-rounds": 2}, dry_run=True))
    tricky = {"experiment-id": "x", "p": 'C:\\data\\"q" \u00fc\n tab\t', "f": 1e-05, "i": 0, "b": False, "big": 1.5e+20}
    assert tomllib.loads(ld.render_run_config_toml(tricky)) == tricky
    for bad in ({"a": None}, {"a": float("nan")}, {"a": float("inf")}, {"a": [1, 2]}, {"a": {"b": 1}}, {"a b": 1}, {"a.b": 1}):
        raises(ld.ConfigError, lambda bad=bad: ld.render_run_config_toml(bad))


@check("K9 flwr itself accepts the generated file: --run-config parse, merge with pyproject (unknown key refused), validator on the merge")
def _():
    try:
        from flwr.common.config import get_fused_config_from_dir, parse_config_args
    except ImportError:
        raise Skip("flwr config helpers not importable here")
    if tomllib is None:
        raise Skip("no tomllib/tomli")
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    for n in range(2, 9):
        c = load(n)
        f = TMP / f"rc{n}.toml"
        f.write_text(ld.render_run_config_toml(c), encoding="utf-8")
        try:
            parsed = parse_config_args([str(f)], flatten=False)
            fused = get_fused_config_from_dir(ROOT, parsed)
        except TypeError:
            raise Skip("flwr config helper signatures differ in this version")
        assert parsed == c, n
        assert dict(fused) == {**declared, **c}, n
        sv.validate_server_config(dict(fused))
    raises(ValueError, lambda: get_fused_config_from_dir(ROOT, {"bogus-key": 1}))


@check("K10 Experiment 1 kwargs equal run_centralized_training's parameters (minus the two hooks) and match the config")
def _():
    c = load(1)
    kwargs = ld.to_centralized_kwargs(c)
    try:
        from training.train_centralized import run_centralized_training
        params = inspect.signature(run_centralized_training).parameters
        names = set(params)
        required = {k for k, p in params.items() if p.default is inspect.Parameter.empty}
    except ImportError:
        tree = ast.parse((ROOT / "training" / "train_centralized.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_centralized_training")
        a = fn.args
        names = {x.arg for x in a.args + a.kwonlyargs}
        required = {x.arg for x in a.args[:len(a.args) - len(a.defaults)]}
    assert set(kwargs) == names - {"wandb_run_id", "on_epoch_end"}, (set(kwargs) ^ (names - {"wandb_run_id", "on_epoch_end"}))
    assert required <= set(kwargs)
    mp = kwargs["client_manifest_paths"]
    assert list(mp) == list(ld.CLIENTS) and all(isinstance(v, Path) for v in mp.values())
    assert all(mp[n] == Path("/data/manifests") / f"{n}_manifest.csv" for n in mp)
    try:
        assert tuple(sv.CLIENT_NAMES) == ld.CLIENTS
    except AttributeError:
        pass
    assert kwargs["checkpoint_dir"] == "/data/ck/exp1-centralized"
    want = {"lora_rank": 8, "lora_alpha": 16, "img_size": 224, "num_classes": 5, "physical_batch_size": 32,
            "accumulation_steps": 2, "effective_batch_size": 64, "num_workers": 8, "lr_lora": 0.0001, "lr_head": 0.001,
            "weight_decay": 0.01, "warmup_epochs": 1.0, "warmup_start_factor": 0.001, "max_epochs": 20, "patience": 5,
            "label_smoothing": 0.1, "seed": 42}
    assert {k: kwargs[k] for k in want} == want
    assert all(type(kwargs[k]) is type(v) for k, v in want.items())
    raises(ld.ConfigError, lambda: ld.to_centralized_kwargs(load(2)))
    raises(ld.ConfigError, lambda: ld.to_centralized_kwargs({**c, "lora-rank": None}))


def _cli(argv, patch_dir=GOOD, clear_env=()):
    out, err = io.StringIO(), io.StringIO()
    old = ld.CONFIG_DIR
    saved_env = {name: os.environ.pop(name, None) for name in clear_env}
    ld.CONFIG_DIR = patch_dir
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = ld.main(argv)
    finally:
        ld.CONFIG_DIR = old
        for name, value in saved_env.items():
            if value is not None:
                os.environ[name] = value
    return rc, out.getvalue(), err.getvalue()


@check("K11 CLI: renders the TOML, prints the setup and the exact flwr command; refuses missing CPU count, nulls, bad flags; Exp 1 prints kwargs")
def _():
    ck = TMP / "ck"
    base = ["--manifest-root", "/data/manifests", "--checkpoint-root", str(ck)]
    rc, out, err = _cli(["4", *base, "--client-cpus", "6"])
    f = ck / "_run-configs" / "exp4-fedprox-eps50.toml"
    assert rc == 0 and f.is_file() and "WARNING" not in err, err
    assert "flwr run . --run-config" in out and "--stream" in out and "num-supernodes = 4" in out
    assert "num-cpus = 6" in out and "--client-resources-num-gpus 1.0" in out and str(f) in out.replace("'", "")
    if tomllib is not None:
        written = tomllib.loads(f.read_text(encoding="utf-8"))
        assert written["run-hook"] == HOOK and written["experiment-id"] == "exp4-fedprox-eps50"
        assert written["manifest-root"] == "/data/manifests" and written["target-epsilon"] == 50.0
    rc, _o, err = _cli(["4", *base], clear_env=("DR_CLIENT_CPUS",))
    assert rc == 2 and "CPU" in err
    rc, _o, err = _cli(
    ["4", "--checkpoint-root", str(ck), "--client-cpus", "6"],
    clear_env=("DR_MANIFEST_ROOT",)
    )
    assert rc == 2 and "manifest-root" in err
    rc, _o, err = _cli(["4", *base, "--client-cpus", "6"], patch_dir=workdir(step={**STEP, "lora-rank": "null"}))
    assert rc == 2 and "lora-rank" in err
    rc, out, _e = _cli(["2", *base, "--client-cpus", "4", "--num-rounds", "2", "--dry-run", "--diagnostic", "--show"])
    assert rc == 0 and "DRY RUN" in out and "exp2-fedavg-dryrun" in out and "num-rounds = 2" in out
    rc, _o, err = _cli(["2", *base, "--client-cpus", "4", "--num-rounds", "2"])
    assert rc == 2 and "dry run" in err
    rc, out, _e = _cli(["1", *base])
    assert rc == 0 and "client_manifest_paths" in out and "exp1-centralized" in out
    for bad in (["9"], ["4", "--client-cpus", "0"], ["4", "--num-workers", "0"], ["4", "--lora-rank", "4"]):
        try:
            _cli([*bad, *base])
            raise RuntimeError(f"accepted {bad}")
        except SystemExit as e:
            assert e.code == 2, bad


@check("K12 hygiene: loader imports only the stdlib and yaml at module level; yaml values are numbers or the expected strings")
def _():
    tree = ast.parse((REAL / "loader.py").read_text(encoding="utf-8"))
    top = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            top |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            top.add(n.module.split(".")[0])
    assert top <= {"__future__", "argparse", "json", "math", "os", "shlex", "sys", "pathlib", "typing", "yaml"}, top
    import yaml
    str_ok = {"algorithm", "accountant", "grad-sample-mode", "experiment-id", "calibration-path"}
    for f in sorted(REAL.glob("*.yaml")):
        for k, v in yaml.safe_load(f.read_text(encoding="utf-8")).items():
            assert v is None or isinstance(v, (bool, int, float)) or k in str_ok, (f.name, k, v)  # 1e-4 style strings
    assert [p.name for p in sorted(REAL.glob("exp*.yaml"))] == [f"exp{i}.yaml" for i in range(1, 9)]


print()
for n, s, msg in RESULTS:
    print(f"{s:4}  {n}" + (f"\n      {msg}" if msg else ""))
print(f"\n{sum(s == 'PASS' for _n, s, _m in RESULTS)}/{len(RESULTS)} passed, {sum(s == 'SKIP' for _n, s, _m in RESULTS)} skipped")
raise SystemExit(any(s == "FAIL" for _n, s, _m in RESULTS))