#!/usr/bin/env bash
# pod/run_smoke_suites.sh
#
# Runs the approved training smoke suites STRICTLY ONE AT A TIME on the pod.
# Each suite gets its own log (/data/logs/smoke/<name>.log) and a 30 min timeout.
# No parallelism, no pip, no edits to any project file.
#
# Usage:
#   bash pod/run_smoke_suites.sh              # run all approved suites in order
#   bash pod/run_smoke_suites.sh --only NAME  # run a single approved suite
#
# A suite is OK only if its exit code is 0. Exit status of this script is 1 if
# any suite failed / timed out / was missing, or if the precondition check fails.

set -u -o pipefail

# ---------------------------------------------------------------- config ----
LOG_DIR="${SMOKE_LOG_DIR:-/data/logs/smoke}"
TIMEOUT_SECS=1800            # 30 minutes per suite
KILL_GRACE_SECS=30           # SIGKILL this long after SIGTERM if still alive
SUITE_DIR="training-smoketests"

# Deliberately NOT included:
# smoke_server.py, smoke_server_final.py, diag_*.py, archived_destructive/*
SUITES=(
  smoke_aug_seeding
  smoke_train_centralized
  smoke_coatnet_lora
  smoke_local_trainer
  smoke_local_trainer_m4
  smoke_checkpoint
  smoke_class_weights
  smoke_dp_checks
  smoke_privacy_calibration
  smoke_client_dp_preflight
  smoke_client
  smoke_serverV2
  smoke_configs
  smoke_exp_tracker
  smoke_runner
  smoke_evaluation
  smoke_stepzero
  smoke_stepzerov2
)

# Repo root = parent of the directory this script lives in (pod/..)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ------------------------------------------------------------------ args ----
ONLY=""
usage() {
  echo "Usage: $0 [--only NAME]"
  echo "Approved suites (run order):"
  printf '  %s\n' "${SUITES[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || { echo "ERROR: --only needs a suite name" >&2; usage >&2; exit 2; }
      ONLY="${2%.py}"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$ONLY" ]]; then
  approved=0
  for s in "${SUITES[@]}"; do [[ "$s" == "$ONLY" ]] && approved=1; done
  if [[ $approved -eq 0 ]]; then
    echo "ERROR: '$ONLY' is not an approved suite; refusing to run it." >&2
    usage >&2
    exit 2
  fi
  SUITES=("$ONLY")
fi

# ---------------------------------------------------------- precondition ----
# Refuse to start if a Flower run is active or run_queue.py / run_experiment1.py
# is running. If we cannot determine the flwr state, we also refuse (fail closed).
check_preconditions() {
  local rc=0 out err_file
  err_file="$(mktemp)"

  # 1) Active Flower run?
  if ! command -v flwr >/dev/null 2>&1; then
    echo "REFUSING TO START: 'flwr' not found on PATH; cannot verify no run is active." >&2
    rm -f "$err_file"; return 1
  fi
  out="$(flwr ls --format json 2>"$err_file")"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "REFUSING TO START: 'flwr ls --format json' failed (exit $rc); cannot verify no run is active." >&2
    sed 's/^/  flwr: /' "$err_file" >&2
    rm -f "$err_file"; return 1
  fi
  rm -f "$err_file"

  # Any run whose status does not start with "finished" is treated as active
  # (pending / starting / running).
  local verdict
  verdict="$(FLWR_JSON="$out" python - <<'PY'
import json, os, sys
raw = os.environ.get("FLWR_JSON", "")
starts = [i for i in (raw.find("{"), raw.find("[")) if i != -1]
if not starts:
    print("UNPARSEABLE"); sys.exit(0)
try:
    data = json.loads(raw[min(starts):])
except Exception:
    print("UNPARSEABLE"); sys.exit(0)

active = []
def walk(o):
    if isinstance(o, dict):
        st = o.get("status")
        if isinstance(st, str) and not st.lower().startswith("finished"):
            active.append("%s (%s)" % (o.get("run-id", o.get("run_id", "?")), st))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(data)
print("ACTIVE: " + ", ".join(active) if active else "IDLE")
PY
)"
  case "$verdict" in
    IDLE) ;;
    ACTIVE*)
      echo "REFUSING TO START: active Flower run(s) detected -> ${verdict#ACTIVE: }" >&2
      return 1 ;;
    *)
      echo "REFUSING TO START: could not parse 'flwr ls --format json' output; cannot verify no run is active." >&2
      return 1 ;;
  esac

  # 2) Queue / experiment processes?
  local procs
  procs="$(pgrep -af 'run_queue\.py|run_experiment1\.py' || true)"
  if [[ -n "$procs" ]]; then
    echo "REFUSING TO START: queue/experiment process is running:" >&2
    printf '  %s\n' "$procs" >&2
    return 1
  fi
  return 0
}

# ------------------------------------------------------------------ run -----
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; exit 2; }

# Guard against two copies of this script racing each other.
mkdir -p "$LOG_DIR" || { echo "ERROR: cannot create $LOG_DIR" >&2; exit 2; }
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOG_DIR/.run_smoke_suites.lock"
  flock -n 9 || { echo "REFUSING TO START: another run_smoke_suites.sh is already running." >&2; exit 1; }
fi

check_preconditions || exit 1

echo "Pod smoke run: ${#SUITES[@]} suite(s), one at a time, ${TIMEOUT_SECS}s timeout each"
echo "Repo: $REPO_ROOT | Logs: $LOG_DIR | $(date '+%F %T')"

R_NAME=(); R_STATUS=(); R_RC=(); R_DUR=()
any_failed=0

# Last "summary-looking" line, else last non-empty line.
last_summary_line() {
  local log="$1" line
  line="$(grep -iE 'summary|passed|failed|[0-9]+ ?/ ?[0-9]+|all .*(ok|pass)|total' "$log" 2>/dev/null | tail -n 1)"
  [[ -z "$line" ]] && line="$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 1)"
  printf '%s' "${line:-<no output>}"
}

for name in "${SUITES[@]}"; do
  script="$SUITE_DIR/$name.py"
  log="$LOG_DIR/$name.log"

  echo
  echo "=============================================================="
  echo ">>> $name"
  echo "=============================================================="

  start=$SECONDS
  if [[ ! -f "$script" ]]; then
    echo "$script not found" >"$log"
    rc=127
    status="MISSING"
  else
    PYTHONUNBUFFERED=1 timeout -k "$KILL_GRACE_SECS" "$TIMEOUT_SECS" \
      python "$script" >"$log" 2>&1 </dev/null
    rc=$?
    if   [[ $rc -eq 0 ]];                  then status="OK"
    elif [[ $rc -eq 124 || $rc -eq 137 ]]; then status="TIMEOUT"
    else                                        status="FAIL"
    fi
  fi
  dur=$(( SECONDS - start ))

  echo "name:     $name"
  echo "status:   $status"
  echo "exit:     $rc"
  echo "duration: ${dur}s"
  echo "summary:  $(last_summary_line "$log")"
  echo "log:      $log"

  fail_lines="$(grep -n 'FAIL' "$log" 2>/dev/null || true)"
  if [[ -n "$fail_lines" ]]; then
    echo "--- lines containing FAIL ($name) ---"
    printf '%s\n' "$fail_lines"
    echo "--- end FAIL lines ---"
  fi

  # Documented smoke_checkpoint N6 ordering failure: shown, never hidden,
  # never auto-forgiven. It still counts as a failed suite.
  if [[ "$name" == "smoke_checkpoint" && "$status" != "OK" ]] && grep -q 'N6' "$log" 2>/dev/null; then
    echo "NOTE: smoke_checkpoint output mentions N6 (documented ordering failure)."
    echo "      Shown above for your judgement; suite is still counted as NOT OK."
    echo "--- N6 lines ---"
    grep -n 'N6' "$log"
    echo "--- end N6 lines ---"
  fi

  [[ "$status" != "OK" ]] && any_failed=1
  R_NAME+=("$name"); R_STATUS+=("$status"); R_RC+=("$rc"); R_DUR+=("${dur}s")
done

# ---------------------------------------------------------------- table -----
echo
echo "=============================== RESULTS ==============================="
printf '%-30s %-8s %-6s %s\n' "SUITE" "STATUS" "EXIT" "DURATION"
printf '%-30s %-8s %-6s %s\n' "-----" "------" "----" "--------"
for i in "${!R_NAME[@]}"; do
  printf '%-30s %-8s %-6s %s\n' "${R_NAME[$i]}" "${R_STATUS[$i]}" "${R_RC[$i]}" "${R_DUR[$i]}"
done
echo "-----------------------------------------------------------------------"
n_ok=0; for s in "${R_STATUS[@]}"; do [[ "$s" == "OK" ]] && n_ok=$((n_ok+1)); done
echo "OK: $n_ok / ${#R_NAME[@]}   Logs: $LOG_DIR"

if [[ $any_failed -ne 0 ]]; then
  echo "OVERALL: FAILED"
  exit 1
fi
echo "OVERALL: ALL OK"
exit 0
