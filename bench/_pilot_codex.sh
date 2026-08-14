#!/usr/bin/env bash
# Secondary pilot block: codex A, A×K and B, started the moment the k3 priority
# block releases the run cap (Tier-0 measured cap = 1, so blocks must not
# overlap).
#
# The frozen Tier-0.5 config lists codex {A, B}. A×K is ADDED here, decided and
# recorded BEFORE any codex cell ran, because the pilot had ~1.5h of wall-clock
# headroom left and the decisive pre-registered contrast (B vs A×K) otherwise
# rests on a single k3 sample. Adding the control arm to a second model can only
# make that contrast harder to over-read.
#
# HANDOFF. `pgrep` alone cannot gate this: it answers "is k3 running NOW", which
# is equally false AFTER the k3 block finished and BEFORE it started, so the
# original wait loop would sail straight through a k3 launcher that was still
# warming up and run two blocks at cap 1. Three gates now, none defaulting open:
#
#   1. bench/.run_cap.lock -- an exclusive flock held for the ENTIRE run. fd 9
#      survives the exec below, so the lock is released only when python exits.
#      EVERY launcher that spends run-cap capacity must take the same lock:
#          flock bench/.run_cap.lock .venv/bin/python -m bench.run_tier1 ...
#      or the `exec 9>` idiom used at the bottom of this script. With both sides
#      locking, the lock alone serialises the blocks; the k3 launcher was an
#      ad-hoc command line that predates it, hence gates 2 and 3.
#   2. a WRITE to the k3 results file NEWER than this script's start -- proof the
#      k3 block really ran while we waited. This is what closes the "k3 has not
#      started yet" half of the pgrep race.
#   3. no live k3 process at the moment we proceed -- run_tier1 rewrites its
#      results file after every cell, so a fresh write means "alive", not "done".
#
# If the k3 block finished BEFORE this script started there can be no write
# newer than our start: re-launch with K3_DONE=1, an explicit operator
# assertion, rather than loosening a gate. Waiting past K3_WAIT_S is a hard
# failure, never a start.
set -u
cd "$(dirname "$0")/.."

LOCK="${RUN_CAP_LOCK:-bench/.run_cap.lock}"   # "none" opts out, explicitly
LOCK_WAIT_S="${LOCK_WAIT_S:-21600}"
K3_OUT="${K3_OUT:-bench/results/tier1_pilot_k3.json}"
K3_PATTERN="${K3_PATTERN:-run_tier1.*tier1_pilot_k3}"
K3_WAIT_S="${K3_WAIT_S:-14400}"
K3_DONE="${K3_DONE:-0}"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
START_EPOCH="$(date -u +%s)"

# Gate 3 then gate 2: alive means wait; done means a fresh, non-empty results
# file. K3_DONE=1 asserts gate 2 by hand for a block that finished earlier.
k3_finished() {
  if pgrep -f "$K3_PATTERN" >/dev/null 2>&1; then
    return 1
  fi
  if [ "$K3_DONE" = 1 ]; then
    return 0
  fi
  "$PY" - "$K3_OUT" "$START_EPOCH" <<'PY'
import json, os, sys

path, since = sys.argv[1], float(sys.argv[2])
try:
    if os.stat(path).st_mtime < since:
        sys.exit(1)          # nothing written since we started waiting
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if (payload.get("runs") or payload.get("failures")) else 1)
PY
}

waited=0
until k3_finished; do
  if [ "$waited" -ge "$K3_WAIT_S" ]; then
    echo "== ABORT after ${waited}s: no evidence the k3 block finished." >&2
    echo "   Needs a write to ${K3_OUT} newer than this script's start AND no" >&2
    echo "   process matching '${K3_PATTERN}'. If the k3 block completed before" >&2
    echo "   this launcher started, re-run with K3_DONE=1." >&2
    exit 3
  fi
  sleep 20
  waited=$((waited + 20))
done
echo "== k3 block finished at $(date -u) (waited ${waited}s); taking the run-cap lock"

if [ "$LOCK" != none ]; then
  if ! command -v flock >/dev/null 2>&1; then
    echo "== ABORT: flock(1) not found; install util-linux or set RUN_CAP_LOCK=none" >&2
    exit 4
  fi
  exec 9>"$LOCK"
  if ! flock -w "$LOCK_WAIT_S" 9; then
    echo "== ABORT: another run held ${LOCK} for ${LOCK_WAIT_S}s; refusing to overlap" >&2
    exit 5
  fi
fi

echo "== starting codex block at $(date -u)"
# fd 9 carries the lock across this exec and drops it when python exits, so the
# cap is held for the whole block rather than just for this shell.
exec "$PY" -m bench.run_tier1 \
  --arms A,AxK,B \
  --models codex/gpt-5.6-sol \
  --tasks iron_plate_throughput \
  --T 1500 --K 2 --m 4 \
  --template-snap snapshot-5fa7769473a710b2 \
  --node-cap 1 \
  --out bench/results/tier1_pilot_codex.json
