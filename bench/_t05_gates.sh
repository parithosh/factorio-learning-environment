#!/usr/bin/env bash
# Tier 0.5 gate track: diversity + 3-step latency for the two models that the
# smoke run did not cover, against the bake bridge (direct URL). Sequential and
# with a /reset between models, because one bridge serialises every endpoint
# except /health and the second model would otherwise inherit the first's
# factory (and its exec latency).
#
# Fails closed: a /reset that does not answer 2xx, or a gate run that exits
# nonzero, is a gate whose numbers cannot be trusted. The script keeps going
# through the remaining models (so one bad model does not cost the other its
# measurement) but exits nonzero, because the merge must not treat a missing or
# half-measured track as evidence.
set -euo pipefail
BAKE="https://8730--sandbox-c73901e65ee3b027.compute.ethpandaops.io"
cd "$(dirname "$0")/.."

failed=()

for pair in "kimi-for-coding:kfc" "codex/gpt-5.6-sol:codex"; do
  model="${pair%%:*}"; tag="${pair##*:}"
  echo "== reset bridge before ${model}"
  # --fail-with-body keeps the bridge's error body while still failing on a
  # non-2xx status; curl < 7.76 only has --fail (status, no body).
  if curl --fail-with-body --help >/dev/null 2>&1; then
    fail_flag=(--fail-with-body)
  else
    fail_flag=(--fail)
  fi
  if ! curl -sS "${fail_flag[@]}" -m 120 -X POST "${BAKE}/reset" \
       -H 'content-type: application/json' -d '{}'; then
    echo
    echo "== FAILED reset for ${model}: the factory it would measure is the"
    echo "   previous model's, so the gate is not run"
    failed+=("${model} (reset)")
    continue
  fi
  echo
  echo "== gates ${model}"
  # The gate run owns its own exit status; do not let -e abort the loop before
  # the second model gets its measurement.
  rc=0
  .venv/bin/python -m bench.tier05 \
    --models "${model}" \
    --phases latency,diversity \
    --live-url "${BAKE}" \
    --latency-steps 3 \
    --out "bench/results/tier05_gate_${tag}.json" \
    --md "bench/results/TIER05_GATE_${tag}.md" || rc=$?
  echo "== exit ${rc} for ${model}"
  if [[ "${rc}" -ne 0 ]]; then
    failed+=("${model} (bench.tier05 exit ${rc})")
  fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "GATES FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "GATES DONE"
