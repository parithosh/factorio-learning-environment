#!/usr/bin/env bash
# Tier 0.5 gate track: diversity + 3-step latency for the two models that the
# smoke run did not cover, against the bake bridge (direct URL). Sequential and
# with a /reset between models, because one bridge serialises every endpoint
# except /health and the second model would otherwise inherit the first's
# factory (and its exec latency).
set -u
BAKE="https://8730--sandbox-c73901e65ee3b027.compute.ethpandaops.io"
cd "$(dirname "$0")/.."

for pair in "kimi-for-coding:kfc" "codex/gpt-5.6-sol:codex"; do
  model="${pair%%:*}"; tag="${pair##*:}"
  echo "== reset bridge before ${model}"
  curl -s -m 120 -X POST "${BAKE}/reset" -H 'content-type: application/json' -d '{}'
  echo
  echo "== gates ${model}"
  .venv/bin/python -m bench.tier05 \
    --models "${model}" \
    --phases latency,diversity \
    --live-url "${BAKE}" \
    --latency-steps 3 \
    --out "bench/results/tier05_gate_${tag}.json" \
    --md "bench/results/TIER05_GATE_${tag}.md"
  echo "== exit $? for ${model}"
done
echo "GATES DONE"
