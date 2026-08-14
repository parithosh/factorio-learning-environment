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
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_tier1.*tier1_pilot_k3" >/dev/null 2>&1; do sleep 20; done
echo "== k3 block finished at $(date -u); starting codex block"
exec .venv/bin/python -m bench.run_tier1 \
  --arms A,AxK,B \
  --models codex/gpt-5.6-sol \
  --tasks iron_plate_throughput \
  --T 1500 --K 2 --m 4 \
  --template-snap snapshot-5fa7769473a710b2 \
  --node-cap 1 \
  --out bench/results/tier1_pilot_codex.json
