# FLE sandbox bridge — P1-P3 fixes + Bridge HTTP API v1 (verification note)

Image: `fle-sandbox:bench` (built from `fle/eval/inspect/sandbox/Dockerfile`).
Smoke container left running for downstream agents:

```
docker run -d --name flebench-smoke --cpus 2 --memory 4g \
  -e FLE_BENCH_MODE=1 -e FLE_ENV_ID=iron_ore_throughput \
  -p 8730:8730 fle-sandbox:bench
```

Readiness: both listeners are created only **after** full environment init, so a
successful `GET /health` implies a ready environment (connection-refused == not ready).
Measured init-to-`/health`-ok: **~5 s** after container start on this host.

Compressed image for transfer: `/tmp/flebench-image.zst` (401 MiB,
`docker save fle-sandbox:bench | zstd -T0 -3`).

## What changed

| # | Fix | Files |
|---|-----|-------|
| P1 | Task is constructed **before** the instance, and `FactorioInstance(all_technologies_researched=...)` now takes the task's setting. The post-`task.setup()` `_gym_env.reset()` — which reset the instance to a blank state and discarded `task.starting_game_state` — is gone; only the gym bookkeeping it refreshed (`initial_score`, `last_observation`) is initialised. `/reset` with no body now resets to `task.starting_game_state` instead of a blank instance. | `bridge_service.py` |
| P2 | The same `BridgeHandler` is served on **TCP 0.0.0.0:8730** (`FLE_BRIDGE_PORT`) in addition to `/tmp/fle_bridge.sock`. Both are `ThreadingHTTPServer`; TCP runs on a daemon thread, UDS on the main thread. | `bridge_service.py`, `Dockerfile` (`EXPOSE 8730`), `supervisord.conf` |
| P3 | `FLE_BENCH_MODE=1` → `FactorioGymEnv(bench_mode=True)`: per-step `task.verify()` skipped (it sleeps through repeated 60 s windows, mutating the world, and flips `terminated` on quota), `terminated` never set from task success, per-step `GameState.from_instance()` skipped, and map rendering disabled (`enable_vision=False`; `/screenshot` is unaffected). | `environment.py`, `bridge_service.py` |
| — | Dependency fix required to rebuild at all: `a2a-sdk` was unpinned and 1.x dropped `a2a.types.TextPart`, which `fle/env/tools/agent/send_message/client.py` imports at tool-load time. Pinned `a2a-sdk<1.0`. | `pyproject.toml` |

## API v1 surface

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status":"ok"}`. **Only lock-free endpoint** — answerable during a 6 s probe. |
| `POST /execute {"code"}` | Gym-step semantics. Adds v1 aliases `automated_score` and `error` next to the legacy `automated_production_score` / `error_occurred`. `game_state_raw` is `null` in bench mode. |
| `POST /probe {"entity"}` | ONE fixed 3600-tick (= 60 in-game s) window; no plateau loop. `{throughput, wall_s, start_tick, end_tick, window_ticks, speed, start_count, end_count, timed_out}`. `entity` defaults to the task's `throughput_entity`. |
| `GET /state-save` | `{"state": GameState.to_raw()}`. |
| `POST /state-restore {"state"}` | `instance.reset(GameState.parse_raw(state))` → `{"ok":true}`. |
| `GET /meta` | `{factorio_pid, elapsed_ticks, game_tick, entity_count, speed, paused, bench_mode, task_key, all_technologies_researched}`. |
| unchanged | `/observe` `/score` `/system-prompt` `/game-state` `/reset` `/screenshot`. |

Concurrency model: one RCON connection drives one Factorio process, so every endpoint
except `/health` serialises behind a single global lock. A `/meta` issued 1 s into a
probe was measured waiting **5.02 s**; `/health` answered in **1.0 ms** during the same
probe. Do not expect concurrent `/execute` + `/probe` on one sandbox to overlap.

Probe implementation is tick-based, not sleep-based: it reads the real `game.tick`,
targets `start_tick + 3600`, polls with a geometric backoff, and normalises the
production delta to exactly 3600 ticks. A UPS shortfall therefore stretches `wall_s`
instead of shortening the measured window. It unpauses the game if `pause_after_action`
left it paused and restores the pause state afterwards; it performs no other mutation
(in particular it does **not** touch FLE's virtual `storage.elapsed_ticks`).

## Measurements (host: Ryzen 9 9950X3D, 2 vCPU / 4 GiB container, game speed 10)

Acceptance checks (a)-(f):

- **(a) `/health`** → `{"status": "ok"}`, 2 ms.
- **(b) `/meta`** → `{"factorio_pid": 8, "elapsed_ticks": 0, "game_tick": 60959, "entity_count": 0, "speed": 10.0, "paused": false, "bench_mode": true, "task_key": "iron_ore_throughput", "all_technologies_researched": true}`.
- **(c) research gate** — `/execute`:
  ```
  automation-2: researched=True
  automation-3: researched=True
  electronics:  researched=True
  AM2 placed at Position(x=0.5, y=4.5) recipe = processing-unit
  ```
  Placing an `assembling-machine-2` and setting the deeply research-gated
  `processing-unit` recipe both succeed. **Negative control** on the pre-fix image
  (`fle-sandbox:latest`, same task): `automation-2: researched=False`,
  `automation-3: researched=False` — i.e. P1 was real and is fixed.
- **(d) trivial `/execute`** — `print('hello from bench mode')` → `0.21 s`,
  `error=false`, `terminated=false`, `game_state_raw=null`, `ticks=0`.
  **Negative control** (pre-fix image, no bench mode): the identical program took
  `6.70 s` and returned a 67 876-byte `game_state_raw` — that 6.5 s is one full 60 s
  in-game verify window, and it advanced the world on every step.
- **(e) `/probe {"entity":"iron-ore"}`** — on an empty map: `throughput 0.0`,
  `wall_s 6.0015`, ticks `65619 → 69221`. After building one coal-fed burner mining
  drill + output chest: **`throughput 20.988`, `wall_s 6.0016 s`, 3602 ticks**
  (`start_count 3.0 → end_count 24.0`).
- **(f) `/state-save` / `/state-restore`** — save: 73 569 bytes in `0.08 s`
  (keys: `agent_messages, entities, inventories, namespaces, research, timestamp`).
  Removed the drill → probe dropped to `0.0`; restored the saved state → `entity_count`
  back to 2 and probe returned exactly `20.988` again. Restore: `0.03 s`.

### `/probe` wall_s and deviation

Nominal window = 3600 ticks / (60 ticks/s × speed 10) = **6.000 s**.
Measured `wall_s` = **6.0015-6.0016 s** across 6 probes; the tick window closed at
**3602 ticks** (+2 ticks, +0.056 %) every time. The overshoot is one RCON poll interval
at the tail of the backoff loop; throughput is normalised by the *actual* tick delta, so
it does not bias the number. Client-observed round trip was 6.011-6.012 s (≈10 ms of
HTTP + RCON overhead).

Repeatability on an identical steady factory (validates the P3 fixed-window scorer):
three consecutive probes returned **20.98833981121599** each, `wall_s` 6.0016 s,
3602 ticks — zero variance.

### Regressions checked

- UDS path (`bridge_client.py` via `docker exec`) still works on the bench image:
  `health` → ok, `execute` → 0.25 s.
- **Non-bench mode** (`fle-sandbox:bench` with `FLE_BENCH_MODE` unset) behaves like
  upstream: trivial `/execute` takes 6.77 s (verify runs) and returns a 67 690-byte
  `game_state_raw` — and now also reports `automation-2 researched=True`, so the P1 fix
  benefits the normal Inspect eval path too.
- Unknown paths → 404 JSON. `/observe` (0.18 s, `map_image` stripped), `/score`,
  `/system-prompt` (116 827 chars) all intact.

### Caveats for the harness

1. `/meta.elapsed_ticks` is FLE's *virtual* counter (`storage.elapsed_ticks`), advanced
   only by the `sleep()` tool. In bench mode nothing sleeps, so it stays `0`. Use
   **`/meta.game_tick`** for real Factorio time in fork-fidelity / parity checks.
   `/execute`'s `ticks` field has the same virtual semantics (kept for gym-step
   compatibility).
2. `production_score` from `namespace.score()` is cumulative and can start negative;
   record a per-branch baseline right after fork/restore (design doc P5).
3. `/state-restore` runs `instance.reset()`, which restores speed 10 / unpaused
   regardless of the pre-restore pause state.

## Test fixtures for Tier-0 fork fidelity

`bench/fixtures/iron_ore_270_entities.state.json` — a **270-entity** iron-ore factory
captured with `/state-save` (72 290 bytes). Load it into any freshly-booted
`iron_ore_throughput` sandbox with `POST /state-restore {"state": <file contents>}`:

- restore takes **0.04 s**, `/meta.entity_count` comes back as exactly **270**;
- `/probe {"entity":"iron-ore"}` on it returns **188.895** ore per 60 in-game seconds,
  identical across probes (4 consecutive probes: 188.895 / 188.895 / 188.895, `wall_s`
  6.0015-6.0016 s) — it does **not** saturate, so it is safe for a whole probe series;
- fuel budget is ~1200 in-game seconds (~20 probe windows) before the drills starve.

Composition: 10 burner mining drills on the bottom two rows of the nearest iron-ore
patch, each dropping directly into its own wooden chest (800-item sink, which is why
throughput is flat), plus 250 medium electric poles as inert 1×1 ballast placed **on**
the ore patch — resource tiles carry no trees, so ballast placement never fails.

The generating program is in `bench/fixtures/build_scaling_factory.py` (an `/execute`
payload, not a host script). Build cost measured at 9.0 s of wall clock. Tune
`N_BALLAST` for a different `entity_count`; the drill count is capped by the 10 wooden
chests in `LAB_PLAY_POPULATED_STARTING_INVENTORY`.

FLE API gotchas found while writing it (all cost a failed build first):

- **Reach matters.** `place_entity` and `insert_item` both need the character within a
  few tiles; `move_to` before every placement (or at least every ~5 tiles along a line).
- **`entity.drop_position` goes stale** after `insert_item` and will hand you the
  *previous* drill's tile. Compute the drop tile: a 2×2 drill at integer origin `(X, Y)`
  covers tiles `(X-1..X, Y-1..Y)` and, facing UP, drops onto tile `(X-1, Y-2)`.
- **Trees block long belt runs.** A 240-tile belt line off the patch lost 40 tiles to
  trees, and `harvest_resource` refused with "Nothing within reach". Staying on resource
  tiles avoids the problem entirely.
- **Belt buffers saturate.** An earlier 12-drill / 200-belt build probed 52.97 once and
  then **0.000** forever (`WAITING_FOR_SPACE_IN_DESTINATION`) — 8 items/tile is nothing
  against ~21 ore/drill/window. Chests (800 items) are the right sink.
- A raw `place_entity` failure reports `entity already exists at the target position`
  for rocks as well as for your own entities; wrap per-unit placement in `try`/`except`
  and skip, rather than aborting the build.

Also worth recording for the design doc's C-arm cost model: `GameState.from_instance`
on this 270-entity factory took **0.20 s** and produced 72 290 bytes — consistent with
the greenfield-scale estimate (v2.2), i.e. capture cost is not a Tier-1 claim.
