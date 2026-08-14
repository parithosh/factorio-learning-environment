#!/usr/bin/env python3
"""FLE Bridge Service - Persistent HTTP daemon inside the Inspect sandbox container.

Maintains FactorioInstance and FactorioGymEnv state. Serves the same handler on two
listeners:

  * a Unix domain socket at /tmp/fle_bridge.sock (used by bridge_client.py, which the
    host-side solver invokes through sandbox().exec()), and
  * TCP 0.0.0.0:8730 (used by the fan-out benchmark harness, which reaches the sandbox
    through a published/exposed TCP route instead of the exec gateway).

Both listeners are created only after the environment is fully initialised, so a
successful GET /health implies a ready environment.

Set FLE_BENCH_MODE=1 to run in benchmark-harness mode: per-step task verification
(which mutates the world and terminates the episode on quota) and the per-step
GameState capture are disabled; scoring is done out-of-band via /probe.
"""

import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("fle_bridge")

# --- HTTP servers ---

# Factorio always simulates 60 ticks per in-game second; game.speed only changes how
# many of those ticks are simulated per wall-clock second.
TICKS_PER_INGAME_SECOND = 60
PROBE_WINDOW_SECONDS = 60
PROBE_WINDOW_TICKS = PROBE_WINDOW_SECONDS * TICKS_PER_INGAME_SECOND

DEFAULT_TCP_PORT = 8730
SOCK_PATH = "/tmp/fle_bridge.sock"


class UnixHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self):
        if os.path.exists(self.server_address):
            os.unlink(self.server_address)
        super().server_bind()
        os.chmod(self.server_address, 0o666)

    def get_request(self):
        request, client_address = super().get_request()
        return request, ("127.0.0.1", 0)


class TcpHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# --- Globals set during init ---

_gym_env = None
_instance = None
_task = None
_bench_mode = False
_game_states = []  # Rolling list for error recovery

# The game instance is a single RCON connection driving a single Factorio process, so
# requests are serialised. /health is deliberately exempt: it touches no game state and
# must stay answerable while a long request (e.g. a 60s-window /probe) is in flight.
_REQ_LOCK = threading.Lock()
_LOCK_FREE_PATHS = frozenset({"/health"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _wait_for_rcon(host="localhost", port=27015, timeout=180):
    """Block until Factorio RCON is reachable."""
    logger.info("Waiting for Factorio RCON on %s:%d ...", host, port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.close()
            logger.info("RCON is reachable.")
            return True
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"Factorio RCON not available after {timeout}s")


def _init_environment():
    """Create FactorioInstance + FactorioGymEnv from environment variables."""
    global _gym_env, _instance, _task, _bench_mode

    env_id = os.environ.get("FLE_ENV_ID", "iron_ore_throughput")
    scenario = os.environ.get("FLE_SCENARIO", "default_lab_scenario")
    num_agents = int(os.environ.get("FLE_NUM_AGENTS", "1"))
    _bench_mode = _env_flag("FLE_BENCH_MODE")

    # For unbounded/open-play tasks, the gym environment is always "open_play"
    # which uses DefaultTask. The env_id (e.g. "open_play_production") is just
    # for task identification in the Inspect eval set.
    from fle.eval.tasks.task_definitions.unbounded.unbounded_tasks import (
        UNBOUNDED_PRODUCTION_TASKS,
    )

    if env_id in UNBOUNDED_PRODUCTION_TASKS:
        task_key = "open_play"
    else:
        task_key = env_id

    logger.info(
        "Initialising environment: env_id=%s, task_key=%s, scenario=%s, num_agents=%d, bench_mode=%s",
        env_id,
        task_key,
        scenario,
        num_agents,
        _bench_mode,
    )

    _wait_for_rcon()

    from fle.env import FactorioInstance
    from fle.env.gym_env.environment import FactorioGymEnv
    from fle.eval.tasks import TaskFactory

    # Build the task FIRST: it carries the research configuration for the instance.
    # Lab-play throughput tasks set all_technology_reserached=True, and building the
    # instance with all_technologies_researched=False would leave research-gated
    # recipes locked for the whole episode.
    _task = TaskFactory.create_task(task_key)
    all_tech = bool(getattr(_task, "all_technology_reserached", False))

    _instance = FactorioInstance(
        address="localhost",
        tcp_port=27015,
        fast=True,
        cache_scripts=True,
        inventory={},
        all_technologies_researched=all_tech,
        num_agents=num_agents,
    )
    _instance.set_speed_and_unpause(10)

    # task.setup() applies the starting inventory, resets with the task's research
    # setting, provisions the task world and captures task.starting_game_state.
    _task.setup(_instance)

    _gym_env = FactorioGymEnv(
        instance=_instance,
        task=_task,
        # Vision renders a full map PNG on every observation; the bridge strips
        # map_image from /observe anyway and exposes /screenshot separately.
        enable_vision=not _bench_mode,
        bench_mode=_bench_mode,
    )
    # Deliberately NOT calling _gym_env.reset(): that resets the instance to a blank
    # state and discards everything task.setup() just established. The instance is
    # already exactly at task.starting_game_state, so only the gym bookkeeping that
    # reset() would have refreshed is initialised here.
    _gym_env.initial_score, _ = _instance.namespaces[0].score()
    _gym_env.last_observation = None

    logger.info(
        "Environment ready (env_id=%s, all_technologies_researched=%s, bench_mode=%s)",
        env_id,
        all_tech,
        _bench_mode,
    )


# --- Game helpers ---


def _rcon_int(command, default=0):
    """Run an RCON command that rcon.print()s a single integer."""
    response = _instance.rcon_client.send_command(command)
    try:
        return int(str(response).strip())
    except (TypeError, ValueError):
        logger.warning("Non-integer RCON response for %r: %r", command, response)
        return default


def _game_tick():
    """The real Factorio tick counter (advances only while the game is unpaused)."""
    return _rcon_int("/sc rcon.print(game.tick)")


def _entity_count():
    """Player-force entities on surface 1, excluding characters."""
    return _rcon_int(
        "/sc local s = game.surfaces[1]; local n = 0; "
        'for _, e in pairs(s.find_entities_filtered{force = "player"}) do '
        'if e.type ~= "character" then n = n + 1 end end; rcon.print(n)'
    )


def _factorio_pid():
    """PID of the Factorio server process in this container (0 if not found)."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        for arg in argv:
            if arg.endswith(b"bin/x64/factorio"):
                return int(entry)
    return 0


def _production_count(entity):
    """Cumulative produced count of `entity` for the player force."""
    from fle.commons.models.achievements import ProductionFlows

    stats = _instance.namespaces[0]._get_production_stats()
    return float(ProductionFlows.from_dict(stats).output.get(entity, 0) or 0)


def _run_probe(entity):
    """One fixed 60-in-game-second production window. No plateau loop, no mutation
    beyond letting the simulation run.

    Robust at any game speed: the window is defined in ticks and the loop polls the
    real game tick, so a UPS shortfall stretches wall time instead of shortening the
    measured window.
    """
    speed = float(_instance.get_speed() or 1.0)
    was_paused = _instance.game_control.is_paused()
    if was_paused:
        _instance.unpause()

    start_tick = _game_tick()
    start_count = _production_count(entity)
    target_tick = start_tick + PROBE_WINDOW_TICKS

    t0 = time.monotonic()
    # Generous guard: 5x the nominal window, at least 30s, so a stalled/paused game
    # cannot wedge the request forever.
    nominal = PROBE_WINDOW_TICKS / TICKS_PER_INGAME_SECOND / speed
    deadline = t0 + max(30.0, nominal * 5.0)

    tick = start_tick
    timed_out = False
    while tick < target_tick:
        remaining_s = (target_tick - tick) / TICKS_PER_INGAME_SECOND / speed
        time.sleep(min(max(remaining_s * 0.9, 0.02), 2.0))
        tick = _rcon_int("/sc rcon.print(game.tick)", tick)
        if time.monotonic() > deadline:
            timed_out = True
            break

    wall_s = time.monotonic() - t0
    end_tick = tick
    end_count = _production_count(entity)

    if was_paused:
        _instance.pause()

    elapsed_ticks = end_tick - start_tick
    if elapsed_ticks > 0:
        # Normalise to exactly one 60-in-game-second window.
        throughput = (end_count - start_count) * PROBE_WINDOW_TICKS / elapsed_ticks
    else:
        throughput = 0.0

    return {
        "entity": entity,
        "throughput": float(throughput),
        "wall_s": float(wall_s),
        "start_tick": int(start_tick),
        "end_tick": int(end_tick),
        "window_ticks": int(PROBE_WINDOW_TICKS),
        "speed": speed,
        "start_count": start_count,
        "end_count": end_count,
        "timed_out": timed_out,
    }


# --- Request handler ---


class BridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the bridge service."""

    protocol_version = "HTTP/1.1"

    # Suppress per-request logs to stderr
    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send_json(self, data, status=200):
        body = json.dumps(data, cls=_NumpyEncoder).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _dispatch(self, routes, verb):
        path = self.path.split("?", 1)[0]
        handler = routes.get(path)
        if handler is None:
            self._send_json({"error": f"Unknown {verb} path: {self.path}"}, 404)
            return
        try:
            if path in _LOCK_FREE_PATHS:
                handler(self)
            else:
                with _REQ_LOCK:
                    handler(self)
        except Exception as exc:
            logger.error("%s %s error: %s", verb, self.path, exc, exc_info=True)
            self._send_json(
                {"error": str(exc), "traceback": traceback.format_exc()}, 500
            )

    # ----- GET routes -----

    def do_GET(self):
        self._dispatch(
            {
                "/health": BridgeHandler._handle_health,
                "/observe": BridgeHandler._handle_observe,
                "/score": BridgeHandler._handle_score,
                "/system-prompt": BridgeHandler._handle_system_prompt,
                "/game-state": BridgeHandler._handle_game_state,
                "/state-save": BridgeHandler._handle_state_save,
                "/meta": BridgeHandler._handle_meta,
            },
            "GET",
        )

    # ----- POST routes -----

    def do_POST(self):
        self._dispatch(
            {
                "/execute": BridgeHandler._handle_execute,
                "/reset": BridgeHandler._handle_reset,
                "/screenshot": BridgeHandler._handle_screenshot,
                "/probe": BridgeHandler._handle_probe,
                "/state-restore": BridgeHandler._handle_state_restore,
            },
            "POST",
        )

    # ----- Handler implementations -----

    def _handle_health(self):
        ready = _gym_env is not None
        self._send_json({"status": "ok" if ready else "initialising"})

    def _handle_meta(self):
        self._send_json(
            {
                "factorio_pid": _factorio_pid(),
                "elapsed_ticks": int(_instance.get_elapsed_ticks()),
                "game_tick": _game_tick(),
                "entity_count": _entity_count(),
                "speed": float(_instance.get_speed()),
                "paused": bool(_instance.game_control.is_paused()),
                "bench_mode": _bench_mode,
                "task_key": getattr(_task, "task_key", None),
                "all_technologies_researched": bool(
                    getattr(_instance, "all_technologies_researched", False)
                ),
            }
        )

    def _handle_observe(self):
        try:
            obs = _gym_env.get_observation()
        except Exception as e:
            # If vision rendering fails (e.g. no sprites), retry with vision disabled
            if _gym_env.enable_vision:
                logger.warning(
                    "get_observation() failed with vision enabled (%s), retrying without vision",
                    e,
                )
                _gym_env.enable_vision = False
                obs = _gym_env.get_observation()
            else:
                raise
        obs_dict = obs.to_dict()
        # Strip bulky map_image from observation to reduce payload;
        # the solver doesn't use it through the bridge (uses /screenshot instead).
        obs_dict.pop("map_image", None)
        # Fix Observation.to_dict() quirks that break from_dict() round-trip:
        # - "progress" and "current_research" can be the string "None" instead of
        #   a proper null/empty value, causing from_dict() to iterate over characters.
        research = obs_dict.get("research", {})
        if isinstance(research, dict):
            if research.get("progress") == "None" or not isinstance(
                research.get("progress"), list
            ):
                research["progress"] = []
            if research.get("current_research") == "None":
                research["current_research"] = None
        self._send_json(obs_dict)

    def _handle_score(self):
        score, automated = _instance.namespaces[0].score()
        self._send_json(
            {
                "production_score": score,
                "automated_production_score": automated or 0,
            }
        )

    def _handle_system_prompt(self):
        import importlib.resources
        from fle.env.utils.controller_loader.system_prompt_generator import (
            SystemPromptGenerator,
        )

        generator = SystemPromptGenerator(str(importlib.resources.files("fle") / "env"))
        prompt = generator.generate_for_agent(agent_idx=0, num_agents=1)
        self._send_json({"system_prompt": prompt})

    def _handle_game_state(self):
        from fle.commons.models.game_state import GameState

        gs = GameState.from_instance(_instance)
        self._send_json({"game_state": gs.to_raw()})

    def _handle_state_save(self):
        from fle.commons.models.game_state import GameState

        gs = GameState.from_instance(_instance)
        self._send_json({"state": gs.to_raw()})

    def _handle_state_restore(self):
        body = self._read_body()
        raw = body.get("state")
        if not raw:
            self._send_json({"error": "missing 'state'"}, 400)
            return

        from fle.commons.models.game_state import GameState

        _instance.reset(GameState.parse_raw(raw))
        _game_states.clear()
        _gym_env.initial_score, _ = _instance.namespaces[0].score()
        _gym_env.last_observation = None
        self._send_json({"ok": True})

    def _handle_probe(self):
        body = self._read_body()
        entity = body.get("entity") or getattr(_task, "throughput_entity", None)
        if not entity:
            self._send_json({"error": "missing 'entity' and task has no default"}, 400)
            return
        self._send_json(_run_probe(str(entity)))

    def _handle_execute(self):
        body = self._read_body()
        code = body.get("code", "")
        agent_idx = body.get("agent_idx", 0)

        from fle.env.gym_env.action import Action
        from fle.env.gym_env.observation_formatter import TreeObservationFormatter

        action = Action(agent_idx=agent_idx, code=code)

        obs, reward, terminated, truncated, info = _gym_env.step(action)
        _gym_env.background_step()

        # Capture game state for rollback (skipped in bench mode: the harness snapshots
        # and forks the whole VM instead, and GameState capture is O(n^2) in entities)
        output_game_state = info.get("output_game_state")
        game_state_raw = output_game_state.to_raw() if output_game_state else None
        if output_game_state is not None:
            _game_states.append(output_game_state)
            # Keep last 5 states
            if len(_game_states) > 5:
                _game_states.pop(0)

        # Format flows
        flow = obs.get("flows", {})
        flows_formatted = TreeObservationFormatter.format_flows_compact(flow)

        production_score = info.get("production_score", 0)
        automated_score = info.get("automated_production_score", 0)
        error_occurred = info.get("error_occurred", False)

        result = {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "result": info.get("result", ""),
            "production_score": production_score,
            "automated_production_score": automated_score,
            # Bridge HTTP API v1 aliases
            "automated_score": automated_score,
            "error": error_occurred,
            "policy_execution_time": info.get("policy_execution_time", 0),
            "error_occurred": error_occurred,
            "flows": flow,
            "flows_formatted": flows_formatted,
            "score": obs.get("score", 0),
            "ticks": info.get("ticks", 0),
            "game_state_raw": game_state_raw,
        }

        self._send_json(result)

    def _handle_reset(self):
        body = self._read_body()
        game_state_raw = body.get("game_state", None)

        if game_state_raw:
            from fle.commons.models.game_state import GameState

            gs = GameState.parse_raw(game_state_raw)
            _gym_env.reset({"game_state": gs})
        else:
            # Reset to the task's starting state rather than a blank instance, so the
            # task's research/inventory/provisioning survives the reset.
            starting = getattr(_task, "starting_game_state", None)
            _gym_env.reset({"game_state": starting} if starting else None)

        _game_states.clear()
        self._send_json({"status": "ok"})

    def _handle_screenshot(self):
        namespace = _instance.namespaces[0]
        result = namespace._render(radius=64, max_render_radius=32, include_status=True)
        base64_data = result.to_base64()
        # Write to file for sandbox().read_file()
        with open("/tmp/screenshot.png", "wb") as f:
            import base64

            f.write(base64.b64decode(base64_data))
        self._send_json(
            {
                "path": "/tmp/screenshot.png",
                "base64": f"data:image/png;base64,{base64_data}",
            }
        )


# --- Main ---


def main():
    tcp_port = int(os.environ.get("FLE_BRIDGE_PORT", str(DEFAULT_TCP_PORT)))

    logger.info("Starting FLE Bridge Service on %s and TCP :%d", SOCK_PATH, tcp_port)

    # Initialise the environment (blocks until RCON is available). Both listeners are
    # created afterwards, so /health answering at all implies a ready environment.
    try:
        _init_environment()
    except Exception:
        logger.error("Failed to initialise environment:\n%s", traceback.format_exc())
        sys.exit(1)

    uds_server = UnixHTTPServer(SOCK_PATH, BridgeHandler)
    tcp_server = TcpHTTPServer(("0.0.0.0", tcp_port), BridgeHandler)

    tcp_thread = threading.Thread(
        target=tcp_server.serve_forever, name="bridge-tcp", daemon=True
    )
    tcp_thread.start()
    logger.info("Bridge service listening on %s and 0.0.0.0:%d", SOCK_PATH, tcp_port)

    try:
        uds_server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge service.")
    finally:
        tcp_server.shutdown()
        tcp_server.server_close()
        uds_server.server_close()
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)


if __name__ == "__main__":
    main()
