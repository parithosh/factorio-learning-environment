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

TCP authentication: set FLE_BRIDGE_TOKEN to require `Authorization: Bearer <token>`
on every TCP request. The UDS listener is exempt (filesystem permissions are its
boundary). With no token set the TCP listener is started only when
FLE_BRIDGE_ALLOW_INSECURE=1; otherwise the service warns once and serves the UDS only.
"""

import hmac
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import uuid
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

# Request bodies are read and parsed before the game-state lock is taken, so a slow or
# oversized client cannot hold the single RCON connection hostage. /state-restore and
# /reset carry a whole serialised GameState, which is legitimately large; everything
# else is a handful of fields.
MAX_BODY_BYTES_DEFAULT = 1 * 1024 * 1024
MAX_BODY_BYTES_BY_PATH = {
    "/state-restore": 64 * 1024 * 1024,
    "/reset": 64 * 1024 * 1024,
}
BODY_READ_TIMEOUT_SECONDS = 60.0


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


def _bridge_token() -> str:
    """Shared secret required on TCP requests ("" = no token configured)."""
    return os.environ.get("FLE_BRIDGE_TOKEN", "").strip()


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

    try:
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
    finally:
        # Never leave the game running because the window failed part-way through.
        if was_paused:
            try:
                _instance.pause()
            except Exception:
                logger.error("Failed to re-pause game after probe", exc_info=True)

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


class _BodyError(Exception):
    """Malformed/oversized request body, mapped to an HTTP status by _dispatch."""

    def __init__(self, status, message, close=False):
        super().__init__(message)
        self.status = status
        self.message = message
        # True when the body was not consumed, so the connection is no longer framed.
        self.close = close


class BridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the bridge service."""

    protocol_version = "HTTP/1.1"

    # Suppress per-request logs to stderr
    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send_json(self, data, status=200, close=False):
        body = json.dumps(data, cls=_NumpyEncoder).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            # send_header() also flips self.close_connection for "Connection: close".
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _is_unix_socket(self):
        return getattr(self.server, "address_family", None) == socket.AF_UNIX

    def _authorised(self):
        """UDS requests are exempt (filesystem permissions are the boundary). TCP
        requests need `Authorization: Bearer <FLE_BRIDGE_TOKEN>` whenever a token is
        configured; with no token configured no TCP listener exists unless the operator
        opted in via FLE_BRIDGE_ALLOW_INSECURE=1 (see main()).
        """
        if self._is_unix_socket():
            return True
        token = _bridge_token()
        if not token:
            return True
        scheme, _, presented = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(
            presented.strip().encode("utf-8", "replace"), token.encode("utf-8")
        )

    def _read_body(self, path):
        """Read and JSON-parse the request body.

        Runs *before* _REQ_LOCK is taken so a slow, oversized or malformed client
        cannot hold the single RCON connection hostage.
        """
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            # Silently treating a chunked body as empty would run the handler with
            # missing arguments, so refuse it instead.
            raise _BodyError(
                411, "chunked request bodies are not supported", close=True
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise _BodyError(400, "malformed Content-Length", close=True)
        if length < 0:
            raise _BodyError(400, "negative Content-Length", close=True)
        if length == 0:
            return {}
        limit = MAX_BODY_BYTES_BY_PATH.get(path, MAX_BODY_BYTES_DEFAULT)
        if length > limit:
            raise _BodyError(413, f"request body exceeds {limit} bytes", close=True)

        previous_timeout = self.connection.gettimeout()
        self.connection.settimeout(BODY_READ_TIMEOUT_SECONDS)
        try:
            raw = self.rfile.read(length)
        except TimeoutError:
            raise _BodyError(408, "timed out reading request body", close=True)
        except OSError as exc:
            raise _BodyError(400, f"error reading request body: {exc}", close=True)
        finally:
            self.connection.settimeout(previous_timeout)
        if len(raw) != length:
            raise _BodyError(400, "truncated request body", close=True)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise _BodyError(400, "malformed JSON body")
        if not isinstance(body, dict):
            raise _BodyError(400, "request body must be a JSON object")
        return body

    def _dispatch(self, routes, verb):
        path = self.path.split("?", 1)[0]

        if not self._authorised():
            logger.warning(
                "Rejected unauthenticated %s %s from %s",
                verb,
                path,
                self.client_address,
            )
            self._send_json({"error": "unauthorized"}, 401, close=True)
            return

        handler = routes.get(path)
        if handler is None:
            # Any request body is left unread, so the connection is no longer framed.
            self._send_json(
                {"error": f"Unknown {verb} path: {self.path}"}, 404, close=True
            )
            return

        try:
            body = self._read_body(path)
        except _BodyError as exc:
            logger.warning("%s %s rejected: %s", verb, path, exc.message)
            self._send_json({"error": exc.message}, exc.status, close=exc.close)
            return

        # Only the game-state work is serialised; body parsing above and the response
        # write below happen outside the lock.
        try:
            if path in _LOCK_FREE_PATHS:
                result = handler(self, body)
            else:
                with _REQ_LOCK:
                    result = handler(self, body)
        except Exception as exc:
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(
                "%s %s failed [%s]: %s\n%s",
                verb,
                path,
                correlation_id,
                exc,
                traceback.format_exc(),
            )
            self._send_json(
                {
                    "error": "internal server error",
                    "correlation_id": correlation_id,
                },
                500,
            )
            return

        payload, status = result if isinstance(result, tuple) else (result, 200)
        self._send_json(payload, status)

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

    def _handle_health(self, body):
        ready = _gym_env is not None
        return {"status": "ok" if ready else "initialising"}

    def _handle_meta(self, body):
        return {
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

    def _handle_observe(self, body):
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
        return obs_dict

    def _handle_score(self, body):
        score, automated = _instance.namespaces[0].score()
        return {
            "production_score": score,
            "automated_production_score": automated or 0,
        }

    def _handle_system_prompt(self, body):
        import importlib.resources
        from fle.env.utils.controller_loader.system_prompt_generator import (
            SystemPromptGenerator,
        )

        generator = SystemPromptGenerator(str(importlib.resources.files("fle") / "env"))
        prompt = generator.generate_for_agent(agent_idx=0, num_agents=1)
        return {"system_prompt": prompt}

    def _handle_game_state(self, body):
        from fle.commons.models.game_state import GameState

        gs = GameState.from_instance(_instance)
        return {"game_state": gs.to_raw()}

    def _handle_state_save(self, body):
        from fle.commons.models.game_state import GameState

        gs = GameState.from_instance(_instance)
        return {"state": gs.to_raw()}

    def _handle_state_restore(self, body):
        raw = body.get("state")
        if not raw:
            return {"error": "missing 'state'"}, 400

        from fle.commons.models.game_state import GameState

        _instance.reset(GameState.parse_raw(raw))
        _game_states.clear()
        _gym_env.initial_score, _ = _instance.namespaces[0].score()
        _gym_env.last_observation = None
        return {"ok": True}

    def _handle_probe(self, body):
        entity = body.get("entity") or getattr(_task, "throughput_entity", None)
        if not entity:
            return {"error": "missing 'entity' and task has no default"}, 400
        return _run_probe(str(entity))

    def _handle_execute(self, body):
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

        return result

    def _handle_reset(self, body):
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
        return {"status": "ok"}

    def _handle_screenshot(self, body):
        namespace = _instance.namespaces[0]
        result = namespace._render(radius=64, max_render_radius=32, include_status=True)
        base64_data = result.to_base64()
        # Write to file for sandbox().read_file()
        with open("/tmp/screenshot.png", "wb") as f:
            import base64

            f.write(base64.b64decode(base64_data))
        return {
            "path": "/tmp/screenshot.png",
            "base64": f"data:image/png;base64,{base64_data}",
        }


# --- Main ---


def main():
    tcp_port = int(os.environ.get("FLE_BRIDGE_PORT", str(DEFAULT_TCP_PORT)))
    token = _bridge_token()
    allow_insecure = _env_flag("FLE_BRIDGE_ALLOW_INSECURE")
    # The TCP listener is reachable from outside the sandbox, so it is only opened when
    # it can be authenticated - or when the operator explicitly accepts an open port.
    serve_tcp = bool(token) or allow_insecure

    if serve_tcp:
        logger.info(
            "Starting FLE Bridge Service on %s and TCP :%d (TCP auth: %s)",
            SOCK_PATH,
            tcp_port,
            "bearer token"
            if token
            else "DISABLED via FLE_BRIDGE_ALLOW_INSECURE=1",
        )
    else:
        logger.warning(
            "FLE_BRIDGE_TOKEN is unset and FLE_BRIDGE_ALLOW_INSECURE is not 1: "
            "serving %s only, TCP :%d will NOT be opened.",
            SOCK_PATH,
            tcp_port,
        )

    # Initialise the environment (blocks until RCON is available). The listeners are
    # created afterwards, so /health answering at all implies a ready environment.
    try:
        _init_environment()
    except Exception:
        logger.error("Failed to initialise environment:\n%s", traceback.format_exc())
        sys.exit(1)

    uds_server = UnixHTTPServer(SOCK_PATH, BridgeHandler)
    tcp_server = None
    if serve_tcp:
        tcp_server = TcpHTTPServer(("0.0.0.0", tcp_port), BridgeHandler)
        threading.Thread(
            target=tcp_server.serve_forever, name="bridge-tcp", daemon=True
        ).start()
        logger.info(
            "Bridge service listening on %s and 0.0.0.0:%d", SOCK_PATH, tcp_port
        )
    else:
        logger.info("Bridge service listening on %s", SOCK_PATH)

    try:
        uds_server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge service.")
    finally:
        if tcp_server is not None:
            tcp_server.shutdown()
            tcp_server.server_close()
        uds_server.server_close()
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)


if __name__ == "__main__":
    main()
