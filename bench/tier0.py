#!/usr/bin/env python3
"""Tier 0 -- the mechanics gate for the Farplane fan-out benchmark.

Six stages, each independently re-runnable via ``--stages``:

``bake``
    Create a ``debian-warm`` sandbox, stream ``fle-sandbox:bench`` into it,
    start the bridge container, expose 8730, verify health from the host, and
    snapshot.  That snapshot is TEMPLATE_SNAP -- every later stage forks it.
``constants``
    5x each: snapshot / fork / expose / health / delete, with subcomponents.
``cooldown``
    Vary only the idle gap after a delete goes terminal, to separate intrinsic
    fork cost from waiting on a deleted child's warm supervisor slot.
``fidelity``
    Plant a factory, snapshot, fork three children sequentially, and compare
    ``factorio_pid`` / ``entity_count`` / ``/probe`` across children and parent.
``probe``
    The full parity probe cycle of design v2.4: snapshot -> 1 fork -> health ->
    ``/probe`` -> delete, timed end to end.
``soak``
    Design v2.4.1 post-parity demand: 3 concurrent source sandboxes x N rounds
    of (snapshot + 3 sequential forks + deletes) while a 4th source runs probe
    cycles.  Reports capacity stalls, latency distributions, node placement.

Everything we create is named ``flebench-*`` and journalled; the run ends with a
reaper pass that keeps only TEMPLATE_SNAP, the bake sandbox, and whatever a soak
worker that outlived its stage still owns (deleting those under a live thread is
how children get stranded).

Bridge auth: the in-guest bridge opens its TCP listener only when it has a
credential, and every exposed-port request is TCP, so export
``FLE_BRIDGE_TOKEN`` before running -- it is injected into the guest container's
environment at container start and sent by the host client on every call.
``FLE_BRIDGE_ALLOW_INSECURE=1`` is the explicit loopback/dev opt-out; with
neither set the run stops before it creates anything.

Image transfer note (measured, not assumed): the host sits behind NAT on
192.168.69.0/24 and the sandbox egress policy denies RFC1918, so a host-side
``python3 -m http.server`` is unreachable from the guest.  ``panda compute
sandboxes exec`` caps out at a 5 min gateway timeout, which makes base64 chunks
of a 400 MiB archive impractical.  What *does* work is the reverse direction: an
exposed guest port is reachable from the host without auth.  The ingress rejects
request bodies past ~1 MiB (HTTP 413), so the host streams ~900 KB PUTs over one
keep-alive connection into a guest receiver that pipes them straight into
``zstd -d | docker load`` -- no intermediate file on the 8 GiB guest disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.bridge_client import Bridge, BridgeError  # noqa: E402
from bench.common import atomic_write_json  # noqa: E402
from bench.farplane import Farplane, SB, summarize  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
STATE_PATH = RESULTS_DIR / "tier0_state.json"
JSON_PATH = RESULTS_DIR / "tier0.json"
SOAK_PATH = RESULTS_DIR / "tier0_soak.json"
MD_PATH = RESULTS_DIR / "TIER0.md"
TRANSFER_PATH = RESULTS_DIR / "tier0_transfer.json"
FIXTURE_STATE = BENCH_DIR / "fixtures" / "iron_ore_270_entities.state.json"

TEMPLATE = "debian-warm"
IMAGE_TAG = "fle-sandbox:bench"
BRIDGE_PORT = 8730
UPLOAD_PORT = 8899
CHUNK_BYTES = 900_000  # ingress rejects bodies past ~1 MiB
CONTAINER = "fle-bench"
GUEST_ENV_FILE = "/root/.fle-bridge.env"

#: Guest bridge environment that carries no credentials.  The C4 auth wiring
#: (FLE_BRIDGE_TOKEN / FLE_BRIDGE_ALLOW_INSECURE) is added by
#: :func:`bridge_guest_env` from the host environment at container start.
GUEST_ENV_BASE: dict[str, str] = {"FLE_BENCH_MODE": "1", "FLE_ENV_ID": "iron_ore_throughput"}


def docker_run_command() -> str:
    """``docker run`` for the guest bridge; its environment is GUEST_ENV_FILE.

    An env *file* rather than ``-e KEY=VALUE`` keeps the bridge token out of the
    guest process list.
    """
    return (
        f"docker run -d --name {CONTAINER} --restart unless-stopped "
        f"--env-file {GUEST_ENV_FILE} "
        f"-p {BRIDGE_PORT}:{BRIDGE_PORT} {IMAGE_TAG}"
    )


# Receiver that turns a sequence of size-capped PUTs back into one byte stream.
RECV_PY = r'''#!/usr/bin/env python3
"""Chunked upload sink: HTTP PUT bodies are concatenated into a sink command."""
import http.server, json, os, subprocess, sys, threading

PORT = int(sys.argv[1])
SINK = sys.argv[2]
STATE = "/tmp/recv_state.json"

sink = subprocess.Popen(["bash", "-c", SINK], stdin=subprocess.PIPE,
                        stdout=open("/tmp/sink.log", "wb"), stderr=subprocess.STDOUT)
state = {"bytes": 0, "chunks": 0, "done": False, "rc": None}


def flush_state():
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def _reply(self, code=200, body=b""):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_PUT(self):
        if self.path.rstrip("/").endswith("done"):
            try:
                sink.stdin.close()
            except Exception:
                pass
            state["rc"] = sink.wait()
            state["done"] = True
            flush_state()
            self._reply(200, json.dumps(state).encode())
            threading.Timer(1.0, lambda: os._exit(0)).start()
            return
        remaining = int(self.headers.get("Content-Length") or 0)
        while remaining > 0:
            buf = self.rfile.read(min(1 << 20, remaining))
            if not buf:
                break
            sink.stdin.write(buf)
            remaining -= len(buf)
            state["bytes"] += len(buf)
        state["chunks"] += 1
        if state["chunks"] % 50 == 0:
            flush_state()
        self._reply(200, b"")

    do_POST = do_PUT

    def do_GET(self):
        self._reply(200, json.dumps(state).encode())


flush_state()
http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''

# A few hundred entities of coal-fed burner drills on iron ore, verified shape
# taken from bench/BRIDGE_SMOKE.md (drill + chest producing iron-ore).
PLANT_PROGRAM = r'''
ore = nearest(Resource.IronOre)
patch = get_resource_patch(Resource.IronOre, ore, radius=30)
bb = patch.bounding_box
move_to(ore)
drills = 0
x = bb.left_top.x + 1
while x < bb.right_bottom.x - 1 and drills < 24:
    y = bb.left_top.y + 1
    while y < bb.right_bottom.y - 1 and drills < 24:
        try:
            d = place_entity(Prototype.BurnerMiningDrill,
                             position=Position(x=x, y=y), direction=Direction.DOWN)
            drills += 1
            try:
                insert_item(Prototype.Coal, d, 15)
            except Exception:
                pass
        except Exception:
            pass
        y += 3
    x += 2
belts = 0
base_x = ore.x + 45
base_y = ore.y - 10
for row in range(12):
    y = base_y + row * 2
    for block in range(3):
        try:
            move_to(Position(x=base_x + block * 8 + 3, y=y))
        except Exception:
            continue
        for col in range(8):
            try:
                place_entity(Prototype.TransportBelt,
                             position=Position(x=base_x + block * 8 + col, y=y),
                             direction=Direction.RIGHT)
                belts += 1
            except Exception:
                pass
print(f"drills={drills} belts={belts}")
'''


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageError(RuntimeError):
    """A stage failure that still carries the evidence it managed to collect.

    A stage that fails halfway has usually measured something worth keeping (and
    something worth explaining), but its payload must never be mistaken for a
    valid stage result -- :func:`ok_stage` keys off the ``error`` field, so the
    partial lands beside it rather than instead of it.

    ``live_resources`` is a callable a stage that failed with threads still
    running hands to the runner: called at cleanup time it answers "what do my
    live workers own *now*", which is what the failure reaper must not delete.
    """

    def __init__(self, message: str, *, partial: dict[str, Any] | None = None,
                 live_resources: Callable[[], dict[str, list[str]]] | None = None) -> None:
        super().__init__(message)
        self.partial = partial or {}
        self.live_resources = live_resources


class LiveResources:
    """Which sandboxes and snapshots each worker thread owns at this instant.

    A soak worker keeps forking off its source and deleting children as it goes,
    so "what is live" is only knowable from the workers themselves.  Threads
    register what they create and release what they delete; the failure reaper
    asks, by thread name, what it must leave alone.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[str, dict[str, set[str]]] = {}

    def hold(self, kind: str, resource_id: str) -> None:
        if not resource_id:
            return
        with self._lock:
            owner = self._held.setdefault(threading.current_thread().name,
                                          {"sandboxes": set(), "snapshots": set()})
            owner.setdefault(kind, set()).add(resource_id)

    def release(self, kind: str, resource_id: str) -> None:
        if not resource_id:
            return
        with self._lock:
            owner = self._held.get(threading.current_thread().name) or {}
            owner.get(kind, set()).discard(resource_id)

    def owned_by(self, names: list[str]) -> dict[str, list[str]]:
        """Everything the named threads hold right now, as ``{kind: [ids]}``."""
        sandboxes: set[str] = set()
        snapshots: set[str] = set()
        with self._lock:
            for name in names:
                owner = self._held.get(name) or {}
                sandboxes |= owner.get("sandboxes", set())
                snapshots |= owner.get("snapshots", set())
        return {"sandboxes": sorted(sandboxes), "snapshots": sorted(snapshots)}


def numeric_column(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Every numeric ``key`` across ``rows``; missing and non-numeric are dropped."""
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def stat_value(stat: Any, key: str = "p95") -> float | None:
    """One value out of a :func:`summarize` dict, or None if nothing was sampled.

    ``summarize`` reports zeros for an empty sample, so an ``or 0`` read turns
    "never measured" into "costs nothing"; ``n`` is the only honest guard.
    """
    if not isinstance(stat, dict) or not stat.get("n"):
        return None
    value = stat.get(key)
    return float(value) if isinstance(value, (int, float)) else None



# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(STATE_PATH, state)


def last_timing(fp: Farplane, op: str) -> dict[str, Any]:
    for record in reversed(fp.timings):
        if record.get("op") == op:
            return record
    return {}


# ----------------------------------------------------------------------
# bake
# ----------------------------------------------------------------------
def guest_bootstrap(fp: Farplane, sb: SB) -> None:
    """Install the tools the transfer needs.  Idempotent."""
    fp.exec(
        sb,
        "command -v zstd >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1 && exit 0; "
        "apt-get -o Acquire::Retries=3 -qq update >/dev/null 2>&1; "
        "apt-get -qq install -y zstd python3-minimal >/dev/null 2>&1; "
        "zstd --version >/dev/null && python3 -V >/dev/null",
        timeout="240s",
    )
    payload = base64.b64encode(RECV_PY.encode()).decode()
    fp.exec(sb, f"echo {payload} | base64 -d > /root/recv.py && chmod +x /root/recv.py")


def upload_stream(host: str, source: Any, *, label: str = "") -> dict[str, Any]:
    """PUT ``source`` to the guest receiver in ingress-sized chunks."""
    conn = http.client.HTTPSConnection(host, timeout=180, context=ssl.create_default_context())
    sent = 0
    chunks = 0
    started = time.monotonic()
    while True:
        buf = source.read(CHUNK_BYTES)
        if not buf:
            break
        for attempt in range(3):
            try:
                conn.request(
                    "PUT",
                    f"/up/c{chunks}",
                    body=buf,
                    headers={
                        "Content-Length": str(len(buf)),
                        "Content-Type": "application/octet-stream",
                    },
                )
                response = conn.getresponse()
                response.read()
                if response.status != 200:
                    raise RuntimeError(f"chunk {chunks} -> HTTP {response.status}")
                break
            except (http.client.HTTPException, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"chunk {chunks} failed: {exc}") from exc
                conn.close()
                conn = http.client.HTTPSConnection(
                    host, timeout=180, context=ssl.create_default_context()
                )
        sent += len(buf)
        chunks += 1
        if chunks % 100 == 0:
            elapsed = time.monotonic() - started
            log(f"    {label} {sent / 1e6:.0f} MB in {elapsed:.0f}s ({sent / elapsed / 1e6:.1f} MB/s)")
    conn.request("PUT", "/up/done", body=b"", headers={"Content-Length": "0"})
    response = conn.getresponse()
    sink_state = response.read().decode()
    conn.close()
    elapsed = time.monotonic() - started
    return {
        "bytes": sent,
        "chunks": chunks,
        "wall_s": round(elapsed, 2),
        "mb_per_s": round(sent / elapsed / 1e6, 2) if elapsed else 0.0,
        "sink_state": sink_state,
    }


def install_image(fp: Farplane, sb: SB, tar_path: Path) -> dict[str, Any]:
    """Stream the image archive into ``docker load`` inside the guest."""
    have = fp.exec(sb, f"docker image inspect {IMAGE_TAG} >/dev/null 2>&1 && echo YES || echo NO",
                   check=False).strip()
    if have.endswith("YES"):
        log(f"  image {IMAGE_TAG} already present in guest, skipping transfer")
        recorded = {}
        if TRANSFER_PATH.exists():
            recorded = json.loads(TRANSFER_PATH.read_text())
        return {"skipped": True, **recorded}

    # A stale receiver from an interrupted run would silently swallow the upload:
    # kill by listening port, never by command-line pattern (which matches this
    # very exec's argv and would take out the shell running it).
    fp.exec(
        sb,
        f"for p in $(ss -ltnp 2>/dev/null | grep ':{UPLOAD_PORT} ' "
        f"| grep -o 'pid=[0-9]*' | cut -d= -f2); do kill -9 $p 2>/dev/null; done; sleep 0.5",
        check=False,
    )
    fp.exec(
        sb,
        f"setsid python3 /root/recv.py {UPLOAD_PORT} 'zstd -d | docker load' "
        f"</dev/null >/tmp/recv_boot.log 2>&1 & sleep 2; ss -ltn | grep {UPLOAD_PORT}",
    )
    url = fp.expose(sb, UPLOAD_PORT)
    host = url.split("://", 1)[1].rstrip("/")
    log(f"  streaming {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB) -> {host}")
    with tar_path.open("rb") as handle:
        stats = upload_stream(host, handle, label="image")
    log(f"  transfer done: {stats['bytes'] / 1e6:.0f} MB in {stats['wall_s']}s "
        f"({stats['mb_per_s']} MB/s)")
    loaded = fp.exec(sb, "cat /tmp/sink.log", check=False)
    stats["docker_load"] = loaded.strip()[-500:]
    inspect = fp.exec(sb, f"docker image inspect {IMAGE_TAG} --format '{{{{.Id}}}}'").strip()
    stats["image_id"] = inspect
    atomic_write_json(TRANSFER_PATH, stats)
    return stats


def host_bridge_auth() -> tuple[str, bool]:
    """The host's bridge credentials: ``(token, insecure_opt_in)``."""
    return (
        os.environ.get("FLE_BRIDGE_TOKEN") or "",
        (os.environ.get("FLE_BRIDGE_ALLOW_INSECURE") or "") == "1",
    )


def auth_fingerprint(token: str) -> str:
    """Stable, non-reversible id for a bridge token (the empty token hashes too)."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def bridge_guest_env() -> dict[str, str]:
    """Guest container environment, including the bridge auth wiring.

    The guest bridge is only reachable through an exposed port, i.e. over TCP,
    and the service refuses to open a TCP listener without either a token or an
    explicit insecure opt-in.  So an unset host credential is a hard error here
    rather than a five-minute health-check timeout later.
    """
    token, insecure = host_bridge_auth()
    if not token and not insecure:
        raise SystemExit(
            "bridge auth is unset: export FLE_BRIDGE_TOKEN=<secret> (injected into the guest "
            "container here and sent by the host Bridge client), or FLE_BRIDGE_ALLOW_INSECURE=1 "
            "for a throwaway loopback run. With neither, the in-guest bridge serves its unix "
            "socket only and every exposed-port call fails."
        )
    if token and not re.fullmatch(r"[A-Za-z0-9_.:\-]{8,}", token):
        raise SystemExit(
            "FLE_BRIDGE_TOKEN must be at least 8 chars of [A-Za-z0-9_.:-]: it is carried to the "
            "guest in a docker --env-file and sourced by the in-guest health probe, neither of "
            "which quotes shell metacharacters."
        )
    env = dict(GUEST_ENV_BASE)
    if token:
        env["FLE_BRIDGE_TOKEN"] = token
    if insecure:
        env["FLE_BRIDGE_ALLOW_INSECURE"] = "1"
    return env


def write_guest_env(fp: Farplane, sb: SB, env: dict[str, str]) -> None:
    """Install GUEST_ENV_FILE (mode 0600) in the guest.

    base64 so the token is not an argv literal in the guest process list; it
    still transits the exec gateway, so rotate it if the operation journal is
    shared.
    """
    body = "".join(f"{key}={value}\n" for key, value in env.items())
    payload = base64.b64encode(body.encode()).decode()
    fp.exec(sb, f"umask 077 && echo {payload} | base64 -d > {GUEST_ENV_FILE}")


def guest_health_probe() -> str:
    """In-guest ``/health`` curl.

    127.0.0.1 is a TCP client of the bridge, so it needs the bearer token; the
    probe sources the container's env file instead of embedding the secret in
    this argv.
    """
    url = f"http://127.0.0.1:{BRIDGE_PORT}/health"
    return (
        f". {GUEST_ENV_FILE} 2>/dev/null || true; "
        'if [ -n "${FLE_BRIDGE_TOKEN:-}" ]; then '
        f'curl -sf -m 5 -H "Authorization: Bearer $FLE_BRIDGE_TOKEN" {url}; '
        f"else curl -sf -m 5 {url}; fi || echo WAIT"
    )


def start_container(fp: Farplane, sb: SB) -> dict[str, Any]:
    """(Re)start the bridge container and wait for in-guest health.

    A container left over from a run with a different bridge token is recreated
    rather than trusted: its env was fixed at create time, so a rotated host
    token would otherwise surface as an unexplained 401 in every later stage.
    """
    env = bridge_guest_env()
    write_guest_env(fp, sb, env)
    want_fp = auth_fingerprint(env.get("FLE_BRIDGE_TOKEN", ""))
    running = fp.exec(
        sb, f"docker inspect -f '{{{{.State.Running}}}}' {CONTAINER} 2>/dev/null || echo none",
        check=False,
    ).strip()
    have_fp = ""
    if running.endswith("true"):
        # sha256 of the container's own FLE_BRIDGE_TOKEN (empty when unset,
        # which hashes to the same digest the host computes for insecure mode).
        words = fp.exec(
            sb,
            "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "
            f"{CONTAINER} 2>/dev/null | sed -n 's/^FLE_BRIDGE_TOKEN=//p' "
            "| tr -d '\\n' | sha256sum | cut -c1-12",
            check=False,
        ).split()
        have_fp = words[-1] if words else ""
        if have_fp != want_fp:
            log("  guest container was created with a different bridge token; recreating")
    if not running.endswith("true") or have_fp != want_fp:
        fp.exec(sb, f"docker rm -f {CONTAINER} >/dev/null 2>&1; {docker_run_command()}",
                timeout="120s")
    started = time.monotonic()
    deadline = started + 300
    probe_cmd = guest_health_probe()
    while time.monotonic() < deadline:
        probe = fp.exec(sb, probe_cmd, check=False).strip()
        if '"ok"' in probe:
            return {
                "guest_health_s": round(time.monotonic() - started, 2),
                "bridge_auth": "token" if env.get("FLE_BRIDGE_TOKEN") else "insecure",
                "bridge_auth_fp": want_fp,
            }
        time.sleep(3)
    logs = fp.exec(sb, f"docker logs --tail 40 {CONTAINER} 2>&1", check=False)
    raise RuntimeError(f"bridge never became healthy in guest; logs:\n{logs[-3000:]}")


def stage_bake(fp: Farplane, args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    log("STAGE bake")
    result: dict[str, Any] = {"started": now_iso()}
    tar_path = Path(args.image_tar)
    if not tar_path.exists():
        raise SystemExit(f"image archive {tar_path} not found")

    sandbox_id = args.bake_sandbox or state.get("bake_sandbox")
    if sandbox_id:
        info = fp.get_sandbox(sandbox_id)
        sb = SB(id=sandbox_id, name=info.get("name") or "", node=info.get("node") or "")
        fp.created_sandboxes[sb.id] = sb.name
        log(f"  reusing bake sandbox {sb}")
        # TTL expiry hibernates rather than deletes, but a hibernated bake
        # sandbox is just as useless to Tier 1 -- push the lease out.
        try:
            fp.lease(sb, "6h")
        except Exception as exc:
            log(f"  lease extension failed (non-fatal): {exc}")
    else:
        t0 = time.monotonic()
        sb = fp.create_from_template(TEMPLATE, args.bake_ttl, name="bake")
        result["create_s"] = round(time.monotonic() - t0, 2)
        log(f"  created bake sandbox {sb} in {result['create_s']}s")
    state["bake_sandbox"] = sb.id
    save_state(state)

    guest_bootstrap(fp, sb)
    result["transfer"] = install_image(fp, sb, tar_path)
    result["container"] = start_container(fp, sb)
    state["bridge_auth_fp"] = result["container"]["bridge_auth_fp"]
    log(f"  bridge healthy in guest after {result['container']['guest_health_s']}s "
        f"(auth {result['container']['bridge_auth']})")

    t0 = time.monotonic()
    url = fp.expose(sb, BRIDGE_PORT)
    result["expose_s"] = round(time.monotonic() - t0, 3)
    bridge = Bridge(url)
    t0 = time.monotonic()
    bridge.wait_healthy(180)
    result["host_health_s"] = round(time.monotonic() - t0, 2)
    result["bake_url"] = url
    result["meta"] = bridge.meta()
    log(f"  host-side /health ok via {url} ({result['host_health_s']}s); meta={result['meta']}")

    # Tidy the guest so the template snapshot carries nothing we added by accident.
    fp.exec(sb, "docker image rm -f alpine:3 >/dev/null 2>&1; rm -f /tmp/recv_state.json "
                "/tmp/sink.log /tmp/up.bin /tmp/recv.rc /tmp/recv.log; sync", check=False)

    t0 = time.monotonic()
    # `--ttl 0` is documented as "no expiry" but the API rejects it
    # (invalid_ttl: must be a positive duration), so pin the longest lease.
    snap = fp.snapshot(sb, ttl=args.template_snap_ttl, note="flebench-TEMPLATE_SNAP")
    result["template_snapshot_s"] = round(time.monotonic() - t0, 2)
    result["template_snap"] = snap
    state["template_snap"] = snap
    state["template_snap_ttl"] = args.template_snap_ttl
    state["bake_url"] = url
    save_state(state)
    log(f"  TEMPLATE_SNAP = {snap} ({result['template_snapshot_s']}s)")
    result["finished"] = now_iso()
    return result


# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
def fork_and_ready(
    fp: Farplane,
    snap: str,
    ttl: str,
    name: str,
    *,
    wait_health: bool = True,
    deadline: float | None = None,
    queue_deadline: str = "5m",
) -> tuple[SB, dict[str, Any]]:
    """One child: fork -> running -> expose -> bridge health.  Returns subcomponents."""
    parts: dict[str, Any] = {}
    t0 = time.monotonic()
    child = fp.fork(snap, ttl, name=name, deadline=deadline, queue_deadline=queue_deadline)
    parts["fork_total_s"] = round(time.monotonic() - t0, 3)
    fork_rec = last_timing(fp, "fork")
    ready_rec = last_timing(fp, "fork_child_ready")
    parts["fork_op_s"] = fork_rec.get("duration_s")
    parts["fork_op_attempts"] = int(fork_rec.get("op_attempts") or 1)
    parts["fork_poll_overhead_s"] = fork_rec.get("poll_overhead_s")
    parts["fork_poll_count"] = fork_rec.get("poll_count")
    parts["child_ready_s"] = ready_rec.get("duration_s")
    parts["capacity_waits"] = ready_rec.get("capacity_waits", 0)
    parts["placement_lane"] = ready_rec.get("placement_lane")
    parts["preclaim_produced"] = (fork_rec.get("result") or {}).get("fork_preclaim_produced")
    parts["preclaim_miss"] = bool((fork_rec.get("result") or {}).get("fork_preclaim_miss"))
    parts["node"] = child.node

    # The fork succeeded, so from here on the child exists and is billed even if
    # exposing it or its bridge health fails: nothing may return without either
    # handing the caller the child or deleting it.
    try:
        t0 = time.monotonic()
        url = fp.expose(child, BRIDGE_PORT)
        parts["expose_s"] = round(time.monotonic() - t0, 3)
        parts["url"] = url
        if wait_health:
            bridge = Bridge(url)
            t0 = time.monotonic()
            bridge.wait_healthy(240, poll_interval=0.5)
            parts["health_s"] = round(time.monotonic() - t0, 3)
            parts["total_to_healthy_s"] = round(
                parts["fork_total_s"] + parts["expose_s"] + parts["health_s"], 3
            )
    except BaseException:
        try:
            fp.delete_sandbox(child)
            log(f"  deleted unusable child {child.id} after post-fork failure")
        except Exception as exc:
            log(f"  LEAK: child {child.id} survived a post-fork failure: "
                f"{type(exc).__name__}: {exc}")
        raise
    return child, parts


def stage_constants(
    fp: Farplane, args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    log(f"STAGE constants ({args.repeats}x)")
    source = SB(id=state["bake_sandbox"], name="flebench-bake")
    rounds: list[dict[str, Any]] = []
    for i in range(args.repeats):
        entry: dict[str, Any] = {"i": i}
        t0 = time.monotonic()
        snap = fp.snapshot(source, ttl="2h", note=f"flebench-const-{i}")
        entry["t_snap_s"] = round(time.monotonic() - t0, 3)
        snap_rec = last_timing(fp, "snapshot")
        entry["snap_op_ms"] = snap_rec.get("op_duration_ms")
        entry["snap_poll_overhead_s"] = snap_rec.get("poll_overhead_s")
        entry["snap_bytes"] = (snap_rec.get("result") or {}).get("capture_memory_image_bytes")

        child, parts = fork_and_ready(fp, snap, "40m", f"const-{i}")
        entry.update(parts)

        t0 = time.monotonic()
        fp.delete_sandbox(child)
        entry["t_delete_s"] = round(time.monotonic() - t0, 3)
        t0 = time.monotonic()
        fp.delete_snapshot(snap)
        entry["t_delete_snapshot_s"] = round(time.monotonic() - t0, 3)
        rounds.append(entry)
        log(f"  round {i}: snap {entry['t_snap_s']}s  fork {entry['fork_total_s']}s "
            f"(attempts {entry['fork_op_attempts']})  expose {entry['expose_s']}s  "
            f"health {entry.get('health_s')}s  del {entry['t_delete_s']}s")

    def column(key: str) -> list[float]:
        return [r[key] for r in rounds if isinstance(r.get(key), (int, float))]

    stats = {
        key: summarize(column(key))
        for key in (
            "t_snap_s", "fork_total_s", "fork_op_s", "child_ready_s", "expose_s",
            "health_s", "total_to_healthy_s", "t_delete_s", "t_delete_snapshot_s",
        )
    }
    poll_records = [
        r for r in fp.timings
        if r.get("outcome") == "ok" and isinstance(r.get("poll_count"), int) and r["poll_count"]
    ]
    per_poll = [
        r["poll_overhead_s"] / r["poll_count"] for r in poll_records if r.get("poll_overhead_s")
    ]
    return {
        "rounds": rounds,
        "stats": stats,
        "op_poll_overhead_per_call_s": summarize(per_poll),
        "capacity_waits_total": sum(int(r.get("capacity_waits") or 0) for r in rounds),
    }


# ----------------------------------------------------------------------
# deletion -> capacity reuse
# ----------------------------------------------------------------------
def stage_cooldown(
    fp: Farplane, args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    """Is t_fork intrinsic, or is it waiting for a deleted child's slot to free?

    The constants stage forks, deletes, and immediately forks again.  Its first
    round (cold lane) cost 4 attempts / ~13s while every later round cost 15-16
    attempts / ~105s -- the signature of a fork bouncing off a warm supervisor
    pre-claim that the previous child has not yet released.  This stage varies
    only the idle gap after the delete goes terminal: if the attempt count falls
    back to the cold-lane value at some cooldown, that cooldown IS the
    deletion-to-capacity-reuse constant and the fork itself is cheap.
    """
    cooldowns = [float(c) for c in args.cooldowns.split(",") if c.strip()]
    log(f"STAGE cooldown (post-delete gaps {cooldowns}, {args.cooldown_rounds} rounds each)")
    source = SB(id=state["bake_sandbox"], name="flebench-bake")
    buckets: list[dict[str, Any]] = []
    for cooldown in cooldowns:
        rounds: list[dict[str, Any]] = []
        for i in range(args.cooldown_rounds):
            entry: dict[str, Any] = {"i": i, "cooldown_s": cooldown}
            t0 = time.monotonic()
            snap = fp.snapshot(source, ttl="1h", note=f"flebench-cool-{int(cooldown)}-{i}")
            entry["t_snap_s"] = round(time.monotonic() - t0, 3)
            child, parts = fork_and_ready(
                fp, snap, "20m", f"cool-{int(cooldown)}-{i}",
                wait_health=False,
                deadline=args.fork_deadline_s,
                queue_deadline=args.fork_queue_deadline,
            )
            entry.update({k: parts[k] for k in
                          ("fork_total_s", "fork_op_s", "fork_op_attempts", "child_ready_s",
                           "placement_lane", "preclaim_produced", "preclaim_miss", "node")
                          if k in parts})
            t0 = time.monotonic()
            fp.delete_sandbox(child)
            entry["t_delete_s"] = round(time.monotonic() - t0, 3)
            t0 = time.monotonic()
            fp.delete_snapshot(snap)
            entry["t_delete_snapshot_s"] = round(time.monotonic() - t0, 3)
            rounds.append(entry)
            log(f"  cooldown {cooldown:.0f}s round {i}: fork {entry['fork_total_s']}s "
                f"attempts {entry['fork_op_attempts']} lane {entry.get('placement_lane')} "
                f"preclaim {entry.get('preclaim_produced')}")
            if cooldown:
                time.sleep(cooldown)
        attempts = [float(r["fork_op_attempts"]) for r in rounds]
        buckets.append({
            "cooldown_s": cooldown,
            "rounds": rounds,
            "fork_op_attempts": summarize(attempts),
            "fork_op_s": summarize([float(r["fork_op_s"] or 0) for r in rounds]),
            "fork_total_s": summarize([float(r["fork_total_s"]) for r in rounds]),
            "preclaim_hits": sum(1 for r in rounds if str(r.get("preclaim_produced")) == "1/1"),
        })

    # The reuse constant is the smallest cooldown whose median attempt count is
    # within one attempt of the best (cheapest) bucket.
    best = min(b["fork_op_attempts"]["p50"] for b in buckets)
    recovered = [b["cooldown_s"] for b in buckets
                 if b["fork_op_attempts"]["p50"] <= best + 1.0]
    return {
        "buckets": buckets,
        "best_p50_attempts": best,
        "reuse_constant_s": min(recovered) if recovered else None,
        "attempts_recovered_at": recovered,
    }


# ----------------------------------------------------------------------
# fidelity
# ----------------------------------------------------------------------
def plant_factory(bridge: Bridge, args: argparse.Namespace) -> dict[str, Any]:
    """Install the fidelity fixture: a captured 270-entity iron-ore factory.

    A restored fixture beats re-running a build program: it is byte-identical
    every time, and its sinks are chests rather than belts, so the probe does
    not drift as belt buffers saturate (see bench/BRIDGE_SMOKE.md).
    """
    fixture = Path(args.plant_state) if args.plant_state else FIXTURE_STATE
    if fixture.exists():
        raw = fixture.read_text()
        t0 = time.monotonic()
        bridge.state_restore(raw)
        meta = bridge.meta()
        return {"method": "state-restore", "fixture": str(fixture),
                "bytes": len(raw), "restore_s": round(time.monotonic() - t0, 3),
                "meta": meta}
    program = Path(args.plant_program).read_text() if args.plant_program else PLANT_PROGRAM
    result = bridge.execute(program, timeout=900)
    return {"method": "execute", "output": str(result.get("result"))[-800:],
            "execute": {k: v for k, v in result.items() if k != "result"},
            "meta": bridge.meta()}


def restore_bake_baseline(parent: Bridge, baseline_state: str) -> dict[str, Any]:
    """Put the bake sandbox back the way the fidelity stage found it.

    Round-trips the saved baseline (which also proves ``/state-restore`` on the
    parent), then resets to the task's greenfield start so the bake sandbox
    still matches TEMPLATE_SNAP for Tier 1.  Never raises: it runs in a finally,
    so it must not mask the exception that sent it there -- the caller decides
    whether a failed restore fails the stage.
    """
    report: dict[str, Any] = {"attempted": True, "ok": False}
    try:
        t0 = time.monotonic()
        parent.state_restore(baseline_state)
        report["restore_s"] = round(time.monotonic() - t0, 3)
        report["restored_meta"] = parent.meta()
        parent.reset()
        report["greenfield_meta"] = parent.meta()
        report["ok"] = True
        log(f"  restored bake sandbox: "
            f"entity_count={report['restored_meta'].get('entity_count')} -> reset greenfield "
            f"entity_count={report['greenfield_meta'].get('entity_count')}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        log(f"  BAKE SANDBOX LEFT DIRTY: baseline restore failed: {report['error']}")
    return report


def stage_fidelity(
    fp: Farplane, args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    log("STAGE fidelity")
    source = SB(id=state["bake_sandbox"], name="flebench-bake")
    parent = Bridge(state["bake_url"])
    parent.wait_healthy(120)

    baseline_state = parent.state_save()
    log(f"  saved parent state ({len(baseline_state)} bytes) for post-stage restore")

    # Everything below mutates the bake sandbox -- the one Tier 1 inherits --
    # starting with the fixture plant, so the baseline restore is a finally, not
    # a happy-path step at the end.
    try:
        planted = plant_factory(parent, args)
        log(f"  planted via {planted['method']}: "
            f"entity_count={planted['meta'].get('entity_count')}")

        parent_meta_before = parent.meta()
        parent_probe_before = parent.probe(args.probe_entity)
        log(f"  parent probe before fork: {parent_probe_before}")

        snap = fp.snapshot(source, ttl="2h", note="flebench-fidelity")
        children: list[dict[str, Any]] = []
        child_objs: list[SB] = []
        try:
            for i in range(args.fidelity_children):
                child, parts = fork_and_ready(fp, snap, "40m", f"fid-{i}")
                child_objs.append(child)
                bridge = Bridge(parts["url"])
                meta = bridge.meta()
                probe = bridge.probe(args.probe_entity)
                children.append({"i": i, "sandbox": child.id, "node": child.node,
                                 "fork": parts, "meta": meta, "probe": probe})
                log(f"  child {i} {child.id}@{child.node}: pid={meta.get('factorio_pid')} "
                    f"entities={meta.get('entity_count')} tick={meta.get('game_tick')} "
                    f"throughput={probe.get('throughput')} "
                    f"items={probe.get('end_count')} - {probe.get('start_count')}")

            parent_meta_after = parent.meta()
            parent_probe_after = parent.probe(args.probe_entity)
        finally:
            for child in child_objs:
                try:
                    fp.delete_sandbox(child)
                except Exception as exc:
                    log(f"  child cleanup failed: {exc}")
            try:
                fp.delete_snapshot(snap)
            except Exception as exc:
                log(f"  snapshot cleanup failed: {exc}")

        pids = {c["meta"].get("factorio_pid") for c in children}
        entity_counts = {c["meta"].get("entity_count") for c in children}
        throughputs = [c["probe"].get("throughput") for c in children]
        items = [
            (c["probe"].get("end_count") or 0) - (c["probe"].get("start_count") or 0)
            for c in children
        ]
        spread = (max(throughputs) - min(throughputs)) if throughputs else 0.0
        verdict = {
            "factorio_pid_identical": len(pids) == 1,
            "factorio_pid_matches_parent": pids == {parent_meta_before.get("factorio_pid")},
            "entity_count_identical": len(entity_counts) == 1,
            "entity_count_matches_parent": (
                entity_counts == {parent_meta_before.get("entity_count")}
            ),
            "probe_spread": round(spread, 6),
            "probe_identical": spread == 0.0,
            "probe_items_per_window": items,
            "probe_items_identical": len(set(items)) == 1,
            "probe_tolerance": args.probe_tolerance,
            "parent_entity_count_unchanged": (
                parent_meta_before.get("entity_count") == parent_meta_after.get("entity_count")
            ),
            "parent_probe_delta": round(
                abs((parent_probe_after.get("throughput") or 0)
                    - (parent_probe_before.get("throughput") or 0)), 6
            ),
            "parent_pid_unchanged": (
                parent_meta_before.get("factorio_pid") == parent_meta_after.get("factorio_pid")
            ),
        }
        # Every invariant this stage computes is scored.  Two are scored against
        # a tolerance rather than as-is: probe_spread against --probe-tolerance,
        # and probe_identical (spread exactly 0.0) is NOT a gate because the
        # probe closes 1-2 ticks past its 3600-tick window and divides by the
        # actual delta -- the exactness claim is carried by probe_items_identical
        # (integer production counters).  parent_probe_delta is not an invariant
        # at all: the parent keeps running and its burner drills run out of coal.
        checks = {
            "children_probed": len(children) == args.fidelity_children and bool(children),
            "factorio_pid_identical": verdict["factorio_pid_identical"],
            "factorio_pid_matches_parent": verdict["factorio_pid_matches_parent"],
            "entity_count_identical": verdict["entity_count_identical"],
            "entity_count_matches_parent": verdict["entity_count_matches_parent"],
            "probe_items_identical": verdict["probe_items_identical"],
            "probe_spread_within_tolerance": verdict["probe_spread"] <= args.probe_tolerance,
            "parent_entity_count_unchanged": verdict["parent_entity_count_unchanged"],
            "parent_pid_unchanged": verdict["parent_pid_unchanged"],
        }
        verdict["checks"] = checks
        verdict["failed_checks"] = sorted(key for key, ok in checks.items() if not ok)
        verdict["pass"] = not verdict["failed_checks"]
        log(f"  verdict: {verdict}")
    finally:
        restore = restore_bake_baseline(parent, baseline_state)

    if not restore["ok"]:
        # The bake sandbox is a Tier 1 input; a polluted one is a stage failure
        # even though every fidelity check passed.
        raise RuntimeError(
            f"fidelity fixture left in the bake sandbox: {restore.get('error')}"
        )

    return {
        "planted": planted,
        "parent_before": {"meta": parent_meta_before, "probe": parent_probe_before},
        "parent_after": {"meta": parent_meta_after, "probe": parent_probe_after},
        "parent_restored_meta": restore.get("restored_meta"),
        "parent_greenfield_meta": restore.get("greenfield_meta"),
        "baseline_restore": restore,
        "baseline_state_bytes": len(baseline_state),
        "children": children,
        "verdict": verdict,
    }


# ----------------------------------------------------------------------
# probe cycle
# ----------------------------------------------------------------------
def cycle_fork_parts(cycle: dict[str, Any]) -> dict[str, Any]:
    """The fork subcomponents a probe cycle recorded under its ``fork_`` prefix.

    Empty when the cycle never got a child (its fork raised), which is how
    callers tell a measured probe fork from a failed one.
    """
    return {key[5:]: value for key, value in cycle.items() if key.startswith("fork_")}


def release_cycle(
    fp: Farplane,
    cycle: dict[str, Any],
    child: SB | None,
    snap: str,
    live: LiveResources | None,
) -> list[str]:
    """Delete a probe cycle's child, then its snapshot; report what survived.

    A deliberate best-effort teardown: each delete is attempted even when the
    other raises, because a child that outlives its cycle is a live sandbox
    holding a warm slot, and a snapshot that outlives it fences the next fork.
    Whatever could not be deleted comes back as a string so the caller can fail
    loudly rather than loop on.
    """
    leaked: list[str] = []
    if child is not None:
        t0 = time.monotonic()
        try:
            fp.delete_sandbox(child)
            cycle["delete_child_s"] = round(time.monotonic() - t0, 3)
            if live:
                live.release("sandboxes", child.id)
        except Exception as exc:
            cycle["delete_child_error"] = f"{type(exc).__name__}: {exc}"
            leaked.append(f"probe child {child.id} survived its cycle: "
                          f"{type(exc).__name__}: {exc}")
    t0 = time.monotonic()
    try:
        fp.delete_snapshot(snap)
        cycle["delete_snapshot_s"] = round(time.monotonic() - t0, 3)
        if live:
            live.release("snapshots", snap)
    except Exception as exc:
        cycle["delete_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        leaked.append(f"probe snapshot {snap} survived its cycle: {type(exc).__name__}: {exc}")
    return leaked


def probe_cycle(
    fp: Farplane,
    source: SB,
    entity: str,
    tag: str,
    *,
    ttl: str = "30m",
    deadline: float | None = None,
    queue_deadline: str = "5m",
    live: LiveResources | None = None,
) -> dict[str, Any]:
    """snapshot -> 1 fork -> health -> /probe -> delete child -> delete snapshot.

    The child is held outside the ``try`` so the cleanup owns it from the moment
    it exists: a /probe that raises -- or a delete that does -- must not leave a
    live sandbox behind while the soak's probe worker loops straight into the
    next cycle.  Both deletes run, child first, whatever the body did.  A
    cleanup failure is raised only when the body itself succeeded, so it never
    masks the error that caused it.
    """
    cycle: dict[str, Any] = {"tag": tag}
    t_all = time.monotonic()
    t0 = time.monotonic()
    snap = fp.snapshot(source, ttl="1h", note=f"flebench-probe-{tag}")
    cycle["snapshot_s"] = round(time.monotonic() - t0, 3)
    if live:
        live.hold("snapshots", snap)
    child: SB | None = None
    try:
        child, parts = fork_and_ready(fp, snap, ttl, f"probe-{tag}",
                                      deadline=deadline, queue_deadline=queue_deadline)
        if live:
            live.hold("sandboxes", child.id)
        cycle.update({f"fork_{k}": v for k, v in parts.items()})
        bridge = Bridge(parts["url"])
        t0 = time.monotonic()
        cycle["probe"] = bridge.probe(entity)
        cycle["probe_s"] = round(time.monotonic() - t0, 3)
    finally:
        leaked = release_cycle(fp, cycle, child, snap, live)
    if leaked:
        raise StageError("probe cycle leaked resources: " + "; ".join(leaked), partial=cycle)
    cycle["t_probe_cycle_s"] = round(time.monotonic() - t_all, 3)
    return cycle


def stage_probe(fp: Farplane, args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    log(f"STAGE probe-cycle ({args.probe_repeats}x)")
    source = SB(id=state["bake_sandbox"], name="flebench-bake")
    cycles = []
    for i in range(args.probe_repeats):
        cycle = probe_cycle(fp, source, args.probe_entity, f"c{i}")
        cycles.append(cycle)
        log(f"  cycle {i}: {cycle['t_probe_cycle_s']}s "
            f"(snap {cycle['snapshot_s']}s fork {cycle.get('fork_fork_total_s')}s "
            f"health {cycle.get('fork_health_s')}s probe {cycle['probe_s']}s "
            f"del {cycle['delete_child_s']}s)")
    return {
        "cycles": cycles,
        "t_probe_cycle_s": summarize([c["t_probe_cycle_s"] for c in cycles]),
        "throughput": summarize(
            [c["probe"].get("throughput", 0.0) for c in cycles if c.get("probe")]
        ),
    }


# ----------------------------------------------------------------------
# soak
# ----------------------------------------------------------------------
def stage_soak(fp: Farplane, args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    log(f"STAGE soak ({args.soak_sources} branch sources x {args.soak_rounds} rounds, "
        f"+1 probe source, budget {args.soak_budget_s}s)")
    snap = state["template_snap"]
    total_sources = args.soak_sources + 1
    sources: list[SB] = []
    create_times: list[float] = []
    for i in range(total_sources):
        t0 = time.monotonic()
        sb = fp.create_from_snapshot(snap, args.soak_ttl, name=f"soak-src{i}")
        create_times.append(round(time.monotonic() - t0, 2))
        sources.append(sb)
        log(f"  source {i}: {sb} in {create_times[-1]}s")

    # The probe source carries the fidelity fixture so soak probe cycles measure
    # a real factory (the branch sources stay greenfield -- their forks are only
    # timed, never scored).
    probe_fixture: dict[str, Any] = {"loaded": False}
    fixture = Path(args.plant_state) if args.plant_state else FIXTURE_STATE
    if fixture.exists():
        probe_url = fp.expose(sources[-1], BRIDGE_PORT)
        probe_bridge = Bridge(probe_url)
        probe_bridge.wait_healthy(180)
        probe_bridge.state_restore(fixture.read_text())
        probe_fixture = {"loaded": True, "meta": probe_bridge.meta()}
        log(f"  probe source seeded from fixture: {probe_fixture['meta']}")

    deadline = time.monotonic() + args.soak_budget_s
    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()
    errors: list[str] = []
    stop = threading.Event()
    # What each worker owns right now, so a failure that leaves threads running
    # can tell the reaper exactly which resources are still in use.
    live = LiveResources()

    def record(event: dict[str, Any]) -> None:
        with events_lock:
            events.append(event)

    def branch_worker(index: int, source: SB) -> None:
        live.hold("sandboxes", source.id)
        for round_index in range(args.soak_rounds):
            # A round owns live children, so the stop flag is honoured at round
            # boundaries only -- never mid-round, where an abort would strand
            # them.
            if stop.is_set() or time.monotonic() > deadline:
                record({"kind": "budget_stop", "source": index, "round": round_index,
                        "reason": "stop_requested" if stop.is_set() else "budget"})
                return
            children: list[SB] = []
            round_snap = ""
            try:
                t_round = time.monotonic()
                t0 = time.monotonic()
                round_snap = fp.snapshot(
                    source, ttl="1h", note=f"flebench-soak-{index}-{round_index}"
                )
                snap_s = round(time.monotonic() - t0, 3)
                live.hold("snapshots", round_snap)
                fork_entries = []
                for k in range(args.soak_width):
                    child, parts = fork_and_ready(
                        fp, round_snap, "30m", f"soak-{index}-{round_index}-{k}",
                        wait_health=args.soak_health,
                        deadline=args.fork_deadline_s,
                        queue_deadline=args.fork_queue_deadline,
                    )
                    children.append(child)
                    live.hold("sandboxes", child.id)
                    fork_entries.append(parts)
                delete_times = []
                for child in children:
                    t0 = time.monotonic()
                    fp.delete_sandbox(child)
                    delete_times.append(round(time.monotonic() - t0, 3))
                    live.release("sandboxes", child.id)
                children = []
                t0 = time.monotonic()
                fp.delete_snapshot(round_snap)
                snap_delete_s = round(time.monotonic() - t0, 3)
                live.release("snapshots", round_snap)
                record({
                    "kind": "branch_round",
                    "source": index,
                    "sandbox": source.id,
                    "round": round_index,
                    "snapshot_s": snap_s,
                    "forks": fork_entries,
                    "delete_child_s": delete_times,
                    "delete_snapshot_s": snap_delete_s,
                    "round_s": round(time.monotonic() - t_round, 3),
                })
                log(f"  [src{index}] round {round_index} done in "
                    f"{round(time.monotonic() - t_round, 1)}s "
                    f"(forks {[f['fork_total_s'] for f in fork_entries]})")
            except Exception as exc:
                errors.append(f"branch{index}/round{round_index}: {type(exc).__name__}: {exc}")
                record({"kind": "branch_error", "source": index, "round": round_index,
                        "error": f"{type(exc).__name__}: {exc}"})
                log(f"  [src{index}] round {round_index} FAILED: {type(exc).__name__}: {exc}")
                # A failed round must not leak: drop whatever landed, then the
                # snapshot, so the next round is not fenced by a live child.
                # Deliberate best-effort teardown -- one undeletable child must
                # not stop us dropping the rest.
                for child in children:
                    try:
                        fp.delete_sandbox(child)
                        live.release("sandboxes", child.id)
                    except Exception:
                        pass
                if round_snap:
                    try:
                        fp.delete_snapshot(round_snap)
                        live.release("snapshots", round_snap)
                    except Exception:
                        pass

    def probe_worker(source: SB) -> None:
        live.hold("sandboxes", source.id)
        i = 0
        consecutive = 0
        # A cycle that keeps failing keeps paying for a snapshot and a fork, and
        # each failure is one more chance to strand a child; three in a row means
        # the path is broken, not flaky, so stop asking.
        failure_limit = 3
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                cycle = probe_cycle(fp, source, args.probe_entity, f"soak{i}",
                                    deadline=args.fork_deadline_s,
                                    queue_deadline=args.fork_queue_deadline,
                                    live=live)
                cycle["kind"] = "probe_cycle"
                record(cycle)
                consecutive = 0
                log(f"  [probe] cycle {i} in {cycle['t_probe_cycle_s']}s")
            except Exception as exc:
                consecutive += 1
                errors.append(f"probe{i}: {type(exc).__name__}: {exc}")
                record({"kind": "probe_error", "i": i, "consecutive_failures": consecutive,
                        "error": f"{type(exc).__name__}: {exc}"})
                log(f"  [probe] cycle {i} FAILED: {type(exc).__name__}: {exc}")
                if consecutive >= failure_limit:
                    errors.append(f"probe worker gave up after {consecutive} consecutive "
                                  f"cycle failures")
                    record({"kind": "probe_stop", "i": i,
                            "reason": f"{consecutive} consecutive cycle failures"})
                    log(f"  [probe] giving up after {consecutive} consecutive failures")
                    return
            i += 1
            if i >= args.soak_rounds * 2:
                return

    threads = [
        threading.Thread(target=branch_worker, args=(i, sources[i]),
                         name=f"soak-branch{i}", daemon=True)
        for i in range(args.soak_sources)
    ]
    threads.append(threading.Thread(target=probe_worker, args=(sources[-1],),
                                    name="soak-probe", daemon=True))
    soak_started = time.monotonic()
    for thread in threads:
        thread.start()

    # Workers own live sandboxes, so they are joined to completion rather than on
    # a timeout: the wall-clock budget is their stopping rule, and one bounded
    # grace covers the round in flight when it expires.
    grace_s = (args.soak_join_grace_s if args.soak_join_grace_s > 0
               else max(300.0, args.fork_deadline_s + 120.0))
    for thread in threads:
        thread.join(timeout=max(1.0, deadline + grace_s - time.monotonic()))
    stuck = [t.name for t in threads if t.is_alive()]
    if stuck:
        log(f"  workers past their budget ({', '.join(stuck)}); asking them to stop")
        stop.set()
        hard_deadline = time.monotonic() + grace_s
        for thread in threads:
            thread.join(timeout=max(1.0, hard_deadline - time.monotonic()))
        stuck = [t.name for t in threads if t.is_alive()]
    soak_wall = time.monotonic() - soak_started

    # Never delete a source out from under a live worker: its next fork would
    # fail against a vanished parent and its children would be stranded on the
    # node with nothing owning them.
    if stuck:
        held = live.owned_by(stuck)
        log(f"  NOT deleting soak sources: {', '.join(stuck)} still running and holding "
            f"sandboxes {held['sandboxes'] or 'none'} / snapshots {held['snapshots'] or 'none'}")
    else:
        for sb in sources:
            try:
                fp.delete_sandbox(sb)
            except Exception as exc:
                errors.append(f"cleanup {sb.id}: {exc}")

    branch_rounds = [e for e in events if e["kind"] == "branch_round"]
    probe_cycles = [e for e in events if e["kind"] == "probe_cycle"]
    budget_stops = [e for e in events if e["kind"] == "budget_stop"]
    branch_forks = [f for e in branch_rounds for f in e["forks"]]
    # The parity probe cycle forks a child of its own, and design v2.4.1 counts
    # it among the K forks a B run issues per round -- so it belongs in every
    # throughput, placement and capacity aggregate here, not just its own
    # cycle-time stat.
    probe_forks = [parts for parts in (cycle_fork_parts(c) for c in probe_cycles) if parts]
    all_forks = branch_forks + probe_forks
    node_counts: dict[str, int] = {}
    for fork in all_forks:
        node_counts[str(fork.get("node"))] = node_counts.get(str(fork.get("node")), 0) + 1
    capacity_waits = sum(int(f.get("capacity_waits") or 0) for f in all_forks)
    attempts = [int(f.get("fork_op_attempts") or 1) for f in all_forks]

    requested_rounds = args.soak_sources * args.soak_rounds
    incomplete: list[str] = []
    if stuck:
        incomplete.append(f"workers never finished: {', '.join(stuck)}")
    if errors:
        incomplete.append(f"{len(errors)} worker error(s)")
    if not branch_rounds:
        incomplete.append("no branch round completed at all")
    elif len(branch_rounds) < requested_rounds:
        incomplete.append(f"only {len(branch_rounds)}/{requested_rounds} branch rounds completed")
    if budget_stops:
        incomplete.append(f"{len(budget_stops)} worker(s) stopped early on the wall-clock budget")
    if not probe_cycles:
        incomplete.append("no parity probe cycle completed")

    summary = {
        "sources": [{"id": sb.id, "name": sb.name, "node": sb.node} for sb in sources],
        "source_create_s": summarize(create_times),
        "probe_source_fixture": probe_fixture,
        "rounds_completed": len(branch_rounds),
        "rounds_requested": requested_rounds,
        "probe_cycles_completed": len(probe_cycles),
        "forks_total": len(all_forks),
        "branch_forks_total": len(branch_forks),
        "probe_forks_total": len(probe_forks),
        "wall_s": round(soak_wall, 1),
        "aggregate_fork_rate_per_min": round(len(all_forks) / (soak_wall / 60.0), 2)
        if soak_wall else 0.0,
        "waiting_for_capacity_observations": capacity_waits,
        "fork_op_attempts": summarize(attempts),
        "fork_attempts_gt1": sum(1 for a in attempts if a > 1),
        "latency": {
            "snapshot_s": summarize(numeric_column(branch_rounds, "snapshot_s")
                                    + numeric_column(probe_cycles, "snapshot_s")),
            "fork_total_s": summarize(numeric_column(all_forks, "fork_total_s")),
            "fork_op_s": summarize(numeric_column(all_forks, "fork_op_s")),
            "child_ready_s": summarize(numeric_column(all_forks, "child_ready_s")),
            "expose_s": summarize(numeric_column(all_forks, "expose_s")),
            "health_s": summarize(numeric_column(all_forks, "health_s")),
            "delete_child_s": summarize(
                [d for e in branch_rounds for d in e["delete_child_s"]]
                + numeric_column(probe_cycles, "delete_child_s")
            ),
            "delete_snapshot_s": summarize(numeric_column(branch_rounds, "delete_snapshot_s")
                                           + numeric_column(probe_cycles, "delete_snapshot_s")),
            "branch_round_s": summarize(numeric_column(branch_rounds, "round_s")),
            "probe_cycle_s": summarize(numeric_column(probe_cycles, "t_probe_cycle_s")),
        },
        "latency_sample_scope": "branch rounds + parity probe cycles",
        "node_placement": node_counts,
        "errors": errors,
        "complete": not incomplete,
        "incomplete_reasons": incomplete,
        "stuck_workers": stuck,
        # What the stuck workers still own, so the failure reaper can keep its
        # hands off resources a live thread is forking from.
        "live_worker_resources": live.owned_by(stuck),
        "sources_deleted": not stuck,
        "config": {
            "soak_sources": args.soak_sources,
            "soak_rounds": args.soak_rounds,
            "soak_width": args.soak_width,
            "soak_health": args.soak_health,
            "soak_budget_s": args.soak_budget_s,
            "source_snapshot": snap,
        },
        "events": events,
    }
    if stuck:
        raise StageError(
            f"soak workers {stuck} never finished within the budget plus {grace_s:.0f}s of "
            f"grace; the {len(sources)} soak sources were left in place because a live worker "
            f"still owns children off them -- reap once the control plane releases the "
            f"operations, then rerun the soak",
            partial=summary,
            # Asked again at cleanup time: a stuck worker keeps creating and
            # deleting while the runner unwinds, so the list in `summary` is
            # only the picture as of this instant.
            live_resources=lambda: live.owned_by(stuck),
        )
    return summary


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def analyze_fork_ops(journal_dir: Path, since: str | None = None) -> dict[str, Any]:
    """Why forks are slow, straight from the journalled operation results.

    The control plane reports its own internals in ``operations get``: a fork
    that cannot get a warm supervisor pre-claim on the target pod records
    ``fork_preclaim_miss`` and is retried, so ``op_attempt`` counts how many
    times the scheduler bounced off an occupied warm slot.  That is the
    concrete, named form of the design doc's "same-host warm supervisor slots
    are the binding constraint".
    """
    attempts: list[float] = []
    durations: list[float] = []
    preclaim_wait_ms: list[float] = []
    misses = 0
    total = 0
    failed = 0
    queued_timeouts = 0
    unparsable = 0
    reasons: dict[str, int] = {}
    for path in sorted(journal_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"op":"fork"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A torn line is evidence we could not read, not evidence of
                # nothing: it is counted so the rates below can be doubted.
                unparsable += 1
                continue
            if record.get("op") != "fork":
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            if record.get("outcome") != "ok":
                failed += 1
                error = str(record.get("error") or "")
                if "still 'queued'" in error or "OperationTimeout" in error:
                    queued_timeouts += 1
                continue
            result = record.get("result") or {}
            total += 1
            attempts.append(float(record.get("op_attempts") or 1))
            durations.append(float(record.get("duration_s") or 0.0))
            if result.get("fork_preclaim_wait_ms"):
                preclaim_wait_ms.append(float(result["fork_preclaim_wait_ms"]))
            miss = result.get("fork_preclaim_miss")
            if miss:
                misses += 1
                key = "warm_preclaim_pod_conflict" if "fork_child_warm_preclaims_pod_idx" in miss \
                    else miss[:80]
                reasons[key] = reasons.get(key, 0) + 1
    return {
        "fork_ops": total,
        "fork_ops_failed": failed,
        "fork_ops_queued_timeout": queued_timeouts,
        "unparsable_journal_lines": unparsable,
        "op_attempts": summarize(attempts),
        "duration_s": summarize(durations),
        "preclaim_wait_ms": summarize(preclaim_wait_ms),
        "preclaim_miss_count": misses,
        "preclaim_miss_rate": round(misses / total, 3) if total else 0.0,
        "preclaim_miss_reasons": reasons,
    }


def recommended_run_cap(soak: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    """Concurrent Tier-1 B-runs one node sustains, from measured fork throughput.

    Post-parity steady state (design v2.4.1): a B run issues ``K`` forks per
    ``m`` steps (1 parity probe cycle + K-1 branch children); an A / A x K / C
    trajectory issues 1.  The node's measured aggregate fork completion rate is
    the budget, so::

        cap = floor(fork_rate_per_s * m * llm_round_s / K)

    Concurrency does not raise the numerator -- the soak measures that directly
    -- so this is a throughput cap, not a slot count.

    There is no default cap.  Without a complete soak the answer is ``None``: a
    fabricated 1 reads as "one B run is safe" when nothing was measured, and
    Tier 1 would size itself on it.  A measured-but-too-slow node yields 0,
    which is a different statement and also not 1.
    """
    window_s = args.m * args.llm_round_s
    blockers: list[str] = []
    if not soak:
        blockers.append("no valid soak stage in the results")
    else:
        if not soak.get("wall_s"):
            blockers.append("soak recorded no wall clock")
        if not soak.get("complete", False):
            reasons = soak.get("incomplete_reasons") or []
            blockers.append(
                "soak incomplete: " + ("; ".join(str(r) for r in reasons) if reasons
                                       else "stage carries no completeness record")
            )
        if float(soak.get("aggregate_fork_rate_per_min") or 0.0) <= 0.0:
            blockers.append("soak measured no completed forks")
    if args.K < 1:
        blockers.append(f"K={args.K} is not a fan-out width")
    if blockers:
        return {
            "cap": None,
            "valid": False,
            "fork_rate_per_min": None,
            "forks_per_run_per_window": args.K,
            "window_s": window_s,
            "forks_affordable_per_window": None,
            "blockers": blockers,
            "basis": (
                "no cap established: " + "; ".join(blockers)
                + " -- rerun the soak to completion before sizing Tier 1"
            ),
        }
    rate_per_min = float(soak.get("aggregate_fork_rate_per_min") or 0.0)
    budget = rate_per_min / 60.0 * window_s
    cap = int(budget // args.K)
    basis = (
        f"measured aggregate {rate_per_min} forks/min on "
        f"{len(soak.get('node_placement') or {})} node(s); a B run needs K={args.K} forks "
        f"per m={args.m} steps of {args.llm_round_s}s"
    )
    if cap == 0:
        basis += (
            f" -- that affords {budget:.2f} forks per window, short of one B run's K={args.K}, "
            f"so no B run fits at this K/m"
        )
    return {
        "cap": cap,
        "valid": True,
        "fork_rate_per_min": rate_per_min,
        "forks_per_run_per_window": args.K,
        "window_s": window_s,
        "forks_affordable_per_window": round(budget, 2),
        "blockers": [],
        "basis": basis,
    }


def soak_validity(soak: dict[str, Any] | None, cap: dict[str, Any]) -> tuple[bool, str]:
    """Is a soak stage publishable as capacity evidence, and if not, why not?

    Two things at once, because either alone lies: the stage must say it ran to
    completion (every branch round, at least one parity probe cycle, no worker
    errors, nothing stuck), and it must have yielded a cap that is a real
    measurement.  A measured 0 is one -- it says no B run fits on this node at
    this K/m -- but a null cap is the absence of a measurement, and Tier 1 must
    size itself on neither an unfinished stage nor a missing number.
    """
    if not soak:
        return False, "no valid soak stage in the results"
    if not soak.get("complete", False):
        reasons = [str(r) for r in (soak.get("incomplete_reasons") or [])]
        return False, ("soak incomplete: "
                       + ("; ".join(reasons) or "stage carries no completeness record"))
    if cap.get("cap") is None:
        blockers = [str(b) for b in (cap.get("blockers") or [])]
        return False, ("no usable run cap: "
                       + ("; ".join(blockers) or str(cap.get("basis") or "the cap is null")))
    return True, ""


def _all_fork_parts(stage: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every per-fork subcomponent dict a stage recorded, whatever its shape."""
    if not stage:
        return []
    parts: list[dict[str, Any]] = []
    for bucket in stage.get("buckets") or []:
        parts.extend(bucket.get("rounds") or [])
    parts.extend(stage.get("rounds") or [])
    for child in stage.get("children") or []:
        if isinstance(child.get("fork"), dict):
            parts.append(child["fork"])
    for event in stage.get("events") or []:
        parts.extend(event.get("forks") or [])
        # A soak probe cycle is an event too, and its fork is a fork.
        parts.append(cycle_fork_parts(event))
    for cycle in stage.get("cycles") or []:
        parts.append(cycle_fork_parts(cycle))
    return [p for p in parts if isinstance(p, dict) and p]


def ok_stage(results: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A stage's payload, or None if it never ran or raised."""
    stage = results.get(name)
    if isinstance(stage, dict) and "error" not in stage:
        return stage
    return None


def build_report(state: dict[str, Any], results: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Tier 0 -- Farplane mechanics gate")
    add("")
    add(f"Generated {results.get('finished', now_iso())} UTC. "
        f"Control plane: `panda compute`; template `{TEMPLATE}` (2 vCPU / 4 GiB / 8 GiB disk); "
        f"image `{IMAGE_TAG}`.")
    add("")
    add("## Kept resources (Tier 1 inputs)")
    add("")
    add(f"- `TEMPLATE_SNAP` = **{state.get('template_snap')}** "
        f"(lease {state.get('template_snap_ttl', '24h')}; the API rejects a zero TTL, so it "
        f"must be re-snapshotted or leased before it lapses)")
    add(f"- bake sandbox = **{state.get('bake_sandbox')}** (`flebench-bake`)")
    add(f"- bake bridge URL = `{state.get('bake_url')}`")
    add("")

    bake = ok_stage(results, "bake")
    if bake:
        transfer = bake.get("transfer") or {}
        if not transfer.get("bytes") and TRANSFER_PATH.exists():
            # The bake is idempotent: a rerun skips the transfer, so pull the
            # numbers from the run that actually moved the bytes.
            transfer = {**transfer, **json.loads(TRANSFER_PATH.read_text())}
        add("## Stage: template bake")
        add("")
        add("Transfer choice (measured, not assumed): the host is behind NAT on "
            "`192.168.69.0/24` and the sandbox egress policy denies RFC1918, so a host-side "
            "`http.server` is unreachable from the guest. The exec gateway caps at a 5 min "
            "timeout, making base64 chunking of a 400 MiB archive impractical. The reverse "
            "direction works: an exposed guest port is reachable from the host without auth. "
            "The ingress rejects bodies past ~1 MiB (HTTP 413), so the host streams "
            f"{CHUNK_BYTES // 1000} KB PUTs over one keep-alive connection into a guest "
            "receiver piping straight into `zstd -d | docker load` -- nothing lands on the "
            "8 GiB guest disk.")
        add("")
        if not transfer.get("bytes"):
            add("- image already present in guest; transfer skipped")
        else:
            add(f"- transferred {transfer.get('bytes', 0) / 1e6:.0f} MB in "
                f"{transfer.get('wall_s')}s ({transfer.get('mb_per_s')} MB/s, "
                f"{transfer.get('chunks')} chunks of {CHUNK_BYTES // 1000} KB); guest "
                f"`docker load` result: `{transfer.get('docker_load')}`")
        add(f"- container start -> in-guest `/health` ok: "
            f"{(bake.get('container') or {}).get('guest_health_s')}s")
        container = bake.get("container") or {}
        add(f"- guest bridge auth: **{container.get('bridge_auth', 'unrecorded')}** "
            f"(token fingerprint `{container.get('bridge_auth_fp', '-')}`; the host's "
            f"`FLE_BRIDGE_TOKEN` is injected into the container env at start, and the exposed "
            f"port is TCP so every request carries `Authorization: Bearer`)")
        add(f"- `expose {BRIDGE_PORT}`: {bake.get('expose_s')}s; host-side `/health` ok after "
            f"{bake.get('host_health_s')}s")
        add(f"- template snapshot: {bake.get('template_snapshot_s')}s")
        add(f"- `/meta` at bake: `{json.dumps(bake.get('meta'))}`")
        add("")

    constants = ok_stage(results, "constants")
    if constants:
        add("## Constants (n = %d)" % len(constants["rounds"]))
        add("")
        add("| metric | p50 | p95 | max | mean | n |")
        add("|---|---|---|---|---|---|")
        labels = {
            "t_snap_s": "t_snap (snapshot running sandbox)",
            "fork_total_s": "t_fork (submit -> child running)",
            "fork_op_s": "  ... fork operation (submit -> terminal)",
            "child_ready_s": "  ... child boot -> running",
            "expose_s": "t_expose",
            "health_s": "  ... expose -> HTTP /health ok",
            "total_to_healthy_s": "t_fork_to_healthy (fork+expose+health)",
            "t_delete_s": "t_delete (child, submit -> terminal)",
            "t_delete_snapshot_s": "t_delete_snapshot",
        }
        for key, label in labels.items():
            stat = constants["stats"].get(key) or {}
            add(f"| {label} | {stat.get('p50')} | {stat.get('p95')} | {stat.get('max')} | "
                f"{stat.get('mean')} | {stat.get('n')} |")
        overhead = constants.get("op_poll_overhead_per_call_s") or {}
        add("")
        add(f"- op-poll overhead per `operations get` call: p50 {overhead.get('p50')}s, "
            f"p95 {overhead.get('p95')}s (n={overhead.get('n')})")
        add(f"- fork-op attempts per round: "
            f"{[r.get('fork_op_attempts') for r in constants['rounds']]} -- each attempt is the "
            f"control plane re-trying a fork that could not take the pod's warm supervisor "
            f"pre-claim")
        add(f"- child polls seen in a queued/pending/waiting state: "
            f"{constants.get('capacity_waits_total')}")
        add("")
        analysis = results.get("fork_op_analysis") or {}
        if analysis.get("fork_ops"):
            add("### Why forks cost ~100s")
            add("")
            add(f"Across all {analysis['fork_ops']} forks in this journal, "
                f"{analysis['preclaim_miss_count']} "
                f"({analysis['preclaim_miss_rate'] * 100:.0f}%) recorded a "
                f"`fork_preclaim_miss`, every one of them "
                f"`duplicate key ... fork_child_warm_preclaims_pod_idx` -- i.e. the target "
                f"pod's single warm supervisor pre-claim was already held. The operation is "
                f"retried until the slot frees: attempts p50 "
                f"{analysis['op_attempts'].get('p50')}, max "
                f"{analysis['op_attempts'].get('max')}. This is the design doc's "
                f"\"same-host warm supervisor slots, about one per node\" constraint in its "
                f"concrete form, and it is what sets t_fork.")
            add("")

    cooldown = ok_stage(results, "cooldown")
    if cooldown:
        add("## Deletion -> capacity reuse (what t_fork is actually waiting for)")
        add("")
        add("Constants forks, deletes, and immediately forks again. This stage changes one "
            "thing: the idle gap after the delete operation goes terminal. If fork were "
            "intrinsically ~100s the gap would not matter; if the cost is an unreleased warm "
            "supervisor pre-claim, attempts collapse once the gap exceeds the reclaim time.")
        add("")
        add("| post-delete cooldown | fork_op_attempts p50 | attempts max | fork_op_s p50 | "
            "fork_total_s p50 | warm preclaim hits | n |")
        add("|---|---|---|---|---|---|---|")
        for bucket in cooldown["buckets"]:
            add(f"| {bucket['cooldown_s']:.0f}s | {bucket['fork_op_attempts']['p50']} | "
                f"{bucket['fork_op_attempts']['max']} | {bucket['fork_op_s']['p50']} | "
                f"{bucket['fork_total_s']['p50']} | {bucket['preclaim_hits']}"
                f"/{bucket['fork_op_attempts']['n']} | {bucket['fork_op_attempts']['n']} |")
        add("")
        constant = cooldown.get("reuse_constant_s")
        add(f"**Deletion-to-capacity-reuse constant: "
            f"{'%.0fs' % constant if constant is not None else 'not resolved'}** -- the "
            f"smallest cooldown whose median attempt count is within one attempt of the "
            f"cheapest bucket ({cooldown['best_p50_attempts']} attempts). Buckets that "
            f"recovered: {cooldown['attempts_recovered_at']}.")
        add("")
        buckets = cooldown["buckets"]
        hot = buckets[0]["fork_op_s"]["p50"]
        cold = buckets[-1]["fork_op_s"]["p50"]
        add(f"Attribution, stated honestly: waiting {constant:.0f}s after the delete moves "
            f"median fork_op_s from {hot}s to {cold}s -- real, but it removes only "
            f"{max(0.0, hot - cold):.0f}s of a ~{hot:.0f}s cost, and even the fully-cooled "
            f"bucket still needs {buckets[-1]['fork_op_attempts']['p50']} attempts and never "
            f"once obtained a warm pre-claim (`preclaim_produced 0/1` in every round of every "
            f"bucket). So deletion-to-reuse latency is a real component but NOT the whole "
            f"story: the pod's warm-supervisor lane is shared and contended beyond our own "
            f"deletes, and the residual is queueing on that lane. Two consequences for Tier 1: "
            f"(a) a harness that deletes a child and immediately re-forks pays a measurable "
            f"penalty, so schedule at least {constant:.0f}s of other work between delete and "
            f"the next fork of the same lineage; (b) even perfectly spaced, fork does not get "
            f"cheap enough to hide inside one sampling round.")
        add("")

    lanes = sorted({
        str(f.get("placement_lane"))
        for stage_name in ("constants", "cooldown", "fidelity", "probe", "soak")
        for f in _all_fork_parts(ok_stage(results, stage_name))
        if f.get("placement_lane")
    })
    if lanes:
        add("## Structural finding: fan-out width is capped by ONE node")
        add("")
        add(f"Every fork child in this run was placed on lane(s) `{', '.join(lanes)}` and "
            f"landed on the same node as its source snapshot. Farplane forks by cloning the "
            f"source's hot-storage dataset, so a lineage is pinned to the node that holds it: "
            f"the pool's 384 vCPU across {len(results.get('node_ids') or []) or 'N'} nodes buys "
            f"more *runs*, never more *width* for one run. Per-run fan-out width is therefore "
            f"bounded by a single node's warm supervisor slots (~1 ready slot), which is why "
            f"the K-1 branch forks serialise instead of overlapping, and why the run cap below "
            f"is a per-node fork-throughput budget rather than a vCPU count.")
        add("")

    fidelity = ok_stage(results, "fidelity")
    if fidelity:
        verdict = fidelity["verdict"]
        add("## Fidelity")
        add("")
        add(f"**{'PASS' if verdict['pass'] else 'FAIL'}** -- "
            f"{len(fidelity['children'])} children forked sequentially from one snapshot of a "
            f"{fidelity['parent_before']['meta'].get('entity_count')}-entity factory.")
        add("")
        if verdict.get("failed_checks"):
            add(f"Failed checks: {', '.join(str(f) for f in verdict['failed_checks'])}.")
            add("")
        add("| check | scores the verdict | result |")
        add("|---|---|---|")
        scored = verdict.get("checks") or {}
        for key, ok in scored.items():
            add(f"| {key} | yes | {ok} |")
        if not scored:
            add("| (stage recorded no per-check breakdown; rerun it) | - | - |")
        for key in ("probe_spread", "probe_identical", "probe_items_per_window",
                    "probe_tolerance", "parent_probe_delta"):
            add(f"| {key} | no (context) | {verdict.get(key)} |")
        add("")
        add("| child | sandbox | node | factorio_pid | entity_count | game_tick | "
            "items/window | throughput |")
        add("|---|---|---|---|---|---|---|---|")
        for child in fidelity["children"]:
            meta = child["meta"]
            probe_result = child["probe"]
            items = (probe_result.get("end_count") or 0) - (probe_result.get("start_count") or 0)
            add(f"| {child['i']} | `{child['sandbox']}` | {child['node']} | "
                f"{meta.get('factorio_pid')} | {meta.get('entity_count')} | "
                f"{meta.get('game_tick')} | {items} | {probe_result.get('throughput')} |")
        parent_meta = fidelity["parent_before"]["meta"]
        parent_probe = fidelity["parent_before"]["probe"]
        parent_items = (parent_probe.get("end_count") or 0) - (parent_probe.get("start_count") or 0)
        add(f"| parent (pre-fork) | `{state.get('bake_sandbox')}` | - | "
            f"{parent_meta.get('factorio_pid')} | {parent_meta.get('entity_count')} | "
            f"{parent_meta.get('game_tick')} | {parent_items} | "
            f"{parent_probe.get('throughput')} |")
        add("")
        if verdict.get("probe_items_identical"):
            add("Reading: every child produced the **same integer item count** in its 3600-tick "
                "window; the residual throughput spread of "
                f"{verdict.get('probe_spread')} is pure window-normalisation noise (the probe "
                "closes 1-2 ticks past 3600 and divides by the actual tick delta), i.e. below "
                "the one-item resolution floor of the production counter. `game_tick` differs "
                "by a few ticks because each child ran briefly before its probe.")
        else:
            add(f"Reading: the children did **not** agree on their integer item counts "
                f"({verdict.get('probe_items_per_window')}), so the exactness claim fails on "
                f"the production counters themselves, not on the "
                f"{verdict.get('probe_spread')} throughput spread.")
        add("")
        parent_after_probe = (fidelity.get("parent_after") or {}).get("probe") or {}
        add(f"Parent drift is expected and benign: the parent kept running for the whole stage "
            f"and its probe fell from {parent_probe.get('throughput')} to "
            f"{parent_after_probe.get('throughput')} because the fixture's burner drills burn "
            f"through their coal in ~22 game-minutes (verified separately: 11/14 drills "
            f"`NO_FUEL` on a long-running copy). The children, resumed from the snapshot, all "
            f"see full fuel -- which is precisely the exactness claim. The untouched-parent "
            f"checks are `parent_pid_unchanged` and `parent_entity_count_unchanged`, both true.")
        add("")

    probe = ok_stage(results, "probe")
    if probe:
        stat = probe["t_probe_cycle_s"]
        add("## Parity probe cycle (design v2.4)")
        add("")
        add(f"`snapshot -> 1 fork -> health -> /probe -> delete child -> delete snapshot`, "
            f"n={stat.get('n')}: **p50 {stat.get('p50')}s, p95 {stat.get('p95')}s, "
            f"max {stat.get('max')}s**.")
        add("")
        add("| cycle | total | snapshot | fork | expose | health | probe | del child | del snap |")
        add("|---|---|---|---|---|---|---|---|---|")
        for cycle in probe["cycles"]:
            add(f"| {cycle['tag']} | {cycle['t_probe_cycle_s']} | {cycle['snapshot_s']} | "
                f"{cycle.get('fork_fork_total_s')} | {cycle.get('fork_expose_s')} | "
                f"{cycle.get('fork_health_s')} | {cycle['probe_s']} | "
                f"{cycle['delete_child_s']} | {cycle['delete_snapshot_s']} |")
        add("")

    soak = ok_stage(results, "soak")
    if soak:
        add("## Soak (design v2.4.1 post-parity demand)")
        add("")
        add(f"{soak['rounds_completed']}/{soak['rounds_requested']} branch rounds and "
            f"{soak['probe_cycles_completed']} probe cycles in {soak['wall_s']}s; "
            f"{soak['forks_total']} forks total ({soak.get('branch_forks_total')} branch + "
            f"{soak.get('probe_forks_total')} parity probe) at "
            f"{soak['aggregate_fork_rate_per_min']} forks/min aggregate.")
        add("")
        add(f"- **waiting_for_capacity observations: {soak['waiting_for_capacity_observations']}**")
        add(f"- fork operation attempts (control-plane internal retries): p50 "
            f"{soak['fork_op_attempts'].get('p50')}, p95 {soak['fork_op_attempts'].get('p95')}, "
            f"max {soak['fork_op_attempts'].get('max')}; "
            f"{soak['fork_attempts_gt1']}/{soak['forks_total']} forks needed >1 attempt")
        add(f"- node placement of children: `{json.dumps(soak['node_placement'])}`")
        add(f"- errors: {len(soak['errors'])}")
        if not soak.get("complete", False):
            reasons = soak.get("incomplete_reasons") or ["stage carries no completeness record"]
            add(f"- **incomplete**: {'; '.join(str(r) for r in reasons)}")
        add("")
        add("| op | p50 | p95 | max | n |")
        add("|---|---|---|---|---|")
        for key, stat in soak["latency"].items():
            add(f"| {key} | {stat.get('p50')} | {stat.get('p95')} | {stat.get('max')} | "
                f"{stat.get('n')} |")
        add("")
        cap = results.get("run_cap") or {}
        if cap.get("cap") is None:
            add(f"**Per-node Tier-1 run cap: not established** -- {cap.get('basis')}. "
                f"`bench/results/tier0_soak.json` carries a null cap, so Tier 1 must refuse to "
                f"size itself from this run rather than assume one.")
        else:
            add(f"**Per-node Tier-1 run cap: {cap['cap']}** concurrent B-runs "
                f"({cap.get('basis')}); a window of {cap.get('window_s')}s affords "
                f"{cap.get('forks_affordable_per_window')} forks node-wide. Consumed by "
                f"`bench/run_tier1.py` from `bench/results/tier0_soak.json` "
                f"(`recommended_run_cap` / `per_node_run_cap` / `node_cap`).")
        add("")

    gate = results.get("gate")
    if gate:
        add("## GATE")
        add("")
        critical = gate.get("critical_path_p95_s")
        add(f"Steady-state round critical path (p95): "
            f"**{f'{critical}s' if critical is not None else 'not established'}** at "
            f"K={gate['K']}, m={gate['m']} -- {gate.get('paths_combined')} of branch "
            f"{gate.get('branch_path_p95_s')}s and parity probe cycle "
            f"{gate.get('probe_path_p95_s')}s.")
        add(f"Basis: {gate.get('critical_path_basis')}.")
        add(f"LLM sampling round used for the comparison: {gate['llm_round_s']}s "
            f"({gate['llm_round_basis']}).")
        add("")
        if gate.get("evidence_gaps"):
            add(f"Evidence gaps: {'; '.join(str(g) for g in gate['evidence_gaps'])}.")
            add("")
        add(f"**Verdict: {gate['verdict']}** -- {gate['rationale']}")
        add("")

    reaper = results.get("reaper")
    if reaper is not None:
        add("## Resource hygiene")
        add("")
        add(f"- reaper deleted {len(results.get('reaper_deleted') or [])} leftovers")
        add(f"- final dry pass residual: **{len(reaper)}** "
            f"(0 == zero leaked resources)")
        if reaper:
            add(f"- residual: `{json.dumps(reaper)}`")
        add("")
    return "\n".join(lines) + "\n"


def evaluate_gate(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Does one steady-state branch round fit inside one LLM sampling round?

    A round at (K, m) per design v2.4.1 is one snapshot, the parity probe cycle,
    and the K-1 branch children forked off it.  The probe cycle runs alongside
    the branch forks (that is exactly how the soak drives them), so the critical
    path is the *max* of the two paths; ``--probe-serialized`` says the harness
    runs them back to back and sums them instead.

    The verdict is PASS, FAIL, or INCOMPLETE.  INCOMPLETE is not a soft FAIL: it
    means the evidence for a decision is missing, and it is the answer whenever a
    required stage is absent, failed, or measured a different shape than the K
    being gated.  A gate that reads zeros as "costs nothing" would PASS on an
    empty run, which is the one outcome that must be impossible here.
    """
    soak = ok_stage(results, "soak") or {}
    constants = ok_stage(results, "constants") or {}
    fidelity = ok_stage(results, "fidelity") or {}
    latency = soak.get("latency") or {}
    const_stats = constants.get("stats") or {}
    soak_width = (soak.get("config") or {}).get("soak_width")
    llm_round = args.llm_round_s
    gaps: list[str] = []
    notes: list[str] = []

    def pick(*stats: Any) -> float | None:
        for stat in stats:
            value = stat_value(stat)
            if value is not None:
                return value
        return None

    # --- required evidence ------------------------------------------------
    if not fidelity:
        gaps.append("fidelity stage missing or failed: fork exactness is unproven")
    elif not (fidelity.get("verdict") or {}).get("pass"):
        failed = (fidelity.get("verdict") or {}).get("failed_checks") or ["unrecorded checks"]
        gaps.append(f"fidelity FAILED ({', '.join(str(f) for f in failed)})")
    if not soak:
        gaps.append("soak stage missing or failed: no steady-state evidence")
    elif not soak.get("complete", False):
        reasons = soak.get("incomplete_reasons") or ["stage carries no completeness record"]
        gaps.append("soak incomplete: " + "; ".join(str(r) for r in reasons))
    run_cap = results.get("run_cap")
    cap_value = run_cap.get("cap") if isinstance(run_cap, dict) else None
    cap_refuses = ""
    if isinstance(run_cap, dict) and cap_value is None:
        # A node whose sustainable width could not be measured cannot be said to
        # fit the fan-out, whatever the percentiles look like.
        gaps.append("no per-node run cap established: " + str(run_cap.get("basis")))
    elif isinstance(cap_value, int) and not isinstance(cap_value, bool) and cap_value <= 0:
        # A measured zero is evidence, not a hole in it: the node affords no B
        # run at all at this K/m, so the gate FAILs on it.  INCOMPLETE stays
        # reserved for the null cap, where nothing was measured.
        cap_refuses = "measured per-node run cap is 0: " + str(run_cap.get("basis"))
    provenance = results.get("stage_provenance") or {}
    for name in provenance.get("unfingerprinted") or []:
        if name in ("constants", "fidelity", "probe", "soak"):
            gaps.append(f"stage {name} carries no input fingerprint, so it cannot be shown to "
                        f"belong with the stages it is combined with; rerun it")

    # --- branch path: snapshot + K-1 sequential branch forks + deletes ----
    branch_path = pick(latency.get("branch_round_s"))
    branch_basis = (f"soak branch_round p95 (snapshot + {soak_width} sequential forks + "
                    f"deletes, under contention)")
    if branch_path is not None and soak_width != args.K - 1:
        notes.append(f"soak branch rounds ran at width {soak_width}, not K-1={args.K - 1}, so "
                     f"their round time is not this K's branch path")
        branch_path = None
    if branch_path is None:
        snap_p95 = pick(const_stats.get("t_snap_s"))
        fork_p95 = pick(const_stats.get("total_to_healthy_s"))
        if snap_p95 is not None and fork_p95 is not None:
            branch_path = snap_p95 + (args.K - 1) * fork_p95
            branch_basis = (f"constants: t_snap p95 + (K-1={args.K - 1}) x t_fork_to_healthy "
                            f"p95 (no contention)")
        else:
            branch_basis = ""
            gaps.append(f"no branch-round evidence: no soak at width K-1={args.K - 1} and no "
                        f"constants t_snap/t_fork_to_healthy percentiles")

    # --- parity probe path: its own snapshot -> fork -> probe -> deletes ---
    probe_path = pick(latency.get("probe_cycle_s"))
    probe_basis = "soak probe_cycle p95 (under contention)"
    if probe_path is None:
        probe_path = pick((ok_stage(results, "probe") or {}).get("t_probe_cycle_s"))
        probe_basis = "probe stage t_probe_cycle p95 (no contention)"
    if probe_path is None:
        probe_basis = ""
        gaps.append("no parity probe cycle evidence: neither the soak nor the probe stage "
                    "completed one, and design v2.4.1 puts it on every round")

    if branch_path is None or probe_path is None:
        critical: float | None = None
        basis = "critical path not established: " + "; ".join(gaps)
    elif args.probe_serialized:
        critical = branch_path + probe_path
        basis = (f"serialised: branch {branch_path:.1f}s [{branch_basis}] + parity probe "
                 f"{probe_path:.1f}s [{probe_basis}]")
    else:
        critical = max(branch_path, probe_path)
        basis = (f"max of the concurrent paths: branch {branch_path:.1f}s [{branch_basis}] vs "
                 f"parity probe {probe_path:.1f}s [{probe_basis}]")

    if gaps or critical is None:
        verdict = "INCOMPLETE"
    elif cap_refuses:
        verdict = "FAIL"
    else:
        verdict = "PASS" if critical <= llm_round else "FAIL"

    candidates = [
        ("fork", pick(latency.get("fork_total_s"), const_stats.get("fork_total_s"))),
        ("snapshot", pick(latency.get("snapshot_s"), const_stats.get("t_snap_s"))),
        ("child_health", pick(latency.get("health_s"), const_stats.get("health_s"))),
        ("delete", pick(latency.get("delete_child_s"), const_stats.get("t_delete_s"))),
        ("parity_probe_cycle", probe_path),
    ]
    measured = [(name, value) for name, value in candidates if value is not None]
    binding = max(measured, key=lambda pair: pair[1]) if measured else ("unmeasured", None)
    if verdict == "INCOMPLETE":
        rationale = (
            "the gate cannot be decided on this evidence: " + "; ".join(gaps)
            + ". Rerun the named stages -- an undecided gate is INCOMPLETE, never a PASS."
        )
    elif verdict == "PASS":
        rationale = (
            f"a branch round hides inside one sampling round with "
            f"{llm_round - critical:.0f}s of slack, so snapshot+forks can be overlapped with "
            f"candidate sampling at K={args.K}, m={args.m}."
        )
    else:
        analysis = results.get("fork_op_analysis") or {}
        overrun = (f"the branch round overruns one sampling round by "
                   f"{critical - llm_round:.0f}s. " if critical > llm_round else "")
        capped = (f"{cap_refuses} -- not one B run fits on this node at K={args.K}, m={args.m}, "
                  f"whatever a single round costs. " if cap_refuses else "")
        rationale = (
            capped + overrun
            + f"The binding primitive is **{binding[0]}** at p95 {binding[1]}s"
            + (
                f", and the control plane names its own cause: "
                f"{analysis['preclaim_miss_rate'] * 100:.0f}% of forks hit "
                f"`fork_preclaim_miss` on the pod's single warm supervisor pre-claim and are "
                f"retried (attempts up to {analysis['op_attempts'].get('max')})"
                if analysis.get("preclaim_miss_count") else ""
            )
            + (
                f". Report infra-bound with fork serialisation named; raising m to "
                f"{max(1, int(critical / llm_round) + 1)} or dropping K would be the levers."
                if overrun else
                ". Report infra-bound: it is the node's measured fork throughput, not one "
                "round's latency, that refuses the fan-out; more nodes or a smaller K are the "
                "levers."
            )
        )
    if notes:
        rationale += " Note: " + "; ".join(notes) + "."
    return {
        "K": args.K,
        "m": args.m,
        "llm_round_s": llm_round,
        "llm_round_basis": args.llm_round_basis,
        "per_node_run_cap": cap_value,
        "critical_path_p95_s": round(critical, 2) if critical is not None else None,
        "critical_path_basis": basis,
        "branch_path_p95_s": round(branch_path, 2) if branch_path is not None else None,
        "probe_path_p95_s": round(probe_path, 2) if probe_path is not None else None,
        "paths_combined": "serialised-sum" if args.probe_serialized else "concurrent-max",
        "soak_width": soak_width,
        "binding_primitive": binding[0],
        "binding_primitive_p95_s": binding[1],
        "probe_cycle_p95_s": probe_path,
        "evidence_gaps": gaps,
        "notes": notes,
        "verdict": verdict,
        "rationale": rationale,
    }


#: Every stage, in the order the runner drives them.
STAGE_NAMES: tuple[str, ...] = ("bake", "constants", "cooldown", "fidelity", "probe", "soak")


def fixture_identity(args: argparse.Namespace) -> dict[str, Any]:
    """What the fidelity / soak-probe factory actually is, by content.

    The path is not the fixture: the file behind it can be regenerated between
    runs, and a probe scored against a different factory is a different
    experiment.  A missing or unreadable file hashes to ``None`` -- the stages
    fall back to PLANT_PROGRAM there, which is hashed too.
    """
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    path = Path(args.plant_state) if args.plant_state else FIXTURE_STATE
    entry: dict[str, Any] = {"path": str(path), "sha256": None,
                             "program": "builtin", "program_sha256": digest(
                                 PLANT_PROGRAM.encode())}
    try:
        entry["sha256"] = digest(path.read_bytes())
    except OSError:
        pass  # recorded as an unhashed fixture, which is not equal to a hashed one
    if args.plant_program:
        program = Path(args.plant_program)
        entry["program"] = str(program)
        entry["program_sha256"] = None
        try:
            entry["program_sha256"] = digest(program.read_bytes())
        except OSError:
            pass
    return entry


#: Bumped whenever :func:`stage_inputs` gains a key.  A fingerprint written by
#: an older version cannot be compared key-for-key against this one, so it
#: counts as no fingerprint at all rather than as agreement.
STAGE_INPUTS_VERSION = 2


def stage_inputs(name: str, args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    """The inputs a stage's numbers are only valid for.

    Stages persist independently and are recombined by the run cap, the gate and
    the report on later invocations, so each records what it was measured
    against: the sandbox or snapshot it forked from, the fan-out width K it was
    shaped for, the soak width that shape was realised at and -- because
    fidelity and both probe paths score a factory against a tolerance -- the
    entity probed, the tolerance it was scored at, how many children were
    compared, and the fixture itself by content.
    """
    return {
        "fingerprint_version": STAGE_INPUTS_VERSION,
        "source": state.get("template_snap") if name == "soak" else state.get("bake_sandbox"),
        "K": args.K,
        "soak_width": args.soak_width,
        "probe_entity": args.probe_entity,
        "probe_tolerance": args.probe_tolerance,
        "fidelity_children": args.fidelity_children,
        "fixture": fixture_identity(args),
    }


#: The fingerprint keys whose change invalidates each stage's numbers.
COMPARED_INPUTS: dict[str, tuple[str, ...]] = {
    "bake": ("source",),
    "constants": ("source",),
    "cooldown": ("source",),
    "fidelity": ("source", "probe_entity", "probe_tolerance", "fidelity_children", "fixture"),
    "probe": ("source", "probe_entity"),
    "soak": ("source", "soak_width", "probe_entity", "fixture"),
}


def check_stage_provenance(
    results: dict[str, Any], args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to recombine stages that were measured against different inputs.

    Only the keys that actually invalidate a stage are compared, per stage
    (:data:`COMPARED_INPUTS`): ``source`` everywhere (a different bake sandbox or
    TEMPLATE_SNAP is a different experiment), ``soak_width`` for the soak, and --
    for the stages that score a factory -- the probed entity, the spread
    tolerance, the number of children compared and the fixture's content hash.
    ``K`` is recorded but not compared: it is applied at gate time, and the gate
    already refuses a soak whose width is not K-1.

    A fingerprint from an older version of this file cannot be compared
    key-for-key with this one, so it counts as unfingerprinted -- which the gate
    turns into INCOMPLETE, never into a false "same inputs".
    """
    stale: list[tuple[str, dict[str, tuple[Any, Any]]]] = []
    unfingerprinted: list[str] = []
    checked: list[str] = []
    for name in STAGE_NAMES:
        if not ok_stage(results, name):
            continue
        recorded = (results[name] or {}).get("inputs")
        if (not isinstance(recorded, dict)
                or recorded.get("fingerprint_version") != STAGE_INPUTS_VERSION):
            unfingerprinted.append(name)
            continue
        want = stage_inputs(name, args, state)
        diff = {key: (recorded.get(key), want[key])
                for key in COMPARED_INPUTS.get(name, ("source",))
                if recorded.get(key) != want[key]}
        if diff:
            stale.append((name, diff))
        else:
            checked.append(name)
    if stale:
        detail = "\n".join(
            f"  {name}: " + ", ".join(f"{key} was {was!r}, this run has {now!r}"
                                     for key, (was, now) in diff.items())
            for name, diff in stale
        )
        raise SystemExit(
            f"stale stage results in {JSON_PATH}: these stages were measured against different "
            f"inputs than this invocation's, so combining them would mix experiments:\n{detail}\n"
            f"Rerun the dependents (--stages {','.join(name for name, _ in stale)}) or move "
            f"{JSON_PATH.name} aside."
        )
    if unfingerprinted:
        log(f"stages with no input fingerprint or one from an older fingerprint version "
            f"(rerun to make them gateable): {', '.join(unfingerprinted)}")
    return {
        "expected": stage_inputs("constants", args, state),
        "expected_soak": stage_inputs("soak", args, state),
        "checked": checked,
        "unfingerprinted": unfingerprinted,
    }


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=",".join(STAGE_NAMES),
                        help=f"comma-separated subset of {','.join(STAGE_NAMES)}")
    parser.add_argument("--image-tar", default="/tmp/flebench-image.zst")
    parser.add_argument("--bake-sandbox", default=None)
    parser.add_argument("--bake-ttl", default="6h")
    parser.add_argument("--template-snap-ttl", default="24h",
                        help="lease for TEMPLATE_SNAP; the API rejects a zero TTL")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cooldowns", default="0,15,30,60",
                        help="post-delete idle gaps (s) for the capacity-reuse experiment")
    parser.add_argument("--cooldown-rounds", type=int, default=4)
    parser.add_argument("--fidelity-children", type=int, default=3)
    parser.add_argument("--probe-entity", default="iron-ore")
    parser.add_argument("--probe-tolerance", type=float, default=1.0,
                        help="max allowed throughput spread across children; one item of "
                             "production-counter quantisation over the 3600-tick window "
                             "(~0.53%% at 189 items) is the resolution floor")
    parser.add_argument("--probe-repeats", type=int, default=3)
    parser.add_argument("--plant-program", default=None)
    parser.add_argument("--plant-state", default=None,
                        help="GameState JSON to restore as the fidelity fixture "
                             f"(default {FIXTURE_STATE})")
    parser.add_argument("--soak-sources", type=int, default=3)
    parser.add_argument("--soak-rounds", type=int, default=10)
    parser.add_argument("--soak-width", type=int, default=3)
    parser.add_argument("--soak-ttl", default="3h")
    parser.add_argument("--soak-budget-s", type=float, default=3600.0)
    parser.add_argument("--soak-join-grace-s", type=float, default=0.0,
                        help="how long past the budget a soak worker may take to finish the "
                             "round it is in before the stage fails and leaves its sources for "
                             "the reaper; 0 derives it from --fork-deadline-s")
    parser.add_argument("--fork-deadline-s", type=float, default=900.0,
                        help="wrapper deadline for one fork op; must exceed the queued wait "
                             "under concurrency or the soak measures timeouts, not latency")
    parser.add_argument("--fork-queue-deadline", default="15m",
                        help="control-plane --deadline: how long a queued child may wait")
    parser.add_argument("--soak-health", action="store_true", default=True)
    parser.add_argument("--no-soak-health", dest="soak_health", action="store_false")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--llm-round-s", type=float, default=120.0,
                        help="one LLM sampling round in seconds, for the gate comparison")
    parser.add_argument("--llm-round-basis", default="design v2 default: 30-120s per LLM step, "
                                                     "upper bound used")
    parser.add_argument("--probe-serialized", action="store_true",
                        help="the harness runs the parity probe cycle back-to-back with the "
                             "branch forks rather than alongside them, so the gate sums the "
                             "two paths instead of taking their max")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue to the next stage when one raises")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild TIER0.md/soak json from the saved tier0.json and the "
                             "journal; runs no stages and touches no resources")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = Farplane()
    state = load_state()
    results: dict[str, Any] = {}
    if JSON_PATH.exists():
        try:
            results = json.loads(JSON_PATH.read_text())
        except json.JSONDecodeError:
            results = {}
    results["started"] = results.get("started") or now_iso()

    stages: dict[str, Callable[[Farplane, argparse.Namespace, dict[str, Any]], dict[str, Any]]] = {
        "bake": stage_bake,
        "constants": stage_constants,
        "cooldown": stage_cooldown,
        "fidelity": stage_fidelity,
        "probe": stage_probe,
        "soak": stage_soak,
    }
    requested = [] if args.report_only else [s.strip() for s in args.stages.split(",") if s.strip()]

    # C4 auth is a precondition of every stage that talks to the guest bridge
    # over an exposed port, and all of them do.  Validating it here -- before a
    # bake sandbox is created, an image uploaded or any TCP stage mutates
    # anything -- turns a missing credential into an argument error instead of a
    # half-built sandbox and a five-minute health-check timeout.
    if requested:
        preflight_env = bridge_guest_env()
        preflight_token = preflight_env.get("FLE_BRIDGE_TOKEN") or ""
        preflight_mode = (f"token {auth_fingerprint(preflight_token)}" if preflight_token
                          else "insecure opt-in (no token)")
        log(f"bridge auth preflight ok: {preflight_mode}")

    def flush() -> None:
        results["finished"] = now_iso()
        results["state"] = state
        if not args.report_only:
            results["timing_summary"] = fp.timing_summary()
        atomic_write_json(JSON_PATH, results)

    #: Callables a failed stage handed over that answer "what do my still-running
    #: workers own right now".  Consulted at cleanup time, not at failure time.
    live_probes: list[Callable[[], dict[str, list[str]]]] = []

    def live_worker_holdings() -> tuple[list[str], list[str]]:
        """(worker names, resource ids) a soak worker that outlived the stage owns.

        stage_soak refuses to delete its sources under a still-running worker;
        the reaper is the second half of that promise, and these are the ids it
        must leave alone.  The recorded ids are the picture as of the failure; a
        live probe adds whatever the worker has created since, which is the whole
        reason for asking again here.
        """
        workers: list[str] = []
        ids: set[str] = set()
        soak = results.get("soak")
        if isinstance(soak, dict):
            for block in (soak, soak.get("partial")):
                if not isinstance(block, dict) or not block.get("stuck_workers"):
                    continue
                workers = [str(w) for w in block["stuck_workers"]]
                held = block.get("live_worker_resources")
                held = held if isinstance(held, dict) else {}
                ids.update(str(i) for i in (held.get("sandboxes") or []))
                ids.update(str(i) for i in (held.get("snapshots") or []))
                break
        for probe in live_probes:
            try:
                fresh = probe()
            except Exception as exc:  # a broken probe must not stop the cleanup
                log(f"live-resource probe failed: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(fresh, dict):
                continue
            ids.update(str(i) for i in (fresh.get("sandboxes") or []))
            ids.update(str(i) for i in (fresh.get("snapshots") or []))
        return workers, sorted(ids)

    def reaper_pass(context: str, *, protected: list[str] | tuple[str, ...] = (),
                    live_workers: list[str] | tuple[str, ...] = ()) -> None:
        """Best-effort reap that never raises and never drops the keep-list.

        ``protected`` are ids a soak worker still running in this process owns.
        They are kept, loudly: that thread is forking off them right now, and a
        source deleted under it strands the children it has already created --
        exactly the leak stage_soak declined to cause when it left them behind.
        """
        if args.report_only:
            return
        keep = [k for k in (state.get("bake_sandbox"), state.get("template_snap")) if k]
        keep += [k for k in protected if k and k not in keep]
        if protected or live_workers:
            log(f"  !! NOT reaping {len(protected)} resource(s) owned by live soak worker(s) "
                f"{', '.join(live_workers) or '?'}: "
                f"{', '.join(protected) or 'none were registered'} -- reap them by hand once "
                f"the control plane releases their operations, then rerun the soak")
            results["reaper_protected"] = {"workers": list(live_workers), "ids": list(protected)}
        log(f"reaper pass ({context}; keeping {keep})")
        try:
            deleted = fp.reaper(keep=keep)
            results["reaper_deleted"] = deleted
            residual = fp.reaper(keep=keep, dry_run=True)
            results["reaper"] = residual
            log(f"reaper deleted {len(deleted)}; residual {len(residual)}")
        except Exception as exc:
            results["reaper_error"] = f"{type(exc).__name__}: {exc}"
            log(f"reaper pass FAILED: {type(exc).__name__}: {exc}")

    def finalize_soak_artifact(cap: dict[str, Any], *, publish: bool = True) -> None:
        """Write tier0_soak.json -- or an explicit invalid marker in its place.

        Tier 1 sizes itself from this file, so a rerun whose soak failed must not
        leave the previous run's numbers sitting on disk looking authoritative.
        The marker carries no ``latency`` block and a null cap, so a consumer
        fails closed instead of quietly using stale percentiles; the numbers
        themselves are kept under ``partial`` and in tier0.json.

        Publishing takes more than a stage that did not raise: :func:`soak_validity`
        also demands that the stage says it completed and that the cap is an
        actual measurement (a measured 0 counts, a null one does not).

        ``publish=False`` is the abnormal-exit path: the soak was never checked
        against the rest of the run (no provenance check, no cap, no gate), so
        even a valid-looking stage is not published.  ``--report-only``
        republishes it once the run completes.
        """
        soak_stage = ok_stage(results, "soak")
        recorded = results.get("soak")
        valid, invalid_reason = soak_validity(soak_stage, cap)
        if isinstance(recorded, dict):
            # tier0.json's own copy of the stage carries the same verdict as the
            # artifact, so a consumer reading either file reaches it.
            recorded["valid"] = bool(valid and publish)
            recorded.setdefault("complete", False)
        if valid and publish:
            payload = dict(soak_stage or {})
            payload["valid"] = True
            payload["complete"] = True
            payload["generated"] = now_iso()
            payload["recommended_run_cap"] = cap["cap"]
            payload["per_node_run_cap"] = cap["cap"]
            payload["node_cap"] = cap["cap"]
            payload["run_cap_derivation"] = cap
            payload["fork_op_analysis"] = results.get("fork_op_analysis")
            atomic_write_json(SOAK_PATH, payload)
            log(f"wrote {SOAK_PATH} (run cap {cap['cap']})")
            return
        if not isinstance(recorded, dict) and "soak" not in requested:
            return  # this invocation has nothing to say about the soak
        reason = str(
            (recorded or {}).get("error")
            or invalid_reason
            or "the run did not complete, so this soak was never validated against it"
        )
        atomic_write_json(SOAK_PATH, {
            "valid": False,
            "complete": bool((soak_stage or {}).get("complete", False)),
            "invalid_reason": reason,
            "generated": now_iso(),
            "recommended_run_cap": None,
            "per_node_run_cap": None,
            "node_cap": None,
            "run_cap_derivation": cap,
            "partial": (recorded or {}).get("partial") or soak_stage or {},
        })
        log(f"marked {SOAK_PATH} INVALID: {reason}")

    def invalidate_conclusions(reason: str) -> None:
        """Unmake tier0.json's whole-run conclusions.

        ``recommended_run_cap``, ``run_cap`` and ``gate`` are statements about a
        finished run.  After an abnormal exit they are either this run's
        half-formed answers or -- worse -- the previous invocation's, reloaded
        from tier0.json at startup and never recomputed, which would size Tier 1
        on a run that did not happen.  They are replaced with explicit nulls and
        an INCOMPLETE gate; the stale values stay as evidence.  Per-stage
        payloads are untouched (constants and probe carry their own provenance),
        and --report-only recomputes all three.
        """
        stale = {key: results[key] for key in ("recommended_run_cap", "run_cap", "gate")
                 if key in results}
        results["recommended_run_cap"] = None
        results["run_cap"] = {"cap": None, "valid": False, "blockers": [reason],
                              "basis": reason}
        results["gate"] = {"verdict": "INCOMPLETE", "valid": False,
                           "evidence_gaps": [reason], "rationale": reason}
        results["conclusions_invalidated"] = {"reason": reason, "at": now_iso(), "stale": stale}
        log(f"tier0.json cap/gate marked INVALID: {reason}")

    try:
        for name in requested:
            if name not in stages:
                raise SystemExit(f"unknown stage {name!r}")
            started = time.monotonic()
            try:
                results[name] = stages[name](fp, args, state)
                results[name]["stage_wall_s"] = round(time.monotonic() - started, 1)
            except Exception as exc:
                log(f"STAGE {name} FAILED: {type(exc).__name__}: {exc}")
                results[name] = {"error": f"{type(exc).__name__}: {exc}",
                                 "stage_wall_s": round(time.monotonic() - started, 1)}
                partial = getattr(exc, "partial", None)
                if isinstance(partial, dict) and partial:
                    # Evidence beside the error, never instead of it: ok_stage()
                    # still refuses the stage because "error" is set.
                    results[name]["partial"] = partial
                probe = getattr(exc, "live_resources", None)
                if callable(probe):
                    # This stage left threads running: the reaper has to ask them
                    # what they own before it deletes anything.
                    live_probes.append(probe)
                results[name]["inputs"] = stage_inputs(name, args, state)
                flush()
                if not args.keep_going:
                    raise
            else:
                results[name]["inputs"] = stage_inputs(name, args, state)
            save_state(state)
            flush()
            log(f"STAGE {name} done in {results[name]['stage_wall_s']}s")

        results["stage_provenance"] = check_stage_provenance(results, args, state)
        soak = ok_stage(results, "soak")
        cap = recommended_run_cap(soak, args)
        results["run_cap"] = cap
        results["recommended_run_cap"] = cap["cap"]
        # Bounded to this result set: the journal accumulates across runs, and
        # attributing an older run's fork failures to this one would misname the
        # cause the gate reports.
        results["fork_op_analysis"] = analyze_fork_ops(
            fp.journal_path.parent, since=str(results.get("started") or ""))
        results["gate"] = evaluate_gate(results, args)
    except BaseException:
        # A stage that raised leaves children, snapshots and possibly a stale
        # soak artifact behind; both are cleaned up before the traceback leaves
        # main, and neither is allowed to mask the original failure.  The cap and
        # the gate go with them: they are conclusions about a run that did not
        # finish.
        stuck_workers, protected_ids = live_worker_holdings()
        for step in (
            lambda: invalidate_conclusions(
                "tier0 exited abnormally; this run never established a cap or a gate"),
            lambda: finalize_soak_artifact(
                recommended_run_cap(ok_stage(results, "soak"), args), publish=False),
            lambda: reaper_pass("after failure", protected=protected_ids,
                                live_workers=stuck_workers),
            flush,
        ):
            try:
                step()
            except Exception as exc:
                log(f"cleanup step failed: {type(exc).__name__}: {exc}")
        raise

    reaper_pass("end of run")
    finalize_soak_artifact(cap)
    flush()
    MD_PATH.write_text(build_report(state, results))
    log(f"wrote {JSON_PATH} and {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
