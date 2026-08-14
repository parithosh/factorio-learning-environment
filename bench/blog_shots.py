"""Render blog-quality PNGs from ``/state-save`` blobs, fully offline.

Zero-contamination pipeline: benchmark runs persist state blobs when
``FLE_BENCH_STATE_DUMPS=1`` (see :meth:`ArmRun._dump_state` in
``bench/arms.py``) -- arm C rounds land as ``rN.state.json`` for free, and
every arm saves ``final-<seat>.state.json`` per node after ``timings.stop()``.
This script restores each blob into an idle LOCAL cluster container
(``fle cluster start``) and renders with the repo renderer, so the measured
run is never touched.

Prerequisites (one-time):
  - ``fle cluster start``            -- local Factorio containers
  - ``fle sprites``                  -- sprite pack in ``.fle/sprites``
    (entity art additionally needs the ``basisu`` CLI on PATH; without it
    machines silently render as nothing -- see beads issue agent-2i6)

Usage:
  uv run python -m bench.blog_shots bench/journal/exp3/<run>.states/*.state.json
  uv run python -m bench.blog_shots --close --out blog/ final-main.state.json
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

CLOSE_RADIUS = 14


def load_state(path: Path) -> str:
    """Return the raw GameState string from a dump file.

    Dumps written by ``ArmRun._dump_state`` are the verbatim ``/state-save``
    payload; fixtures occasionally wrap it as ``{"state": "..."}``.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict) and isinstance(obj.get("state"), str):
        return obj["state"]
    return raw


def make_instance(address: str | None, tcp_port: int | None):
    from fle.commons.cluster_ips import get_local_container_ips
    from fle.env import FactorioInstance

    if address is None or tcp_port is None:
        ips, _udp, tcp_ports = get_local_container_ips()
        if not ips:
            sys.exit("no local cluster containers found -- run `fle cluster start`")
        # Last container by convention: least likely to be in use.
        address, tcp_port = address or ips[-1], tcp_port or tcp_ports[-1]
    return FactorioInstance(
        address=address, tcp_port=tcp_port, fast=True, cache_scripts=True,
        inventory={}, all_technologies_researched=True,
    )


def render_one(instance, state: str, out_stem: Path, *, close: bool) -> list[Path]:
    from fle.commons.models.game_state import GameState
    from fle.env.entities import Position

    instance.reset(GameState.parse_raw(state))
    ns = instance.namespaces[0]

    entities = ns.get_entities(radius=1000)
    if entities:
        xs = [e.position.x for e in entities]
        ys = [e.position.y for e in entities]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        # Frame the factory bbox with a margin; the renderer draws a square.
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        radius = min(64, max(12, math.ceil(span / 2) + 6))
    else:
        cx, cy, radius = 0.0, 0.0, 32

    written: list[Path] = []
    shots = [("wide", radius)] + ([("close", CLOSE_RADIUS)] if close else [])
    for suffix, r in shots:
        img = ns._render(position=Position(x=cx, y=cy), radius=r,
                         max_render_radius=max(1, r - 2), include_status=True)
        out = out_stem.with_name(f"{out_stem.name}_{suffix}.png")
        out.write_bytes(base64.b64decode(img.to_base64()))
        written.append(out)
    return written


def _render_shard(address: str, tcp_port: int, states: list[Path],
                  out_dir: Path, close: bool) -> list[str]:
    """One worker process: a dedicated container, a serial slice of blobs."""
    instance = make_instance(address, tcp_port)
    written: list[str] = []
    for state_path in states:
        stem = out_dir / state_path.name.removesuffix(".state.json")
        written += [str(p) for p in
                    render_one(instance, load_state(state_path), stem,
                               close=close)]
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("states", nargs="+", type=Path,
                    help="*.state.json dump files (see FLE_BENCH_STATE_DUMPS)")
    ap.add_argument("--out", type=Path, default=Path("bench/results/blog_shots"),
                    help="output directory (default: bench/results/blog_shots)")
    ap.add_argument("--close", action="store_true",
                    help=f"also render a radius-{CLOSE_RADIUS} close-up per state")
    ap.add_argument("--address", default=None, help="cluster container IP override")
    ap.add_argument("--tcp-port", type=int, default=None, help="RCON port override")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers, one local cluster container each "
                         "(max = containers up; default 1 uses the last one)")
    args = ap.parse_args(argv)

    sprites = Path(".fle/sprites")
    if not sprites.is_dir() or not any(sprites.iterdir()):
        sys.exit("`.fle/sprites` is missing or empty -- run `fle sprites` first "
                 "(entity art needs the basisu CLI on PATH)")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1 or len(args.states) <= 1 or args.address:
        for out in _render_shard(args.address, args.tcp_port, args.states,
                                 args.out, args.close):
            print(out)
        return 0

    from fle.commons.cluster_ips import get_local_container_ips
    ips, _udp, tcp_ports = get_local_container_ips()
    if not ips:
        sys.exit("no local cluster containers found -- run `fle cluster start`")
    containers = list(zip(ips, tcp_ports))[-args.workers:]
    # Round-robin shard: each worker owns ONE container for its whole slice
    # (an RCON connection per process; containers are never shared).
    shards = [args.states[i::len(containers)] for i in range(len(containers))]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(containers)) as pool:
        futures = [
            pool.submit(_render_shard, ip, port, shard, args.out, args.close)
            for (ip, port), shard in zip(containers, shards) if shard
        ]
        for fut in futures:
            for out in fut.result():
                print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
