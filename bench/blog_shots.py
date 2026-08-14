"""Post-hoc local-cluster rendering of ``/state-save`` blobs into blog PNGs.

Not offline: the renderer needs a running LOCAL Factorio container to restore
into (``fle cluster start``).  What it does guarantee is no benchmark
contamination -- the measured run is never touched, because its blobs are
re-rendered afterwards on a different cluster.  Benchmark runs persist them when
``FLE_BENCH_STATE_DUMPS=1`` (see :meth:`ArmRun._dump_state` in
``bench/arms.py``) -- arm C rounds land as ``rN.state.json`` for free, and
every arm saves ``final-<seat>.state.json`` per node after ``timings.stop()``.

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


def strip_namespaces(state: object) -> int:
    """Blank the pickled agent namespaces on a parsed state; return how many.

    ``instance.reset(game_state)`` hands ``game_state.namespaces[i]`` straight to
    ``FactorioNamespace.load`` -> ``pickle.loads``, so restoring a dump executes
    whatever wrote it. These PNGs need only entities, inventories and research;
    the agent namespace (persistent vars and pickled helper functions) is never
    read by the renderer. So the blobs are replaced with empty bytes before the
    restore -- ``load`` swallows the resulting unpickling error -- and a dump off
    a shared node, a colleague's run or an archive is rendered, not executed.
    """
    blobs = list(getattr(state, "namespaces", []) or [])
    state.namespaces = [b"" for _ in blobs]  # type: ignore[attr-defined]
    return sum(1 for ns in blobs if ns)


def render_one(instance, state: str, out_stem: Path, *, close: bool) -> list[Path]:
    from fle.commons.models.game_state import GameState
    from fle.env.entities import Position

    game_state = GameState.parse_raw(state)
    dropped = strip_namespaces(game_state)
    if dropped:
        print(f"{out_stem.name}: dropped {dropped} pickled namespace blob(s) "
              "before restore (rendering needs entities/inventories only)",
              file=sys.stderr)
    instance.reset(game_state)
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


def output_stem(state_path: Path, out_dir: Path) -> Path:
    """Where one dump's PNGs go: ``<out_dir>/<run>_<dump>``.

    Dump basenames are NOT unique -- every arm writes ``final-<seat>.state.json``
    and arm C writes ``rN.state.json`` per round -- so a basename-only stem lets
    the second run silently overwrite the first run's PNGs. The immediate parent
    (``<run>.states/``) is the run component.
    """
    name = state_path.name.removesuffix(".state.json")
    run = state_path.parent.name.removesuffix(".states")
    run = "".join(c if (c.isalnum() or c in "-.") else "_" for c in run).strip("_.")
    return out_dir / (f"{run}_{name}" if run else name)


def plan_outputs(states: list[Path], out_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every input with its output stem, or refuse the whole batch.

    Two dumps that compute the same stem would have the later render overwrite
    the earlier one, which is worse than useless in a figure: the caption would
    name a run the image does not show. A collision is a hard error naming both
    inputs, checked in the parent before any container is touched.
    """
    plan: list[tuple[Path, Path]] = []
    owner: dict[Path, Path] = {}
    for state_path in states:
        stem = output_stem(state_path, out_dir)
        clash = owner.get(stem)
        if clash is not None:
            sys.exit(f"output collision: {clash} and {state_path} both render to "
                     f"{stem}_wide.png -- render them into separate --out dirs")
        owner[stem] = state_path
        plan.append((state_path, stem))
    return plan


def _render_shard(address: str, tcp_port: int, jobs: list[tuple[Path, Path]],
                  close: bool) -> list[str]:
    """One worker process: a dedicated container, a serial slice of blobs."""
    instance = make_instance(address, tcp_port)
    written: list[str] = []
    for state_path, stem in jobs:
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
    plan = plan_outputs(args.states, args.out)

    if args.workers <= 1 or len(plan) <= 1 or args.address:
        for out in _render_shard(args.address, args.tcp_port, plan, args.close):
            print(out)
        return 0

    from fle.commons.cluster_ips import get_local_container_ips
    ips, _udp, tcp_ports = get_local_container_ips()
    if not ips:
        sys.exit("no local cluster containers found -- run `fle cluster start`")
    containers = list(zip(ips, tcp_ports))[-args.workers:]
    # Round-robin shard: each worker owns ONE container for its whole slice
    # (an RCON connection per process; containers are never shared).
    shards = [plan[i::len(containers)] for i in range(len(containers))]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(containers)) as pool:
        futures = [
            pool.submit(_render_shard, ip, port, shard, args.close)
            for (ip, port), shard in zip(containers, shards) if shard
        ]
        for fut in futures:
            for out in fut.result():
                print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
