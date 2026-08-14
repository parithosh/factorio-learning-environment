"""Exp-3 checkpoint prep: secure S2B, the checkpoint every Exp-3 arm forks from.

Two paths, auto-selected (recorded as ``how``):

``clone``
    S2 is still restorable -> create-from-S2, verify the bridge answers
    ``/health`` through ``expose``, snapshot -> **S2B** (fresh lease), delete the
    intermediate sandbox.  Nothing is rebuilt and nothing is spent on the model.
``rebake``
    S2 lapsed (``state: deleted`` -- the image list keeps a tombstone, warm boot
    refuses with ``409 ... is not published``).  TEMPLATE_SNAP lapsed with it, so
    the recipe runs end to end: fresh ``debian-warm`` sandbox -> guest bootstrap
    -> stream ``/tmp/flebench-image.zst`` into ``docker load`` -> start the bridge
    container -> snapshot (the new TEMPLATE_SNAP) -> ONE codex agent to the Exp-1
    milestone (probe >= 2x quota, or 15 steps) -> snapshot -> **S2B**.

Every resource is named ``flebench-exp3-*`` and every op is journaled under
``bench/journal/exp3/``.  Results land in ``bench/results/exp3_prep.json``; no
Exp-1/Exp-2/tier-0 artifact is written.  ``--how record`` re-runs the three dry
gates (arm mechanisms, orchestrated block, orchestrator launch blockers) and
records their green output into the same file, so the launch record holds the
checkpoint, the personas, the block contract and the evidence in one place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from bench import tier0
from bench.bridge_client import Bridge
from bench.common import RunJournal
from bench.exp1 import Exp1Config, Exp1Runner
from bench.farplane import Farplane
from bench.run_tier1 import S2_SNAP
from bench.tier0 import BRIDGE_PORT

OUT = Path("bench/results/exp3_prep.json")
JOURNAL_DIR = Path("bench/journal/exp3")
PREFIX = "flebench-exp3-"
#: Exp-3 arms run 2 x 4200s per cell x 4 cells, so the checkpoint must outlive
#: ~10h of block plus prep.  24h is the longest lease the API accepts here.
SNAP_TTL = "24h"
IMAGE_TAR = Path("/tmp/flebench-image.zst")


def _load_out() -> dict[str, Any]:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(section: dict[str, Any], key: str = "checkpoint") -> None:
    payload = _load_out()
    payload[key] = section
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def snapshot_restorable(fp: Farplane, snap_id: str) -> tuple[bool, str]:
    """Is ``snap_id`` a snapshot we can still boot from?

    The image list keeps a tombstone for an expired snapshot (``state:
    deleted``), and ``sandboxes create --snapshot`` on one fails with HTTP 409
    ``warm boot source is not ready``.  So presence in the list is not liveness;
    the snapshot's own state is.
    """
    for image in fp.list_images():
        if image.get("id") != snap_id and image.get("snapshotId") != snap_id:
            continue
        snap = image.get("snapshot") or {}
        state = str(snap.get("state") or "")
        return state not in ("deleted", "expired", "failed"), state or "unknown"
    return False, "absent"


# ---------------------------------------------------------------------------
# clone path
# ---------------------------------------------------------------------------


def clone_checkpoint(fp: Farplane, journal: RunJournal, out: dict[str, Any]) -> str:
    sb = None
    try:
        t0 = time.monotonic()
        sb = fp.create_from_snapshot(S2_SNAP, "3h", name="s2b-clone")
        out["clone_sandbox"] = sb.id
        out["create_s"] = round(time.monotonic() - t0, 2)
        journal.event("exp3_clone_sandbox", sandbox=sb.id, node=sb.node,
                      create_s=out["create_s"])

        url = fp.expose(sb, BRIDGE_PORT)
        bridge = Bridge(url)
        out["health_wait_s"] = round(bridge.wait_healthy(300.0), 2)
        out["healthy"] = bool(bridge.health())
        out["meta"] = bridge.meta()
        journal.event("exp3_clone_health", sandbox=sb.id, healthy=out["healthy"],
                      wait_s=out["health_wait_s"], meta=out["meta"])
        if not out["healthy"]:
            raise RuntimeError("bridge never became healthy on the S2 clone")

        t0 = time.monotonic()
        s2b = fp.snapshot(sb, note="flebench-exp3-S2B", ttl=SNAP_TTL)
        out["snapshot_s"] = round(time.monotonic() - t0, 2)
        return s2b
    finally:
        if sb is not None:
            try:
                fp.delete_sandbox(sb)
                out["clone_deleted"] = True
            except Exception as exc:  # journal, continue
                out["clone_deleted"] = False
                out["cleanup_error"] = str(exc)
            journal.event("exp3_clone_cleanup", sandbox=sb.id,
                          outcome="deleted" if out.get("clone_deleted") else "failed")


# ---------------------------------------------------------------------------
# rebake path
# ---------------------------------------------------------------------------


def rebake_template(fp: Farplane, journal: RunJournal, out: dict[str, Any]) -> str:
    """Rebuild TEMPLATE_SNAP: tier-0's bake phase, exp-3 paths only."""
    if not IMAGE_TAR.exists():
        raise SystemExit(f"image archive {IMAGE_TAR} not found; cannot rebake")
    # tier-0's bake writes its transfer stats to a tier-0 results file; Exp 3
    # keeps its own so the frozen tier-0 artifact is never rewritten.
    tier0.TRANSFER_PATH = Path("bench/results/exp3_transfer.json")

    stage: dict[str, Any] = {"template": tier0.TEMPLATE, "image_tar": str(IMAGE_TAR)}
    out["template_stage"] = stage
    t0 = time.monotonic()
    sb = fp.create_from_template(tier0.TEMPLATE, "6h", name="tmplbake")
    stage["sandbox"] = sb.id
    stage["node"] = sb.node
    stage["create_s"] = round(time.monotonic() - t0, 2)
    journal.event("exp3_template_sandbox", sandbox=sb.id, node=sb.node,
                  create_s=stage["create_s"])
    _save(out)

    tier0.guest_bootstrap(fp, sb)
    stage["transfer"] = tier0.install_image(fp, sb, IMAGE_TAR)
    stage["container"] = tier0.start_container(fp, sb)
    journal.event("exp3_template_container", sandbox=sb.id, **stage["container"])

    t0 = time.monotonic()
    url = fp.expose(sb, BRIDGE_PORT)
    stage["expose_s"] = round(time.monotonic() - t0, 3)
    bridge = Bridge(url)
    t0 = time.monotonic()
    bridge.wait_healthy(180)
    stage["host_health_s"] = round(time.monotonic() - t0, 2)
    stage["meta"] = bridge.meta()
    journal.event("exp3_template_health", sandbox=sb.id, url=url,
                  host_health_s=stage["host_health_s"], meta=stage["meta"])

    # Same hygiene as tier 0: the template must not carry transfer scaffolding.
    fp.exec(sb, "docker image rm -f alpine:3 >/dev/null 2>&1; rm -f /tmp/recv_state.json "
                "/tmp/sink.log /tmp/up.bin /tmp/recv.rc /tmp/recv.log; sync", check=False)

    t0 = time.monotonic()
    template_snap = fp.snapshot(sb, ttl=SNAP_TTL, note="flebench-exp3-TEMPLATE_SNAP")
    stage["template_snap"] = template_snap
    stage["snapshot_s"] = round(time.monotonic() - t0, 2)
    journal.event("exp3_template_snap", snapshot=template_snap,
                  snapshot_s=stage["snapshot_s"])
    _save(out)

    # The template sandbox is scaffolding: the milestone bake creates its own
    # sandbox FROM the snapshot, and this deployment has ~1 warm slot per node,
    # so holding it would compete with the run it exists to enable.
    fp.delete_sandbox(sb)
    stage["sandbox_deleted"] = True
    journal.event("exp3_template_teardown", sandbox=sb.id, outcome="deleted")
    return template_snap


class _Exp3Bake(Exp1Runner):
    """Exp-1's milestone bake, journaled and noted as Exp 3's own."""

    async def _snapshot(self, node: Any, *, note: str) -> str:  # noqa: D102
        return await super()._snapshot(node, note="flebench-exp3-S2B")


async def rebake_checkpoint(template_snap: str, model: str,
                            out: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One agent from the fresh template to the Exp-1 milestone, then snapshot."""
    cfg = Exp1Config(
        template_snap=template_snap,
        model=model,
        bake_steps=15,
        probe_every=4,
        bake_target_multiple=2.0,
        snapshot_ttl=SNAP_TTL,
        budget_s=5400.0,
        reserve_s=600.0,
        results_path="bench/results/exp3_bake.json",
        report_path="bench/results/EXP3_BAKE.md",
        journal_dir=str(JOURNAL_DIR),
        prefix=PREFIX,
        run_id="exp3bake",
    )
    runner = _Exp3Bake(cfg)
    # Keep the substrate op journal with the rest of Exp 3's evidence.
    runner.fp.journal_path = JOURNAL_DIR / "exp3-bake-farplane.jsonl"
    try:
        snap = await runner.bake()
        return snap, dict(runner.state.get("s2") or {})
    finally:
        await runner.aclose()
        # ``Exp1Runner`` hardcodes ``exp1.jsonl`` inside its journal dir. This
        # bake is Exp 3's, so the file is renamed once it is flushed -- an
        # ``exp1.jsonl`` under bench/journal/exp3/ would read as Exp-1 evidence.
        stale = JOURNAL_DIR / "exp1.jsonl"
        bake_journal = JOURNAL_DIR / "exp3-bake.jsonl"
        if stale.exists():
            stale.replace(bake_journal)
        out["bake_journals"] = {
            "run": str(bake_journal),
            "farplane": str(runner.fp.journal_path),
            "results": cfg.results_path,
        }


def verify_checkpoint(fp: Farplane, journal: RunJournal, s2b: str) -> dict[str, Any]:
    """Boot S2B once: health, meta, and ONE fixed probe as Exp 3's start line.

    Four live cells descend from this id, so it is booted before the block, not
    during it. The probe is the design's "decay read" reference: any Exp-3
    endpoint below the checkpoint's OWN throughput regressed from a byte-
    identical start, which is a different failure mode from "the hybrid did not
    help". The sandbox is deleted immediately -- warm slots are the scarce
    resource on this deployment.
    """
    from bench.arms import task_spec

    _goal, entity, quota = task_spec("iron_plate_throughput")
    rec: dict[str, Any] = {"snapshot": s2b, "entity": entity, "quota": quota}
    sb = None
    try:
        t0 = time.monotonic()
        sb = fp.create_from_snapshot(s2b, "1h", name="s2b-verify")
        rec["sandbox"] = sb.id
        rec["create_s"] = round(time.monotonic() - t0, 2)
        bridge = Bridge(fp.expose(sb, BRIDGE_PORT))
        rec["health_wait_s"] = round(bridge.wait_healthy(300.0), 2)
        rec["healthy"] = bool(bridge.health())
        rec["meta"] = bridge.meta()
        probe = bridge.probe(entity)
        rec["reference_throughput"] = float(probe.get("throughput", 0.0) or 0.0)
        rec["probe"] = probe
        journal.event("exp3_checkpoint_verified", **rec)
    finally:
        if sb is not None:
            try:
                fp.delete_sandbox(sb)
                rec["deleted"] = True
            except Exception as exc:  # journal, continue
                rec["deleted"] = False
                rec["cleanup_error"] = str(exc)
            journal.event("exp3_verify_cleanup", sandbox=sb.id,
                          outcome="deleted" if rec.get("deleted") else "failed")
    return rec


def start_line_comparison(out: dict[str, Any]) -> dict[str, Any]:
    """How S2B's start line compares with Exp 1's S2, stated not implied.

    Both checkpoints met the SAME pre-registered milestone (a fixed probe at or
    above 2x quota), but they met it differently: k3 needed 12 steps and built 48
    entities to 76 plates/60s, codex needed 4 and built 8 to ~37. S2B is
    therefore a THINNER shared prefix than S2, which is admissible -- the design
    only requires that all Exp-3 arms share ONE checkpoint, and it already forbids
    comparing Exp-3 endpoints numerically against Exp 2's -- but the reference
    line for Exp 3's decay read is S2B's own throughput, not S2's, and that
    number has to be on the record before the block runs.
    """
    rec: dict[str, Any] = {
        "s2b": {
            "snapshot": out.get("S2B"),
            "milestone": out.get("milestone"),
            "reference_throughput": (out.get("verified") or {}).get(
                "reference_throughput"),
            "entities": (out.get("bake") or {}).get("entities_end"),
            "steps": (out.get("bake") or {}).get("steps"),
        },
        "note": (
            "Exp-3 endpoints are compared only against each other and against "
            "S2B's own reference throughput; never against Exp-2 numbers (S2, "
            "different T, different model start line)."
        ),
    }
    try:
        s2 = json.loads(Path("bench/results/exp1.json").read_text())["s2"]
        rec["s2_exp1"] = {
            "snapshot": s2.get("snapshot"),
            "milestone_throughput": (s2.get("milestone") or {}).get(
                "reached_throughput"),
            "entities": (s2.get("bake") or {}).get("entities_end"),
            "steps": (s2.get("bake") or {}).get("steps"),
            "state": "deleted (24h lease lapsed 2026-08-11T09:42Z)",
        }
    except (OSError, KeyError, json.JSONDecodeError):
        rec["s2_exp1"] = None
    return rec



# ---------------------------------------------------------------------------
# launch record: personas, block contract, dry evidence
# ---------------------------------------------------------------------------


def _trim(section: dict[str, Any]) -> dict[str, Any]:
    """A dry-report section without the per-bucket timing dump."""
    return {k: v for k, v in section.items() if k != "timings"}


def record_dry() -> dict[str, Any]:
    """Re-run Exp 3's dry gates and record what they proved.

    1. ``bench.arms.dry_run()`` -- the arm mechanisms (persona placement and
       rotation, one selection over all seats, the refork wave and its width
       floor, the strict control's negatives, max-over-seats endpoints, leaks,
       timing partition).
    2. ``bench.run_tier1.main(--exp3-block --round 1 --parallel-round --dry)`` --
       the block IN THE MODE IT WILL LAUNCH: one round, three rungs at once,
       shared codex gate with Control exempt, block-level assertions.
    3. ``bench.run_tier1.main(--exp3-block --dry)`` -- the same six cells
       sequentially, so the ladder is proven in both modes.
    4. ``bench.run_tier1.dry_validate()`` -- the orchestrator's launch blockers
       (manifest order, round gate, width floor on RESULTS, parallel caps and
       provider exemption, rerun selectors).

    Any violation raises out of the gate itself; reaching the end IS the record.
    """
    import contextlib
    import io

    from bench import run_tier1 as t1
    from bench.arms import (
        EXP3_ARMS,
        EXP3_PHASE2_ADMISSION,
        EXP3_WIDTH_FLOOR,
        PERSONAS,
        dry_run,
    )

    arms_report = asyncio.run(dry_run())
    gate_path = Path("bench/results/exp3_dry_gate.json")
    gate_path.write_text(json.dumps(arms_report, indent=2, default=str) + "\n",
                         encoding="utf-8")

    def run_block(args: list[str]) -> tuple[str, dict[str, Any]]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = t1.main(args)
        assert rc == 0, f"{' '.join(args)} exited {rc}"
        tail = next(line for line in buf.getvalue().splitlines()
                    if line.startswith("EXP3 DRY BLOCK OK"))
        payload = json.loads(
            Path("bench/results/exp3_block_dry.json").read_text()
        )
        return tail, payload

    par_tail, par_block = run_block(
        ["--exp3-block", "--round", "1", "--parallel-round", "--dry"]
    )
    seq_tail, seq_block = run_block(["--exp3-block", "--dry"])
    validate = asyncio.run(t1.dry_validate())
    live = t1.build_config(t1._cli().parse_args(["--exp3-block"]))
    par_live = t1.build_config(
        t1._cli().parse_args(["--exp3-block", "--round", "1", "--parallel-round"])
    )

    keep_prefixes = ("exp3", "width_floor", "lease_guard", "provider_tripwire",
                     "tripwire")
    exp3_assertions = {
        k: v for k, v in arms_report["assertions"].items()
        if k.startswith(keep_prefixes)
    }
    exp3_assertions.update(
        {k: v for k, v in validate["assertions"].items()
         if k.startswith(keep_prefixes)}
    )

    def cell_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "cell": r["cell"],
                "status": r["status"],
                "endpoint": r["endpoint_throughput"],
                "endpoint_source": r["endpoint_source"],
                "seat_endpoints": [s["throughput"]
                                   for s in r.get("seat_endpoints") or []],
                "dose": r["branch_points"],
                "k_effective": ((r.get("exp3") or {}).get("validity")
                                or {}).get("k_effective"),
                "steps": r["steps"],
                "provider_high_water": (r.get("provider_gate")
                                        or {}).get("in_flight_high_water"),
            }
            for r in payload["runs"]
        ]

    rounds = {
        n: [c.key for c in t1.exp3_block_cells(live) if c.replicate == n]
        for n in sorted({rep for _, rep in t1.EXP3_BLOCK})
    }
    return {
        "arms": list(t1.EXP3_LADDER),
        "ladder": {
            "Control": "1 seat, no persona, no fork -- the bottom rung",
            "AxK-S": "8 persona seats forked once, never converged",
            "Hybrid": "8 persona seats, one halftime regroup, personas rotated",
            "reads": ["AxK-S - Control = value of forking wide",
                      "Hybrid - AxK-S = value of one convergence"],
        },
        "personas": list(PERSONAS),
        "config": {
            "model": live.models[0],
            "task": live.tasks[0],
            "P_s": live.leg_s,
            "T_total_s": live.T_s,
            "build_time_matched": True,
            "K": live.K,
            "m": live.m,
            "cells": [c.key for c in t1.exp3_block_cells(live)],
            "rounds": rounds,
            "round_selector": "--round N (== --cells "
                              f"{','.join(f'{a}:N' for a in t1.EXP3_LADDER)})",
            "width_floor": EXP3_WIDTH_FLOOR,
            "phase2_admission_fraction": EXP3_PHASE2_ADMISSION,
            "ttl_s": live.ttl_s,
            "ttl_derivation": f"T_total + {t1.EXP3_TTL_MARGIN_S:.0f}s margin "
                              "(exp3_ttl_s); covers seats AND halftime refork "
                              "children -- round 1 hibernated at 7200s",
            "create_deadline_s": live.create_deadline_s,
            "sequential": {"run_cap": live.run_cap,
                           "provider_concurrency": live.provider_concurrency},
            "parallel": {
                "run_cap": par_live.run_cap,
                "max_sandboxes": par_live.max_sandboxes,
                "provider_concurrency": par_live.provider_concurrency,
                "exempt_arms": list(t1.EXP3_PROVIDER_EXEMPT_ARMS),
                "provision_stagger_s": par_live.provision_stagger_s,
                "provision_order": list(t1.EXP3_LADDER),
                "bias": (
                    "Control unthrottled while fan-out seats share 8 in flight "
                    "-> DEFLATES the Control->A×K-S gap: conservative for the "
                    "fork-value claim, and read as such"
                ),
            },
            "out": live.out,
            "journal_dir": live.journal_dir,
            "checkpoint": live.template_snap,
            "keep": list(live.keep),
        },
        "launch_command": (
            "python -m bench.run_tier1 --exp3-block --round 1 --parallel-round"
            "   # round 2 only on an explicit go; checkpoint, model, P/T/K/m, "
            "caps, keep list and the provider regime are all pinned by the block"
        ),
        "dry": {
            "arm_gate": {
                "arithmetic": arms_report["exp3_arithmetic"],
                "Hybrid": _trim(arms_report["Hybrid"]),
                "Hybrid_truncated": _trim(arms_report["Hybrid-truncated"]),
                "AxK-S": _trim(arms_report["AxK-S"]),
                "Control": _trim(arms_report["Control"]),
                "command": "python -m bench.arms --dry",
                "report": str(gate_path),
                "journal_dir": arms_report["journal_dir"],
                "lease_guard": arms_report["lease_guard"],
                "provider_tripwire": arms_report["provider_tripwire"],
            },
            "parallel_round": {
                "command": "python -m bench.run_tier1 --exp3-block --round 1 "
                           "--parallel-round --dry",
                "tail": par_tail,
                "caps": par_block["caps"],
                "provider": par_block["provider"],
                "cells": cell_rows(par_block),
            },
            "sequential_block": {
                "command": "python -m bench.run_tier1 --exp3-block --dry",
                "tail": seq_tail,
                "cells": cell_rows(seq_block),
            },
            "orchestrator_gate": {
                "command": "python -m bench.run_tier1 --dry-validate",
                "exp3_block": validate["exp3_block"],
                "width_floor": validate["width_floor"],
                "lease_guard_presets": validate["lease_guard"],
                "parallel_round": validate["parallel_round"],
                "provider_tripwire": validate["provider_tripwire"],
            },
            "assertions": exp3_assertions,
        },
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exp-3 checkpoint prep (S2B)")
    ap.add_argument("--how", default="auto",
                    choices=("auto", "clone", "rebake", "verify", "record"))
    ap.add_argument("--model", default="codex/gpt-5.6-sol",
                    help="model for the rebake milestone agent")
    ap.add_argument("--template-snap", default="",
                    help="skip the template rebuild and bake from this snapshot")
    ap.add_argument("--s2b", default="",
                    help="verify this checkpoint instead of securing a new one")
    args = ap.parse_args(argv)

    if args.how == "record":
        # No substrate, no spend: the dry gates plus the launch contract.
        record = record_dry()
        _save(record, key="exp3")
        print(record["dry"]["parallel_round"]["tail"])
        print(record["dry"]["sequential_block"]["tail"])
        print(f"recorded {len(record['personas'])} personas, "
              f"{len(record['arms'])} rungs, "
              f"{len(record['config']['cells'])} cells "
              f"({len(record['config']['rounds'])} rounds) and "
              f"{len(record['dry']['assertions'])} dry assertions to {OUT}")
        return 0

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    journal = RunJournal(
        str(JOURNAL_DIR / "exp3-prep.jsonl"),
        run_id="flebench-exp3-prep",
        meta={"experiment": "exp3-hybrid", "phase": "checkpoint", "source": S2_SNAP},
    )
    fp = Farplane(str(JOURNAL_DIR / "exp3-prep-farplane.jsonl"), prefix=PREFIX)

    prior = (_load_out().get("checkpoint") or {}) if (args.s2b or args.how == "verify") else {}
    out: dict[str, Any] = {**prior, "source_snapshot": S2_SNAP,
                           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime())}
    try:
        alive, state = snapshot_restorable(fp, S2_SNAP)
        out["s2_state"] = state
        how = args.how
        if how == "auto":
            how = "clone" if alive else "rebake"
        if how == "verify":
            s2b = args.s2b or str(out.get("S2B") or "")
            if not s2b:
                raise SystemExit("--how verify needs --s2b or a recorded S2B")
            out["verified"] = verify_checkpoint(fp, journal, s2b)
            out["start_line"] = start_line_comparison(out)
            print(f"S2B = {s2b} verified: healthy={out['verified'].get('healthy')} "
                  f"reference_throughput={out['verified'].get('reference_throughput')}")
            return 0
        out["how"] = how
        journal.event("exp3_prep_start", source=S2_SNAP, s2_state=state, how=how,
                      model=args.model if how == "rebake" else "")
        _save(out)

        if how == "clone":
            s2b = clone_checkpoint(fp, journal, out)
        else:
            template = args.template_snap or rebake_template(fp, journal, out)
            out["template_snap"] = template
            snap, s2_rec = asyncio.run(rebake_checkpoint(template, args.model, out))
            s2b, out["milestone"] = snap, s2_rec.get("milestone")
            out["bake"] = s2_rec.get("bake")
        out["S2B"] = s2b
        out["verified"] = verify_checkpoint(fp, journal, s2b)
        out["start_line"] = start_line_comparison(out)
        journal.event("exp3_checkpoint", snapshot=s2b, how=how,
                      source=S2_SNAP if how == "clone" else out.get("template_snap"),
                      ttl=SNAP_TTL)
        print(f"S2B = {s2b}  (how={how})")
        return 0
    except BaseException as exc:  # journal the blocker, then re-raise
        out["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        journal.event("exp3_prep_error", error=out["error"])
        raise
    finally:
        out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save(out)
        journal.close()


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
