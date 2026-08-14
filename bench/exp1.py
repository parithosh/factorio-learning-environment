"""Experiment 1 -- decorrelation gate (Farplane fan-out benchmark).

Question (design doc, "Experiment 1"): when K branches share a past and receive
divergent strategies, does the outcome space open enough that selecting a best
branch is meaningful?

Protocol implemented here, exactly as specified:

1. BAKE.  A fresh sandbox from ``TEMPLATE_SNAP`` runs one k3 agent on
   ``iron_plate_throughput`` with the standard conversation (system prompt from
   the bridge + task goal) and the standard probe cadence (every 4 steps) until
   the milestone fires -- a probe at or above 2x quota -- or 15 steps, whichever
   comes first.  The sandbox is then snapshotted into **S2** (kept) and deleted:
   every branch of every wave forks from S2, so the bake sandbox is pure
   scaffolding once the snapshot exists.
2. WAVES.  For each wave, K=8 sandboxes are forked from S2 *sequentially* (forks
   of a lineage pin to the source's node and serialise), exposed and health
   checked.  A fork's world keeps RUNNING at game speed while the remaining
   forks are taken, so the first child would enter its rollout minutes of
   in-game time ahead of the last one -- and because the strategy hints are
   POSITIONAL, that skew is a repeatable confound rather than noise.  The wave
   is therefore released through a BARRIER: the least-advanced child's world is
   captured with ``/state-save`` and restored onto EVERY child (the capture
   source included, so the treatment is uniform) with ``/state-restore``
   immediately before the rollouts start, so all K branches begin from
   identical world content.  Per-child game ticks are recorded either way; if
   the bridge's state endpoints are unavailable the wave runs only while the
   measured tick skew is inside ``Exp1Config.barrier_skew_tolerance_ticks`` and
   is ABORTED with no draws past it.  Then all 8 branches run CONCURRENTLY,
   each with ONE divergent strategy hint injected into its first user turn
   through the existing hints mechanism (:data:`bench.llm.HINT_TEMPLATE`).
   Each branch runs 12 steps and is probed EN ROUTE after steps 4, 8 and 12
   (nested design), so probe side effects are uniform across branches and the
   m=12 endpoints keep Exp 2's probe cadence -- the barrier restore is Exp 1's
   own addition on top of the A*K-from-S fork path and is reported as such.
   Branch sandboxes are deleted at the end of the wave; the next wave forks
   fresh children from the SAME S2.
3. ANALYSIS.  Per (wave, m): ``spread = (max-min)/median`` and
   ``gain = (max-median)/median``.  A wave is read only if it drew its full
   pre-registered complement at the read point (n == K, or
   ``Exp1Config.min_draws_per_wave``); a wave short of it leaves the gate
   ``borderline-undecided`` instead of deciding it on a thinner pool.  Gate at
   the best m, which must hold in BOTH waves: PASS if spread >= 0.25 and
   gain >= 0.15; CONDITIONAL if spread in [0.10, 0.25) and gain >= 0.10; FAIL
   otherwise.  Edge rules: median == 0 with max > 0 is a PASS-signal (maximal
   selectability); all eight zero is a FAIL-signal at that m.  Wave
   disagreement at the best m calls a third wave and takes the majority,
   flagged borderline -- read at the m the FIRST TWO waves selected, persisted
   when the third wave is called so the extra wave cannot move the read point
   it was called to settle; with no budget for a third wave the verdict is
   ``borderline-undecided``.
   Double duty on the m=12 endpoints: an empirical best-of-K curve and a power
   check for Exp 2's paired n=3.  The best-of-K bootstrap respects the design
   instead of pooling every draw: one resample takes one outcome per configured
   strategy arm, and a K below the number of arms is a WITHOUT-replacement
   subset of arms -- a fan-out wave runs K DIFFERENT strategies, so drawing the
   same arm twice is not something the deployment can do.
   Pooling guard: the two waves are pooled only if the location shift is within
   0.10 of the pooled median AND the spread ratio is within 2x; otherwise every
   read is per wave and the power check uses the larger variance.

Everything runs on the primitives the rest of the harness already uses
(:class:`bench.arms.ArmRun` for the agent step / probe cycle,
:class:`bench.arms.Infra` for journaled substrate ops, :class:`bench.farplane`
for the CLI and the reaper).  No FLE source is touched and no FLE-specific state
code is in the measurement path.

Entry points::

    python -m bench.exp1 --phase run                # bake + waves + analysis + cleanup
    python -m bench.exp1 --phase wave --wave 3      # extra (third) wave from the saved S2
    python -m bench.exp1 --phase analyze            # recompute analysis + report only
    python -m bench.exp1 --phase reap               # sweep and report residual
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from bench.arms import (
    ArmConfig,
    ArmRun,
    Node,
    Trajectory,
    _tick_of,
    default_bridge_factory,
)
from bench.common import RunJournal, atomic_write_json
from bench.farplane import Farplane, summarize
from bench.llm import HINT_TEMPLATE, make_client

#: Baked greenfield template every checkpoint descends from (settled input).
TEMPLATE_SNAP = "snapshot-5fa7769473a710b2"

#: Settled mechanics constant this run's forks are checked against (drift guard).
FORK_CONSTANT_S = 32.0

#: The eight divergent strategies. Positional: branch i always gets hint i, so
#: a wave's assignment is reproducible and comparable across waves. Each is
#: on-task for iron-plate throughput and mutually exclusive in what it changes.
STRATEGY_HINTS: tuple[tuple[str, str], ...] = (
    (
        "expand-wide",
        "Expand the factory WIDE: add more mining drills and more furnaces "
        "together, in parallel, so ore supply and smelting capacity grow at the "
        "same time. Do not stop to optimise -- add capacity.",
    ),
    (
        "rebuild-ratios",
        "Rebuild the line's RATIOS from scratch: work out how many drills feed "
        "how many furnaces and what the belt can carry, then lay the stages out "
        "matched to those numbers instead of extending what is there.",
    ),
    (
        "second-cell",
        "Build an INDEPENDENT SECOND production cell somewhere else: its own "
        "miners, its own furnaces, its own output chest. Leave the existing "
        "line untouched so the two cells add up.",
    ),
    (
        "power-electrify",
        "Fix and extend POWER first -- offshore pump, boilers, steam engines, "
        "poles with real coverage -- and then ELECTRIFY mining so drills never "
        "stall for fuel.",
    ),
    (
        "logistics-optimise",
        "Optimise the LOGISTICS of the existing line only: inserters, belt "
        "routing, buffering and hand-off points. Add no new smelting capacity; "
        "make what exists move plates without stalling.",
    ),
    (
        "vertical-furnaces",
        "Go VERTICAL on the ore you already deliver: keep the current patch and "
        "current miners, and stack many more furnaces onto that same ore flow.",
    ),
    (
        "relayout-compact",
        "TEAR DOWN the current layout and re-lay it COMPACTLY: shortest belts, "
        "fewest transfer hops, furnaces packed next to the drills.",
    ),
    (
        "new-outpost",
        "DIVERSIFY the ore supply: prospect a different iron patch, build a new "
        "mining outpost there and run its ore into smelting.",
    ),
)

#: FLE agent API surface (``fle/env/tools/agent``), used to summarise from the
#: transcript what a branch actually built. ``print`` is excluded: it is noise.
FLE_VERBS = frozenset(
    {
        "can_place_entity", "connect_entities", "craft_item", "extract_item",
        "get_connection_amount", "get_entities", "get_entity",
        "get_prototype_recipe", "get_research_progress", "get_resource_patch",
        "harvest_resource", "insert_item", "inspect_inventory", "launch_rocket",
        "move_to", "nearest", "nearest_buildable", "pickup_entity",
        "place_entity", "place_entity_next_to", "rotate_entity", "score",
        "send_message", "set_entity_recipe", "set_research", "shift_entity",
        "sleep",
    }
)

_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")

#: Gate thresholds, verbatim from the design.
PASS_SPREAD, PASS_GAIN = 0.25, 0.15
COND_SPREAD_LO, COND_SPREAD_HI, COND_GAIN = 0.10, 0.25, 0.10

_TIER_RANK = {"FAIL": 0, "CONDITIONAL": 1, "PASS": 2}

#: Two-sided Student-t 0.975 quantiles by degrees of freedom (n-1). Used by the
#: power check's rejection rule; scipy is not a dependency of this harness.
T_CRIT_975: dict[int, float] = {
    1: 12.70620, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
    16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
}

TARGET_POWER = 0.80

#: Experiment identity written into (and demanded of) the results artefact.
EXPERIMENT_KEY = "exp1-decorrelation-gate"

#: Config keys that define what a recorded draw MEANS. Draws taken under
#: different values of these are different measurements and are never mixed, so
#: a resume whose config disagrees on any of them is refused. Everything else
#: (paths, budgets, bootstrap seed and resamples) is operational: it may change
#: on a resume and the change is recorded rather than rejected.
MEASUREMENT_KEYS: tuple[str, ...] = (
    "template_snap",
    "model",
    "task_key",
    "k",
    "steps",
    "probe_every",
    "bake_steps",
    "bake_target_multiple",
    "min_draws_per_wave",
    "barrier_skew_tolerance_ticks",
)


def measurement_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    """The measurement-affecting slice of a config, plus a short digest of it."""
    keys = {k: config.get(k) for k in MEASUREMENT_KEYS}
    digest = hashlib.blake2b(
        json.dumps(keys, sort_keys=True, default=str).encode("utf-8"), digest_size=8
    ).hexdigest()
    return {"keys": keys, "digest": digest}


def config_delta(
    saved: Mapping[str, Any], current: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """``{key: {"saved": ..., "now": ...}}`` for every ``key`` that differs."""
    return {
        k: {"saved": saved.get(k), "now": current.get(k)}
        for k in keys
        if saved.get(k) != current.get(k)
    }


#: Timing-summary fields that are RATIOS or bookkeeping, not additive seconds.
_TIMING_DERIVED = ("infra_fraction_attributed", "infra_fraction_raw", "invocations")


def merge_counters(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Add ``current``'s counters onto ``prior``'s, recursing into sub-dicts.

    Values that are not numbers (or dicts of them) are labels, not counts, and
    are taken from ``current``. Merging an all-zero contribution is a no-op, so
    an invocation that measured nothing cannot dilute the record.
    """
    out: dict[str, Any] = dict(prior)
    for key, value in current.items():
        have = out.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, dict)):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = merge_counters(have, value) if isinstance(have, dict) else dict(value)
        elif isinstance(have, (int, float)) and not isinstance(have, bool):
            out[key] = have + value
        else:
            out[key] = value
    return out


def merge_timings(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Cumulative timing across invocations of one results file.

    Bucket seconds, wall clock and interval counts add up; the infra fractions
    are RECOMPUTED from the merged totals, because adding two ratios is not a
    ratio. ``invocations`` says how many runs contributed, so a resumed run's
    report cannot read as if one invocation produced all of it.
    """
    if not prior:
        return {**dict(current), "invocations": 1}
    merged = merge_counters(
        {k: v for k, v in prior.items() if k not in _TIMING_DERIVED},
        {k: v for k, v in current.items() if k not in _TIMING_DERIVED},
    )
    wall = float(merged.get("wall_s") or 0.0)
    attributed = merged.get("attributed_s") or {}
    raw = merged.get("raw_s") or {}
    infra_attributed = sum(v for b, v in attributed.items() if b.startswith("infra_"))
    infra_raw = sum(v for b, v in raw.items() if b.startswith("infra_"))
    merged["infra_fraction_attributed"] = (
        round(infra_attributed / wall, 6) if wall else 0.0
    )
    merged["infra_fraction_raw"] = round(infra_raw / wall, 6) if wall else 0.0
    merged["invocations"] = int(prior.get("invocations") or 1) + 1
    return merged


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Exp1Config:
    template_snap: str = TEMPLATE_SNAP
    model: str = "k3"
    task_key: str = "iron_plate_throughput"
    k: int = 8
    waves: int = 2
    steps: int = 12
    probe_every: int = 4
    bake_steps: int = 15
    #: Draws a wave must hold at the read point for the gate to be decidable.
    #: ``None`` = the pre-registered default, the full complement K. A wave
    #: short of it leaves the gate undecided; a missing draw is never resampled.
    min_draws_per_wave: int | None = None
    #: Fork-barrier tolerance, in game ticks, used ONLY when the bridge cannot
    #: equalise the children with /state-save + /state-restore. 3600 ticks = one
    #: probe window (60 in-game seconds at speed 1) of divergence between the
    #: first and last child of a wave.
    barrier_skew_tolerance_ticks: int = 3600
    #: Milestone: a bake probe at or above this multiple of the task quota.
    bake_target_multiple: float = 2.0
    #: Sandbox lease. Never a cleanup mechanism -- only insurance that a slow
    #: wave cannot hibernate under us. Deletion is always explicit.
    ttl_s: int = 10800
    #: S2 must outlive every wave (and stays on the keep-list afterwards).
    snapshot_ttl: str = "24h"
    max_llm_concurrency: int = 8
    budget_s: float = 6 * 3600.0
    #: Held back for analysis, reports and cleanup no matter what.
    reserve_s: float = 1800.0
    #: A wave is not started unless this much (plus the reserve) remains.
    wave_estimate_s: float = 1500.0
    resamples: int = 10000
    seed: int = 20260810
    results_path: str = "bench/results/exp1.json"
    report_path: str = "bench/results/EXP1.md"
    journal_dir: str = "bench/journal/exp1"
    prefix: str = "flebench-"
    run_id: str = "exp1"

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Metrics, gate, bootstrap, power  (pure functions -- the spec's arithmetic)
# ---------------------------------------------------------------------------


def _fin(value: float) -> float | None:
    """JSON has no infinity; an unbounded ratio is recorded as null + edge."""
    return None if math.isinf(value) else round(value, 6)


def classify(spread: float, gain: float) -> str:
    """The gate's tier for one (wave, m), verbatim from the design."""
    if spread >= PASS_SPREAD and gain >= PASS_GAIN:
        return "PASS"
    if COND_SPREAD_LO <= spread < COND_SPREAD_HI and gain >= COND_GAIN:
        return "CONDITIONAL"
    return "FAIL"


def metrics_at(values: Sequence[float | None]) -> dict[str, Any]:
    """Spread / gain / tier over one wave's branch scores at one m.

    Edge rules: ``median == 0 and max > 0`` -> PASS-signal with unbounded
    ratios; all draws zero -> FAIL-signal; no draws at all -> undecidable.
    """
    vals = [float(v) for v in values if v is not None]
    out: dict[str, Any] = {
        "n": len(vals),
        "scores": sorted(vals, reverse=True),
    }
    if not vals:
        out.update(median=None, max=None, min=None, spread=None, gain=None,
                   edge="no_draws", tier="undecided",
                   _spread=-math.inf, _gain=-math.inf)
        return out
    mx, mn = max(vals), min(vals)
    med = statistics.median(vals)
    out.update(median=round(med, 6), max=round(mx, 6), min=round(mn, 6))
    if med == 0.0 and mx > 0.0:
        out.update(spread=None, gain=None, edge="median_zero", tier="PASS",
                   _spread=math.inf, _gain=math.inf)
        return out
    if mx == 0.0:
        out.update(spread=0.0, gain=0.0, edge="all_zero", tier="FAIL",
                   _spread=0.0, _gain=0.0)
        return out
    spread, gain = (mx - mn) / med, (mx - med) / med
    out.update(spread=_fin(spread), gain=_fin(gain), edge=None,
               tier=classify(spread, gain), _spread=spread, _gain=gain)
    return out


def wave_metrics(branches: Sequence[dict[str, Any]], ms: Sequence[int]) -> dict[str, Any]:
    """Per-m metrics for one wave, plus its missing-draw accounting."""
    out: dict[str, Any] = {}
    for m in ms:
        draws = [b.get("scores", {}).get(str(m)) for b in branches]
        rec = metrics_at(draws)
        rec["missing"] = [
            b["branch"] for b in branches if b.get("scores", {}).get(str(m)) is None
        ]
        out[str(m)] = rec
    return out


def pick_best_m(waves: Sequence[dict[str, Any]], ms: Sequence[int]) -> dict[str, Any]:
    """The m the gate is read at: the one best supported in EVERY wave.

    The gate must hold in both waves, so an m is ranked by its BINDING (worst
    across waves) tier, then by binding spread, then by binding gain. Ties
    break towards the later m -- the longer horizon carries more signal and is
    the one that is protocol-identical to Exp 2's endpoint.
    """
    if not waves:
        raise ValueError("pick_best_m needs at least one wave; there is no m to rank")
    if not ms:
        raise ValueError("pick_best_m needs at least one candidate m")
    ranked = []
    for m in ms:
        per = [w["metrics"][str(m)] for w in waves]
        tiers = [_TIER_RANK.get(p["tier"], -1) for p in per]
        ranked.append(
            {
                "m": m,
                "binding_tier_rank": min(tiers),
                "binding_spread": min(p["_spread"] for p in per),
                "binding_gain": min(p["_gain"] for p in per),
                "tiers": [p["tier"] for p in per],
            }
        )
    best = max(
        ranked,
        key=lambda r: (r["binding_tier_rank"], r["binding_spread"], r["binding_gain"], r["m"]),
    )
    return {
        "best_m": best["m"],
        "ranking": [
            {
                "m": r["m"],
                "binding_tier": [t for t, rank in _TIER_RANK.items() if rank == r["binding_tier_rank"]][0]
                if r["binding_tier_rank"] in _TIER_RANK.values() else "undecided",
                "binding_spread": _fin(r["binding_spread"]) if r["binding_spread"] != -math.inf else None,
                "binding_gain": _fin(r["binding_gain"]) if r["binding_gain"] != -math.inf else None,
                "wave_tiers": r["tiers"],
            }
            for r in ranked
        ],
    }


def _thresholds() -> dict[str, Any]:
    """The gate's numbers, verbatim from the design (fresh dict per result)."""
    return {
        "PASS": {"spread_min": PASS_SPREAD, "gain_min": PASS_GAIN},
        "CONDITIONAL": {
            "spread_range": [COND_SPREAD_LO, COND_SPREAD_HI], "gain_min": COND_GAIN
        },
    }


def _undecidable_gate(reason: str, *, read_point: dict[str, Any]) -> dict[str, Any]:
    """A gate result that carries no verdict, with every key a reader expects.

    Used for the evidence failures that are NOT outcomes of the criterion: no
    draws at all, and a pinned read point this run cannot be read at. Neither
    is a FAIL -- a FAIL is a measured flat outcome space.
    """
    return {
        "best_m": read_point.get("m"),
        "best_m_rule": "not applicable -- the gate was not read",
        "ranking": [],
        "per_wave": [],
        "thresholds": _thresholds(),
        "edge_rules_applied": [],
        "borderline": False,
        "third_wave_required": False,
        "read_point": read_point,
        "verdict": "borderline-undecided",
        "reason": reason,
    }


def evaluate_gate(
    waves: Sequence[dict[str, Any]],
    ms: Sequence[int],
    *,
    third_wave_possible: bool,
    required_n: int | Sequence[int],
    fixed_m: int | None = None,
) -> dict[str, Any]:
    """PASS / CONDITIONAL / FAIL / borderline-undecided with its exact numbers.

    ``required_n`` is the pre-registered draw complement a wave must hold at the
    read point -- one int for all waves, or one per wave in order. A wave below
    it does NOT decide the gate: the verdict is ``borderline-undecided``, since
    a wave that lost draws is a thinner pool whose spread and gain are not the
    pre-registered statistic (and a missing draw is never resampled).

    ``fixed_m`` pins the read point to the m the FIRST TWO waves selected. A
    third wave is called to break a tie AT that m, so it must not be allowed to
    move it; a pin that is not in ``ms`` means the waves were not measured at a
    common horizon and fails closed.
    """
    if not waves:
        return _undecidable_gate(
            "no wave produced a draw, so there is no m to read the gate at and no "
            "verdict exists: this is an INCOMPLETE run, not a FAIL",
            read_point={"m": None, "pinned": False, "free_best_m": None},
        )
    if fixed_m is not None and int(fixed_m) not in [int(x) for x in ms]:
        return _undecidable_gate(
            f"the pinned read point m={fixed_m} (selected by the first two waves) "
            f"is not among this run's probe steps {[int(x) for x in ms]}, so the "
            "waves were not all measured at a common horizon",
            read_point={"m": int(fixed_m), "pinned": True, "in_ms": False},
        )
    if isinstance(required_n, int):
        needs = [required_n] * len(waves)
    else:
        needs = [int(v) for v in required_n]
        if len(needs) != len(waves):
            raise ValueError(
                f"required_n has {len(needs)} entries for {len(waves)} wave(s)"
            )
    pick = pick_best_m(waves, ms)
    free_m = pick["best_m"]
    pinned = fixed_m is not None
    m = int(fixed_m) if pinned else free_m
    per_wave = [
        {
            "wave": w["wave"],
            "n": w["metrics"][str(m)]["n"],
            "required_n": need,
            "complete": w["metrics"][str(m)]["n"] >= need,
            "median": w["metrics"][str(m)]["median"],
            "max": w["metrics"][str(m)]["max"],
            "min": w["metrics"][str(m)]["min"],
            "spread": w["metrics"][str(m)]["spread"],
            "gain": w["metrics"][str(m)]["gain"],
            "edge": w["metrics"][str(m)]["edge"],
            "tier": w["metrics"][str(m)]["tier"],
            "missing": w["metrics"][str(m)].get("missing") or [],
        }
        for w, need in zip(waves, needs)
    ]
    tiers = [p["tier"] for p in per_wave]
    out: dict[str, Any] = {
        "best_m": m,
        "best_m_rule": (
            "PINNED to the m the first two waves selected (binding tier, then "
            "binding spread, then binding gain, ties to the later m), so the "
            "third wave cannot move the read point it was called to settle"
            if pinned else
            "binding (worst-across-waves) tier, then binding spread, then binding "
            "gain, ties to the later m"
        ),
        "ranking": pick["ranking"],
        "per_wave": per_wave,
        "thresholds": _thresholds(),
        "edge_rules_applied": sorted({p["edge"] for p in per_wave if p["edge"]}),
        "borderline": False,
        "third_wave_required": False,
        "read_point": {
            "m": m,
            "pinned": pinned,
            "free_best_m": free_m,
            "moved_by_pin": bool(pinned and free_m != m),
        },
        "required_n": needs,
    }
    underfilled = [
        {"wave": p["wave"], "n": p["n"], "required_n": p["required_n"],
         "missing": p["missing"]}
        for p in per_wave if not p["complete"]
    ]
    if underfilled:
        out.update(
            verdict="borderline-undecided",
            underfilled_waves=underfilled,
            reason=(
                f"at m={m} "
                + "; ".join(
                    f"wave {u['wave']} holds {u['n']} of {u['required_n']} "
                    f"pre-registered draws"
                    + (f" (missing {', '.join(u['missing'])})" if u["missing"] else "")
                    for u in underfilled
                )
                + ". The gate is read only on waves that drew their full "
                "complement, and a missing draw is never resampled, so the "
                "criterion is formally undecidable on this evidence -- a further "
                "wave cannot repair a wave that lost a draw"
            ),
        )
        return out
    if len(waves) < 2:
        out.update(
            verdict="borderline-undecided",
            reason=(
                f"only {len(waves)} wave completed; the gate requires the criterion "
                "to hold in BOTH waves, so it is formally undecidable"
            ),
        )
        return out
    if len(set(tiers)) == 1:
        out.update(verdict=tiers[0], reason=f"all {len(tiers)} wave(s) agree at m={m}")
        if len(waves) >= 3:
            out["borderline"] = True
            out["reason"] += " (third wave was run after a two-wave disagreement)"
        return out
    counts = {t: tiers.count(t) for t in set(tiers)}
    top = max(counts.values())
    majority = sorted(t for t, c in counts.items() if c == top)
    if len(waves) >= 3 and len(majority) == 1:
        out.update(
            verdict=majority[0],
            borderline=True,
            reason=(
                f"waves disagree at m={m} ({', '.join(tiers)}); majority of "
                f"{len(waves)} waves = {majority[0]}, flagged borderline"
                + (
                    f" (read at the pinned two-wave m={m}"
                    + (f", not the {len(waves)}-wave choice m={free_m}"
                       if free_m != m else "")
                    + ")"
                    if pinned else ""
                )
            ),
        )
        return out
    out.update(
        verdict="borderline-undecided",
        third_wave_required=len(waves) < 3,
        reason=(
            f"waves disagree at m={m} ({', '.join(tiers)}) and "
            + (
                "no majority exists across the waves run"
                if len(waves) >= 3
                else (
                    "a third wave was affordable but its result is not in this report"
                    if third_wave_possible
                    else "the budget could not afford a third wave"
                )
            )
        ),
    )
    return out


def pooling_guard(
    endpoints: Sequence[Sequence[float]], spreads: Sequence[float]
) -> dict[str, Any]:
    """Pool the waves only if location shift <= 0.10 x pooled median AND
    spread ratio <= 2x. Otherwise every read stays per wave."""
    pooled = [v for wave in endpoints for v in wave]
    pooled_med = statistics.median(pooled) if pooled else 0.0
    meds = [statistics.median(w) if w else 0.0 for w in endpoints]
    shift = max(meds) - min(meds) if meds else 0.0
    limit = 0.10 * pooled_med
    shift_ok = shift <= limit if pooled_med > 0 else shift == 0.0
    finite = [s for s in spreads if not math.isinf(s)]
    if not finite or len(finite) < len(spreads):
        ratio: float = math.inf
    elif min(finite) > 0:
        ratio = max(finite) / min(finite)
    else:
        ratio = 1.0 if max(finite) == 0 else math.inf
    ratio_ok = ratio <= 2.0
    return {
        "poolable": bool(shift_ok and ratio_ok),
        "wave_medians": [round(v, 6) for v in meds],
        "pooled_median": round(pooled_med, 6),
        "location_shift": round(shift, 6),
        "location_shift_limit": round(limit, 6),
        "location_shift_ok": bool(shift_ok),
        "wave_spreads": [_fin(s) for s in spreads],
        "spread_ratio": _fin(ratio),
        "spread_ratio_limit": 2.0,
        "spread_ratio_ok": bool(ratio_ok),
    }


def best_of_k(
    draws_by_arm: Mapping[str, Sequence[float]], ks: Sequence[int] = (1, 2, 4, 8), *,
    arms: Sequence[str] | None = None,
    resamples: int = 10000, rng: random.Random | None = None,
) -> dict[str, Any]:
    """Empirical best-of-K curve, resampled at the level the design fans out on.

    A wave is ONE outcome per configured strategy arm, and a deployment running
    best-of-K runs K DIFFERENT arms -- it cannot draw the same strategy twice.
    So a resample here picks the arm subset WITHOUT replacement whenever K does
    not exceed the number of observed arms, and then takes one observed outcome
    for each picked arm (uniformly over the waves that arm was drawn in). The
    old pooled with-replacement bootstrap ignored both facts and inflated the
    curve's variance with combinations the design cannot produce.

    ``draws_by_arm`` maps arm label -> that arm's endpoint outcomes across waves;
    ``arms`` is the configured arm order (labels absent from it are reported as
    unobserved, never silently dropped).
    """
    rng = rng or random.Random(0)
    configured = list(arms) if arms is not None else sorted(draws_by_arm)
    labels = [a for a in configured if draws_by_arm.get(a)]
    extra = [a for a in sorted(draws_by_arm) if a not in configured and draws_by_arm[a]]
    labels += extra
    pool = [float(v) for a in labels for v in draws_by_arm[a]]
    design = {
        "resample_unit": "configured strategy arm (one outcome per arm per resample)",
        "arms_configured": len(configured),
        "arms_observed": len(labels),
        "arms_unobserved": [a for a in configured if not draws_by_arm.get(a)],
        "arms_off_the_configured_list": extra,
        "outcomes_per_arm": {a: len(draws_by_arm[a]) for a in labels},
        "rule": (
            "K <= observed arms -> arm subset WITHOUT replacement; K > observed "
            "arms -> with replacement, flagged per K, because there are not that "
            "many distinct arms to run"
        ),
    }
    if not labels:
        return {
            "status": "no_draws",
            "n_draws": 0,
            "resamples": resamples,
            "median_of_draws": None,
            "curve": {},
            "design": design,
            "note": (
                "no configured arm produced an endpoint draw, so no best-of-K "
                "curve exists (reported as absent, never as zero)"
            ),
        }
    med = statistics.median(pool)
    n_arms = len(labels)
    curve: dict[str, Any] = {}
    base: float | None = None
    for k in ks:
        without = k <= n_arms
        maxima: list[float] = []
        for _ in range(resamples):
            picked = (
                rng.sample(labels, k) if without
                else [rng.choice(labels) for _ in range(k)]
            )
            maxima.append(max(rng.choice(draws_by_arm[a]) for a in picked))
        maxima.sort()
        mean = statistics.fmean(maxima)
        if k == 1:
            base = mean
        curve[str(k)] = {
            "expected_best": round(mean, 6),
            "median_best": round(statistics.median(maxima), 6),
            "sd_best": round(statistics.stdev(maxima), 6) if len(maxima) > 1 else 0.0,
            "p10": round(maxima[int(0.10 * (len(maxima) - 1))], 6),
            "p90": round(maxima[int(0.90 * (len(maxima) - 1))], 6),
            "vs_median_of_draws": round(mean / med, 6) if med > 0 else None,
            "gain_over_k1": round(mean / base - 1.0, 6) if base else None,
            "arm_sampling": "without_replacement" if without else "with_replacement",
            "arms_drawn": min(k, n_arms) if without else k,
        }
    return {
        "status": "ok",
        "n_draws": len(pool),
        "resamples": resamples,
        "median_of_draws": round(med, 6),
        "curve": curve,
        "design": design,
    }


def mc_paired_power(
    n: int, delta: float, sd_diff: float, *, sims: int = 10000,
    rng: random.Random | None = None,
) -> float:
    """Two-sided paired t-test power at alpha=0.05, by Monte-Carlo simulation.

    ``sims`` experiments of ``n`` paired differences drawn from
    ``Normal(delta, sd_diff)``; each is rejected when ``|t| > t_{0.975, n-1}``.
    """
    if n < 2 or n - 1 not in T_CRIT_975:
        return float("nan")
    if delta == 0.0:
        return 0.05
    if sd_diff == 0.0:
        return 1.0
    rng = rng or random.Random(0)
    crit = T_CRIT_975[n - 1]
    root = math.sqrt(n)
    hits = 0
    for _ in range(sims):
        sample = [rng.gauss(delta, sd_diff) for _ in range(n)]
        mean = statistics.fmean(sample)
        sd = statistics.stdev(sample)
        if sd == 0.0:
            hits += 1
            continue
        if abs(mean / (sd / root)) > crit:
            hits += 1
    return hits / sims


def _normal_power(n: int, delta: float, sd_diff: float) -> float:
    """Large-n normal approximation of the same two-sided test.

    Used only to answer "how many replicates WOULD it take" beyond the exact
    Monte-Carlo table: it ignores the t distribution's small-sample penalty, so
    it is OPTIMISTIC and the reported n is a lower bound.
    """
    if sd_diff <= 0.0:
        return 1.0
    z = 1.959963985
    ncp = abs(delta) * math.sqrt(n) / sd_diff
    phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    return phi(ncp - z) + phi(-ncp - z)


def mde(
    n: int, sd_diff: float, *, target: float = TARGET_POWER, sims: int = 4000,
    seed: int = 0, hi_mult: float = 20.0,
) -> float | None:
    """Smallest effect paired n resolves at ``target`` power (bisection on the
    Monte-Carlo power, re-seeded per evaluation so the search is monotone)."""
    if sd_diff <= 0.0:
        return 0.0
    lo, hi = 0.0, hi_mult * sd_diff
    if mc_paired_power(n, hi, sd_diff, sims=sims, rng=random.Random(seed)) < target:
        return None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if mc_paired_power(n, mid, sd_diff, sims=sims, rng=random.Random(seed)) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def power_check(
    draws: Sequence[float], *, sigma: float, sims: int = 10000,
    rng: random.Random | None = None, n_max: int = 12,
    scale: float | None = None, scale_kind: str = "median of the endpoints",
) -> dict[str, Any]:
    """Can paired n=3 resolve a CONDITIONAL-magnitude effect (0.10 x median)?

    ``scale`` overrides the reference level the 0.10 effect is taken of (used
    for the secondary arm-level read, where the arm reports a best-of-K
    endpoint rather than a single branch draw).
    """
    med = statistics.median(draws) if draws else 0.0
    ref = med if scale is None else float(scale)
    delta = 0.10 * ref
    sd_diff = sigma * math.sqrt(2.0)
    curve = {
        str(n): round(mc_paired_power(n, delta, sd_diff, sims=sims, rng=rng), 4)
        for n in range(2, n_max + 1)
    }
    p3 = curve.get("3", float("nan"))
    n_approx: int | None = None
    mde3 = mde(3, sd_diff, sims=max(2000, sims // 4))
    if delta == 0.0:
        rec, detail = "primary-only", (
            "the endpoint median is 0, so a 0.10-of-median effect is not a "
            "defined quantity; Exp 2 cannot be powered against it"
        )
        n_needed: int | None = None
    elif p3 >= TARGET_POWER:
        rec, n_needed = "n=3 OK", 3
        detail = f"paired n=3 reaches power {p3:.2f} >= {TARGET_POWER:.2f}"
    else:
        n_needed = next(
            (n for n in range(3, n_max + 1) if curve[str(n)] >= TARGET_POWER), None
        )
        n_approx = next(
            (n for n in range(2, 4001) if _normal_power(n, delta, sd_diff) >= TARGET_POWER),
            None,
        )
        if n_needed is not None:
            rec = f"raise n to {n_needed}"
            detail = (
                f"paired n=3 reaches only power {p3:.2f}; n={n_needed} is the "
                f"smallest replicate count at or above {TARGET_POWER:.2f}"
            )
        else:
            rec = "primary-only"
            detail = (
                f"paired n=3 reaches only power {p3:.2f} and no n <= {n_max} "
                f"reaches {TARGET_POWER:.2f}"
                + (
                    f"; the normal approximation puts the requirement at n ~ "
                    f"{n_approx} (optimistic lower bound), far beyond the design's "
                    "n=3 -- so either raise n by that order or read the primary "
                    "contrast only"
                    if n_approx else
                    "; the requirement is beyond n=4000, so read the primary "
                    "contrast only"
                )
            )
    return {
        "method": (
            "Monte-Carlo two-sided paired t-test at alpha=0.05: sims experiments "
            "of n paired differences drawn from Normal(delta, sd_diff), rejected "
            "when |t| > t_{0.975,n-1}. sd_diff = sigma_within x sqrt(2), i.e. no "
            "credit for within-pair correlation (conservative). n beyond the "
            "tabulated range and the minimum detectable effect come from the same "
            "test -- the former by normal approximation (optimistic, a lower "
            "bound), the latter by bisection on the Monte-Carlo power."
        ),
        "sims": sims,
        "alpha": 0.05,
        "target_power": TARGET_POWER,
        "endpoint_median": round(med, 6),
        "reference_level": round(ref, 6),
        "reference_kind": scale_kind,
        "target_effect": round(delta, 6),
        "sigma_within": round(sigma, 6),
        "sd_paired_diff": round(sd_diff, 6),
        "power_by_n": curve,
        "power_n3": p3,
        "n_required": n_needed,
        "n_required_normal_approx": n_approx,
        "mde_n3": None if mde3 is None else round(mde3, 6),
        "mde_n3_as_fraction_of_reference": (
            None if (mde3 is None or ref <= 0) else round(mde3 / ref, 4)
        ),
        "recommendation": rec,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Transcript-derived branch summaries
# ---------------------------------------------------------------------------


def api_calls(messages: Iterable[dict[str, str]]) -> dict[str, int]:
    """FLE API calls the branch actually wrote, counted from its transcript."""
    counts: dict[str, int] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for name in _CALL_RE.findall(msg.get("content") or ""):
            if name in FLE_VERBS:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def action_steps(messages: Iterable[dict[str, str]]) -> tuple[int, int]:
    """(steps that called at least one FLE API, steps that called none).

    The shared extractor (``fle.agents.llm.parsing.PythonParser`` via
    :func:`bench.llm.extract_code`) turns a response with NO code block into a
    single comment line, which executes as a valid no-op program rather than
    raising an unparseable-response incident. Such a step advances nothing, so
    a branch whose every step is prose-only is a de-facto do-nothing draw and
    has to be identified as one instead of being read as a failed build.
    """
    acted = prose = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if any(name in FLE_VERBS for name in _CALL_RE.findall(msg.get("content") or "")):
            acted += 1
        else:
            prose += 1
    return acted, prose


def read_transcripts(journal_path: str) -> dict[str, list[dict[str, str]]]:
    """Latest archived transcript per branch, from the run journal."""
    out: dict[str, list[dict[str, str]]] = {}
    try:
        with open(journal_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"branch_archive"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "branch_archive" and rec.get("messages"):
                    out[str(rec.get("branch"))] = rec["messages"]
    except OSError:
        pass
    return out


def branch_summary(rec: dict[str, Any], ms: Sequence[int]) -> str:
    """One line, entirely derived from the branch's own journal records."""
    if rec.get("status") != "ok":
        return f"{rec.get('hint_label', '?')}: FAILED -- {rec.get('error', 'unknown')}"
    scores = " / ".join(
        ("-" if rec["scores"].get(str(m)) is None else f"{rec['scores'][str(m)]:.1f}")
        for m in ms
    )
    top = ", ".join(f"{k}x{v}" for k, v in list(rec.get("api_calls", {}).items())[:4])
    return (
        f"{rec['hint_label']}: entities {rec.get('entities_start')}->"
        f"{rec.get('entities_end')}, production {rec.get('production_start', 0):.0f}->"
        f"{rec.get('production_end', 0):.0f}, probe m="
        f"{'/'.join(str(m) for m in ms)} {scores}; "
        f"{rec.get('steps', 0)} steps"
        + (
            f" ({rec['acted_steps']} of {rec.get('steps', 0)} called the game API)"
            if rec.get("acted_steps") is not None else ""
        )
        + f", {rec.get('errors', 0)} error(s); "
        f"top calls {top or 'none'}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BarrierSkewError(RuntimeError):
    """A wave's children could not be released from a common start.

    Raised after the wave's evidence (barrier record, per-branch reasons) is
    saved: an unequal start is a repeatable confound under positional hints, so
    the wave is recorded with no draws instead of being measured.
    """


def _tick_or_none(meta: Mapping[str, Any]) -> int | None:
    """Real game tick from ``/meta``, or ``None`` when it cannot be read.

    :func:`bench.arms._tick_of` defaults to 0, which is right for a curve label
    and wrong for a skew check -- 0 would answer a question nothing measured.
    """
    for key in ("game_tick", "elapsed_ticks"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _skew(ticks: Iterable[int | None]) -> int | None:
    """Spread of the readable ticks; ``None`` when none of them is readable."""
    known = [t for t in ticks if t is not None]
    return max(known) - min(known) if known else None


class Exp1Runner:
    def __init__(self, cfg: Exp1Config) -> None:
        self.cfg = cfg
        # The pre-registered draw complement a wave must hold for the gate to be
        # readable (see :func:`evaluate_gate`); nonsense here would silently
        # weaken the gate, so it is refused before any artefact is created.
        want = cfg.min_draws_per_wave
        if want is not None and not 1 <= int(want) <= cfg.k:
            raise ValueError(
                f"min_draws_per_wave={want!r} is not a usable complement for "
                f"K={cfg.k}: it must be between 1 and K"
            )
        self.required_draws = cfg.k if want is None else int(want)
        os.makedirs(cfg.journal_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
        self.journal = RunJournal(
            os.path.join(cfg.journal_dir, "exp1.jsonl"),
            run_id=cfg.run_id,
            meta={"experiment": EXPERIMENT_KEY, "config": cfg.to_dict()},
        )
        self.fp = Farplane(prefix=cfg.prefix)
        self.llm = make_client(
            cfg.model, journal=self.journal, max_concurrency=cfg.max_llm_concurrency
        )
        arm_cfg = ArmConfig(
            arm="AxK",
            model=cfg.model,
            task_key=cfg.task_key,
            K=cfg.k,
            m=cfg.probe_every,
            T_s=cfg.budget_s,
            template_snap=cfg.template_snap,
            ttl_s=cfg.ttl_s,
            prefix=cfg.prefix,
            run_id=cfg.run_id,
            journal_dir=cfg.journal_dir,
            # ``T_s`` here is the WHOLE-EXPERIMENT budget (waves plus analysis),
            # not one sandbox's horizon: Exp 1 creates and deletes a fresh set of
            # branch sandboxes per 12-step wave, each far inside ``ttl_s``. The
            # pre-flight lease guard is about a single line running to T, so it
            # does not apply to this container config.
            lease_guard=False,
        )
        self.run = ArmRun(
            arm_cfg,
            farplane=self.fp,
            bridge_factory=default_bridge_factory(),
            llm=self.llm,
            journal=self.journal,
        )
        self.run.timings.start()
        self.t0 = time.monotonic()
        self.ms = tuple(
            range(cfg.probe_every, cfg.steps + 1, cfg.probe_every)
        )
        self.state = self._load_state()
        # Evidence carried over from earlier invocations of this results file.
        # Cumulative counters are merged onto these on every save, so a resumed
        # wave or an analysis-only pass adds to the record instead of replacing
        # it with its own (near-empty) slice.
        self._prior_incidents: list[dict[str, Any]] = list(
            self.state.get("incidents") or []
        )
        self._prior_usage: dict[str, Any] = dict(self.state.get("llm_usage") or {})
        self._prior_timings: dict[str, Any] = dict(self.state.get("timings") or {})

    # -- state -------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        """Resume this results file, or refuse to.

        The file is EVIDENCE: its draws were taken under one measurement config.
        Resuming it with a different value for any of :data:`MEASUREMENT_KEYS`
        would mix two measurements under one verdict, and an unreadable file used
        to be replaced by a fresh state (silently overwriting the run it could
        not parse). Both are refused. Operational changes -- paths, budget,
        bootstrap seed and resamples -- are allowed and recorded.
        """
        path = self.cfg.results_path
        current = self.cfg.to_dict()
        if not os.path.exists(path):
            return {
                "experiment": EXPERIMENT_KEY,
                "design_ref": "fanout-benchmark-design.md :: Experiment 1",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": current,
                "config_fingerprint": measurement_fingerprint(current),
                "model_info": self.llm.model_info(),
                "task": {},
                "s2": None,
                "waves": [],
                "analysis": None,
                "incidents": [],
                "residual": None,
            }
        try:
            with open(path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{path} exists but is not readable as results JSON "
                f"({type(exc).__name__}: {exc}). It is this experiment's evidence, "
                "so it is never overwritten: move it aside, or pass a new "
                "--results path."
            ) from exc
        if not isinstance(state, dict) or state.get("experiment") != EXPERIMENT_KEY:
            raise RuntimeError(
                f"{path} is not an {EXPERIMENT_KEY} results file (experiment="
                f"{(state.get('experiment') if isinstance(state, dict) else None)!r}); "
                "pass a new --results path instead of overwriting it."
            )
        saved = state.get("config")
        if not isinstance(saved, dict) or not saved:
            raise RuntimeError(
                f"{path} records no config, so its draws cannot be shown to belong "
                "to this measurement; pass a new --results path."
            )
        recorded = [k for k in MEASUREMENT_KEYS if k in saved]
        mismatch = config_delta(saved, current, recorded)
        if mismatch:
            raise RuntimeError(
                f"refusing to resume {path}: it holds draws measured under a "
                "different config ("
                + "; ".join(
                    f"{k}: saved {v['saved']!r}, now {v['now']!r}"
                    for k, v in sorted(mismatch.items())
                )
                + "). Draws from two measurement configs are never mixed under one "
                "verdict: run this config into a NEW --results (and --report) path, "
                "or restore the recorded config to continue this one."
            )
        fingerprint = measurement_fingerprint(current)
        # Keys this file predates cannot be checked -- they are named in the
        # artefact rather than waved through silently.
        legacy = [k for k in MEASUREMENT_KEYS if k not in saved]
        if legacy:
            fingerprint["legacy_unrecorded_keys"] = legacy
        ops = config_delta(
            saved, current, [k for k in current if k not in MEASUREMENT_KEYS]
        )
        if ops:
            state.setdefault("config_changes", []).append({
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "changed": ops,
            })
        state["config"] = current
        state["config_fingerprint"] = fingerprint
        return state

    def save(self) -> None:
        self.state["config"] = self.cfg.to_dict()
        self.state["config_fingerprint"] = self.state.get(
            "config_fingerprint"
        ) or measurement_fingerprint(self.cfg.to_dict())
        self.state["model_info"] = self.llm.model_info()
        self.state["task"] = {
            "key": self.cfg.task_key,
            "entity": self.run.entity,
            "quota": self.run.quota,
            "goal": self.run.goal,
        }
        # Cumulative across invocations of this results file: a resumed wave or
        # an analysis-only pass has incidents, usage and intervals of its OWN
        # only, so replacing these would delete the run's evidence. Merging is
        # a no-op when this invocation contributed nothing.
        self.state["incidents"] = self._prior_incidents + list(self.run.incidents)
        self.state["llm_usage"] = merge_counters(self._prior_usage, self.llm.usage())
        if self.run.timings.intervals:
            self.state["timings"] = merge_timings(
                self._prior_timings, self.run.timings.summary()
            )
        elif self._prior_timings:
            self.state["timings"] = self._prior_timings
        else:
            self.state["timings"] = self.run.timings.summary()
        self.state["elapsed_s"] = round(
            max(time.monotonic() - self.t0, float(self.state.get("elapsed_s") or 0.0)), 1
        )
        self.state["journals"] = {
            "run": str(self.journal.path),
            "farplane": str(self.fp.journal_path),
        }
        atomic_write_json(self.cfg.results_path, self.state)

    # -- budget ------------------------------------------------------------
    def remaining_s(self) -> float:
        return self.cfg.budget_s - (time.monotonic() - self.t0)

    def usable_s(self) -> float:
        return self.remaining_s() - self.cfg.reserve_s

    # -- bake --------------------------------------------------------------
    async def bake(self) -> str:
        cfg = self.cfg
        target = cfg.bake_target_multiple * self.run.quota
        self.journal.event(
            "bake_start", template_snap=cfg.template_snap, target_throughput=target,
            max_steps=cfg.bake_steps, quota=self.run.quota,
        )
        node = await self.run.provision_main("s2bake")
        healthy = await asyncio.to_thread(node.bridge.health)
        meta0 = await self.run.infra.meta(node, branch="bake")
        baseline = await self.run.read_baseline(node, "bake")
        if baseline is None:
            # Every production number the bake reports is a delta from this, and
            # S2 is the input to every wave: a checkpoint whose starting counters
            # were never read cannot be described, so it is not taken.
            raise RuntimeError(
                "bake baseline unreadable (/execute could not read the cumulative "
                "counters); refusing to snapshot an S2 this run cannot describe"
            )
        prod0, auto0 = baseline["production"], baseline["automated"]
        traj = Trajectory(tid="bake", node=node, conv=self.run.new_conversation())
        traj.last_production, traj.last_automated = prod0, auto0
        traj.last_ticks = _tick_of(meta0)
        self.journal.event(
            "bake_ready", sandbox=node.id, sandbox_name=node.name,
            fp_node=getattr(node.sb, "node", ""), health=bool(healthy),
            entity_count=meta0.get("entity_count"), tick=traj.last_ticks,
            production=prod0,
        )
        probes: list[dict[str, Any]] = []
        reason = "step_cap"
        for step in range(1, cfg.bake_steps + 1):
            await self.run.agent_step(traj)
            last = step == cfg.bake_steps
            if step % cfg.probe_every == 0 or last:
                probe = await self.run.probe_line(
                    node, branch="bake", step=step, kind="bake"
                )
                if probe is None:
                    continue
                traj.conv.inject(self.run.probe_block(probe))
                traj.curve.add(t_s=time.monotonic() - self.t0, step=step,
                               throughput=probe["throughput"], branch="bake",
                               kind="bake")
                probes.append({"step": step, "throughput": probe["throughput"]})
                if probe["throughput"] >= target:
                    reason = "milestone_throughput"
                    break
        meta1 = await self.run.infra.meta(node, branch="bake")
        self.journal.event(
            "bake_milestone", reason=reason, target_throughput=target,
            steps=traj.step, probes=probes, errors=traj.errors,
            entity_count=meta1.get("entity_count"), tick=_tick_of(meta1),
            production=traj.last_production, automated=traj.last_automated,
        )
        snap = await self._snapshot(node, note="exp1-S2")
        self.journal.archive_branch(
            branch="bake", step=traj.step, messages=traj.conv.messages,
            score={"probes": probes}, reason="exp1-bake",
        )
        self.state["s2"] = {
            "snapshot": snap,
            "source_sandbox": node.id,
            "source_name": node.name,
            "fp_node": getattr(node.sb, "node", ""),
            "snapshot_ttl": self.cfg.snapshot_ttl,
            "milestone": {
                "reason": reason,
                "target_throughput": target,
                "reached_throughput": probes[-1]["throughput"] if probes else None,
                "steps": traj.step,
                "probes": probes,
            },
            "bake": {
                "template_snap": cfg.template_snap,
                "steps": traj.step,
                "errors": traj.errors,
                "entities_start": meta0.get("entity_count"),
                "entities_end": meta1.get("entity_count"),
                "production_start": prod0,
                "production_end": traj.last_production,
                "automated_end": traj.last_automated,
                "api_calls": api_calls(traj.conv.messages),
                "curve": traj.curve.points,
            },
        }
        # The bake sandbox is scaffolding once S2 exists: every branch of every
        # wave forks from the SNAPSHOT, so keeping the source alive would only
        # hold a warm slot on the node the forks have to land on. Deleted here,
        # journaled as a decision.
        self.journal.event(
            "bake_teardown_decision",
            decision="delete bake sandbox after snapshot",
            rationale=(
                "branches fork from S2, not from the bake sandbox; holding it "
                "alive would occupy a slot on the same node the forks pin to"
            ),
            sandbox=node.id,
        )
        await self.run.infra.delete(node)
        self.save()
        return snap

    async def _snapshot(self, node: Node, *, note: str) -> str:
        """``Infra.snapshot`` with an explicit snapshot TTL (S2 must outlive the run)."""
        t0 = time.monotonic()
        try:
            snap = await asyncio.to_thread(
                lambda: self.fp.snapshot(
                    node.sb, ttl=self.cfg.snapshot_ttl, note=note
                )
            )
        except BaseException as exc:  # noqa: BLE001 - journaled then re-raised
            dt = time.monotonic() - t0
            self.run.timings.record("infra_snapshot", t0, t0 + dt, f"snapshot:{node.name}")
            self.journal.infra_op(
                op="snapshot", bucket="infra_snapshot", duration_s=dt,
                outcome="error", target=node.name,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            raise
        dt = time.monotonic() - t0
        self.run.timings.record("infra_snapshot", t0, t0 + dt, f"snapshot:{node.name}")
        self.journal.infra_op(
            op="snapshot", bucket="infra_snapshot", duration_s=dt, target=node.name,
            snapshot=snap, ttl=self.cfg.snapshot_ttl, note=note,
        )
        self.run.infra.snapshots_created += 1
        return snap

    # -- wave --------------------------------------------------------------
    async def wave(self, w: int) -> dict[str, Any]:
        cfg = self.cfg
        s2 = (self.state.get("s2") or {}).get("snapshot")
        if not s2:
            raise RuntimeError("no S2 snapshot in state; run the bake phase first")
        self.journal.event("wave_start", wave=w, snapshot=s2, k=cfg.k,
                           steps=cfg.steps, probe_at=list(self.ms),
                           remaining_s=round(self.remaining_s(), 1))
        forks: list[dict[str, Any]] = []
        nodes: dict[int, Node] = {}
        for i in range(1, cfg.k + 1):
            role = f"w{w}b{i}"
            mark = len(self.fp.timings)
            t0 = time.monotonic()
            try:
                node = await self.run.infra.fork(s2, role)
            except asyncio.CancelledError:
                raise  # T-deadline cancel: stop forking, do not fill missing draws
            except BaseException as exc:  # noqa: BLE001 - a missing draw, journaled
                detail = f"{type(exc).__name__}: {exc}"
                self.run.incident("fork_failed", detail, branch=role, wave=w)
                forks.append({
                    "branch": role, "outcome": "error", "error": detail[:500],
                    "wall_s": round(time.monotonic() - t0, 3),
                    **self._fork_journal_facts(mark),
                })
                continue
            nodes[i] = node
            forks.append({
                "branch": role, "outcome": "ok", "sandbox": node.id,
                "name": node.name, "fp_node": getattr(node.sb, "node", ""),
                "wall_s": round(time.monotonic() - t0, 3),
                **self._fork_journal_facts(mark),
            })
        self.journal.event("wave_forks_ready", wave=w, ok=len(nodes),
                           requested=cfg.k,
                           fork_wall_s=[f["wall_s"] for f in forks])
        # Children evolve while the remaining forks are taken; nothing is
        # measured until they are back on a common start (see _release_barrier).
        unusable: dict[int, str] = {}
        barrier = await self._release_barrier(w, nodes, unusable)
        if barrier.get("outcome") == "aborted_skew":
            # An unequal start is not a measurement: the children are torn down,
            # the wave is recorded with no draws, and the abort is re-raised
            # after the evidence is saved.
            await asyncio.gather(
                *(self._delete(node) for node in nodes.values()),
                return_exceptions=True,
            )
            for i in list(nodes):
                unusable[i] = barrier["abort_reason"][:500]
            nodes.clear()
        results = await asyncio.gather(
            *(
                self.branch(w, i, nodes[i], STRATEGY_HINTS[(i - 1) % len(STRATEGY_HINTS)])
                for i in sorted(nodes)
            ),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, asyncio.CancelledError):
                raise res  # wave aborted: propagate, never record k failed draws
        branches: list[dict[str, Any]] = []
        for i, res in zip(sorted(nodes), results):
            if isinstance(res, BaseException):
                label, _ = STRATEGY_HINTS[(i - 1) % len(STRATEGY_HINTS)]
                detail = f"{type(res).__name__}: {res}"
                self.run.incident("branch_crashed", detail, branch=f"w{w}b{i}", wave=w)
                branches.append({
                    "branch": f"w{w}b{i}", "hint_label": label, "status": "failed",
                    "error": detail[:500], "scores": {},
                })
            else:
                branches.append(res)
        for i in range(1, cfg.k + 1):
            if i not in nodes:
                label, hint = STRATEGY_HINTS[(i - 1) % len(STRATEGY_HINTS)]
                branches.append({
                    "branch": f"w{w}b{i}", "hint_label": label, "hint": hint,
                    "status": "failed",
                    "error": unusable.get(i, "fork/health failed -- no sandbox"),
                    "scores": {},
                })
        branches.sort(key=lambda b: b["branch"])
        # Branch sandboxes are wave-local: the next wave forks fresh children
        # from the same S2, so nothing here is reused.
        await asyncio.gather(
            *(self._delete(node) for node in nodes.values()), return_exceptions=True
        )
        for rec in branches:
            rec["summary"] = branch_summary(rec, self.ms)
        wave_rec = {
            "wave": w,
            "snapshot": s2,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "aborted" if barrier.get("outcome") == "aborted_skew" else "ok",
            "k_requested": cfg.k,
            "k_drawn": sum(1 for b in branches if b["status"] == "ok"),
            "required_draws": self.required_draws,
            "barrier": barrier,
            "forks": forks,
            "branches": branches,
            "metrics": wave_metrics(branches, self.ms),
        }
        self.state["waves"] = [
            x for x in self.state.get("waves", []) if x.get("wave") != w
        ] + [wave_rec]
        self.state["waves"].sort(key=lambda x: x["wave"])
        self.journal.event("wave_done", wave=w, k_drawn=wave_rec["k_drawn"],
                           status=wave_rec["status"],
                           barrier=barrier.get("outcome"),
                           metrics={m: {k: v for k, v in rec.items()
                                        if not k.startswith("_")}
                                    for m, rec in wave_rec["metrics"].items()})
        self.save()
        if barrier.get("outcome") == "aborted_skew":
            raise BarrierSkewError(barrier["abort_reason"])
        return wave_rec

    async def _child_meta(
        self, w: int, nodes: dict[int, Node]
    ) -> dict[int, dict[str, Any]]:
        """``/meta`` for every child concurrently: real game tick + entity count.

        A tick that cannot be read stays ``None``. It is never defaulted to 0 --
        a fabricated tick would make the barrier's skew check answer a question
        it did not measure.
        """
        order = sorted(nodes)
        results = await asyncio.gather(
            *(self.run.infra.meta(nodes[i], branch=f"w{w}b{i}") for i in order),
            return_exceptions=True,
        )
        out: dict[int, dict[str, Any]] = {}
        for i, res in zip(order, results):
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, BaseException):
                detail = f"{type(res).__name__}: {res}"
                self.run.incident("barrier_meta_failed", detail, wave=w,
                                  branch=f"w{w}b{i}")
                out[i] = {"tick": None, "entity_count": None, "error": detail[:300]}
                continue
            out[i] = {"tick": _tick_or_none(res),
                      "entity_count": res.get("entity_count")}
        return out

    async def _release_barrier(
        self, w: int, nodes: dict[int, Node], unusable: dict[int, str]
    ) -> dict[str, Any]:
        """Put every forked child back on a common start, then release the wave.

        Forks of one lineage serialise, so by the time child K exists child 1's
        world has been RUNNING for the whole fork loop -- and because the strategy
        hints are positional, that head start would be a repeatable confound
        rather than noise. The strongest equaliser the bridge API offers is a
        state round-trip: capture the LEAST advanced child's world (the closest
        thing to S2 still in hand) with ``/state-save`` and restore it onto every
        child with ``/state-restore``, the capture source included so the
        restore's own losses (ore replenished, fluid boxes dropped, production
        counters reset) land on all K identically. Each branch reads its own
        baseline after this, so those resets are inside every draw's baseline.

        A child whose restore fails is DROPPED as a missing draw -- never left to
        run from an unequal start. If the state endpoints are unavailable at all,
        the wave proceeds only while the measured tick skew is inside
        ``cfg.barrier_skew_tolerance_ticks``, and is aborted past it.
        """
        if not nodes:
            return {"outcome": "no_children", "k": 0,
                    "note": "no child was forked, so there was nothing to equalise"}
        t0 = time.monotonic()
        pre = await self._child_meta(w, nodes)
        rec: dict[str, Any] = {
            "k": len(nodes),
            "method": "state-save on the least advanced child, state-restore onto all",
            "tolerance_ticks": self.cfg.barrier_skew_tolerance_ticks,
            "pre_ticks": {f"w{w}b{i}": pre[i]["tick"] for i in sorted(pre)},
            "pre_skew_ticks": _skew(m["tick"] for m in pre.values()),
            "pre_entity_counts": {
                f"w{w}b{i}": pre[i]["entity_count"] for i in sorted(pre)
            },
        }
        known = {i: m["tick"] for i, m in pre.items() if m["tick"] is not None}
        # The least advanced child is the best proxy for S2 still in hand; with
        # no readable tick, the last child forked is the youngest by construction.
        source = min(known, key=lambda i: known[i]) if known else max(nodes)
        rec["source_child"] = f"w{w}b{source}"
        blob: str | None = None
        try:
            blob = await self.run.infra.state_save(
                nodes[source], branch=f"w{w}b{source}"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - journaled, degrades to the guard
            detail = f"{type(exc).__name__}: {exc}"
            self.run.incident("barrier_capture_failed", detail, wave=w,
                              branch=f"w{w}b{source}")
            rec["capture_error"] = detail[:500]
        if blob:
            rec["state_chars"] = len(blob)
            order = sorted(nodes)
            results = await asyncio.gather(
                *(
                    self.run.infra.state_restore(nodes[i], blob, branch=f"w{w}b{i}")
                    for i in order
                ),
                return_exceptions=True,
            )
            restored: list[str] = []
            failed: list[dict[str, Any]] = []
            for i, res in zip(order, results):
                if isinstance(res, asyncio.CancelledError):
                    raise res
                if isinstance(res, BaseException):
                    detail = f"{type(res).__name__}: {res}"
                    self.run.incident("barrier_restore_failed", detail, wave=w,
                                      branch=f"w{w}b{i}")
                    failed.append({"branch": f"w{w}b{i}", "error": detail[:300]})
                    unusable[i] = f"barrier restore failed -- {detail}"[:500]
                    continue
                restored.append(f"w{w}b{i}")
            for entry in failed:
                idx = int(entry["branch"].split("b")[-1])
                node = nodes.pop(idx, None)
                if node is not None:
                    await self._delete(node)
            post = await self._child_meta(w, nodes) if nodes else {}
            counts = [
                m["entity_count"] for m in post.values() if m["entity_count"] is not None
            ]
            rec.update(
                # No survivor is not an equalised wave: it is a wave with no
                # usable child left, and every branch is already recorded as a
                # missing draw by the caller.
                outcome="equalized" if restored else "all_restores_failed",
                restored=restored,
                restore_failed=failed,
                post_ticks={f"w{w}b{i}": post[i]["tick"] for i in sorted(post)},
                post_skew_ticks=_skew(m["tick"] for m in post.values()),
                post_entity_counts={
                    f"w{w}b{i}": post[i]["entity_count"] for i in sorted(post)
                },
                # Restoring identical state must leave identical world content.
                # The reads are a few milliseconds apart on separate running
                # worlds, so a mismatch is reported (and journaled) rather than
                # treated as proof of failure.
                entity_counts_equal=bool(counts) and len(set(counts)) == 1,
                tick_note=(
                    "/state-restore transplants the world without rewinding the "
                    "Factorio clock, so per-child game ticks stay apart; what the "
                    "probe reads is world content, and every branch takes its own "
                    "baseline right after this barrier"
                ),
                wall_s=round(time.monotonic() - t0, 3),
            )
            if not rec["entity_counts_equal"]:
                self.run.incident(
                    "barrier_content_unverified",
                    f"post-restore entity counts differ: {rec['post_entity_counts']}",
                    wave=w,
                )
            self.journal.event("wave_barrier", wave=w, **rec)
            return rec
        # Degraded path: no state round-trip available, so the only defence is
        # the measured skew itself.
        skew = rec["pre_skew_ticks"]
        tolerance = self.cfg.barrier_skew_tolerance_ticks
        if skew is None:
            rec.update(
                outcome="aborted_skew",
                abort_reason=(
                    f"wave {w}: the children could not be equalised "
                    f"({rec.get('capture_error', 'no state capture')}) and no "
                    "child's game tick could be read, so the start skew is "
                    "unknown -- the wave is aborted rather than measured from an "
                    "unverified start"
                ),
            )
        elif skew > tolerance:
            rec.update(
                outcome="aborted_skew",
                abort_reason=(
                    f"wave {w}: the children could not be equalised "
                    f"({rec.get('capture_error', 'no state capture')}) and their "
                    f"start skew is {skew} ticks, past the "
                    f"{tolerance}-tick tolerance -- the positional hints would "
                    "turn that head start into a repeatable confound"
                ),
            )
        else:
            rec.update(
                outcome="skew_within_tolerance",
                note=(
                    f"no state round-trip ({rec.get('capture_error', 'unavailable')}); "
                    f"released on the measured skew of {skew} ticks, inside the "
                    f"{tolerance}-tick tolerance"
                ),
            )
            self.run.incident(
                "barrier_degraded", rec["note"], wave=w,
                skew_ticks=skew, tolerance_ticks=tolerance,
            )
        rec["wall_s"] = round(time.monotonic() - t0, 3)
        self.journal.event("wave_barrier", wave=w, **rec)
        if rec["outcome"] == "aborted_skew":
            self.run.incident("barrier_aborted", rec["abort_reason"], wave=w)
        return rec

    def _fork_journal_facts(self, mark: int) -> dict[str, Any]:
        """Fork attempts / placement, read out of the farplane journal records."""
        facts: dict[str, Any] = {}
        for rec in self.fp.timings[mark:]:
            if rec.get("op") == "fork":
                facts["fork_op_s"] = rec.get("duration_s")
                facts["attempts"] = rec.get("op_attempts")
                facts["poll_count"] = rec.get("poll_count")
            elif rec.get("op") == "fork_child_ready":
                facts["child_ready_s"] = rec.get("duration_s")
                facts["capacity_waits"] = rec.get("capacity_waits")
                facts["placement_lane"] = rec.get("placement_lane")
                facts["not_found_polls"] = rec.get("not_found_polls")
        return facts

    async def _delete(self, node: Node) -> None:
        try:
            await self.run.infra.delete(node)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            self.run.incident("delete_failed", f"{type(exc).__name__}: {exc}",
                              target=node.name)

    async def branch(
        self, w: int, i: int, node: Node, hint_pair: tuple[str, str]
    ) -> dict[str, Any]:
        label, hint = hint_pair
        tid = f"w{w}b{i}"
        rec: dict[str, Any] = {
            "branch": tid, "hint_label": label, "hint": hint, "status": "ok",
            "sandbox": node.id, "name": node.name,
            "fp_node": getattr(node.sb, "node", ""), "scores": {},
            "missing_probes": [], "steps": 0, "errors": 0,
        }
        traj: Trajectory | None = None
        t0 = time.monotonic()
        try:
            conv = self.run.new_conversation()
            # Existing hints mechanism (llm.HINT_TEMPLATE), placed in the
            # branch's FIRST user turn so the strategy is committed to the
            # transcript and stays in context for all 12 steps.
            conv.inject(HINT_TEMPLATE.format(hint=hint))
            traj = Trajectory(tid=tid, node=node, conv=conv)
            meta0 = await self.run.infra.meta(node, branch=tid)
            baseline = await self.run.read_baseline(node, tid)
            if baseline is None:
                # Every score in this experiment is a DELTA from the P5 baseline,
                # so a branch without one is unscorable: it becomes a reported
                # missing draw instead of being credited with the whole
                # cumulative score of the checkpoint it inherited.
                raise RuntimeError(
                    "P5 baseline unreadable -- this branch cannot be scored "
                    "against S2 and is reported as a missing draw"
                )
            prod0, auto0 = baseline["production"], baseline["automated"]
            traj.last_production, traj.last_automated = prod0, auto0
            traj.last_ticks = _tick_of(meta0)
            rec.update(entities_start=meta0.get("entity_count"),
                       tick_start=traj.last_ticks, production_start=prod0)
            self.journal.event("branch_start", wave=w, branch=tid, hint=hint,
                               hint_label=label, sandbox=node.id,
                               entity_count=meta0.get("entity_count"),
                               production=prod0)
            for step in range(1, self.cfg.steps + 1):
                await self.run.agent_step(traj)
                if step % self.cfg.probe_every:
                    continue
                probe = await self.run.probe_line(
                    node, branch=tid, step=step,
                    kind="terminal" if step == self.cfg.steps else "nested",
                )
                if probe is None:
                    rec["missing_probes"].append(step)
                    continue
                traj.conv.inject(self.run.probe_block(probe))
                traj.curve.add(t_s=time.monotonic() - self.t0, step=step,
                               throughput=probe["throughput"], branch=tid,
                               kind="nested")
                rec["scores"][str(step)] = probe["throughput"]
            meta1 = await self.run.infra.meta(node, branch=tid)
            rec.update(
                entities_end=meta1.get("entity_count"), tick_end=_tick_of(meta1),
                production_end=traj.last_production,
                automated_end=traj.last_automated,
                api_calls=api_calls(traj.conv.messages),
                curve=traj.curve.points,
            )
        except asyncio.CancelledError:
            # An aborted wave must propagate, not masquerade as k failed draws.
            raise
        except BaseException as exc:  # noqa: BLE001 - a missing draw, never a resample
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            self.run.incident("branch_failed", rec["error"], branch=tid, wave=w)
        finally:
            if traj is not None:
                rec["steps"] = traj.step
                rec["errors"] = traj.errors
                rec["rollout_s"] = round(time.monotonic() - t0, 1)
                self.journal.archive_branch(
                    branch=tid, step=traj.step, messages=traj.conv.messages,
                    score=rec.get("scores", {}), reason="exp1-branch",
                )
        return rec

    # -- analysis ----------------------------------------------------------
    def analyze(self) -> dict[str, Any]:
        waves = [w for w in self.state.get("waves", []) if w.get("k_drawn")]
        for w in waves:
            w["metrics"] = wave_metrics(w["branches"], self.ms)
        transcripts = read_transcripts(str(self.journal.path))
        for w in waves:
            for b in w["branches"]:
                msgs = transcripts.get(b["branch"])
                if msgs is None:
                    continue
                acted, prose = action_steps(msgs)
                b["acted_steps"], b["prose_only_steps"] = acted, prose
                b["summary"] = branch_summary(b, self.ms)
        rng = random.Random(self.cfg.seed)
        endpoint_m = self.cfg.steps
        third_possible = self.usable_s() >= self.cfg.wave_estimate_s
        # The read point the FIRST TWO waves selected, once a third wave has been
        # called: the extra wave breaks a tie AT that m and must not be able to
        # move the horizon it was called to settle.
        pin = (self.state.get("read_point") or {}).get("m")
        gate = evaluate_gate(
            waves, self.ms, third_wave_possible=third_possible,
            required_n=self.required_draws, fixed_m=pin,
        )
        if pin is None and gate.get("third_wave_required") and len(waves) >= 2:
            self.state["read_point"] = {
                "m": gate["best_m"],
                "pinned_after_waves": [w["wave"] for w in waves],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rule": gate.get("best_m_rule"),
                "reason": (
                    "the first two waves disagreed at this m and called a third "
                    "wave; the majority is read HERE so the extra wave cannot "
                    "shift the horizon it was called to settle"
                ),
            }
            self.journal.event("read_point_pinned", **self.state["read_point"])
        endpoints = [
            [b["scores"][str(endpoint_m)] for b in w["branches"]
             if b.get("scores", {}).get(str(endpoint_m)) is not None]
            for w in waves
        ]
        # The same endpoint draws keyed by the CONFIGURED strategy arm: the
        # best-of-K bootstrap resamples arms (one outcome each), not anonymous
        # pooled draws, because a fan-out wave runs K DIFFERENT strategies.
        arm_labels = list(dict.fromkeys(
            STRATEGY_HINTS[(i - 1) % len(STRATEGY_HINTS)][0]
            for i in range(1, self.cfg.k + 1)
        ))
        endpoints_by_arm: list[dict[str, list[float]]] = []
        for w in waves:
            per_arm: dict[str, list[float]] = {}
            for b in w["branches"]:
                value = b.get("scores", {}).get(str(endpoint_m))
                if value is None:
                    continue
                key = b.get("hint_label") or b["branch"]
                per_arm.setdefault(key, []).append(float(value))
            endpoints_by_arm.append(per_arm)
        pooled_by_arm: dict[str, list[float]] = {}
        for per_arm in endpoints_by_arm:
            for label, values in per_arm.items():
                pooled_by_arm.setdefault(label, []).extend(values)
        analysis: dict[str, Any] = {
            "ms": list(self.ms),
            "endpoint_m": endpoint_m,
            "waves_completed": [w["wave"] for w in waves],
            "waves_aborted": [
                x["wave"] for x in self.state.get("waves", [])
                if x.get("status") == "aborted"
            ],
            "required_draws_per_wave": self.required_draws,
            "read_point": self.state.get("read_point"),
            "barriers": [
                {
                    "wave": x["wave"],
                    # A wave recorded before the barrier existed says so, rather
                    # than reading as a barrier with every field empty.
                    "outcome": "not_recorded",
                    **{
                        k: v for k, v in (x.get("barrier") or {}).items()
                        if k in (
                            "outcome", "source_child", "tolerance_ticks",
                            "pre_skew_ticks", "post_skew_ticks",
                            "entity_counts_equal", "restore_failed",
                        )
                    },
                }
                for x in self.state.get("waves", [])
            ],
            "draws": {
                "requested": self.cfg.k * len(waves),
                "obtained_at_endpoint": sum(len(e) for e in endpoints),
                "missing": [
                    b["branch"] for w in waves for b in w["branches"]
                    if b.get("scores", {}).get(str(endpoint_m)) is None
                ],
            },
            "gate": gate,
            "fork_latency": self.fork_latency(),
            "execution_quality": {
                "note": (
                    "A step counts as an ACTION when the response's program calls "
                    "at least one FLE API. The shared extractor turns a response "
                    "with no code block into a single comment, which executes as a "
                    "valid no-op, so prose-only steps are silent: they are counted "
                    "here rather than raising an incident. A branch with zero "
                    "action steps is a de-facto do-nothing draw and its endpoint "
                    "measures S2 left running, not a strategy outcome."
                ),
                "per_branch": {
                    b["branch"]: {
                        "acted": b.get("acted_steps"),
                        "prose_only": b.get("prose_only_steps"),
                        "endpoint": b.get("scores", {}).get(str(self.cfg.steps)),
                    }
                    for w in waves for b in w["branches"]
                },
                "zero_action_branches": [
                    b["branch"] for w in waves for b in w["branches"]
                    if b.get("acted_steps") == 0
                ],
                "action_steps_total": sum(
                    b.get("acted_steps") or 0 for w in waves for b in w["branches"]
                ),
                "steps_total": sum(
                    (b.get("acted_steps") or 0) + (b.get("prose_only_steps") or 0)
                    for w in waves for b in w["branches"]
                ),
            },
        }
        zero_action = set(analysis["execution_quality"]["zero_action_branches"])
        if zero_action and len(waves) >= 2:
            filtered = []
            needs: list[int] = []
            for w in waves:
                kept = [b for b in w["branches"] if b["branch"] not in zero_action]
                filtered.append({
                    "wave": w["wave"], "branches": kept, "k_drawn": len(kept),
                    "metrics": wave_metrics(kept, self.ms),
                })
                # Dropping the no-ops may not smuggle in a missing draw: every
                # RETAINED branch still has to hold its endpoint.
                needs.append(len(kept))
            # Deliberately NOT pinned to the primary read point -- if dropping
            # the no-ops moves the best m, that is exactly what has to show.
            alt = evaluate_gate(
                filtered, self.ms, third_wave_possible=False, required_n=needs,
            )
            analysis["gate_sensitivity_excluding_no_ops"] = {
                "rationale": (
                    "Robustness check only -- the pre-registered gate above keeps "
                    "every draw. This re-reads it with the zero-action branches "
                    "removed: the primary is robust to them only if BOTH the "
                    "verdict and the m it is read at come back unchanged, so a "
                    "matching verdict at a DIFFERENT m is a change, not a "
                    "confirmation."
                ),
                "excluded": sorted(zero_action),
                "best_m": alt["best_m"],
                "verdict": alt["verdict"],
                "per_wave": alt["per_wave"],
                "required_n": needs,
                "reason": alt.get("reason"),
                "same_verdict": alt["verdict"] == gate.get("verdict"),
                "same_best_m": alt["best_m"] == gate.get("best_m"),
                "matches_primary": bool(
                    alt["verdict"] == gate.get("verdict")
                    and alt["best_m"] == gate.get("best_m")
                ),
            }
        # The pooling guard compares locations and spreads: a wave with no
        # endpoint draw has neither, and running the guard on it would compare
        # against a median of 0 that nothing measured.
        if not any(endpoints):
            analysis["pooling"] = {"poolable": False, "note": "no endpoint draws"}
        elif not all(endpoints):
            analysis["pooling"] = {
                "poolable": False,
                "note": (
                    "wave(s) "
                    + ", ".join(
                        str(waves[i]["wave"]) for i, e in enumerate(endpoints) if not e
                    )
                    + f" produced no m={endpoint_m} endpoint draw, so there is "
                    "nothing to compare locations and spreads across; every read "
                    "stays per wave"
                ),
            }
        elif len(waves) >= 2:
            spreads = [
                w["metrics"][str(endpoint_m)]["_spread"] for w in waves
            ]
            analysis["pooling"] = pooling_guard(endpoints, spreads)
        else:
            analysis["pooling"] = {
                "poolable": False,
                "note": "fewer than two waves completed; nothing to pool",
            }

        sigmas = [statistics.stdev(e) if len(e) > 1 else 0.0 for e in endpoints]
        # Only a wave that HAS endpoint draws can be the variance source; a wave
        # with none has an SD of 0 by convention, which is not a small variance.
        drawn = [i for i, e in enumerate(endpoints) if e]
        worst = max(drawn, key=lambda idx: sigmas[idx]) if drawn else None
        if analysis["pooling"].get("poolable") and len(endpoints) >= 2:
            pool = [v for e in endpoints for v in e]
            analysis["best_of_k"] = {
                "source": "pooled (both waves)",
                **best_of_k(pooled_by_arm, arms=arm_labels,
                            resamples=self.cfg.resamples, rng=rng),
            }
            sigma = statistics.stdev(pool) if len(pool) > 1 else 0.0
            analysis["power"] = {
                "variance_source": f"pooled m={endpoint_m} endpoints",
                **power_check(pool, sigma=sigma, sims=self.cfg.resamples, rng=rng),
            }
        elif worst is not None:
            analysis["best_of_k"] = {
                "source": "per wave (pooling guard refused)",
                "per_wave": {
                    str(w["wave"]): best_of_k(
                        per_arm, arms=arm_labels,
                        resamples=self.cfg.resamples, rng=rng,
                    )
                    for w, per_arm in zip(waves, endpoints_by_arm) if per_arm
                },
            }
            analysis["power"] = {
                "variance_source": (
                    f"wave {waves[worst]['wave']} m={endpoint_m} endpoints "
                    "(larger variance, "
                    "per the pooling guard)"
                ),
                **power_check(
                    endpoints[worst], sigma=sigmas[worst],
                    sims=self.cfg.resamples, rng=rng,
                ),
            }
        else:
            # Not one endpoint draw anywhere: no curve, no variance, no power
            # recommendation. Reported as ABSENT, never as zero.
            analysis["best_of_k"] = {
                "source": "none",
                "status": "no_draws",
                "note": (
                    f"no branch produced an m={endpoint_m} endpoint draw, so there "
                    "is no pool to bootstrap a best-of-K curve from"
                ),
            }
            analysis["power"] = {
                "status": "no_draws",
                "note": (
                    f"no branch produced an m={endpoint_m} endpoint draw, so this "
                    "run measures no within-S variance and cannot recommend an n "
                    "for Exp 2"
                ),
            }
        if "recommendation" in analysis["power"]:
            curve = analysis["best_of_k"].get("curve")
            if curve is None and worst is not None:
                curve = (
                    (analysis["best_of_k"].get("per_wave") or {})
                    .get(str(waves[worst]["wave"]), {})
                    .get("curve")
                )
            sel = None
            if curve:
                sel = curve.get(str(self.cfg.k)) or curve[max(curve, key=int)]
            arm_rationale = (
                "Exp 2's arms report ONE endpoint per run -- the SELECTED best "
                "of K branches -- so the replicate-level variance an Exp 2 "
                "contrast actually faces is the variance of the best-of-K "
                "statistic, not of a single branch draw. SECONDARY read: the "
                "recommendation above is the design's (single-branch within-S "
                "variance). The best-of-K SD is an arm-level bootstrap over the "
                "same endpoints, so it inherits their discreteness."
            )
            if sel is None:
                analysis["power"]["secondary_arm_level"] = {
                    "status": "unavailable",
                    "rationale": arm_rationale,
                    "note": (
                        "The best-of-K curve for this variance source is absent, so "
                        "the arm-level read is reported missing rather than computed "
                        "off a substitute pool."
                    ),
                }
            elif float(sel.get("sd_best") or 0.0) <= 0.0:
                analysis["power"]["secondary_arm_level"] = {
                    "status": "not_estimable",
                    "rationale": arm_rationale,
                    "k": self.cfg.k,
                    "expected_best_of_k": sel["expected_best"],
                    "sd_best_of_k": sel.get("sd_best"),
                    "note": (
                        f"The best-of-{self.cfg.k} statistic has no spread across "
                        "this resample (K covers every observed arm and each arm "
                        "has a single outcome, so there is nothing left to vary), "
                        "so its replicate-level variance is NOT estimable from "
                        "these waves -- reported as such instead of as a power of 1."
                    ),
                }
            else:
                analysis["power"]["secondary_arm_level"] = {
                    "status": "ok",
                    "rationale": arm_rationale,
                    "k": self.cfg.k,
                    "expected_best_of_k": sel["expected_best"],
                    "sd_best_of_k": sel["sd_best"],
                    **power_check(
                        [sel["expected_best"]], sigma=sel["sd_best"],
                        scale=sel["expected_best"],
                        scale_kind=f"expected best-of-{self.cfg.k} endpoint",
                        sims=self.cfg.resamples, rng=rng,
                    ),
                }
        analysis["third_wave_affordable_at_analysis"] = third_possible
        self.state["analysis"] = analysis
        self.save()
        return analysis

    def fork_latency(self) -> dict[str, Any]:
        """Drift guard: this run's fork latency against the settled ~32s solo."""
        waves = self.state.get("waves", [])
        oks = [f for w in waves for f in w.get("forks", []) if f.get("outcome") == "ok"]
        vals = [float(f["wall_s"]) for f in oks if f.get("wall_s") is not None]
        stats = summarize(vals)
        attempts = [f.get("attempts") for f in oks if f.get("attempts") is not None]
        waits = [f.get("capacity_waits") or 0 for f in oks]
        lanes: dict[str, int] = {}
        for f in oks:
            lane = str(f.get("placement_lane") or "")
            lanes[lane] = lanes.get(lane, 0) + 1
        drift = "no data"
        if vals:
            drift = (
                "within 1.5x of the constant"
                if stats["p50"] <= 1.5 * FORK_CONSTANT_S
                else f"DRIFT: p50 {stats['p50']:.1f}s vs {FORK_CONSTANT_S:.0f}s constant"
            )
        return {
            "n": len(vals),
            "constant_s": FORK_CONSTANT_S,
            "wall_s": stats,
            "per_fork_wall_s": [round(v, 2) for v in vals],
            "attempts": attempts,
            "capacity_waits_total": sum(waits),
            "placement_lanes": lanes,
            "failed_forks": [f["branch"] for w in waves
                             for f in w.get("forks", []) if f.get("outcome") != "ok"],
            "drift_guard": drift,
        }

    # -- cleanup -----------------------------------------------------------
    def _prior_sweeps(self) -> list[dict[str, Any]]:
        """Every resource earlier sweeps of this experiment already deleted.

        Read back from the run journal, so a later sweep cannot erase the
        evidence of what an earlier one reaped.
        """
        out: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        try:
            with open(self.journal.path, encoding="utf-8") as fh:
                for line in fh:
                    if '"cleanup"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("name") != "cleanup":
                        continue
                    for v in rec.get("sweep") or []:
                        key = (v.get("kind"), v.get("id"))
                        if key not in seen:
                            seen.add(key)
                            out.append(v)
        except OSError:
            pass
        return out

    async def cleanup(self) -> dict[str, Any]:
        keep = [self.cfg.template_snap]
        s2 = (self.state.get("s2") or {}).get("snapshot")
        if s2:
            keep.append(s2)
        sweep = await asyncio.to_thread(
            lambda: self.fp.reaper(self.cfg.prefix, keep=keep)
        )
        sandboxes = await asyncio.to_thread(self.fp.list_sandboxes)
        images = await asyncio.to_thread(self.fp.list_images)
        live_snaps: set[str] = set()
        expiry: dict[str, Any] = {}
        for img in images:
            snap = img.get("snapshot") or {}
            for value in (
                img.get("id"),
                (img.get("template") or {}).get("sourceSnapshotId"),
                img.get("snapshotId"),
            ):
                if value:
                    live_snaps.add(str(value))
                    exp = snap.get("expiresAt") or img.get("expiresAt")
                    if exp:
                        expiry[str(value)] = exp
        ledger_sb, ledger_snap = self.fp.journal_ledger()
        history = self._prior_sweeps()
        seen = {(v.get("kind"), v.get("id")) for v in history}
        for v in sweep:
            if (v.get("kind"), v.get("id")) not in seen:
                history.append(v)
                seen.add((v.get("kind"), v.get("id")))
        residual = {
            "keep_list": keep,
            "sweep": history,
            "sweep_this_pass": sweep,
            "sandboxes": [
                {"id": sb.get("id"), "name": sb.get("name"), "state": sb.get("state")}
                for sb in sandboxes
                if str(sb.get("name") or "").startswith(self.cfg.prefix)
                or sb.get("id") in ledger_sb
            ],
            "snapshots": sorted(
                {s for s in ledger_snap if s in live_snaps} | {k for k in keep if k in live_snaps}
            ),
            "template_snap_alive": self.cfg.template_snap in live_snaps,
            "s2_alive": bool(s2 and s2 in live_snaps),
            "keep_list_expiry": {k: expiry.get(k) for k in keep},
        }
        residual["clean"] = (
            not residual["sandboxes"]
            and set(residual["snapshots"]) <= set(keep)
            and residual["template_snap_alive"]
            and (residual["s2_alive"] or not s2)
        )
        self.journal.event("cleanup", **residual)
        self.state["residual"] = residual
        self.save()
        return residual

    async def aclose(self) -> None:
        try:
            await self.llm.aclose()
        finally:
            self.run.timings.stop()
            self.journal.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(state: dict[str, Any]) -> str:
    cfg = state.get("config", {})
    analysis = state.get("analysis") or {}
    gate = analysis.get("gate", {})
    ms = analysis.get("ms", [4, 8, 12])
    waves = [w for w in state.get("waves", []) if w.get("k_drawn")]
    s2 = state.get("s2") or {}
    task = state.get("task", {})
    L: list[str] = []
    A = L.append

    A("# Experiment 1 -- decorrelation gate (Farplane fan-out benchmark)")
    A("")
    verdict = gate.get("verdict", "not evaluated")
    flag = " *(borderline)*" if gate.get("borderline") else ""
    A(f"## GATE VERDICT: **{verdict}**{flag}")
    A("")
    A(
        f"- Read at **m = {gate.get('best_m') or '-'}** "
        f"({gate.get('best_m_rule', '-')})."
    )
    rp = gate.get("read_point") or {}
    if rp.get("pinned"):
        A(
            f"- Read point PINNED at m={rp.get('m')} by the first two waves"
            + (
                f"; a free re-selection over all waves would read m="
                f"{rp.get('free_best_m')} instead, which is why it is pinned."
                if rp.get("moved_by_pin") else
                " (a free re-selection over all waves lands on the same m)."
            )
        )
    A(f"- {gate.get('reason', '-')}")
    for pw in gate.get("per_wave", []):
        A(
            f"- wave {pw['wave']}: n={pw['n']}"
            + (
                f" of {pw['required_n']} required"
                + ("" if pw.get("complete") else " -- **INCOMPLETE**")
                if pw.get("required_n") is not None else ""
            )
            + f", median {_fmt(pw['median'],3)}, "
            f"max {_fmt(pw['max'],3)}, min {_fmt(pw['min'],3)} -> "
            f"spread {_fmt(pw['spread'])}, gain {_fmt(pw['gain'])}"
            + (f" [edge: {pw['edge']}]" if pw.get("edge") else "")
            + f" -> **{pw['tier']}**"
        )
    A("")
    A(
        "Gate thresholds (design, verbatim): PASS = spread >= 0.25 AND gain >= 0.15 "
        "in BOTH waves; CONDITIONAL = spread in [0.10, 0.25) AND gain >= 0.10; "
        "FAIL otherwise. Edge rules: median 0 with max > 0 is a PASS-signal; all "
        "draws zero is a FAIL-signal at that m."
    )
    A("")
    consequence = {
        "PASS": "run full Exp 2.",
        "CONDITIONAL": "run Exp 2's primary contrast only.",
        "FAIL": (
            "the curve is flat by construction. Final answer: \"one-shot fan-out "
            "suffices; snapshots are for provisioning, checkpointing, rewind, and "
            "destructive measurement\" -- no further convergence spend."
        ),
        "borderline-undecided": (
            "the gate is formally undecided; do not read it as either outcome."
        ),
    }.get(verdict, "-")
    A(f"**Consequence per the design:** {consequence}")
    A("")
    sens = analysis.get("gate_sensitivity_excluding_no_ops")
    if sens:
        # The primary is robust to the exclusion only if BOTH the verdict and the
        # m it is read at survive it. Anything else is a CHANGE, and saying "so
        # the verdict does not rest on them" would then be false.
        same_verdict = sens.get("same_verdict")
        if same_verdict is None:
            same_verdict = sens.get("verdict") == gate.get("verdict")
        same_m = sens.get("same_best_m")
        if same_m is None:
            same_m = sens.get("best_m") == gate.get("best_m")
        matches = bool(same_verdict and same_m)
        if matches:
            change = (
                ", the same verdict at the same m, so the verdict does not rest "
                "on them."
            )
        elif same_verdict:
            change = (
                f" -- same verdict, but the READ POINT MOVES (primary m="
                f"{gate.get('best_m') or '-'} -> m={sens['best_m']}): the tier "
                "survives the exclusion, which horizon is best supported does not."
            )
        else:
            change = (
                f" -- **CHANGED** from the primary read (**{gate.get('verdict')}** "
                f"at m={gate.get('best_m') or '-'}): the verdict above DOES depend "
                "on the no-op draws being in the pool, and this run does not "
                "establish it without them."
            )
        A(
            f"**Robustness:** dropping the {len(sens['excluded'])} zero-action "
            f"branch(es) ({', '.join('`' + b + '`' for b in sens['excluded'])}) and "
            f"re-reading the same gate gives **{sens['verdict']}** at m="
            f"{sens['best_m']} ("
            + ("; ".join(
                f"wave {p['wave']} spread {_fmt(p['spread'])}, gain {_fmt(p['gain'])}"
                for p in sens["per_wave"]
            ) or "no readable wave")
            + ")"
            + change
            + (
                f" Robustness read's own basis: {sens['reason']}."
                if sens.get("reason") else ""
            )
            + f" {sens['rationale']}"
        )
        A("")

    A("## Setup")
    A("")
    A(
        f"- Model `{cfg.get('model')}` (max_tokens "
        f"{(state.get('model_info') or {}).get('max_tokens')}, temperature "
        f"{(state.get('model_info') or {}).get('temperature')}, locked), task "
        f"`{task.get('key')}`, endpoint item `{task.get('entity')}`, quota "
        f"{task.get('quota')} per 60 in-game seconds."
    )
    A(
        f"- Template snapshot `{cfg.get('template_snap')}` -> bake -> **S2 = "
        f"`{s2.get('snapshot')}`** (kept; TTL {s2.get('snapshot_ttl')})."
    )
    mile = s2.get("milestone", {})
    A(
        f"- Bake milestone: stopped on `{mile.get('reason')}` after "
        f"{mile.get('steps')} steps; target was >= "
        f"{_fmt(mile.get('target_throughput'),1)} {task.get('entity')}/60s "
        f"(2x quota), last bake probe {_fmt(mile.get('reached_throughput'),3)}."
    )
    bake = s2.get("bake", {})
    A(
        f"- S2 state at snapshot: {bake.get('entities_end')} entities "
        f"(from {bake.get('entities_start')}), production score "
        f"{_fmt(bake.get('production_end'),0)}, bake probes "
        + ", ".join(
            f"step {p['step']}: {p['throughput']:.2f}"
            for p in mile.get("probes", [])
        )
        + "."
    )
    equalized_waves = [
        b.get("wave") for b in (analysis.get("barriers") or [])
        if b.get("outcome") == "equalized"
    ]
    A(
        f"- {len(waves)} wave(s) of K={cfg.get('k')} forks from the SAME S2, "
        f"{cfg.get('steps')} steps per branch, probed en route at m="
        f"{'/'.join(str(m) for m in ms)} (nested design; probe cadence = Exp 2's "
        f"parity cadence, so the m={analysis.get('endpoint_m')} endpoints are "
        "read on Exp 2's A*K-from-S cadence"
        + (
            f"; wave(s) {', '.join(str(w) for w in equalized_waves)} were released "
            "through Exp 1's own state barrier on top of that fork path -- see "
            "*Fork barrier* below"
            if equalized_waves else ""
        )
        + ")."
    )
    draws = analysis.get("draws", {})
    A(
        f"- Draws: {draws.get('obtained_at_endpoint')} of "
        f"{draws.get('requested')} at m={analysis.get('endpoint_m')}"
        + (
            f"; MISSING (reported, never resampled): {', '.join(draws.get('missing', []))}"
            if draws.get("missing") else "; no missing draws"
        )
        + "."
    )
    A("")

    A("## Spread and gain per (wave, m)")
    A("")
    A("| wave | m | n | median | max | min | spread (max-min)/median | gain (max-median)/median | tier | edge |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for w in waves:
        for m in ms:
            r = w["metrics"][str(m)]
            A(
                f"| {w['wave']} | {m} | {r['n']} | {_fmt(r['median'],3)} | "
                f"{_fmt(r['max'],3)} | {_fmt(r['min'],3)} | {_fmt(r['spread'])} | "
                f"{_fmt(r['gain'])} | {r['tier']} | {r.get('edge') or '-'} |"
            )
    A("")
    A("Comparators from the design: pilot sibling spreads 0.02 (k3) / 0.098 (codex) were uninformative; probe window-normalisation noise ~0.003; probe repeatability on identical state is exact.")
    A("")
    A("### Which m the gate is read at")
    A("")
    A("| m | binding tier (worst wave) | binding spread | binding gain | per-wave tiers |")
    A("|---|---|---|---|---|")
    for r in gate.get("ranking", []):
        A(
            f"| {r['m']} | {r['binding_tier']} | {_fmt(r['binding_spread'])} | "
            f"{_fmt(r['binding_gain'])} | {', '.join(r['wave_tiers'])} |"
        )
    A("")

    A("## Per-branch scores (raw probe, item/60 in-game seconds)")
    A("")
    A(
        "| branch | strategy | status | " + " | ".join(f"m={m}" for m in ms)
        + " | steps | action steps | errors |"
    )
    A("|---|---|---|" + "---|" * (len(ms) + 3))
    for w in waves:
        for b in w["branches"]:
            cells = " | ".join(_fmt(b.get("scores", {}).get(str(m)), 3) for m in ms)
            acted = b.get("acted_steps")
            A(
                f"| `{b['branch']}` | {b.get('hint_label','-')} | {b.get('status')} | "
                f"{cells} | {b.get('steps',0)} | "
                f"{'-' if acted is None else acted} | {b.get('errors',0)} |"
            )
    A("")
    eq = analysis.get("execution_quality") or {}
    if eq:
        zero = eq.get("zero_action_branches") or []
        A(
            f"Action steps: {eq.get('action_steps_total')} of "
            f"{eq.get('steps_total')} steps across all branches. {eq.get('note')}"
        )
        A("")
        if zero:
            A(
                "- Zero-action branches (their endpoint is S2 left running, i.e. a "
                "passive-decay draw, NOT a strategy outcome): "
                + ", ".join(
                    f"`{b}` (endpoint {_fmt((eq['per_branch'].get(b) or {}).get('endpoint'),3)})"
                    for b in zero
                )
                + "."
            )
            A("")
    A("")

    A("## What each strategy actually built (from the branch journals)")
    A("")
    for w in waves:
        A(f"**Wave {w['wave']}**")
        A("")
        for b in w["branches"]:
            A(f"- `{b['branch']}` {b.get('summary','-')}")
        A("")

    bok = analysis.get("best_of_k", {})
    A(
        "## Best-of-K curve (bootstrap over the "
        f"m={analysis.get('endpoint_m')} endpoints)"
    )
    A("")
    pooling = analysis.get("pooling", {})
    A(
        f"- Pooling guard: **{'POOLED' if pooling.get('poolable') else 'NOT pooled'}**"
        + (
            f" -- location shift {_fmt(pooling.get('location_shift'),3)} vs limit "
            f"{_fmt(pooling.get('location_shift_limit'),3)} (0.10 x pooled median "
            f"{_fmt(pooling.get('pooled_median'),3)}), spread ratio "
            f"{_fmt(pooling.get('spread_ratio'),3)} vs limit 2.0."
            if "location_shift" in pooling else f" -- {pooling.get('note','')}"
        )
    )
    A(f"- Source: {bok.get('source','-')}.")
    bok_design = bok.get("design") or {}
    if bok_design:
        A(
            f"- Resample unit: {bok_design.get('resample_unit')} -- "
            f"{bok_design.get('arms_observed')} of "
            f"{bok_design.get('arms_configured')} configured arms observed"
            + (
                f", unobserved: {', '.join(bok_design.get('arms_unobserved') or [])}"
                if bok_design.get("arms_unobserved") else ""
            )
            + f". {bok_design.get('rule')}."
        )
    if bok.get("note"):
        A(f"- {bok['note']}")
    A("")

    def _curve_table(payload: dict[str, Any], title: str) -> None:
        design = payload.get("design") or {}
        A(f"**{title}** ({payload.get('n_draws')} draws over "
          f"{design.get('arms_observed', '-')} arm(s), "
          f"{payload.get('resamples')} resamples, median of draws "
          f"{_fmt(payload.get('median_of_draws'),3)})")
        A("")
        A("| K | E[best-of-K] | median | SD | p10 | p90 | / median of draws | gain over K=1 | arm sampling |")
        A("|---|---|---|---|---|---|---|---|---|")
        for k, row in payload.get("curve", {}).items():
            A(
                f"| {k} | {_fmt(row['expected_best'],3)} | {_fmt(row['median_best'],3)} | "
                f"{_fmt(row.get('sd_best'),3)} | "
                f"{_fmt(row['p10'],3)} | {_fmt(row['p90'],3)} | "
                f"{_fmt(row['vs_median_of_draws'],3)} | {_fmt(row['gain_over_k1'],4)} | "
                f"{row.get('arm_sampling','-')} |"
            )
        A("")

    if bok.get("curve"):
        _curve_table(bok, "Best-of-K")
    for wave_id, payload in (bok.get("per_wave") or {}).items():
        _curve_table(payload, f"Best-of-K, wave {wave_id}")

    power = analysis.get("power", {})
    A("## Power check for Exp 2 (before spending it)")
    A("")
    if "recommendation" in power:
        A(f"- **Recommendation: {power['recommendation']}** -- {power['detail']}")
        A(f"- Variance source: {power.get('variance_source','-')}.")
        A(
            f"- Within-S sigma {_fmt(power['sigma_within'],4)}; paired-difference SD "
            f"{_fmt(power['sd_paired_diff'],4)}; target effect "
            f"{_fmt(power['target_effect'],4)} (= 0.10 x endpoint median "
            f"{_fmt(power['endpoint_median'],3)})."
        )
        A(f"- Method: {power['method']}")
        A("")
        A("| n (paired) | power at alpha=0.05 |")
        A("|---|---|")
        for n, p in power.get("power_by_n", {}).items():
            A(f"| {n} | {_fmt(p,3)} |")
        A("")
        A(
            f"- Minimum detectable effect at paired n=3: "
            f"{_fmt(power.get('mde_n3'),3)}"
            + (
                f" = {_fmt(power.get('mde_n3_as_fraction_of_reference'),2)} x the "
                "reference level"
                if power.get("mde_n3_as_fraction_of_reference") is not None else ""
            )
            + (
                f"; normal-approximation replicate requirement for the 0.10 effect: "
                f"n ~ {power['n_required_normal_approx']} (optimistic lower bound)."
                if power.get("n_required_normal_approx") else "."
            )
        )
        A("")
        sec = power.get("secondary_arm_level")
        if sec and sec.get("status") in (None, "ok"):
            A(
                f"**Secondary read -- arm-level variance (best-of-{sec['k']}).** "
                f"{sec['rationale']}"
            )
            A("")
            A(
                f"- E[best-of-{sec['k']}] {_fmt(sec['expected_best_of_k'],3)}, its SD "
                f"{_fmt(sec['sd_best_of_k'],3)}; target effect "
                f"{_fmt(sec['target_effect'],3)} (= 0.10 x that level); paired n=3 "
                f"power {_fmt(sec['power_n3'],3)} -> **{sec['recommendation']}** "
                f"({sec['detail']})."
            )
            A(
                f"- Minimum detectable effect at n=3 on this variance: "
                f"{_fmt(sec.get('mde_n3'),3)}"
                + (
                    f" = {_fmt(sec.get('mde_n3_as_fraction_of_reference'),2)} x the "
                    "arm's expected endpoint."
                    if sec.get("mde_n3_as_fraction_of_reference") is not None else "."
                )
            )
            A("")
        elif sec:
            A(
                f"**Secondary read -- arm-level variance: {sec['status']}.** "
                f"{sec.get('note', '')} {sec.get('rationale', '')}"
            )
            A("")
        A(
            f"Caveat carried from the design: m={analysis.get('endpoint_m')} is "
            "shorter than Exp 2's full-T "
            "horizon, so this within-S variance is a LOWER bound and the n "
            "decision is a floor."
        )
    else:
        A(f"- {power.get('note','not evaluated')}")
    A("")

    fl = analysis.get("fork_latency", {})
    A("## Fork latency vs the settled constant (drift guard)")
    A("")
    A(
        f"- {fl.get('n')} successful fork(s), wall clock p50 "
        f"{_fmt((fl.get('wall_s') or {}).get('p50'),2)}s, p95 "
        f"{_fmt((fl.get('wall_s') or {}).get('p95'),2)}s, max "
        f"{_fmt((fl.get('wall_s') or {}).get('max'),2)}s against the settled "
        f"{_fmt(fl.get('constant_s'),0)}s solo constant -> **{fl.get('drift_guard')}**."
    )
    A(
        f"- Capacity waits total {fl.get('capacity_waits_total')}; placement lanes "
        f"{fl.get('placement_lanes')}; failed forks "
        f"{fl.get('failed_forks') or 'none'}."
    )
    A(f"- Fork op attempts per child (control-plane retries): {fl.get('attempts')}.")
    tim = state.get("timings") or {}
    attr = tim.get("attributed_s") or {}
    if attr:
        A(
            f"- Charged to the experiment's wall clock: infra_fork "
            f"{_fmt(attr.get('infra_fork'),1)}s of {_fmt(tim.get('wall_s'),1)}s total "
            f"({_fmt(100.0 * (attr.get('infra_fork') or 0.0) / (tim.get('wall_s') or 1.0),1)}%), "
            f"infra_snapshot {_fmt(attr.get('infra_snapshot'),1)}s, probe "
            f"{_fmt(attr.get('probe'),1)}s, llm_wait "
            f"{_fmt(attr.get('llm_wait'),1)}s (attributed partition; concurrent "
            "latency is charged to the dominant activity and the hidden remainder "
            "is reported separately in `exp1.json -> timings`)."
        )
    A(f"- Per-fork wall seconds: {fl.get('per_fork_wall_s')}")
    A(
        "  Note: this is fork + expose + health (the whole time-to-usable-branch), "
        "which is what a wave actually pays; the settled 32s constant is the same "
        "measurement taken solo."
    )
    A("")
    barriers = analysis.get("barriers") or []
    if barriers:
        A("## Fork barrier (what the branches were released from)")
        A("")
        A(
            "Forks of one lineage serialise, so the first child of a wave has been "
            "RUNNING for the whole fork loop by the time the last one exists. The "
            "strategy hints are POSITIONAL, so that head start would be a "
            "repeatable confound, not noise. Every wave is therefore released "
            "through a barrier: the least advanced child's world is captured with "
            "`/state-save` and restored onto all K children (the capture source "
            "included) with `/state-restore` immediately before the concurrent "
            "rollouts start."
        )
        A("")
        A("| wave | outcome | capture source | start skew (ticks) | skew after restore | tolerance | content equal | dropped |")
        A("|---|---|---|---|---|---|---|---|")
        for b in barriers:
            failed = b.get("restore_failed") or []
            A(
                f"| {b.get('wave')} | {b.get('outcome','-')} | "
                f"{b.get('source_child') or '-'} | "
                f"{b.get('pre_skew_ticks') if b.get('pre_skew_ticks') is not None else '-'} | "
                f"{b.get('post_skew_ticks') if b.get('post_skew_ticks') is not None else '-'} | "
                f"{b.get('tolerance_ticks','-')} | "
                f"{b.get('entity_counts_equal') if 'entity_counts_equal' in b else '-'} | "
                f"{', '.join(f['branch'] for f in failed) or 'none'} |"
            )
        A("")
        A(
            "`/state-restore` transplants the world without rewinding Factorio's "
            "clock, so per-child game ticks stay apart after the barrier; what a "
            "probe reads is world CONTENT, and each branch takes its own P5 "
            "baseline immediately after the barrier. A child whose restore failed "
            "is dropped as a missing draw rather than run from an unequal start; a "
            "wave that could neither be equalised nor shown to be inside the tick "
            "tolerance is aborted with no draws."
        )
        unbarriered = [b.get("wave") for b in barriers
                       if b.get("outcome") == "not_recorded"]
        if unbarriered:
            A("")
            A(
                f"Wave(s) {', '.join(str(w) for w in unbarriered)} carry NO barrier "
                "record: they were drawn before the barrier existed, so their "
                "children were released straight out of the sequential fork loop "
                "and their start skew was never measured. Under positional hints "
                "that skew is systematic; those waves are evidence of what the "
                "unbarriered protocol returns, not of the barriered one."
            )
        A("")

    A("## LIMITATIONS (read before any number above is quoted)")
    A("")
    lims = [
        f"**Single model.** Every draw is `{cfg.get('model')}`, temperature-locked "
        "by the provider, so branch diversity is provider-default sampling plus "
        "the pre-registered per-branch strategy hints. Nothing here says how a "
        "different model's outcome space opens.",
        f"**Single task.** One lab-play task (`{task.get('key')}`) at one "
        "checkpoint depth (S2). Floor/ceiling effects on other tasks are "
        "unmeasured; a flat gate on this task is not a flat gate everywhere.",
        f"**m={analysis.get('endpoint_m')} is a horizon proxy.** Exp 2 runs to a "
        f"full T; {cfg.get('steps')} steps is what the nested design can afford. "
        "The endpoint "
        "variance is therefore a LOWER bound and the power recommendation is a "
        "floor, not an estimate.",
        "**Single checkpoint, single S2 lineage.** All "
        f"{draws.get('obtained_at_endpoint')} draws descend from one baked state, "
        "so between-checkpoint variation is not in any interval here.",
        "**Probe is one 60s window.** Precise (repeatability on identical state is "
        "exact) but it is a snapshot of a possibly non-steady factory: a branch "
        "mid-rebuild at its probe step measures low, and that is indistinguishable "
        "from a real difference at n=1 per branch-step.",
        "**Bootstrap is not new data.** The best-of-K curve resamples the same "
        f"{(bok.get('n_draws') if 'n_draws' in bok else 'available')} endpoints at "
        "ARM level (one outcome per configured strategy, arm subsets without "
        "replacement); its intervals describe selection over THIS draw pool, not "
        "sampling of new waves.",
        "**Warm-slot width cap (deployment, not primitive).** Forks pin to the "
        "source's node and serialise, so K=8 was provisioned as 8 sequential forks "
        "on one node; this is a property of this deployment's warm supervisor lane, "
        "separate from fork exactness/speed as a primitive.",
    ]
    equalized = [b for b in (analysis.get("barriers") or [])
                 if b.get("outcome") == "equalized"]
    if equalized:
        lims.append(
            "**The barrier is a state round-trip, not a re-fork.** Every wave's "
            "children were put back on a common world with `/state-restore` before "
            "their rollouts, which is what removes the sequential-fork head start "
            "from the positional hints -- but a restore is not fork-exact: it "
            "replenishes ore, drops fluid boxes and resets production counters. "
            "Those losses land on all K children identically and every branch's P5 "
            "baseline is taken after them, so they cannot favour a strategy; they "
            "do mean the branch lines are Exp 1's own release procedure on top of "
            "the A*K-from-S fork path, not a byte-identical copy of it."
        )
    degraded = [b for b in (analysis.get("barriers") or [])
                if b.get("outcome") == "skew_within_tolerance"]
    if degraded:
        lims.insert(
            0,
            "**Wave(s) "
            + ", ".join(str(b.get("wave")) for b in degraded)
            + " were released WITHOUT the state barrier.** The bridge could not "
            "round-trip the children's state, so they started with the measured "
            "tick skew of the fork loop (inside the pre-registered tolerance, and "
            "recorded per wave above). Under positional hints that residual skew "
            "is systematic, not noise."
        )
    aborted = analysis.get("waves_aborted") or []
    if aborted:
        lims.insert(
            0,
            "**Wave(s) "
            + ", ".join(str(w) for w in aborted)
            + " were ABORTED at the barrier** and contribute no draws: their "
            "children could not be shown to start from a common state, and an "
            "unequal start under positional hints is a repeatable confound rather "
            "than a measurement."
        )
    underfilled = (gate.get("underfilled_waves") or []) if gate else []
    if underfilled:
        lims.insert(
            0,
            "**Incomplete wave(s) at the read point.** "
            + "; ".join(
                f"wave {u['wave']} holds {u['n']} of {u['required_n']} "
                "pre-registered draws"
                for u in underfilled
            )
            + ". The gate is read only on waves that drew their full complement, "
            "so the verdict above is `borderline-undecided` by construction: the "
            "spread and gain of a thinner pool are not the pre-registered "
            "statistic, and a missing draw is never resampled."
        )
    zero = (analysis.get("execution_quality") or {}).get("zero_action_branches") or []
    if zero:
        # Whether the gate rests on the no-ops is exactly what the sensitivity
        # re-read answers; it is never asserted from the side.
        if not sens:
            rests = (
                " Whether the gate rests on them is not established here: the "
                "robustness re-read needs two waves and at least one excluded "
                "branch."
            )
        elif matches:
            rests = (
                " The gate does not rest on them: re-reading it with those draws "
                "removed gives the same verdict at the same m (see *Robustness* "
                "above)."
            )
        elif same_verdict:
            rests = (
                f" The tier does not rest on them (it is still {sens.get('verdict')} "
                f"without them), but the read point does: excluded, the gate reads "
                f"best at m={sens.get('best_m')} rather than "
                f"m={gate.get('best_m') or '-'} (see *Robustness* above)."
            )
        else:
            rests = (
                f" The gate DOES rest on them: re-reading it without those draws "
                f"gives **{sens.get('verdict')}** at m={sens.get('best_m')} "
                "instead (see *Robustness* above)."
            )
        lims.insert(
            1,
            f"**{len(zero)} of {draws.get('obtained_at_endpoint')} draws are agent "
            "no-ops, not strategies.** "
            + ", ".join(f"`{b}`" for b in zero)
            + " answered every step with prose and no code block; the shared "
            "extractor turns that into a comment, which executes as a valid no-op, "
            "so those branches never touched the factory and their endpoints "
            "measure S2 left running. They are legitimate draws of what a fan-out "
            "wave actually returns on this model -- and part of the measured spread "
            "is therefore agent unreliability rather than strategy divergence."
            + rests,
        )
    if draws.get("missing"):
        lims.insert(
            0,
            "**Missing draws.** "
            + ", ".join(draws["missing"])
            + " produced no endpoint probe and are reported as missing, never "
            "resampled; the affected wave's metrics are computed over fewer than "
            "K draws.",
        )
    if len(waves) < 2:
        lims.insert(
            0,
            f"**Only {len(waves)} wave completed.** The gate requires the "
            "criterion to hold in BOTH waves, so the verdict above is formally "
            "undecidable and no extrapolation to a second wave is offered.",
        )
    for i, item in enumerate(lims, 1):
        A(f"{i}. {item}")
    A("")

    res = state.get("residual") or {}
    A("## Resource hygiene")
    A("")
    if res:
        A(f"- Keep-list: {', '.join(f'`{k}`' for k in res.get('keep_list', []))}.")
        sweep = res.get("sweep") or []
        A(
            f"- Reaper sweep: {len(sweep)} resource(s) deleted across this run and "
            "its audit passes (cumulative; a later sweep never erases an earlier "
            "one's record)."
        )
        if sweep:
            A("")
            A("| kind | id | name | reason | outcome |")
            A("|---|---|---|---|---|")
            for v in sweep:
                A(
                    f"| {v.get('kind')} | `{v.get('id')}` | {v.get('name') or '-'} | "
                    f"{v.get('reason')} | {v.get('outcome')} |"
                )
        A("")
        A(
            f"- Residual after the sweep: sandboxes "
            f"{[s['id'] for s in res.get('sandboxes', [])] or 'none'}; snapshots "
            f"{res.get('snapshots')}. TEMPLATE_SNAP alive: "
            f"{res.get('template_snap_alive')}; S2 alive: {res.get('s2_alive')}. "
            f"Clean: **{res.get('clean')}**."
        )
        A("")
        exps = res.get("keep_list_expiry") or {}
        if any(exps.values()):
            A(
                "- Keep-list leases (a lease is NOT cleanup; it is how long the "
                "artefact stays available to Exp 2): "
                + "; ".join(f"`{k}` until {v or 'no expiry'}" for k, v in exps.items())
                + ". Snapshot TTL can only be set when the snapshot is taken -- "
                "there is no lease-extend for images -- so Exp 2 must start before "
                "S2's lease lapses or re-bake its own checkpoint."
            )
            A("")
        A(
            "- Naming note (deployment fact, not a harness gap): `panda compute "
            "images fork` takes no `--name`, so fork children land with "
            "control-plane names (`fork-fork-*`) and cannot carry the "
            "`flebench-exp1-*` prefix. Their harness labels "
            "(`flebench-exp1-w<w>b<i>`) are in every journal record, and the "
            "reaper establishes ownership for them by ledger (the fork child id "
            "it recorded) and by source snapshot, not by name -- which is why the "
            "residual audit above is ledger-based. Sandboxes created with "
            "`sandboxes create` (the bake) DO carry the prefix."
        )
    else:
        A("- Not swept in this invocation.")
    A("")

    A("## Traceability")
    A("")
    j = state.get("journals", {})
    A(f"- Run journal (steps, probes, LLM calls with full requests, branch transcripts, incidents): `{j.get('run')}`")
    A(f"- Farplane op journal (every substrate call, attempts, placement): `{j.get('farplane')}`")
    A(f"- Machine-readable results: `{cfg.get('results_path')}`")
    A(f"- Driver: `bench/exp1.py` (seed {cfg.get('seed')}, {cfg.get('resamples')} bootstrap resamples)")
    usage = state.get("llm_usage", {})
    A(
        f"- LLM usage: {usage.get('calls')} calls, {usage.get('total_tokens')} tokens, "
        f"{usage.get('retries')} retries, {usage.get('failures')} hard failures."
    )
    A(f"- Wall clock: {_fmt(state.get('elapsed_s'),1)}s.")
    inc = state.get("incidents") or []
    A(f"- Incidents: {len(inc)}" + (
        " (" + "; ".join(f"{i.get('kind')}" for i in inc[:12]) + ")" if inc else ""
    ))
    A("")
    return "\n".join(L)


def write_report(state: dict[str, Any], path: str) -> str:
    text = render_report(state)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# Phases / CLI
# ---------------------------------------------------------------------------


async def run_experiment(cfg: Exp1Config, *, phase: str, wave: int | None) -> int:
    runner = Exp1Runner(cfg)
    try:
        if phase in ("run", "bake"):
            if (runner.state.get("s2") or {}).get("snapshot"):
                runner.journal.event(
                    "bake_skipped", reason="S2 already in state",
                    snapshot=runner.state["s2"]["snapshot"],
                )
            else:
                await runner.bake()
        if phase == "run":
            # A wave aborted at the barrier contributed no draws, so it is not
            # done: the next invocation re-runs that wave number.
            done = {
                w["wave"] for w in runner.state.get("waves", [])
                if w.get("status") != "aborted"
            }
            for w in range(1, cfg.waves + 1):
                if w in done:
                    continue
                if runner.usable_s() < cfg.wave_estimate_s:
                    runner.run.incident(
                        "wave_skipped_budget",
                        f"{runner.usable_s():.0f}s usable < {cfg.wave_estimate_s:.0f}s "
                        f"estimate for wave {w}",
                        wave=w,
                    )
                    break
                await runner.wave(w)
            analysis = runner.analyze()
            gate = analysis["gate"]
            # Third wave ONLY on a two-wave disagreement at the best m, and only
            # if the budget can pay for it (the design's rule, not a retry).
            if gate.get("third_wave_required") and runner.usable_s() >= cfg.wave_estimate_s:
                runner.journal.event(
                    "third_wave_triggered", reason=gate.get("reason"),
                    best_m=gate.get("best_m"),
                    usable_s=round(runner.usable_s(), 1),
                )
                next_wave = max(
                    (w["wave"] for w in runner.state.get("waves", [])
                     if w.get("status") != "aborted"),
                    default=0,
                ) + 1
                await runner.wave(next_wave)
                runner.analyze()
        elif phase == "wave":
            if wave is None:
                raise SystemExit("--wave N is required for --phase wave")
            await runner.wave(wave)
            runner.analyze()
        elif phase == "analyze":
            runner.analyze()
        if phase in ("run", "reap"):
            await runner.cleanup()
        runner.save()
        write_report(runner.state, cfg.report_path)
        analysis = runner.state.get("analysis") or {}
        gate = analysis.get("gate", {})
        print(
            json.dumps(
                {
                    "verdict": gate.get("verdict"),
                    "best_m": gate.get("best_m"),
                    "per_wave": gate.get("per_wave"),
                    "s2": (runner.state.get("s2") or {}).get("snapshot"),
                    "draws": analysis.get("draws"),
                    "power": (analysis.get("power") or {}).get("recommendation"),
                    "residual_clean": (runner.state.get("residual") or {}).get("clean"),
                    "results": cfg.results_path,
                    "report": cfg.report_path,
                },
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        # A failure mid-run leaves branch sandboxes holding warm slots on the
        # very node the next attempt has to fork onto, and the reaper lives in
        # ``cleanup`` (which builds its own keep-list: TEMPLATE_SNAP and S2).
        # Best-effort, never masking the failure, and only for the phases that
        # can have created substrate.
        if phase in ("run", "bake", "wave", "reap"):
            try:
                await runner.cleanup()
            except asyncio.CancelledError:
                raise
            except BaseException as sweep_exc:  # noqa: BLE001 - journaled, not raised
                runner.journal.event(
                    "cleanup_best_effort_failed",
                    after=f"{type(exc).__name__}: {exc}"[:500],
                    error=f"{type(sweep_exc).__name__}: {sweep_exc}"[:500],
                )
        raise
    finally:
        await runner.aclose()


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Experiment 1 -- decorrelation gate")
    ap.add_argument("--phase", default="run",
                    choices=("run", "bake", "wave", "analyze", "reap"))
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--waves", type=int, default=Exp1Config.waves)
    ap.add_argument("--k", type=int, default=Exp1Config.k)
    ap.add_argument("--steps", type=int, default=Exp1Config.steps)
    ap.add_argument("--bake-steps", type=int, default=Exp1Config.bake_steps)
    ap.add_argument("--model", default=Exp1Config.model)
    ap.add_argument("--task", default=Exp1Config.task_key)
    ap.add_argument("--template-snap", default=Exp1Config.template_snap)
    ap.add_argument("--budget-s", type=float, default=Exp1Config.budget_s)
    ap.add_argument("--reserve-s", type=float, default=Exp1Config.reserve_s)
    ap.add_argument("--resamples", type=int, default=Exp1Config.resamples)
    ap.add_argument("--results", default=Exp1Config.results_path)
    ap.add_argument("--report", default=Exp1Config.report_path)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    cfg = Exp1Config(
        template_snap=args.template_snap,
        model=args.model,
        task_key=args.task,
        k=args.k,
        waves=args.waves,
        steps=args.steps,
        bake_steps=args.bake_steps,
        budget_s=args.budget_s,
        reserve_s=args.reserve_s,
        resamples=args.resamples,
        results_path=args.results,
        report_path=args.report,
    )
    return asyncio.run(run_experiment(cfg, phase=args.phase, wave=args.wave))


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
