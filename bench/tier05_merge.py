"""Assemble the canonical Tier 0.5 result from the per-track measurement files.

Tier 0.5 could not run as one process: the diversity gate and step latency for
``kimi-for-coding`` and ``codex/gpt-5.6-sol`` run against the already-baked
bridge over its exposed URL (loopback substrate, no Farplane spend), while the
task-selection mini-trajectories need a FRESH sandbox per candidate task and so
run on the Farplane substrate from TEMPLATE_SNAP. ``k3``'s latency and
diversity were measured earlier in the same way (tier05_smoke.json) and are
reused verbatim rather than re-bought.

This module merges those tracks into ``tier05.json`` / ``TIER05.md``, recomputes
the per-model verdicts and the pilot sizing over the merged latency table, and
freezes T from the measured numbers under two pre-registered constraints (see
:func:`choose_T`). Every merged block carries the file it came from in
``sources`` so each number in the report stays traceable.

The merge fails closed. A requested track that is missing, a gate measured at a
different K, two contradictory rows for the same (phase, model), a journal that
cannot account for every requested sample, or a T that no ladder point can
justify all stop the merge (exit 2, no artifact) instead of quietly producing a
number. Evidence that is merely PARTIAL -- a secondary model with no rows, a
latency track that died after one step -- is merged and labelled: ``status`` is
``incomplete``, every gap is listed in ``incomplete``, the frozen config is
marked non-executable and the exit code is 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Sequence

from bench.common import atomic_write_json, load_journal_records
from bench.llm import distinct_program_rate, extract_code, normalize_program
from bench.tier05 import (
    DIVERSITY_CONDITIONAL,
    DIVERSITY_PASS,
    _verdict,
    overlap_gate,
    peak_sandboxes,
    select_tasks,
    size_pilot,
    write_markdown,
)

#: Tier-0 measured provisioning constants (bench/results/tier0.json
#: timing_summary), used to price a pilot cell.
CREATE_FROM_SNAPSHOT_P50_S = 73.45
DELETE_SANDBOX_P50_S = 1.70
#: Tier-0 measured direct-probe cost on a cold fork (probe.cycles[0].probe_s).
PROBE_COLD_S = 22.21
#: Keys of the Tier-0 soak artifact that price one branch materialisation: what
#: arm B has to hide under its sampling wait every branch round. Whatever it
#: cannot hide is a tail charged straight to T (see
#: :func:`bench.tier05.overlap_gate`). This one is NOT pinned as a constant:
#: unlike the three above it decides whether arm B is admitted at all, so it is
#: read from the artifact and is missing evidence when the artifact is missing.
MATERIALIZE_SOAK_KEYS = ("snapshot_s", "fork_total_s")

#: Sandboxes a priority-block cell provisions before T starts.
PRIORITY_BLOCK = ("A", "AxK", "B", "C")

#: T is frozen from this ladder only. The ceiling is the mission wall-clock
#: reserve (the pilot must also afford the secondary cells and the analysis
#: phase); the floor is imposed by the branch-round constraint below.
T_CANDIDATES: tuple[float, ...] = (900.0, 1200.0, 1500.0)

#: Arm B must converge at least this many times for T to be frozen without an
#: explicit relaxation. The design's HARD floor is 2 (below that arm B never
#: converges twice and the B-vs-A×K contrast is untestable) and that is what
#: per-model admission and :func:`bench.tier05.size_pilot` enforce; the T ladder
#: asks for one more so that losing a round to a slow step does not void the
#: cell. The two thresholds are deliberately different, not inconsistent.
MIN_BRANCH_ROUNDS = 3


class MergeError(RuntimeError):
    """An input this merge cannot reconcile.

    Every number in the canonical artifact has to be traceable to a file that
    measured the gate being frozen, so a missing track, a track measured at a
    different K, two contradictory rows for the same (phase, model) or an
    unjustifiable pilot config all stop the merge instead of being papered over
    with the last value read.
    """


class GateReconstructionError(MergeError):
    """A journal cannot be replayed into a COMPLETE gate result."""


class InfeasiblePilot(MergeError):
    """No point on the T ladder satisfies the pre-registered constraints."""


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def materialize_from_tier0(path: str) -> tuple[float | None, str]:
    """Snapshot+fork p50 from a Tier-0 soak artifact, or ``None`` with a reason.

    Returns evidence, never a stand-in: arm B's admission hangs off this number
    (:func:`bench.tier05.overlap_gate` answers ``unknown`` -> B not admitted when
    it is missing), so an absent or incomplete artifact yields ``None`` rather
    than a pinned constant no file backs.
    """
    if not path:
        return None, "no Tier-0 artifact requested (--tier0 '')"
    if not os.path.exists(path):
        return None, f"{path} does not exist"
    soak_latency = ((_load(path).get("soak") or {}).get("latency") or {})
    parts: list[float] = []
    for key in MATERIALIZE_SOAK_KEYS:
        p50 = (soak_latency.get(key) or {}).get("p50")
        if not isinstance(p50, (int, float)) or isinstance(p50, bool) or p50 <= 0:
            return None, f"{path}: soak.latency.{key}.p50 not measured"
        parts.append(float(p50))
    keys = " + ".join(f"soak.latency.{k}.p50" for k in MATERIALIZE_SOAK_KEYS)
    return round(sum(parts), 3), f"{path}: {keys}"


def _journal(path: str) -> list[dict[str, Any]]:
    """Records of the LATEST session in a run journal.

    A journal is an append stream: re-running a gate appends a second session to
    the same file. Replaying both would double-count seats and silently mix a
    failed attempt with its retry, so only the last session is replayed (see
    :func:`bench.common.load_journal_records`) and a session boundary inside
    that slice is an error rather than a guess.
    """
    records = load_journal_records(path, session="latest")
    sessions = {r.get("session") for r in records}
    if len(sessions) > 1:
        raise GateReconstructionError(
            f"{path}: the latest-session slice spans {len(sessions)} sessions "
            f"({sorted(str(s) for s in sessions)}); refusing to merge records "
            "from different runs of the same gate"
        )
    return records


def _branch_root(label: Any) -> str:
    """``"hinted#2"`` -> ``"hinted"``; the seat's gate branch."""
    return str(label or "").split("#", 1)[0]


def _seats(
    calls: Sequence[dict[str, Any]], *, branch: str, gate_K: int, source: str
) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Resolve one gate branch into exactly ``gate_K`` seats.

    A seat is one REQUESTED sample. ``None`` means the seat was requested and
    every attempt for it failed: it produced no program, so it stays in the
    denominator as an unusable seat. Anything that leaves the seat count
    ambiguous -- a missing seat label, more samples than the gate asked for, a
    branch that stops mid-gate -- raises, because scoring only the seats that
    happened to answer is exactly the survivor bias the gate exists to catch.

    Returns ``(seats, errors)`` with one error line per dead seat.
    """
    group = [c for c in calls if _branch_root(c.get("branch")) == branch]
    if not group:
        raise GateReconstructionError(
            f"{source}: no {branch!r} llm_call records; the journal does not "
            f"cover the {branch} branch of the diversity gate"
        )
    by_label: dict[str, list[dict[str, Any]]] = {}
    for call in group:
        by_label.setdefault(str(call.get("branch") or ""), []).append(call)
    fanned = sorted(lbl for lbl in by_label if "#" in lbl)
    collapsed = sorted(lbl for lbl in by_label if "#" not in lbl)
    if fanned and collapsed:
        raise GateReconstructionError(
            f"{source}: the {branch} branch mixes per-seat labels {fanned} with "
            f"collapsed label(s) {collapsed}; seat identity is ambiguous"
        )
    seats: list[dict[str, Any] | None] = []
    errors: list[str] = []
    if fanned:
        # One journalled call per seat (hinted branch, or a provider without
        # server-side n): the seat labels themselves must cover the gate.
        expected = [f"{branch}#{i}" for i in range(gate_K)]
        if fanned != sorted(expected):
            raise GateReconstructionError(
                f"{source}: the {branch} branch journalled seats {fanned}, the "
                f"gate asks for {expected}; the journal is incomplete for "
                f"K={gate_K}"
            )
        for label in expected:
            recs = by_label[label]
            ok = [r for r in recs if r.get("outcome") == "ok"]
            if len(ok) > 1:
                raise GateReconstructionError(
                    f"{source}: seat {label} has {len(ok)} successful samples; "
                    "one seat is one sample"
                )
            if ok:
                seats.append(ok[0])
                continue
            seats.append(None)
            last = max(recs, key=lambda r: int(r.get("attempt") or 0))
            errors.append(
                f"{label}: no sample after {len(recs)} attempt(s) "
                f"({str(last.get('error') or 'failed')[:200]})"
            )
        return seats, errors
    # One call answered with n samples server-side; each returned sample is
    # journalled separately under the same branch label.
    label = collapsed[0]
    recs = by_label[label]
    ok = [r for r in recs if r.get("outcome") == "ok"]
    failed = [r for r in recs if r.get("outcome") != "ok"]
    if len(ok) > gate_K:
        raise GateReconstructionError(
            f"{source}: the {branch} branch journalled {len(ok)} samples for a "
            f"K={gate_K} gate; this journal measured a different gate"
        )
    if len(ok) < gate_K and not failed:
        raise GateReconstructionError(
            f"{source}: the {branch} branch journalled {len(ok)} of {gate_K} "
            "samples and no failure record; the journal stops mid-gate, so the "
            "missing seats cannot be told apart from a truncated file"
        )
    seats = list(ok) + [None] * (gate_K - len(ok))
    if len(ok) < gate_K:
        last = max(failed, key=lambda r: int(r.get("attempt") or 0))
        errors.append(
            f"{label}: {gate_K - len(ok)} of {gate_K} samples never arrived "
            f"({str(last.get('error') or 'failed')[:200]})"
        )
    return seats, errors


def _require_response_text(
    seats: Sequence[dict[str, Any] | None], *, source: str
) -> None:
    """A seat that answered must carry its response text in the journal.

    ``response_text`` is only journalled when the client logs full requests.
    Without it every program normalises to empty and the reconstruction would
    report a 0.0 distinct-program rate for a model that actually answered -- a
    fabricated verdict, so refuse instead of scoring the blanks.
    """
    blind = [
        s for s in seats
        if s is not None
        and int(s.get("code_chars") or 0) > 0
        and not (s.get("response_text") or "").strip()
    ]
    if blind:
        raise GateReconstructionError(
            f"{source}: {len(blind)} sample(s) journalled a program "
            "(code_chars > 0) but no response_text -- the run was journalled "
            "without full request logging, so its programs cannot be replayed "
            "and its diversity rate cannot be rebuilt"
        )


def gate_from_journal(
    model: str, journal_dir: str, *, gate_K: int, lat_steps: int
) -> dict[str, Any]:
    """Rebuild a model's gate result from its Tier-0.5 journals.

    Needed when a gate track is stopped before it can serialise its payload:
    the journal already holds every request, response and outcome, so the
    diversity rate and the step latency are recoverable exactly rather than
    re-bought. Returns the same shape ``run_tier05`` would have produced, plus
    ``reconstructed_from``, ``reconstruction`` (the seat accounting) and
    ``incomplete`` (explicit, never a silent gap).

    ``gate_K`` is the canonical gate width this merge freezes, NOT whatever the
    journal happens to contain: the rate's denominator is the number of samples
    requested, a seat whose every attempt failed is a seat that produced no
    program, and a journal that cannot account for all ``gate_K`` seats is
    rejected instead of being scored on its survivors. ``lat_steps`` is the step
    count the latency track was asked for, so a run that died after one step is
    reported as aborted instead of passing as a full measurement.
    """
    if gate_K < 1:
        raise ValueError(f"gate_K must be >= 1, got {gate_K}")
    if lat_steps < 1:
        raise ValueError(f"lat_steps must be >= 1, got {lat_steps}")
    slug = model.replace("/", "-")
    div_path = os.path.join(journal_dir, f"t05-div-{slug}.jsonl")
    lat_path = os.path.join(journal_dir, f"t05-lat-{slug}.jsonl")
    if not os.path.exists(div_path):
        raise GateReconstructionError(
            f"{model}: no diversity journal at {div_path}, so there is no "
            "evidence to rebuild the gate from"
        )
    div = _journal(div_path)
    calls = [r for r in div if r.get("kind") == "llm_call"]
    if not calls:
        raise GateReconstructionError(
            f"{div_path}: no llm_call records in the latest session"
        )
    plain_seats, plain_errors = _seats(
        calls, branch="plain", gate_K=gate_K, source=div_path
    )
    hinted_seats, hinted_errors = _seats(
        calls, branch="hinted", gate_K=gate_K, source=div_path
    )
    _require_response_text(plain_seats + hinted_seats, source=div_path)
    ok = [c for c in calls if c.get("outcome") == "ok"]
    failures = [c for c in calls if c.get("outcome") != "ok"]

    def summarize(
        seats: Sequence[dict[str, Any] | None], label: str, seat_errors: list[str]
    ) -> dict[str, Any]:
        present = [s for s in seats if s is not None]
        codes = [
            extract_code(s.get("response_text") or "") if s is not None else None
            for s in seats
        ]
        norm = [normalize_program(c or "") for c in codes]
        lats = [float(s.get("latency_s") or 0.0) for s in present]
        return {
            "label": label,
            # k is the gate width, so the rate below is scored against what the
            # gate asked for and not against the seats that came back.
            "k": len(seats),
            "parsed": sum(1 for c in codes if c),
            "failed_seats": sum(1 for s in seats if s is None),
            "empty_responses": sum(
                1 for s in present if not (s.get("response_text") or "").strip()
            ),
            "truncated": sum(
                1 for s in present if "length" in str(s.get("finish_reason") or "")
            ),
            "finish_reasons": [
                (s.get("finish_reason") if s is not None else None) for s in seats
            ],
            "distinct_program_rate": round(distinct_program_rate(codes), 3),
            "distinct_programs": len({n for n in norm if n}),
            "median_latency_s": (
                round(statistics.median(lats), 3) if lats else None
            ),
            "mean_completion_tokens": (
                round(statistics.fmean(
                    [float(s.get("completion_tokens") or 0) for s in present]), 1)
                if present else None
            ),
            "errors": list(seat_errors),
            "code_heads": [(c or "")[:160] for c in codes],
        }

    plain_sum = summarize(plain_seats, "plain", plain_errors)
    hinted_sum = summarize(hinted_seats, "hinted", hinted_errors)
    verdict, rationale = _verdict(
        plain_sum["distinct_program_rate"], hinted_sum["distinct_program_rate"]
    )
    unusable = plain_sum["k"] - plain_sum["parsed"]
    if unusable:
        rationale += (
            f" NOTE: {unusable}/{plain_sum['k']} plain seats yielded no "
            f"extractable program (dead seats={plain_sum['failed_seats']}, "
            f"empty={plain_sum['empty_responses']}, "
            f"truncated={plain_sum['truncated']}); the rate is scored against "
            "the K the gate requested, so the effective width is smaller than "
            "requested."
        )

    incomplete: list[str] = []
    lat: list[dict[str, Any]] = []
    lat_reason = ""
    if os.path.exists(lat_path):
        lat = _journal(lat_path)
    else:
        lat_reason = f"no step-latency journal at {lat_path}"
        incomplete.append(f"{model}: {lat_reason}")
    lat_calls = [r for r in lat if r.get("kind") == "llm_call"]
    lat_ok = [c for c in lat_calls if c.get("outcome") == "ok"]
    lat_fail = [c for c in lat_calls if c.get("outcome") != "ok"]
    steps = [r for r in lat if r.get("kind") == "step"]
    # A step costs every attempt it took, not just the one that answered.
    step_walls: list[float] = []
    spent = 0.0
    for call in lat_calls:
        spent += float(call.get("latency_s") or 0.0)
        if call.get("outcome") == "ok":
            step_walls.append(round(spent, 3))
            spent = 0.0
    exec_walls = [float(s.get("exec_s") or 0.0) for s in steps]
    for i, wall in enumerate(step_walls):
        if i < len(exec_walls):
            step_walls[i] = round(wall + exec_walls[i], 3)
    if not lat_reason and len(steps) < lat_steps:
        lat_reason = (
            f"{len(steps)} of {lat_steps} requested steps completed before the "
            "track stopped"
        )
        incomplete.append(f"{model}: {lat_reason}")
    total_calls = len(calls) + len(lat_calls)
    total_fail = len(failures) + len(lat_fail)
    return {
        "model": model,
        "reconstructed_from": [div_path, lat_path],
        "journals": {"diversity": div_path, "latency": lat_path},
        "incomplete": incomplete,
        "reconstruction": {
            "gate_K": gate_K,
            "requested_latency_steps": lat_steps,
            "diversity_records": len(div),
            "latency_records": len(lat),
            "session": next((r.get("session") for r in div if r.get("session")), None),
            "plain_seats_usable": plain_sum["parsed"],
            "plain_seats_dead": plain_sum["failed_seats"],
            "hinted_seats_usable": hinted_sum["parsed"],
            "hinted_seats_dead": hinted_sum["failed_seats"],
            "seat_errors": plain_sum["errors"] + hinted_sum["errors"],
        },
        "diversity": {
            "model": model,
            "temperature": 1.0,
            "temperature_locked": True,
            "gate_K": gate_K,
            "plain": plain_sum,
            "hinted": hinted_sum,
            "verdict": verdict,
            "rationale": rationale,
            "unusable_samples": unusable,
            "k_way_sampling_latency_s": plain_sum["median_latency_s"],
            "usage": {"calls": len(ok), "retries": len(failures)},
            "provider_retries": len(failures),
            "provider_retry_rate": round(
                len(failures) / max(1, len(calls)), 3
            ),
        },
        "latency": {
            "model": model,
            "measured": bool(step_walls),
            "steps": [
                {"step": i + 1, "wall_s": w} for i, w in enumerate(step_walls)
            ],
            "median_step_s": (
                round(statistics.median(step_walls), 3) if step_walls else None
            ),
            "mean_step_s": (
                round(statistics.fmean(step_walls), 3) if step_walls else None
            ),
            "max_step_s": round(max(step_walls), 3) if step_walls else None,
            "median_llm_s": (
                round(statistics.median(
                    [float(c.get("latency_s") or 0.0) for c in lat_ok]), 3)
                if lat_ok else None
            ),
            "median_exec_s": (
                round(statistics.median(exec_walls), 3) if exec_walls else None
            ),
            "tokens_per_step": (
                round(statistics.fmean(
                    [float(c.get("completion_tokens") or 0) for c in lat_ok]), 1)
                if lat_ok else None
            ),
            "completed_steps": len(steps),
            "requested_steps": lat_steps,
            "aborted": len(steps) < lat_steps,
            "incomplete_reason": lat_reason,
            "incidents": [
                f"{c.get('error', '')[:120]}" for c in lat_fail
            ],
        },
        "reliability": {
            "total_calls": total_calls,
            "failed_attempts": total_fail,
            "failure_rate": round(total_fail / max(1, total_calls), 3),
            "timeouts": sum(
                1 for c in failures + lat_fail if "Timeout" in str(c.get("error", ""))
            ),
            "empty_completions": sum(
                1 for c in failures + lat_fail
                if "EmptyCompletion" in str(c.get("error", ""))
            ),
            "max_attempt_latency_s": round(
                max((float(c.get("latency_s") or 0.0) for c in calls + lat_calls),
                    default=0.0), 1
            ),
        },
    }


def block_wall_s(T: float, K: int, arms: Sequence[str] = PRIORITY_BLOCK) -> float:
    """Wall clock of the priority model's block at run cap 1 (sequential).

    ``arms`` is the block the priority model actually runs, so a pilot that
    drops a cell is not priced as if it still bought it.
    """
    total = 0.0
    for arm in arms:
        n = peak_sandboxes(arm, K)
        total += T + n * CREATE_FROM_SNAPSHOT_P50_S + n * DELETE_SANDBOX_P50_S
    return total


def choose_T(
    *,
    slowest_step_s: float,
    m: int,
    K: int,
    block_budget_s: float,
    arms: Sequence[str] = PRIORITY_BLOCK,
    materialize_tail_s: float = 0.0,
    allow_relaxation: bool = False,
) -> dict[str, Any]:
    """Largest T on the ladder that satisfies both pre-registered constraints.

    1. the priority block (``arms`` at run cap 1) fits ``block_budget_s``;
    2. arm B gets >= :data:`MIN_BRANCH_ROUNDS` branch rounds at the priority
       model's measured step latency, counting the direct probe AND the
       snapshot+fork tail the model could not hide under its sampling wait
       (``materialize_tail_s``, same charge :func:`bench.tier05.size_pilot`
       applies, so the frozen T and the reported sizing agree on what a branch
       round costs).

    If no ladder point satisfies both, T is NOT frozen: :class:`InfeasiblePilot`
    is raised and the caller has to decide, because picking a point anyway hides
    a pre-registered constraint violation inside a number that reads like a
    measurement. ``allow_relaxation=True`` is that decision, made explicitly: it
    gives up branch rounds (the pre-registered relaxation order) while KEEPING
    the block budget, and the surviving violation is reported in ``relaxed`` /
    ``violations``. The block budget itself is never relaxed -- an over-budget
    block cannot be run at all, so there is no T to freeze.
    """
    if slowest_step_s <= 0:
        raise InfeasiblePilot(
            "T cannot be frozen without a measured median step for the priority "
            f"model (got slowest_step_s={slowest_step_s!r})"
        )
    if materialize_tail_s < 0:
        raise ValueError(f"materialize_tail_s must be >= 0, got {materialize_tail_s}")
    round_s = m * slowest_step_s + PROBE_COLD_S + materialize_tail_s
    rows = []
    for T in T_CANDIDATES:
        rounds = math.floor(T / round_s)
        block = block_wall_s(T, K, arms)
        rows.append(
            {
                "T_s": T,
                "branch_rounds": rounds,
                "block_wall_s": round(block, 1),
                "block_wall_h": round(block / 3600.0, 2),
                "fits_budget": block <= block_budget_s,
                "enough_rounds": rounds >= MIN_BRANCH_ROUNDS,
                "ok": block <= block_budget_s and rounds >= MIN_BRANCH_ROUNDS,
            }
        )
    ok = [r for r in rows if r["ok"]]
    violations: list[str] = []
    if ok:
        chosen, relaxed = max(ok, key=lambda r: r["T_s"]), ""
    else:
        affordable = [r for r in rows if r["fits_budget"]]
        ladder_txt = "; ".join(
            f"T={r['T_s']:.0f}s -> {r['branch_rounds']} round(s), block "
            f"{r['block_wall_h']}h ({'fits' if r['fits_budget'] else 'over'} "
            "budget)" for r in rows
        )
        if not affordable:
            raise InfeasiblePilot(
                "no T on the ladder keeps the priority block inside "
                f"{block_budget_s:.0f}s ({ladder_txt}); the block budget cannot "
                "be relaxed because an over-budget block cannot be run at all -- "
                "raise --block-budget-s or shrink K/the arm set"
            )
        if not allow_relaxation:
            raise InfeasiblePilot(
                f"no T on the ladder gives arm B >= {MIN_BRANCH_ROUNDS} branch "
                f"round(s) at a {slowest_step_s:.0f}s median step and a "
                f"{materialize_tail_s:.0f}s unhidden snapshot+fork tail "
                f"({ladder_txt}); "
                "re-run with --allow-t-relaxation to freeze the largest "
                "affordable T anyway and have the violation recorded in the "
                "artifact"
            )
        # Pre-registered relaxation order: give up branch rounds before the
        # block budget, and say so in the artifact.
        chosen = max(affordable, key=lambda r: r["T_s"])
        violations.append("min_branch_rounds")
        relaxed = (
            "no ladder point met both constraints; --allow-t-relaxation kept the "
            f"block budget and accepted {chosen['branch_rounds']} branch "
            f"round(s) instead of the pre-registered {MIN_BRANCH_ROUNDS}"
        )
    return {
        "chosen_T_s": chosen["T_s"],
        "ladder": rows,
        "round_s": round(round_s, 1),
        "materialize_tail_s": round(materialize_tail_s, 1),
        "slowest_step_s": slowest_step_s,
        "m": m,
        "K": K,
        "block_budget_s": block_budget_s,
        "min_branch_rounds": MIN_BRANCH_ROUNDS,
        "relaxation_allowed": bool(allow_relaxation),
        "relaxed": relaxed,
        "violations": violations,
        "rule": (
            "largest T with (a) the priority model's block A/A×K/B/C at run cap "
            f"1 inside the block budget and (b) >= {MIN_BRANCH_ROUNDS} branch "
            "rounds at m x measured-median-step + one cold direct probe + the "
            "unhidden snapshot+fork tail"
        ),
    }


def verdict_row(div: dict[str, Any], lat: dict[str, Any]) -> dict[str, Any]:
    return {
        "diversity_verdict": div.get("verdict", "not_measured"),
        "diversity_rate_plain": div.get("plain", {}).get("distinct_program_rate"),
        "diversity_rate_hinted": div.get("hinted", {}).get("distinct_program_rate"),
        "parsed_plain": div.get("plain", {}).get("parsed"),
        "unusable_samples": div.get("unusable_samples"),
        "median_step_s": lat.get("median_step_s"),
        "median_llm_s": lat.get("median_llm_s"),
        "tokens_per_step": lat.get("tokens_per_step"),
        "provider_retry_rate": div.get("provider_retry_rate"),
        "hints_required": div.get("verdict") in ("pass_with_hints", "conditional"),
        "enters_tier1": div.get("verdict") in ("pass", "pass_with_hints", "conditional"),
        "notes": div.get("rationale", ""),
    }


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _claim(
    table: dict[str, Any],
    sources: dict[str, str],
    substrates: dict[str, str],
    *,
    phase: str,
    model: str,
    row: dict[str, Any],
    path: str,
    substrate: str,
) -> None:
    """Record one imported measurement, refusing to overwrite a different one.

    Two tracks may legitimately carry the same row (a re-serialised payload), but
    two DIFFERENT measurements of the same (phase, model) make the artifact
    depend on the order of ``--track`` arguments. That is the operator's call to
    make, not a silent last-write-wins.
    """
    if model in table:
        if _canonical(table[model]) != _canonical(row):
            raise MergeError(
                f"{phase} row for {model!r} appears in {sources[model]} and in "
                f"{path} with different measurements; drop one --track or "
                "reconcile them before merging"
            )
        sources[model] = f"{sources[model]}; {path} (identical)"
        return
    table[model] = row
    sources[model] = path
    substrates[model] = substrate


def _check_gate_row(
    path: str, cfg: dict[str, Any], model: str, row: dict[str, Any], gate_K: int
) -> None:
    """A diversity row only counts if it measured the gate being frozen.

    The distinct-program rate's denominator is the gate width, so importing a
    row measured at a different K silently changes what "pass" means.
    """
    src_K = cfg.get("K")
    if src_K != gate_K:
        raise MergeError(
            f"{path}: diversity was measured at K={src_K!r} but this merge "
            f"freezes a K={gate_K} gate; re-measure the model or merge with "
            f"--gate-K {src_K!r}"
        )
    if row.get("model") not in (None, model):
        raise MergeError(
            f"{path}: diversity row keyed {model!r} carries model "
            f"{row.get('model')!r}"
        )
    for branch in ("plain", "hinted"):
        block = row.get(branch)
        if not isinstance(block, dict):
            raise MergeError(
                f"{path}: diversity row for {model!r} has no {branch!r} block"
            )
        if block.get("k") != gate_K:
            raise MergeError(
                f"{path}: the {branch} block for {model!r} has "
                f"k={block.get('k')!r} samples, the gate this merge freezes is "
                f"K={gate_K}; its rate is not a rate over the gate width"
            )


def merge(
    *,
    tracks: Sequence[str],
    reconstruct: Sequence[str],
    journal_dir: str,
    tasks_path: str,
    models: Sequence[str],
    m: int,
    K: int,
    gate_K: int,
    block_budget_s: float,
    arms: Sequence[str],
    c_model: str,
    replicates: int,
    lat_steps: int = 3,
    reconstruct_substrate: str = "loopback (bake bridge)",
    branch_materialize_s: float | None = None,
    tier0_path: str = "bench/results/tier0.json",
    allow_relaxation: bool = False,
) -> dict[str, Any]:
    if c_model not in models:
        raise MergeError(
            f"--c-model {c_model!r} is not in --models {list(models)}; the model "
            "carrying the four-arm block has to be gated like any other"
        )
    latency: dict[str, Any] = {}
    diversity: dict[str, Any] = {}
    sources: dict[str, Any] = {"latency": {}, "diversity": {}, "tasks": tasks_path}
    # Substrate is a property of a (phase, model) measurement, not of a model:
    # the gates ran on the bake bridge and the task sanity on Farplane, and a
    # diversity-only track used to lose its substrate entirely.
    substrate_by_phase: dict[str, dict[str, str]] = {"latency": {}, "diversity": {}}
    incomplete: list[str] = []
    for path in tracks:
        if not os.path.exists(path):
            raise MergeError(
                f"track {path} was requested with --track but does not exist; "
                "the canonical artifact would silently drop its rows"
            )
        data = _load(path)
        cfg = data.get("config")
        if not isinstance(cfg, dict):
            raise MergeError(
                f"{path}: no config block, so the gate condition it measured "
                "cannot be checked against this merge"
            )
        substrate = str(data.get("substrate") or "")
        if not substrate:
            raise MergeError(f"{path}: does not record the substrate it ran on")
        for phase in ("diversity", "latency"):
            block = data.get(phase)
            if block is not None and not isinstance(block, dict):
                raise MergeError(f"{path}: {phase!r} is not a table of rows")
            for model, row in (block or {}).items():
                if not isinstance(row, dict):
                    raise MergeError(
                        f"{path}: the {phase} row for {model!r} is not an object"
                    )
        for model, row in (data.get("diversity") or {}).items():
            if not row.get("verdict"):
                incomplete.append(
                    f"{path}: diversity row for {model!r} has no verdict "
                    f"({row.get('error') or 'gate did not finish'})"
                )
                continue
            _check_gate_row(path, cfg, model, row, gate_K)
            _claim(diversity, sources["diversity"], substrate_by_phase["diversity"],
                   phase="diversity", model=model, row=row, path=path,
                   substrate=substrate)
        for model, row in (data.get("latency") or {}).items():
            if row.get("median_step_s") is None:
                incomplete.append(
                    f"{path}: latency row for {model!r} has no median step "
                    f"({row.get('error') or 'no step completed'})"
                )
                continue
            _claim(latency, sources["latency"], substrate_by_phase["latency"],
                   phase="latency", model=model, row=row, path=path,
                   substrate=substrate)

    reliability: dict[str, Any] = {}
    reconstructions: dict[str, Any] = {}
    for model in reconstruct:
        rebuilt = gate_from_journal(
            model, journal_dir, gate_K=gate_K, lat_steps=lat_steps
        )
        _claim(diversity, sources["diversity"], substrate_by_phase["diversity"],
               phase="diversity", model=model, row=rebuilt["diversity"],
               path=rebuilt["journals"]["diversity"],
               substrate=f"{reconstruct_substrate} (attested, reconstructed)")
        if rebuilt["latency"]["median_step_s"] is None:
            incomplete.append(
                f"{model}: no step latency recoverable from "
                f"{rebuilt['journals']['latency']}"
            )
        else:
            _claim(latency, sources["latency"], substrate_by_phase["latency"],
                   phase="latency", model=model, row=rebuilt["latency"],
                   path=rebuilt["journals"]["latency"],
                   substrate=f"{reconstruct_substrate} (attested, reconstructed)")
        reliability[model] = rebuilt["reliability"]
        reconstructions[model] = rebuilt["reconstruction"]
        incomplete.extend(rebuilt["incomplete"])

    tasks_payload: dict[str, Any] = {}
    task_rows: list[dict[str, Any]] = []
    task_errors: list[dict[str, Any]] = []
    if tasks_path:
        if not os.path.exists(tasks_path):
            raise MergeError(
                f"tasks track {tasks_path} was requested but does not exist; "
                "pass --tasks '' to merge without the task-sanity phase"
            )
        tasks_payload = _load(tasks_path)
        rows = tasks_payload.get("tasks") or []
        task_rows = [r for r in rows if not r.get("error")]
        task_errors = [r for r in rows if r.get("error")]
        if not task_rows:
            incomplete.append(
                f"{tasks_path}: no usable task-sanity rows, so no task can be "
                "selected for the pilot"
            )

    for model in models:
        if model not in diversity:
            incomplete.append(f"{model}: no diversity gate row was merged")
        if model not in latency:
            incomplete.append(f"{model}: no step-latency row was merged")

    verdicts = {
        model: verdict_row(diversity.get(model, {}), latency.get(model, {}))
        for model in models
    }

    # Arm B has to hide a snapshot+fork materialisation under its sampling wait
    # every branch round; whatever it cannot hide is wall clock inside T. The
    # gate (and the tail it charges) is bench.tier05.overlap_gate, so the frozen
    # T, the per-model feasibility check and size_pilot all price a branch round
    # the same way.
    b_in_arms = "B" in arms
    caps_block = tasks_payload.get("caps") or {}
    materialize_s: float | None = branch_materialize_s
    materialize_source = "--branch-materialize-s (operator)"
    if materialize_s is None:
        materialize_s = caps_block.get("branch_materialize_s")
        materialize_source = f"{tasks_path or 'caps'}: caps.branch_materialize_s"
    if materialize_s is None:
        materialize_s, materialize_source = materialize_from_tier0(tier0_path)
    if materialize_s is None and b_in_arms:
        # Fail closed: unmeasured materialisation is not hidden materialisation.
        # overlap_gate answers 'unknown' below, which keeps arm B out.
        incomplete.append(
            "arm B is in the arm set but the snapshot+fork materialisation it "
            f"has to hide is not measured ({materialize_source}); pass "
            "--branch-materialize-s or a Tier-0 soak artifact with "
            + " and ".join(f"soak.latency.{k}.p50" for k in MATERIALIZE_SOAK_KEYS)
        )
    for model, v in verdicts.items():
        overlap = overlap_gate((latency.get(model) or {}).get("median_llm_s"),
                               materialize_s)
        v["overlap_verdict"] = overlap["verdict"]
        v["overlap_tail_s"] = overlap["tail_s"]
        v["overlap_detail"] = overlap["detail"]
        v["b_arm_admitted"] = bool(overlap["b_arm_admitted"]) if b_in_arms else None

    def tail_for(model: str) -> float:
        """Per-branch-round tail this model pays inside T (0 without arm B)."""
        if not b_in_arms:
            return 0.0
        return float((verdicts.get(model) or {}).get("overlap_tail_s") or 0.0)

    # T is sized on the PRIORITY model (the one carrying the four-arm block);
    # a secondary model that turns out slower does not get to shrink the block
    # the whole contrast depends on -- it gets its own feasibility check below.
    priority_step = (latency.get(c_model) or {}).get("median_step_s")
    if priority_step is None:
        raise MergeError(
            f"the priority model {c_model!r} has no merged median step latency, "
            "so T cannot be frozen from measured numbers and there is no pilot "
            "config to freeze"
        )
    if not verdicts[c_model]["enters_tier1"]:
        raise MergeError(
            f"the priority model {c_model!r} did not pass the diversity gate "
            f"(verdict {verdicts[c_model]['diversity_verdict']!r}); the four-arm "
            "block it carries cannot be run, so nothing can be frozen"
        )
    t_choice = choose_T(
        slowest_step_s=priority_step, m=m, K=K, block_budget_s=block_budget_s,
        arms=tuple(arms),
        materialize_tail_s=tail_for(c_model),
        allow_relaxation=allow_relaxation,
    )
    T = t_choice["chosen_T_s"]

    # Design rule, applied per model: a cell is only worth running if arm B can
    # converge at least twice inside T at that model's measured step latency.
    # A model that fails this is skipped with the arithmetic as the reason.
    for model, v in verdicts.items():
        step = (latency.get(model) or {}).get("median_step_s")
        v["reliability"] = reliability.get(model)
        if step:
            rounds = math.floor(T / (m * step + PROBE_COLD_S + tail_for(model)))
            v["branch_rounds_at_T"] = rounds
            v["enters_pilot"] = bool(v["enters_tier1"]) and rounds >= 2
            v["pilot_skip_reason"] = "" if v["enters_pilot"] else (
                f"measured median step {step:.0f}s gives {rounds} branch round(s) "
                f"at T={T:.0f}s (m={m}); the design requires >= 2 for arm B to "
                "converge twice, so no testable B cell exists for this model "
                "inside any T this pilot can afford"
                if v["enters_tier1"] else
                f"diversity gate verdict {v['diversity_verdict']}"
            )
        else:
            v["branch_rounds_at_T"] = None
            v["enters_pilot"] = False
            v["pilot_skip_reason"] = (
                "step latency not measured"
                if v["enters_tier1"] else
                f"diversity gate verdict {v['diversity_verdict']}"
            )
    admitted = [m_ for m_, v in verdicts.items() if v["enters_pilot"]]
    if c_model not in admitted:
        raise MergeError(
            f"the priority model {c_model!r} is not admitted to the pilot "
            f"({verdicts[c_model]['pilot_skip_reason']}); the priority block is "
            "its block, so there is no pilot config to freeze"
        )

    # Arm B on a model whose materialisation does not hide would measure
    # Farplane latency instead of the arm (bench.tier05.overlap_gate), so its B
    # cell is not shipped and the artifact says why.
    b_blocked = [
        m_ for m_ in admitted if b_in_arms and not verdicts[m_]["b_arm_admitted"]
    ]
    for m_ in b_blocked:
        incomplete.append(
            f"{m_}: arm B not admitted -- {verdicts[m_]['overlap_detail']}"
        )
    b_tails = [
        verdicts[m_]["overlap_tail_s"] for m_ in admitted
        if b_in_arms and verdicts[m_]["overlap_tail_s"] is not None
    ]
    # Charge the worst admitted tail, the same convention size_pilot documents.
    tail_for_sizing = max(b_tails) if b_tails else 0.0

    selection = select_tasks(task_rows, want=2) if task_rows else {
        "selected": [], "want": 2, "candidates": [], "shortfall": 2,
        "criterion": "task phase produced no usable rows",
    }
    # The pilot affords ONE task at run cap 1; the ranked runner-up is recorded
    # as the pre-registered substitute if the primary turns out degenerate.
    primary = selection["selected"][:1]
    if not primary:
        incomplete.append(
            "the pilot has no task to run: "
            + ("the task-sanity phase was not merged (--tasks '')" if not tasks_path
               else "no candidate passed the task-sanity probe")
        )

    sizing = size_pilot(
        models=admitted,
        arms=tuple(arms),
        c_model=c_model,
        latency=latency,
        caps=tasks_payload.get("caps") or {"run_cap": 1, "max_sandboxes": 24},
        probe_s=PROBE_COLD_S,
        provision_s=CREATE_FROM_SNAPSHOT_P50_S,
        teardown_s=DELETE_SANDBOX_P50_S,
        m=m,
        K=K,
        safety_factor=1.25,
        materialize_tail_s=tail_for_sizing,
    )
    # T is frozen by choose_T (mission wall clock + branch-round floor) and the
    # task count and replicates are frozen by the selection above, so the
    # reported sizing is the ladder point for THAT config -- not the most
    # ambitious point that happens to share the frozen T.
    want_tasks = len(primary) or 1
    matching = [
        e for e in sizing["ladder"]
        if e["T_s"] == T and e["n_tasks"] == want_tasks
        and e["replicates"] == replicates
    ]
    if not matching:
        raise MergeError(
            f"the frozen config (T={T:.0f}s, {want_tasks} task(s), "
            f"{replicates} replicate(s)) is not a point on the sizing ladder; "
            "the reported sizing would not be the sizing for the frozen config"
        )
    sizing["chosen"] = matching[0]
    sizing["T_choice"] = t_choice
    if sizing.get("error"):
        incomplete.append(f"pilot sizing: {sizing['error']}")
    sizing_fits = bool(sizing["chosen"].get("fits"))
    if not sizing_fits:
        incomplete.append(
            f"pilot sizing: the frozen config (T={T:.0f}s, {want_tasks} task(s), "
            f"{replicates} replicate(s), {len(admitted)} model(s)) is estimated "
            f"at {sizing['chosen'].get('est_wall_h')}h and does not fit the pilot "
            "wall-clock budget; raise the budget or drop a cell"
        )

    # Phases are reported from what was actually imported, never from the fixed
    # list this script hopes for.
    phases_requested = ["latency", "diversity"] + (["tasks"] if tasks_path else [])
    phases_run = [
        phase for phase, present in (
            ("latency", bool(latency)),
            ("diversity", bool(diversity)),
            ("tasks", bool(task_rows)),
        ) if present
    ]
    substrates: dict[str, str] = {}
    ordered = list(models) + sorted(
        (set(substrate_by_phase["latency"]) | set(substrate_by_phase["diversity"]))
        - set(models)
    )
    for model in ordered:
        per_phase = {
            phase: substrate_by_phase[phase].get(model)
            for phase in ("latency", "diversity")
        }
        seen = {v for v in per_phase.values() if v}
        if not seen:
            continue
        substrates[model] = (
            next(iter(seen)) if len(seen) == 1
            else "; ".join(f"{p}={v}" for p, v in per_phase.items() if v)
        )

    payload: dict[str, Any] = {
        "ts": time.time(),
        "tier": "0.5 (canonical)",
        "status": "incomplete" if incomplete else "complete",
        "incomplete": incomplete,
        "config": {
            "models": list(models),
            "m": m,
            "K": K,
            "diversity_gate_K": gate_K,
            "safety_factor": 1.25,
            "block_budget_s": block_budget_s,
            "reconstructed_models": list(reconstruct),
            "reconstruct_latency_steps": lat_steps,
            "allow_t_relaxation": bool(allow_relaxation),
            "task_pool": [r["task"] for r in task_rows] + [
                r["task"] for r in task_errors
            ],
            "task_steps": (tasks_payload.get("config") or {}).get("task_steps"),
            "latency_steps_by_track": {
                mdl: len((latency.get(mdl) or {}).get("steps") or [])
                for mdl in latency
            },
        },
        "sources": sources,
        "substrate_by_model": substrates,
        "substrate_by_phase_model": substrate_by_phase,
        "substrate": "mixed (loopback bake bridge for gates; farplane "
                     "TEMPLATE_SNAP for task sanity)",
        "phases_requested": phases_requested,
        "phases_run": phases_run,
        "caps": tasks_payload.get("caps") or {},
        "latency": latency,
        "diversity": diversity,
        "tasks": task_rows,
        "task_errors": task_errors,
        "system_prompt_source": tasks_payload.get("system_prompt_source", "bridge"),
        "system_prompt_chars": tasks_payload.get("system_prompt_chars"),
        "verdicts": verdicts,
        "reliability": reliability,
        "reconstructions": reconstructions,
        "task_selection": selection,
        "pilot_sizing": sizing,
        "overlap_gate": {
            "branch_materialize_s": materialize_s,
            "branch_materialize_source": materialize_source,
            "b_arm_in_arms": b_in_arms,
            "charged_tail_s": tail_for_sizing,
            "priority_tail_s": tail_for(c_model),
            # b_arm_models is the key bench.tier05.write_markdown renders.
            "b_arm_models": [
                mdl for mdl in admitted
                if b_in_arms and verdicts[mdl]["b_arm_admitted"]
            ],
            "b_arm_blocked_models": b_blocked,
            "per_model": {
                mdl: {"verdict": v["overlap_verdict"], "tail_s": v["overlap_tail_s"],
                      "b_arm_admitted": v["b_arm_admitted"],
                      "detail": v["overlap_detail"]}
                for mdl, v in verdicts.items()
            },
        },
    }
    payload["frozen_pilot_config"] = {
        "arms": list(arms),
        "models": admitted,
        "c_model": c_model,
        "tasks": primary,
        "task_substitute": selection["selected"][1:2],
        "replicates": replicates,
        "T_s": T,
        "K": K,
        "diversity_gate_K": gate_K,
        "m": m,
        "run_cap": int((tasks_payload.get("caps") or {}).get("run_cap", 1)),
        "max_sandboxes": int(
            (tasks_payload.get("caps") or {}).get("max_sandboxes", 24)
        ),
        "hints_required_models": [
            mdl for mdl, v in verdicts.items() if v["hints_required"]
        ],
        # The priority block belongs to the c_model, whichever model that is;
        # every other admitted model contributes the A/B contrast only -- and
        # only for arms this pilot actually runs. A model whose arm B is not
        # admitted ships no B cell: it would measure Farplane materialisation
        # rather than the arm.
        "priority_cells": (
            [f"{c_model}|{a}" for a in arms
             if a != "B" or c_model not in b_blocked]
            + [f"{mdl}|{a}" for mdl in admitted if mdl != c_model
               for a in ("A", "B")
               if a in arms and (a != "B" or mdl not in b_blocked)]
        ),
        "b_arm_blocked_models": b_blocked,
        "est_priority_block_h": round(block_wall_s(T, K, tuple(arms)) / 3600.0, 2),
        "T_rule": t_choice["rule"],
        "T_relaxed": t_choice["relaxed"],
        "constraint_violations": t_choice["violations"],
        "status": (
            "FROZEN" if sizing_fits and primary and not b_blocked else "REFUSED"
        ),
        "executable": bool(sizing_fits and primary and not b_blocked),
        "pre_registered": True,
        "labelled": "TIER-1 PILOT (reduced T/tasks/replicates), not the full matrix",
    }
    return payload


def append_provenance(payload: dict[str, Any], path: str) -> None:
    """Append the merge-specific sections write_markdown does not know about."""
    t_choice = payload["pilot_sizing"]["T_choice"]
    c_model = payload["frozen_pilot_config"]["c_model"]
    lines = ["", "## How T was frozen", "", f"Rule: {t_choice['rule']}.", ""]
    gate = payload["overlap_gate"]
    materialize_txt = (
        f"{gate['branch_materialize_s']}s materialisation"
        if gate["branch_materialize_s"] is not None
        else "materialisation NOT MEASURED"
    )
    lines.append(
        f"Priority model's measured median step: {t_choice['slowest_step_s']}s; "
        f"one branch round at m={t_choice['m']} plus one cold direct probe "
        f"({PROBE_COLD_S}s) plus the unhidden snapshot+fork tail "
        f"({t_choice['materialize_tail_s']}s of {materialize_txt}, source "
        f"{gate['branch_materialize_source']}) = {t_choice['round_s']}s. T is "
        "sized on the model that carries the four-arm block; a slower secondary "
        "model does not shrink T, it gets the per-model feasibility check below "
        "instead."
    )
    lines.append("")
    lines.append(f"| T s | branch rounds for B | `{c_model}` priority block "
                 "(A,A×K,B,C) h | fits block budget | "
                 f">= {MIN_BRANCH_ROUNDS} rounds |")
    lines.append("|---|---|---|---|---|")
    for row in t_choice["ladder"]:
        lines.append(
            f"| {row['T_s']:.0f} | {row['branch_rounds']} | {row['block_wall_h']} | "
            f"{'yes' if row['fits_budget'] else 'no'} | "
            f"{'yes' if row['enough_rounds'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        f"Chosen **T = {t_choice['chosen_T_s']:.0f}s**. Block cost uses the Tier-0 "
        f"measured create-from-snapshot p50 {CREATE_FROM_SNAPSHOT_P50_S}s and "
        f"delete p50 {DELETE_SANDBOX_P50_S}s per sandbox, at "
        f"{{A:1, A×K:{payload['config']['K']}, B:{payload['config']['K']}, "
        f"C:{payload['config']['K']}}} sandboxes per cell and run cap 1."
    )
    if t_choice["relaxed"]:
        lines.append("")
        lines.append(
            f"**Constraint relaxed** (violations: "
            f"{', '.join(t_choice['violations']) or 'none'}): "
            f"{t_choice['relaxed']}."
        )
    lines.append("")
    lines.append("## Pilot admission (diversity gate AND branch-round feasibility)")
    lines.append("")
    lines.append("| model | diversity verdict | median step s | overlap verdict "
                 "| unhidden tail s | branch rounds at T | attempt failure rate "
                 "| enters pilot | reason if not |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model, v in payload["verdicts"].items():
        rel = v.get("reliability") or {}
        lines.append(
            f"| `{model}` | {v['diversity_verdict']} | {v['median_step_s']} | "
            f"{v.get('overlap_verdict', '-')} | {v.get('overlap_tail_s', '-')} | "
            f"{v['branch_rounds_at_T']} | "
            f"{rel.get('failure_rate', v.get('provider_retry_rate', '-'))} | "
            f"{'**yes**' if v['enters_pilot'] else 'no'} | "
            f"{v['pilot_skip_reason'] or '-'} |"
        )
    lines.append("")
    lines.append(
        "The second gate is the design's own feasibility rule ('a pilot point is "
        "only feasible with >= 2 branch rounds, otherwise arm B never converges "
        "twice and the B-vs-A×K contrast is untestable') evaluated per model at "
        "the frozen T. It is arithmetic on measured latency, not a judgement "
        "call, and it is what excludes a model that passed diversity."
    )
    if gate["b_arm_in_arms"]:
        lines.append("")
        lines.append(
            f"Arm B pays a snapshot+fork {materialize_txt} per branch round "
            f"(source {gate['branch_materialize_source']}) and can only hide it "
            "under the model's sampling wait; the exposed remainder is wall "
            f"clock inside T (charged into the sizing at "
            f"{gate['charged_tail_s']}s, the worst admitted tail). A model whose "
            "tail exceeds the pre-registered ratio -- or whose materialisation "
            "was never measured -- ships no B cell:"
        )
        lines.append("")
        for model, row in gate["per_model"].items():
            lines.append(
                f"- `{model}`: {row['verdict']} -- {row['detail']}"
            )
    for model, v in payload["verdicts"].items():
        rel = v.get("reliability")
        if rel and rel.get("failed_attempts"):
            lines.append("")
            lines.append(
                f"- `{model}` reliability during its gate: {rel['failed_attempts']} "
                f"of {rel['total_calls']} attempts failed "
                f"({rel['timeouts']} client timeouts at "
                f"{rel['max_attempt_latency_s']}s, {rel['empty_completions']} "
                "empty 200s), each one billed to wall clock and retried by the "
                "harness."
            )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| model | latency from | diversity from | latency substrate | "
                 "diversity substrate |")
    lines.append("|---|---|---|---|---|")
    by_phase = payload["substrate_by_phase_model"]
    for model in payload["config"]["models"]:
        lines.append(
            f"| `{model}` | `{payload['sources']['latency'].get(model, '-')}` | "
            f"`{payload['sources']['diversity'].get(model, '-')}` | "
            f"{by_phase['latency'].get(model, '-')} | "
            f"{by_phase['diversity'].get(model, '-')} |"
        )
    lines.append("")
    lines.append(
        "The gate track runs on the bake sandbox's exposed bridge (`/reset` "
        "between models, so the second model does not inherit the first's "
        "factory) and spends no Farplane capacity. The task-sanity track "
        "provisions one fresh sandbox per candidate task from TEMPLATE_SNAP and "
        "runs the four candidates concurrently -- they are independent "
        "trajectories on independent sandboxes, and the fork lane (the Tier-0 "
        "binding primitive) is not involved in `create_from_snapshot`."
    )
    lines.append("")
    lines.append(
        "One baked template serves every candidate task: all 24 lab-play "
        "throughput tasks are greenfield with `starting_game_state=None`, quota "
        "16 and identical trajectory settings, differing only in "
        "`throughput_entity` (verified against `THROUGHPUT_TASKS`). The entity "
        "is host-side -- it selects the goal text and is passed explicitly to "
        "`/probe` -- so the sandbox's own `FLE_ENV_ID=iron_ore_throughput` only "
        "matters for `task.verify()`, which bench mode disables (P3)."
    )
    if payload["task_errors"]:
        lines.append("")
        lines.append("**Task-sanity failures:**")
        for row in payload["task_errors"]:
            lines.append(f"- `{row['task']}`: {row.get('error')}")
    if payload.get("reconstructions"):
        lines.append("")
        lines.append("## Gate reconstruction from journals")
        lines.append("")
        lines.append("| model | gate K | usable plain seats | dead plain seats | "
                     "usable hinted seats | dead hinted seats | session |")
        lines.append("|---|---|---|---|---|---|---|")
        for model, rec in payload["reconstructions"].items():
            lines.append(
                f"| `{model}` | {rec['gate_K']} | {rec['plain_seats_usable']} | "
                f"{rec['plain_seats_dead']} | {rec['hinted_seats_usable']} | "
                f"{rec['hinted_seats_dead']} | `{rec.get('session') or '-'}` |"
            )
        lines.append("")
        lines.append(
            "A seat is one sample the gate REQUESTED. A seat whose every attempt "
            "failed produced no program and stays in the rate's denominator, so a "
            "reconstruction is never scored on its survivors; a journal that "
            "cannot account for all K seats is rejected instead of merged."
        )
        for model, rec in payload["reconstructions"].items():
            for err in rec.get("seat_errors") or []:
                lines.append(f"- `{model}` dead seat: {err}")
    lines.append("")
    lines.append(f"## Artifact status: {payload['status'].upper()}")
    lines.append("")
    if payload["incomplete"]:
        lines.append(
            "The merge ran on partial evidence. Every gap is listed here rather "
            "than filled with a default, and the frozen config below is marked "
            f"`{payload['frozen_pilot_config']['status']}`:"
        )
        lines.append("")
        for item in payload["incomplete"]:
            lines.append(f"- {item}")
    else:
        lines.append(
            "Every model requested on the command line contributed a diversity "
            "row and a step-latency row, and the task-sanity phase produced a "
            "selectable task."
        )
    lines.append("")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Merge Tier 0.5 measurement tracks")
    ap.add_argument("--track", action="append", default=[],
                    help="a tier05 payload JSON contributing latency/diversity "
                         "(a path that does not exist is an error, not a skip)")
    ap.add_argument("--reconstruct", action="append", default=[],
                    help="model whose gate must be rebuilt from its Tier-0.5 "
                         "journals (track stopped before it serialised)")
    ap.add_argument("--journal-dir", default="bench/journal/tier05")
    ap.add_argument("--tasks", default="bench/results/tier05_tasks.json",
                    help="task-sanity payload; pass '' to merge without the "
                         "task phase")
    ap.add_argument("--models", default="k3,kimi-for-coding,codex/gpt-5.6-sol")
    ap.add_argument("--arms", default="A,AxK,B,C")
    ap.add_argument("--c-model", default="k3")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gate-K", type=int, default=4,
                    help="canonical diversity gate width; every merged gate row "
                         "and every reconstruction must be this K")
    ap.add_argument("--reconstruct-latency-steps", type=int, default=3,
                    help="steps the reconstructed latency track was asked for, "
                         "so a track that died early reports as aborted")
    ap.add_argument("--reconstruct-substrate", default="loopback (bake bridge)",
                    help="substrate attested for reconstructed gate journals "
                         "(journals do not record it)")
    ap.add_argument("--branch-materialize-s", type=float, default=None,
                    help="snapshot+fork p50 arm B must hide under its sampling "
                         "wait each branch round; default reads caps."
                         "branch_materialize_s, then --tier0. Unmeasured means "
                         "arm B is not admitted, never a stand-in number")
    ap.add_argument("--tier0", default="bench/results/tier0.json",
                    help="Tier-0 soak artifact the materialisation p50 is read "
                         "from; pass '' to declare no evidence")
    ap.add_argument("--allow-t-relaxation", action="store_true",
                    help="freeze the largest affordable T even when no ladder "
                         f"point reaches {MIN_BRANCH_ROUNDS} branch rounds; the "
                         "violation is recorded in the artifact")
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--block-budget-s", type=float, default=9000.0)
    ap.add_argument("--out", default="bench/results/tier05.json")
    ap.add_argument("--md", default="bench/results/TIER05.md")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """0 = complete artifact, 1 = artifact written but INCOMPLETE, 2 = refused."""
    args = _cli().parse_args(argv)
    try:
        payload = merge(
            tracks=args.track,
            reconstruct=args.reconstruct,
            journal_dir=args.journal_dir,
            tasks_path=args.tasks,
            models=tuple(m for m in args.models.split(",") if m),
            m=args.m,
            K=args.K,
            gate_K=args.gate_K,
            block_budget_s=args.block_budget_s,
            arms=tuple(a for a in args.arms.split(",") if a),
            c_model=args.c_model,
            replicates=args.replicates,
            lat_steps=args.reconstruct_latency_steps,
            reconstruct_substrate=args.reconstruct_substrate,
            branch_materialize_s=args.branch_materialize_s,
            tier0_path=args.tier0,
            allow_relaxation=args.allow_t_relaxation,
        )
    except MergeError as exc:
        # No artifact: a merge that cannot justify its numbers writes nothing,
        # so a stale tier05.json is never mistaken for this run's result.
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    atomic_write_json(args.out, payload)
    write_markdown(payload, args.md)
    append_provenance(payload, args.md)
    print(json.dumps({
        "status": payload["status"],
        "incomplete": payload["incomplete"],
        "verdicts": payload["verdicts"],
        "selected_tasks": payload["task_selection"]["selected"],
        "frozen_pilot_config": payload["frozen_pilot_config"],
        "gate_thresholds": {"pass": DIVERSITY_PASS,
                            "conditional": DIVERSITY_CONDITIONAL},
        "json": args.out, "md": args.md,
    }, indent=2, default=str))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
