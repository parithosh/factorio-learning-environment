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
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from typing import Any, Sequence

from bench.llm import distinct_program_rate, extract_code, normalize_program
from bench.tier05 import (
    DIVERSITY_CONDITIONAL,
    DIVERSITY_PASS,
    _verdict,
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

#: Sandboxes a priority-block cell provisions before T starts.
PRIORITY_BLOCK = ("A", "AxK", "B", "C")

#: T is frozen from this ladder only. The ceiling is the mission wall-clock
#: reserve (the pilot must also afford the secondary cells and the analysis
#: phase); the floor is imposed by the branch-round constraint below.
T_CANDIDATES: tuple[float, ...] = (900.0, 1200.0, 1500.0)

#: Arm B must converge at least this many times or the B-vs-A×K contrast has no
#: selection events to test (design: ">= 2 branch rounds"; the pilot asks for 3
#: so that losing one round to a slow step does not void the cell).
MIN_BRANCH_ROUNDS = 3


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)



def _journal(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gate_from_journal(model: str, journal_dir: str) -> dict[str, Any]:
    """Rebuild a model's gate result from its Tier-0.5 journals.

    Needed when a gate track is stopped before it can serialise its payload:
    the journal already holds every request, response and outcome, so the
    diversity rate and the step latency are recoverable exactly rather than
    re-bought. Returns the same shape ``run_tier05`` would have produced, plus
    ``reconstructed_from``.
    """
    slug = model.replace("/", "-")
    div = _journal(os.path.join(journal_dir, f"t05-div-{slug}.jsonl"))
    lat = _journal(os.path.join(journal_dir, f"t05-lat-{slug}.jsonl"))
    calls = [r for r in div if r.get("kind") == "llm_call"]
    ok = [c for c in calls if c.get("outcome") == "ok"]
    plain = [c for c in ok if not c.get("hint")]
    hinted = [c for c in ok if c.get("hint")]
    failures = [c for c in calls if c.get("outcome") != "ok"]

    def summarize(samples: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
        codes = [extract_code(s.get("response_text") or "") for s in samples]
        norm = [normalize_program(c or "") for c in codes]
        lats = [float(s.get("latency_s") or 0.0) for s in samples]
        return {
            "label": label,
            "k": len(samples),
            "parsed": sum(1 for c in codes if c),
            "empty_responses": sum(
                1 for s in samples if not (s.get("response_text") or "").strip()
            ),
            "truncated": sum(
                1 for s in samples if "length" in str(s.get("finish_reason") or "")
            ),
            "finish_reasons": [s.get("finish_reason") for s in samples],
            "distinct_program_rate": round(distinct_program_rate(codes), 3),
            "distinct_programs": len({n for n in norm if n}),
            "median_latency_s": (
                round(statistics.median(lats), 3) if lats else None
            ),
            "mean_completion_tokens": (
                round(statistics.fmean(
                    [float(s.get("completion_tokens") or 0) for s in samples]), 1)
                if samples else None
            ),
            "errors": [],
            "code_heads": [(c or "")[:160] for c in codes],
        }

    plain_sum = summarize(plain, "plain")
    hinted_sum = summarize(hinted, "hinted")
    verdict, rationale = _verdict(
        plain_sum["distinct_program_rate"], hinted_sum["distinct_program_rate"]
    )
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
    total_calls = len(calls) + len(lat_calls)
    total_fail = len(failures) + len(lat_fail)
    return {
        "model": model,
        "reconstructed_from": [
            os.path.join(journal_dir, f"t05-div-{slug}.jsonl"),
            os.path.join(journal_dir, f"t05-lat-{slug}.jsonl"),
        ],
        "diversity": {
            "model": model,
            "temperature": 1.0,
            "temperature_locked": True,
            "plain": plain_sum,
            "hinted": hinted_sum,
            "verdict": verdict,
            "rationale": rationale,
            "unusable_samples": plain_sum["k"] - plain_sum["parsed"],
            "k_way_sampling_latency_s": plain_sum["median_latency_s"],
            "usage": {"calls": len(ok), "retries": len(failures)},
            "provider_retries": len(failures),
            "provider_retry_rate": round(
                len(failures) / max(1, len(calls)), 3
            ),
        },
        "latency": {
            "model": model,
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
            "requested_steps": 3,
            "aborted": len(steps) < 3,
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

def block_wall_s(T: float, K: int) -> float:
    """Wall clock of the k3 priority block at run cap 1 (strictly sequential)."""
    total = 0.0
    for arm in PRIORITY_BLOCK:
        n = peak_sandboxes(arm, K)
        total += T + n * CREATE_FROM_SNAPSHOT_P50_S + n * DELETE_SANDBOX_P50_S
    return total


def choose_T(
    *, slowest_step_s: float, m: int, K: int, block_budget_s: float
) -> dict[str, Any]:
    """Largest T on the ladder that satisfies both pre-registered constraints.

    1. the k3 priority block (A, A×K, B, C, run cap 1) fits ``block_budget_s``;
    2. arm B gets >= :data:`MIN_BRANCH_ROUNDS` branch rounds at the slowest
       admitted model's measured step latency, counting the direct probe.
    """
    round_s = m * slowest_step_s + PROBE_COLD_S
    rows = []
    for T in T_CANDIDATES:
        rounds = math.floor(T / round_s) if round_s > 0 else 0
        block = block_wall_s(T, K)
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
    if ok:
        chosen, relaxed = max(ok, key=lambda r: r["T_s"]), ""
    else:
        # Relaxation order is pre-registered: give up the third branch round
        # before giving up the block budget, because an over-budget block
        # cannot be run at all.
        affordable = [r for r in rows if r["fits_budget"]]
        chosen = max(affordable or rows, key=lambda r: r["T_s"])
        relaxed = (
            "no ladder point met both constraints; kept the block budget and "
            f"accepted {chosen['branch_rounds']} branch round(s)"
        )
    return {
        "chosen_T_s": chosen["T_s"],
        "ladder": rows,
        "round_s": round(round_s, 1),
        "slowest_step_s": slowest_step_s,
        "m": m,
        "K": K,
        "block_budget_s": block_budget_s,
        "min_branch_rounds": MIN_BRANCH_ROUNDS,
        "relaxed": relaxed,
        "rule": (
            "largest T with (a) k3 priority block A/A×K/B/C at run cap 1 inside "
            f"the block budget and (b) >= {MIN_BRANCH_ROUNDS} branch rounds at "
            "m x measured-median-step + one cold direct probe"
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
) -> dict[str, Any]:
    latency: dict[str, Any] = {}
    diversity: dict[str, Any] = {}
    sources: dict[str, Any] = {"latency": {}, "diversity": {}, "tasks": tasks_path}
    substrates: dict[str, str] = {}
    for path in tracks:
        if not os.path.exists(path):
            continue
        data = _load(path)
        for model, row in (data.get("latency") or {}).items():
            if row.get("median_step_s") is not None:
                latency[model] = row
                sources["latency"][model] = path
                substrates[model] = data.get("substrate", "?")
        for model, row in (data.get("diversity") or {}).items():
            if row.get("verdict"):
                diversity[model] = row
                sources["diversity"][model] = path

    reliability: dict[str, Any] = {}
    for model in reconstruct:
        rebuilt = gate_from_journal(model, journal_dir)
        if rebuilt["diversity"]["plain"]["k"]:
            diversity[model] = rebuilt["diversity"]
            sources["diversity"][model] = "; ".join(rebuilt["reconstructed_from"])
        if rebuilt["latency"]["median_step_s"] is not None:
            latency[model] = rebuilt["latency"]
            sources["latency"][model] = "; ".join(rebuilt["reconstructed_from"])
            substrates[model] = "loopback (bake bridge)"
        reliability[model] = rebuilt["reliability"]

    tasks_payload = _load(tasks_path) if os.path.exists(tasks_path) else {}
    task_rows = [r for r in (tasks_payload.get("tasks") or []) if not r.get("error")]
    task_errors = [r for r in (tasks_payload.get("tasks") or []) if r.get("error")]

    verdicts = {
        model: verdict_row(diversity.get(model, {}), latency.get(model, {}))
        for model in models
    }
    admitted = [m_ for m_, v in verdicts.items() if v["enters_tier1"]]

    # T is sized on the PRIORITY model (the one carrying the four-arm block);
    # a secondary model that turns out slower does not get to shrink the block
    # the whole contrast depends on -- it gets its own feasibility check below.
    priority_step = (latency.get(c_model) or {}).get("median_step_s") or 0.0
    t_choice = choose_T(
        slowest_step_s=priority_step, m=m, K=K, block_budget_s=block_budget_s
    )
    T = t_choice["chosen_T_s"]

    # Design rule, applied per model: a cell is only worth running if arm B can
    # converge at least twice inside T at that model's measured step latency.
    # A model that fails this is skipped with the arithmetic as the reason.
    for model, v in verdicts.items():
        step = (latency.get(model) or {}).get("median_step_s")
        v["reliability"] = reliability.get(model)
        if step:
            rounds = math.floor(T / (m * step + PROBE_COLD_S))
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

    sizing = size_pilot(
        models=admitted or list(models),
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
    )
    # T is frozen by choose_T (mission wall clock + branch-round floor), not by
    # the generic ladder; report the ladder point that matches it.
    matching = [e for e in sizing["ladder"] if e["T_s"] == T]
    if matching:
        sizing["chosen"] = max(matching, key=lambda e: (e["n_tasks"], e["replicates"]))
    sizing["chosen"]["T_s"] = T
    sizing["T_choice"] = t_choice

    selection = select_tasks(task_rows, want=2) if task_rows else {
        "selected": [], "want": 2, "candidates": [], "shortfall": 2,
        "criterion": "task phase produced no usable rows",
    }
    # The pilot affords ONE task at run cap 1; the ranked runner-up is recorded
    # as the pre-registered substitute if the primary turns out degenerate.
    primary = selection["selected"][:1]

    payload: dict[str, Any] = {
        "ts": time.time(),
        "tier": "0.5 (canonical)",
        "config": {
            "models": list(models),
            "m": m,
            "K": K,
            "diversity_gate_K": gate_K,
            "safety_factor": 1.25,
            "block_budget_s": block_budget_s,
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
        "substrate": "mixed (loopback bake bridge for gates; farplane "
                     "TEMPLATE_SNAP for task sanity)",
        "phases_requested": ["latency", "diversity", "tasks"],
        "phases_run": ["latency", "diversity"] + (["tasks"] if task_rows else []),
        "caps": tasks_payload.get("caps") or {},
        "latency": latency,
        "diversity": diversity,
        "tasks": task_rows,
        "task_errors": task_errors,
        "system_prompt_source": tasks_payload.get("system_prompt_source", "bridge"),
        "system_prompt_chars": tasks_payload.get("system_prompt_chars"),
        "verdicts": verdicts,
        "reliability": reliability,
        "task_selection": selection,
        "pilot_sizing": sizing,
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
        "priority_cells": (
            [f"k3|{a}" for a in arms]
            + [f"{mdl}|{a}" for mdl in admitted if mdl != "k3" for a in ("A", "B")]
        ),
        "est_priority_block_h": round(block_wall_s(T, K) / 3600.0, 2),
        "T_rule": t_choice["rule"],
        "pre_registered": True,
        "labelled": "TIER-1 PILOT (reduced T/tasks/replicates), not the full matrix",
    }
    return payload


def append_provenance(payload: dict[str, Any], path: str) -> None:
    """Append the merge-specific sections write_markdown does not know about."""
    t_choice = payload["pilot_sizing"]["T_choice"]
    lines = ["", "## How T was frozen", "", f"Rule: {t_choice['rule']}.", ""]
    lines.append(
        f"Priority model's measured median step: {t_choice['slowest_step_s']}s; "
        f"one branch round at m={t_choice['m']} plus one cold direct probe "
        f"({PROBE_COLD_S}s) = {t_choice['round_s']}s. T is sized on the model "
        "that carries the four-arm block; a slower secondary model does not "
        "shrink T, it gets the per-model feasibility check below instead."
    )
    lines.append("")
    lines.append("| T s | branch rounds for B | k3 priority block (A,A×K,B,C) h | "
                 "fits block budget | >= 3 rounds |")
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
        lines.append(f"**Constraint relaxed:** {t_choice['relaxed']}.")
    lines.append("")
    lines.append("## Pilot admission (diversity gate AND branch-round feasibility)")
    lines.append("")
    lines.append("| model | diversity verdict | median step s | branch rounds at T "
                 "| attempt failure rate | enters pilot | reason if not |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, v in payload["verdicts"].items():
        rel = v.get("reliability") or {}
        lines.append(
            f"| `{model}` | {v['diversity_verdict']} | {v['median_step_s']} | "
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
    lines.append("| model | latency from | diversity from | substrate |")
    lines.append("|---|---|---|---|")
    for model in payload["config"]["models"]:
        lines.append(
            f"| `{model}` | `{payload['sources']['latency'].get(model, '-')}` | "
            f"`{payload['sources']['diversity'].get(model, '-')}` | "
            f"{payload['substrate_by_model'].get(model, '-')} |"
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
    lines.append("")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Merge Tier 0.5 measurement tracks")
    ap.add_argument("--track", action="append", default=[],
                    help="a tier05 payload JSON contributing latency/diversity")
    ap.add_argument("--reconstruct", action="append", default=[],
                    help="model whose gate must be rebuilt from its Tier-0.5 "
                         "journals (track stopped before it serialised)")
    ap.add_argument("--journal-dir", default="bench/journal/tier05")
    ap.add_argument("--tasks", default="bench/results/tier05_tasks.json")
    ap.add_argument("--models", default="k3,kimi-for-coding,codex/gpt-5.6-sol")
    ap.add_argument("--arms", default="A,AxK,B,C")
    ap.add_argument("--c-model", default="k3")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gate-K", type=int, default=4)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--block-budget-s", type=float, default=9000.0)
    ap.add_argument("--out", default="bench/results/tier05.json")
    ap.add_argument("--md", default="bench/results/TIER05.md")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
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
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    write_markdown(payload, args.md)
    append_provenance(payload, args.md)
    print(json.dumps({
        "verdicts": payload["verdicts"],
        "selected_tasks": payload["task_selection"]["selected"],
        "frozen_pilot_config": payload["frozen_pilot_config"],
        "gate_thresholds": {"pass": DIVERSITY_PASS,
                            "conditional": DIVERSITY_CONDITIONAL},
        "json": args.out, "md": args.md,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
