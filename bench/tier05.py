"""Tier 0.5 -- per-model calibration and pilot sizing.

Three measurements, all per model, exactly as the design doc requires before
any Tier-1 spend:

(a) STEP LATENCY   5 sequential real agent steps on a live sandbox. The Tier-0
    overlap GATE is evaluated per model: faster sampling leaves a smaller window
    to hide snapshot+forks under.
(b) DIVERSITY GATE (blocker-class) K=4 completions at ONE fixed FLE prompt;
    report the distinct-program rate over whitespace/comment-normalised code.
    Both providers here are temperature-locked, so the gate is measured twice --
    plain and with the pre-registered per-branch strategy hints -- which is what
    decides whether the diversification knob is mandatory for Tier 1.
(c) TASK SANITY    a 6-step mini-trajectory per model per candidate task, scored
    with the same fixed 60s-window fork probe used everywhere else. Tasks are
    then selected JOINTLY: non-zero for at least two models, not saturating
    (below quota) for any model.

A model is ADMITTED to the pilot only when all three measurements exist and
succeeded for it; missing or errored evidence keeps it out (there is no
"not measured, therefore fine"). Arm B needs one more thing: the Tier-0
OVERLAP GATE, which puts the soak's snapshot+fork p50 against the model's
median LLM wait. Whatever sampling cannot hide is a per-branch-round tail that
is charged into the round cost that sizes T.

Outputs ``bench/results/tier05.json`` and ``bench/results/TIER05.md``, including
the FROZEN pilot config (T, tasks, replicates) sized so the whole Tier-1 pilot
fits in <= 3h of wall clock at achievable concurrency. When the evidence does
not support one -- no admitted model, no eligible task, or no ladder point that
fits the budget with >= 2 branch rounds -- the config is REFUSED
(``executable: false``, with the blockers) and the CLI exits 1 instead of
emitting a runnable config the measurements do not back.

Tier-0 CAPACITY is read with PRESENCE semantics: an explicitly null per-node
run cap means "never measured" and blocks executable sizing (liftable only by
an operator ``--node-cap``), a cap of 0 is a measurement that refuses the
pilot outright, and only a MISSING key falls back to the configured default.
An invalid or incomplete Tier-0 soak retracts every soak-derived number in
BOTH Tier-0 artifacts, so a stale copy cannot re-supply what was retracted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from bench.arms import (
    DIVERSITY_HINTS,
    GOAL_TEMPLATE,
    ArmConfig,
    FakeLLM,
    Trajectory,
    build_run,
    fake_substrate,
    task_spec,
)
from bench.common import RunJournal, TimingBuckets, atomic_write_json
from bench.llm import (
    DEFAULT_MODELS,
    LLMClient,
    distinct_program_rate,
    make_client,
    normalize_program,
)

# Candidate pool from the design doc (depth-graded; the final six are chosen
# jointly from measurements, not from this ordering).
DEFAULT_TASK_POOL: tuple[str, ...] = (
    "iron_ore_throughput",
    "iron_plate_throughput",
    "automation_science_pack_throughput",
    "electronic_circuit_throughput",
    "plastic_bar_throughput",
    "advanced_circuit_throughput",
)

#: Pre-registered thresholds for the diversity gate at K=4.
DIVERSITY_PASS = 0.75          # >=3 of 4 candidates distinct
DIVERSITY_CONDITIONAL = 0.50   # 2 of 4 distinct: enters Tier 1 only with hints

#: The single fixed prompt for the diversity gate. Identical for every model, so
#: the distinct-program rate is comparable across the matrix.
DIVERSITY_TASK = "iron_plate_throughput"
DIVERSITY_OBSERVATION = """## Step 1 Execution Results

**Program Output (STDOUT/STDERR):**
```
Inventory: {'iron-plate': 40, 'stone-furnace': 2, 'burner-mining-drill': 2,
 'transport-belt': 60, 'assembling-machine-1': 1, 'electric-mining-drill': 2,
 'small-electric-pole': 20, 'offshore-pump': 1, 'boiler': 1, 'steam-engine': 1,
 'pipe': 20, 'coal': 50}
Nearest iron-ore patch: (12, -8), size 6 tiles. Nearest coal: (-16, 4).
Nearest water: (-24, 12). No entities placed yet.
```

**Performance Results:**
- Production score: 0.0 (was 0.0)
- Score change: +0.0
- Automated production score: 0.0
- Elapsed game time: 0:00:10
- Ticks: 600 (cost +600)

Continue to step 2."""

FALLBACK_SYSTEM_PROMPT = (
    "You are an agent playing Factorio through a Python API. You may call "
    "place_entity, place_entity_next_to, connect_entities, insert_item, "
    "extract_item, harvest_resource, nearest, get_entities, inspect_inventory, "
    "get_production_stats, craft_item, set_entity_recipe and sleep. Write a "
    "single Python program per step, wrapped in a ```python block. Variables do "
    "not persist between programs; the game state does."
)

PILOT_WALL_BUDGET_S = 3 * 3600.0

#: Ladder searched for the frozen pilot config, in pre-registered preference
#: order: task coverage first, then replicates (P6 sampling replicates), then T.
T_LADDER: tuple[float, ...] = (1800.0, 1500.0, 1200.0, 900.0, 600.0)
TASKS_LADDER: tuple[int, ...] = (3, 2, 1)
REPLICATE_LADDER: tuple[int, ...] = (2, 1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Tier05Config:
    models: tuple[str, ...] = DEFAULT_MODELS
    tasks: tuple[str, ...] = DEFAULT_TASK_POOL
    template_snap: str = ""
    phases: tuple[str, ...] = ("latency", "diversity", "tasks")
    latency_steps: int = 5
    task_steps: int = 6
    #: K for the DIVERSITY GATE (pre-registered at 4: >=3 of 4 distinct).
    K: int = 4
    #: K frozen into the Tier-1 pilot config. v2.6 caps it at 2: the Tier-0
    #: same-host warm-supervisor lane cannot materialise more width per run.
    pilot_K: int = 2
    m: int = 4
    ttl_s: int = 3600
    prefix: str = "flebench-t05-"
    results_dir: str = "bench/results"
    journal_dir: str = "bench/journal/tier05"
    #: Cost model inputs for pilot sizing; overridden from tier0 results when present.
    #: v2.6: a probe is a direct /probe on the line's own sandbox -- measured at
    #: 22.2s cold (Tier 0 `probe.cycles[0].probe_s`) and ~6s warm; there is no
    #: snapshot/fork/delete cycle around it any more.
    probe_s: float = 22.0
    provision_s: float = 90.0
    teardown_s: float = 30.0
    run_cap: int = 6
    #: Operator-declared per-node concurrent run cap (``--node-cap``). It lifts
    #: an UNMEASURED (explicit-null or retracted) Tier-0 cap so sizing can
    #: proceed on a declared number; it never overrides a MEASURED cap of 0.
    node_cap_override: int | None = None
    max_sandboxes: int = 24
    dry: bool = False
    #: Base URL of a single already-running bridge. Enables the latency phase
    #: (and the real system prompt for the diversity gate) before any Farplane
    #: capacity exists. The tasks phase needs a FRESH sandbox per (model, task),
    #: which one shared container cannot provide, so it is refused here.
    live_url: str = ""
    allow_loopback_tasks: bool = False
    safety_factor: float = 1.25

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Substrate plumbing
# ---------------------------------------------------------------------------


@dataclass
class Substrate:
    farplane: Any
    bridge_factory: Any
    template_snap: str
    world: Any = None


def real_substrate(cfg: Tier05Config, tag: str) -> Substrate:
    from bench.bridge_client import Bridge
    from bench.farplane import Farplane

    os.makedirs(cfg.journal_dir, exist_ok=True)
    fp = Farplane(os.path.join(cfg.journal_dir, f"farplane-{tag}.jsonl"))
    return Substrate(farplane=fp, bridge_factory=lambda url: Bridge(url),
                     template_snap=cfg.template_snap)


def loopback_substrate(cfg: Tier05Config) -> Substrate:
    """One shared live bridge; 'forks' resolve back to it (see LoopbackFarplane)."""
    from bench.arms import LoopbackFarplane
    from bench.bridge_client import Bridge

    return Substrate(farplane=LoopbackFarplane(cfg.live_url),
                     bridge_factory=lambda url: Bridge(url),
                     template_snap="loopback")


def dry_substrate() -> Substrate:
    world, fp, bridge_factory, template = fake_substrate(latency=0.005)
    return Substrate(farplane=fp, bridge_factory=bridge_factory,
                     template_snap=template, world=world)


def make_llm(cfg: Tier05Config, model: str, journal: RunJournal) -> LLMClient:
    if cfg.dry:
        return FakeLLM(journal=journal, log_full_requests=False,
                       max_concurrency=cfg.K)
    return make_client(model, journal=journal, max_concurrency=max(4, cfg.K * 2))


def _arm_config(cfg: Tier05Config, model: str, task: str, run_id: str) -> ArmConfig:
    return ArmConfig(
        arm="A", model=model, task_key=task, T_s=1e9, K=cfg.K, m=cfg.m,
        template_snap=cfg.template_snap, ttl_s=cfg.ttl_s, prefix=cfg.prefix,
        terminal_reserve_s=0.0, probe_cost_estimate_s=0.0,
        journal_dir=cfg.journal_dir, results_dir=cfg.results_dir, run_id=run_id,
        dry=cfg.dry,
        # T_s=1e9 is Tier 0.5's "no wall-clock stop" sentinel for latency
        # probes -- there is no horizon for a lease to cover, and the sandboxes
        # here are short-lived by construction.
        lease_guard=False,
    )


def _step_breakdown(timings: TimingBuckets, t0: float, t1: float) -> dict[str, float]:
    """Bucket totals of intervals that fall inside one step's window."""
    out: dict[str, float] = {}
    for iv in timings.intervals:
        inside = max(0.0, min(iv.t1, t1) - max(iv.t0, t0))
        if inside > 0:
            out[iv.bucket] = out.get(iv.bucket, 0.0) + inside
    return {k: round(v, 4) for k, v in out.items()}


# ---------------------------------------------------------------------------
# (a) Step latency
# ---------------------------------------------------------------------------


async def measure_step_latency(
    cfg: Tier05Config, model: str, substrate: Substrate
) -> dict[str, Any]:
    run_id = f"t05-lat-{model.replace('/', '-')}"
    journal = RunJournal(os.path.join(cfg.journal_dir, f"{run_id}.jsonl"),
                         run_id=run_id, meta={"phase": "latency", "model": model})
    llm = make_llm(cfg, model, journal)
    arm_cfg = _arm_config(cfg, model, DIVERSITY_TASK, run_id)
    run = build_run(arm_cfg, farplane=substrate.farplane,
                    bridge_factory=substrate.bridge_factory, llm=llm, journal=journal)
    steps: list[dict[str, Any]] = []
    node = None
    try:
        node = await run.provision_main("lat")
        traj = Trajectory(tid="lat", node=node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        for _ in range(cfg.latency_steps):
            before = llm.usage()
            errors_before = traj.errors
            t0 = time.monotonic()
            outcome = await run.agent_step(traj)
            t1 = time.monotonic()
            after = llm.usage()
            steps.append(
                {
                    "step": traj.step,
                    "wall_s": round(t1 - t0, 3),
                    "buckets_s": _step_breakdown(run.timings, t0, t1),
                    "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                    "completion_tokens": after["completion_tokens"]
                    - before["completion_tokens"],
                    "production_score": traj.last_production,
                    # Per-step OUTCOME, not the cumulative error counter: a step
                    # is unparseable only when ITS OWN response carried no
                    # program, and an /execute failure is not a parse failure.
                    "parsed": bool(outcome.get("parsed")),
                    "step_error": bool(outcome.get("error")),
                    "errors_delta": traj.errors - errors_before,
                    "errors": traj.errors,
                }
            )
        run.timings.stop()
    finally:
        if node is not None:
            await run.teardown([node])
        await llm.aclose()
        journal.close()
    walls = [s["wall_s"] for s in steps]
    llm_waits = [s["buckets_s"].get("llm_wait", 0.0) for s in steps]
    exec_s = [s["buckets_s"].get("rollout_exec", 0.0) for s in steps]
    return {
        "model": model,
        "steps": steps,
        "median_step_s": round(statistics.median(walls), 3) if walls else None,
        "mean_step_s": round(statistics.fmean(walls), 3) if walls else None,
        "max_step_s": round(max(walls), 3) if walls else None,
        "median_llm_s": round(statistics.median(llm_waits), 3) if llm_waits else None,
        "median_exec_s": round(statistics.median(exec_s), 3) if exec_s else None,
        "tokens_per_step": (
            round(statistics.fmean([s["completion_tokens"] for s in steps]), 1)
            if steps else None
        ),
        "unparseable_steps": sum(1 for s in steps if not s["parsed"]),
        "exec_error_steps": sum(1 for s in steps if s["parsed"] and s["step_error"]),
        "incidents": run.incidents,
        "timings": run.timings.summary(),
    }


# ---------------------------------------------------------------------------
# (b) Diversity gate
# ---------------------------------------------------------------------------


def diversity_prompt(system_prompt: str) -> list[dict[str, str]]:
    goal, _entity, _quota = task_spec(DIVERSITY_TASK)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": GOAL_TEMPLATE.format(goal=goal)},
        {"role": "assistant", "content":
            "```python\nprint(inspect_inventory())\nprint(nearest(Resource.IronOre))\n```"},
        {"role": "user", "content": DIVERSITY_OBSERVATION},
    ]


def _verdict(rate_plain: float, rate_hinted: float) -> tuple[str, str]:
    if rate_plain >= DIVERSITY_PASS:
        return "pass", "sampling alone yields non-trivial diversity"
    if rate_hinted >= DIVERSITY_PASS:
        return "pass_with_hints", (
            "temperature-locked: enters Tier 1 only with the pre-registered "
            "per-branch strategy hints enabled"
        )
    if max(rate_plain, rate_hinted) >= DIVERSITY_CONDITIONAL:
        return "conditional", (
            "partial diversity; B risks degenerating toward A at K x token cost -- "
            "report per-branch identity rate alongside results"
        )
    return "fail", (
        "K branches come back near-identical even with hints: B would be A at K x "
        "cost. Excluded from arm B."
    )


async def measure_diversity(
    cfg: Tier05Config, model: str, system_prompt: str
) -> dict[str, Any]:
    run_id = f"t05-div-{model.replace('/', '-')}"
    journal = RunJournal(os.path.join(cfg.journal_dir, f"{run_id}.jsonl"),
                         run_id=run_id, meta={"phase": "diversity", "model": model})
    llm = make_llm(cfg, model, journal)
    messages = diversity_prompt(system_prompt)
    hints = [DIVERSITY_HINTS[i % len(DIVERSITY_HINTS)] for i in range(cfg.K)]
    try:
        # Time the OUTER K-way call: that is the wall clock one branch round
        # pays. Per-sample latencies hide the fan-out (K concurrent requests,
        # provider queueing and harness retries all land in the outer call).
        t0 = time.monotonic()
        plain = await llm.sample_detailed(messages, n=cfg.K, branch="plain")
        plain_wall_s = time.monotonic() - t0
        t0 = time.monotonic()
        hinted = await llm.sample_detailed(messages, n=cfg.K, hints=hints,
                                          branch="hinted")
        hinted_wall_s = time.monotonic() - t0
        usage = llm.usage()
    finally:
        await llm.aclose()
        journal.close()

    def summarize(samples: Sequence[Any], label: str, wall_s: float) -> dict[str, Any]:
        codes = [s.code for s in samples]
        rate = distinct_program_rate(codes)
        norm = [normalize_program(c or "") for c in codes]
        return {
            "label": label,
            "k": len(samples),
            "parsed": sum(1 for c in codes if c),
            "empty_responses": sum(1 for s in samples if not (s.text or "").strip()),
            "truncated": sum(
                1 for s in samples
                if "length" in (s.finish_reason or "")
                or "incomplete" in (s.finish_reason or "")
            ),
            "finish_reasons": [s.finish_reason for s in samples],
            "distinct_program_rate": round(rate, 3),
            "distinct_programs": len({n for n in norm if n}),
            "k_way_wall_s": round(wall_s, 3),
            "median_latency_s": round(
                statistics.median([s.latency_s for s in samples]), 3
            ) if samples else None,
            "mean_completion_tokens": round(
                statistics.fmean([s.completion_tokens for s in samples]), 1
            ) if samples else None,
            "errors": [s.error for s in samples if s.error],
            "code_heads": [(c or "")[:160] for c in codes],
        }

    plain_sum = summarize(plain, "plain", plain_wall_s)
    hinted_sum = summarize(hinted, "hinted", hinted_wall_s)
    verdict, rationale = _verdict(
        plain_sum["distinct_program_rate"], hinted_sum["distinct_program_rate"]
    )
    unusable = plain_sum["k"] - plain_sum["parsed"]
    if unusable:
        # A branch with no program is not a branch. This is a decoding/budget
        # problem, not diversity, and it must not be silently averaged away.
        rationale += (
            f" NOTE: {unusable}/{plain_sum['k']} plain samples yielded no "
            f"extractable program (empty={plain_sum['empty_responses']}, "
            f"truncated={plain_sum['truncated']}); raise max_tokens or the "
            "effective K is smaller than requested."
        )
    return {
        "model": model,
        "temperature": llm.spec.temperature,
        "temperature_locked": llm.spec.temperature_locked,
        "max_tokens": llm.spec.max_tokens,
        "plain": plain_sum,
        "hinted": hinted_sum,
        "verdict": verdict,
        "rationale": rationale,
        "unusable_samples": unusable,
        "k_way_sampling_latency_s": plain_sum["k_way_wall_s"],
        "k_way_sampling_latency_hinted_s": hinted_sum["k_way_wall_s"],
        "median_sample_latency_s": plain_sum["median_latency_s"],
        "usage": usage,
        # Retries here are almost all empty 200s from the provider; each one
        # costs real wall clock inside T, so the rate belongs in the calibration.
        "provider_retries": usage["retries"],
        "provider_retry_rate": (
            round(usage["retries"] / max(1, usage["calls"] + usage["retries"]), 3)
        ),
    }


# ---------------------------------------------------------------------------
# (c) Task sanity probe
# ---------------------------------------------------------------------------


async def task_sanity(
    cfg: Tier05Config, model: str, task: str, substrate: Substrate
) -> dict[str, Any]:
    run_id = f"t05-task-{model.replace('/', '-')}-{task}"
    journal = RunJournal(os.path.join(cfg.journal_dir, f"{run_id}.jsonl"),
                         run_id=run_id,
                         meta={"phase": "tasks", "model": model, "task": task})
    llm = make_llm(cfg, model, journal)
    arm_cfg = _arm_config(cfg, model, task, run_id)
    run = build_run(arm_cfg, farplane=substrate.farplane,
                    bridge_factory=substrate.bridge_factory, llm=llm, journal=journal)
    node = None
    probes: list[float] = []
    traj: Trajectory | None = None
    try:
        node = await run.provision_main("task")
        traj = Trajectory(tid="task", node=node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        for i in range(cfg.task_steps):
            await run.agent_step(traj)
            # One probe mid-trajectory and one at the end: enough to see whether
            # the task is reachable at all without paying for six probes.
            if i + 1 in (cfg.task_steps // 2, cfg.task_steps):
                probe = await run.probe_line(node, branch="task", step=traj.step,
                                             kind="sanity")
                if probe:
                    probes.append(probe["throughput"])
                    traj.conv.inject(run.probe_block(probe))
        run.timings.stop()
    finally:
        if node is not None:
            await run.teardown([node])
        await llm.aclose()
        journal.close()
    best = max(probes) if probes else None
    return {
        "model": model,
        "task": task,
        "entity": run.entity,
        "quota": run.quota,
        # No probe came back at all: the MEASUREMENT failed. Reporting 0.0 here
        # is indistinguishable from a task the model genuinely cannot move, and
        # selection would then disqualify the task as a floor on no evidence.
        "status": "ok" if probes else "no_probe",
        "steps": traj.step if traj else 0,
        "probes": probes,
        "best_throughput": best,
        "quota_fraction": (
            round(best / run.quota, 4) if best is not None and run.quota else None
        ),
        "final_production_score": traj.last_production if traj else 0.0,
        "errors": traj.errors if traj else 0,
        "incidents": run.incidents,
        "wall_s": run.timings.wall_s,
        "tokens": llm.usage(),
    }


def _sanity_measured(row: dict[str, Any]) -> bool:
    """True when this sanity row carries a real throughput measurement."""
    if row.get("error") or row.get("status", "ok") != "ok":
        return False
    return isinstance(row.get("best_throughput"), (int, float))


def _sanity_failure(row: dict[str, Any]) -> str:
    return str(row.get("error") or row.get("status") or "no measurement")


def _task_quota(task: str, rows: Sequence[dict[str, Any]]) -> tuple[int | None, str]:
    """Quota from FLE's own task registry, never from a measurement row.

    A failed sanity run carries whatever quota its half-built run object had
    (0 in the error path), and a 0 quota silently turns every ratio in the
    ranking into nothing. The registry is the authority; a row is only a last
    resort for hosts where FLE is not importable.
    """
    try:
        _goal, _entity, quota = task_spec(task)
    except Exception:  # noqa: BLE001 - FLE absent on analysis hosts, or bad key
        pass
    else:
        if quota > 0:
            return int(quota), "task_spec"
    for row in rows:
        value = row.get("quota")
        if isinstance(value, (int, float)) and value > 0:
            return int(value), "result_row"
    return None, "unknown"


def select_tasks(
    task_results: Sequence[dict[str, Any]], *, want: int
) -> dict[str, Any]:
    """Pick candidate tasks for the continuous endpoint.

    The primary endpoint is CONTINUOUS (best verified throughput of the target
    item, quota-normalised), so the quota is a NORMALISER, not a ceiling:
    exceeding it does not cap what an arm can still show, and v2.2 explicitly
    admits a task that "sits near saturation for the strongest model". The one
    fatal case is the FLOOR -- a task that reads exactly zero, where every arm
    ties at nothing and the cell carries no information.

    So: disqualify floor tasks; among the rest, rank by how close the sanity
    probe sat to the quota, because a task whose short-trajectory throughput is
    already orders of magnitude above quota is one whose structure the model
    solves in a couple of steps, leaving little for an arm difference to move.
    Tasks above quota are reported in ``above_quota_models`` but stay eligible.

    A sanity run that ERRORED, or that returned no probe at all, is missing
    evidence -- not a zero. Those rows are excluded from the ranking entirely
    and listed in ``missing_evidence``; they cannot make a task look like a
    floor, and they cannot count toward its model coverage either.

    When the sanity phase could only afford ONE model (the v2.6 pilot probes
    tasks with k3 alone), "non-zero for >=2 models" is unsatisfiable, so the
    coverage requirement degrades to every model that was actually probed and
    the reduced evidence is stated in ``criterion``.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in task_results:
        by_task.setdefault(r["task"], []).append(r)
    n_models = len({r["model"] for r in task_results})
    need_nonzero = min(2, n_models) if n_models else 2

    scored: list[dict[str, Any]] = []
    for task, rows in by_task.items():
        quota, quota_source = _task_quota(task, rows)
        usable = [r for r in rows if _sanity_measured(r)]
        missing = {
            str(r.get("model")): _sanity_failure(r)
            for r in rows if not _sanity_measured(r)
        }
        bests = {r["model"]: float(r["best_throughput"]) for r in usable}
        nonzero = [m for m, v in bests.items() if v > 0]
        above = [m for m, v in bests.items() if quota and v >= quota]
        ratios = [v / quota for v in bests.values() if quota and v > 0]
        # Mean ABSOLUTE distance in orders of magnitude from the quota; 0 means
        # the sanity probe landed exactly at quota, which is the most legible
        # region. Averaging SIGNED logs let opposite deviations cancel, so a
        # task 100x above quota for one model and 100x below for another read
        # as sitting exactly at quota.
        distance = (
            round(statistics.fmean([abs(math.log10(r)) for r in ratios]), 4)
            if ratios else None
        )
        reasons: list[str] = []
        if quota is None:
            reasons.append(
                "quota unknown: the FLE task registry is unavailable and no "
                "sanity row carried one, so the endpoint cannot be normalised"
            )
        if len(nonzero) < need_nonzero:
            reasons.append(
                f"floor: non-zero for only {len(nonzero)} of {need_nonzero} "
                f"required model(s) across {len(usable)} usable measurement(s)"
            )
            if missing:
                reasons.append(
                    "missing evidence: "
                    + "; ".join(f"{m} ({why})" for m, why in sorted(missing.items()))
                )
        scored.append(
            {
                "task": task,
                "quota": quota,
                "quota_source": quota_source,
                "best_by_model": bests,
                "nonzero_models": len(nonzero),
                "usable_measurements": len(usable),
                "missing_evidence": missing,
                "above_quota_models": above,
                "mean_quota_ratio": (
                    round(statistics.fmean(list(bests.values())) / quota, 4)
                    if quota and bests else None
                ),
                "log10_distance_from_quota": distance,
                "eligible": not reasons,
                "reasons": reasons,
                "sanity_errors": sum(int(r.get("errors") or 0) for r in rows),
                "sanity_incidents": sum(len(r.get("incidents") or []) for r in rows),
            }
        )
    eligible = [s for s in scored if s["eligible"]]
    eligible.sort(
        key=lambda s: (-s["nonzero_models"],
                       s["log10_distance_from_quota"]
                       if s["log10_distance_from_quota"] is not None else 9.9)
    )
    selected = [s["task"] for s in eligible[:want]]
    return {
        "selected": selected,
        "want": want,
        "candidates": scored,
        "criterion": (
            f"non-zero best-probe throughput for >={need_nonzero} of the "
            f"{n_models} model(s) probed (floor tasks and tasks with no "
            "registry quota are the only disqualifiers -- the endpoint is "
            "continuous, so the quota is a normaliser and not a ceiling); "
            "failed or probe-less sanity runs are missing evidence and are "
            "excluded from the ranking rather than counted as zero; ranked by "
            "model coverage, then by the MEAN ABSOLUTE number of orders of "
            "magnitude the sanity probe sat from the quota (absolute, so "
            "deviations in opposite directions cannot cancel out)"
        ),
        "shortfall": max(0, want - len(selected)),
        "excluded_measurements": sum(
            len(s["missing_evidence"]) for s in scored
        ),
    }


# ---------------------------------------------------------------------------
# Pilot sizing
# ---------------------------------------------------------------------------


def _stat(node: Any, *keys: str) -> float | None:
    """Positive float at ``keys`` inside a nested Tier-0 summary dict."""
    cur: Any = node
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None
    return float(cur) if cur > 0 else None


def _stat_n(node: Any, key: str) -> int | None:
    """Sample count behind a Tier-0 summary stat, when it reports one."""
    stat = node.get(key) if isinstance(node, dict) else None
    n = stat.get("n") if isinstance(stat, dict) else None
    return int(n) if isinstance(n, int) else None


#: Cap keys Tier 0 publishes, in descending preference. Every one of them is
#: derived from the soak stage -- including tier0.json's top-level copy -- so a
#: retracted soak poisons all of them in BOTH artifacts (R2C2).
TIER0_CAP_KEYS: tuple[str, ...] = ("recommended_run_cap", "per_node_run_cap",
                                   "node_cap")


def _cap_present(data: dict[str, Any], soak: dict[str, Any],
                 key: str) -> tuple[bool, Any]:
    """``(present, value)`` at ``key``: top level first, then the soak block.

    PRESENCE, not truthiness. Tier 0 publishes an explicit ``null`` cap for a
    node whose capacity was never measured and ``0`` for one measured to
    sustain nothing; ``dict.get`` collapses both into "absent", which is how an
    unmeasured node came to be sized on the configured default of 6.
    """
    if key in data:
        return True, data[key]
    if key in soak:
        return True, soak[key]
    return False, None


def _cap_status(value: Any) -> str:
    """R2C2 classification of one PRESENT cap value."""
    if value is None:
        return "unmeasured"
    if isinstance(value, bool) or not isinstance(value, int):
        return "malformed"
    if value >= 1:
        return "measured"
    return "zero" if value == 0 else "malformed"


def load_tier0_caps(cfg: Tier05Config,
                    *, journal: RunJournal | None = None) -> dict[str, Any]:
    """Tier-0 evidence for pilot sizing: run cap, slots, materialisation, probe.

    Reads the REAL Tier-0 schema. The soak stage reports distributions, not
    flat scalars, and it is spelled differently in each artifact
    (``latency.snapshot_s.p50`` in the flat ``tier0_soak.json``,
    ``soak.latency.snapshot_s.p50`` inside ``tier0.json``); the direct probe
    cost only exists in the full ``tier0.json`` under ``probe.cycles[*].
    probe_s``. So both files are read, each measurement is taken from the first
    file that actually carries it, and the recorded evidence path is the one
    that file really uses -- a citation nobody can resolve is not evidence.

    CAPACITY (R2C2). The per-node run cap has four states and they are four
    different statements:

    * ``int >= 1``  measured capacity; sizing uses it.
    * ``null``      PRESENT and unmeasured. Sizing on the configured default
      would invent capacity Tier 0 explicitly refused to claim, so this is a
      capacity BLOCKER, liftable only by an operator ``--node-cap``.
    * ``0``         measured ZERO capacity. Refused outright: a flag does not
      argue with a measurement.
    * ABSENT        an artifact too old to carry the key. This is the only case
      that falls back to the configured default, with a journalled warning.

    A retracted soak (INVALID marker, ``valid: false``, ``complete: false`` or
    ``conclusions_invalidated``) poisons soak-derived latency AND every cap read
    in BOTH artifacts, so tier0.json's stale top-level copy cannot re-supply
    what the marker just withdrew. Constants and probe evidence have
    independent provenance and stay usable.

    Anything Tier 0 did not measure falls back to the configured default AND is
    reported in ``warnings`` (journalled when a journal is given); anything it
    refused lands in ``blockers``, which refuses the frozen config. The
    snapshot+fork materialisation has no default at all: without it the arm-B
    overlap gate is ``unknown``, which is not the same as fast.
    """
    out: dict[str, Any] = {
        "source": "defaults",
        "sources": [],
        "run_cap": cfg.run_cap,
        "max_sandboxes": cfg.max_sandboxes,
        "evidence": {},
        "warnings": [],
        "blockers": [],
        "cap_status": "absent",
        "run_cap_backed": False,
        "soak_invalid": False,
        "soak_invalid_reasons": [],
        "pilot_K": cfg.pilot_K,
    }
    evidence: dict[str, Any] = out["evidence"]
    warnings: list[str] = out["warnings"]
    blockers: list[str] = out["blockers"]
    #: First PRESENT cap claim, whatever it says. There is no cap shopping: a
    #: null or 0 is an answer, and moving on to the next key or the next file
    #: until something reads >= 1 is how a retracted soak got re-published as
    #: capacity from tier0.json.
    cap_seen: dict[str, Any] | None = None
    #: Parsed artifacts in preference order: ``(name, data, soak, flat)``.
    artifacts: list[tuple[str, dict[str, Any], dict[str, Any], bool]] = []
    for name in ("tier0_soak.json", "tier0.json"):
        path = os.path.join(cfg.results_dir, name)
        if not os.path.exists(path):
            warnings.append(f"{name}: absent")
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{name}: not a JSON object")
            continue
        out["sources"].append(name)
        # tier0_soak.json IS the soak payload; tier0.json nests it under "soak".
        nested = data.get("soak")
        flat = not isinstance(nested, dict)
        artifacts.append((name, data, data if flat else nested, flat))

    # Tier 0 RETRACTS a soak it could not validate: the flat artifact is
    # replaced by an INVALID marker, and an abnormal exit marks the nested block
    # and files conclusions_invalidated. The stale "partial" numbers stay in
    # place, so a retraction found in EITHER file poisons the soak-derived
    # numbers in BOTH -- and it is settled BEFORE anything is read, so a stale
    # sibling that still claims validity cannot win by being read first.
    for name, data, soak, flat in artifacts:
        retracted: list[str] = []
        if data.get("valid") is False:
            retracted.append(
                f"{name}: INVALID marker "
                f"({data.get('invalid_reason') or 'no reason recorded'})"
            )
        if not flat and soak.get("valid") is False:
            retracted.append(f"{name}: soak block marked invalid")
        if soak.get("complete") is False:
            reasons = soak.get("incomplete_reasons") or []
            retracted.append(
                f"{name}: soak incomplete ("
                + ("; ".join(str(r) for r in reasons) if reasons
                   else "stage carries no completeness record")
                + ")"
            )
        stale = data.get("conclusions_invalidated")
        if isinstance(stale, dict):
            retracted.append(
                f"{name}: Tier 0 invalidated its own conclusions "
                f"({stale.get('reason') or 'no reason recorded'})"
            )
        out["soak_invalid_reasons"].extend(retracted)
        warnings.extend(
            f"{reason}; soak-derived latency and capacity are ABSENT, not zero, "
            "in every Tier-0 artifact" for reason in retracted
        )
    out["soak_invalid"] = bool(out["soak_invalid_reasons"])

    for name, data, soak, flat in artifacts:
        if cap_seen is None and not out["soak_invalid"]:
            run_cap_block = data.get("run_cap")
            if not isinstance(run_cap_block, dict):
                run_cap_block = {}
            if run_cap_block.get("valid") is False:
                cap_seen = {
                    "file": name, "path": "run_cap.valid", "value": None,
                    "status": "invalidated",
                    "detail": "; ".join(
                        str(b) for b in (run_cap_block.get("blockers") or [])
                    ),
                }
            else:
                for key in TIER0_CAP_KEYS:
                    present, value = _cap_present(data, soak, key)
                    if not present:
                        continue
                    cap_seen = {"file": name, "path": key, "value": value,
                                "status": _cap_status(value)}
                    break
        if "max_sandboxes" not in evidence:
            for key in ("max_sandboxes", "total_slots", "warm_slots"):
                if key in data:
                    value = data[key]
                elif key in soak:
                    value = soak[key]
                else:
                    continue
                # A slot ceiling that resolves out of the soak block (or out of
                # the flat soak artifact) is soak-derived like every other
                # number in it, so a retraction takes it too.
                if (flat or key not in data) and out["soak_invalid"]:
                    continue
                if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                    out["max_sandboxes"] = value
                    evidence["max_sandboxes"] = {"file": name, "path": key,
                                                 "value": value}
                    break

        latency = soak.get("latency")
        if not isinstance(latency, dict):
            latency = {}
        constants = data.get("constants")
        const_stats = constants.get("stats") if isinstance(constants, dict) else {}
        if not isinstance(const_stats, dict):
            const_stats = {}
        # The evidence path is the one THIS file uses: the flat soak artifact
        # has no "soak." prefix, and citing one made the evidence unresolvable.
        soak_prefix = "latency." if flat else "soak.latency."
        # v2.6 keeps snapshot + fork for B's branch round (the probe itself no
        # longer forks). The soak p50 is the operative number; the constants
        # stage is the fallback when the soak never ran -- and the only source
        # left when the soak was retracted.
        wanted: list[tuple[str, dict[str, Any], tuple[str, ...], str]] = []
        if not out["soak_invalid"]:
            wanted += [
                ("t_snap_s", latency, ("snapshot_s", "p50"), soak_prefix),
                ("t_fork_s", latency, ("fork_total_s", "p50"), soak_prefix),
            ]
        wanted += [
            ("t_snap_s", const_stats, ("t_snap_s", "p50"), "constants.stats."),
            ("t_fork_s", const_stats, ("fork_total_s", "p50"), "constants.stats."),
        ]
        for field_name, node, keys, prefix in wanted:
            if field_name in evidence:
                continue
            value = _stat(node, *keys)
            if value is not None:
                out[field_name] = value
                evidence[field_name] = {
                    "file": name,
                    "path": prefix + ".".join(keys),
                    "value": value,
                    "n": _stat_n(node, keys[0]),
                }
        if "probe_s" not in evidence:
            probe = data.get("probe")
            cycles = probe.get("cycles") if isinstance(probe, dict) else None
            values = [
                float(c["probe_s"]) for c in (cycles or [])
                if isinstance(c, dict)
                and isinstance(c.get("probe_s"), (int, float))
                and not isinstance(c.get("probe_s"), bool)
                and c["probe_s"] > 0
            ]
            if values:
                out["probe_s"] = round(statistics.median(values), 3)
                evidence["probe_s"] = {"file": name,
                                       "path": "probe.cycles[*].probe_s",
                                       "value": out["probe_s"], "n": len(values)}

    # -- capacity resolution (R2C2) ----------------------------------------
    override = cfg.node_cap_override
    if override is not None and _cap_status(override) != "measured":
        warnings.append(
            f"--node-cap {override!r} is not a positive integer: ignored")
        override = None
    if cap_seen is None and out["soak_invalid"]:
        cap_seen = {"file": ", ".join(out["sources"]) or "none",
                    "path": "retracted soak", "value": None,
                    "status": "retracted"}
    status = cap_seen["status"] if cap_seen else "absent"
    out["cap_status"] = status
    if cap_seen:
        out["cap_key"] = cap_seen["path"]
        out["cap_file"] = cap_seen["file"]
    if status == "measured":
        measured = int(cap_seen["value"])
        out.update(source=cap_seen["file"], run_cap=measured, run_cap_backed=True)
        evidence["run_cap"] = {"file": cap_seen["file"], "path": cap_seen["path"],
                               "value": measured}
        if override is not None and override != measured:
            out.update(source="operator", run_cap=override)
            evidence["run_cap"] = {
                "file": "operator", "path": "--node-cap", "value": override,
                "overrides": {"file": cap_seen["file"], "path": cap_seen["path"],
                              "value": measured},
            }
            warnings.append(
                f"--node-cap {override} overrides the Tier-0 MEASURED per-node "
                f"run cap {measured} ({cap_seen['file']}:{cap_seen['path']}); "
                "sizing is operator-declared from here on"
            )
    elif status == "zero":
        # A MEASUREMENT, not a gap: Tier 0 timed this node and it sustains no
        # concurrent Tier-1 run. No flag argues with a measurement.
        blockers.append(
            f"Tier 0 measured a per-node run cap of 0 ({cap_seen['file']}:"
            f"{cap_seen['path']}): the node sustains no concurrent Tier-1 run, "
            "so there is no executable pilot to size (--node-cap cannot "
            "override a measurement: fix the node or re-run the Tier-0 soak)"
        )
        if override is not None:
            warnings.append(
                f"--node-cap {override} IGNORED: an operator declaration cannot "
                "override a MEASURED zero capacity"
            )
        warnings.append(
            f"per-node run cap: measured 0; the ladder below uses the configured "
            f"default {out['run_cap']} for DIAGNOSIS only and the frozen config "
            "is refused"
        )
    elif status in ("unmeasured", "retracted", "invalidated", "malformed"):
        detail = {
            "unmeasured": (
                f"Tier 0 published an explicit null per-node run cap "
                f"({cap_seen['file']}:{cap_seen['path']}): capacity was never "
                "measured"
            ),
            "retracted": (
                "Tier 0 retracted its soak, so every soak-derived per-node run "
                "cap in both artifacts is withdrawn: "
                + "; ".join(out["soak_invalid_reasons"])
            ),
            "malformed": (
                f"Tier 0's per-node run cap is not an integer "
                f"({cap_seen['file']}:{cap_seen['path']} = "
                f"{cap_seen['value']!r})"
            ),
            "invalidated": (
                f"Tier 0 marked its own run-cap derivation invalid "
                f"({cap_seen['file']}:run_cap.valid = false"
                + (f": {cap_seen.get('detail')}" if cap_seen.get("detail") else "")
                + ")"
            ),
        }[status]
        if override is not None:
            out.update(source="operator", run_cap=override, run_cap_backed=True)
            evidence["run_cap"] = {"file": "operator", "path": "--node-cap",
                                   "value": override, "unmeasured": detail}
            warnings.append(
                f"{detail}; sizing proceeds on the OPERATOR-DECLARED cap "
                f"--node-cap {override}, which is a declaration and not Tier-0 "
                "evidence"
            )
        else:
            blockers.append(
                f"{detail}; refusing to size an executable pilot on the "
                f"configured default {out['run_cap']} -- re-run the Tier-0 soak, "
                "or declare the capacity with --node-cap N"
            )
    else:
        # ABSENT: Tier 0 publishes an explicit cap on EVERY exit path it
        # controls, so a missing key means the artifact predates that contract
        # -- never that this run declined to answer (that is an explicit null).
        # This is the ONE case the configured default may cover, and it is
        # journalled.
        warnings.append(
            "per-node run cap: no cap key at all in "
            f"{', '.join(out['sources']) or 'any Tier-0 artifact'} -- an "
            "artifact predating Tier 0's explicit-cap contract; falling back to "
            f"the configured default {out['run_cap']} (re-run Tier 0 for a "
            "measured cap)"
        )
        if override is not None:
            out.update(source="operator", run_cap=override, run_cap_backed=True)
            evidence["run_cap"] = {"file": "operator", "path": "--node-cap",
                                   "value": override}
            warnings.append(
                f"--node-cap {override} supplies the per-node run cap no Tier-0 "
                "artifact carried"
            )

    for field_name, default, what in (
        ("max_sandboxes", cfg.max_sandboxes, "sandbox slot ceiling"),
        ("probe_s", cfg.probe_s, "direct /probe cost"),
    ):
        if field_name not in evidence:
            out.setdefault(field_name, default)
            warnings.append(
                f"{what}: not measured by Tier 0; falling back to the configured "
                f"default {out[field_name]}"
            )
    # One arm-B branch round materialises a snapshot plus (pilot K - 1) forks.
    # Charging a single fork understated the round by every extra branch the
    # pilot actually opens, which flattered the overlap gate at K > 2.
    n_forks = int(cfg.pilot_K) - 1
    out["branch_materialize_forks"] = n_forks
    if n_forks < 1:
        out["branch_materialize_s"] = None
        warnings.append(
            f"snapshot+fork materialisation: pilot K={cfg.pilot_K} opens no "
            "branch fork at all, so there is no arm-B branch round to charge; "
            "the overlap gate stays unknown and arm B is not admitted"
        )
    elif "t_snap_s" in evidence and "t_fork_s" in evidence:
        out["branch_materialize_s"] = round(
            out["t_snap_s"] + n_forks * out["t_fork_s"], 3
        )
        out["branch_materialize_detail"] = (
            f"snapshot {out['t_snap_s']:.3f}s + {n_forks} fork(s) x "
            f"{out['t_fork_s']:.3f}s at pilot K={cfg.pilot_K}"
        )
    else:
        # No default: a fabricated materialisation cost would silently pass the
        # arm-B overlap gate that exists precisely to measure it.
        out["branch_materialize_s"] = None
        warnings.append(
            "snapshot+fork materialisation: not measured by Tier 0 "
            "(latency.snapshot_s.p50 / latency.fork_total_s.p50 in "
            "tier0_soak.json, soak.latency.* in tier0.json, or "
            "constants.stats.*); the arm-B overlap gate is unknown and no "
            "default is substituted"
        )
    if journal is not None:
        journal.event("tier0_caps", source=out["source"], sources=list(out["sources"]),
                      run_cap=out["run_cap"], cap_status=out["cap_status"],
                      run_cap_backed=out["run_cap_backed"],
                      soak_invalid=out["soak_invalid"],
                      max_sandboxes=out["max_sandboxes"],
                      branch_materialize_s=out["branch_materialize_s"],
                      branch_materialize_forks=n_forks,
                      probe_s=out.get("probe_s"), evidence=evidence)
        for warning in warnings:
            journal.incident(kind="tier0_evidence_missing", detail=warning)
        for blocker in blockers:
            journal.incident(kind="tier0_capacity_blocker", detail=blocker)
    return out


#: Arm-B OVERLAP GATE (design doc (a)). One B branch round materialises a
#: snapshot plus K-1 forks and hides that work under the model's sampling wait.
#: What sampling cannot hide is a per-round TAIL charged straight to T:
#:     tail = max(0, branch_materialize_s - median_llm_s)
#: Pre-registered: a tail no larger than the sampling wait it had to hide under
#: is tolerated (B pays it, and pilot sizing charges it into the branch-round
#: cost). Beyond that, more than half of every branch round is Farplane
#: latency rather than the arm, and B is not admitted for that model.
OVERLAP_TAIL_RATIO = 1.0


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def overlap_gate(median_llm_s: Any, branch_materialize_s: Any) -> dict[str, Any]:
    """Tier-0 overlap verdict for one model: can B hide snap+fork under sampling?

    ``hidden``   materialisation fits inside the sampling wait; B is free.
    ``partial``  a tail is exposed but stays within ``OVERLAP_TAIL_RATIO`` x the
                 sampling wait; B is admitted and the tail is charged into the
                 branch-round cost that sizes T.
    ``exposed``  the tail is larger than that; B would measure Farplane rather
                 than the arm, so B is not admitted for the model.
    ``unknown``  a side is missing. Fail closed: B is not admitted, because an
                 unmeasured materialisation is not the same as a hidden one.
    """
    llm_s = _positive(median_llm_s)
    mat_s = _positive(branch_materialize_s)
    if llm_s is None or mat_s is None:
        missing = []
        if llm_s is None:
            missing.append("median_llm_s (Tier 0.5 latency phase)")
        if mat_s is None:
            missing.append("branch_materialize_s (Tier 0 soak snapshot+fork p50)")
        return {
            "verdict": "unknown",
            "median_llm_s": llm_s,
            "branch_materialize_s": mat_s,
            "hidden_s": None,
            "tail_s": None,
            "b_arm_admitted": False,
            "detail": "overlap gate not evaluable: missing " + ", ".join(missing),
        }
    tail = max(0.0, mat_s - llm_s)
    if tail <= 0.0:
        verdict, admitted = "hidden", True
        detail = (f"snapshot+fork {mat_s:.1f}s hides entirely under the "
                  f"{llm_s:.1f}s sampling wait")
    elif tail <= OVERLAP_TAIL_RATIO * llm_s:
        verdict, admitted = "partial", True
        detail = (f"snapshot+fork {mat_s:.1f}s vs {llm_s:.1f}s sampling wait: "
                  f"{tail:.1f}s per branch round is exposed and is charged to T")
    else:
        verdict, admitted = "exposed", False
        detail = (f"snapshot+fork {mat_s:.1f}s vs {llm_s:.1f}s sampling wait: "
                  f"{tail:.1f}s per branch round is exposed, more than the "
                  f"{OVERLAP_TAIL_RATIO:g}x sampling wait the gate allows -- arm "
                  "B would measure Farplane latency, not the arm")
    return {
        "verdict": verdict,
        "median_llm_s": round(llm_s, 3),
        "branch_materialize_s": round(mat_s, 3),
        "hidden_s": round(min(llm_s, mat_s), 3),
        "tail_s": round(tail, 3),
        "b_arm_admitted": admitted,
        "detail": detail,
    }


def pilot_cells(models: Sequence[str], arms: Sequence[str], *,
                arm_b_models: Sequence[str], c_model: str | None) -> list[str]:
    """``"model|arm"`` cells this pilot actually runs (R2C1 ``priority_cells``).

    Mirrors :func:`size_pilot`'s arithmetic exactly: every admitted model runs
    every non-C arm, arm B only for the models the Tier-0 overlap gate admitted
    (a B cell for a blocked model would measure Farplane materialisation rather
    than the arm), and arm C is the one within-model control, on ``c_model``.
    A consumer counting planned cells has to get the same matrix the sizing was
    charged for, so this is derived here and frozen into the config rather than
    re-guessed downstream from arms x models.
    """
    b_ok = set(arm_b_models)
    cells: list[str] = []
    for model in models:
        for arm in arms:
            if arm == "B" and model not in b_ok:
                continue
            if arm == "C" and model != c_model:
                continue
            cells.append(f"{model}|{arm}")
    return cells


def peak_sandboxes(arm: str, K: int) -> int:
    """Peak concurrent sandboxes one run of ``arm`` holds.

    v2.6: probes are direct, so no arm holds a disposable measurement fork.
    A owns one sandbox; A×K owns K trajectories; B holds main + (K-1) live
    branch forks; C holds main + a (K-1) restore pool. Exp 3: ``Control`` is a
    single seat like A (a block passes K=8 to every cell, and reserving 8 slots
    for a one-agent run would idle the scheduler); ``AxK-S`` holds K seats, and
    ``Hybrid`` also peaks at K -- its refork wave starts only after the K-1
    losers are deleted.
    """
    return 1 if arm in ("A", "Control") else max(2, K)


def size_pilot(
    *,
    models: Sequence[str],
    arms: Sequence[str],
    c_model: str | None,
    latency: dict[str, dict[str, Any]],
    caps: dict[str, Any],
    probe_s: float,
    provision_s: float,
    teardown_s: float,
    m: int,
    K: int,
    safety_factor: float,
    budget_s: float = PILOT_WALL_BUDGET_S,
    materialize_tail_s: float | None = None,
) -> dict[str, Any]:
    """Largest ladder point whose estimated wall clock fits the pilot budget.

    ``materialize_tail_s`` is the worst per-branch-round snapshot+fork tail the
    admitted models could NOT hide under their sampling wait (see
    ``overlap_gate``); it is real wall clock inside T, so it is charged into the
    branch-round cost that decides how many rounds T buys. ``None`` means no
    overlap evidence was supplied and the tail is reported as unknown.

    Nothing fits => ``chosen`` is None and ``error`` says why. There is no
    "cheapest infeasible" fallback: emitting min(ladder) as a frozen config
    hands the pilot a point that was measured NOT to fit.
    """
    run_cap = int(caps.get("run_cap", 6))
    max_sandboxes = int(caps.get("max_sandboxes", 24))
    steps = {mm: _positive(latency.get(mm, {}).get("median_step_s")) for mm in models}
    unmeasured = sorted(mm for mm, v in steps.items() if v is None)
    slowest = max((v for v in steps.values() if v is not None), default=0.0)
    tail_s = _positive(materialize_tail_s) or 0.0

    def n_runs(n_tasks: int, replicates: int) -> int:
        base = len(models) * len([a for a in arms if a != "C"]) * n_tasks * replicates
        if "C" in arms and c_model:
            base += n_tasks * replicates
        return base

    def peak_weighted(n_tasks: int, replicates: int) -> float:
        total = 0.0
        for arm in arms:
            if arm == "C":
                if not c_model:
                    continue
                total += peak_sandboxes("C", K) * n_tasks * replicates
            else:
                total += peak_sandboxes(arm, K) * len(models) * n_tasks * replicates
        return total

    def estimate(T: float, n_tasks: int, replicates: int) -> dict[str, Any]:
        per_run_s = T + provision_s + teardown_s
        runs = n_runs(n_tasks, replicates)
        # Lower bounds: (1) one full run, (2) run-count waves under the run cap,
        # (3) sandbox-slot capacity. The binding one is the estimate.
        by_runs = math.ceil(runs / max(1, run_cap)) * per_run_s
        slot_seconds = peak_weighted(n_tasks, replicates) * per_run_s
        by_slots = slot_seconds / max(1, max_sandboxes)
        est = max(per_run_s, by_runs, by_slots) * safety_factor
        # A branch round is m agent steps, one direct probe, and whatever
        # snapshot+fork the sampling wait could not hide.
        round_s = m * slowest + probe_s + tail_s if slowest > 0 else None
        rounds = math.floor(T / round_s) if round_s else None
        return {
            "T_s": T,
            "n_tasks": n_tasks,
            "replicates": replicates,
            "n_runs": runs,
            "per_run_s": round(per_run_s, 1),
            "bound_by_runs_s": round(by_runs, 1),
            "bound_by_slots_s": round(by_slots, 1),
            "est_wall_s": round(est, 1),
            "est_wall_h": round(est / 3600.0, 2),
            "branch_round_s": round(round_s, 1) if round_s else None,
            "materialize_tail_s": round(tail_s, 1),
            "branch_rounds_slowest_model": rounds,
            # Fail closed: an unmeasured step latency (rounds None) used to
            # count as fitting, which passed the >=2-round requirement on no
            # evidence at all.
            "fits": (est <= budget_s and rounds is not None and rounds >= 2
                     and not unmeasured),
        }

    ladder = [
        estimate(T, n_tasks, reps)
        for n_tasks in TASKS_LADDER
        for reps in REPLICATE_LADDER
        for T in T_LADDER
    ]
    feasible = [e for e in ladder if e["fits"]]
    # Pre-registered preference: task coverage, then replicates, then T.
    feasible.sort(key=lambda e: (-e["n_tasks"], -e["replicates"], -e["T_s"]))
    out: dict[str, Any] = {
        "chosen": feasible[0] if feasible else None,
        "feasible_points": len(feasible),
        "ladder": ladder,
        "inputs": {
            "models": list(models),
            "arms": list(arms),
            "c_model": c_model,
            "run_cap": run_cap,
            "max_sandboxes": max_sandboxes,
            "probe_s": probe_s,
            "provision_s": provision_s,
            "teardown_s": teardown_s,
            "slowest_median_step_s": slowest,
            "models_without_latency": unmeasured,
            "materialize_tail_s": (
                round(tail_s, 3) if materialize_tail_s is not None else None
            ),
            "m": m,
            "safety_factor": safety_factor,
            "budget_s": budget_s,
            "peak_sandboxes_per_arm": {a: peak_sandboxes(a, K) for a in arms},
        },
        "budget_respected": bool(feasible),
    }
    if not feasible:
        cheapest = min(ladder, key=lambda e: e["est_wall_s"]) if ladder else None
        if unmeasured:
            why = ("no measured step latency for " + ", ".join(unmeasured)
                   + ": branch rounds at T are unknown")
        elif cheapest is not None and cheapest["est_wall_s"] > budget_s:
            why = (f"cheapest ladder point still needs "
                   f"{cheapest['est_wall_h']}h against a "
                   f"{budget_s / 3600.0:.1f}h budget")
        else:
            why = (f"every ladder point buys fewer than 2 branch rounds at a "
                   f"{m}-step round of "
                   f"{(m * slowest + probe_s + tail_s):.0f}s "
                   f"(tail {tail_s:.0f}s)")
        out["error"] = f"no feasible pilot point: {why}"
        out["cheapest_infeasible"] = cheapest
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def get_system_prompt(cfg: Tier05Config, substrate: Substrate) -> tuple[str, str]:
    """Real FLE system prompt from a throwaway sandbox, else the fallback."""
    if not substrate.template_snap:
        return FALLBACK_SYSTEM_PROMPT, "fallback"
    run_id = "t05-sysprompt"
    journal = RunJournal(os.path.join(cfg.journal_dir, f"{run_id}.jsonl"),
                         run_id=run_id, meta={"phase": "system_prompt"})
    arm_cfg = _arm_config(cfg, cfg.models[0], DIVERSITY_TASK, run_id)
    llm = make_llm(cfg, cfg.models[0], journal)
    run = build_run(arm_cfg, farplane=substrate.farplane,
                    bridge_factory=substrate.bridge_factory, llm=llm, journal=journal)
    node = None
    try:
        node = await run.provision_main("sysprompt")
        return run.system_prompt, "bridge"
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001 - fall back, but record why
        journal.incident(kind="system_prompt_unavailable", detail=f"{exc}")
        return FALLBACK_SYSTEM_PROMPT, f"fallback ({type(exc).__name__}: {exc})"
    finally:
        if node is not None:
            await run.teardown([node])
        await llm.aclose()
        journal.close()


async def reset_loopback_bridge(cfg: Tier05Config, substrate: Substrate, *,
                                model: str, journal: RunJournal) -> str:
    """``/reset`` the one shared --live-url bridge before a model's trajectory.

    On the loopback substrate every 'fork' resolves back to the SAME container,
    so without this the second model starts on the first model's factory --
    inheriting its entity count and therefore its /execute latency, which is
    exactly the number this phase exists to measure. The first model is reset
    too: the container may have been left dirty by a bake, a smoke run, or an
    earlier invocation.

    Returns "" on success, the failure detail otherwise.
    """
    bridge = substrate.bridge_factory(cfg.live_url)
    t0 = time.monotonic()
    try:
        await asyncio.to_thread(bridge.reset)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        detail = f"{type(exc).__name__}: {exc}"
        journal.incident(kind="loopback_reset_failed", detail=detail, model=model)
        return detail
    else:
        journal.event("loopback_reset", model=model,
                      wall_s=round(time.monotonic() - t0, 2))
        return ""
    finally:
        close = getattr(bridge, "close", None)
        if callable(close):
            close()


async def run_tier05(cfg: Tier05Config) -> dict[str, Any]:
    os.makedirs(cfg.journal_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    journal = RunJournal(os.path.join(cfg.journal_dir, "t05-orchestrator.jsonl"),
                         run_id="t05",
                         meta={"phase": "orchestrator", "models": list(cfg.models),
                               "phases": list(cfg.phases)})
    try:
        return await _run_tier05(cfg, journal)
    finally:
        journal.close()


async def _run_tier05(cfg: Tier05Config, journal: RunJournal) -> dict[str, Any]:
    if cfg.dry:
        substrate = dry_substrate()
        substrate_kind = "dry"
    elif cfg.template_snap:
        substrate = real_substrate(cfg, "tier05")
        substrate_kind = "farplane"
    elif cfg.live_url:
        substrate = loopback_substrate(cfg)
        substrate_kind = "loopback"
    else:
        substrate = Substrate(farplane=None, bridge_factory=None, template_snap="")
        substrate_kind = "none"
    # The substrate owns the template id (the dry substrate bakes its own), so
    # every ArmConfig built below reads it from here.
    cfg.template_snap = substrate.template_snap
    caps = load_tier0_caps(cfg, journal=journal)
    probe_s = float(caps.get("probe_s", cfg.probe_s))
    materialize_s = caps.get("branch_materialize_s")

    phases = list(cfg.phases)
    skipped: dict[str, str] = {}
    if substrate_kind == "loopback" and "tasks" in phases and not cfg.allow_loopback_tasks:
        phases.remove("tasks")
        skipped["tasks"] = (
            "loopback substrate: the task sanity probe needs a FRESH sandbox per "
            "(model, task); one shared container would carry state across tasks. "
            "Run with --template-snap, or --allow-loopback-tasks to override."
        )
    if substrate_kind == "none":
        for phase in ("latency", "tasks"):
            if phase in phases:
                phases.remove(phase)
                skipped[phase] = (
                    "no TEMPLATE_SNAP and no --live-url: sandbox phases need one"
                )

    payload: dict[str, Any] = {
        "ts": time.time(),
        "config": cfg.to_dict(),
        "substrate": substrate_kind,
        "phases_requested": list(cfg.phases),
        "phases_run": [],
        "caps": caps,
        "latency": {},
        "diversity": {},
        "tasks": [],
    }
    if skipped:
        payload["skipped"] = skipped
    cfg.phases = tuple(phases)

    system_prompt, prompt_source = await get_system_prompt(cfg, substrate)
    payload["system_prompt_source"] = prompt_source
    payload["system_prompt_chars"] = len(system_prompt)

    if "diversity" in cfg.phases:
        payload["phases_run"].append("diversity")
        for model in cfg.models:
            try:
                payload["diversity"][model] = await measure_diversity(
                    cfg, model, system_prompt
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - verbatim error, continue
                payload["diversity"][model] = {
                    "model": model, "verdict": "error",
                    "rationale": f"{type(exc).__name__}: {exc}",
                }

    if "latency" in cfg.phases and substrate.template_snap:
        payload["phases_run"].append("latency")
        for model in cfg.models:
            if substrate_kind == "loopback":
                failure = await reset_loopback_bridge(cfg, substrate, model=model,
                                                      journal=journal)
                if failure:
                    # Without a clean container this model's steps would be
                    # timed on the previous model's factory. Refuse the
                    # measurement rather than publish a contaminated one.
                    payload["latency"][model] = {
                        "model": model,
                        "error": f"loopback /reset failed before the latency "
                                 f"trajectory: {failure}",
                    }
                    continue
            try:
                payload["latency"][model] = await measure_step_latency(
                    cfg, model, substrate
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                payload["latency"][model] = {
                    "model": model, "error": f"{type(exc).__name__}: {exc}"
                }

    if "tasks" in cfg.phases and substrate.template_snap:
        payload["phases_run"].append("tasks")
        # Candidate tasks are independent trajectories on independent fresh
        # sandboxes, so they run concurrently: the sanity phase is otherwise
        # n_tasks x task_steps x step_latency of pure serial waiting, which
        # does not fit the pilot's wall-clock budget.
        async def _one(model: str, task: str) -> dict[str, Any]:
            try:
                return await task_sanity(cfg, model, task, substrate)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                # NOT a zero: a failed sanity run is missing evidence, and
                # select_tasks excludes it instead of reading it as a floor.
                return {"model": model, "task": task, "status": "error",
                        "best_throughput": None, "quota": None,
                        "error": f"{type(exc).__name__}: {exc}"}

        for model in cfg.models:
            payload["tasks"].extend(
                await asyncio.gather(*(_one(model, task) for task in cfg.tasks))
            )

    # -- verdicts ----------------------------------------------------------
    # Admission needs all three measurements. A phase that never ran leaves no
    # evidence, and "no evidence" is not "no objection": the model stays out.
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for row in payload["tasks"]:
        task_rows.setdefault(str(row.get("model")), []).append(row)

    verdicts: dict[str, Any] = {}
    for model in cfg.models:
        div = payload["diversity"].get(model, {})
        lat = payload["latency"].get(model, {})
        rows = task_rows.get(model, [])
        usable_tasks = [r for r in rows if _sanity_measured(r)]
        div_verdict = div.get("verdict", "not_measured")
        overlap = overlap_gate(lat.get("median_llm_s"), materialize_s)

        model_blockers: list[str] = []
        if div_verdict == "fail":
            model_blockers.append(
                "diversity gate FAIL: K branches come back near-identical")
        elif div_verdict not in ("pass", "pass_with_hints", "conditional"):
            model_blockers.append(
                f"no diversity evidence (verdict {div_verdict})"
                + (f": {div.get('rationale')}" if div_verdict == "error" else "")
            )
        if _positive(lat.get("median_step_s")) is None:
            model_blockers.append(
                "no step-latency evidence"
                + (f": {lat['error']}" if lat.get("error") else "")
            )
        if not usable_tasks:
            model_blockers.append(
                f"no usable task-sanity evidence ({len(rows)} row(s) recorded)")
        admitted_model = not model_blockers

        verdicts[model] = {
            "diversity_verdict": div_verdict,
            "diversity_rate_plain": div.get("plain", {}).get("distinct_program_rate"),
            "diversity_rate_hinted": div.get("hinted", {}).get("distinct_program_rate"),
            "parsed_plain": div.get("plain", {}).get("parsed"),
            "unusable_samples": div.get("unusable_samples"),
            "median_step_s": lat.get("median_step_s"),
            "median_llm_s": lat.get("median_llm_s"),
            "tokens_per_step": lat.get("tokens_per_step"),
            "provider_retry_rate": div.get("provider_retry_rate"),
            "usable_task_measurements": len(usable_tasks),
            "overlap_verdict": overlap["verdict"],
            "overlap_tail_s": overlap["tail_s"],
            "overlap_detail": overlap["detail"],
            "b_arm_admitted": bool(admitted_model and overlap["b_arm_admitted"]),
            "hints_required": div_verdict in ("pass_with_hints", "conditional"),
            # enters_pilot == enters_tier1 HERE: this producer's admission
            # already folds the diversity, latency and task-sanity evidence, so
            # there is no second pilot-only gate to disagree with. Both keys are
            # emitted so a consumer never has to know which producer wrote the
            # artifact (R2C1).
            "enters_tier1": admitted_model,
            "enters_pilot": admitted_model,
            "pilot_skip_reason": "; ".join(model_blockers),
            "admission_blockers": model_blockers,
            "notes": div.get("rationale", ""),
        }
        journal.event("model_admission", model=model, admitted=admitted_model,
                      diversity_verdict=div_verdict,
                      overlap_verdict=overlap["verdict"],
                      b_arm_admitted=verdicts[model]["b_arm_admitted"],
                      blockers=model_blockers)
    payload["verdicts"] = verdicts

    admitted = [m for m, v in verdicts.items() if v["enters_tier1"]]
    b_models = [m for m in admitted if verdicts[m]["b_arm_admitted"]]
    b_tails = [verdicts[m]["overlap_tail_s"] for m in b_models
               if verdicts[m]["overlap_tail_s"] is not None]
    # The tail is a cost of arm B's branch rounds: charge the worst admitted
    # one. With no B in the pilot there are no branch materialisations to hide.
    tail_for_sizing = max(b_tails) if b_tails else (0.0 if admitted else None)
    payload["overlap_gate"] = {
        "branch_materialize_s": materialize_s,
        # One branch round = snapshot + (pilot K - 1) forks, so the charge is
        # K-dependent and the report has to say which K it was measured for.
        "branch_materialize_forks": caps.get("branch_materialize_forks"),
        "branch_materialize_detail": caps.get("branch_materialize_detail"),
        "pilot_K": cfg.pilot_K,
        "tail_ratio_allowed": OVERLAP_TAIL_RATIO,
        "b_arm_models": b_models,
        "charged_tail_s": tail_for_sizing,
        "per_model": {m: {"verdict": v["overlap_verdict"], "tail_s": v["overlap_tail_s"],
                          "detail": v["overlap_detail"]}
                      for m, v in verdicts.items()},
    }

    if admitted:
        arms = ["A", "AxK"] + (["B"] if b_models else []) + ["C"]
        sizing = size_pilot(
            models=admitted,
            arms=tuple(arms),
            c_model=admitted[0],
            latency={m: payload["latency"].get(m, {}) for m in admitted},
            caps=caps,
            probe_s=probe_s,
            provision_s=cfg.provision_s,
            teardown_s=cfg.teardown_s,
            m=cfg.m,
            K=cfg.pilot_K,
            safety_factor=cfg.safety_factor,
            materialize_tail_s=tail_for_sizing,
        )
    else:
        sizing = {
            "chosen": None,
            "feasible_points": 0,
            "ladder": [],
            "inputs": {
                "models": [], "arms": [], "c_model": None,
                "run_cap": int(caps.get("run_cap", cfg.run_cap)),
                "max_sandboxes": int(caps.get("max_sandboxes", cfg.max_sandboxes)),
                "probe_s": probe_s,
                "provision_s": cfg.provision_s,
                "teardown_s": cfg.teardown_s,
                "slowest_median_step_s": 0.0,
                "models_without_latency": list(cfg.models),
                "materialize_tail_s": tail_for_sizing,
                "m": cfg.m,
                "safety_factor": cfg.safety_factor,
                "budget_s": PILOT_WALL_BUDGET_S,
                "peak_sandboxes_per_arm": {},
            },
            "budget_respected": False,
            "error": ("no model cleared Tier 0.5 calibration, so there is "
                      "nothing to size"),
        }
    chosen = sizing["chosen"]
    want = int(chosen["n_tasks"]) if chosen else max(TASKS_LADDER)
    selection = select_tasks(payload["tasks"], want=want) if payload["tasks"] else {
        "selected": [], "want": want, "candidates": [], "shortfall": want,
        "criterion": "task phase not run", "excluded_measurements": 0,
    }
    payload["task_selection"] = selection
    payload["pilot_sizing"] = sizing

    blockers: list[str] = []
    if not admitted:
        blockers.append(
            "no model cleared calibration: admission needs step latency, a "
            "diversity verdict and usable task-sanity evidence for the SAME model"
        )
    if chosen is None:
        blockers.append(str(sizing.get("error")
                            or "no ladder point fits the pilot wall-clock budget"))
    if not selection["selected"]:
        blockers.append(f"no task cleared sanity selection ({selection['criterion']})")
    # Tier-0 capacity refusals (explicit-null or measured-zero run cap, or a
    # retracted soak) are calibration blockers like any other: the ladder above
    # is diagnostic, but nothing executable may be frozen on top of them.
    blockers.extend(caps.get("blockers") or [])

    if blockers:
        payload["frozen_pilot_config"] = {
            "status": "REFUSED",
            "executable": False,
            "error": ("Tier 0.5 calibration is incomplete: refusing to emit an "
                      "executable Tier-1 pilot config"),
            "blockers": blockers,
            "per_model_blockers": {
                m: v["admission_blockers"] for m, v in verdicts.items()
                if v["admission_blockers"]
            },
            # Nothing here is runnable; the keys stay present and EMPTY so a
            # consumer that reads them cannot mistake a refusal for a config.
            "arms": [],
            "models": [],
            "tasks": [],
            "priority_cells": [],
            "models_admitted": admitted,
            "arm_b_models": b_models,
            "candidate_tasks": selection["selected"],
            "pre_registered": True,
            "labelled": "TIER-1 PILOT (reduced T/tasks/replicates), not the full matrix",
        }
        journal.incident(kind="frozen_config_refused", detail=" | ".join(blockers))
    else:
        frozen = {
            "status": "FROZEN",
            "executable": True,
            "arms": list(sizing["inputs"]["arms"]),
            "models": admitted,
            "c_model": sizing["inputs"]["c_model"],
            "tasks": selection["selected"],
            "replicates": int(chosen["replicates"]),
            "T_s": float(chosen["T_s"]),
            "K": cfg.pilot_K,
            "diversity_gate_K": cfg.K,
            "m": cfg.m,
            "run_cap": sizing["inputs"]["run_cap"],
            "max_sandboxes": sizing["inputs"]["max_sandboxes"],
            "hints_required_models": [
                m for m in admitted if verdicts[m]["hints_required"]
            ],
            "arm_b_models": b_models,
            "priority_cells": pilot_cells(
                admitted, sizing["inputs"]["arms"],
                arm_b_models=b_models, c_model=sizing["inputs"]["c_model"],
            ),
            "overlap_verdicts": {m: verdicts[m]["overlap_verdict"] for m in admitted},
            "materialize_tail_s": chosen["materialize_tail_s"],
            "branch_rounds_slowest_model": chosen["branch_rounds_slowest_model"],
            "est_wall_h": chosen["est_wall_h"],
            "n_runs": chosen["n_runs"],
            "pre_registered": True,
            "labelled": "TIER-1 PILOT (reduced T/tasks/replicates), not the full matrix",
        }
        if not b_models:
            frozen["warnings"] = [
                "arm B is excluded for every admitted model by the Tier-0 "
                "overlap gate; the B-vs-AxK contrast is NOT measurable in this "
                "pilot: "
                + "; ".join(f"{m}: {verdicts[m]['overlap_detail']}" for m in admitted)
            ]
        if selection["shortfall"]:
            frozen.setdefault("warnings", []).append(
                f"{selection['shortfall']} fewer eligible task(s) than the sized "
                f"point wants ({want}); the pilot runs the eligible subset"
            )
        payload["frozen_pilot_config"] = frozen
        journal.event("frozen_pilot_config", **{
            k: frozen[k] for k in ("arms", "models", "tasks", "T_s", "replicates",
                                   "est_wall_h", "n_runs", "arm_b_models",
                                   "priority_cells")
        })
    return payload


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_markdown(payload: dict[str, Any], path: str) -> str:
    cfg = payload["config"]
    frozen = payload["frozen_pilot_config"]
    chosen = payload["pilot_sizing"]["chosen"]
    lines: list[str] = []
    lines.append("# Tier 0.5 -- per-model calibration and frozen pilot config")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['ts']))}. "
                 f"Phases run: {', '.join(payload['phases_run']) or 'none'}. "
                 f"System prompt source: {payload['system_prompt_source']}.")
    if payload.get("skipped"):
        lines.append("")
        for phase, why in payload["skipped"].items():
            lines.append(f"- SKIPPED `{phase}`: {why}")
    lines.append("")
    lines.append("## Per-model verdicts")
    lines.append("")
    lines.append("| model | diversity (plain) | diversity (hinted) | usable programs "
                 "(plain) | verdict | median step s | median LLM s | tokens/step | "
                 "hints required | overlap gate (arm B) | admitted |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, v in payload["verdicts"].items():
        usable = (
            f"{v['parsed_plain']}/{v['parsed_plain'] + v['unusable_samples']}"
            if v.get("parsed_plain") is not None
            and v.get("unusable_samples") is not None
            else "-"
        )
        overlap = str(v.get("overlap_verdict") or "-")
        tail = v.get("overlap_tail_s")
        if isinstance(tail, (int, float)) and tail > 0:
            overlap += f" (tail {tail:.0f}s)"
        lines.append(
            f"| `{model}` | {v['diversity_rate_plain']} | {v['diversity_rate_hinted']} "
            f"| {usable} | **{v['diversity_verdict']}** | {v['median_step_s']} | "
            f"{v['median_llm_s']} | {v['tokens_per_step']} | "
            f"{'yes' if v['hints_required'] else 'no'} | {overlap} | "
            f"{'yes' if v.get('enters_tier1') else '**NO**'} |"
        )
    lines.append("")
    for model, v in payload["verdicts"].items():
        if v["notes"]:
            lines.append(f"- `{model}`: {v['notes']}")
    for model, v in payload["verdicts"].items():
        if v.get("admission_blockers"):
            lines.append(f"- `{model}` NOT ADMITTED: "
                         + "; ".join(v["admission_blockers"]))
    lines.append("")
    lines.append("Gate thresholds (pre-registered, K=4): distinct-program rate "
                 f">= {DIVERSITY_PASS} passes; >= {DIVERSITY_CONDITIONAL} is "
                 "conditional; below that arm B is excluded for the model. "
                 "Both providers in this matrix are temperature-locked "
                 "(Kimi: temperature must equal 1; Codex: temperature rejected), "
                 "so the hinted column is the operative one.")
    lines.append("")
    gate = payload.get("overlap_gate")
    if gate:
        materialize = gate.get("branch_materialize_s")
        lines.append(
            "Tier-0 OVERLAP GATE (arm B): snapshot+fork materialisation "
            + (f"{materialize:.1f}s "
               f"({gate.get('branch_materialize_detail') or 'Tier-0 soak p50'})"
               if isinstance(materialize, (int, float))
               else "NOT MEASURED by Tier 0")
            + " against each model's median LLM wait; the unhidden tail is "
            f"charged to T. A tail above {OVERLAP_TAIL_RATIO:g}x the sampling "
            "wait excludes arm B for that model. Admitted for B: "
            + (", ".join(f"`{m}`" for m in gate.get("b_arm_models") or []) or "none")
            + f"; tail charged into the branch round: {gate.get('charged_tail_s')}s."
        )
        lines.append("")

    lines.append("## Task sanity probe")
    lines.append("")
    if payload["tasks"]:
        lines.append("| task | quota | best throughput by model | non-zero models | "
                     "above quota (not disqualifying) | log10 distance from quota | "
                     "missing evidence | sanity errors | eligible |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cand in payload["task_selection"]["candidates"]:
            bests = ", ".join(
                f"{m}={v:g}" for m, v in (cand.get("best_by_model") or {}).items()
            ) or "-"
            missing = ", ".join(
                f"{m} ({why})"
                for m, why in sorted((cand.get("missing_evidence") or {}).items())
            ) or "-"
            lines.append(
                f"| `{cand['task']}` | {cand['quota']} | {bests} | "
                f"{cand['nonzero_models']} | "
                f"{', '.join(cand.get('above_quota_models') or []) or '-'} | "
                f"{cand.get('log10_distance_from_quota', '-')} | "
                f"{missing} | "
                f"{cand.get('sanity_errors', '-')} | "
                f"{'yes' if cand['eligible'] else 'no: ' + '; '.join(cand['reasons'])} |"
            )
        lines.append("")
        lines.append(f"Selection criterion: {payload['task_selection']['criterion']}.")
        if payload["task_selection"]["shortfall"]:
            lines.append(
                f"**Shortfall {payload['task_selection']['shortfall']} task(s):** fewer "
                "eligible tasks than the pilot wants; the pilot runs the eligible "
                "subset and the shortfall is reported with the results."
            )
    else:
        lines.append("Task phase not run (needs a baked TEMPLATE_SNAP).")
    lines.append("")

    lines.append("## Frozen pilot config")
    lines.append("")
    if frozen.get("executable") is False:
        lines.append(f"**REFUSED -- {frozen.get('error')}**")
        lines.append("")
        for blocker in frozen.get("blockers") or []:
            lines.append(f"- {blocker}")
        lines.append("")
        lines.append("No executable pilot config is emitted: re-run the missing "
                     "calibration phases (or fix the failing ones) and freeze again.")
        lines.append("")
    for warning in frozen.get("warnings") or []:
        lines.append(f"**WARNING:** {warning}")
        lines.append("")
    lines.append("```json")
    lines.append(json.dumps(frozen, indent=2))
    lines.append("```")
    lines.append("")
    inputs = payload["pilot_sizing"].get("inputs") or {}
    if chosen is None:
        lines.append(f"Sizing: no feasible ladder point. "
                     f"{payload['pilot_sizing'].get('error', '')}")
        cheapest = payload["pilot_sizing"].get("cheapest_infeasible")
        if cheapest:
            lines.append("")
            lines.append(
                f"Cheapest point searched: T={cheapest['T_s']:.0f}s, "
                f"{cheapest['n_tasks']} task(s), {cheapest['replicates']} "
                f"replicate(s) -> {cheapest['est_wall_h']}h and "
                f"{cheapest['branch_rounds_slowest_model']} branch round(s). "
                "Reported for diagnosis only; it is NOT frozen, because it was "
                "measured not to fit."
            )
    else:
        lines.append(
            f"Sizing: {chosen['n_runs']} runs, {chosen['per_run_s']:.0f}s per run "
            f"(T={chosen['T_s']:.0f}s + provisioning + teardown), run cap "
            f"{inputs.get('run_cap')}, sandbox slots "
            f"{inputs.get('max_sandboxes')}; binding bound "
            f"{'run-count' if chosen['bound_by_runs_s'] >= chosen['bound_by_slots_s'] else 'sandbox-slot'}"
            f", estimated wall clock **{chosen['est_wall_h']}h** against a 3h budget "
            f"(safety factor {cfg['safety_factor']})."
        )
        lines.append("")
        lines.append(
            f"Branch rounds available at T for the slowest model: "
            f"{chosen['branch_rounds_slowest_model']} (m={cfg['m']}, direct probe "
            f"{float(inputs.get('probe_s') or 0.0):.0f}s, unhidden snapshot+fork "
            f"tail {chosen.get('materialize_tail_s')}s -> branch round "
            f"{chosen.get('branch_round_s')}s). A pilot point is only feasible "
            "with >= 2 branch rounds, otherwise arm B never converges twice and "
            "the B-vs-A×K contrast is untestable."
        )
    lines.append("")
    lines.append("Ladder searched (pre-registered preference: task coverage, then "
                 "replicates, then T):")
    lines.append("")
    if payload["pilot_sizing"]["ladder"]:
        lines.append("| T s | tasks | replicates | runs | branch rounds | est wall h "
                     "| fits |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in payload["pilot_sizing"]["ladder"]:
            lines.append(
                f"| {e['T_s']:.0f} | {e['n_tasks']} | {e['replicates']} | "
                f"{e['n_runs']} | {e.get('branch_rounds_slowest_model')} | "
                f"{e['est_wall_h']} | {'yes' if e['fits'] else 'no'} |"
            )
    else:
        lines.append("Ladder not searched: nothing to size.")
    lines.append("")
    lines.append("## Standing deviations (v2.5)")
    lines.append("")
    lines.append("- Kimi models routed via the direct Kimi API, OpenAI via the Codex "
                 "subscription provider: no OpenRouter, therefore no middle-out "
                 "context transform for any model (uniform across the matrix).")
    lines.append("- Kimi rejects `n`, so K-way sampling is K concurrent requests for "
                 "every model; per-provider concurrency is capped by the run cap above.")
    lines.append("- Decoding is provider-locked, not chosen: Kimi rejects any "
                 "temperature except 1, the Codex backend rejects `temperature`, "
                 "`top_p` and `max_output_tokens` outright. Branch diversity "
                 "therefore comes from sampling at the provider default plus the "
                 "pre-registered per-branch strategy hints.")
    lines.append("- Measured provider reliability (empty 200 responses, retried by "
                 "the harness with backoff): "
                 + ", ".join(
                     f"`{m}` {v.get('provider_retry_rate')}"
                     for m, v in payload["verdicts"].items()
                     if v.get("provider_retry_rate") is not None
                 )
                 + ". Every retry costs wall clock inside T and is journaled.")
    lines.append("- This is a LABELLED Tier-1 PILOT. It is never reported as the full "
                 "180-run matrix.")
    lines.append("")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tier 0.5 calibration",
        epilog="exit 0 = frozen pilot config emitted; exit 1 = config REFUSED "
               "(incomplete/failed calibration, no eligible task, no ladder "
               "point that fits, or no Tier-0 capacity evidence to size "
               "against) -- the evidence is still written",
    )
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASK_POOL))
    ap.add_argument("--template-snap", default=os.environ.get("TEMPLATE_SNAP", ""))
    ap.add_argument("--live-url", default="",
                    help="base URL of one already-running bridge: enables the "
                         "latency phase and the real system prompt without "
                         "Farplane. The bridge is /reset before EVERY model's "
                         "latency trajectory, since all 'forks' resolve to it")
    ap.add_argument("--allow-loopback-tasks", action="store_true",
                    help="run the task phase on the shared --live-url container "
                         "(state carries across tasks; selection is then unsound)")
    ap.add_argument("--phases", default="latency,diversity,tasks")
    ap.add_argument("--K", type=int, default=4,
                    help="K for the diversity gate (pre-registered at 4)")
    ap.add_argument("--pilot-K", type=int, default=2,
                    help="K frozen into the Tier-1 pilot config (v2.6: 2)")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--latency-steps", type=int, default=5)
    ap.add_argument("--task-steps", type=int, default=6)
    ap.add_argument("--run-cap", type=int, default=6)
    ap.add_argument("--node-cap", type=int, default=None,
                    help="operator-declared per-node concurrent run cap. Lifts an "
                         "UNMEASURED Tier-0 cap (explicit null, or one retracted "
                         "with the soak) so sizing can proceed on a declared "
                         "number; it never overrides a MEASURED cap of 0")
    ap.add_argument("--max-sandboxes", type=int, default=24)
    ap.add_argument("--dry", action="store_true",
                    help="exercise all phases against the in-memory fakes")
    ap.add_argument("--out", default="bench/results/tier05.json")
    ap.add_argument("--md", default="bench/results/TIER05.md")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    cfg = Tier05Config(
        models=tuple(m for m in args.models.split(",") if m),
        tasks=tuple(t for t in args.tasks.split(",") if t),
        template_snap=args.template_snap,
        phases=tuple(p for p in args.phases.split(",") if p),
        K=args.K, pilot_K=args.pilot_K, m=args.m, latency_steps=args.latency_steps,
        task_steps=args.task_steps, run_cap=args.run_cap,
        max_sandboxes=args.max_sandboxes, dry=args.dry,
        node_cap_override=args.node_cap,
        live_url=args.live_url, allow_loopback_tasks=args.allow_loopback_tasks,
    )
    if cfg.dry:
        cfg.models = ("fake-model",)
        cfg.journal_dir = "bench/journal/tier05-dry"
    payload = asyncio.run(run_tier05(cfg))
    atomic_write_json(args.out, payload)
    write_markdown(payload, args.md)
    frozen = payload["frozen_pilot_config"]
    print(json.dumps(
        {
            "phases_run": payload["phases_run"],
            "verdicts": payload["verdicts"],
            "task_selection": payload["task_selection"]["selected"],
            "frozen_pilot_config": frozen,
            "json": args.out,
            "md": args.md,
        },
        indent=2, default=str,
    ))
    # 1 = the calibration did not produce an executable frozen pilot config
    # (missing/failed evidence, or no ladder point that fits). The evidence is
    # still written; what is refused is the executable config.
    return 0 if frozen.get("executable") else 1


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
