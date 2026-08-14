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

Outputs ``bench/results/tier05.json`` and ``bench/results/TIER05.md``, including
the FROZEN pilot config (T, tasks, replicates) sized so the whole Tier-1 pilot
fits in <= 3h of wall clock at achievable concurrency.
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
from bench.common import RunJournal, TimingBuckets
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
            t0 = time.monotonic()
            await run.agent_step(traj)
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
        "unparseable_steps": sum(1 for s in steps if s["errors"]),
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
        plain = await llm.sample_detailed(messages, n=cfg.K, branch="plain")
        hinted = await llm.sample_detailed(messages, n=cfg.K, hints=hints,
                                          branch="hinted")
        usage = llm.usage()
    finally:
        await llm.aclose()
        journal.close()

    def summarize(samples: Sequence[Any], label: str) -> dict[str, Any]:
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
            "median_latency_s": round(
                statistics.median([s.latency_s for s in samples]), 3
            ) if samples else None,
            "mean_completion_tokens": round(
                statistics.fmean([s.completion_tokens for s in samples]), 1
            ) if samples else None,
            "errors": [s.error for s in samples if s.error],
            "code_heads": [(c or "")[:160] for c in codes],
        }

    plain_sum = summarize(plain, "plain")
    hinted_sum = summarize(hinted, "hinted")
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
        "k_way_sampling_latency_s": plain_sum["median_latency_s"],
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
    best = max(probes, default=0.0)
    return {
        "model": model,
        "task": task,
        "entity": run.entity,
        "quota": run.quota,
        "steps": traj.step if traj else 0,
        "probes": probes,
        "best_throughput": best,
        "quota_fraction": round(best / run.quota, 4) if run.quota else 0.0,
        "final_production_score": traj.last_production if traj else 0.0,
        "errors": traj.errors if traj else 0,
        "incidents": run.incidents,
        "wall_s": run.timings.wall_s,
        "tokens": llm.usage(),
    }


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
        quota = rows[0]["quota"]
        bests = {r["model"]: r["best_throughput"] for r in rows}
        nonzero = [m for m, v in bests.items() if v > 0]
        above = [m for m, v in bests.items() if quota and v >= quota]
        ratios = [v / quota for v in bests.values() if quota and v > 0]
        # Distance in orders of magnitude from the quota; 0 means the sanity
        # probe landed exactly at quota, which is the most legible region.
        distance = (
            round(abs(statistics.fmean([math.log10(r) for r in ratios])), 4)
            if ratios else None
        )
        reasons: list[str] = []
        if len(nonzero) < need_nonzero:
            reasons.append(
                f"floor: non-zero for only {len(nonzero)} of {need_nonzero} "
                "required model(s)"
            )
        scored.append(
            {
                "task": task,
                "quota": quota,
                "best_by_model": bests,
                "nonzero_models": len(nonzero),
                "above_quota_models": above,
                "mean_quota_ratio": (
                    round(statistics.fmean(list(bests.values())) / quota, 4)
                    if quota and bests else 0.0
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
            f"{n_models} model(s) probed (floor tasks are the only "
            "disqualifier -- the endpoint is continuous, so the quota is a "
            "normaliser and not a ceiling); ranked by model coverage, then by "
            "how few orders of magnitude the sanity probe sat from the quota"
        ),
        "shortfall": max(0, want - len(selected)),
    }


# ---------------------------------------------------------------------------
# Pilot sizing
# ---------------------------------------------------------------------------


def load_tier0_caps(cfg: Tier05Config) -> dict[str, Any]:
    """Read the Tier-0 soak's per-node run cap, if Tier 0 has already run."""
    out: dict[str, Any] = {"source": "defaults", "run_cap": cfg.run_cap,
                           "max_sandboxes": cfg.max_sandboxes}
    for name in ("tier0_soak.json", "tier0.json"):
        path = os.path.join(cfg.results_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        soak = data.get("soak", data)
        for key in ("recommended_run_cap", "per_node_run_cap", "node_cap"):
            value = data.get(key, soak.get(key) if isinstance(soak, dict) else None)
            if isinstance(value, int) and value >= 1:
                out.update(source=name, run_cap=value, cap_key=key)
                break
        for key in ("max_sandboxes", "total_slots", "warm_slots"):
            value = data.get(key, soak.get(key) if isinstance(soak, dict) else None)
            if isinstance(value, int) and value >= 1:
                out["max_sandboxes"] = value
                break
        for key in ("t_snap_p50_s", "t_snap_s", "snapshot_p50_s"):
            value = (data.get(key) or (soak.get(key) if isinstance(soak, dict) else None))
            if isinstance(value, (int, float)) and value > 0:
                out["t_snap_s"] = float(value)
                break
        for key in ("t_fork_p50_s", "t_fork_s", "fork_p50_s"):
            value = (data.get(key) or (soak.get(key) if isinstance(soak, dict) else None))
            if isinstance(value, (int, float)) and value > 0:
                out["t_fork_s"] = float(value)
                break
        break
    if "t_snap_s" in out and "t_fork_s" in out:
        # v2.6 kept these for B's branch round (snapshot + K-1 forks); the probe
        # itself no longer forks, so the probe cost is the bare /probe call.
        out["branch_materialize_s"] = out["t_snap_s"] + out["t_fork_s"]
    return out


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
) -> dict[str, Any]:
    """Largest ladder point whose estimated wall clock fits the pilot budget."""
    run_cap = int(caps.get("run_cap", 6))
    max_sandboxes = int(caps.get("max_sandboxes", 24))
    slowest = max(
        (latency.get(mm, {}).get("median_step_s") or 0.0 for mm in models), default=0.0
    )

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
        rounds = (
            math.floor(T / (m * slowest + probe_s)) if slowest > 0 else None
        )
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
            "branch_rounds_slowest_model": rounds,
            "fits": est <= budget_s and (rounds is None or rounds >= 2),
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
    chosen = feasible[0] if feasible else min(ladder, key=lambda e: e["est_wall_s"])
    return {
        "chosen": chosen,
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
            "m": m,
            "safety_factor": safety_factor,
            "budget_s": budget_s,
            "peak_sandboxes_per_arm": {a: peak_sandboxes(a, K) for a in arms},
        },
        "budget_respected": bool(feasible),
    }


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
    except BaseException as exc:  # noqa: BLE001 - fall back, but record why
        journal.incident(kind="system_prompt_unavailable", detail=f"{exc}")
        return FALLBACK_SYSTEM_PROMPT, f"fallback ({type(exc).__name__}: {exc})"
    finally:
        if node is not None:
            await run.teardown([node])
        await llm.aclose()
        journal.close()


async def run_tier05(cfg: Tier05Config) -> dict[str, Any]:
    os.makedirs(cfg.journal_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
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
    caps = load_tier0_caps(cfg)
    probe_s = float(caps.get("probe_s", cfg.probe_s))

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
            except BaseException as exc:  # noqa: BLE001 - verbatim error, continue
                payload["diversity"][model] = {
                    "model": model, "verdict": "error",
                    "rationale": f"{type(exc).__name__}: {exc}",
                }

    if "latency" in cfg.phases and substrate.template_snap:
        payload["phases_run"].append("latency")
        for model in cfg.models:
            try:
                payload["latency"][model] = await measure_step_latency(
                    cfg, model, substrate
                )
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
            except BaseException as exc:  # noqa: BLE001
                return {"model": model, "task": task, "best_throughput": 0.0,
                        "quota": 0, "error": f"{type(exc).__name__}: {exc}"}

        for model in cfg.models:
            payload["tasks"].extend(
                await asyncio.gather(*(_one(model, task) for task in cfg.tasks))
            )

    # -- verdicts ----------------------------------------------------------
    verdicts: dict[str, Any] = {}
    for model in cfg.models:
        div = payload["diversity"].get(model, {})
        lat = payload["latency"].get(model, {})
        verdicts[model] = {
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
            "enters_tier1": div.get("verdict") in ("pass", "pass_with_hints",
                                                   "conditional", "not_measured"),
            "notes": div.get("rationale", ""),
        }
    payload["verdicts"] = verdicts

    admitted = [m for m, v in verdicts.items() if v["enters_tier1"]]
    sizing = size_pilot(
        models=admitted or list(cfg.models),
        arms=("A", "AxK", "B", "C"),
        c_model=(admitted or list(cfg.models))[0] if (admitted or cfg.models) else None,
        latency={m: payload["latency"].get(m, {}) for m in cfg.models},
        caps=caps,
        probe_s=probe_s,
        provision_s=cfg.provision_s,
        teardown_s=cfg.teardown_s,
        m=cfg.m,
        K=cfg.pilot_K,
        safety_factor=cfg.safety_factor,
    )
    chosen = sizing["chosen"]
    selection = select_tasks(payload["tasks"], want=int(chosen["n_tasks"])) if \
        payload["tasks"] else {"selected": [], "want": int(chosen["n_tasks"]),
                               "candidates": [], "shortfall": int(chosen["n_tasks"]),
                               "criterion": "task phase not run"}
    payload["task_selection"] = selection
    payload["pilot_sizing"] = sizing
    payload["frozen_pilot_config"] = {
        "arms": ["A", "AxK", "B", "C"],
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
            m for m, v in verdicts.items() if v["hints_required"]
        ],
        "est_wall_h": chosen["est_wall_h"],
        "n_runs": chosen["n_runs"],
        "pre_registered": True,
        "labelled": "TIER-1 PILOT (reduced T/tasks/replicates), not the full matrix",
    }
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
                 "hints required |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model, v in payload["verdicts"].items():
        usable = (
            f"{v['parsed_plain']}/{v['parsed_plain'] + v['unusable_samples']}"
            if v.get("parsed_plain") is not None
            and v.get("unusable_samples") is not None
            else "-"
        )
        lines.append(
            f"| `{model}` | {v['diversity_rate_plain']} | {v['diversity_rate_hinted']} "
            f"| {usable} | **{v['diversity_verdict']}** | {v['median_step_s']} | "
            f"{v['median_llm_s']} | {v['tokens_per_step']} | "
            f"{'yes' if v['hints_required'] else 'no'} |"
        )
    lines.append("")
    for model, v in payload["verdicts"].items():
        if v["notes"]:
            lines.append(f"- `{model}`: {v['notes']}")
    lines.append("")
    lines.append("Gate thresholds (pre-registered, K=4): distinct-program rate "
                 f">= {DIVERSITY_PASS} passes; >= {DIVERSITY_CONDITIONAL} is "
                 "conditional; below that arm B is excluded for the model. "
                 "Both providers in this matrix are temperature-locked "
                 "(Kimi: temperature must equal 1; Codex: temperature rejected), "
                 "so the hinted column is the operative one.")
    lines.append("")

    lines.append("## Task sanity probe")
    lines.append("")
    if payload["tasks"]:
        lines.append("| task | quota | best throughput by model | non-zero models | "
                     "above quota (not disqualifying) | log10 distance from quota | "
                     "sanity errors | eligible |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cand in payload["task_selection"]["candidates"]:
            bests = ", ".join(f"{m}={v:g}" for m, v in cand["best_by_model"].items())
            lines.append(
                f"| `{cand['task']}` | {cand['quota']} | {bests} | "
                f"{cand['nonzero_models']} | "
                f"{', '.join(cand.get('above_quota_models') or []) or '-'} | "
                f"{cand.get('log10_distance_from_quota', '-')} | "
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
    lines.append("```json")
    lines.append(json.dumps(frozen, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(
        f"Sizing: {chosen['n_runs']} runs, {chosen['per_run_s']:.0f}s per run "
        f"(T={chosen['T_s']:.0f}s + provisioning + teardown), run cap "
        f"{payload['pilot_sizing']['inputs']['run_cap']}, sandbox slots "
        f"{payload['pilot_sizing']['inputs']['max_sandboxes']}; binding bound "
        f"{'run-count' if chosen['bound_by_runs_s'] >= chosen['bound_by_slots_s'] else 'sandbox-slot'}"
        f", estimated wall clock **{chosen['est_wall_h']}h** against a 3h budget "
        f"(safety factor {cfg['safety_factor']})."
    )
    lines.append("")
    lines.append(
        f"Branch rounds available at T for the slowest model: "
        f"{chosen['branch_rounds_slowest_model']} (m={cfg['m']}, direct probe "
        f"{payload['pilot_sizing']['inputs']['probe_s']:.0f}s). A pilot point is "
        "only feasible with >= 2 branch rounds, otherwise arm B never converges "
        "twice and the B-vs-A×K contrast is untestable."
    )
    lines.append("")
    lines.append("Ladder searched (pre-registered preference: task coverage, then "
                 "replicates, then T):")
    lines.append("")
    lines.append("| T s | tasks | replicates | runs | est wall h | fits |")
    lines.append("|---|---|---|---|---|---|")
    for e in payload["pilot_sizing"]["ladder"]:
        lines.append(
            f"| {e['T_s']:.0f} | {e['n_tasks']} | {e['replicates']} | {e['n_runs']} | "
            f"{e['est_wall_h']} | {'yes' if e['fits'] else 'no'} |"
        )
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
    ap = argparse.ArgumentParser(description="Tier 0.5 calibration")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASK_POOL))
    ap.add_argument("--template-snap", default=os.environ.get("TEMPLATE_SNAP", ""))
    ap.add_argument("--live-url", default="",
                    help="base URL of one already-running bridge: enables the "
                         "latency phase and the real system prompt without Farplane")
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
        live_url=args.live_url, allow_loopback_tasks=args.allow_loopback_tasks,
    )
    if cfg.dry:
        cfg.models = ("fake-model",)
        cfg.journal_dir = "bench/journal/tier05-dry"
    payload = asyncio.run(run_tier05(cfg))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    write_markdown(payload, args.md)
    print(json.dumps(
        {
            "phases_run": payload["phases_run"],
            "verdicts": payload["verdicts"],
            "task_selection": payload["task_selection"]["selected"],
            "frozen_pilot_config": payload["frozen_pilot_config"],
            "json": args.out,
            "md": args.md,
        },
        indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
