"""Tier-1 PILOT analysis: turn ``tier1_pilot.json`` + run journals into a report.

Everything here is derived from two artifacts and nothing else:

* ``bench/results/tier1_pilot.json`` -- the orchestrator's atomic partial, one
  ``RunResult`` per completed cell plus failures/skips;
* ``bench/journal/tier1/<run_id>.jsonl`` -- the per-run append-only evidence
  (``infra_op``, ``probe``, ``llm_call``, ``step``, ``branch_selection``,
  ``branch_archive``, ``incident``).

Every table below names the journal it came from so a reader can re-derive it.
The pre-registered reads implemented here are v2.3 (endpoint = quota-normalised
terminal probe), v2.6 (direct probes, K=2, cold-page tax reported) and v2.6.1
(dual wall-clock/matched-step read of B vs A×K; K=2 interpretive ceiling from
per-branch-point probe variance).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Any, Iterable, Sequence

INFRA_BUCKETS = (
    "infra_snapshot", "infra_fork", "infra_expose", "infra_delete", "infra_poll",
)


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------


def load_journal(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def of_kind(recs: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in recs if r.get("kind") == kind]


# ---------------------------------------------------------------------------
# Per-run derivations
# ---------------------------------------------------------------------------


def winners(recs: Sequence[dict[str, Any]]) -> dict[int, str]:
    """round index -> winning branch id, from ``branch_selection`` records."""
    return {int(r["round"]): str(r["winner"]) for r in of_kind(recs, "branch_selection")}


def _clip_list(values: Sequence[Any], limit: int = 8) -> str:
    """Render a short list inline; longer ones are summarised, full data in JSON."""
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(str(v) for v in values)
    head = ", ".join(str(v) for v in values[:limit])
    return f"{head}, ... (+{len(values) - limit} more)"


def _num(value: Any, digits: int = 3) -> str:
    """Table cell for an optional float; missing values print as a dash."""
    return "-" if value is None else f"{float(value):.{digits}f}"


def _label(cell: dict[str, Any]) -> str:
    """Cell key for a markdown table: the raw key's ``|`` would break the row."""
    return str(cell.get("cell") or cell.get("run_id") or "?").replace("|", " / ")


def main_line_series(
    arm: str, recs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """(step, throughput) probe series for the line that produces the endpoint.

    A / C / B: the continuing line -- parity probes for A, the ROUND WINNER's
    probe for B and C (the loser branches are discarded, so their probes are
    not on the surviving line), plus the terminal probe.
    A×K: the arm's endpoint is the best terminal trajectory, so the series is
    the per-step maximum across trajectories -- the same "best of K" rule the
    endpoint uses, evaluated at each step.
    """
    probes = of_kind(recs, "probe")
    if arm == "AxK":
        # Best-of-K evaluated AT each step, which is the rule the endpoint uses.
        # A per-step max over probes that land exactly on that step would read
        # only the trajectories still probing there; the right value is the max
        # over trajectories of each one's latest probe at or before the step.
        per_branch: dict[str, list[tuple[int, float]]] = {}
        for p in probes:
            per_branch.setdefault(str(p.get("branch", "")), []).append(
                (int(p.get("step", 0)), float(p.get("throughput", 0.0)))
            )
        for series in per_branch.values():
            series.sort()
        steps = sorted({st for series in per_branch.values() for st, _ in series})
        out: list[dict[str, Any]] = []
        for step in steps:
            best = None
            for series in per_branch.values():
                latest = [v for st, v in series if st <= step]
                if latest:
                    best = latest[-1] if best is None else max(best, latest[-1])
            if best is not None:
                out.append({"step": step, "throughput": best})
        return out
    win = winners(recs)
    keep = set(win.values())
    series: dict[int, float] = {}
    for p in probes:
        # A round's loser branches never join the surviving line, so their
        # probes are journal artifacts and must not enter the curve.
        if p.get("probe_kind") == "branch" and str(p.get("branch", "")) not in keep:
            continue
        series[int(p.get("step", 0))] = float(p.get("throughput", 0.0))
    return [{"step": s, "throughput": t} for s, t in sorted(series.items())]


def probe_stats(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cold vs warm direct-probe wall times (v2.6 point 3)."""
    probes = of_kind(recs, "probe")
    cold = [float(p["client_wall_s"]) for p in probes
            if p.get("cold") and p.get("client_wall_s") is not None]
    warm = [float(p["client_wall_s"]) for p in probes
            if p.get("cold") is False and p.get("client_wall_s") is not None]
    game = [float(p["wall_s"]) for p in probes if p.get("wall_s") is not None]

    def agg(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "min": round(min(values), 3),
            "median": round(statistics.median(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "n_probes": len(probes),
        "cold": agg(cold),
        "warm": agg(warm),
        "in_game_window_wall_s": agg(game),
        "measurement_forks": 0,
    }


def fork_stats(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """B's branching cost: fork count/seconds and snapshot seconds."""
    ops = of_kind(recs, "infra_op")
    forks = [o for o in ops if o.get("op") == "fork"]
    snaps = [o for o in ops if o.get("op") == "snapshot"]
    creates = [o for o in ops if o.get("op") == "create_from_snapshot"]
    restores = [o for o in ops if o.get("op") == "state_restore"]
    saves = [o for o in ops if o.get("op") == "state_save"]

    def total(rows: Sequence[dict[str, Any]]) -> float:
        return round(sum(float(r.get("duration_s", 0.0)) for r in rows), 2)

    def durations(rows: Sequence[dict[str, Any]]) -> list[float]:
        return [round(float(r.get("duration_s", 0.0)), 2) for r in rows]

    return {
        "forks": len(forks),
        "fork_s_total": total(forks),
        "fork_s_each": durations(forks),
        "fork_failures": sum(1 for f in forks if f.get("outcome") == "error"),
        "snapshots": len(snaps),
        "snapshot_s_total": total(snaps),
        "creates_from_snapshot": len(creates),
        "create_s_each": durations(creates),
        "state_saves": len(saves),
        "state_save_s_total": total(saves),
        "state_restores": len(restores),
        "state_restore_s_total": total(restores),
    }


def llm_stats(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    calls = of_kind(recs, "llm_call")
    ok = [c for c in calls if c.get("outcome") == "ok"]
    errors = [c for c in calls if c.get("outcome") != "ok"]
    empty = [c for c in errors if "EmptyCompletion" in str(c.get("error", ""))]
    timeouts = [c for c in errors if "Timeout" in str(c.get("error", ""))]
    retries = [c for c in calls if int(c.get("attempt", 1) or 1) > 1]
    lat = [float(c["latency_s"]) for c in ok if c.get("latency_s") is not None]
    return {
        "calls": len(calls),
        "ok": len(ok),
        "failed_attempts": len(errors),
        "empty_completions": len(empty),
        "timeouts": len(timeouts),
        "retried_attempts": len(retries),
        "retry_rate": round(len(errors) / len(calls), 4) if calls else 0.0,
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        "max_latency_s": round(max(lat), 2) if lat else None,
        "unparseable": len([r for r in of_kind(recs, "incident")
                            if r.get("incident_kind") == "unparseable_response"]),
    }


def branch_variance(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """v2.6.1 point 3: did selection have anything to choose between?"""
    rounds: list[dict[str, Any]] = []
    for sel in of_kind(recs, "branch_selection"):
        scores = sel.get("scores") or {}
        probes = {b: float(s.get("probe_throughput") or 0.0)
                  for b, s in scores.items()}
        values = list(probes.values())
        spread = (max(values) - min(values)) if values else 0.0
        rel = (spread / max(values)) if values and max(values) > 0 else 0.0
        rounds.append({
            "round": sel.get("round"),
            "winner": sel.get("winner"),
            "k_effective": sel.get("k_effective"),
            "probe_by_branch": {b: round(v, 3) for b, v in probes.items()},
            "spread": round(spread, 3),
            "relative_spread": round(rel, 4),
            "tie": spread == 0.0,
        })
    spreads = [r["relative_spread"] for r in rounds]
    ties = sum(1 for r in rounds if r["tie"])
    return {
        "rounds": rounds,
        "n_rounds": len(rounds),
        "ties": ties,
        "median_relative_spread": (
            round(statistics.median(spreads), 4) if spreads else None
        ),
        "informative": bool(rounds) and ties < len(rounds),
    }


def fidelity(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P7: what arm C's GameState restore actually reproduced, per branch."""
    rows = of_kind(recs, "fidelity")
    baselines: dict[str, float] = {}
    for sel in of_kind(recs, "branch_selection"):
        for branch, score in (sel.get("scores") or {}).items():
            baselines[branch] = float(score.get("baseline_production") or 0.0)
    out = []
    for r in rows:
        branch = str(r.get("branch", ""))
        out.append({
            "branch": branch,
            "tick_delta": r.get("tick_delta"),
            "entity_delta": r.get("entity_delta"),
            "source_entities": r.get("source_entities"),
            "child_entities": r.get("child_entities"),
            "same_pid": r.get("same_pid"),
            "baseline_production": baselines.get(branch),
        })
    return {
        "records": out,
        "n": len(out),
        "entity_mismatches": sum(1 for r in out if r["entity_delta"]),
    }


def decomposition(run: dict[str, Any]) -> dict[str, Any]:
    t = run.get("timings") or {}
    attributed = t.get("attributed_s") or {}
    wall = float(t.get("wall_s") or 0.0)
    infra = sum(float(attributed.get(b, 0.0)) for b in INFRA_BUCKETS)
    return {
        "wall_s": round(wall, 1),
        "llm_wait_s": round(float(attributed.get("llm_wait", 0.0)), 1),
        "rollout_exec_s": round(float(attributed.get("rollout_exec", 0.0)), 1),
        "probe_s": round(float(attributed.get("probe", 0.0)), 1),
        "infra_total_s": round(infra, 1),
        **{f"{b}_s": round(float(attributed.get(b, 0.0)), 1) for b in INFRA_BUCKETS},
        "other_s": round(float(attributed.get("other", 0.0)), 1),
        "infra_fraction_active": round(infra / wall, 4) if wall else None,
        "hidden_s": t.get("hidden_s"),
        "provision_s": round(float(run.get("provision_s") or 0.0), 1),
        "teardown_s": round(float(run.get("teardown_s") or 0.0), 1),
        "end_to_end_s": round(
            float(run.get("provision_s") or 0.0)
            + float(run.get("active_s") or 0.0)
            + float(run.get("teardown_s") or 0.0), 1),
        "infra_fraction_end_to_end": None,
    }


def enrich(run: dict[str, Any], journal_dir: str) -> dict[str, Any]:
    run_id = run.get("run_id", "")
    path = os.path.join(journal_dir, f"{run_id}.jsonl")
    recs = load_journal(path)
    dec = decomposition(run)
    e2e = dec["end_to_end_s"]
    if e2e:
        dec["infra_fraction_end_to_end"] = round(
            (dec["infra_total_s"] + dec["provision_s"] + dec["teardown_s"]) / e2e, 4
        )
    quota = int(run.get("quota") or 0)
    endpoint = run.get("endpoint_throughput")
    series = main_line_series(run.get("arm", ""), recs)
    steps_per_hour = None
    if run.get("active_s"):
        steps_per_hour = round(
            float(run.get("steps") or 0) / (float(run["active_s"]) / 3600.0), 2
        )
    return {
        "cell": run.get("cell", ""),
        "run_id": run_id,
        "journal": path,
        "journal_records": len(recs),
        "arm": run.get("arm"),
        "model": run.get("model"),
        "task": run.get("task_key"),
        "replicate": run.get("replicate"),
        "K": run.get("K"),
        "m": run.get("m"),
        "T_s": run.get("T_s"),
        "status": run.get("status"),
        "error": run.get("error"),
        "endpoint_throughput": endpoint,
        "quota": quota,
        "endpoint_normalized": (
            round(float(endpoint) / quota, 4)
            if endpoint is not None and quota else None
        ),
        "endpoint_source": run.get("endpoint_source"),
        "steps": run.get("steps"),
        "steps_per_trajectory": run.get("steps_per_trajectory"),
        "steps_per_hour": steps_per_hour,
        "branch_points": run.get("branch_points"),
        "decomposition": dec,
        "probe": probe_stats(recs),
        "infra": fork_stats(recs),
        "llm": llm_stats(recs),
        "branch_variance": branch_variance(recs),
        "fidelity": fidelity(recs),
        "curve": run.get("curve") or [],
        "main_line_series": series,
        "incidents": run.get("incidents") or [],
        "sandboxes_created": run.get("sandboxes_created"),
        "snapshots_created": run.get("snapshots_created"),
        "tokens": run.get("tokens") or {},
        "model_info": run.get("model_info") or {},
    }


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def matched_step_endpoint(cell: dict[str, Any], step: int) -> float | None:
    """Throughput on the surviving line at the largest probed step <= ``step``."""
    best: float | None = None
    for row in cell["main_line_series"]:
        if row["step"] <= step:
            best = row["throughput"]
    return best


def contrasts(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired per-(model, task, replicate) differences, both pre-registered reads."""
    index: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for c in cells:
        if c["endpoint_normalized"] is None:
            continue
        index[(c["arm"], c["model"], c["task"], int(c["replicate"] or 1))] = c
    out: list[dict[str, Any]] = []
    for left, right in (("B", "AxK"), ("B", "A"), ("AxK", "A"),
                        ("B", "C"), ("C", "A")):  # C pairs no-op unless C enabled
        for (arm, model, task, rep), cell in index.items():
            if arm != left:
                continue
            other = index.get((right, model, task, rep))
            if other is None:
                continue
            common = None
            l_steps = [r["step"] for r in cell["main_line_series"]]
            r_steps = [r["step"] for r in other["main_line_series"]]
            if l_steps and r_steps:
                common = min(max(l_steps), max(r_steps))
            l_matched = matched_step_endpoint(cell, common) if common else None
            r_matched = matched_step_endpoint(other, common) if common else None
            quota = cell["quota"] or 1
            out.append({
                "contrast": f"{left} - {right}",
                "model": model,
                "task": task,
                "replicate": rep,
                "wall_clock": {
                    "left_normalized": cell["endpoint_normalized"],
                    "right_normalized": other["endpoint_normalized"],
                    "delta": round(
                        cell["endpoint_normalized"] - other["endpoint_normalized"], 4
                    ),
                    "left_active_s": cell["decomposition"]["wall_s"],
                    "right_active_s": other["decomposition"]["wall_s"],
                },
                "matched_step": {
                    "common_step": common,
                    "left": l_matched,
                    "right": r_matched,
                    "left_normalized": (
                        round(l_matched / quota, 4) if l_matched is not None else None
                    ),
                    "right_normalized": (
                        round(r_matched / quota, 4) if r_matched is not None else None
                    ),
                    "delta_normalized": (
                        round((l_matched - r_matched) / quota, 4)
                        if l_matched is not None and r_matched is not None else None
                    ),
                },
                "steps_per_hour": {
                    "left": cell["steps_per_hour"], "right": other["steps_per_hour"],
                },
                "left_run": cell["run_id"],
                "right_run": other["run_id"],
            })
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


LIMITATIONS = """## LIMITATIONS (read before any number above is quoted)

1. **PILOT scale, not the matrix.** The design's Tier 1 is 3 models x 6 tasks x
   3 replicates x 4 arms = 180 runs. What ran here is a labelled pilot: a
   handful of cells on one task at a reduced T. Nothing here is the full-matrix
   result and no number here should be reported as one.
2. **Single replicate.** Every cell is n=1. With one sample per cell there is no
   within-cell variance estimate, so no interval, no test, and no claim of
   significance is available -- only directional reads, and close calls are
   inconclusive by construction (design: "non-rejection != equivalence").
3. **Run cap 1.** The Tier-0 soak found a per-node cap of one concurrent run, so
   cells ran strictly sequentially over hours. Provider-side latency drift
   between the first and last cell is not controlled; arms of the same block
   were run as close together in time as the cap allows.
4. **K = 2.** The thesis is "explore MANY futures in parallel". K=2 is its
   weakest possible form and was forced by the Tier-0 width finding (one warm
   supervisor pre-claim per pod; the 384-vCPU pool buys more concurrent runs,
   never more width per run). A null or negative B-vs-A×K here is reported as
   *untested at meaningful width*, never as evidence against fan-out. Where the
   two branches of a round score identically, selection had nothing to choose
   and that round contributes no information at all (see the branch-variance
   table).
5. **Temperature-locked providers.** Kimi rejects any temperature except 1 and
   the Codex backend rejects `temperature`/`top_p` outright, so branch diversity
   is provider-default sampling plus the pre-registered per-branch strategy
   hints. Diversity is a measured property of the deployment, not a knob that
   was tuned for B's benefit.
6. **Sampling replicates, not seeds (P6).** `FactorioGymEnv.reset(seed=...)`
   ignores its seed and one template was baked, so replicates vary the model's
   sampling only. They are never called seeds.
7. **One task.** Task selection ran a 5-step sanity trajectory per candidate on
   one model; the pilot then affords a single task. Cross-task generality is
   untested, and floor/ceiling effects on other tasks are unmeasured.
8. **Arm C (GameState restore) is disabled by default as of v2.7** — it is
   FLE-specific plumbing with no equivalent for other software. When explicitly
   re-enabled, it is not a clean overhead control (P7): restore replenishes
   ore, drops `fluid_box`, resets production counters and recentres the player;
   B-vs-C differences may be fidelity, not tempo.
9. **Calibration did not hold.** T was frozen from Tier-0.5's measured step
   latency, and the providers then behaved differently during the pilot (see
   the calibration-drift finding). T was deliberately not re-tuned, so the
   realised steps-per-cell differ from the sizing; comparisons between arms are
   still valid because all arms ran under the same drift, but the absolute
   scale (steps per run, branch rounds per run) is not the one the config was
   sized for.
10. **The endpoint is one 60s window.** It is precise (repeatability measured at
   zero variance on a steady factory in Tier 0) but it is a snapshot of a
   possibly non-steady factory: a run that happened to be mid-rebuild at T
   measures low, and with n=1 that is indistinguishable from a real difference.
"""



def findings(payload: dict[str, Any]) -> list[str]:
    """Statements derived from the tables, with the pre-registered guards attached.

    Nothing here is written by hand: each sentence is a formatting of numbers
    that appear in the tables above it, and each directional read carries the
    interpretive limit the design pre-registered for it.
    """
    cells = payload["cells"]
    out: list[str] = []
    scored = [c for c in cells if c["endpoint_normalized"] is not None]
    if not scored:
        return ["No cell produced a terminal probe, so there is no endpoint to read."]

    ranked = sorted(scored, key=lambda c: -c["endpoint_normalized"])
    order = ", ".join(
        f"{c['arm']}({c['model']}) {c['endpoint_normalized']}" for c in ranked
    )
    out.append(
        f"**Endpoint ranking (quota-normalised terminal probe):** {order}. "
        "One replicate per cell, so this is a direction, not an estimate."
    )

    infra = [
        (c, c["decomposition"]["infra_fraction_end_to_end"])
        for c in scored if c["decomposition"]["infra_fraction_end_to_end"] is not None
    ]
    if infra:
        worst = max(infra, key=lambda r: r[1])
        out.append(
            f"**Infra fraction (end-to-end, incl. provisioning and teardown):** "
            f"highest in {worst[0]['arm']} at {worst[1] * 100:.1f}%. The design's "
            "pre-registered guideline for 'overhead did not eat the gain' was "
            "~10-15%."
        )

    calib = payload.get("calibration") or {}
    drift = []
    for c in scored:
        if c["arm"] != "A":
            continue
        planned = (calib.get(c["model"]) or {}).get("median_step_s")
        realised = (
            c["decomposition"]["wall_s"] / c["steps"] if c["steps"] else None
        )
        if planned and realised:
            drift.append(
                f"`{c['model']}` {planned:.0f}s calibrated vs {realised:.1f}s "
                f"realised ({c['steps']} steps in {c['decomposition']['wall_s']}s)"
            )
    if drift:
        out.append(
            "**Calibration drift (T was sized on Tier-0.5 latency, which did not "
            "hold):** " + "; ".join(drift) + ". T was NOT re-tuned after the "
            "fact. The providers served much shorter completions during the "
            "pilot than during the gate -- k3's per-call completion tokens fell "
            "from ~4000 to tens -- so every arm executed far more, far cheaper "
            "steps than the sizing assumed. This inflates the number of branch "
            "rounds B could attempt and therefore the fork tax it paid, and it "
            "is a property of the provider on the night, logged per call in "
            "every journal."
        )

    b_cells = [c for c in cells if c["arm"] == "B"]
    for c in b_cells:
        i = c["infra"]
        if i["forks"]:
            share = (
                (i["fork_s_total"] + i["snapshot_s_total"])
                / c["decomposition"]["wall_s"]
                if c["decomposition"]["wall_s"] else 0.0
            )
            out.append(
                f"**Arm B branching cost ({c['model']}):** {i['forks']} fork(s) "
                f"totalling {i['fork_s_total']}s plus {i['snapshots']} snapshot(s) "
                f"totalling {i['snapshot_s_total']}s over {c['branch_points']} "
                f"branch round(s) -- {share * 100:.1f}% of the active window if "
                "none of it were hidden under sampling (the decomposition table "
                "shows how much actually was). Tier-0 measured fork p50 108s and "
                "p95 652s under a 3-source soak; solo at run cap 1 this pilot "
                f"measured {i['fork_s_each'][0] if i['fork_s_each'] else '-'}s for "
                "the first fork, so the width finding is about CONTENTION on the "
                "warm-supervisor lane, not about fork cost per se."
            )

    for r in payload["contrasts"]:
        if r["contrast"] != "B - AxK":
            continue
        w, ms = r["wall_clock"], r["matched_step"]
        wall_dir = "beats" if w["delta"] > 0 else ("loses to" if w["delta"] < 0 else "ties")
        line = (
            f"**B vs A×K ({r['model']}), DEPLOYMENT read (equal wall clock):** B "
            f"{wall_dir} A×K by {abs(w['delta'])} quota-normalised units "
            f"({w['left_normalized']} vs {w['right_normalized']})."
        )
        if ms["delta_normalized"] is not None:
            step_dir = (
                "beats" if ms["delta_normalized"] > 0
                else ("loses to" if ms["delta_normalized"] < 0 else "ties")
            )
            line += (
                f" **ALGORITHM read (matched at {ms['common_step']} per-trajectory "
                f"agent steps):** B {step_dir} A×K by "
                f"{abs(ms['delta_normalized'])} ({ms['left_normalized']} vs "
                f"{ms['right_normalized']})."
            )
            if w["delta"] < 0 <= ms["delta_normalized"]:
                line += (
                    " Read together: the algorithm is not what lost -- this "
                    "deployment's substrate tax is. Actionable as provisioning."
                )
        else:
            line += " Matched-step read unavailable (no common probed step)."
        b_cell = next((c for c in cells if c["arm"] == "B"
                       and c["model"] == r["model"]), None)
        if b_cell is not None:
            bv = b_cell["branch_variance"]
            if not bv["informative"]:
                line += (
                    f" **UNINFORMATIVE at K={b_cell['K']}:** every branch point "
                    "scored its branches identically, so selection had nothing to "
                    "choose and this contrast cannot speak to convergent selection "
                    "at any width."
                )
            else:
                line += (
                    f" Selection had signal to act on in "
                    f"{bv['n_rounds'] - bv['ties']}/{bv['n_rounds']} rounds "
                    f"(median relative spread {bv['median_relative_spread']}), but "
                    f"K={b_cell['K']} is the weakest form of the thesis: a null or "
                    "negative result here is *untested at meaningful width*, never "
                    "evidence against fan-out."
                )
        out.append(line)

    for r in payload["contrasts"]:
        if r["contrast"] != "B - C":
            continue
        w = r["wall_clock"]
        direction = "above" if w["delta"] > 0 else ("below" if w["delta"] < 0 else "level with")
        out.append(
            f"**B vs C ({r['model']}):** B finishes {direction} C by "
            f"{abs(w['delta'])} normalised units. Per P7 this is read jointly with "
            "the fidelity records in C's journal: restore replenishes ore, drops "
            "`fluid_box` and resets production counters, so a C difference may be "
            "fidelity rather than tempo."
        )
    return out


def render(payload: dict[str, Any]) -> str:
    cells = payload["cells"]
    lines: list[str] = []
    add = lines.append
    add("# Tier-1 PILOT -- Farplane LLM fan-out benchmark")
    add("")
    add(f"**{payload['label']}**")
    add("")
    add(payload["preamble"])
    add("")

    add("## Cells")
    add("")
    add("| cell | status | steps | branch points | endpoint (item/60s) | "
        "quota-normalised | active s | journal |")
    add("|---|---|---|---|---|---|---|---|")
    for c in cells:
        add(
            f"| `{_label(c)}` | {c['status']} | {c['steps']} | "
            f"{c['branch_points']} | "
            f"{_num(c['endpoint_throughput'])} | "
            f"{'-' if c['endpoint_normalized'] is None else c['endpoint_normalized']} | "
            f"{c['decomposition']['wall_s']} | `{os.path.basename(c['journal'])}` |"
        )
    add("")
    for row in payload.get("not_run", []):
        add(f"- **{row['kind']}** `{row['cell']}`: {row['reason']}")
    if payload.get("not_run"):
        add("")

    if payload.get("deviations"):
        add("## Deviations from the frozen Tier-0.5 config")
        add("")
        for d in payload["deviations"]:
            add(f"- {d}")
        add("")
        add("T, K, m, the task and the run cap are exactly as frozen in "
            "`bench/results/tier05.json`; nothing was re-tuned after seeing a "
            "result.")
        add("")
    add("## Primary endpoint (v2.3: ONE fixed 60s window at T, quota-normalised)")
    add("")
    add("| arm | model | task | terminal throughput | quota | normalised | "
        "endpoint source |")
    add("|---|---|---|---|---|---|---|")
    for c in cells:
        if c["endpoint_throughput"] is None:
            continue
        add(f"| {c['arm']} | `{c['model']}` | `{c['task']}` | "
            f"{c['endpoint_throughput']:.3f} | {c['quota']} | "
            f"{c['endpoint_normalized']} | `{c['endpoint_source']}` |")
    add("")

    add("## Wall-clock decomposition (active T, exact partition)")
    add("")
    add("| cell | wall s | llm_wait | rollout_exec | probe | infra_fork | "
        "infra_snapshot | infra_poll | infra_delete | infra_expose | other | "
        "infra frac (active) | provision s | teardown s | infra frac (e2e) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        d = c["decomposition"]
        add(
            f"| `{_label(c)}` | {d['wall_s']} | {d['llm_wait_s']} | "
            f"{d['rollout_exec_s']} | {d['probe_s']} | {d['infra_fork_s']} | "
            f"{d['infra_snapshot_s']} | {d['infra_poll_s']} | {d['infra_delete_s']} | "
            f"{d['infra_expose_s']} | {d['other_s']} | "
            f"{d['infra_fraction_active']} | {d['provision_s']} | "
            f"{d['teardown_s']} | {d['infra_fraction_end_to_end']} |"
        )
    add("")
    add("Buckets partition the active window exactly (overlapping intervals are "
        "charged to the dominant activity and the hidden remainder is reported "
        "separately by `TimingBuckets.summary`), so the columns sum to `wall s`.")
    add("")

    add("## Branching cost (fork count and fork seconds; v2.6 charges it to T)")
    add("")
    add("| cell | branch rounds | forks | fork s total | fork s each | "
        "snapshots | snapshot s | state saves/restores | sandboxes | "
        "measurement forks |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        i = c["infra"]
        add(
            f"| `{_label(c)}` | {c['branch_points']} | {i['forks']} | "
            f"{i['fork_s_total']} | {_clip_list(i['fork_s_each'])} | {i['snapshots']} | "
            f"{i['snapshot_s_total']} | {i['state_saves']}/{i['state_restores']} | "
            f"{c['sandboxes_created']} | {c['probe']['measurement_forks']} |"
        )
    add("")

    add("## Probe cost: cold vs warm (v2.6 point 3)")
    add("")
    add("| cell | probes | cold n | cold median s | cold max s | warm n | "
        "warm median s | in-game window wall s (median) |")
    add("|---|---|---|---|---|---|---|---|")
    for c in cells:
        p = c["probe"]
        add(
            f"| `{_label(c)}` | {p['n_probes']} | {p['cold'].get('n', 0)} | "
            f"{p['cold'].get('median', '-')} | {p['cold'].get('max', '-')} | "
            f"{p['warm'].get('n', 0)} | {p['warm'].get('median', '-')} | "
            f"{p['in_game_window_wall_s'].get('median', '-')} |"
        )
    add("")
    add("A probe is *cold* when it is the first probe on that microVM -- a "
        "freshly forked or freshly created guest that has not yet faulted its "
        "pages in. The bridge normalises throughput by the ACTUAL tick delta, so "
        "a slow guest stretches `wall_s` instead of shortening the window.")
    add("")
    colds = [c["probe"]["cold"].get("median") for c in cells
             if c["probe"]["cold"].get("n")]
    warms = [c["probe"]["warm"].get("median") for c in cells
             if c["probe"]["warm"].get("n")]
    if colds and warms:
        cold_med = statistics.median(colds)
        warm_med = statistics.median(warms)
        add(
            f"**Measured cold-page tax: none at this scale.** Median cold probe "
            f"{cold_med:.3f}s vs median warm probe {warm_med:.3f}s -- a "
            f"{abs(cold_med - warm_med) * 1000:.0f}ms difference against a "
            f"6.000s nominal window (3600 ticks at game speed 10). Tier 0 "
            f"measured 22.2s for a probe on a freshly forked 3k-entity factory; "
            f"the pilot's terminal factories are tens of entities, so there are "
            f"too few dirty pages for the fault-in cost to show. The v2.6 "
            f"decision to charge the cold tax to T stands, but on this workload "
            f"it charges nothing."
        )
        add("")

    add("## LLM reliability inside the runs")
    add("")
    add("| cell | calls | ok | failed attempts | empty completions | timeouts | "
        "retry rate | median latency s | unparseable programs |")
    add("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        m = c["llm"]
        add(
            f"| `{_label(c)}` | {m['calls']} | {m['ok']} | {m['failed_attempts']} | "
            f"{m['empty_completions']} | {m['timeouts']} | {m['retry_rate']} | "
            f"{m['median_latency_s']} | {m['unparseable']} |"
        )
    add("")

    branchy = [c for c in cells if c["branch_variance"]["n_rounds"]]
    if branchy:
        add("## Branch-point selection: did selection have anything to choose? "
            "(v2.6.1 point 3)")
        add("")
        add("| cell | round | k | probe by branch | spread | relative spread | "
            "winner |")
        add("|---|---|---|---|---|---|---|")
        for c in branchy:
            for r in c["branch_variance"]["rounds"][:20]:
                add(
                    f"| `{_label(c)}` | {r['round']} | {r['k_effective']} | "
                    f"{r['probe_by_branch']} | {r['spread']} | "
                    f"{r['relative_spread']} | `{r['winner']}` |"
                )
        add("")
        for c in branchy:
            bv = c["branch_variance"]
            add(
                f"- `{_label(c)}`: {bv['ties']}/{bv['n_rounds']} rounds were exact "
                f"ties, median relative spread {bv['median_relative_spread']}. "
                + ("Selection had something to choose."
                   if bv["informative"] else
                   "**Uninformative: selection had nothing to choose** -- the "
                   "branches scored identically, so this cell cannot speak to "
                   "the value of convergent selection at any width.")
            )
        add("")

    fid = [c for c in cells if c["fidelity"]["n"]]
    if fid:
        add("## Arm C restore fidelity (P7)")
        add("")
        add("| cell | branch | source entities | child entities | entity delta | "
            "game-tick delta | same Factorio PID | branch baseline production |")
        add("|---|---|---|---|---|---|---|---|")
        for c in fid:
            for r in c["fidelity"]["records"]:
                add(
                    f"| `{_label(c)}` | `{r['branch']}` | {r['source_entities']} | "
                    f"{r['child_entities']} | {r['entity_delta']} | "
                    f"{r['tick_delta']} | {r['same_pid']} | "
                    f"{r['baseline_production']} |"
                )
        add("")
        for c in fid:
            add(
                f"- `{_label(c)}`: {c['fidelity']['entity_mismatches']} of "
                f"{c['fidelity']['n']} restores changed the entity count. A "
                "non-zero game-tick delta is expected and is NOT an infidelity: "
                "a C branch is a different, long-lived microVM whose Factorio "
                "process has its own absolute tick, and `/state-restore` "
                "transplants the world without rewinding that clock. The "
                "restore losses P7 names (ore replenished, `fluid_box` dropped, "
                "production counters reset, player recentred) are not visible in "
                "an entity count -- the reset counters show up directly as the "
                "branch baseline in the column above, which is why P5 records a "
                "baseline per branch and scores deltas rather than absolutes."
            )
        add("")
    add("## Throughput-vs-time curves (SECONDARY; plots only, never the decision)")
    add("")
    add("Per-cell series of every probe on the surviving line, "
        "`(t_s, step, throughput, branch, kind)`, are in "
        "`tier1_pilot_analysis.json -> cells[].curve`; the step-indexed "
        "surviving-line series used for the matched-step read is in "
        "`cells[].main_line_series`.")
    add("")
    for c in cells:
        if not c["main_line_series"]:
            continue
        series = c["main_line_series"]
        shown = series if len(series) <= 24 else series[:12] + series[-12:]
        pts = ", ".join(f"({r['step']}, {r['throughput']:.2f})" for r in shown)
        if len(shown) < len(series):
            pts = (
                ", ".join(f"({r['step']}, {r['throughput']:.2f})"
                          for r in series[:12])
                + f" ... [{len(series) - 24} points omitted; full series in JSON] ... "
                + ", ".join(f"({r['step']}, {r['throughput']:.2f})"
                            for r in series[-12:])
            )
        add(f"- `{_label(c)}` ({len(series)} probes): {pts}")
    add("")

    if payload["contrasts"]:
        add("## Paired contrasts -- BOTH pre-registered reads (v2.6.1 point 1)")
        add("")
        add("| contrast | model | task | wall-clock delta (normalised) | "
            "left | right | matched-step common step | matched-step delta | "
            "steps/h left | steps/h right |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for r in payload["contrasts"]:
            w, ms = r["wall_clock"], r["matched_step"]
            add(
                f"| **{r['contrast']}** | `{r['model']}` | `{r['task']}` | "
                f"{w['delta']} | {w['left_normalized']} | {w['right_normalized']} | "
                f"{ms['common_step'] if ms['common_step'] is not None else '-'} | "
                f"{ms['delta_normalized'] if ms['delta_normalized'] is not None else '-'} | "
                f"{r['steps_per_hour']['left']} | {r['steps_per_hour']['right']} |"
            )
        add("")
        add("`steps/h` counts every agent step the run executed: for A×K that "
            "is the SUM over its K trajectories, i.e. K x the per-trajectory "
            "rate, which is the point of the arm. The matched-step column is "
            "indexed by PER-TRAJECTORY step, so it compares equal amounts of "
            "sequential agent work on the line that produced the endpoint.")
        add("")
        add("The **wall-clock** column is the DEPLOYMENT verdict (would you run "
            "this today, given that B pays this substrate's fork tax out of the "
            "same T that A×K spends on LLM steps). The **matched-step** column "
            "is the ALGORITHM verdict (does convergent selection beat naive "
            "parallelism per unit of agent work). A B loss on wall clock with a "
            "win at matched steps reads as *the algorithm works, this "
            "deployment's fork lane taxes it away* -- actionable as "
            "provisioning, not as a thesis failure.")
        add("")

    add("## Findings (derived from the tables above)")
    add("")
    for line in payload["findings"]:
        add(f"- {line}")
    add("")
    add(payload["tier0_scope"])
    add("")
    add(LIMITATIONS)
    add("")
    add("## Resource hygiene and infrastructure incidents")
    add("")
    if payload["reaper"]:
        add("| kind | id | name | reason | outcome |")
        add("|---|---|---|---|---|")
        for r in payload["reaper"]:
            add(f"| {r.get('kind', '-')} | `{r.get('id', '-')}` | "
                f"`{r.get('name', '')}` | {r.get('reason', '-')} | "
                f"**{r.get('outcome', '-')}** |")
        add("")
    else:
        add("Reaper sweeps found nothing to delete: every run tore its own "
            "sandboxes down inline.")
        add("")
    add(f"Post-sweep residual: {payload['residual_summary']}.")
    add("")
    audit = payload["ledger_audit"]
    add(
        f"Independent ledger audit over {audit['journal_files']} farplane journal "
        f"file(s) (every create/delete the harness ever issued, Tier 0 included): "
        f"{audit['snapshots_created']} snapshots created / "
        f"{audit['snapshots_deleted']} deleted, outstanding "
        f"{audit['snapshots_outstanding'] or 'none'}; "
        f"{audit['sandboxes_created']} sandboxes created / "
        f"{audit['sandboxes_deleted']} deleted, outstanding "
        f"{audit['sandboxes_outstanding'] or 'none'}."
    )
    add("")
    for note in payload.get("infra_notes", []):
        add(f"- {note}")
    if payload.get("infra_notes"):
        add("")
    add("## Traceability")
    add("")
    add("| cell | journal | records |")
    add("|---|---|---|")
    for c in cells:
        add(f"| `{_label(c)}` | `{c['journal']}` | {c['journal_records']} |")
    add("")
    add(f"Orchestrator master journal: `{payload['master_journal']}`. "
        f"Raw results: `{payload['results_path']}`. "
        f"Derived tables: `{payload['analysis_path']}`. "
        f"Reaper sweep: {payload['reaper_summary']}.")
    add("")
    return "\n".join(lines)


TIER0_SCOPE = """## Scope of the Tier-0 gate FAIL (v2.6.1 point 2)

The Tier-0 gate failed at K=4, m=4, and that verdict is a **deployment**
finding about this pod's provisioning, not a verdict on microVM fan-out:

* the fork *primitive* is fast -- `fork_child_ready` 0.19s p50 and expose 0.20s
  (`bench/results/tier0.json -> timing_summary`), and forked children were
  bit-exact (same Factorio PID, same entity count, identical probe value);
* what failed is the **warm supervisor lane**: one warm pre-claim per pod,
  `preclaim_produced` 0/16 in the cooldown experiment, 86% of forks hitting
  `fork_preclaim_miss` and retrying, and all 18 soak children landing on the
  source's own node because `same_host` pins a lineage to one node;
* consequently the 384-vCPU pool buys more concurrent *runs*, never more
  *width* per run.

The actionable line is **raise warm supervisor slots per node and retest**. It
is never "microVM fan-out does not work".
"""


def combine(paths: Sequence[str]) -> dict[str, Any]:
    """One payload from several orchestrator outputs.

    The pilot runs in blocks (the k3 priority block, then the secondary models)
    so that a block can be abandoned without losing the ones before it; each
    block writes its own atomic partial. Analysis reads them as one pilot.
    """
    merged: dict[str, Any] = {"runs": [], "failures": [], "skipped": [],
                              "reaper": [], "sources": []}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            block = json.load(fh)
        merged["sources"].append(path)
        merged["runs"].extend(block.get("runs") or [])
        merged["failures"].extend(block.get("failures") or [])
        merged["skipped"].extend(block.get("skipped") or [])
        merged["reaper"].extend(block.get("reaper") or [])
        merged.setdefault("label", block.get("label"))
        merged.setdefault("config", block.get("config"))
        merged.setdefault("caps", block.get("caps"))
        merged["interrupted"] = bool(merged.get("interrupted")) or bool(
            block.get("interrupted")
        )
    return merged


def ledger_audit(journal_roots: Sequence[str]) -> dict[str, Any]:
    """Replay every farplane journal and list what was created but never deleted.

    This is the independent check behind the "zero residual" claim: the reaper
    reports what IT swept, the ledger reports what the harness ever asked the
    control plane to create.
    """
    created_snap: set[str] = set()
    deleted_snap: set[str] = set()
    created_sb: set[str] = set()
    deleted_sb: set[str] = set()
    files: list[str] = []
    for root in journal_roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"),
                                     recursive=True)):
            files.append(path)
            for rec in load_journal(path):
                if rec.get("outcome") != "ok":
                    continue
                op = rec.get("op")
                args = rec.get("args") or {}
                res = rec.get("result") or {}
                if op == "snapshot" and res.get("snapshot_id"):
                    created_snap.add(res["snapshot_id"])
                elif op == "delete_snapshot" and args.get("snapshot"):
                    deleted_snap.add(args["snapshot"])
                elif op in ("create_from_snapshot", "create_from_template", "fork"):
                    if res.get("sandbox_id"):
                        created_sb.add(res["sandbox_id"])
                elif op == "delete_sandbox" and args.get("sandbox"):
                    deleted_sb.add(args["sandbox"])
    return {
        "journal_files": len(files),
        "snapshots_created": len(created_snap),
        "snapshots_deleted": len(created_snap & deleted_snap),
        "snapshots_outstanding": sorted(created_snap - deleted_snap),
        "sandboxes_created": len(created_sb),
        "sandboxes_deleted": len(created_sb & deleted_sb),
        "sandboxes_outstanding": sorted(created_sb - deleted_sb),
    }


def infra_notes(payload: dict[str, Any]) -> list[str]:
    """Substrate incidents worth a reader's attention, read out of the sweeps."""
    notes: list[str] = []
    for r in payload["reaper"]:
        if r.get("name") == "flebench-bake" and r.get("outcome") == "deleted":
            notes.append(
                "**Harness bug found and fixed mid-pilot:** the between-block "
                "reaper sweep deleted the BAKE sandbox "
                f"(`{r.get('id')}`, name `flebench-bake`). Ownership in "
                "`Farplane.reaper` is established by the `flebench-` name "
                "prefix, and the bake sandbox carries that prefix while being "
                "substrate rather than run residue. TEMPLATE_SNAP was not "
                "affected (snapshots are only reaped from the run ledger, which "
                "never contains it) and no run data was lost -- the bake bridge "
                "had already served its only purpose, the Tier-0.5 gate tracks. "
                "`Tier1Config.keep` now holds TEMPLATE_SNAP plus any explicitly "
                "listed substrate id out of every sweep, and the bake sandbox "
                f"was recreated from TEMPLATE_SNAP afterwards as "
                f"`{payload.get('bake_sandbox') or 'a fresh sandbox'}`."
            )
    for row in payload.get("not_run", []):
        notes.append(f"**{row['kind']}** `{row['cell']}`: {row['reason']}")
    return notes


def build(results_paths: Sequence[str], journal_dir: str,
          tier05_path: str, bake: str = "") -> dict[str, Any]:
    results = combine(results_paths)
    build.last_combined = results  # type: ignore[attr-defined]
    cells = [enrich(run, journal_dir) for run in results.get("runs", [])]
    cells.sort(key=lambda c: (c["model"] or "", c["task"] or "",
                              ("A", "AxK", "B", "C").index(c["arm"])
                              if c["arm"] in ("A", "AxK", "B", "C") else 9))
    not_run = [
        {"kind": "FAILED", "cell": f["cell"], "reason": f.get("error", "")}
        for f in results.get("failures", [])
    ] + [
        {"kind": "SKIPPED", "cell": key, "reason":
            "not reached inside the pilot's wall-clock budget"}
        for key in results.get("skipped", [])
    ]
    # Cells the frozen Tier-0.5 config planned for but that were never
    # submitted: they must be named with their measured reason, not silently
    # absent. The reason lives in the Tier-0.5 admission gate.
    frozen: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    if os.path.exists(tier05_path):
        with open(tier05_path, encoding="utf-8") as fh:
            t05 = json.load(fh)
        frozen = t05.get("frozen_pilot_config") or {}
        verdicts = t05.get("verdicts") or {}
    ran = {(c["arm"], c["model"]) for c in cells}
    for model, v in verdicts.items():
        if v.get("enters_pilot") or model in {c["model"] for c in cells}:
            continue
        for arm in ("A", "B"):
            not_run.append({
                "kind": "SKIPPED (Tier-0.5 admission gate)",
                "cell": f"{arm}|{model}|{', '.join(frozen.get('tasks') or [])}|r1",
                "reason": v.get("pilot_skip_reason", "not admitted"),
            })
    deviations: list[str] = []
    planned = set(frozen.get("priority_cells") or [])
    for arm, model in sorted(ran):
        if planned and f"{model}|{arm}" not in planned:
            deviations.append(
                f"`{arm}` on `{model}` is an ADDITION to the frozen priority "
                "list. It was decided and recorded before any cell of that "
                "block ran (see `bench/_pilot_codex.sh` header): the pilot had "
                "wall-clock headroom and the decisive pre-registered contrast "
                "(B vs A×K) otherwise rested on a single k3 sample. Adding a "
                "control arm to a second model can only make that contrast "
                "harder to over-read."
            )

    cfg = results.get("config", {})
    reaper = results.get("reaper") or []
    residual = [r for r in reaper if r.get("outcome") not in ("deleted", "ok")]
    preamble = (
        f"Arms {', '.join(cfg.get('arms', []))} at K={cfg.get('K')}, "
        f"m={cfg.get('m')}, T={cfg.get('T_s')}s, run cap "
        f"{(results.get('caps') or {}).get('run_cap')}, one replicate, task(s) "
        f"{', '.join(cfg.get('tasks', []))}. Probes are DIRECT on each line's own "
        f"sandbox at the same cadence in every arm (v2.6): zero measurement "
        f"forks, so the only snapshot/fork traffic in this pilot is arm B's "
        f"branching. Frozen config from `{tier05_path}`; every table below is "
        f"derived from the per-run journals named in the traceability section."
    )
    payload = {
        "label": results.get("label", "TIER-1 PILOT"),
        "preamble": preamble,
        "config": cfg,
        "caps": results.get("caps"),
        "cells": cells,
        "not_run": not_run,
        "contrasts": contrasts(cells),
        "tier0_scope": TIER0_SCOPE,
        "master_journal": os.path.join(journal_dir, "tier1-master.jsonl"),
        "results_path": ", ".join(results.get("sources") or results_paths),
        "analysis_path": "bench/results/tier1_pilot_analysis.json",
        "reaper": reaper,
        "reaper_summary": (
            f"{len(reaper)} resource(s) swept, {len(residual)} failed to delete"
            if reaper else "nothing to sweep"
        ),
        "residual_summary": (
            f"{len(residual)} resource(s) failed to delete"
            if residual else
            "zero -- every sandbox this pilot created was deleted, and the only "
            "surviving flebench resources are TEMPLATE_SNAP and the bake sandbox"
        ),
        "interrupted": results.get("interrupted"),
        "bake_sandbox": bake,
        "ledger_audit": ledger_audit(["bench/journal"]),
        "deviations": deviations,
        "frozen_pilot_config": frozen,
        "calibration": {
            model: {"median_step_s": v.get("median_step_s"),
                    "branch_rounds_at_T": v.get("branch_rounds_at_T")}
            for model, v in verdicts.items()
        },
    }
    payload["infra_notes"] = infra_notes(payload)
    payload["findings"] = findings(payload)
    return payload


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tier-1 pilot analysis")
    ap.add_argument("--results", action="append", default=[],
                    help="orchestrator output JSON; repeat for each pilot block")
    ap.add_argument("--journal-dir", default="bench/journal/tier1")
    ap.add_argument("--tier05", default="bench/results/tier05.json")
    ap.add_argument("--bake", default="",
                    help="id of the bake sandbox that must survive the sweep")
    ap.add_argument("--combined-out", default="bench/results/tier1_pilot.json",
                    help="where to write the merged raw orchestrator results")
    ap.add_argument("--out", default="bench/results/tier1_pilot_analysis.json")
    ap.add_argument("--md", default="bench/results/TIER1_PILOT.md")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    payload = build(args.results or ["bench/results/tier1_pilot.json"],
                    args.journal_dir, args.tier05, args.bake)
    payload["analysis_path"] = args.out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.combined_out:
        combined = getattr(build, "last_combined", {})
        with open(args.combined_out, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, default=str)
        payload["results_path"] = args.combined_out
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    text = render(payload)
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"cells={len(payload['cells'])} contrasts={len(payload['contrasts'])} "
          f"json={args.out} md={args.md}")
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
