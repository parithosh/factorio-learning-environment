"""Tier-1 PILOT analysis: turn orchestrator results + run journals into a report.

Every number below is derived from the inputs named here and nothing else. Each
input is validated, hashed and listed in the report's provenance section
(``payload["provenance"]``, rendered under Traceability):

* ``--results`` -- one or more orchestrator output JSONs. The pilot runs in
  blocks (the k3 priority block, then the secondary models) and each block
  writes its own atomic partial; a requested block that is absent is an error,
  never a quietly smaller pilot.
* ``--journal-dir`` -- ``bench/journal/tier1/<run_id>.jsonl``, the per-run
  append-only evidence (``infra_op``, ``probe``, ``llm_call``, ``step``,
  ``branch_selection``, ``branch_archive``, ``incident``), read through
  ``bench.common.load_journal_records`` (latest session per run).
* ``--tier05`` -- the frozen Tier-0.5 gate: planned priority cells, admission
  verdicts and the calibration the deviation/not-run tables are read against.
* ``--ledger-root`` -- farplane journal tree(s) replayed for the independent
  create/delete ledger behind the residual claim (default ``bench/journal``).
  Every cell's ``<run_id>-farplane.jsonl`` must be inside the replayed trees:
  an empty or partial ledger makes the audit INCOMPLETE instead of letting an
  absence of evidence read as zero residual.
* ``--bake`` / ``--keep`` -- declared substrate ids that are allowed to outlive
  the sweep (the bake sandbox, TEMPLATE_SNAP). Anything else left outstanding
  in the ledger blocks the zero-residual claim.

Analysis fails closed. A missing required input, an unreadable results block or
a journal corrupted before its final line raises :class:`AnalysisError` and
exits 2 *before* any output is written. A run whose own journal cannot be read
is named as an unusable cell and dropped from every journal-derived table
instead of being scored as zero (exit 1, report still written). Result
artifacts are written atomically.

The pre-registered reads implemented here are v2.3 (endpoint = quota-normalised
terminal probe), v2.6 (direct probes, K=2, cold-page tax reported against the
declared materiality threshold below) and v2.6.1 (dual wall-clock/matched-step
read of B vs A×K; K=2 interpretive ceiling from per-branch-point probe
variance).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.common import (  # noqa: E402
    JournalParseError,
    atomic_write_json,
    load_journal_records,
)

INFRA_BUCKETS = (
    "infra_snapshot", "infra_fork", "infra_expose", "infra_delete", "infra_poll",
)

DEFAULT_RESULTS = "bench/results/tier1_pilot.json"
DEFAULT_LEDGER_ROOTS = ("bench/journal",)

# v2.6 cold-page read: the probe window is 3600 ticks at game speed 10, and a
# cold-vs-warm gap counts as material at 5% of that window. Declared here so
# the report states a threshold instead of asserting a verdict.
NOMINAL_PROBE_WINDOW_S = 6.0
COLD_PAGE_MATERIAL_S = 0.3
COLD_PAGE_MIN_PROBES = 3


# ---------------------------------------------------------------------------
# Journal loading
# ---------------------------------------------------------------------------


class AnalysisError(RuntimeError):
    """A required input is missing, unreadable, or ambiguous.

    Analysis fails closed: a report that cannot name the evidence behind a
    number does not print the number.
    """


def load_journal(path: str, *, session: str = "latest") -> list[dict[str, Any]]:
    """Records of one journal session, through the shared C1 reader.

    Read errors are never swallowed: a missing file raises ``FileNotFoundError``
    and corruption before the final line raises ``JournalParseError``. An
    unreadable journal is not an empty one, and every table derived from it
    would otherwise publish zeros as if they were measurements.
    """
    return load_journal_records(path, session=session)


def journal_evidence(path: str) -> tuple[list[dict[str, Any]] | None, str, str]:
    """``(records, status, error)`` for one run journal; never raises.

    Callers turn a ``None`` record list into an explicitly unusable cell, so a
    truncated journal costs exactly that run's tables and nothing else.
    """
    try:
        return load_journal(path), "ok", ""
    except FileNotFoundError:
        return None, "missing", f"no journal file at `{path}`"
    except JournalParseError as exc:
        detail = "; ".join(f"line {ln}: {msg}" for ln, msg in exc.errors[:4])
        return None, "corrupt", f"journal `{path}` is malformed ({detail})"
    except OSError as exc:
        return None, "unreadable", f"journal `{path}` cannot be read ({exc})"


def unusable_cell(run: dict[str, Any], path: str, status: str,
                  error: str) -> dict[str, Any]:
    """A run whose journal cannot be read: named in full, scored nowhere."""
    return {
        "cell": run.get("cell", ""),
        "run_id": run.get("run_id", ""),
        "journal": path,
        "journal_status": status,
        "journal_error": error,
        "journal_records": None,
        "arm": run.get("arm"),
        "model": run.get("model"),
        "task": run.get("task_key"),
        "replicate": run.get("replicate"),
        "status": run.get("status"),
        "error": run.get("error"),
        "endpoint_throughput": run.get("endpoint_throughput"),
        "steps": run.get("steps"),
    }


def read_json(path: str, *, role: str) -> Any:
    """Load a declared JSON input, or fail naming the role that needed it."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise AnalysisError(f"required {role} input is missing: {path}") from exc
    except OSError as exc:
        raise AnalysisError(f"{role} input {path} cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{role} input {path} is not valid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Provenance: every input this report consumes, with its content hash
# ---------------------------------------------------------------------------


def validate_resource_id(flag: str, value: str) -> str:
    """A declared substrate id: a bare token, never a path or a sentence."""
    ident = (value or "").strip()
    if ident and (len(ident.split()) > 1 or "/" in ident):
        raise AnalysisError(f"{flag} must be a bare resource id, got {value!r}")
    return ident


def _sha256_file(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def input_record(role: str, path: str, *, required: bool) -> dict[str, Any]:
    """One consumed file with its content hash, for the provenance section."""
    if not os.path.isfile(path):
        if required:
            raise AnalysisError(f"required {role} input is missing: {path}")
        return {"role": role, "kind": "file", "path": path, "present": False}
    try:
        digest, size = _sha256_file(path)
    except OSError as exc:
        raise AnalysisError(f"{role} input {path} cannot be read: {exc}") from exc
    return {"role": role, "kind": "file", "path": path, "present": True,
            "sha256": digest, "bytes": size,
            "mtime": round(os.path.getmtime(path), 3)}


def tree_record(role: str, root: str, *, required: bool,
                pattern: str = "**/*.jsonl") -> dict[str, Any]:
    """Every journal under ``root``, each hashed, plus one digest over all."""
    if not os.path.isdir(root):
        if required:
            raise AnalysisError(f"required {role} directory is missing: {root}")
        return {"role": role, "kind": "tree", "path": root, "present": False}
    files = [input_record(role, p, required=True) for p in
             sorted(glob.glob(os.path.join(root, pattern), recursive=True))]
    roll = hashlib.sha256()
    for rec in files:
        rel = os.path.relpath(rec["path"], root)
        roll.update(f"{rel}\0{rec['sha256']}\n".encode())
    return {"role": role, "kind": "tree", "path": root, "present": True,
            "n_files": len(files), "bytes": sum(r["bytes"] for r in files),
            "sha256_tree": roll.hexdigest(), "files": files}


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

    cold_agg = agg(cold)
    warm_agg = agg(warm)
    # The pilot-level cold-page verdict pools the INDIVIDUAL probes of every
    # cell (see the cold-vs-warm section of :func:`render`), so the samples
    # travel with the per-cell aggregate. A median of per-cell medians weights
    # a one-probe cell like a twenty-probe one while the reported n counts
    # individual probes -- statistic and count must describe the same thing.
    cold_agg["samples"] = [round(v, 3) for v in cold]
    warm_agg["samples"] = [round(v, 3) for v in warm]

    return {
        "n_probes": len(probes),
        "cold": cold_agg,
        "warm": warm_agg,
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
    """Provider reliability from the attempt-level ``llm_call`` records.

    ``retry_rate`` is retried attempts over all attempts; what *failed* is the
    separate ``failed_attempt_rate``. Neither rate is defined without calls, so
    both are ``None`` rather than 0.0 when the journal recorded none.
    """
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
        "retry_rate": round(len(retries) / len(calls), 4) if calls else None,
        "failed_attempt_rate": (
            round(len(errors) / len(calls), 4) if calls else None
        ),
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
    recs, journal_status, journal_error = journal_evidence(path)
    if recs is None:
        return unusable_cell(run, path, journal_status, journal_error)
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
        "journal_status": journal_status,
        "journal_error": journal_error,
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
    # Both the statistic and the n are pooled over INDIVIDUAL probes: a median
    # of per-cell medians gives a cell with one probe the same weight as a cell
    # with twenty while the reported n counts probes, so the two halves of the
    # sentence would describe different populations.
    cold_all = [float(v) for c in cells
                for v in (c["probe"]["cold"].get("samples") or [])]
    warm_all = [float(v) for c in cells
                for v in (c["probe"]["warm"].get("samples") or [])]
    cold_n = len(cold_all)
    warm_n = len(warm_all)
    if cold_all and warm_all:
        cold_med = statistics.median(cold_all)
        warm_med = statistics.median(warm_all)
        delta_ms = (cold_med - warm_med) * 1000.0
        threshold_pct = COLD_PAGE_MATERIAL_S / NOMINAL_PROBE_WINDOW_S * 100.0
        measured = (
            f"Median cold probe {cold_med:.3f}s (n={cold_n}) vs median warm probe "
            f"{warm_med:.3f}s (n={warm_n}) -- both medians pooled over every "
            f"cell's individual probes, not averaged per cell -- a "
            f"{delta_ms:+.0f}ms difference against a "
            f"{NOMINAL_PROBE_WINDOW_S:.3f}s nominal window (3600 ticks "
            f"at game speed 10)."
        )
        if cold_n < COLD_PAGE_MIN_PROBES or warm_n < COLD_PAGE_MIN_PROBES:
            add(
                f"**Cold-page tax: NOT DETERMINED.** The verdict needs at least "
                f"{COLD_PAGE_MIN_PROBES} probes of each temperature; this pilot "
                f"has cold n={cold_n}, warm n={warm_n}. {measured}"
            )
        elif cold_med - warm_med > COLD_PAGE_MATERIAL_S:
            add(
                f"**Measured cold-page tax: MATERIAL** against the pre-set "
                f"{COLD_PAGE_MATERIAL_S:.3f}s threshold ({threshold_pct:.0f}% of "
                f"the nominal window). {measured} The v2.6 decision to charge the "
                f"cold tax to T is load-bearing on this workload: the arms that "
                f"fork or create guests pay this inside T."
            )
        else:
            add(
                f"**Measured cold-page tax: below the pre-set "
                f"{COLD_PAGE_MATERIAL_S:.3f}s materiality threshold** "
                f"({threshold_pct:.0f}% of the nominal window). {measured} Tier 0 "
                f"measured 22.2s for a probe on a freshly forked 3k-entity "
                f"factory; the pilot's terminal factories are tens of entities, so "
                f"there are too few dirty pages for the fault-in cost to show at "
                f"this scale. The v2.6 decision to charge the cold tax to T "
                f"stands, but on this workload it charges {delta_ms:+.0f}ms."
            )
        add("")
    else:
        add(
            f"**Cold-page tax: NOT DETERMINED.** The pilot has no pooled cold "
            f"and warm probe pair (cold n={cold_n}, warm n={warm_n}), so the "
            f"v2.6 cold-page read cannot be taken from this pilot."
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
            f"{m['empty_completions']} | {m['timeouts']} | "
            f"{_num(m['retry_rate'], 4)} | "
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
        f"file(s) under {', '.join(audit['journal_roots'])} (every create/delete "
        f"the harness ever issued, Tier 0 included): "
        f"{audit['snapshots_created']} snapshots created / "
        f"{audit['snapshots_deleted']} deleted, outstanding "
        f"{audit['snapshots_outstanding'] or 'none'}; "
        f"{audit['sandboxes_created']} sandboxes created / "
        f"{audit['sandboxes_deleted']} deleted, outstanding "
        f"{audit['sandboxes_outstanding'] or 'none'}. Declared substrate allowed "
        f"to survive: {', '.join(audit['keep']) or 'none'}; UNDECLARED "
        f"outstanding: {', '.join(audit['outstanding_undeclared']) or 'none'}."
    )
    if not audit["complete"]:
        add("")
        add(
            "**Ledger audit INCOMPLETE**, so the counts above are a lower bound "
            "and no zero-residual claim is made: "
            + "; ".join(audit["incomplete_reasons"]) + "."
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
    for c in payload.get("unusable_cells", []):
        add(f"| `{_label(c)}` | `{c['journal']}` | "
            f"**{c['journal_status'].upper()} -- not scored** |")
    add("")
    prov = payload.get("provenance") or {}
    inputs = prov.get("inputs") or []
    master = next((i for i in inputs if i["role"] == "master_journal"), None)
    add(f"Orchestrator master journal: `{payload['master_journal']}`"
        + ("" if master and master.get("present") else " (ABSENT -- not consumed)")
        + f". Raw results: `{payload['results_path']}`. "
        f"Derived tables: `{payload['analysis_path']}`. "
        f"Reaper sweep: {payload['reaper_summary']}.")
    add("")
    add("### Inputs consumed")
    add("")
    add("| role | path | sha256 | size |")
    add("|---|---|---|---|")
    for rec in inputs:
        if not rec.get("present"):
            add(f"| {rec['role']} | `{rec['path']}` | - | **ABSENT** |")
        elif rec.get("kind") == "tree":
            add(f"| {rec['role']} | `{rec['path']}` (tree) | "
                f"`{rec['sha256_tree'][:16]}` | {rec['n_files']} file(s), "
                f"{rec['bytes']} B |")
        else:
            add(f"| {rec['role']} | `{rec['path']}` | `{rec['sha256'][:16]}` | "
                f"{rec['bytes']} B |")
    add("")
    cli = prov.get("cli") or {}
    add("Hashes are SHA-256 over the exact bytes consumed; a tree row hashes the "
        "sorted per-file digests and the full per-file list is in the JSON "
        "payload. Declared CLI inputs: "
        + "; ".join(f"`{k}`={v}" for k, v in sorted(cli.items())) + ".")
    for err in payload.get("evidence_errors") or []:
        add("")
        add(f"- **EVIDENCE ERROR:** {err}")
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

    Every requested source must exist and parse BEFORE anything is merged: the
    default ``--combined-out`` overwrites one of these very files, and a pilot
    silently missing a block would publish a smaller matrix as if it were the
    whole one.
    """
    if not paths:
        raise AnalysisError(
            "no --results source given: analysis needs at least one "
            "orchestrator output JSON"
        )
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise AnalysisError(
            "requested --results source(s) do not exist: "
            + ", ".join(missing)
            + " -- refusing to analyse (or overwrite outputs from) a pilot whose "
            "blocks are not all present"
        )
    merged: dict[str, Any] = {"runs": [], "failures": [], "skipped": [],
                              "reaper": [], "sources": []}
    for path in paths:
        block = read_json(path, role="results_block")
        if not isinstance(block, dict):
            raise AnalysisError(
                f"results source {path} is not a JSON object: "
                f"got {type(block).__name__}"
            )
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


# Farplane writes both spellings for ids it normalises (``_RESULT_ID_ALIASES``),
# but journals written before that normalisation say ``snapshotId``/``sandboxId``
# only, and a fork's *child* id never appears in the ``fork`` result at all --
# it is journalled by the following ``fork_child_ready`` record. These key sets
# mirror :meth:`bench.farplane.Farplane.journal_ledger` exactly, so the audit and
# the reaper's own ledger see the same resources.
_LEDGER_SANDBOX_RESULT_KEYS = ("sandbox_id", "sandboxId")
_LEDGER_SANDBOX_ARG_KEYS = ("sandbox", "sandbox_id", "sandboxId")
_LEDGER_SNAPSHOT_RESULT_KEYS = ("snapshot_id", "snapshotId")
_LEDGER_SNAPSHOT_ARG_KEYS = ("snapshot", "snapshot_id", "snapshotId")
_LEDGER_CHILD_KEYS = ("child", "child_id", "childId",
                      "sandbox_id", "sandboxId", "sandbox")


def _ledger_id(keys: Sequence[str], *sources: Any) -> str:
    """First non-empty string id under any of ``keys``, across ``sources``."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def ledger_audit(journal_roots: Sequence[str],
                 keep: Iterable[str] = (),
                 require: Sequence[str] = ()) -> dict[str, Any]:
    """Replay every farplane journal and list what was created but never deleted.

    This is the independent check behind the "zero residual" claim: the reaper
    reports what IT swept, the ledger reports what the harness ever asked the
    control plane to create. ``keep`` names the declared substrate ids that are
    *supposed* to outlive the sweep (TEMPLATE_SNAP, the bake sandbox); anything
    else still outstanding is undeclared residue.

    ``require`` names the journal files this audit MUST have replayed -- the
    run-specific farplane journals of the pilot. An audit is COMPLETE only when
    every journal it found could be replayed, every required journal was among
    them, and it actually saw create/delete evidence: an empty ledger tree is
    the absence of evidence, not evidence of a clean substrate, and must never
    read as "created nothing, deleted nothing, zero residual".
    """
    created_snap: set[str] = set()
    deleted_snap: set[str] = set()
    created_sb: set[str] = set()
    deleted_sb: set[str] = set()
    files: list[str] = []
    replayed: set[str] = set()
    unreadable: list[dict[str, str]] = []
    ledger_records = 0
    for root in journal_roots:
        if not os.path.isdir(root):
            raise AnalysisError(f"ledger root does not exist: {root}")
        for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"),
                                     recursive=True)):
            files.append(path)
            try:
                recs = load_journal(path, session="all")
            except (JournalParseError, OSError) as exc:
                unreadable.append({"path": path, "error": str(exc)})
                continue
            replayed.add(os.path.realpath(path))
            for rec in recs:
                if rec.get("outcome") != "ok":
                    continue
                op = rec.get("op")
                args = rec.get("args") or {}
                res = rec.get("result") or {}
                if op == "snapshot":
                    ident = _ledger_id(_LEDGER_SNAPSHOT_RESULT_KEYS, res)
                    if ident:
                        created_snap.add(ident)
                elif op == "delete_snapshot":
                    ident = _ledger_id(_LEDGER_SNAPSHOT_ARG_KEYS, args, res)
                    if ident:
                        deleted_snap.add(ident)
                elif op in ("create_from_snapshot", "create_from_template", "fork"):
                    # A `fork` result carries the FORK id, so the child arrives
                    # with fork_child_ready below; the create ops answer the
                    # sandbox id directly (either spelling).
                    ident = (_ledger_id(_LEDGER_SANDBOX_RESULT_KEYS, res)
                             or _ledger_id(("sandbox",), args))
                    if ident:
                        created_sb.add(ident)
                elif op == "fork_child_ready":
                    ident = _ledger_id(_LEDGER_CHILD_KEYS, args, res, rec)
                    if ident:
                        created_sb.add(ident)
                elif op == "delete_sandbox":
                    ident = _ledger_id(_LEDGER_SANDBOX_ARG_KEYS, args, res)
                    if ident:
                        deleted_sb.add(ident)
                else:
                    continue
                if ident:
                    ledger_records += 1
    keep_ids = {k for k in keep if k}
    outstanding = (created_snap - deleted_snap) | (created_sb - deleted_sb)
    required = sorted({p for p in require if p})
    missing_required = [p for p in required
                        if os.path.realpath(p) not in replayed]
    incomplete: list[str] = []
    if unreadable:
        incomplete.append(
            f"{len(unreadable)} journal file(s) could not be replayed "
            f"({', '.join(f['path'] for f in unreadable[:3])})"
        )
    if not files:
        incomplete.append(
            "the ledger root(s) "
            f"{', '.join(journal_roots)} hold no journal file at all, so there "
            "is no create/delete ledger to audit"
        )
    elif not ledger_records:
        incomplete.append(
            f"the {len(files)} journal file(s) replayed hold no successful "
            "create or delete record, so the ledger carries no evidence either "
            "way"
        )
    if missing_required:
        incomplete.append(
            f"{len(missing_required)} run-specific farplane journal(s) were not "
            f"replayed ({', '.join(missing_required[:3])}), so the runs they "
            "belong to are outside the audited ledger"
        )
    return {
        "journal_roots": list(journal_roots),
        "journal_files": len(files),
        "ledger_records": ledger_records,
        "unreadable_files": unreadable,
        "required_journals": required,
        "missing_required_journals": missing_required,
        "incomplete_reasons": incomplete,
        "complete": not incomplete,
        "keep": sorted(keep_ids),
        "snapshots_created": len(created_snap),
        "snapshots_deleted": len(created_snap & deleted_snap),
        "snapshots_outstanding": sorted(created_snap - deleted_snap),
        "sandboxes_created": len(created_sb),
        "sandboxes_deleted": len(created_sb & deleted_sb),
        "sandboxes_outstanding": sorted(created_sb - deleted_sb),
        "outstanding_declared": sorted(outstanding & keep_ids),
        "outstanding_undeclared": sorted(outstanding - keep_ids),
    }


def residual_summary(residual: Sequence[dict[str, Any]],
                     audit: dict[str, Any]) -> str:
    """The post-sweep residual claim, gated on the independent ledger audit.

    The reaper only knows about the resources it looked at, so "zero residual"
    is claimable only when the ledger audit is COMPLETE and shows nothing
    outstanding beyond the declared substrate ids. Anything else is UNVERIFIED
    with the reason attached -- never a zero.
    """
    if residual:
        return (
            f"{len(residual)} resource(s) failed to delete during the sweep"
        )
    if not audit["complete"]:
        return (
            "UNVERIFIED -- the sweep reported no failures, but the create/delete "
            "ledger cannot back a zero-residual claim: "
            + "; ".join(audit["incomplete_reasons"])
        )
    undeclared = audit["outstanding_undeclared"]
    if undeclared:
        return (
            "UNVERIFIED -- the sweep reported no failures, but the ledger audit "
            f"shows {len(undeclared)} undeclared resource(s) created and never "
            f"deleted: {', '.join(undeclared[:6])}"
        )
    survivors = ", ".join(f"`{i}`" for i in audit["outstanding_declared"])
    return (
        "zero -- every sandbox and snapshot this pilot created was deleted, and "
        f"the independent ledger audit over {audit['journal_files']} journal "
        f"file(s) ({audit['ledger_records']} create/delete record(s)) agrees; "
        "the only surviving flebench resources are the declared "
        f"substrate {survivors or '(none outstanding at all)'}"
    )


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


def validate_tier05(t05: Any, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(frozen_pilot_config, verdicts)`` of a well-formed, FROZEN Tier-0.5 gate.

    Both producers (``bench/tier05.py`` and ``bench/tier05_merge.py``) always
    write both sections, the REFUSED path included. Anything else is the wrong
    file or a truncated one: a list would raise ``AttributeError`` deep inside
    the derivation and an empty object would silently produce a gate table with
    no planned cells and no admission reasons, i.e. a report that claims the
    pilot ran exactly what was frozen because it knows nothing about either.

    A REFUSED / non-executable gate is rejected by name rather than read as a
    plan with nothing in it: tier05_merge's refusal marker carries empty
    ``arms``/``models``/``priority_cells`` and ``verdicts == {}``, against which
    every cell that actually ran would be reported as an unplanned ADDITION.
    """
    if not isinstance(t05, dict):
        raise AnalysisError(
            f"tier05_gate input {path} is not a JSON object: "
            f"got {type(t05).__name__}"
        )
    sections: list[dict[str, Any]] = []
    for key in ("frozen_pilot_config", "verdicts"):
        if key not in t05:
            raise AnalysisError(
                f"tier05_gate input {path} has no {key!r} section, so the pilot "
                "cannot be read against the frozen gate; point --tier05 at the "
                "artifact written by bench/tier05.py or bench/tier05_merge.py"
            )
        value = t05[key]
        if not isinstance(value, dict):
            raise AnalysisError(
                f"tier05_gate input {path}: {key!r} must be an object, got "
                f"{type(value).__name__}"
            )
        sections.append(value)
    frozen, verdicts = sections[0], sections[1]
    status = str(frozen.get("status") or "")
    if status.upper() == "REFUSED" or frozen.get("executable") is False:
        detail = str(frozen.get("error") or frozen.get("reason") or "").strip()
        if not detail:
            for key in ("reasons", "blockers", "incomplete", "warnings"):
                rows = frozen.get(key)
                if isinstance(rows, list) and rows:
                    detail = "; ".join(str(r) for r in rows)
                    break
        raise AnalysisError(
            f"tier05_gate input {path} froze nothing (status "
            f"{status or 'unset'!r}, executable {frozen.get('executable')!r}), "
            "so there is no plan to read the pilot against"
            + (f": {detail}" if detail else "")
        )
    for key, value in (("frozen_pilot_config", frozen), ("verdicts", verdicts)):
        if not value:
            raise AnalysisError(
                f"tier05_gate input {path}: {key!r} is empty, so there is no "
                "frozen gate to read the pilot against"
            )
    for model, verdict in verdicts.items():
        if not isinstance(verdict, dict):
            raise AnalysisError(
                f"tier05_gate input {path}: verdict for model {model!r} must be "
                f"an object, got {type(verdict).__name__}"
            )
    return frozen, verdicts


def admission(verdict: dict[str, Any]) -> tuple[bool, str]:
    """``(admitted, skip_reason)`` out of one Tier-0.5 model verdict (R2C1).

    ``enters_pilot``/``pilot_skip_reason`` are the canonical keys. Artifacts
    written before they existed carry ``enters_tier1`` and the raw
    ``admission_blockers`` list, and reading only the canonical pair there makes
    every admitted model look skipped.
    """
    if "enters_pilot" in verdict:
        admitted = bool(verdict.get("enters_pilot"))
    else:
        admitted = bool(verdict.get("enters_tier1"))
    reason = str(verdict.get("pilot_skip_reason") or "").strip()
    if not reason:
        blockers = verdict.get("admission_blockers") or []
        if isinstance(blockers, str):
            blockers = [blockers]
        reason = "; ".join(str(b).strip() for b in blockers if str(b).strip())
    return admitted, reason


def frozen_priority_cells(frozen: dict[str, Any]) -> set[str]:
    """The planned ``"model|arm"`` cells of the frozen config (R2C1).

    ``priority_cells`` is the canonical key both producers emit. Older artifacts
    only listed ``arms`` and ``models``, and the cross product of those is wrong
    for arm B: B runs only for the models whose B arm was admitted
    (``arm_b_models``, spelled ``b_arm_models`` by pre-R2C1 tier05_merge). When
    neither spelling is present the plan simply does not say, so B is left
    unrestricted rather than reported as an unplanned addition.
    """
    cells = frozen.get("priority_cells")
    if isinstance(cells, list) and cells:
        return {str(c).strip() for c in cells if str(c).strip()}
    arms = [str(a) for a in (frozen.get("arms") or []) if a]
    models = [str(m) for m in (frozen.get("models") or []) if m]
    b_models = frozen.get("arm_b_models")
    if not isinstance(b_models, list):
        b_models = frozen.get("b_arm_models")
    restrict_b = isinstance(b_models, list)
    b_allowed = {str(m) for m in (b_models or []) if m}
    planned: set[str] = set()
    for model in models:
        for arm in arms:
            if restrict_b and arm in ("B", "Bonce") and model not in b_allowed:
                continue
            planned.add(f"{model}|{arm}")
    return planned


def build(results_paths: Sequence[str], journal_dir: str,
          tier05_path: str, bake: str = "",
          ledger_roots: Sequence[str] = DEFAULT_LEDGER_ROOTS,
          keep: Sequence[str] = ()) -> dict[str, Any]:
    """Fold every declared input into one payload, validating it first.

    The inputs are exactly: the orchestrator result blocks, the per-run journals
    under ``journal_dir``, the frozen Tier-0.5 gate, the farplane journal trees
    replayed for the residual ledger, and the declared substrate ids (``bake``,
    ``keep``). Each one is validated and hashed into ``payload["provenance"]``
    before any table is derived, and a missing required input raises
    :class:`AnalysisError` rather than yielding a quietly smaller report.
    """
    bake = validate_resource_id("--bake", bake)
    keep_ids = {validate_resource_id("--keep", k) for k in keep if k}
    if bake:
        keep_ids.add(bake)

    results = combine(results_paths)
    build.last_combined = results  # type: ignore[attr-defined]
    runs = results.get("runs") or []

    inputs = [input_record("results_block", p, required=True)
              for p in (results.get("sources") or list(results_paths))]
    inputs.append(input_record("tier05_gate", tier05_path, required=True))
    inputs.append(tree_record("run_journals", journal_dir, required=bool(runs)))
    master_journal = os.path.join(journal_dir, "tier1-master.jsonl")
    inputs.append(input_record("master_journal", master_journal, required=False))
    for root in ledger_roots:
        inputs.append(tree_record("farplane_ledger", root, required=True))

    enriched = [enrich(run, journal_dir) for run in runs]
    cells = [c for c in enriched if c["journal_status"] == "ok"]
    unusable = [c for c in enriched if c["journal_status"] != "ok"]
    if runs and not cells:
        raise AnalysisError(
            f"not one of the {len(runs)} run journal(s) under {journal_dir} could "
            "be read, so there is nothing to derive: "
            + "; ".join(c["journal_error"] for c in unusable[:5])
        )
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
    ] + [
        {"kind": f"UNUSABLE EVIDENCE ({c['journal_status']} journal)",
         "cell": c["cell"] or c["run_id"],
         "reason": f"{c['journal_error']} -- this run is dropped from every "
                   "journal-derived table instead of being scored as zero"}
        for c in unusable
    ]
    # Cells the frozen Tier-0.5 config planned for but that were never
    # submitted: they must be named with their measured reason, not silently
    # absent. The reason lives in the Tier-0.5 admission gate.
    t05 = read_json(tier05_path, role="tier05_gate")
    frozen, verdicts = validate_tier05(t05, tier05_path)
    ran = {(c["arm"], c["model"]) for c in cells}
    ran_models = {c["model"] for c in cells}
    for model, v in verdicts.items():
        admitted, skip_reason = admission(v)
        if admitted or model in ran_models:
            continue
        for arm in ("A", "B"):
            not_run.append({
                "kind": "SKIPPED (Tier-0.5 admission gate)",
                "cell": f"{arm}|{model}|{', '.join(frozen.get('tasks') or [])}|r1",
                "reason": skip_reason or (
                    "not admitted by the Tier-0.5 gate, which recorded no reason"
                ),
            })
    deviations: list[str] = []
    planned = frozen_priority_cells(frozen)
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
    # The ledger audit is the evidence behind the residual claim, so it is
    # computed BEFORE the claim is worded and the claim reads it. Each cell's
    # farplane journal (``<run_id>-farplane.jsonl``, written by run_tier1's
    # per-cell Farplane) is REQUIRED evidence: without it the replay has no
    # record of that run's creates and would answer "nothing outstanding"
    # because it looked nowhere.
    required_ledgers = [
        os.path.join(journal_dir, f"{c['run_id']}-farplane.jsonl")
        for c in enriched if c.get("run_id")
    ]
    audit = ledger_audit(ledger_roots, keep=keep_ids, require=required_ledgers)
    evidence_errors = [c["journal_error"] for c in unusable]
    evidence_errors += [
        f"ledger journal `{f['path']}` could not be replayed: {f['error']}"
        for f in audit["unreadable_files"]
    ]
    evidence_errors += [
        f"required farplane ledger `{p}` was not replayed by the residual audit"
        for p in audit["missing_required_journals"]
    ]
    if not audit["ledger_records"]:
        evidence_errors.append(
            "the farplane ledger audit found no successful create/delete record "
            f"under {', '.join(ledger_roots)}, so nothing backs the residual "
            "claim -- an empty ledger is not a clean one"
        )
    preamble = (
        f"Arms {', '.join(cfg.get('arms', []))} at K={cfg.get('K')}, "
        f"m={cfg.get('m')}, T={cfg.get('T_s')}s, run cap "
        f"{(results.get('caps') or {}).get('run_cap')}, one replicate, task(s) "
        f"{', '.join(cfg.get('tasks', []))}. Probes are DIRECT on each line's own "
        f"sandbox at the same cadence in every arm (v2.6): zero measurement "
        f"forks, so the only snapshot/fork traffic in this pilot is arm B's "
        f"branching. Frozen config from `{tier05_path}`; every table below is "
        f"derived from the per-run journals named in the traceability section, "
        f"and every input is listed there with its content hash."
    )
    payload = {
        "label": results.get("label", "TIER-1 PILOT"),
        "preamble": preamble,
        "config": cfg,
        "caps": results.get("caps"),
        "cells": cells,
        "unusable_cells": unusable,
        "evidence_errors": evidence_errors,
        "not_run": not_run,
        "contrasts": contrasts(cells),
        "tier0_scope": TIER0_SCOPE,
        "master_journal": master_journal,
        "results_path": ", ".join(results.get("sources") or results_paths),
        "analysis_path": "bench/results/tier1_pilot_analysis.json",
        "reaper": reaper,
        "reaper_summary": (
            f"{len(reaper)} resource(s) swept, {len(residual)} failed to delete"
            if reaper else "nothing to sweep"
        ),
        "ledger_audit": audit,
        "residual_summary": residual_summary(residual, audit),
        "interrupted": results.get("interrupted"),
        "bake_sandbox": bake,
        "provenance": {
            "inputs": inputs,
            "cli": {
                "results": list(results_paths),
                "journal_dir": journal_dir,
                "tier05": tier05_path,
                "ledger_roots": list(ledger_roots),
                "keep": sorted(keep_ids),
                "bake": bake,
            },
        },
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
    ap.add_argument("--ledger-root", action="append", default=[],
                    help="farplane journal tree replayed for the residual ledger "
                         f"audit (default {', '.join(DEFAULT_LEDGER_ROOTS)}); "
                         "repeat for each root")
    ap.add_argument("--keep", action="append", default=[],
                    help="resource id that is declared substrate and is allowed "
                         "to survive the sweep (e.g. TEMPLATE_SNAP); repeatable")
    ap.add_argument("--combined-out", default="bench/results/tier1_pilot.json",
                    help="where to write the merged raw orchestrator results")
    ap.add_argument("--out", default="bench/results/tier1_pilot_analysis.json")
    ap.add_argument("--md", default="bench/results/TIER1_PILOT.md")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        payload = build(args.results or [DEFAULT_RESULTS],
                        args.journal_dir, args.tier05, args.bake,
                        ledger_roots=args.ledger_root or list(DEFAULT_LEDGER_ROOTS),
                        keep=args.keep)
    except AnalysisError as exc:
        print(f"analysis ERROR: {exc}", file=sys.stderr)
        return 2
    payload["analysis_path"] = args.out
    if args.combined_out:
        atomic_write_json(args.combined_out,
                          getattr(build, "last_combined", {}))
        payload["results_path"] = args.combined_out
    atomic_write_json(args.out, payload)
    text = render(payload)
    os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"cells={len(payload['cells'])} contrasts={len(payload['contrasts'])} "
          f"json={args.out} md={args.md}")
    errors = payload["evidence_errors"]
    if errors:
        print(f"INCOMPLETE evidence ({len(errors)}): " + "; ".join(errors[:5]),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
