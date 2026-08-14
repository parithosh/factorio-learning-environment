"""Shared instrumentation for the Farplane LLM fan-out benchmark.

Three concerns live here, all of them measurement plumbing that every arm and
every tier script shares:

* :class:`TimingBuckets` -- the fixed decomposition of run wall clock demanded
  by the design doc ("LLM wait vs itemized infra wait vs rollout vs scoring").
* :class:`RunJournal` -- append-only JSONL evidence: every LLM call with tokens
  and latency, every infra operation, every probe result.
* :class:`Budget` -- wall clock T is the SOLE stopping rule (design v2.3), so
  the deadline is an object that phases interrogate rather than an ad-hoc
  ``time.time()`` comparison scattered through the arm loops.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

# ---------------------------------------------------------------------------
# Timing buckets
# ---------------------------------------------------------------------------

#: The only buckets that exist. Every arm loop must account its wall clock into
#: exactly these; adding a bucket is a protocol change, not an implementation
#: detail, so the list is closed and validated at runtime.
BUCKETS: tuple[str, ...] = (
    "llm_wait",
    "infra_snapshot",
    "infra_fork",
    "infra_expose",
    "infra_delete",
    "infra_poll",
    "probe",
    "rollout_exec",
    "other",
)

#: Attribution priority for *overlapping* intervals, lowest number wins.
#:
#: Fan-out deliberately hides infra latency under LLM sampling (design: "K-1
#: sequential forks overlapped with candidate sampling"). If both raw sums were
#: reported as the decomposition they would exceed wall clock and the infra
#: fraction -- the number decision rule 3 turns on -- would be inflated by
#: exactly the latency that fan-out successfully hid. So concurrent wall clock
#: is charged to the *dominant* activity and the hidden remainder is reported
#: separately (see :meth:`TimingBuckets.summary`).
_PRIORITY: dict[str, int] = {
    "llm_wait": 0,
    "rollout_exec": 1,
    "probe": 2,
    "infra_snapshot": 3,
    "infra_fork": 4,
    "infra_expose": 5,
    "infra_delete": 6,
    "infra_poll": 7,
    "other": 8,
}


@dataclass(frozen=True, slots=True)
class Interval:
    bucket: str
    t0: float
    t1: float
    label: str = ""

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


class TimingBuckets:
    """Records labelled intervals and decomposes wall clock over them.

    Two views are produced, both needed:

    ``raw``
        Sum of interval durations per bucket. Concurrency-inflated; this is
        "how much latency did this activity cost, if it had been alone".
    ``attributed``
        A partition of the measured wall clock: every instant of
        ``[t_start, t_end]`` is charged to exactly one bucket (the highest
        priority activity live at that instant, ``other`` when nothing is
        live). ``sum(attributed.values()) == wall_s`` by construction.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.intervals: list[Interval] = []
        self.t_start: float | None = None
        self.t_end: float | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, at: float | None = None) -> float:
        """Mark the beginning of the accounted window (T starts here)."""
        with self._lock:
            self.t_start = at if at is not None else time.monotonic()
            return self.t_start

    def stop(self, at: float | None = None) -> float:
        with self._lock:
            self.t_end = at if at is not None else time.monotonic()
            return self.t_end

    @property
    def wall_s(self) -> float:
        if self.t_start is None:
            return 0.0
        end = self.t_end if self.t_end is not None else time.monotonic()
        return max(0.0, end - self.t_start)

    # -- recording ---------------------------------------------------------
    @contextmanager
    def bucket(self, name: str, label: str = "") -> Iterator[None]:
        """``with timings.bucket('llm_wait'):`` -- records even on exception."""
        if name not in _PRIORITY:
            raise KeyError(f"unknown timing bucket {name!r}; allowed: {BUCKETS}")
        t0 = time.monotonic()
        if self.t_start is None:
            self.start(t0)
        try:
            yield
        finally:
            self.record(name, t0, time.monotonic(), label)

    def record(self, name: str, t0: float, t1: float, label: str = "") -> Interval:
        if name not in _PRIORITY:
            raise KeyError(f"unknown timing bucket {name!r}; allowed: {BUCKETS}")
        iv = Interval(name, t0, max(t0, t1), label)
        with self._lock:
            self.intervals.append(iv)
        return iv

    def merge(self, other: "TimingBuckets") -> None:
        """Fold a child accounting (e.g. one A×K trajectory) into this one."""
        with self._lock:
            self.intervals.extend(other.intervals)
            if other.t_start is not None:
                self.t_start = (
                    other.t_start
                    if self.t_start is None
                    else min(self.t_start, other.t_start)
                )
            if other.t_end is not None:
                self.t_end = (
                    other.t_end if self.t_end is None else max(self.t_end, other.t_end)
                )

    # -- views -------------------------------------------------------------
    def raw(self) -> dict[str, float]:
        out = {b: 0.0 for b in BUCKETS}
        with self._lock:
            for iv in self.intervals:
                out[iv.bucket] += iv.duration_s
        return out

    def attributed(self) -> dict[str, float]:
        """Partition wall clock across buckets (sums to :attr:`wall_s`)."""
        out = {b: 0.0 for b in BUCKETS}
        if self.t_start is None:
            return out
        start = self.t_start
        end = self.t_end if self.t_end is not None else time.monotonic()
        if end <= start:
            return out
        with self._lock:
            ivs = [iv for iv in self.intervals if iv.t1 > start and iv.t0 < end]

        # Sweep the clipped interval endpoints; between two consecutive
        # boundaries the set of live activities is constant.
        edges = {start, end}
        for iv in ivs:
            edges.add(max(start, iv.t0))
            edges.add(min(end, iv.t1))
        ordered = sorted(edges)
        for left, right in zip(ordered, ordered[1:]):
            if right <= left:
                continue
            mid = (left + right) / 2.0
            winner = "other"
            best = _PRIORITY["other"]
            for iv in ivs:
                if iv.t0 <= mid < iv.t1 and _PRIORITY[iv.bucket] < best:
                    best = _PRIORITY[iv.bucket]
                    winner = iv.bucket
            out[winner] += right - left
        return out

    def outside_window(self) -> dict[str, float]:
        """Interval time recorded outside ``[t_start, t_end]``.

        Provisioning before T and teardown after T land here: real cost, but
        end-to-end cost, not part of the active-agent decomposition.
        """
        out = {b: 0.0 for b in BUCKETS}
        if self.t_start is None:
            return out
        start = self.t_start
        end = self.t_end if self.t_end is not None else time.monotonic()
        with self._lock:
            ivs = list(self.intervals)
        for iv in ivs:
            inside = max(0.0, min(iv.t1, end) - max(iv.t0, start))
            out[iv.bucket] += iv.duration_s - inside
        return out

    def summary(self) -> dict[str, Any]:
        raw = self.raw()
        attributed = self.attributed()
        outside = self.outside_window()
        wall = self.wall_s
        # Latency that ran concurrently with a higher-priority activity inside
        # the window, i.e. cost that fan-out successfully hid. Reported, never
        # silently dropped, and kept distinct from out-of-window cost.
        hidden = {
            b: round(max(0.0, raw[b] - attributed[b] - outside[b]), 6) for b in BUCKETS
        }
        infra_attributed = sum(
            v for b, v in attributed.items() if b.startswith("infra_")
        )
        return {
            "wall_s": round(wall, 6),
            "raw_s": {b: round(v, 6) for b, v in raw.items()},
            "attributed_s": {b: round(v, 6) for b, v in attributed.items()},
            "hidden_s": hidden,
            "outside_window_s": {b: round(v, 6) for b, v in outside.items()},
            "attributed_total_s": round(sum(attributed.values()), 6),
            "infra_fraction_attributed": (
                round(infra_attributed / wall, 6) if wall else 0.0
            ),
            "infra_fraction_raw": (
                round(sum(v for b, v in raw.items() if b.startswith("infra_")) / wall, 6)
                if wall
                else 0.0
            ),
            "n_intervals": len(self.intervals),
        }

    def check_sums(self, tol: float = 1e-6) -> None:
        """Assert the attribution really is a partition of wall clock."""
        total = sum(self.attributed().values())
        wall = self.wall_s
        if abs(total - wall) > max(tol, wall * 1e-9):
            raise AssertionError(
                f"timing attribution {total:.9f}s != wall {wall:.9f}s "
                f"(delta {total - wall:.3e}s)"
            )


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class RunJournal:
    """Append-only JSONL evidence file for one run (or one tier script).

    Every record carries ``ts`` (unix), ``mono`` (monotonic, comparable with
    timing intervals), ``seq`` and ``run_id``. Writes are flushed immediately:
    a run that dies mid-flight must still leave a readable journal, because
    graceful partial results are a requirement of the orchestrator.
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.path.stem
        self._lock = threading.Lock()
        self._seq = 0
        self._fh = self.path.open("a", encoding="utf-8")
        self.counts: dict[str, int] = {}
        if meta is not None:
            self.event("journal_open", **meta)

    # -- primitives --------------------------------------------------------
    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "ts": time.time(),
                "mono": time.monotonic(),
                "run_id": self.run_id,
                "kind": kind,
            }
            rec.update(fields)
            self._fh.write(json.dumps(rec, default=_jsonable) + "\n")
            self._fh.flush()
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return rec

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        return self.write("event", name=name, **fields)

    # -- typed records -----------------------------------------------------
    def llm_call(
        self,
        *,
        model: str,
        provider: str,
        attempt: int,
        latency_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        upstream: str = "",
        n_messages: int = 0,
        temperature: Any = None,
        request_id: str = "",
        branch: str = "",
        step: int | None = None,
        outcome: str = "ok",
        error: str = "",
        response_chars: int = 0,
        code_chars: int = 0,
        finish_reason: str = "",
        hint: str = "",
        request: Any = None,
        response_text: str = "",
    ) -> dict[str, Any]:
        return self.write(
            "llm_call",
            model=model,
            provider=provider,
            attempt=attempt,
            latency_s=round(latency_s, 6),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            upstream=upstream,
            total_tokens=prompt_tokens + completion_tokens,
            n_messages=n_messages,
            temperature=temperature,
            request_id=request_id,
            branch=branch,
            step=step,
            outcome=outcome,
            error=error,
            response_chars=response_chars,
            code_chars=code_chars,
            finish_reason=finish_reason,
            hint=hint,
            request=request,
            response_text=response_text,
        )

    def infra_op(
        self,
        *,
        op: str,
        bucket: str,
        duration_s: float,
        outcome: str = "ok",
        op_id: str = "",
        target: str = "",
        branch: str = "",
        error: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        return self.write(
            "infra_op",
            op=op,
            bucket=bucket,
            duration_s=round(duration_s, 6),
            outcome=outcome,
            op_id=op_id,
            target=target,
            branch=branch,
            error=error,
            **extra,
        )

    def probe_result(
        self,
        *,
        entity: str,
        throughput: float,
        wall_s: float,
        start_tick: int,
        end_tick: int,
        branch: str = "",
        step: int | None = None,
        kind: str = "parity",
        sandbox: str = "",
        fork_of: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        return self.write(
            "probe",
            entity=entity,
            throughput=throughput,
            wall_s=round(wall_s, 6),
            start_tick=start_tick,
            end_tick=end_tick,
            branch=branch,
            step=step,
            probe_kind=kind,
            sandbox=sandbox,
            fork_of=fork_of,
            **extra,
        )

    def step_result(
        self,
        *,
        step: int,
        branch: str,
        code_chars: int,
        production_score: float,
        automated_score: float,
        ticks: int,
        error: bool,
        exec_s: float,
        output_head: str = "",
    ) -> dict[str, Any]:
        return self.write(
            "step",
            step=step,
            branch=branch,
            code_chars=code_chars,
            production_score=production_score,
            automated_score=automated_score,
            ticks=ticks,
            error=error,
            exec_s=round(exec_s, 6),
            output_head=output_head,
        )

    def archive_branch(self, *, branch: str, step: int, messages: Sequence[Any],
                       score: Any, reason: str = "loser") -> dict[str, Any]:
        """P4: loser transcripts are artifacts. They live here and nowhere else."""
        return self.write(
            "branch_archive",
            branch=branch,
            step=step,
            reason=reason,
            score=score,
            n_messages=len(messages),
            messages=list(messages),
        )

    def incident(self, *, kind: str, detail: str, **extra: Any) -> dict[str, Any]:
        """Fidelity / failure incidents (design P7 fidelity log, failed children)."""
        return self.write("incident", incident_kind=kind, detail=detail, **extra)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> "RunJournal":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(obj: Any) -> Any:
    for attr in ("model_dump", "to_dict", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive
                pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


# ---------------------------------------------------------------------------
# Wall clock budget (design v2.3: wall clock is the SOLE stopping rule)
# ---------------------------------------------------------------------------


class BudgetExhausted(RuntimeError):
    """Raised by :meth:`Budget.require` when a phase cannot be afforded."""

    def __init__(self, phase: str, remaining_s: float, need_s: float) -> None:
        super().__init__(
            f"budget exhausted before {phase!r}: {remaining_s:.1f}s left, "
            f"{need_s:.1f}s needed"
        )
        self.phase = phase
        self.remaining_s = remaining_s
        self.need_s = need_s


@dataclass
class Budget:
    """Active-agent wall clock T.

    ``T`` starts at the first LLM call *after* readiness, so :meth:`start` is
    called by the arm at that exact moment; provisioning before it is measured
    separately as end-to-end time.

    ``reserve_s`` is held back for the mandatory terminal probe: reaching T
    with no time left to measure the endpoint would waste the whole run.
    """

    total_s: float
    reserve_s: float = 0.0
    t0: float | None = None

    def start(self, at: float | None = None) -> float:
        self.t0 = at if at is not None else time.monotonic()
        return self.t0

    @property
    def started(self) -> bool:
        return self.t0 is not None

    def elapsed_s(self) -> float:
        return 0.0 if self.t0 is None else time.monotonic() - self.t0

    def remaining_s(self, *, with_reserve: bool = True) -> float:
        if self.t0 is None:
            return self.total_s - (self.reserve_s if with_reserve else 0.0)
        budget = self.total_s - (self.reserve_s if with_reserve else 0.0)
        return budget - self.elapsed_s()

    def expired(self, *, with_reserve: bool = True) -> bool:
        return self.remaining_s(with_reserve=with_reserve) <= 0.0

    def can_afford(self, need_s: float, *, with_reserve: bool = True) -> bool:
        return self.remaining_s(with_reserve=with_reserve) >= need_s

    def require(self, phase: str, need_s: float = 0.0, *, with_reserve: bool = True) -> None:
        """Deadline check between phases; raises :class:`BudgetExhausted`."""
        remaining = self.remaining_s(with_reserve=with_reserve)
        if remaining < need_s or remaining <= 0.0:
            raise BudgetExhausted(phase, remaining, need_s)

    def deadline_for(self, need_s: float, *, with_reserve: bool = False) -> float:
        """Seconds an operation may take before it must be abandoned."""
        return max(0.0, min(need_s, self.remaining_s(with_reserve=with_reserve)))


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def new_run_id(arm: str, model: str, task: str, replicate: int) -> str:
    """Stable-ish, collision-free run id used for names, journals and tags."""
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{arm}-{safe_model}-{task}-r{replicate}-{uuid.uuid4().hex[:6]}"


def resource_name(prefix: str, run_id: str, role: str) -> str:
    """Farplane resource name; prefix is the reaper's contract."""
    stem = f"{prefix}{run_id}-{role}"
    return stem[:60].rstrip("-")


@dataclass
class ScoreRecord:
    """P5 baseline/endpoint bookkeeping for one branch.

    ``namespace.score()`` is cumulative and C's restore resets the counters, so
    a branch is only ever compared through *deltas* taken from a baseline
    recorded immediately after fork/restore.
    """

    baseline_production: float = 0.0
    baseline_automated: float = 0.0
    endpoint_production: float = 0.0
    endpoint_automated: float = 0.0
    probe_throughput: float | None = None

    @property
    def production_delta(self) -> float:
        return self.endpoint_production - self.baseline_production

    @property
    def automated_delta(self) -> float:
        return self.endpoint_automated - self.baseline_automated

    def rank_key(self) -> tuple[float, float]:
        """Probe throughput first; automated-production delta as tie-break.

        Labelled in the writeup as a historical-flow tie-break (P5): it values
        what already flowed, not future potential.
        """
        return (self.probe_throughput or 0.0, self.automated_delta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_production": self.baseline_production,
            "baseline_automated": self.baseline_automated,
            "endpoint_production": self.endpoint_production,
            "endpoint_automated": self.endpoint_automated,
            "production_delta": self.production_delta,
            "automated_delta": self.automated_delta,
            "probe_throughput": self.probe_throughput,
        }


@dataclass
class Curve:
    """SECONDARY endpoint: throughput-vs-time, plots only, never the decision."""

    points: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, t_s: float, step: int, throughput: float, branch: str = "",
            kind: str = "parity") -> None:
        self.points.append(
            {
                "t_s": round(t_s, 3),
                "step": step,
                "throughput": throughput,
                "branch": branch,
                "kind": kind,
            }
        )

    def best(self) -> float:
        return max((p["throughput"] for p in self.points), default=0.0)
