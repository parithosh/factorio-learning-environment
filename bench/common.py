"""Shared instrumentation for the Farplane LLM fan-out benchmark.

Three concerns live here, all of them measurement plumbing that every arm and
every tier script shares:

* :class:`TimingBuckets` -- the fixed decomposition of run wall clock demanded
  by the design doc ("LLM wait vs itemized infra wait vs rollout vs scoring").
* :class:`RunJournal` -- append-only JSONL evidence: every LLM call with tokens
  and latency, every infra operation, every probe result. Re-runs append, so
  records are grouped into sessions and read back with
  :func:`load_journal_records`. A tail torn by a killed writer is quarantined
  to a ``<journal>.torn`` sidecar when the journal is reopened.
* :class:`Budget` -- wall clock T is the SOLE stopping rule (design v2.3), so
  the deadline is an object that phases interrogate rather than an ad-hoc
  ``time.time()`` comparison scattered through the arm loops.

Alongside them sit the two helpers every script has to agree on:
:func:`resource_name` (Farplane's 60-char name budget, with prefix and role
kept intact) and :func:`atomic_write_json` (every result artifact, so no
reader ever sees a half-written file).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

__all__ = [
    "BUCKETS",
    "Budget",
    "BudgetExhausted",
    "Curve",
    "Interval",
    "JournalParseError",
    "RunJournal",
    "ScoreRecord",
    "TimingBuckets",
    "atomic_write_json",
    "load_journal_records",
    "new_run_id",
    "resource_name",
]

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


#: Sidecar holding tails quarantined out of a journal; see :class:`RunJournal`.
_TORN_SUFFIX = ".torn"
#: Backwards scan granularity when locating the start of an unterminated line.
_TORN_SCAN_BLOCK = 65536


def _quarantine_torn_tail(path: Path) -> tuple[int, Path] | None:
    """Move an unterminated final line out of ``path`` into ``path + '.torn'``.

    A journal is reopened in append mode, so a torn last line -- the half
    written record of a process killed mid-flush -- would be *concatenated*
    with the next session's ``journal_open``. That spliced line is
    newline-terminated, so :func:`load_journal_records` no longer sees the one
    corruption it is allowed to drop (an unterminated tail): it sees
    permanently malformed evidence in the middle of the file and raises
    :class:`JournalParseError` on every future strict read, while the new
    session also loses the ``journal_open`` marker that delimits it. So the
    fragment is quarantined before the append handle is opened.

    The fragment is appended to the sidecar (newline-terminated, one fragment
    per line) rather than dropped: it is still evidence about how the previous
    writer died. Returns ``(bytes_quarantined, sidecar_path)``, or ``None``
    when there was nothing to repair.

    ``fcntl.flock`` serialises concurrent *repairs* of one journal (two
    scripts starting on the same path at once). It does not lock out a live
    writer: every record is a single flushed append, so an unterminated tail
    means that writer is gone. A journal we cannot even open for repair is
    left alone -- the append below will report the real problem -- but a
    failure while writing the sidecar or truncating propagates: silently
    appending onto a known-torn tail is exactly the corruption to avoid.
    """
    try:
        fh = path.open("r+b")
    except OSError:
        return None  # absent, or not ours to touch
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except OSError:  # pragma: no cover - fs without advisory locks
            pass
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return None
        fh.seek(size - 1)
        if fh.read(1) == b"\n":
            return None
        # Walk backwards to the start of the unterminated line.
        cut, tail, pos = 0, b"", size
        while pos > 0:
            step = min(_TORN_SCAN_BLOCK, pos)
            pos -= step
            fh.seek(pos)
            chunk = fh.read(step)
            nl = chunk.rfind(b"\n")
            if nl >= 0:
                cut = pos + nl + 1
                tail = chunk[nl + 1:] + tail
                break
            tail = chunk + tail
        sidecar = path.with_name(path.name + _TORN_SUFFIX)
        with sidecar.open("ab") as sink:
            sink.write(tail + b"\n")
            sink.flush()
            os.fsync(sink.fileno())
        fh.truncate(cut)
        fh.flush()
        os.fsync(fh.fileno())
        return len(tail), sidecar


class RunJournal:
    """Append-only JSONL evidence file for one run (or one tier script).

    Every record carries ``ts`` (unix), ``mono`` (monotonic, comparable with
    timing intervals), ``seq``, ``run_id`` and ``session``. Writes are flushed
    immediately: a run that dies mid-flight must still leave a readable
    journal, because graceful partial results are a requirement of the
    orchestrator.

    Consumer contract -- the file is opened in append mode, so re-running the
    same run id (or invoking a tier script twice) adds a *new session* to the
    same path rather than overwriting evidence. Each instance stamps its own
    ``session`` id on every record, restarts ``seq`` at 1, and always writes a
    ``journal_open`` event first. So ``seq`` orders records only *within* one
    session, and the file as a whole is not a single evidence stream: summing
    it blindly double-counts reruns. Readers MUST go through
    :func:`load_journal_records`, which returns one session at a time and
    defaults to the latest.

    A torn tail -- the unterminated last line of a writer that died mid-flush
    -- is quarantined at construction, *before* the append handle is opened:
    the fragment moves to ``<path>.torn`` (append mode, one newline-terminated
    fragment per line) and is truncated from the journal, under
    ``fcntl.flock`` on the journal file. Without that, this session's
    ``journal_open`` would be spliced onto the fragment, producing a single
    permanently malformed line that breaks every later strict read and hides
    the session boundary. The quarantine is journalled as an incident.
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.path.stem
        #: Tags the records this instance appends; see the class docstring.
        self.session = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._seq = 0
        # Before the append handle exists: a tail torn by a dead writer must
        # not get this session's journal_open spliced onto it (R2C5).
        torn = _quarantine_torn_tail(self.path)
        self._fh = self.path.open("a", encoding="utf-8")
        self.counts: dict[str, int] = {}
        # Unconditional: the session boundary is what keeps an appended journal
        # readable, so it must not depend on the caller passing meta.
        self.event("journal_open", **(meta or {}))
        if torn is not None:
            torn_bytes, sidecar = torn
            self.incident(
                kind="journal_torn_tail",
                detail=(f"quarantined {torn_bytes} unterminated byte(s) left by a "
                        "previous writer"),
                torn_bytes=torn_bytes,
                sidecar=str(sidecar),
            )

    # -- primitives --------------------------------------------------------
    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "session": self.session,
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


def _type_name(obj: Any) -> str:
    try:
        return type(obj).__name__
    except Exception:  # pragma: no cover - hostile metaclass
        return "object"


def _unserializable(obj: Any) -> str:
    """The one fixed shape a value degrades to when nothing else works."""
    return f"<unserializable: {_type_name(obj)}>"


#: Depth ceiling for :func:`_jsonable`. Cycles are broken by identity, but a
#: legitimately deep structure would still blow the stack, here or inside
#: ``json.dumps``; a sentinel at depth is better than a dead run.
_JSONABLE_MAX_DEPTH = 32


def _jsonable(obj: Any) -> Any:
    """``json.dumps(default=...)`` hook that is guaranteed not to raise.

    Journal writes are evidence collection: a payload the harness cannot
    represent must degrade to a placeholder, never take the run down. Every
    step here touches arbitrary user code -- ``model_dump``/``to_dict``/
    ``_asdict`` lookup *and* call, ``vars()``, ``repr()``, container iteration
    -- so each is guarded individually. And the *whole* returned structure is
    sanitized, not just its top level, because ``json.dumps`` walks what we
    hand back and would otherwise still raise on an unsupported dict key or a
    reference cycle (its own circular-reference check never sees the fresh
    containers built here). Values json already renders are passed through
    untouched, so well-behaved payloads serialize to identical bytes.
    """
    return _sanitize(obj, set(), 0)


def _sanitize(obj: Any, seen: set[int], depth: int) -> Any:
    # Atoms json renders itself; passing them through byte-for-byte is what
    # keeps well-behaved payloads unchanged.
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if depth >= _JSONABLE_MAX_DEPTH:
        return _unserializable(obj)
    oid = id(obj)
    if oid in seen:
        return f"<cycle: {_type_name(obj)}>"
    # Identity is tracked along the current path only, so a value shared by
    # two siblings is still serialized twice rather than called a cycle.
    seen.add(oid)
    try:
        if isinstance(obj, dict):
            return _sanitize_mapping(obj, seen, depth)
        if isinstance(obj, (list, tuple)):
            return _sanitize_sequence(obj, seen, depth)
        return _sanitize_object(obj, seen, depth)
    finally:
        seen.discard(oid)


def _sanitize_key(key: Any) -> Any:
    # json accepts exactly these key types (rendering them as strings) and
    # refuses everything else outright, so only the rest needs converting.
    if key is None or isinstance(key, (str, bool, int, float)):
        return key
    try:
        return str(key)
    except Exception:
        return _unserializable(key)


def _sanitize_mapping(obj: Any, seen: set[int], depth: int) -> Any:
    try:
        items = list(obj.items())
    except Exception:
        return _unserializable(obj)
    out: dict[Any, Any] = {}
    for key, value in items:
        # Two exotic keys can stringify the same; last one wins, as it would
        # in json's own key rendering.
        out[_sanitize_key(key)] = _sanitize(value, seen, depth + 1)
    return out


def _sanitize_sequence(obj: Any, seen: set[int], depth: int) -> Any:
    out: list[Any] = []
    try:
        for item in obj:
            out.append(_sanitize(item, seen, depth + 1))
    except Exception:  # a hostile __iter__ / __next__ midway
        out.append(_unserializable(obj))
    return out


def _sanitize_object(obj: Any, seen: set[int], depth: int) -> Any:
    for attr in ("model_dump", "to_dict", "_asdict"):
        try:
            fn = getattr(obj, attr, None)
            if not callable(fn):
                continue
            dumped = fn()
        except Exception:  # __getattr__ or the call itself misbehaving
            continue
        return _sanitize(dumped, seen, depth + 1)
    try:
        namespace = vars(obj)
        items = list(namespace.items())
    except Exception:  # no __dict__, or a hostile mapping proxy
        items = None
    if items is not None:
        public = {k: v for k, v in items
                  if not (isinstance(k, str) and k.startswith("_"))}
        return _sanitize_mapping(public, seen, depth)
    try:
        return repr(obj)
    except Exception:
        return _unserializable(obj)


class JournalParseError(ValueError):
    """A run journal held lines that are not readable evidence.

    Raised by :func:`load_journal_records` in strict mode. ``errors`` is the
    full list of ``(lineno, message)`` pairs (1-based line numbers), so a
    caller can report exactly which evidence is unreadable instead of quietly
    averaging over a damaged file.
    """

    def __init__(self, path: str | os.PathLike[str],
                 errors: Sequence[tuple[int, str]]) -> None:
        self.path = Path(path)
        self.errors: list[tuple[int, str]] = list(errors)
        head = "; ".join(f"line {ln}: {msg}" for ln, msg in self.errors[:3])
        more = "" if len(self.errors) <= 3 else f" (+{len(self.errors) - 3} more)"
        super().__init__(
            f"{self.path}: {len(self.errors)} unreadable journal line(s): {head}{more}"
        )


def _is_journal_open(rec: dict[str, Any]) -> bool:
    return rec.get("kind") == "event" and rec.get("name") == "journal_open"


def load_journal_records(
    path: str | os.PathLike[str],
    *,
    session: str = "latest",
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Read a run journal, one session at a time.

    A journal is an append stream that may hold several sessions (see
    :class:`RunJournal`), so every consumer has to say which one it means:

    * ``session="latest"`` (default) -- the records of the last
      ``journal_open``. Records written before any ``journal_open`` are legacy
      evidence and count as one implicit session, so a journal with no open
      record at all yields every record it has.
    * ``session="all"`` -- every record, in file order. Only for tools that
      group by ``session`` themselves.
    * anything else -- the records carrying that exact ``session`` id.

    Failing closed is the point. A missing file raises
    :class:`FileNotFoundError`; unreadable lines raise
    :class:`JournalParseError` rather than silently shrinking the evidence.
    The single tolerated corruption is an unterminated final line -- the torn
    append of a process killed mid-write -- which is dropped with a warning.
    With ``strict=False`` unreadable lines are skipped with a warning and the
    caller owns the resulting ambiguity.
    """
    if not isinstance(session, str) or not session:
        raise TypeError(f"session must be a non-empty str, got {session!r}")
    p = Path(path)
    records: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []
    with p.open("rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            # Only the final line of a file can lack its newline, so this flag
            # is exactly "torn write" and nothing else.
            terminated = raw.endswith(b"\n")
            problem: str | None = None
            rec: Any = None
            try:
                text = raw.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                problem = f"invalid utf-8 ({exc.reason})"
            else:
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except ValueError as exc:
                    problem = f"malformed json ({exc})"
                else:
                    if not isinstance(rec, dict):
                        problem = f"expected a JSON object, got {type(rec).__name__}"
            if problem is None:
                records.append(rec)
            elif not terminated:
                warnings.warn(
                    f"{p}: dropping torn final line {lineno}: {problem}",
                    RuntimeWarning, stacklevel=2,
                )
            else:
                errors.append((lineno, problem))
    if errors:
        if strict:
            raise JournalParseError(p, errors)
        warnings.warn(
            f"{p}: skipping {len(errors)} unreadable journal line(s), first at "
            f"line {errors[0][0]}: {errors[0][1]}",
            RuntimeWarning, stacklevel=2,
        )
    if session == "all":
        return records
    if session == "latest":
        opens = [i for i, rec in enumerate(records) if _is_journal_open(rec)]
        if not opens:
            return records  # legacy journal: the whole file is one session
        start = opens[-1]
        sid = records[start].get("session")
        if isinstance(sid, str) and sid:
            # Select by id, not by position: two writers appending to one path
            # interleave their lines and only the id survives that.
            return [rec for rec in records if rec.get("session") == sid]
        return records[start:]  # legacy open record, no session id to match on
    picked = [rec for rec in records if rec.get("session") == session]
    if not picked:
        if strict:
            raise ValueError(f"{p}: no records for session {session!r}")
        warnings.warn(
            f"{p}: no records for session {session!r}",
            RuntimeWarning, stacklevel=2,
        )
    return picked


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


def _fsync_dir(directory: Path) -> None:
    """Make a rename itself durable; best effort, not every fs permits it."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *,
                      indent: int | None = 2) -> Path:
    """Write ``payload`` as JSON so no reader ever sees a partial artifact.

    Result artifacts are read by other scripts, and by humans, while runs are
    still in flight; a crash halfway through ``json.dump`` would leave a file
    that parses as nothing at all. So the payload goes to a temp file in the
    *same* directory, is flushed and fsynced, and is then moved into place with
    :func:`os.replace`, which is atomic on POSIX. Journals are exempt: they are
    append streams by design (see :class:`RunJournal`).

    Values that are not JSON-native take the same :func:`_jsonable` fallback
    the journal uses, so a stray dataclass never costs a caller its artifact.
    Returns the final path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, default=_jsonable)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(p.parent)
    return p


def new_run_id(arm: str, model: str, task: str, replicate: int) -> str:
    """Stable-ish, collision-free run id used for names, journals and tags."""
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{arm}-{safe_model}-{task}-r{replicate}-{uuid.uuid4().hex[:6]}"


#: Farplane refuses longer resource names, so this is a hard budget, not taste.
_NAME_MAX = 60
#: Hex chars of digest spliced in when a name has to be shortened; 32 bits is
#: ample to keep the run ids of one benchmark matrix apart.
_NAME_HASH_CHARS = 8


def resource_name(prefix: str, run_id: str, role: str) -> str:
    """Farplane resource name, at most 60 chars, prefix and role intact.

    Both ends are load-bearing: ``prefix`` is the reaper's contract (it deletes
    by prefix) and ``role`` is the only thing telling the eight concurrent
    Hybrid seats of one run apart. Truncating the whole stem erased the role
    and made those seats collide on a single name, so instead only the
    ``run_id`` middle is cut, with an 8-char blake2b digest of the *full*
    untruncated stem spliced in where the cut happened. Names stay
    deterministic, and distinct ``(run_id, role)`` pairs keep distinct names.
    """
    stem = f"{prefix}{run_id}-{role}" if role else f"{prefix}{run_id}"
    if len(stem) <= _NAME_MAX:
        return stem.rstrip("-")
    digest = hashlib.blake2b(
        stem.encode("utf-8"), digest_size=_NAME_HASH_CHARS // 2
    ).hexdigest()
    tail = f"-{digest}-{role}" if role else f"-{digest}"
    keep = _NAME_MAX - len(prefix) - len(tail)
    if keep < 1:
        raise ValueError(
            f"resource name budget of {_NAME_MAX} chars leaves no room for a "
            f"run id: prefix {prefix!r} plus role {role!r} already need "
            f"{len(prefix) + len(tail)} chars"
        )
    return f"{prefix}{run_id[:keep].rstrip('-')}{tail}"


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
