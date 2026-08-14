"""Journalled, deadline-bounded wrapper around the ``panda compute`` CLI.

Every mutation is asynchronous on the control plane: the CLI returns an
operation id and we poll ``panda compute operations get <id>`` until the
operation reaches a terminal state.  All calls -- mutations and reads alike --
append a JSONL record to ``bench/journal/<date>.jsonl`` and a structured entry
to :attr:`Farplane.timings`, which is what Tier 0 turns into its constants.

The wrapper is deliberately thin: it shells out, parses ``-o json``, and owns
only the state that the CLI does not keep for us (which resources *we* created,
so the reaper never touches a stranger's sandbox).

Every public call takes ONE absolute deadline: the submit, the operation poll,
the readiness wait and every CLI timeout and retry backoff underneath them clamp
to what is left of it, and exhausting it raises :class:`OperationTimeout`.
Operations that never settled -- and forks whose child was never confirmed -- are
remembered and driven to a terminal state by the reaper, so a sweep that reports
"clean" is not racing a create that lands thirty seconds later.  Argv reaches
errors and journal records with ``--env`` values masked.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "SB",
    "Farplane",
    "FarplaneError",
    "OperationFailed",
    "OperationTimeout",
    "CLIError",
    "summarize",
]

PANDA_BIN = os.environ.get("PANDA_BIN", "panda")

#: Operation states that mean "done, and it worked".
TERMINAL_OK = frozenset({"succeeded", "success", "completed", "done"})
#: Operation states that mean "done, and it did not work".
TERMINAL_BAD = frozenset(
    {"failed", "failure", "cancelled", "canceled", "expired", "timeout", "error", "aborted"}
)
#: Sandbox states that mean the microVM is up and its guest agent answers.
SB_READY = frozenset({"running"})
#: Sandbox states from which "running" is unreachable.
SB_DEAD = frozenset({"failed", "deleted", "terminated", "error", "expired"})
#: Substrings marking a child parked behind the scheduler rather than booting.
CAPACITY_MARKERS = ("capacity", "queued", "pending", "waiting")
#: Prefixes that mark an id as naming an OPERATION rather than a resource.  A
#: submission payload's ``id`` is an operation handle often enough that taking it
#: for a sandbox id yields 404s on every follow-up call.
_OP_ID_PREFIXES = ("op-", "op_", "operation-", "operation_", "oper-")
#: camelCase -> snake_case result ids, normalised into the journal so the ledger
#: replay and the ledger audit only have to know one spelling.
_RESULT_ID_ALIASES = (
    ("snapshotId", "snapshot_id"),
    ("sandboxId", "sandbox_id"),
    ("forkId", "fork_id"),
    ("imageId", "image_id"),
)
#: Keys a paginated ``list`` payload may carry its continuation token under.
_CURSOR_KEYS = (
    "nextCursor", "next_cursor", "nextPageToken", "next_page_token", "next", "cursor",
)
#: Runaway guard while following those cursors.
_MAX_LIST_PAGES = 200
#: CLI complaints that mean "this build does not know that flag".
_UNKNOWN_FLAG_MARKERS = (
    "unknown flag",
    "unknown shorthand flag",
    "flag provided but not defined",
    "unknown option",
    "unrecognized",
)

_DEFAULT_DEADLINE = 300.0


class FarplaneError(RuntimeError):
    """Base class for every failure raised by this module."""


class CLIError(FarplaneError):
    """``panda`` exited non-zero or emitted output we could not parse."""


class OperationFailed(FarplaneError):
    """An async operation reached a terminal non-success state."""


class OperationTimeout(FarplaneError):
    """An async operation did not reach a terminal state before the deadline."""


@dataclasses.dataclass
class SB:
    """A sandbox we created.  ``base_url`` is set once :meth:`Farplane.expose` runs."""

    id: str
    name: str = ""
    node: str = ""
    base_url: str | None = None

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return f"{self.id}({self.name or '-'}@{self.node or '-'})"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _state_of(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("state", "status", "phase"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value.lower()
    return ""


def _looks_capacity_bound(state: str) -> bool:
    return any(marker in state for marker in CAPACITY_MARKERS)


def _redact_arg(pair: str) -> str:
    """``KEY=VALUE`` -> ``KEY=***``; a bare token comes back unchanged."""
    key, sep, _ = pair.partition("=")
    return f"{key}{sep}***" if sep else pair


def _redact_cmd(cmd: Sequence[str]) -> str:
    """Render a ``panda`` argv for humans with every ``--env`` VALUE masked.

    Guest environments carry provider API keys and the bridge token, and this
    argv is interpolated into :class:`CLIError` messages and journal records
    alike -- both of which get pasted into issues and archived with the results.
    Only the value is masked: the KEY is what makes a failure diagnosable.
    """
    parts: list[str] = []
    mask_next = False
    for arg in cmd:
        if mask_next:
            mask_next = False
            parts.append(_redact_arg(arg))
            continue
        if arg in ("--env", "-e"):
            mask_next = True
            parts.append(arg)
            continue
        for flag in ("--env=", "-e="):
            if arg.startswith(flag):
                parts.append(flag + _redact_arg(arg[len(flag):]))
                break
        else:
            parts.append(arg)
    return " ".join(parts)


def _env_secrets(cmd: Sequence[str]) -> list[str]:
    """The VALUES of every ``--env`` pair in an argv -- what must not be echoed."""
    secrets: list[str] = []
    take_next = False
    for arg in cmd:
        pair = ""
        if take_next:
            take_next = False
            pair = arg
        elif arg in ("--env", "-e"):
            take_next = True
            continue
        elif arg.startswith("--env=") or arg.startswith("-e="):
            pair = arg.split("=", 1)[1]
        if pair:
            _, sep, value = pair.partition("=")
            if sep and value:
                secrets.append(value)
    return secrets


def _scrub(text: str, secrets: Sequence[str]) -> str:
    """Mask secret values a failing CLI echoed back at us out of its own argv."""
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def _normalized_result(result: Any) -> Any:
    """Add snake_case twins for the camelCase ids an operation result may use.

    The journal is replayed by :meth:`Farplane.journal_ledger` and by the ledger
    audit, both of which look for ``snapshot_id``/``sandbox_id``; a control plane
    that answered ``snapshotId`` once already hid a snapshot from the replay.
    """
    if not isinstance(result, dict):
        return result
    out = dict(result)
    for camel, snake in _RESULT_ID_ALIASES:
        value = out.get(camel)
        if isinstance(value, str) and value and not out.get(snake):
            out[snake] = value
    return out


def _looks_like_op_id(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(marker) for marker in _OP_ID_PREFIXES)


def _resource_ref(value: Any, keys: Sequence[str]) -> str:
    """A resource id read out of a string or a nested dict field, or ``""``.

    ``sandboxes/sandbox-abc`` and ``sandbox-abc`` name the same thing; an
    operation handle names neither, and is rejected here rather than handed on as
    a resource id.
    """
    if isinstance(value, dict):
        for key in keys:
            found = _resource_ref(value.get(key), keys)
            if found:
                return found
        return ""
    if not isinstance(value, str):
        return ""
    text = value.strip().rstrip("/").rsplit("/", 1)[-1]
    if not text or _looks_like_op_id(text):
        return ""
    return text


_SANDBOX_KEYS = ("sandbox_id", "sandboxId", "sandbox", "id")


def _sandbox_id_from(op: Any, submit: Any) -> str:
    """The sandbox a completed operation produced, or ``""``.

    The submission payload's ``id`` is frequently the OPERATION id, so the
    completed operation's own result/target wins and a submission id is taken
    only when it does not look like an operation handle.  An id that names a
    sandbox outright beats any bare handle we cannot classify.
    """
    candidates: list[str] = []

    def offer(value: Any) -> None:
        ref = _resource_ref(value, _SANDBOX_KEYS)
        if ref and ref not in candidates:
            candidates.append(ref)

    result = op.get("result") if isinstance(op, dict) else None
    if isinstance(result, dict):
        for key in ("sandbox_id", "sandboxId", "sandbox", "target_id", "targetId", "id"):
            offer(result.get(key))
    if isinstance(op, dict):
        for key in ("target_id", "targetId", "target", "sandbox_id", "sandboxId",
                    "sandbox", "resource"):
            offer(op.get(key))
    if isinstance(submit, dict):
        for key in ("sandbox_id", "sandboxId", "sandbox", "id"):
            offer(submit.get(key))
    for candidate in candidates:
        if candidate.startswith("sandbox-"):
            return candidate
    return candidates[0] if candidates else ""


def _snapshot_id_from(op: Any) -> str:
    keys = ("snapshot_id", "snapshotId", "snapshot", "image_id", "imageId")
    result = op.get("result") if isinstance(op, dict) else None
    for source in (result, op):
        if not isinstance(source, dict):
            continue
        for key in keys:
            ref = _resource_ref(source.get(key), keys + ("id",))
            if ref:
                return ref
    return ""


def _fork_id_from(op: Any, submit: Any) -> str:
    keys = ("fork_id", "forkId", "fork")
    result = op.get("result") if isinstance(op, dict) else None
    for source in (result, op, submit):
        if not isinstance(source, dict):
            continue
        for key in keys:
            ref = _resource_ref(source.get(key), keys + ("id",))
            if ref:
                return ref
    return ""


def _resolve_sandbox(label: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Journalling resolver: name the sandbox a create produced, or fail loudly."""

    def resolve(submit: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
        sandbox_id = _sandbox_id_from(op, submit)
        if sandbox_id:
            return {"sandbox_id": sandbox_id}
        result = op.get("result") if isinstance(op.get("result"), dict) else {}
        raise CLIError(
            f"{label}: no usable sandbox id (operation target "
            f"{op.get('targetId') or op.get('target')!r}, result keys {sorted(result)}, "
            f"submit id {submit.get('id')!r} -- an operation id is not a sandbox id)"
        )

    return resolve


def _looks_missing(error: BaseException) -> bool:
    text = str(error).lower()
    return "not_found" in text or "not found" in text or "404" in text


def _rejects_flag(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _UNKNOWN_FLAG_MARKERS)


def _next_cursor(payload: Any, used: str) -> str:
    """The cursor for the NEXT page, or ``""`` when the CLI says we are done."""
    if not isinstance(payload, dict):
        return ""
    sources = [payload]
    for key in ("page", "pagination", "meta"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        for key in _CURSOR_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != used:
                return value.strip()
    return ""


class Farplane:
    """Thin subprocess wrapper over ``panda compute``.

    Parameters
    ----------
    journal_path:
        JSONL file every call is appended to.  Defaults to
        ``bench/journal/<UTC date>.jsonl`` next to this module.
    prefix:
        Name prefix stamped on everything we create; also the reaper's filter.
    default_deadline:
        Seconds an operation may take before :class:`OperationTimeout`.
    """

    def __init__(
        self,
        journal_path: str | os.PathLike[str] | None = None,
        *,
        prefix: str = "flebench-",
        default_deadline: float = _DEFAULT_DEADLINE,
        datasource: str | None = None,
        cli_timeout: float = 180.0,
    ) -> None:
        if journal_path is None:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            journal_path = Path(__file__).resolve().parent / "journal" / f"{day}.jsonl"
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.default_deadline = float(default_deadline)
        self.datasource = datasource
        self.cli_timeout = float(cli_timeout)

        #: Structured record of every wrapper call, in completion order.
        self.timings: list[dict[str, Any]] = []
        #: Ledger of resources this process created, so the reaper is precise.
        self.created_sandboxes: dict[str, str] = {}
        self.created_snapshots: dict[str, str] = {}
        #: Operations we submitted but never saw settle (timeout, lost poll).  A
        #: timed-out create can still land a sandbox minutes later, so the reaper
        #: drives these to a terminal state before it decides what is residue.
        self.unresolved_ops: dict[str, dict[str, Any]] = {}
        #: Fork id (or the submitting op id, when the fork id never arrived) ->
        #: its source snapshot and child.  Deleting a source snapshot while a fork
        #: of it is still pending orphans the child.
        self.pending_forks: dict[str, dict[str, Any]] = {}

        self._lock = threading.Lock()
        self._session_owner: str | None = None

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _journal(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, separators=(",", ":"))
        with self._lock:
            self.timings.append(record)
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _expiry(self, deadline: float | None) -> float:
        """Absolute ``time.monotonic()`` stamp one public call may not run past."""
        budget = self.default_deadline if deadline is None else float(deadline)
        return time.monotonic() + max(0.0, budget)

    @staticmethod
    def _left(expiry: float | None) -> float:
        """Seconds of budget left; ``inf`` when the caller set no expiry."""
        return float("inf") if expiry is None else expiry - time.monotonic()

    def _backoff(
        self,
        delay: float,
        expiry: float | None,
        cmd: Sequence[str],
        last_error: Exception,
    ) -> None:
        """Sleep before a CLI retry, clamped to the budget.

        Running out of budget is a deadline miss, not a CLI bug, so it surfaces as
        :class:`OperationTimeout` -- that is what callers key their orphan
        handling off.
        """
        remaining = self._left(expiry)
        if remaining > 0.0:
            time.sleep(min(delay, remaining))
            remaining = self._left(expiry)
        if remaining <= 0.0:
            raise OperationTimeout(
                f"deadline exhausted retrying {_redact_cmd(cmd)}: {last_error}"
            ) from last_error

    def _cli(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        json_out: bool = True,
        retries: int = 0,
        expiry: float | None = None,
    ) -> Any:
        """Run ``panda compute <args>`` and return parsed JSON (or raw text).

        ``expiry`` is the absolute stamp of the caller's budget: the subprocess
        timeout and every retry backoff clamp to what is left of it, so no CLI
        call can outlive the phase that owns it by one timeout per attempt.
        """
        # Flags must land before the `--` argv separator, otherwise `panda`
        # hands them to the guest command instead of parsing them itself.
        head = list(args)
        tail: list[str] = []
        if "--" in head:
            split = head.index("--")
            head, tail = head[:split], head[split:]
        cmd: list[str] = [PANDA_BIN, "compute"]
        if self.datasource:
            cmd += ["--datasource", self.datasource]
        cmd += head
        if json_out:
            cmd += ["-o", "json"]
        cmd += tail

        want = self.cli_timeout if timeout is None else float(timeout)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            remaining = self._left(expiry)
            if remaining <= 0.0:
                raise OperationTimeout(
                    f"deadline exhausted before {_redact_cmd(cmd)}"
                    + (f": {last_error}" if last_error else "")
                )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=min(want, remaining),
                )
            except subprocess.TimeoutExpired as exc:
                last_error = CLIError(f"panda timed out: {_redact_cmd(cmd)}")
                if self._left(expiry) <= 0.0:
                    # We killed it, not the CLI: the budget ran out mid-call.
                    raise OperationTimeout(
                        f"deadline exhausted running {_redact_cmd(cmd)}"
                    ) from exc
                if attempt < retries:
                    self._backoff(1.0 + attempt, expiry, cmd, last_error)
                    continue
                raise last_error from exc
            if proc.returncode != 0:
                secrets = _env_secrets(cmd)
                last_error = CLIError(
                    f"panda exited {proc.returncode}: {_redact_cmd(cmd)}\n"
                    f"stdout: {_scrub(proc.stdout.strip(), secrets)[:2000]}\n"
                    f"stderr: {_scrub(proc.stderr.strip(), secrets)[:2000]}"
                )
                if attempt < retries:
                    self._backoff(1.0 + attempt, expiry, cmd, last_error)
                    continue
                raise last_error
            if not json_out:
                return proc.stdout
            text = proc.stdout.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = CLIError(
                    f"non-JSON output from {_redact_cmd(cmd)}: "
                    f"{_scrub(text, _env_secrets(cmd))[:2000]}"
                )
                if attempt < retries:
                    self._backoff(1.0 + attempt, expiry, cmd, last_error)
                    continue
                raise last_error from exc
        raise last_error or CLIError("unreachable")

    @staticmethod
    def _extract_op_id(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("op_id", "opId", "operationId", "operation_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        value = payload.get("id")
        if isinstance(value, str) and value.startswith("op-"):
            return value
        return None

    def _poll_operation(
        self,
        op_id: str,
        *,
        expiry: float,
        label: str = "",
    ) -> dict[str, Any]:
        """Poll ``operations get`` until terminal, inside ONE absolute budget.

        ``expiry`` is a ``time.monotonic()`` stamp, not a duration: the poll
        interval, the ``operations get`` timeout and its retry backoff all clamp
        to what is left of it, so a poll cannot walk past its caller's deadline by
        one CLI timeout per attempt.  Budget exhaustion is an
        :class:`OperationTimeout`, never a :class:`CLIError`.
        """
        started = time.monotonic()
        polls = 0
        poll_cost = 0.0
        interval = 0.4
        last_state = ""
        while True:
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                raise OperationTimeout(
                    f"{label or 'operation'} {op_id} still {last_state or 'unpolled'!r} "
                    f"after {time.monotonic() - started:.0f}s"
                )
            call_t0 = time.monotonic()
            op = self._cli(
                ["operations", "get", op_id],
                retries=3,
                timeout=min(60.0, remaining),
                expiry=expiry,
            )
            poll_cost += time.monotonic() - call_t0
            polls += 1
            state = _state_of(op)
            last_state = state or last_state
            if state in TERMINAL_OK:
                op = dict(op)
                op["_poll_count"] = polls
                op["_poll_overhead_s"] = round(poll_cost, 4)
                op["_wait_s"] = round(time.monotonic() - started, 4)
                return op
            if state in TERMINAL_BAD:
                raise OperationFailed(
                    f"{label or 'operation'} {op_id} -> {state}: {op.get('err') or op}"
                )
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                continue  # the guard at the top of the loop raises, with the state
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.4, 2.0)

    def _run_mutation(
        self,
        op_name: str,
        args: Sequence[str],
        call_args: dict[str, Any],
        *,
        expiry: float,
        resolve: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
        """Submit a mutation, poll to terminal, journal it.

        ``expiry`` is the caller's absolute deadline (see :meth:`_poll_operation`),
        shared by the submit and the poll.  ``resolve`` reads the ids the caller
        needs out of ``(submit, op)``; it runs inside the journalled block, so a
        payload we cannot read is an ``error`` record rather than an ``ok`` record
        followed by a raise, and the ids it finds land under ``result`` where the
        ledger replay looks for them.

        An operation we never saw reach a terminal state is remembered in
        :attr:`unresolved_ops` and stamped into the journal record, because the
        reaper -- possibly in another process -- must settle it before it decides
        what counts as residue.  Returns ``(submit, op, duration, resolved ids)``.
        """
        t0 = time.monotonic()
        op_id: str | None = None
        settled = False
        try:
            submit = self._cli(args, expiry=expiry)
            op_id = self._extract_op_id(submit)
            if op_id is None:
                raise CLIError(f"{op_name}: no operation id in {submit!r}")
            op = self._poll_operation(op_id, expiry=expiry, label=op_name)
            settled = True
            extra = resolve(submit if isinstance(submit, dict) else {}, op) if resolve else {}
        except Exception as exc:
            # OperationFailed IS a verdict; a timeout or a lost poll is not, and
            # the operation behind it can still land a resource on us later.
            unresolved = bool(op_id) and not settled and not isinstance(exc, OperationFailed)
            if unresolved:
                self._note_unresolved(op_id or "", op_name, call_args, exc)
            dt = time.monotonic() - t0
            self._journal(
                {
                    "ts": _utcnow(),
                    "op": op_name,
                    "args": call_args,
                    "op_id": op_id,
                    "duration_s": round(dt, 4),
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "unresolved_op": op_id if unresolved else None,
                }
            )
            raise
        dt = time.monotonic() - t0
        result = _normalized_result(op.get("result"))
        if extra:
            result = {**result, **extra} if isinstance(result, dict) else dict(extra)
        self._journal(
            {
                "ts": _utcnow(),
                "op": op_name,
                "args": call_args,
                "op_id": op_id,
                "duration_s": round(dt, 4),
                "outcome": "ok",
                "op_duration_ms": op.get("durationMs"),
                "poll_count": op["_poll_count"],
                "poll_overhead_s": op["_poll_overhead_s"],
                "op_attempts": (op.get("result") or {}).get("op_attempt"),
                "result": result,
            }
        )
        return (submit if isinstance(submit, dict) else {}), op, dt, extra

    def _record_read(
        self,
        op_name: str,
        call_args: dict[str, Any],
        dt: float,
        outcome: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": _utcnow(),
            "op": op_name,
            "args": call_args,
            "op_id": None,
            "duration_s": round(dt, 4),
            "outcome": outcome,
        }
        if extra:
            record.update(extra)
        self._journal(record)

    def _note_unresolved(
        self,
        op_id: str,
        op_name: str,
        call_args: dict[str, Any],
        error: BaseException,
    ) -> None:
        """Remember an operation whose fate we do not know.

        A fork is remembered twice over: as an operation to settle, and (keyed by
        the submitting op id, because a fork that timed out never told us its fork
        id) as a claim on its source snapshot.
        """
        if not op_id:
            return
        entry = {
            "op": op_name,
            "args": dict(call_args),
            "error": f"{type(error).__name__}: {error}",
            "ts": _utcnow(),
        }
        with self._lock:
            self.unresolved_ops[op_id] = entry
            if op_name == "fork":
                source = str(call_args.get("snapshot") or "")
                claim = self.pending_forks.setdefault(
                    op_id, {"snapshot": source, "child": "", "op_id": op_id}
                )
                if source and not claim.get("snapshot"):
                    claim["snapshot"] = source

    def _note_pending_fork(self, fork_id: str, *, snapshot: str, child: str = "") -> None:
        """A dispatched fork whose child is not yet confirmed ours-and-live."""
        with self._lock:
            claim = self.pending_forks.setdefault(
                fork_id, {"snapshot": snapshot, "child": "", "op_id": ""}
            )
            if snapshot:
                claim["snapshot"] = snapshot
            if child:
                claim["child"] = child

    def _clear_pending_fork(self, fork_id: str) -> None:
        """The child is ours, live and in the sandbox ledger: nothing to retain."""
        with self._lock:
            self.pending_forks.pop(fork_id, None)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def owner(self) -> str:
        """Authenticated handle, used to keep the reaper off other people's boxes."""
        if self._session_owner is None:
            session = self._cli(["session"], retries=2, timeout=60, expiry=self._expiry(None))
            user = (session or {}).get("user") or {}
            self._session_owner = user.get("handle") or user.get("subject") or ""
        return self._session_owner

    def get_sandbox(self, sandbox_id: str, *, expiry: float | None = None) -> dict[str, Any]:
        # A read reached from a polling loop inherits that loop's expiry; reached
        # on its own it gets one of its own, so no entry point is unbounded.
        return self._cli(
            ["sandboxes", "get", sandbox_id],
            retries=3,
            timeout=60,
            expiry=self._expiry(None) if expiry is None else expiry,
        )

    def list_sandboxes(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._list_pages("sandboxes", limit, expiry=self._expiry(None))

    def list_images(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._list_pages("images", limit, expiry=self._expiry(None))

    def _list_pages(
        self, kind: str, limit: int, *, expiry: float | None = None
    ) -> list[dict[str, Any]]:
        """Every page of ``<kind> list``, cursor by cursor.

        The reaper's precision rests on seeing the WHOLE list: a page silently
        truncated at ``--limit`` leaves residue behind while the sweep reports
        itself clean.  So the CLI's continuation cursor is followed, and a build
        that fills a page without offering a cursor -- or refuses ``--cursor``
        while offering another page -- is an error here, not a partial scan.
        """
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        used: set[str] = set()
        cursor = ""
        pages = 0
        while True:
            args = [kind, "list", "--limit", str(limit)]
            if cursor:
                args += ["--cursor", cursor]
            try:
                payload = self._cli(args, retries=3, timeout=90, expiry=expiry)
            except CLIError as exc:
                if cursor and _rejects_flag(exc):
                    raise CLIError(
                        f"{kind} list offers another page (cursor {cursor!r}) but this "
                        f"panda build rejects --cursor, so a sweep would under-scan: {exc}"
                    ) from exc
                raise
            page = list((payload or {}).get("items") or [])
            pages += 1
            for item in page:
                item_id = str(item.get("id") or "")
                if item_id:
                    if item_id in seen:
                        continue
                    seen.add(item_id)
                items.append(item)
            used.add(cursor)
            cursor = _next_cursor(payload, cursor)
            if not cursor:
                if len(page) >= limit:
                    raise CLIError(
                        f"{kind} list filled its page ({len(page)} items at --limit "
                        f"{limit}) and offered no cursor: the rest of the list is "
                        f"invisible, and a sweep from it would under-scan"
                    )
                return items
            if cursor in used:
                raise CLIError(f"{kind} list re-offered cursor {cursor!r}: pagination loops")
            if pages >= _MAX_LIST_PAGES:
                raise CLIError(
                    f"{kind} list still paginating after {pages} pages (cursor {cursor!r})"
                )

    def get_fork(self, fork_id: str, *, expiry: float | None = None) -> dict[str, Any]:
        return self._cli(
            ["forks", "get", fork_id],
            retries=3,
            timeout=60,
            expiry=self._expiry(None) if expiry is None else expiry,
        )

    def nodes(self) -> list[dict[str, Any]]:
        payload = self._cli(["nodes", "list"], retries=2, timeout=60, expiry=self._expiry(None))
        return list((payload or {}).get("items") or [])

    # ------------------------------------------------------------------
    # sandbox lifecycle
    # ------------------------------------------------------------------
    def wait_running(self, sandbox_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        """Block until the sandbox reports ``running`` (``deadline`` is seconds)."""
        return self._wait_running(sandbox_id, expiry=self._expiry(deadline))

    def _wait_running(self, sandbox_id: str, *, expiry: float) -> dict[str, Any]:
        """Readiness poll bounded by the caller's ABSOLUTE expiry.

        A fork child's sandbox record becomes queryable a beat after the fork
        operation goes terminal, so a 404 here means "not yet", not "gone".
        """
        started = time.monotonic()
        interval = 0.4
        capacity_waits = 0
        not_found = 0
        state = ""
        while True:
            if expiry - time.monotonic() <= 0.0:
                raise OperationTimeout(
                    f"sandbox {sandbox_id} still {state or 'unpolled'!r} after "
                    f"{time.monotonic() - started:.0f}s"
                )
            try:
                info = self.get_sandbox(sandbox_id, expiry=expiry)
            except CLIError as exc:
                if "not_found" not in str(exc) and "404" not in str(exc):
                    raise
                not_found += 1
                info = {}
                state = "not_found"
            else:
                state = _state_of(info)
                if state in SB_READY:
                    info = dict(info)
                    info["_capacity_waits"] = capacity_waits
                    info["_not_found_polls"] = not_found
                    info["_wait_s"] = round(time.monotonic() - started, 4)
                    return info
                if state in SB_DEAD:
                    raise OperationFailed(f"sandbox {sandbox_id} entered terminal state {state!r}")
                if _looks_capacity_bound(state):
                    capacity_waits += 1
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                continue  # the guard at the top of the loop raises, with the state
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.4, 2.0)

    def create_from_template(
        self,
        template: str,
        ttl: str,
        vcpu: int | None = None,
        mem: int | None = None,
        name: str | None = None,
        *,
        labels: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> SB:
        """Create a sandbox from a named template (bare name, no ``@version``).

        ``deadline`` is the budget for the WHOLE call: the submit, the operation
        poll and the readiness wait share ONE absolute expiry instead of taking a
        fresh one each.
        """
        expiry = self._expiry(deadline)
        name = self._stamp(name)
        args = ["sandboxes", "create", "--template", template, "--ttl", ttl, "--name", name]
        if vcpu:
            args += ["--vcpu", str(vcpu)]
        if mem:
            args += ["--memory-mb", str(mem)]
        for key, value in (labels or {}).items():
            args += ["--label", f"{key}={value}"]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        call_args = {"template": template, "ttl": ttl, "vcpu": vcpu, "mem": mem, "name": name}
        _, _, _, ids = self._run_mutation(
            "create_from_template", args, call_args, expiry=expiry,
            resolve=_resolve_sandbox("create_from_template"),
        )
        sandbox_id = ids["sandbox_id"]
        self.created_sandboxes[sandbox_id] = name
        info = self._wait_running(sandbox_id, expiry=expiry)
        return SB(id=sandbox_id, name=name, node=info.get("node") or "")

    def create_from_snapshot(
        self,
        snap_id: str,
        ttl: str,
        name: str | None = None,
        *,
        deadline: float | None = None,
    ) -> SB:
        """Reconstitute a single sandbox from a snapshot (not a fan-out).

        The CLI advertises ``--boot-flavor``/``--flavor`` but the API rejects the
        field (``unknown field "flavor"``); warm resume is the server default.
        ``deadline`` bounds the whole call, its retries and the readiness wait
        included, as ONE absolute expiry.
        """
        expiry = self._expiry(deadline)
        name = self._stamp(name)
        args = [
            "sandboxes", "create",
            "--snapshot", snap_id,
            "--ttl", ttl,
            "--name", name,
        ]
        call_args = {"snapshot": snap_id, "ttl": ttl, "name": name}
        # agent-2p3 mitigation: two control-plane failure classes are FAST
        # fails where no sandbox was ever created (the placement query never
        # reached the API server / the reservation transaction raced), so a
        # bounded retry is orphan-safe -- this race killed 4 cells today. The
        # 'still queued after Ns' timeout is deliberately NOT retried: that op
        # may still land later and a retry would leak its sandbox.
        retryable = (
            "durable placement reservation store",
            "client rate limiter Wait",
        )
        last_exc: OperationFailed | None = None
        ids: dict[str, Any] = {}
        for attempt in range(3):
            try:
                _, _, _, ids = self._run_mutation(
                    "create_from_snapshot", args, call_args, expiry=expiry,
                    resolve=_resolve_sandbox("create_from_snapshot"),
                )
                break
            except OperationFailed as exc:
                if not any(marker in str(exc) for marker in retryable):
                    raise
                last_exc = exc
                if attempt >= 2:
                    break
                remaining = expiry - time.monotonic()
                if remaining <= 0.0:
                    raise OperationTimeout(
                        f"create_from_snapshot of {snap_id}: no budget left to retry after {exc}"
                    ) from exc
                time.sleep(min(5.0 * (attempt + 1) + random.uniform(0.0, 3.0), remaining))
        if not ids:
            assert last_exc is not None
            raise last_exc
        sandbox_id = ids["sandbox_id"]
        self.created_sandboxes[sandbox_id] = name
        info = self._wait_running(sandbox_id, expiry=expiry)
        return SB(id=sandbox_id, name=name, node=info.get("node") or "")

    def snapshot(
        self,
        sb: SB | str,
        *,
        ttl: str | None = None,
        note: str | None = None,
        deadline: float | None = None,
    ) -> str:
        """Snapshot a running sandbox; returns the snapshot id."""
        expiry = self._expiry(deadline)
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        args = ["sandboxes", "snapshot", sandbox_id]
        if ttl is not None:
            args += ["--ttl", ttl]
        if note:
            args += ["--note", note]
        call_args = {"sandbox": sandbox_id, "ttl": ttl, "note": note}

        def resolve(submit: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
            snap_id = _snapshot_id_from(op)
            if not snap_id:
                raise CLIError(
                    f"snapshot: no snapshot id in operation result {op.get('result')!r}"
                )
            return {"snapshot_id": snap_id}

        _, _, _, ids = self._run_mutation(
            "snapshot", args, call_args, expiry=expiry, resolve=resolve
        )
        snap_id = ids["snapshot_id"]
        self.created_snapshots[snap_id] = note or (sb.name if isinstance(sb, SB) else sandbox_id)
        return snap_id

    def fork(
        self,
        snap_id: str,
        ttl: str,
        name: str | None = None,
        *,
        deadline: float | None = None,
        queue_deadline: str = "5m",
    ) -> SB:
        """Width-1 fan-out from a snapshot, polled all the way to ``running``.

        Width 1 is deliberate: the control plane serialises children anyway and a
        pending fork fences the next snapshot of the same source, so callers must
        sequence ``snapshot -> forks terminal -> next snapshot`` per sandbox.

        ``deadline`` bounds the WHOLE call -- submit, child dispatch, readiness --
        as one absolute expiry.  Until the child is confirmed ours and live the
        fork stays in :attr:`pending_forks`, which keeps the reaper off the SOURCE
        snapshot: a source deleted under a pending fork orphans the child.
        """
        expiry = self._expiry(deadline)
        name = self._stamp(name)
        args = [
            "images", "fork", snap_id,
            "--count", "1",
            "--ttl", ttl,
            "--identity-rng", "reseed",
            "--identity-clock", "correct",
            "--deadline", queue_deadline,
        ]
        call_args = {"snapshot": snap_id, "ttl": ttl, "name": name}

        def resolve(submit: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
            fork_id = _fork_id_from(op, submit)
            if not fork_id:
                raise CLIError(f"fork: no fork id in {op.get('result')!r} / {submit!r}")
            return {"fork_id": fork_id}

        _, _, _, ids = self._run_mutation(
            "fork", args, call_args, expiry=expiry, resolve=resolve
        )
        fork_id = ids["fork_id"]
        self._note_pending_fork(fork_id, snapshot=snap_id)
        child_id, capacity_waits, lane = self._await_fork_child(
            fork_id, expiry=expiry, snap_id=snap_id
        )
        self._note_pending_fork(fork_id, snapshot=snap_id, child=child_id)
        self.created_sandboxes[child_id] = name
        info = self._wait_running(child_id, expiry=expiry)
        self._journal(
            {
                "ts": _utcnow(),
                "op": "fork_child_ready",
                "args": {"snapshot": snap_id, "fork_id": fork_id, "child": child_id},
                "op_id": fork_id,
                "duration_s": info.get("_wait_s", 0.0),
                "outcome": "ok",
                "capacity_waits": capacity_waits + int(info.get("_capacity_waits") or 0),
                "not_found_polls": info.get("_not_found_polls", 0),
                "placement_lane": lane,
                "node": info.get("node"),
            }
        )
        # The child is in the sandbox ledger now and the reaper deletes sandboxes
        # before snapshots, so the source needs no further protection.
        self._clear_pending_fork(fork_id)
        return SB(id=child_id, name=name, node=info.get("node") or "")

    def _await_fork_child(
        self, fork_id: str, *, expiry: float, snap_id: str
    ) -> tuple[str, int, str]:
        """Poll ``forks get`` until the single child is live, inside the budget.

        A fork operation reports ``succeeded`` as soon as it has dispatched, and
        during its retry loop ``forks get`` can still be showing an abandoned
        attempt's sandbox id -- taking that id yields a 404.  So only accept a
        child that is out of the queued/pending states, and report the lane the
        scheduler used (``same_host`` pins the child to the source's node).
        """
        started = time.monotonic()
        interval = 0.4
        capacity_waits = 0
        last_seen: Any = None
        while True:
            if expiry - time.monotonic() <= 0.0:
                raise OperationTimeout(
                    f"fork {fork_id} of {snap_id} produced no child in "
                    f"{time.monotonic() - started:.0f}s: {last_seen}"
                )
            detail = self.get_fork(fork_id, expiry=expiry)
            last_seen = detail
            for child in _fork_children(detail):
                state = _state_of(child)
                child_id = (
                    child.get("sandboxId")
                    or child.get("sandbox_id")
                    or child.get("sandbox")
                    or child.get("id")
                )
                if not (isinstance(child_id, str) and child_id.startswith("sandbox-")):
                    continue
                if state in ("failed", "cancelled", "canceled"):
                    raise OperationFailed(f"fork {fork_id} child {child_id} -> {state}: {child}")
                if _looks_capacity_bound(state):
                    capacity_waits += 1
                    continue
                lane = str(child.get("placement_lane") or child.get("placementLane") or "")
                return child_id, capacity_waits, lane
            state = _state_of(detail)
            if state in TERMINAL_BAD:
                raise OperationFailed(f"fork {fork_id} of {snap_id} -> {state}: {detail}")
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                continue  # the guard at the top of the loop raises, with last_seen
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.4, 2.0)

    def expose(self, sb: SB | str, port: int) -> str:
        """Publish a guest TCP port through the ingress gateway; returns the base URL."""
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        t0 = time.monotonic()
        try:
            payload = self._cli(
                ["sandboxes", "expose", sandbox_id, str(port)],
                retries=2,
                expiry=self._expiry(None),
            )
        except Exception as exc:
            self._record_read(
                "expose", {"sandbox": sandbox_id, "port": port},
                time.monotonic() - t0, "error", {"error": str(exc)},
            )
            raise
        dt = time.monotonic() - t0
        url = (payload or {}).get("url")
        if not url:
            self._record_read(
                "expose", {"sandbox": sandbox_id, "port": port}, dt, "error",
                {"error": f"no url in {payload!r}"},
            )
            raise CLIError(f"expose: no url in {payload!r}")
        self._record_read("expose", {"sandbox": sandbox_id, "port": port}, dt, "ok", {"url": url})
        if isinstance(sb, SB):
            sb.base_url = url
        return url

    def exec(
        self,
        sb: SB | str,
        cmd: str | Sequence[str],
        *,
        timeout: str = "60s",
        check: bool = True,
    ) -> str:
        """Run a command in the guest through the exec gateway; returns stdout."""
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        argv = ["bash", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        wall_timeout = _parse_go_duration(timeout) + 30.0
        t0 = time.monotonic()
        try:
            payload = self._cli(
                ["sandboxes", "exec", sandbox_id, "--timeout", timeout, "--", *argv],
                timeout=wall_timeout,
                expiry=time.monotonic() + wall_timeout,
            )
        except Exception as exc:
            self._record_read(
                "exec", {"sandbox": sandbox_id, "cmd": _clip(argv)},
                time.monotonic() - t0, "error", {"error": str(exc)},
            )
            raise
        dt = time.monotonic() - t0
        payload = payload or {}
        code = int(payload.get("exit_code", payload.get("exitCode", 0)) or 0)
        stdout = payload.get("stdout") or ""
        stderr = payload.get("stderr") or ""
        self._record_read(
            "exec", {"sandbox": sandbox_id, "cmd": _clip(argv)}, dt,
            "ok" if code == 0 else "nonzero", {"exit_code": code},
        )
        if check and code != 0:
            raise CLIError(
                f"exec {_clip(argv)} in {sandbox_id} exited {code}\n"
                f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
            )
        return stdout

    def delete_sandbox(self, sb: SB | str, *, deadline: float | None = None) -> None:
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        self._run_mutation(
            "delete_sandbox",
            ["sandboxes", "delete", sandbox_id],
            {"sandbox": sandbox_id},
            expiry=self._expiry(deadline),
        )
        self.created_sandboxes.pop(sandbox_id, None)

    def delete_snapshot(self, snap_id: str, *, deadline: float | None = None) -> None:
        self._run_mutation(
            "delete_snapshot",
            ["images", "delete", snap_id],
            {"snapshot": snap_id},
            expiry=self._expiry(deadline),
        )
        self.created_snapshots.pop(snap_id, None)

    def lease(self, sb: SB | str, extend: str) -> None:
        """Extend a sandbox lease so a long bake does not hibernate under us."""
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        t0 = time.monotonic()
        self._cli(
            ["sandboxes", "lease", sandbox_id, "--extend", extend],
            retries=2,
            expiry=self._expiry(None),
        )
        self._record_read(
            "lease", {"sandbox": sandbox_id, "extend": extend}, time.monotonic() - t0, "ok"
        )

    # ------------------------------------------------------------------
    # reaper
    # ------------------------------------------------------------------
    def journal_ledger(self) -> tuple[set[str], set[str]]:
        """Replay every journal file in our directory into (sandbox ids, snapshot ids)."""
        sandboxes: set[str] = set()
        snapshots: set[str] = set()
        for path in sorted(self.journal_path.parent.glob("*.jsonl")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("outcome") != "ok":
                    continue
                op = record.get("op")
                args = record.get("args") or {}
                result = record.get("result") or {}
                if op in ("create_from_template", "create_from_snapshot", "fork"):
                    sid = (
                        result.get("sandbox_id")
                        or result.get("sandboxId")
                        or args.get("sandbox")
                    )
                    if sid:
                        sandboxes.add(sid)
                elif op == "fork_child_ready":
                    child = args.get("child")
                    if child:
                        sandboxes.add(child)
                elif op == "snapshot":
                    # Either spelling: _normalized_result writes the snake_case
                    # twin now, but journals written before it say snapshotId only.
                    snap = result.get("snapshot_id") or result.get("snapshotId")
                    if snap:
                        snapshots.add(snap)
                elif op == "delete_sandbox":
                    sandboxes.discard(args.get("sandbox") or "")
                elif op == "delete_snapshot":
                    snapshots.discard(args.get("snapshot") or "")
        sandboxes.update(self.created_sandboxes)
        snapshots.update(self.created_snapshots)
        sandboxes.discard("")
        snapshots.discard("")
        return sandboxes, snapshots

    def journal_unresolved_ops(self) -> dict[str, dict[str, Any]]:
        """Operations any process writing to this journal dir left un-settled.

        The final sweep usually runs in a FRESH process (``reaper.jsonl``), so the
        :attr:`unresolved_ops` of the run that timed out is not in memory for it
        -- the journal is.  Ids an earlier sweep already drove to a terminal state
        are dropped again here, so nothing is chased forever.
        """
        noted: dict[str, dict[str, Any]] = {}
        settled: set[str] = set()
        for path in sorted(self.journal_path.parent.glob("*.jsonl")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                op_id = record.get("unresolved_op")
                if isinstance(op_id, str) and op_id:
                    noted[op_id] = {
                        "op": record.get("op"),
                        "args": record.get("args") or {},
                        "error": record.get("error"),
                        "ts": record.get("ts"),
                    }
                for entry in record.get("operations") or ():
                    if not isinstance(entry, dict) or not entry.get("terminal"):
                        continue
                    done = entry.get("op_id")
                    if isinstance(done, str) and done:
                        settled.add(done)
        for op_id in settled:
            noted.pop(op_id, None)
        return noted

    def _drive_operation(self, op_id: str, expiry: float, *, cancel: bool) -> dict[str, Any]:
        """Poll (and optionally cancel) ONE operation until terminal or out of window.

        A create or fork that timed out on its caller is not dead: it can still
        land a sandbox later, which is exactly how a "clean" sweep leaves a live
        microVM behind.  Cancellation is best effort -- the poll after it is the
        verdict -- and a 404 counts as terminal: the control plane has forgotten
        the operation, so nothing more can come of it.
        """
        state = ""
        op: dict[str, Any] = {}
        error = ""
        asked = False
        interval = 1.0
        while expiry - time.monotonic() > 0.0:
            try:
                payload = self._cli(
                    ["operations", "get", op_id],
                    retries=1,
                    timeout=min(30.0, expiry - time.monotonic()),
                    expiry=expiry,
                )
            except FarplaneError as exc:
                error = f"{type(exc).__name__}: {exc}"
                if _looks_missing(exc):
                    return {"terminal": True, "state": state or "gone", "op": op,
                            "cancel_asked": asked, "error": error}
                break
            if isinstance(payload, dict):
                op = payload
                state = _state_of(op) or state
            if state in TERMINAL_OK or state in TERMINAL_BAD:
                return {"terminal": True, "state": state, "op": op,
                        "cancel_asked": asked, "error": error}
            if cancel and not asked:
                asked = True
                try:
                    self._cli(
                        ["operations", "cancel", op_id],
                        timeout=min(30.0, max(0.1, expiry - time.monotonic())),
                        expiry=expiry,
                    )
                except FarplaneError as exc:
                    error = f"cancel refused: {type(exc).__name__}: {exc}"
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.5, 5.0)
        return {"terminal": False, "state": state or "unknown", "op": op,
                "cancel_asked": asked, "error": error}

    def _drive_fork(self, fork_id: str, expiry: float) -> tuple[bool, bool, set[str]]:
        """``(terminal, children_known, child ids)`` for one dispatched fork.

        "Terminal" means nothing more can appear under this fork: the record says
        so, or every child it lists is out of the queued/pending states.  A fork
        we cannot look up at all leaves ``children_known`` False, which retains
        its source snapshot -- a retained snapshot expires on its TTL, an orphaned
        child does not.
        """
        children: set[str] = set()
        known = False
        interval = 1.0
        while expiry - time.monotonic() > 0.0:
            try:
                detail = self.get_fork(fork_id, expiry=expiry)
            except FarplaneError as exc:
                if _looks_missing(exc):
                    # A fork id the control plane never heard of is finished; an
                    # OPERATION id standing in for one proves nothing about it.
                    return not _looks_like_op_id(fork_id), False, children
                break
            known = True
            pending = False
            for child in _fork_children(detail):
                child_id = (
                    child.get("sandboxId")
                    or child.get("sandbox_id")
                    or child.get("sandbox")
                    or child.get("id")
                )
                if isinstance(child_id, str) and child_id.startswith("sandbox-"):
                    children.add(child_id)
                if _looks_capacity_bound(_state_of(child)):
                    pending = True
            state = _state_of(detail)
            if not pending and (state in TERMINAL_OK or state in TERMINAL_BAD or children):
                return True, known, children
            remaining = expiry - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(interval, remaining))
            interval = min(interval * 1.5, 5.0)
        return False, known, children

    def _quiesce(self, window_s: float, *, cancel: bool) -> dict[str, Any]:
        """Drive every un-settled operation and pending fork to a verdict.

        Bounded by ``window_s``.  Whatever the settled operations produced joins
        the ledger (that is the sandbox a timed-out create landed after its caller
        gave up); whatever refuses to settle is reported, so the sweep says
        "partial" instead of "clean".
        """
        expiry = time.monotonic() + max(0.0, float(window_s))
        with self._lock:
            pending_ops = {k: dict(v) for k, v in self.unresolved_ops.items()}
            pending_forks = {k: dict(v) for k, v in self.pending_forks.items()}
        for op_id, info in self.journal_unresolved_ops().items():
            pending_ops.setdefault(op_id, info)

        operations: list[dict[str, Any]] = []
        sandboxes: set[str] = set()
        snapshots: set[str] = set()
        for op_id, info in sorted(pending_ops.items()):
            drive = self._drive_operation(op_id, expiry, cancel=cancel)
            op = drive.pop("op")
            entry: dict[str, Any] = {"op_id": op_id, "op": info.get("op"), **drive}
            sandbox_id = _sandbox_id_from(op, {})
            if not sandbox_id.startswith("sandbox-"):
                sandbox_id = ""
            if drive["terminal"]:
                if sandbox_id:
                    sandboxes.add(sandbox_id)
                    entry["sandbox"] = sandbox_id
                snap = _snapshot_id_from(op)
                if snap:
                    snapshots.add(snap)
                    entry["snapshot"] = snap
                with self._lock:
                    self.unresolved_ops.pop(op_id, None)
            else:
                entry["submit_error"] = info.get("error")
            if info.get("op") == "fork":
                fork_id = _fork_id_from(op, {}) or op_id
                source = str((info.get("args") or {}).get("snapshot") or "")
                claim = pending_forks.setdefault(
                    fork_id, {"snapshot": source, "child": "", "op_id": op_id}
                )
                if source and not claim.get("snapshot"):
                    claim["snapshot"] = source
                if sandbox_id and not claim.get("child"):
                    claim["child"] = sandbox_id
                if not drive["terminal"]:
                    claim["op_unsettled"] = True
                entry["fork_id"] = fork_id
                if fork_id != op_id:
                    # The claim was filed under the SUBMITTING op id (a fork that
                    # timed out never told us its fork id).  Now that it has, move
                    # it: `forks get <op id>` would 404 forever and retain the
                    # source snapshot on every future sweep.
                    stale = pending_forks.pop(op_id, None)
                    if stale and not claim.get("child"):
                        claim["child"] = str(stale.get("child") or "")
                    self._clear_pending_fork(op_id)
            operations.append(entry)

        forks: list[dict[str, Any]] = []
        for fork_id, claim in sorted(pending_forks.items()):
            terminal, known, children = self._drive_fork(fork_id, expiry)
            child = str(claim.get("child") or "")
            if child:
                children.add(child)
                known = True
            if claim.get("op_unsettled"):
                terminal = False
            sandboxes |= children
            forks.append(
                {
                    "fork_id": fork_id,
                    "snapshot": str(claim.get("snapshot") or ""),
                    "terminal": terminal,
                    "children_known": known,
                    "children": sorted(children),
                }
            )
        return {
            "window_s": round(max(0.0, float(window_s)), 3),
            "operations": operations,
            "forks": forks,
            "sandboxes": sandboxes,
            "snapshots": snapshots,
        }

    def _settle_fork_sources(
        self,
        forks: list[dict[str, Any]],
        results: list[dict[str, Any]],
        keep_set: set[str],
        *,
        dry_run: bool,
    ) -> dict[str, str]:
        """Chase pending forks' children on a FRESH scan; hold sources we cannot clear.

        A source snapshot is released only when its fork is terminal AND the
        children it produced are provably gone -- "provably" means a sandbox list
        taken AFTER this sweep's deletions, because a child that materialised
        during the sweep is invisible to the scan that opened it.  Every such child
        found alive is deleted here: our fork made it, so it is ours.
        """
        holds: dict[str, str] = {}
        if not forks:
            return holds
        try:
            live = {str(item.get("id") or "") for item in self.list_sandboxes()}
        except FarplaneError as exc:
            reason = f"fresh sandbox scan failed ({type(exc).__name__}: {exc})"
            for entry in forks:
                snapshot = str(entry.get("snapshot") or "")
                if snapshot:
                    holds[snapshot] = reason
            return holds
        for entry in forks:
            fork_id = str(entry.get("fork_id") or "?")
            snapshot = str(entry.get("snapshot") or "")
            alive = [
                child for child in entry.get("children") or ()
                if child in live and child not in keep_set
            ]
            failed: list[str] = []
            for child in alive:
                record: dict[str, Any] = {
                    "kind": "sandbox",
                    "id": child,
                    "name": "",
                    "node": None,
                    "reason": f"pending-fork child of {snapshot or fork_id}",
                }
                if dry_run:
                    record["outcome"] = "would-delete"
                else:
                    try:
                        self.delete_sandbox(child)
                        record["outcome"] = "deleted"
                    except Exception as exc:  # keep reaping the rest
                        record["outcome"] = "failed"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        failed.append(child)
                results.append(record)
            if not snapshot:
                continue
            if not entry.get("terminal"):
                holds[snapshot] = f"fork {fork_id} never reached a terminal state"
            elif not entry.get("children_known"):
                holds[snapshot] = f"fork {fork_id}: child list unavailable"
            elif dry_run and alive:
                holds[snapshot] = f"fork {fork_id} child {', '.join(alive)} still live (dry run)"
            elif failed:
                holds[snapshot] = f"fork {fork_id} child {', '.join(failed)} could not be deleted"
            else:
                self._clear_pending_fork(fork_id)
        return holds

    def reaper(
        self,
        prefix: str | None = None,
        *,
        dry_run: bool = False,
        keep: Iterable[str] = (),
        quiesce_s: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Delete every sandbox/snapshot we own that matches ``prefix``.

        Ownership must be established at least one of three ways: the display
        name starts with ``prefix``; the journal says we created it; or it is a
        fork child of one of our snapshots.  Anything else -- foreign snapshots,
        other people's sandboxes -- is never touched.  ``keep`` protects specific
        ids (TEMPLATE_SNAP and the bake sandbox).

        Two things happen BEFORE the resource scan, both about operations that
        were still moving when their caller gave up (``quiesce_s`` bounds both):

        * every operation we never saw settle is polled -- and cancelled, unless
          this is a dry run -- to a terminal state, and whatever it produced joins
          the ledger.  Scanning first is how a timed-out create lands a live
          sandbox thirty seconds after a sweep called itself clean.
        * a fork whose child was never confirmed keeps its SOURCE snapshot until
          the fork is terminal and a fresh sandbox scan finds (and deletes) the
          child.  A retained snapshot expires on its TTL; an orphaned child does
          not.

        Anything that will not settle inside the window comes back as an
        ``unsettled`` operation record and is journalled ``outcome: partial`` --
        never quietly dropped.
        """
        t0 = time.monotonic()
        prefix = self.prefix if prefix is None else prefix
        keep_set = {k for k in keep if k}
        owner = self.owner()
        quiesce = self._quiesce(quiesce_s, cancel=not dry_run)
        ledger_sandboxes, ledger_snapshots = self.journal_ledger()
        ledger_sandboxes |= quiesce["sandboxes"]
        ledger_snapshots |= quiesce["snapshots"]

        results: list[dict[str, Any]] = []
        for entry in quiesce["operations"]:
            if entry.get("terminal"):
                continue
            results.append(
                {
                    "kind": "operation",
                    "id": entry.get("op_id"),
                    "name": entry.get("op"),
                    "reason": "never reached a terminal state",
                    "state": entry.get("state"),
                    "outcome": "unsettled",
                }
            )

        victims: list[dict[str, Any]] = []
        for item in self.list_sandboxes():
            sandbox_id = item.get("id")
            if not sandbox_id or sandbox_id in keep_set:
                continue
            name = item.get("name") or ""
            if owner and item.get("owner") and item.get("owner") != owner:
                continue
            by_name = bool(prefix) and name.startswith(prefix)
            by_ledger = sandbox_id in ledger_sandboxes
            by_source = (item.get("sourceSnapshot") or "") in ledger_snapshots
            if not (by_name or by_ledger or by_source):
                continue
            victims.append(
                {
                    "kind": "sandbox",
                    "id": sandbox_id,
                    "name": name,
                    "node": item.get("node"),
                    "reason": "name" if by_name else ("ledger" if by_ledger else "fork-of-ours"),
                }
            )
        for victim in victims:
            record = dict(victim)
            if dry_run:
                record["outcome"] = "would-delete"
            else:
                try:
                    self.delete_sandbox(victim["id"])
                    record["outcome"] = "deleted"
                except Exception as exc:  # keep reaping the rest
                    record["outcome"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)

        # Only now, with this sweep's deletions applied, can a fork's child be
        # called absent -- and only then may its source snapshot go.
        retained = self._settle_fork_sources(
            quiesce["forks"], results, keep_set, dry_run=dry_run
        )

        live_images: set[str] = set()
        for image in self.list_images():
            image_id = image.get("id") or ""
            if image_id:
                live_images.add(image_id)
            snap = ((image.get("template") or {}).get("sourceSnapshotId")) or image.get("snapshotId")
            if snap:
                live_images.add(snap)
        for snap_id in sorted(ledger_snapshots):
            if snap_id in keep_set or snap_id not in live_images:
                continue
            hold = retained.get(snap_id)
            if hold:
                results.append(
                    {"kind": "snapshot", "id": snap_id, "name": "", "reason": hold,
                     "outcome": "retained"}
                )
                continue
            record = {"kind": "snapshot", "id": snap_id, "name": "", "reason": "ledger"}
            if dry_run:
                record["outcome"] = "would-delete"
            else:
                try:
                    self.delete_snapshot(snap_id)
                    record["outcome"] = "deleted"
                except Exception as exc:  # keep reaping the rest
                    record["outcome"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)

        unsettled = [e for e in quiesce["operations"] if not e.get("terminal")]
        self._journal(
            {
                "ts": _utcnow(),
                "op": "reaper",
                "args": {
                    "prefix": prefix,
                    "dry_run": dry_run,
                    "keep": sorted(keep_set),
                    "quiesce_s": quiesce["window_s"],
                },
                "op_id": None,
                "duration_s": round(time.monotonic() - t0, 4),
                "outcome": "partial" if (unsettled or retained) else "ok",
                "victims": results,
                "operations": quiesce["operations"],
                "pending_forks": quiesce["forks"],
                "retained_snapshots": retained,
            }
        )
        return results

    # ------------------------------------------------------------------
    def _stamp(self, name: str | None) -> str:
        if not name:
            name = f"{self.prefix}{uuid.uuid4().hex[:10]}"
        elif not name.startswith(self.prefix):
            name = f"{self.prefix}{name}"
        return name[:63]

    def timing_summary(self) -> dict[str, dict[str, float]]:
        """Per-op duration stats over everything recorded so far."""
        buckets: dict[str, list[float]] = {}
        with self._lock:
            records = list(self.timings)
        for record in records:
            if record.get("outcome") not in ("ok", "nonzero"):
                continue
            duration = record.get("duration_s")
            if isinstance(duration, (int, float)):
                buckets.setdefault(str(record.get("op")), []).append(float(duration))
        return {op: summarize(values) for op, values in sorted(buckets.items())}


def _fork_children(detail: Any) -> list[dict[str, Any]]:
    """Pull the per-child list out of a ``forks get`` payload, shape-agnostically."""
    if not isinstance(detail, dict):
        return []
    for key in ("children", "items", "childSandboxes", "child_sandboxes", "sandboxes"):
        value = detail.get(key)
        if isinstance(value, list):
            return [child for child in value if isinstance(child, dict)]
    for key in ("child", "sandbox"):
        value = detail.get(key)
        if isinstance(value, dict):
            return [value]
    ids = detail.get("sandboxIds") or detail.get("sandbox_ids")
    if isinstance(ids, list):
        return [{"sandboxId": sid} for sid in ids if isinstance(sid, str)]
    return []


def _clip(value: Any, limit: int = 240) -> str:
    text = " ".join(value) if isinstance(value, list) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_go_duration(text: str) -> float:
    """Minimal Go-duration parser for the units the CLI accepts (s/m/h)."""
    total = 0.0
    number = ""
    for char in str(text):
        if char.isdigit() or char == ".":
            number += char
            continue
        if not number:
            continue
        total += float(number) * {"s": 1.0, "m": 60.0, "h": 3600.0}.get(char, 1.0)
        number = ""
    if number:
        total += float(number)
    return total or 60.0


def summarize(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/max/mean over a sample; an empty sample yields zeros."""
    data = sorted(float(v) for v in values)
    if not data:
        return {"n": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}

    def pct(fraction: float) -> float:
        if len(data) == 1:
            return data[0]
        position = fraction * (len(data) - 1)
        low = int(position)
        high = min(low + 1, len(data) - 1)
        return data[low] + (data[high] - data[low]) * (position - low)

    return {
        "n": len(data),
        "min": round(data[0], 4),
        "p50": round(pct(0.5), 4),
        "p95": round(pct(0.95), 4),
        "max": round(data[-1], 4),
        "mean": round(sum(data) / len(data), 4),
    }
