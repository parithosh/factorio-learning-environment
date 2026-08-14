"""Journalled, deadline-bounded wrapper around the ``panda compute`` CLI.

Every mutation is asynchronous on the control plane: the CLI returns an
operation id and we poll ``panda compute operations get <id>`` until the
operation reaches a terminal state.  All calls -- mutations and reads alike --
append a JSONL record to ``bench/journal/<date>.jsonl`` and a structured entry
to :attr:`Farplane.timings`, which is what Tier 0 turns into its constants.

The wrapper is deliberately thin: it shells out, parses ``-o json``, and owns
only the state that the CLI does not keep for us (which resources *we* created,
so the reaper never touches a stranger's sandbox).
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
from typing import Any, Iterable, Sequence

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

    def _cli(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        json_out: bool = True,
        retries: int = 0,
    ) -> Any:
        """Run ``panda compute <args>`` and return parsed JSON (or raw text)."""
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

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout if timeout is not None else self.cli_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = CLIError(f"panda timed out: {' '.join(cmd)}")
                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                raise last_error from exc
            if proc.returncode != 0:
                last_error = CLIError(
                    f"panda exited {proc.returncode}: {' '.join(cmd)}\n"
                    f"stdout: {proc.stdout.strip()[:2000]}\nstderr: {proc.stderr.strip()[:2000]}"
                )
                if attempt < retries:
                    time.sleep(1.0 + attempt)
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
                last_error = CLIError(f"non-JSON output from {' '.join(cmd)}: {text[:2000]}")
                if attempt < retries:
                    time.sleep(1.0 + attempt)
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
        deadline: float | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        """Poll ``operations get`` until terminal.  Returns the final operation."""
        budget = self.default_deadline if deadline is None else float(deadline)
        started = time.monotonic()
        polls = 0
        poll_cost = 0.0
        interval = 0.4
        last_state = ""
        while True:
            call_t0 = time.monotonic()
            op = self._cli(["operations", "get", op_id], retries=3, timeout=60)
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
            if time.monotonic() - started > budget:
                raise OperationTimeout(
                    f"{label or 'operation'} {op_id} still {last_state!r} after {budget:.0f}s"
                )
            time.sleep(interval)
            interval = min(interval * 1.4, 2.0)

    def _run_mutation(
        self,
        op_name: str,
        args: Sequence[str],
        call_args: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        """Submit a mutation, poll to terminal, journal it.  Returns (submit, op, dt)."""
        t0 = time.monotonic()
        op_id: str | None = None
        try:
            submit = self._cli(args)
            op_id = self._extract_op_id(submit)
            if op_id is None:
                raise CLIError(f"{op_name}: no operation id in {submit!r}")
            op = self._poll_operation(op_id, deadline=deadline, label=op_name)
        except Exception as exc:
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
                }
            )
            raise
        dt = time.monotonic() - t0
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
                "result": op.get("result"),
            }
        )
        return (submit if isinstance(submit, dict) else {}), op, dt

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

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def owner(self) -> str:
        """Authenticated handle, used to keep the reaper off other people's boxes."""
        if self._session_owner is None:
            session = self._cli(["session"], retries=2, timeout=60)
            user = (session or {}).get("user") or {}
            self._session_owner = user.get("handle") or user.get("subject") or ""
        return self._session_owner

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        return self._cli(["sandboxes", "get", sandbox_id], retries=3, timeout=60)

    def list_sandboxes(self, limit: int = 500) -> list[dict[str, Any]]:
        payload = self._cli(["sandboxes", "list", "--limit", str(limit)], retries=3, timeout=90)
        return list((payload or {}).get("items") or [])

    def list_images(self, limit: int = 500) -> list[dict[str, Any]]:
        payload = self._cli(["images", "list", "--limit", str(limit)], retries=3, timeout=90)
        return list((payload or {}).get("items") or [])

    def get_fork(self, fork_id: str) -> dict[str, Any]:
        return self._cli(["forks", "get", fork_id], retries=3, timeout=60)

    def nodes(self) -> list[dict[str, Any]]:
        payload = self._cli(["nodes", "list"], retries=2, timeout=60)
        return list((payload or {}).get("items") or [])

    # ------------------------------------------------------------------
    # sandbox lifecycle
    # ------------------------------------------------------------------
    def wait_running(self, sandbox_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        """Block until the sandbox reports ``running``.

        A fork child's sandbox record becomes queryable a beat after the fork
        operation goes terminal, so a 404 here means "not yet", not "gone".
        """
        budget = self.default_deadline if deadline is None else float(deadline)
        started = time.monotonic()
        interval = 0.4
        capacity_waits = 0
        not_found = 0
        state = ""
        while True:
            try:
                info = self.get_sandbox(sandbox_id)
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
            if time.monotonic() - started > budget:
                raise OperationTimeout(f"sandbox {sandbox_id} still {state!r} after {budget:.0f}s")
            time.sleep(interval)
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
        """Create a sandbox from a named template (bare name, no ``@version``)."""
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
        submit, op, _ = self._run_mutation(
            "create_from_template", args, call_args, deadline=deadline
        )
        sandbox_id = submit.get("id") or op.get("targetId") or op.get("target")
        if not sandbox_id:
            raise CLIError(f"create_from_template: no sandbox id in {submit!r} / {op!r}")
        self.created_sandboxes[sandbox_id] = name
        info = self.wait_running(sandbox_id, deadline=deadline)
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
        """
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
        for attempt in range(3):
            try:
                submit, op, _ = self._run_mutation(
                    "create_from_snapshot", args, call_args, deadline=deadline
                )
                break
            except OperationFailed as exc:
                if not any(marker in str(exc) for marker in retryable):
                    raise
                last_exc = exc
                time.sleep(5.0 * (attempt + 1) + random.uniform(0.0, 3.0))
        else:
            assert last_exc is not None
            raise last_exc
        sandbox_id = submit.get("id") or op.get("targetId") or op.get("target")
        if not sandbox_id:
            raise CLIError(f"create_from_snapshot: no sandbox id in {submit!r}")
        self.created_sandboxes[sandbox_id] = name
        info = self.wait_running(sandbox_id, deadline=deadline)
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
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        args = ["sandboxes", "snapshot", sandbox_id]
        if ttl is not None:
            args += ["--ttl", ttl]
        if note:
            args += ["--note", note]
        call_args = {"sandbox": sandbox_id, "ttl": ttl, "note": note}
        _, op, _ = self._run_mutation("snapshot", args, call_args, deadline=deadline)
        result = op.get("result") or {}
        snap_id = result.get("snapshot_id") or result.get("snapshotId")
        if not snap_id:
            raise CLIError(f"snapshot: no snapshot id in operation result {result!r}")
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
        """
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
        submit, op, _ = self._run_mutation("fork", args, call_args, deadline=deadline)
        result = op.get("result") or {}
        fork_id = (
            result.get("fork_id")
            or result.get("forkId")
            or submit.get("fork_id")
            or submit.get("forkId")
        )
        if not fork_id:
            raise CLIError(f"fork: no fork id in {result!r} / {submit!r}")
        child_id, capacity_waits, lane = self._await_fork_child(
            fork_id, deadline=deadline, snap_id=snap_id
        )
        self.created_sandboxes[child_id] = name
        info = self.wait_running(child_id, deadline=deadline)
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
        return SB(id=child_id, name=name, node=info.get("node") or "")

    def _await_fork_child(
        self, fork_id: str, *, deadline: float | None, snap_id: str
    ) -> tuple[str, int, str]:
        """Poll ``forks get`` until the single child is live.

        A fork operation reports ``succeeded`` as soon as it has dispatched, and
        during its retry loop ``forks get`` can still be showing an abandoned
        attempt's sandbox id -- taking that id yields a 404.  So only accept a
        child that is out of the queued/pending states, and report the lane the
        scheduler used (``same_host`` pins the child to the source's node).
        """
        budget = self.default_deadline if deadline is None else float(deadline)
        started = time.monotonic()
        interval = 0.4
        capacity_waits = 0
        last_seen: Any = None
        while True:
            detail = self.get_fork(fork_id)
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
            if time.monotonic() - started > budget:
                raise OperationTimeout(
                    f"fork {fork_id} of {snap_id} produced no child in {budget:.0f}s: {last_seen}"
                )
            time.sleep(interval)
            interval = min(interval * 1.4, 2.0)

    def expose(self, sb: SB | str, port: int) -> str:
        """Publish a guest TCP port through the ingress gateway; returns the base URL."""
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        t0 = time.monotonic()
        try:
            payload = self._cli(["sandboxes", "expose", sandbox_id, str(port)], retries=2)
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
            deadline=deadline,
        )
        self.created_sandboxes.pop(sandbox_id, None)

    def delete_snapshot(self, snap_id: str, *, deadline: float | None = None) -> None:
        self._run_mutation(
            "delete_snapshot",
            ["images", "delete", snap_id],
            {"snapshot": snap_id},
            deadline=deadline,
        )
        self.created_snapshots.pop(snap_id, None)

    def lease(self, sb: SB | str, extend: str) -> None:
        """Extend a sandbox lease so a long bake does not hibernate under us."""
        sandbox_id = sb.id if isinstance(sb, SB) else sb
        t0 = time.monotonic()
        self._cli(["sandboxes", "lease", sandbox_id, "--extend", extend], retries=2)
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
                if op in ("create_from_template", "create_from_snapshot"):
                    sid = result.get("sandbox_id") or args.get("sandbox")
                    if sid:
                        sandboxes.add(sid)
                elif op == "fork_child_ready":
                    child = args.get("child")
                    if child:
                        sandboxes.add(child)
                elif op == "snapshot":
                    snap = result.get("snapshot_id")
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

    def reaper(
        self,
        prefix: str | None = None,
        *,
        dry_run: bool = False,
        keep: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Delete every sandbox/snapshot we own that matches ``prefix``.

        Ownership must be established at least one of three ways: the display
        name starts with ``prefix``; the journal says we created it; or it is a
        fork child of one of our snapshots.  Anything else -- foreign snapshots,
        other people's sandboxes -- is never touched.  ``keep`` protects specific
        ids (TEMPLATE_SNAP and the bake sandbox).
        """
        prefix = self.prefix if prefix is None else prefix
        keep_set = {k for k in keep if k}
        owner = self.owner()
        ledger_sandboxes, ledger_snapshots = self.journal_ledger()

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
            victims.append({"kind": "snapshot", "id": snap_id, "name": "", "reason": "ledger"})

        results: list[dict[str, Any]] = []
        for victim in victims:
            record = dict(victim)
            if dry_run:
                record["outcome"] = "would-delete"
                results.append(record)
                continue
            try:
                if victim["kind"] == "sandbox":
                    self.delete_sandbox(victim["id"])
                else:
                    self.delete_snapshot(victim["id"])
                record["outcome"] = "deleted"
            except Exception as exc:  # keep reaping the rest
                record["outcome"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            results.append(record)

        self._journal(
            {
                "ts": _utcnow(),
                "op": "reaper",
                "args": {"prefix": prefix, "dry_run": dry_run, "keep": sorted(keep_set)},
                "op_id": None,
                "duration_s": 0.0,
                "outcome": "ok",
                "victims": results,
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
