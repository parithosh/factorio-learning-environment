"""HTTP client for the FLE sandbox Bridge API v1.

The bridge listens on TCP 8730 inside the guest container; Farplane's ingress
publishes it as ``https://8730--<sandbox>.compute.ethpandaops.io``.  Everything
here is synchronous and blocking -- callers that need concurrency wrap it in
``asyncio.to_thread``.  One :class:`requests.Session` per instance keeps the
TLS handshake off the hot path.

Two rules the caller inherits from this module:

* **Retries are idempotent-only.**  A GET may be replayed; ``POST /execute``,
  ``/probe``, ``/reset`` and ``/state-restore`` mutate the game, so a failure
  that happened once the request was already on the wire is reported as
  :class:`BridgeError` with ``ambiguous=True`` instead of being replayed.
* **Auth.**  When ``FLE_BRIDGE_TOKEN`` is set in the environment (or ``token=``
  is passed explicitly) every request carries ``Authorization: Bearer
  <token>``; a bridge started with a token answers 401 to anything else on its
  TCP listener.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

__all__ = ["Bridge", "BridgeError"]

# urllib3 raises one of these *before* a single request byte is written, so a
# failure carrying one in its cause chain cannot have been executed remotely.
_PRE_SEND_CAUSES = frozenset(
    {"NewConnectionError", "NameResolutionError", "ConnectTimeoutError"}
)


def _never_left_the_socket(exc: requests.RequestException) -> bool:
    """True only when ``exc`` proves the request was never delivered.

    Used to decide whether a failed *mutating* call is ambiguous.  We fail
    closed: everything we cannot positively classify as pre-send -- a read
    timeout above all, which is exactly what a long ``/execute`` looks like
    when the ingress drops the connection -- counts as possibly executed.
    """
    if isinstance(
        exc,
        (
            requests.ConnectTimeout,
            requests.URLRequired,
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema,
            requests.exceptions.InvalidURL,
            requests.exceptions.InvalidHeader,
        ),
    ):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False
    if isinstance(exc, (requests.exceptions.ProxyError, requests.exceptions.SSLError)):
        return False  # may equally have broken mid-transfer
    # requests wraps connect failures as ConnectionError(MaxRetryError(reason=...)).
    cause: object | None = exc.args[0] if exc.args else None
    for _ in range(4):
        if cause is None:
            return False
        if type(cause).__name__ in _PRE_SEND_CAUSES:
            return True
        cause = getattr(cause, "reason", None)
    return False


def _error_detail(response: requests.Response, *, limit: int = 1000) -> str:
    """Readable tail for an error response.

    The bridge reports errors as ``{"error": ...}`` and adds a
    ``correlation_id`` on a 500 -- that id is the only handle on the traceback,
    which stays in the container log -- so lift both out instead of dumping the
    raw body.  Ingress errors are HTML and fall through to collapsed text.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        cid = body.get("correlation_id")
        detail = body["error"]
        return f"{detail} [correlation_id={cid}]" if isinstance(cid, str) else detail
    return " ".join(response.text[:limit].split())


class BridgeError(RuntimeError):
    """The bridge answered with a non-2xx status or an unusable body.

    ``ambiguous`` is true when a mutating request was already on the wire and
    we never learned whether the bridge applied it.  Such a call must not be
    replayed: the caller has to re-derive the world state (``/meta``,
    ``/state-save``) or abandon the branch.  ``status`` is the HTTP status when
    the bridge (or the ingress) answered at all, and ``None`` for a transport
    failure.
    """

    def __init__(
        self, message: str, *, ambiguous: bool = False, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.ambiguous = bool(ambiguous)
        self.status = status


class Bridge:
    """Client for one sandbox's bridge.

    Parameters
    ----------
    base_url:
        Scheme + host of the exposed route, e.g. ``https://8730--sandbox-x.…``.
    timeout:
        Default per-request timeout in seconds.  ``execute`` and ``probe``
        override it because they block on the game loop.
    token:
        Shared secret for the bridge's TCP listener.  Defaults to
        ``$FLE_BRIDGE_TOKEN``; when it resolves to a non-empty value every
        request carries ``Authorization: Bearer <token>``.  Pass ``token=""``
        to force an unauthenticated client regardless of the environment.
    """

    def __init__(
        self, base_url: str, timeout: float = 60.0, *, token: str | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.token = (os.environ.get("FLE_BRIDGE_TOKEN") if token is None else token) or None
        self.session = requests.Session()
        # The route is a single upstream; keep the connection pinned.
        self.session.headers.update({"Connection": "keep-alive"})
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
        # Why the last health probe failed; wait_healthy uses it to tell "not up
        # yet" from "never will be" (see :meth:`health`).
        self.last_health_error: BridgeError | None = None

    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        idempotent: bool,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        retries: int = 3,
    ) -> Any:
        """Issue one bridge call, replaying it only when ``idempotent``.

        The Farplane ingress occasionally answers 502/503/504 for a route whose
        upstream is perfectly healthy (design v2 calls this "504
        reconciliation").  Replaying that is safe for a GET and *unsafe* for
        anything that mutates the game: a 504 in particular means the ingress
        gave up while the bridge was most likely still executing the request,
        so a retried ``POST /execute`` would run the program twice and a
        retried ``/probe`` would burn a second 60 s window.  ``idempotent=False``
        therefore disables every retry path and converts a gateway status or a
        post-send transport error into ``BridgeError(ambiguous=True)``, leaving
        reconciliation to the caller, which is the only layer that knows what
        the world should look like.  A 4xx from the bridge itself is a real
        error and is raised immediately either way.
        """
        url = f"{self.base_url}{path}"
        last: BridgeError | None = None
        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json_body,
                    timeout=timeout if timeout is not None else self.timeout,
                )
            except requests.RequestException as exc:
                ambiguous = not idempotent and not _never_left_the_socket(exc)
                last = BridgeError(
                    f"{method} {url} failed: {type(exc).__name__}: {exc}"
                    + (" (sent, outcome unknown -- not retried)" if ambiguous else ""),
                    ambiguous=ambiguous,
                )
                if idempotent and attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                raise last from exc
            if response.status_code in (502, 503, 504):
                gateway = BridgeError(
                    f"{method} {url} -> HTTP {response.status_code} (gateway): "
                    f"{_error_detail(response, limit=200)}",
                    ambiguous=not idempotent,
                    status=response.status_code,
                )
                if idempotent and attempt < retries:
                    last = gateway
                    time.sleep(1.0 + attempt)
                    continue
                raise gateway
            # The bridge itself only ever answers 401, and its body is the generic
            # {"error": "unauthorized"} -- so the diagnosis has to come from our own
            # state.  A 403 can only be an ingress/proxy denial; same treatment.
            if response.status_code in (401, 403):
                why = "no token sent" if not self.token else "token mismatch"
                raise BridgeError(
                    f"{method} {url} -> HTTP {response.status_code}: bridge rejected the "
                    f"request ({why}); FLE_BRIDGE_TOKEN must match the value the sandbox "
                    f"was started with -- every TCP path is gated, /health included",
                    status=response.status_code,
                )
            if response.status_code >= 300:
                # 400/408/411/413 are our bug (body too big, chunked encoding, ...) and
                # 500 carries a correlation id; neither is ever retried.
                raise BridgeError(
                    f"{method} {url} -> HTTP {response.status_code}: {_error_detail(response)}",
                    status=response.status_code,
                )
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise BridgeError(
                    f"{method} {url}: non-JSON body {response.text[:500]!r}",
                    status=response.status_code,
                ) from exc
        raise last or BridgeError(f"{method} {url}: exhausted retries")

    # ------------------------------------------------------------------
    def health(self, *, timeout: float = 10.0) -> bool:
        """True once the bridge answers ``{"status": "ok"}``; never raises.

        The failure, if any, is kept in :attr:`last_health_error` so
        :meth:`wait_healthy` can distinguish a bridge that is still booting from
        one that rejects us outright.
        """
        try:
            # No retries: wait_healthy is the retry loop, and inflating a single
            # health probe would corrupt the measured fork-to-ready latency.
            payload = self._request(
                "GET", "/health", timeout=timeout, retries=0, idempotent=True
            )
        except BridgeError as exc:
            self.last_health_error = exc
            return False
        self.last_health_error = None
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def reset(self, *, timeout: float = 900.0) -> dict[str, Any]:
        """Reset the env to the task's greenfield starting state (P1 semantics).

        Mutating: never auto-retried (see :meth:`_request`).
        """
        payload = self._request(
            "POST", "/reset", json_body={}, timeout=timeout, idempotent=False
        )
        return payload if isinstance(payload, dict) else {}

    def wait_healthy(self, deadline_s: float = 300.0, *, poll_interval: float = 2.0) -> float:
        """Block until :meth:`health` is true.  Returns seconds waited.

        The deadline is absolute: each poll's own timeout and the sleep after it
        are clamped to the remaining budget, and a bridge that only answers
        *after* the deadline has passed is a timeout, not a success -- callers
        size their next phase off the returned latency.

        A 401/403 aborts the wait immediately: the token is wrong (``/health`` is
        gated too), and no amount of polling fixes that.
        """
        started = time.monotonic()
        deadline = started + max(0.0, float(deadline_s))

        def expired(waited: float) -> TimeoutError:
            return TimeoutError(
                f"bridge at {self.base_url} not healthy after {waited:.1f}s "
                f"(deadline {deadline_s:g}s)"
            )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise expired(time.monotonic() - started)
            if self.health(timeout=min(10.0, remaining)):
                waited = time.monotonic() - started
                if time.monotonic() > deadline:
                    raise expired(waited)  # answered, but too late to be useful
                return waited
            rejected = self.last_health_error
            if rejected is not None and rejected.status in (401, 403):
                raise rejected
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise expired(time.monotonic() - started)
            time.sleep(min(poll_interval, remaining))

    def execute(self, code: str, *, timeout: float = 600.0) -> dict[str, Any]:
        """Run one program in the REPL; gym-step semantics minus verify/quota.

        Mutating: never auto-retried.  A transport failure after the program was
        sent raises ``BridgeError(ambiguous=True)`` -- it may have run.
        """
        payload = self._request(
            "POST", "/execute", json_body={"code": code}, timeout=timeout, idempotent=False
        )
        if not isinstance(payload, dict):
            raise BridgeError(f"/execute returned {payload!r}")
        return payload

    def probe(self, entity: str | None = None, *, timeout: float = 300.0) -> dict[str, Any]:
        """One fixed 60 in-game-second throughput window.

        ``entity`` defaults to the task's ``throughput_entity``: the key is
        omitted from the body when it is ``None`` so the server picks it.
        Mutating (it advances the game): never auto-retried.
        """
        body: dict[str, Any] = {}
        if entity is not None:
            if not entity.strip():
                raise ValueError("probe(entity=) must be a real entity name or None")
            body["entity"] = entity
        payload = self._request(
            "POST", "/probe", json_body=body, timeout=timeout, idempotent=False
        )
        if not isinstance(payload, dict):
            raise BridgeError(f"/probe returned {payload!r}")
        return payload

    def state_save(self, *, timeout: float = 900.0) -> str:
        """``GameState.to_raw()`` JSON string for the live game."""
        payload = self._request("GET", "/state-save", timeout=timeout, idempotent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state"), str):
            raise BridgeError(f"/state-save returned {payload!r}")
        return payload["state"]

    def state_restore(self, state: str, *, timeout: float = 900.0) -> None:
        payload = self._request(
            "POST", "/state-restore", json_body={"state": state}, timeout=timeout,
            idempotent=False,
        )
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise BridgeError(f"/state-restore returned {payload!r}")

    def system_prompt(self, *, timeout: float = 120.0) -> str:
        payload = self._request("GET", "/system-prompt", timeout=timeout, idempotent=True)
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("system_prompt", "prompt", "result"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        raise BridgeError(f"/system-prompt returned {payload!r}")

    def meta(self, *, timeout: float = 60.0) -> dict[str, Any]:
        """``{factorio_pid, elapsed_ticks, entity_count}``."""
        payload = self._request("GET", "/meta", timeout=timeout, idempotent=True)
        if not isinstance(payload, dict):
            raise BridgeError(f"/meta returned {payload!r}")
        return payload

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
