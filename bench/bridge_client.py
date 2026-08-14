"""HTTP client for the FLE sandbox Bridge API v1.

The bridge listens on TCP 8730 inside the guest container; Farplane's ingress
publishes it as ``https://8730--<sandbox>.compute.ethpandaops.io``.  Everything
here is synchronous and blocking -- callers that need concurrency wrap it in
``asyncio.to_thread``.  One :class:`requests.Session` per instance keeps the
TLS handshake off the hot path.
"""

from __future__ import annotations

import time
from typing import Any

import requests

__all__ = ["Bridge", "BridgeError"]


class BridgeError(RuntimeError):
    """The bridge answered with a non-2xx status or an unusable body."""


class Bridge:
    """Client for one sandbox's bridge.

    Parameters
    ----------
    base_url:
        Scheme + host of the exposed route, e.g. ``https://8730--sandbox-x.…``.
    timeout:
        Default per-request timeout in seconds.  ``execute`` and ``probe``
        override it because they block on the game loop.
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        # The route is a single upstream; keep the connection pinned.
        self.session.headers.update({"Connection": "keep-alive"})

    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        retries: int = 3,
    ) -> Any:
        """Issue one bridge call, retrying only transient ingress failures.

        The Farplane ingress occasionally answers 502/503/504 for a route whose
        upstream is perfectly healthy (design v2 calls this "504 reconciliation").
        Those are retried with a short backoff; a 4xx from the bridge itself is a
        real error and is raised immediately.
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
                last = BridgeError(f"{method} {url} failed: {type(exc).__name__}: {exc}")
                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                raise last from exc
            if response.status_code in (502, 503, 504) and attempt < retries:
                last = BridgeError(f"{method} {url} -> HTTP {response.status_code} (gateway)")
                time.sleep(1.0 + attempt)
                continue
            if response.status_code >= 300:
                raise BridgeError(
                    f"{method} {url} -> HTTP {response.status_code}: {response.text[:1000]}"
                )
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise BridgeError(
                    f"{method} {url}: non-JSON body {response.text[:500]!r}"
                ) from exc
        raise last or BridgeError(f"{method} {url}: exhausted retries")

    # ------------------------------------------------------------------
    def health(self) -> bool:
        """True once the bridge answers ``{"status": "ok"}``; never raises."""
        try:
            # No retries: wait_healthy is the retry loop, and inflating a single
            # health probe would corrupt the measured fork-to-ready latency.
            payload = self._request("GET", "/health", timeout=10.0, retries=0)
        except BridgeError:
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def reset(self, *, timeout: float = 900.0) -> dict[str, Any]:
        """Reset the env to the task's greenfield starting state (P1 semantics)."""
        payload = self._request("POST", "/reset", json_body={}, timeout=timeout)
        return payload if isinstance(payload, dict) else {}

    def wait_healthy(self, deadline_s: float = 300.0, *, poll_interval: float = 2.0) -> float:
        """Block until :meth:`health` is true.  Returns seconds waited."""
        started = time.monotonic()
        while True:
            if self.health():
                return time.monotonic() - started
            waited = time.monotonic() - started
            if waited > deadline_s:
                raise TimeoutError(
                    f"bridge at {self.base_url} not healthy after {deadline_s:.0f}s"
                )
            time.sleep(poll_interval)

    def execute(self, code: str, *, timeout: float = 600.0) -> dict[str, Any]:
        """Run one program in the REPL; gym-step semantics minus verify/quota."""
        payload = self._request("POST", "/execute", json_body={"code": code}, timeout=timeout)
        if not isinstance(payload, dict):
            raise BridgeError(f"/execute returned {payload!r}")
        return payload

    def probe(self, entity: str, *, timeout: float = 300.0) -> dict[str, Any]:
        """One fixed 60 in-game-second throughput window on ``entity``."""
        payload = self._request("POST", "/probe", json_body={"entity": entity}, timeout=timeout)
        if not isinstance(payload, dict):
            raise BridgeError(f"/probe returned {payload!r}")
        return payload

    def state_save(self, *, timeout: float = 900.0) -> str:
        """``GameState.to_raw()`` JSON string for the live game."""
        payload = self._request("GET", "/state-save", timeout=timeout)
        if not isinstance(payload, dict) or not isinstance(payload.get("state"), str):
            raise BridgeError(f"/state-save returned {payload!r}")
        return payload["state"]

    def state_restore(self, state: str, *, timeout: float = 900.0) -> None:
        payload = self._request(
            "POST", "/state-restore", json_body={"state": state}, timeout=timeout
        )
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise BridgeError(f"/state-restore returned {payload!r}")

    def system_prompt(self, *, timeout: float = 120.0) -> str:
        payload = self._request("GET", "/system-prompt", timeout=timeout)
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
        payload = self._request("GET", "/meta", timeout=timeout)
        if not isinstance(payload, dict):
            raise BridgeError(f"/meta returned {payload!r}")
        return payload

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
