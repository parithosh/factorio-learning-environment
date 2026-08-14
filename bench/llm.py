"""Async LLM clients for the fan-out benchmark (Kimi direct + Codex/Claude OAuth).

Design v2.5 declares the routing deviation this module implements: Kimi models
go through the direct Kimi API (``KIMI_BASE_URL``, models ``k3`` and
``kimi-for-coding``), OpenAI goes through the repo's Codex subscription
provider. That default matrix uses no aggregator, so no middle-out transform
touches any of its models. ``openrouter/*`` models are an opt-in extra and get
middle-out disabled by an explicit per-request opt-out instead; each route
reports its own note (:func:`routing_notes`).

Two measured provider constraints shape the API here (verified against the
live endpoints, not assumed):

* Kimi rejects ``n`` (``invalid value for param n``) and rejects any
  temperature other than 1 (``only 1 is allowed for this model``).
* Codex (ChatGPT backend) rejects ``temperature``/``top_p``/``max_output_tokens``
  outright (see ``fle/eval/inspect/codex/provider.py:_UNSUPPORTED_PARAMS``) and
  only speaks streaming Responses.

So K-way sampling is always K concurrent calls, and the design's DIVERSITY GATE
knob for temperature-locked models -- per-branch strategy hints/personas -- is a
first-class argument (``hints=``) rather than a silent temperature default.
"""

from __future__ import annotations

import ast
import asyncio
import os
import random
import re
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from bench.common import RunJournal, TimingBuckets, atomic_write_json

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

KIMI_MODELS = ("k3", "kimi-for-coding", "k3-256k", "kimi-for-coding-highspeed")
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_PLACEHOLDER_KEY = "chatgpt-oauth"
CLAUDE_BASE_URL = "https://api.anthropic.com"
#: Keeps the anthropic SDK on the Bearer-auth path; the real token is attached
#: per-request by :func:`_claude_auth` (mirrors the Inspect provider).
CLAUDE_PLACEHOLDER_TOKEN = "claude-oauth"
#: Anthropic rejects subscription tokens unless the request carries the Claude
#: Code identity: these betas/headers plus the spoofed first system block.
#: Kept in lockstep with ``fle.eval.inspect.claude.provider`` (not imported:
#: that module pulls inspect_ai into every bench process).
CLAUDE_OAUTH_BETAS = "claude-code-20250219,oauth-2025-04-20"
CLAUDE_CODE_VERSION = "2.1.75"
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)

ROUTING_NOTES = {
    "middle_out": False,
    "note": (
        "v2.5 default matrix: Kimi via direct Kimi API, OpenAI via Codex "
        "subscription. No aggregator on this route, so no middle-out context "
        "transform for ANY of these models; context-overflow behaviour is a "
        "per-model property reported with results."
    ),
}

#: ``openrouter/*`` models do NOT route like the default matrix, so they never
#: report :data:`ROUTING_NOTES` -- its "no aggregator" claim would be false in
#: their own result artifact. Middle-out is still off, but by an explicit
#: per-request opt-out (``transforms: []``) instead of by construction.
OPENROUTER_ROUTING_NOTES = {
    "middle_out": False,
    "note": (
        "OpenRouter (metered aggregator): middle-out compression disabled "
        "explicitly with transforms=[] on every request; routing constrained "
        "by OPENROUTER_QUANTIZATIONS (exactly one value) and/or "
        "OPENROUTER_PROVIDER; the serving upstream is journaled per call."
    ),
}


def routing_notes(provider: str) -> dict[str, Any]:
    """Routing metadata for the provider a model actually RESOLVES to."""
    if provider == "openrouter":
        return OPENROUTER_ROUTING_NOTES
    return ROUTING_NOTES


#: Per-branch diversification knob (design DIVERSITY GATE). Injected as an
#: extra user turn so it is visible in the journal and never a hidden decoding
#: default.
HINT_TEMPLATE = "[Branch strategy hint] {hint}"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    api_model: str
    #: Value that MUST be sent, ``None`` when the parameter is unsupported.
    temperature: float | None
    supports_n: bool
    max_tokens: int | None
    notes: str = ""

    @property
    def temperature_locked(self) -> bool:
        return self.provider == "codex" or self.temperature is not None


def _kimi_spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="kimi",
        api_model=key,
        # Measured: the endpoint 400s on any temperature != 1.
        temperature=1.0,
        supports_n=False,
        # Reasoning tokens are billed inside completion_tokens on this endpoint:
        # measured 4096 truncates k3 mid-reasoning and returns EMPTY content on
        # a full FLE prompt, so the budget has to cover reasoning + program.
        max_tokens=16384,
        notes="temperature locked to 1 by provider; n unsupported -> K concurrent calls",
    )


def _codex_spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="codex",
        api_model=key.split("/", 1)[1] if "/" in key else key,
        temperature=None,
        supports_n=False,
        max_tokens=None,
        notes="ChatGPT Codex backend: no temperature/top_p/max_output_tokens, SSE only",
    )


def _claude_spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="claude",
        api_model=key.split("/", 1)[1] if "/" in key else key,
        # Anthropic's default, sent explicitly: the value lands in the journal
        # and temperature_locked stays True (the Exp-3 persona/diversity gate
        # keys on it -- a None here would silently run identical seats).
        temperature=1.0,
        supports_n=False,
        # The Messages API REQUIRES max_tokens. 16384 covers program + prose
        # on every FLE step observed; thinking is off, so none of it is burned
        # on reasoning.
        max_tokens=16384,
        notes=(
            "Claude Pro/Max subscription (OAuth): Claude Code identity headers "
            "+ spoofed first system block; n unsupported -> K concurrent calls"
        ),
    )


def _local_spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="local",
        api_model=key.split("/", 1)[1] if "/" in key else key,
        # Sent explicitly: journaled, and temperature_locked stays True (the
        # Exp-3 persona/diversity gate keys on it).
        temperature=1.0,
        supports_n=False,
        # Reasoning model: thinking blocks bill inside output. The endpoint
        # allows 128k completion tokens; 32768 keeps a spike from truncating.
        max_tokens=32768,
        notes=(
            "Self-hosted Anthropic-Messages endpoint (LOCAL_LLM_BASE_URL / "
            "LOCAL_LLM_API_KEY). Operator limits: 4 concurrent hard, 2 safe "
            "(KV cache ~1.3M tokens); run with --provider-concurrency 2."
        ),
    )


def _openrouter_spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        provider="openrouter",
        api_model=key.split("/", 1)[1] if "/" in key else key,
        # Sent explicitly: journaled, and temperature_locked stays True (the
        # Exp-3 persona/diversity gate keys on it).
        temperature=1.0,
        supports_n=False,
        # Reasoning models bill thinking inside completion_tokens (measured
        # 11.5k reasoning on one FLE step for deepseek-v4-flash); 32768 keeps
        # a spike from truncating to an empty answer.
        max_tokens=32768,
        notes=(
            "OpenRouter (metered). Middle-out disabled explicitly "
            "(transforms=[] per request); "
            "routing constrained via OPENROUTER_QUANTIZATIONS (one value, "
            "fallbacks on) and/or OPENROUTER_PROVIDER (preference order; "
            "strict pin when no quantization filter); anthropic/* models get "
            "an explicit cache_control breakpoint on the system turn; the "
            "serving upstream is journaled per call (upstream field)."
        ),
    )


DEFAULT_MODELS: tuple[str, ...] = ("k3", "kimi-for-coding", "codex/gpt-5.6-sol")


def resolve_model(name: str) -> ModelSpec:
    if name.startswith("codex/") or name.startswith("codex:"):
        return _codex_spec(name.replace("codex:", "codex/", 1))
    if name.startswith("claude/") or name.startswith("claude:"):
        return _claude_spec(name.replace("claude:", "claude/", 1))
    if name in KIMI_MODELS or name.startswith("kimi") or name.startswith("k3"):
        return _kimi_spec(name)
    if name.startswith("openrouter/") or name.startswith("or/"):
        return _openrouter_spec(
            "openrouter/" + name.split("/", 1)[1]
        )
    if name.startswith("local/"):
        return _local_spec(name)
    raise ValueError(
        f"unknown model {name!r}; use one of {KIMI_MODELS}, 'codex/<model>', "
        "'claude/<model>', 'openrouter/<vendor>/<model>' or 'local/<model>'"
    )


# ---------------------------------------------------------------------------
# Samples & code extraction
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    text: str
    code: str | None
    model: str
    provider: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    #: Prompt-cache counters (0 where unsupported). Folded INTO prompt_tokens
    #: for quota-honest totals; journaled separately so a silently-failing
    #: cache path is visible per call.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: Serving upstream for aggregator routes (OpenRouter): which host
    #: actually served the call. Empty for direct providers.
    upstream: str = ""
    attempts: int = 1
    request_id: str = ""
    hint: str = ""
    #: Provider stop reason. ``length`` means the model was cut off -- with
    #: reasoning models that often means reasoning consumed the whole budget
    #: and no program came back at all.
    finish_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def billed_usage(self) -> BilledUsage:
        """Usage in the shape the journal and the retry accounting want."""
        return BilledUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            upstream=self.upstream,
            finish_reason=self.finish_reason,
        )


_PARSER: Any = None


def _parser() -> Any:
    """Repo's program extractor (``fle.agents.llm.parsing.PythonParser``)."""
    global _PARSER
    if _PARSER is None:
        from fle.agents.llm.parsing import PythonParser

        _PARSER = PythonParser
    return _PARSER


def extract_code(text: str) -> str | None:
    """Extract an executable Python program from a model response.

    Same order of attempts as the repo's ``parse_response`` path
    (``PythonParser.extract_code``): reasoning stripped, whole response if it
    already parses, then fenced blocks, then valid-Python chunks.
    """
    if not text:
        return None
    parser = _parser()
    try:
        _reasoning, final = parser.extract_reasoning_content(text)
    except Exception:
        final = text
    working = final or text
    if parser.is_valid_python(working):
        return working
    block = parser.extract_all_backtick_blocks(working)
    if block:
        return block
    cleaned = working.replace("```python", "").replace("```", "")
    chunks = parser.extract_all_valid_python_chunks(cleaned)
    return chunks or None


_COMMENT_RE = re.compile(r"(?m)#.*$")
_DOCSTRING_RE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')


class _DocstringStripper(ast.NodeTransformer):
    """Strip documentation from EVERY docstring-bearing node.

    Only module-level prose used to be dropped, so two programs differing
    solely in the docstrings of their functions or classes counted as two
    distinct branches -- exactly the cosmetic difference the diversity gate
    must not credit. Module docstrings fall out of the same ``visit_Expr``.
    """

    def visit_Expr(self, node: ast.Expr) -> Any:
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            return None  # docstring or bare prose: documentation, not behaviour
        return self.generic_visit(node)

    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        if not node.body:
            # A docstring-only body IS a ``pass`` body; canonicalize both ways.
            node.body = [ast.Pass()]
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def normalize_program(code: str) -> str:
    """Canonical form for comparing two candidate programs.

    Structural (AST) when the program parses: identical logic written with
    different spacing, comments or docstrings must count as ONE program, or the
    diversity gate would credit cosmetic differences as branch diversity. Falls
    back to comment-stripped, whitespace-collapsed text when it does not parse.
    """
    if not code:
        return ""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, RecursionError):
        stripped = _DOCSTRING_RE.sub("", code)
        stripped = _COMMENT_RE.sub("", stripped)
        return " ".join(stripped.split())
    try:
        return ast.dump(_DocstringStripper().visit(tree), annotate_fields=False)
    except (ValueError, RecursionError):
        return " ".join(code.split())


def distinct_program_rate(codes: Sequence[str | None]) -> float:
    """Distinct normalized programs divided by the number of samples REQUESTED.

    The denominator is K, not the number that parsed: a model that returns two
    empty completions and one program has not given arm B four futures to choose
    between, and scoring it 1.0 would hide exactly the degeneration the
    diversity gate exists to catch.
    """
    if not codes:
        return 0.0
    usable = {normalize_program(c) for c in codes if c}
    usable.discard("")
    return len(usable) / len(codes)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    attempts: int = 4
    base_s: float = 2.0
    max_s: float = 45.0
    jitter: float = 0.3

    def sleep_s(self, attempt: int) -> float:
        raw = min(self.max_s, self.base_s * (2 ** (attempt - 1)))
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


@dataclass(frozen=True)
class BilledUsage:
    """Tokens a provider charged for an attempt that returned nothing usable.

    Travels with the exception so the retry wrapper can account for the attempt
    before it throws the response away.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    upstream: str = ""
    finish_reason: str = ""


class EmptyCompletion(RuntimeError):
    """The provider answered 200 with no content (typically truncated reasoning).

    The attempt was BILLED: the prompt was processed and the completion budget
    was usually spent in full on reasoning. ``billed`` carries that usage so a
    retry never silently deletes it from the run's token totals (~11% of k3
    calls come back empty).
    """

    def __init__(self, message: str, billed: BilledUsage | None = None) -> None:
        super().__init__(message)
        self.billed = billed or BilledUsage()


class CodexStreamError(RuntimeError):
    """The Codex SSE stream broke, errored mid-flight or never terminated.

    Retryable by classification (:data:`_RETRYABLE_NAMES`): the stream only
    exists once the request passed validation, so a failure here is transport
    or stream-integrity noise -- the same class as an empty-200, not a bad
    request. A bare RuntimeError made every one of them terminal, which then
    fed the provider tripwire noise it is explicitly not supposed to count.
    """


_RETRYABLE_NAMES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "APIStatusError",
        "ConnectError",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "TimeoutError",
        "IncompleteRead",
        "OpenAIResponseError",
        "EmptyCompletion",
        "CodexStreamError",
    }
)


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return type(exc).__name__ in _RETRYABLE_NAMES


# ---------------------------------------------------------------------------
# Provider tripwire (online): a dead provider must not burn a whole block
# ---------------------------------------------------------------------------

#: Consecutive TERMINAL failures on one provider that mean it is gone. Terminal =
#: the retry wrapper gave up, so transient 429/5xx/empty-200 noise that a retry
#: absorbs never reaches this counter. 30 is ~2 minutes of a 429 storm at Exp-2's
#: call rate and far outside anything the measured retry noise produces.
PROVIDER_DEAD_CONSECUTIVE = 30
#: ...or this long with ZERO successful calls provider-wide while calls are
#: failing. The Exp-3 codex round burned 2.6h on ~17k straight 429s before a human
#: noticed; five minutes is the longest silence worth paying for.
PROVIDER_DEAD_WINDOW_S = 300.0


class ProviderDead(RuntimeError):
    """A provider stopped answering: abort the block instead of measuring noise.

    Carries the trigger and the stats that fired it so the orchestrator can
    journal them, mark every affected cell INVALID_PROVIDER and exit nonzero with
    one line a human can act on.
    """

    def __init__(self, provider: str, trigger: str, stats: dict[str, Any]) -> None:
        self.provider = provider
        self.trigger = trigger
        self.stats = stats
        detail = (
            f"{stats.get('consecutive_failures')} consecutive terminal failures"
            if trigger == "consecutive_failures"
            else f"no successful call for {stats.get('silence_s')}s"
        )
        super().__init__(
            f"provider {provider!r} looks dead: {detail} "
            f"(successes={stats.get('successes')}, "
            f"terminal_failures={stats.get('failures')}, "
            f"retry_noise={stats.get('retry_noise')}); "
            f"last error: {stats.get('last_error')}"
        )


def provider_of(model: str) -> str:
    """Rate limits are per PROVIDER, not per model or per run.

    Total by construction: the tripwire keys provider health on this, and a model
    that mapped to something the orchestrator does not recognise would leave its
    siblings running against a dead quota. Every alias :func:`resolve_model`
    accepts (``codex:``, ``claude:``, ``or/``) MUST land on the same canonical
    provider here, or one seat's health would be filed under ``or`` while its
    sibling's goes to ``openrouter``.
    """
    if model.startswith("fake"):
        return "fake"  # the in-memory fake used by every dry gate
    try:
        return resolve_model(model).provider
    except ValueError:
        return model.split("/", 1)[0] or "unknown"


@dataclass
class ProviderHealth:
    """Process-global health of one provider, in the shape the tripwire needs.

    ``consecutive_failures`` counts only TERMINAL failures -- attempts that failed
    AFTER classification and after the retry policy gave up. k3's empty-200s are
    ~11% retryable noise, not death: they land in ``retry_noise`` and reset
    nothing. Any success resets the counter and the silence window, so a provider
    that is merely slow or lossy never trips.
    """

    provider: str
    successes: int = 0
    failures: int = 0
    retry_noise: int = 0
    consecutive_failures: int = 0
    first_attempt_ts: float = 0.0
    last_success_ts: float = 0.0
    last_failure_ts: float = 0.0
    last_error: str = ""
    dead: "ProviderDead | None" = None

    def _stamp(self, now: float) -> None:
        if not self.first_attempt_ts:
            self.first_attempt_ts = now

    def record_success(self, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._stamp(now)
        self.successes += 1
        self.consecutive_failures = 0
        self.last_success_ts = now

    def record_retry_noise(self, error: str, *, now: float | None = None) -> None:
        """A failed attempt the retry policy absorbed: recorded, never counted."""
        now = time.time() if now is None else now
        self._stamp(now)
        self.retry_noise += 1
        self.last_error = error[:500]

    def record_failure(self, error: str, *,
                       now: float | None = None) -> "ProviderDead | None":
        """A TERMINAL failure. Returns :class:`ProviderDead` when it trips.

        The silence window is only ever evaluated here -- on a failure -- so a
        genuinely idle provider (long ``/execute`` calls, between cells) can never
        be declared dead for being quiet.
        """
        now = time.time() if now is None else now
        self._stamp(now)
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure_ts = now
        self.last_error = error[:500]
        silent_since = self.last_success_ts or self.first_attempt_ts
        silence_s = max(0.0, now - silent_since)
        trigger = ""
        if self.consecutive_failures >= PROVIDER_DEAD_CONSECUTIVE:
            trigger = "consecutive_failures"
        elif silence_s >= PROVIDER_DEAD_WINDOW_S:
            trigger = "silence_window"
        if not trigger:
            return None
        if self.dead is None:
            self.dead = ProviderDead(self.provider, trigger,
                                     self.snapshot(silence_s=silence_s))
        return self.dead

    def snapshot(self, *, silence_s: float | None = None) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "successes": self.successes,
            "failures": self.failures,
            "retry_noise": self.retry_noise,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_limit": PROVIDER_DEAD_CONSECUTIVE,
            "silence_s": (round(silence_s, 1) if silence_s is not None else
                          round(max(0.0, time.time() - (self.last_success_ts
                                                        or self.first_attempt_ts)), 1)),
            "silence_limit_s": PROVIDER_DEAD_WINDOW_S,
            "last_error": self.last_error,
        }


_PROVIDER_HEALTH: dict[str, ProviderHealth] = {}
_PROVIDER_HEALTH_LOCK = threading.Lock()


def provider_health(provider: str) -> ProviderHealth:
    """The process-global health record for ``provider`` (created on first use).

    Global on purpose: the tripwire's question is "is this provider answering
    ANYONE", and in a parallel round three cells share one quota.
    """
    with _PROVIDER_HEALTH_LOCK:
        health = _PROVIDER_HEALTH.get(provider)
        if health is None:
            health = ProviderHealth(provider=provider)
            _PROVIDER_HEALTH[provider] = health
        return health


def provider_health_snapshot() -> dict[str, dict[str, Any]]:
    with _PROVIDER_HEALTH_LOCK:
        return {p: h.snapshot() for p, h in _PROVIDER_HEALTH.items()}


def reset_provider_health(provider: str | None = None) -> None:
    """Forget provider health (a fresh block, or a dry case that just killed one)."""
    with _PROVIDER_HEALTH_LOCK:
        if provider is None:
            _PROVIDER_HEALTH.clear()
        else:
            _PROVIDER_HEALTH.pop(provider, None)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class LLMClient:
    """Common sampling surface: ``sample()`` returns K response texts."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        journal: RunJournal | None = None,
        timings: TimingBuckets | None = None,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 8,
        #: Shared across clients to cap *provider* concurrency when many runs
        #: sample at once (rate limits are per provider, not per run).
        semaphore: asyncio.Semaphore | None = None,
        log_full_requests: bool = True,
    ) -> None:
        self.spec = spec
        self.journal = journal
        self.timings = timings
        self.retry = retry or RetryPolicy()
        self._sem = semaphore or asyncio.Semaphore(max_concurrency)
        self.log_full_requests = log_full_requests
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        #: Attempts the provider BILLED but that came back with no content and
        #: were retried away. Their tokens are folded into the counters above,
        #: so quota totals stay honest; this is how many calls paid for nothing.
        self.billed_empty_calls = 0
        self.retries = 0
        self.failures = 0

    # -- public API --------------------------------------------------------
    @property
    def model(self) -> str:
        return self.spec.key

    def model_info(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "provider": self.spec.provider,
            "api_model": self.spec.api_model,
            "temperature": self.spec.temperature,
            "supports_n": self.spec.supports_n,
            "max_tokens": self.spec.max_tokens,
            "temperature_locked": self.spec.temperature_locked,
            "notes": self.spec.notes,
            "routing": routing_notes(self.provider),
        }

    def usage(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "billed_empty_calls": self.billed_empty_calls,
            "retries": self.retries,
            "failures": self.failures,
        }

    async def sample(
        self,
        messages: Sequence[dict[str, Any]],
        n: int = 1,
        temperature: float | None = None,
        model: str | None = None,
        *,
        hints: Sequence[str] | None = None,
        branch: str = "",
        step: int | None = None,
    ) -> list[str]:
        """K-parallel sampling -> K response texts (empty string on failure)."""
        samples = await self.sample_detailed(
            messages, n=n, temperature=temperature, model=model,
            hints=hints, branch=branch, step=step,
        )
        return [s.text for s in samples]

    async def sample_detailed(
        self,
        messages: Sequence[dict[str, Any]],
        n: int = 1,
        temperature: float | None = None,
        model: str | None = None,
        *,
        hints: Sequence[str] | None = None,
        branch: str = "",
        step: int | None = None,
    ) -> list[Sample]:
        if model and model != self.spec.key:
            raise ValueError(
                f"client is bound to {self.spec.key!r}; got model={model!r}. "
                "Create a separate client per model (per-model decoding params "
                "are frozen across arms)."
            )
        if n < 1:
            raise ValueError("n must be >= 1")
        hint_list = list(hints) if hints else []
        if hint_list and len(hint_list) != n:
            raise ValueError(f"hints must have length n={n}, got {len(hint_list)}")

        t0 = time.monotonic()
        if self.spec.supports_n and not hint_list and n > 1:
            samples = await self._call_with_retry(
                messages, n=n, temperature=temperature, hint="",
                branch=branch, step=step,
            )
        else:
            tasks = [
                self._call_with_retry(
                    self._with_hint(messages, hint_list[i] if hint_list else ""),
                    n=1,
                    temperature=temperature,
                    hint=hint_list[i] if hint_list else "",
                    branch=f"{branch}#{i}" if branch else str(i),
                    step=step,
                )
                for i in range(n)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # A dead provider is NOT a per-seat sampling failure: swallowing it
            # here would hand every seat an empty program and let the block run
            # its full T against nothing. It propagates.
            dead = next((r for r in results if isinstance(r, ProviderDead)), None)
            if dead is not None:
                raise dead
            samples = []
            for i, res in enumerate(results):
                if isinstance(res, BaseException):
                    self.failures += 1
                    samples.append(
                        Sample(
                            text="",
                            code=None,
                            model=self.spec.key,
                            provider=self.spec.provider,
                            latency_s=time.monotonic() - t0,
                            hint=hint_list[i] if hint_list else "",
                            error=f"{type(res).__name__}: {res}",
                        )
                    )
                else:
                    samples.extend(res)
        t1 = time.monotonic()
        # One interval for the whole K-way gather: concurrent calls cost one
        # sampling round of wall clock, not K.
        if self.timings is not None:
            self.timings.record("llm_wait", t0, t1, f"{self.spec.key}:n={n}")
        for s in samples:
            # Failed samples never reached extraction in the retry wrapper.
            if s.code is None and s.text:
                s.code = extract_code(s.text)
        return samples

    def _with_hint(
        self, messages: Sequence[dict[str, Any]], hint: str
    ) -> list[dict[str, Any]]:
        msgs = list(messages)
        if hint:
            msgs = msgs + [{"role": "user", "content": HINT_TEMPLATE.format(hint=hint)}]
        return msgs

    # -- retry wrapper -----------------------------------------------------
    async def _call_with_retry(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        hint: str,
        branch: str,
        step: int | None,
    ) -> list[Sample]:
        request_id = uuid.uuid4().hex[:12]
        last_exc: BaseException | None = None
        # END-TO-END clock for the value the caller sees: what a seat actually
        # waited for, retries and backoff included. Per-attempt latency stays in
        # the journal records, so "slow provider" and "retried provider" remain
        # distinguishable in the evidence.
        started = time.monotonic()
        # Every provider call in every arm funnels through here, which makes this
        # the only honest place to judge whether the provider is still answering.
        health = provider_health(self.provider)
        for attempt in range(1, self.retry.attempts + 1):
            t0 = time.monotonic()
            try:
                async with self._sem:
                    samples = await self._generate(
                        messages, n=n, temperature=temperature, request_id=request_id
                    )
            except asyncio.CancelledError:
                raise  # a deadline, not the provider's fault
            except BaseException as exc:  # noqa: BLE001 - classified below
                latency = time.monotonic() - t0
                detail = f"{type(exc).__name__}: {exc}"
                # A billed attempt that returned nothing usable (empty-200) is
                # accounted BEFORE the retry throws it away: k3 comes back empty
                # on ~11% of calls having burned its whole completion budget on
                # reasoning, and dropping that would under-report the quota the
                # run really consumed.
                billed = getattr(exc, "billed", None)
                billed = billed if isinstance(billed, BilledUsage) else None
                if billed is not None:
                    self.billed_empty_calls += 1
                    self.prompt_tokens += billed.prompt_tokens
                    self.completion_tokens += billed.completion_tokens
                    self.reasoning_tokens += billed.reasoning_tokens
                    self.cache_read_tokens += billed.cache_read_tokens
                    self.cache_write_tokens += billed.cache_write_tokens
                self._log_call(
                    None, attempt, latency, messages, request_id, hint, branch, step,
                    outcome="error", error=detail[:1000], billed=billed,
                )
                last_exc = exc
                if attempt >= self.retry.attempts or not is_retryable(exc):
                    # TERMINAL: classified, retried as far as policy allows, still
                    # failed. This is the only class the tripwire counts -- an
                    # empty-200 that a retry absorbs is noise, an empty-200 that
                    # outlives every retry is the provider not answering.
                    dead = health.record_failure(detail)
                    if dead is not None:
                        self._log_provider_dead(dead, branch=branch, step=step)
                        raise dead from exc
                    raise
                self.retries += 1
                health.record_retry_noise(detail)
                await asyncio.sleep(self.retry.sleep_s(attempt))
                continue
            latency = time.monotonic() - t0
            health.record_success()
            for s in samples:
                s.attempts = attempt
                s.hint = hint
                s.request_id = request_id
                # End-to-end: every retry and every backoff the caller paid for.
                s.latency_s = time.monotonic() - started
                # Extract BEFORE journaling: code_chars in the journal is the
                # evidence that a program actually came back.
                s.code = extract_code(s.text) if s.text else None
                self.calls += 1
                self.prompt_tokens += s.prompt_tokens
                self.completion_tokens += s.completion_tokens
                self.reasoning_tokens += s.reasoning_tokens
                self.cache_read_tokens += s.cache_read_tokens
                self.cache_write_tokens += s.cache_write_tokens
                self._log_call(
                    s, attempt, latency, messages, request_id, hint, branch, step,
                )
            return samples
        assert last_exc is not None
        raise last_exc

    @property
    def provider(self) -> str:
        """Tripwire key: the quota this client shares with every other cell."""
        return self.spec.provider or provider_of(self.spec.key)

    def _log_provider_dead(self, dead: ProviderDead, *, branch: str = "",
                           step: int | None = None) -> None:
        if self.journal is None:
            return
        self.journal.write("provider_dead", provider=dead.provider,
                           trigger=dead.trigger, model=self.spec.key,
                           branch=branch, step=step, detail=str(dead),
                           # NESTED, not splatted: the snapshot carries its own
                           # "provider" key and a duplicate keyword would raise a
                           # TypeError inside the handler -- swallowing the very
                           # ProviderDead it is trying to record.
                           stats=dead.stats)

    def _log_call(
        self,
        sample: Sample | None,
        attempt: int,
        latency: float,
        messages: Sequence[dict[str, Any]],
        request_id: str,
        hint: str,
        branch: str,
        step: int | None,
        outcome: str = "ok",
        error: str = "",
        #: Usage for a FAILED attempt the provider still billed (empty-200).
        billed: BilledUsage | None = None,
    ) -> None:
        if self.journal is None:
            return
        usage = sample.billed_usage() if sample is not None else (
            billed or BilledUsage()
        )
        self.journal.llm_call(
            model=self.spec.key,
            provider=self.spec.provider,
            attempt=attempt,
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            upstream=usage.upstream,
            n_messages=len(messages),
            temperature=self.spec.temperature,
            request_id=request_id,
            branch=branch,
            step=step,
            outcome=outcome,
            error=error,
            response_chars=len(sample.text) if sample else 0,
            code_chars=len(sample.code or "") if sample else 0,
            finish_reason=usage.finish_reason,
            hint=hint,
            request=_request_digest(messages) if self.log_full_requests else None,
            response_text=(
                sample.text[:20000] if sample and self.log_full_requests else ""
            ),
        )

    # -- provider hook -----------------------------------------------------
    async def _generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        request_id: str,
    ) -> list[Sample]:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


def _request_digest(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full request logging, capped per message so journals stay readable."""
    return [
        {
            "role": m.get("role", "?"),
            "chars": len(str(m.get("content", ""))),
            "content": str(m.get("content", ""))[:8000],
        }
        for m in messages
    ]


class KimiClient(LLMClient):
    """Direct Kimi API (OpenAI-compatible chat completions)."""

    def __init__(self, spec: ModelSpec, *, api_key: str, base_url: str,
                 timeout_s: float = 300.0, **kw: Any) -> None:
        super().__init__(spec, **kw)
        from openai import AsyncOpenAI

        self.base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0
        )

    async def _generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        request_id: str,
    ) -> list[Sample]:
        # Provider-locked decoding: temperature must be exactly spec.temperature.
        kwargs: dict[str, Any] = {
            "model": self.spec.api_model,
            "messages": list(messages),
        }
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        if self.spec.max_tokens:
            kwargs["max_tokens"] = self.spec.max_tokens
        if n > 1 and self.spec.supports_n:
            kwargs["n"] = n
        resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        choices = resp.choices or []
        share = max(1, len(choices))
        out: list[Sample] = []
        for choice in choices:
            content = choice.message.content or ""
            if isinstance(content, list):  # defensive: structured content
                content = "\n".join(getattr(p, "text", "") or "" for p in content)
            out.append(
                Sample(
                    text=content,
                    code=None,
                    model=self.spec.key,
                    provider="kimi",
                    latency_s=0.0,
                    prompt_tokens=prompt_tokens // share,
                    completion_tokens=completion_tokens // share,
                    reasoning_tokens=reasoning // share,
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                    request_id=request_id,
                )
            )
        if not any(s.text.strip() for s in out):
            # 200 with no content: k3 spends the whole token budget on reasoning
            # and returns nothing. Retryable -- a silent empty program would
            # otherwise be scored as a step the agent chose to waste.
            raise EmptyCompletion(
                f"{self.spec.key} returned no content "
                f"(finish_reason={[s.finish_reason for s in out]}, "
                f"completion_tokens={completion_tokens}, reasoning={reasoning}, "
                f"max_tokens={self.spec.max_tokens})",
                # Whole-call usage, not the per-choice share: the provider
                # billed this request once.
                BilledUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning,
                    finish_reason=",".join(
                        sorted({s.finish_reason for s in out if s.finish_reason})
                    ),
                ),
            )
        return out

    async def aclose(self) -> None:
        await self._client.close()


class _CodexCredentialCache:
    """Process-wide, single-flight credential cache.

    Mirrors ``provider._CredentialCache``: refresh tokens rotate on use, so
    concurrent samples must not each kick off their own refresh.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._credentials: Any = None

    async def get(self) -> Any:
        from fle.eval.inspect.codex.auth import ensure_credentials

        async with self._lock:
            if self._credentials is None or self._credentials.is_expired():
                self._credentials = await ensure_credentials()
            return self._credentials


_CODEX_CACHE = _CodexCredentialCache()


def _codex_auth() -> Any:
    import httpx

    class _CodexOAuth(httpx.Auth):
        async def async_auth_flow(self, request: Any):  # type: ignore[override]
            credentials = await _CODEX_CACHE.get()
            request.headers["Authorization"] = f"Bearer {credentials.access_token}"
            request.headers["chatgpt-account-id"] = credentials.account_id
            yield request

    return _CodexOAuth()


class CodexClient(LLMClient):
    """ChatGPT-subscription Codex backend (streaming Responses API).

    Auth/token loading is the repo's own (``fle.eval.inspect.codex.auth``), so
    ``fle codex login`` / the Codex CLI file / codex-proxy all keep working and
    rotated refresh tokens are written back exactly once.
    """

    def __init__(self, spec: ModelSpec, *, timeout_s: float = 600.0,
                 reasoning_effort: str | None = "medium", **kw: Any) -> None:
        super().__init__(spec, **kw)
        import httpx
        from openai import AsyncOpenAI

        self.reasoning_effort = reasoning_effort
        self.session_id = str(uuid.uuid4())
        self._http = httpx.AsyncClient(auth=_codex_auth(), timeout=timeout_s)
        self._client = AsyncOpenAI(
            api_key=CODEX_PLACEHOLDER_KEY,
            base_url=CODEX_BASE_URL,
            http_client=self._http,
            max_retries=0,
            default_headers={
                "OpenAI-Beta": "responses=experimental",
                "originator": "codex_cli_rs",
                "session_id": self.session_id,
            },
        )

    async def _generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        request_id: str,
    ) -> list[Sample]:
        instructions, items = _split_for_responses(messages)
        kwargs: dict[str, Any] = {
            "model": self.spec.api_model,
            "input": items,
            # store=False and stream=True are non-negotiable for this backend.
            "store": False,
            "stream": True,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        stream = await self._client.responses.create(**kwargs)
        text, usage, status = await _collect_stream(stream)
        if not text.strip():
            raise EmptyCompletion(
                f"{self.spec.key} returned no output text (status={status}, "
                f"output_tokens={usage['output_tokens']}, "
                f"reasoning={usage['reasoning_tokens']})",
                BilledUsage(
                    prompt_tokens=usage["input_tokens"],
                    completion_tokens=usage["output_tokens"],
                    reasoning_tokens=usage["reasoning_tokens"],
                    finish_reason=status,
                ),
            )
        return [
            Sample(
                text=text,
                code=None,
                model=self.spec.key,
                provider="codex",
                latency_s=0.0,
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                finish_reason=status,
                request_id=request_id,
            )
        ]

    async def aclose(self) -> None:
        await self._http.aclose()


def _split_for_responses(
    messages: Sequence[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """System turns -> ``instructions``; the rest -> Responses input items."""
    system_parts: list[str] = []
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            system_parts.append(content)
        else:
            items.append({"role": role, "content": content})
    return "\n\n".join(system_parts), items


_TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


async def _collect_stream(stream: Any) -> tuple[str, dict[str, int], str]:
    """Rebuild text, usage and status from the Codex SSE stream.

    The terminal event reports an empty ``output`` array on this backend, so
    finished items are gathered from ``response.output_item.done`` (same
    workaround as ``provider._collect_streamed_response``).
    """
    items: list[Any] = []
    response: Any = None
    async for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_item.done":
            items.append(event.item)
        elif etype in _TERMINAL_EVENTS:
            response = event.response
        elif etype == "error":
            # Mid-stream failure of a request the backend already accepted:
            # transport noise, retryable (a bare RuntimeError made it terminal
            # and fed the tripwire failures it must not count).
            raise CodexStreamError(
                f"codex stream error {getattr(event, 'code', '')}: "
                f"{getattr(event, 'message', '')}"
            )
    if response is None:
        raise CodexStreamError(
            "codex stream ended without a terminal response event"
        )
    if getattr(response, "error", None) is not None:
        err = response.error
        raise CodexStreamError(f"codex response error {err.code}: {err.message}")
    if not getattr(response, "output", None):
        response.output = items
    texts: list[str] = []
    for item in response.output or items:
        if getattr(item, "type", "") == "reasoning":
            continue
        for part in getattr(item, "content", None) or []:
            t = getattr(part, "text", None)
            if t:
                texts.append(t)
    usage_obj = getattr(response, "usage", None)
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    if usage_obj is not None:
        usage["input_tokens"] = getattr(usage_obj, "input_tokens", 0) or 0
        usage["output_tokens"] = getattr(usage_obj, "output_tokens", 0) or 0
        details = getattr(usage_obj, "output_tokens_details", None)
        usage["reasoning_tokens"] = (
            (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        )
    status = str(getattr(response, "status", "") or "")
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is not None and getattr(incomplete, "reason", None):
        status = f"{status}:{incomplete.reason}"
    return "\n".join(texts), usage, status

# ---------------------------------------------------------------------------
# Claude (Anthropic Messages API, Pro/Max subscription OAuth)
# ---------------------------------------------------------------------------


class _ClaudeCredentialCache:
    """Process-wide, single-flight credential cache.

    Mirrors ``_CodexCredentialCache``: refresh tokens rotate on use, so
    concurrent samples must not each kick off their own refresh.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._credentials: Any = None

    async def get(self) -> Any:
        from fle.eval.inspect.claude.auth import ensure_credentials

        async with self._lock:
            if self._credentials is None or self._credentials.is_expired():
                self._credentials = await ensure_credentials()
            return self._credentials


_CLAUDE_CACHE = _ClaudeCredentialCache()


def _claude_auth() -> Any:
    import httpx

    class _ClaudeOAuth(httpx.Auth):
        async def async_auth_flow(self, request: Any):  # type: ignore[override]
            credentials = await _CLAUDE_CACHE.get()
            request.headers["Authorization"] = f"Bearer {credentials.access_token}"
            # The API rejects requests carrying both header styles.
            if "x-api-key" in request.headers:
                del request.headers["x-api-key"]
            yield request

    return _ClaudeOAuth()


def _split_for_anthropic(
    messages: Sequence[dict[str, Any]],
    *,
    spoof: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """System turns -> ``system`` blocks; the rest -> alternating Messages turns.

    Subscription tokens are only authorized for Claude Code, and the API checks
    the FIRST system block for its identity -- the bench's own system prompt
    follows as later blocks, never replaces it. Consecutive same-role turns are
    merged (branch hints append a second user turn, which the Messages API
    would otherwise 400 on).

    TWO cache breakpoints (Anthropic allows 4):
    - last system block: the ~117KB FLE system prompt is byte-identical across
      every seat, branch and step of a run;
    - last conversation turn (rotating): Anthropic's incremental-caching
      pattern -- each request extends the cached prefix by one exchange, so the
      GROWING history is read from cache instead of re-billed in full every
      step. Measured without this: a Control cell's ~64k prompts cached only
      the 33k system share (25-27%) and burned a whole 5h subscription window
      by ~40% of T.
    """
    system_blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PROMPT}] if spoof else []
    )
    turns: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        role = msg.get("role", "user")
        if role == "system":
            system_blocks.append({"type": "text", "text": content})
        elif turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    if system_blocks:
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
    if turns:
        last = turns[-1]
        last["content"] = [{
            "type": "text",
            "text": last["content"],
            "cache_control": {"type": "ephemeral"},
        }]
    return system_blocks, turns


class ClaudeClient(LLMClient):
    """Claude Pro/Max subscription (Anthropic Messages API, OAuth Bearer).

    Auth/token loading is the repo's own (``fle.eval.inspect.claude.auth``), so
    ``fle claude login`` / Claude Code's credentials file both keep working and
    rotated refresh tokens are written back exactly once. Request shaping
    mirrors ``fle.eval.inspect.claude.provider``: OAuth betas + Claude Code
    identity headers, spoofed first system block.
    """

    #: Subclasses that talk to plain Anthropic-compatible endpoints (no OAuth)
    #: set this False: the Claude Code identity block is an OAuth requirement,
    #: not part of the Messages API.
    _spoof = True

    def __init__(self, spec: ModelSpec, *, timeout_s: float = 600.0,
                 **kw: Any) -> None:
        super().__init__(spec, **kw)
        import httpx
        from anthropic import AsyncAnthropic

        self._http = httpx.AsyncClient(auth=_claude_auth(), timeout=timeout_s)
        self._client = AsyncAnthropic(
            base_url=CLAUDE_BASE_URL,
            auth_token=CLAUDE_PLACEHOLDER_TOKEN,
            http_client=self._http,
            max_retries=0,
            default_headers={
                "anthropic-beta": CLAUDE_OAUTH_BETAS,
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
            },
        )

    async def _generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        request_id: str,
    ) -> list[Sample]:
        system_blocks, turns = _split_for_anthropic(messages, spoof=self._spoof)
        # STREAMED, then folded back into one Message: long thinking
        # generations exceed proxy idle timeouts on non-streaming requests
        # (measured: Cloudflare 524s on a self-hosted endpoint at >100s), and
        # Anthropic itself requires streaming for long requests. The final
        # message is identical to the non-streaming shape, so nothing below
        # changes.
        async with self._client.messages.stream(
            model=self.spec.api_model,
            system=system_blocks or None,
            messages=turns,
            max_tokens=self.spec.max_tokens or 16384,
            # Provider-frozen decoding, same policy as Kimi: the spec value is
            # the one sent, never the caller's.
            temperature=self.spec.temperature,
        ) as stream:
            resp = await stream.get_final_message()
        text = "\n".join(
            block.text
            for block in resp.content or []
            if getattr(block, "type", "") == "text" and block.text
        )
        usage = resp.usage
        # Quota-honest accounting: cache reads and writes are still context the
        # provider processed for this call.
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        prompt_tokens = (usage.input_tokens or 0) + cache_write + cache_read
        completion_tokens = usage.output_tokens or 0
        stop_reason = str(resp.stop_reason or "")
        if not text.strip():
            raise EmptyCompletion(
                f"{self.spec.key} returned no text content "
                f"(stop_reason={stop_reason}, output_tokens={completion_tokens}, "
                f"max_tokens={self.spec.max_tokens})",
                BilledUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                    finish_reason=stop_reason,
                ),
            )
        return [
            Sample(
                text=text,
                code=None,
                model=self.spec.key,
                provider=self.spec.provider,
                latency_s=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=0,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                finish_reason=stop_reason,
                request_id=request_id,
            )
        ]

    async def aclose(self) -> None:
        await self._http.aclose()


class LocalAnthropicClient(ClaudeClient):
    """Self-hosted Anthropic-Messages-compatible endpoint, plain API key.

    Endpoint + key come from LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY. No OAuth,
    no Claude Code identity spoof; thinking blocks in responses are skipped by
    the text-block filter in :meth:`ClaudeClient._generate`. cache_control
    breakpoints are sent (measured: the endpoint tolerates them; a server
    without a prompt cache just ignores them and reports zero cache tokens).
    """

    _spoof = False

    def __init__(self, spec: ModelSpec, *, base_url: str, api_key: str,
                 timeout_s: float = 600.0, **kw: Any) -> None:
        LLMClient.__init__(self, spec, **kw)
        import httpx
        from anthropic import AsyncAnthropic

        self._http = httpx.AsyncClient(timeout=timeout_s)
        self._client = AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
            http_client=self._http,
            max_retries=0,
        )



# ---------------------------------------------------------------------------
# OpenRouter (metered aggregator, OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(LLMClient):
    """OpenRouter chat completions (metered; any ``openrouter/<vendor>/<m>``).

    Deviations that are deliberate and journaled:
    - Middle-out compression is disabled EXPLICITLY on every request
      (``transforms: []``). OpenRouter compresses the middle of an
      over-long prompt by DEFAULT, so merely omitting the key would make
      ROUTING_NOTES' ``middle_out=False`` a false claim about this route.
    - Seats within one cell must not straddle differently-QUANTIZED
      deployments mid-run. OPENROUTER_QUANTIZATIONS pins EXACTLY ONE
      quantization (more than one is rejected: it re-admits the straddle the
      filter exists to forbid) as a routing filter with fallbacks ON;
      OPENROUTER_PROVIDER is then a preference order. Order WITHOUT a
      quantization filter is a strict pin (fallbacks disabled) -- a measured
      SPOF: one upstream's shared-pool TPM starved 17 seats. The upstream
      that served each call is journaled either way (``upstream``).
    - anthropic/* models get an explicit ``cache_control`` breakpoint on the
      system turn; other vendors cache implicitly. Cached tokens are read from
      ``usage.prompt_tokens_details.cached_tokens`` and folded INTO
      prompt_tokens (quota-honest), journaled separately.
    """

    def __init__(self, spec: ModelSpec, *, api_key: str,
                 timeout_s: float = 600.0,
                 provider_order: tuple[str, ...] = (),
                 quantizations: tuple[str, ...] = (), **kw: Any) -> None:
        super().__init__(spec, **kw)
        if len(quantizations) > 1:
            raise ValueError(
                "OPENROUTER_QUANTIZATIONS takes exactly one quantization, got "
                f"{list(quantizations)}: several values put the seats of one "
                "cell on differently-quantized deployments, which is precisely "
                "the straddle the filter exists to prevent."
            )
        self.quantizations = quantizations
        from openai import AsyncOpenAI

        self.provider_order = provider_order
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=OPENROUTER_BASE_URL,
            timeout=timeout_s, max_retries=0,
        )

    def _sort(self) -> str:
        # Default routing is price-sorted, which lands on the SLOWEST cheap
        # host once the cheapest is saturated (measured: 373s/call on
        # OpenInference vs 90s on Baidu). Sorting by throughput keeps a
        # latency-starved seat from wasting the build clock.
        return os.environ.get("OPENROUTER_SORT", "throughput")

    def model_info(self) -> dict[str, Any]:
        """OpenRouter's routing note plus the knobs this client really sends."""
        info = super().model_info()
        info["routing"] = {
            **info["routing"],
            "transforms": [],
            "quantizations": list(self.quantizations),
            "provider_order": list(self.provider_order),
            "allow_fallbacks": (
                bool(self.quantizations) if self.provider_order else True
            ),
            "sort": self._sort() if self.quantizations else "",
        }
        return info

    def _system_content(self, text: str) -> Any:
        if self.spec.api_model.startswith("anthropic/"):
            return [{"type": "text", "text": text,
                     "cache_control": {"type": "ephemeral"}}]
        return text

    async def _generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        request_id: str,
    ) -> list[Sample]:
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                msgs.append({"role": "system",
                             "content": self._system_content(m.get("content") or "")})
            else:
                msgs.append({"role": m.get("role", "user"),
                             "content": m.get("content") or ""})
        # transforms=[] is the EXPLICIT middle-out opt-out: OpenRouter would
        # otherwise compress over-long prompts for us and silently change what
        # the model saw (ROUTING_NOTES claims middle_out=False for real).
        extra_body: dict[str, Any] = {"usage": {"include": True}, "transforms": []}
        # Routing policy (design: seats must not straddle differently-
        # quantized deployments). A quantization filter enforces that
        # invariant directly and leaves fallbacks ON, so one upstream's
        # shared-pool TPM limit cannot starve the round (measured: Baidu
        # alone 429s at 17 seats). A bare order pin (no quantizations)
        # keeps the strict single-upstream behavior.
        provider_body: dict[str, Any] = {}
        if self.quantizations:
            provider_body["quantizations"] = list(self.quantizations)
            provider_body["sort"] = self._sort()
        if self.provider_order:
            provider_body["order"] = list(self.provider_order)
            provider_body["allow_fallbacks"] = bool(self.quantizations)
        if provider_body:
            extra_body["provider"] = provider_body
        resp = await self._client.chat.completions.create(
            model=self.spec.api_model,
            messages=msgs,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
            extra_body=extra_body,
        )
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        pd = getattr(usage, "prompt_tokens_details", None)
        cache_read = (getattr(pd, "cached_tokens", 0) or 0) if pd else 0
        cd = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(cd, "reasoning_tokens", 0) or 0) if cd else 0
        upstream = str(getattr(resp, "provider", "") or "")
        choice = (resp.choices or [None])[0]
        content = (choice.message.content or "") if choice else ""
        if isinstance(content, list):  # defensive: structured content
            content = "\n".join(getattr(p, "text", "") or "" for p in content)
        finish = (getattr(choice, "finish_reason", "") or "") if choice else ""
        if not content.strip():
            raise EmptyCompletion(
                f"{self.spec.key} returned no content (finish_reason={finish}, "
                f"completion_tokens={completion_tokens}, reasoning={reasoning}, "
                f"upstream={upstream or '?'})",
                BilledUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning,
                    cache_read_tokens=cache_read,
                    upstream=upstream,
                    finish_reason=finish,
                ),
            )
        return [
            Sample(
                text=content,
                code=None,
                model=self.spec.key,
                provider="openrouter",
                latency_s=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning,
                cache_read_tokens=cache_read,
                upstream=upstream,
                finish_reason=finish,
                request_id=request_id,
            )
        ]

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENV_LOADED = False


def load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv ships with the repo
        pass
    _ENV_LOADED = True


def make_client(
    model: str,
    *,
    journal: RunJournal | None = None,
    timings: TimingBuckets | None = None,
    retry: RetryPolicy | None = None,
    max_concurrency: int = 8,
    semaphore: asyncio.Semaphore | None = None,
    timeout_s: float | None = None,
    reasoning_effort: str | None = "medium",
    log_full_requests: bool = True,
) -> LLMClient:
    """Build the client: ``k3``/``kimi-*``, ``codex/``, ``claude/`` or ``openrouter/``."""
    load_env()
    spec = resolve_model(model)
    common: dict[str, Any] = {
        "journal": journal,
        "timings": timings,
        "retry": retry,
        "max_concurrency": max_concurrency,
        "semaphore": semaphore,
        "log_full_requests": log_full_requests,
    }
    if spec.provider == "kimi":
        api_key = os.environ.get("KIMI_API_KEY")
        base_url = os.environ.get("KIMI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("KIMI_API_KEY / KIMI_BASE_URL missing (expected in .env)")
        return KimiClient(
            spec,
            api_key=api_key.strip().strip('"').strip("'"),
            base_url=base_url.strip().strip('"').strip("'"),
            timeout_s=timeout_s or 300.0,
            **common,
        )
    if spec.provider == "claude":
        return ClaudeClient(spec, timeout_s=timeout_s or 600.0, **common)
    if spec.provider == "local":
        base_url = os.environ.get("LOCAL_LLM_BASE_URL")
        api_key = os.environ.get("LOCAL_LLM_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY missing (expected in .env)"
            )
        return LocalAnthropicClient(
            spec,
            base_url=base_url.strip().strip('"').strip("'"),
            api_key=api_key.strip().strip('"').strip("'"),
            timeout_s=timeout_s or 600.0,
            **common,
        )
    if spec.provider == "openrouter":
        api_key = (os.environ.get("OPENROUTER_API_KEY")
                   or os.environ.get("OPENROUTER_KEY"))
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY / OPENROUTER_KEY missing (expected in .env)"
            )
        order = tuple(
            p.strip() for p in os.environ.get("OPENROUTER_PROVIDER", "").split(",")
            if p.strip()
        )
        quants = tuple(
            q.strip() for q in os.environ.get(
                "OPENROUTER_QUANTIZATIONS", ""
            ).split(",")
            if q.strip()
        )
        # A single pinned upstream (order without quantizations) is a SPOF by
        # design and even a quantization pool can burst-429; the retry policy
        # absorbs those (up to ~4 min of backoff) so transient storms never
        # reach the tripwire's 30-consecutive-terminal budget.
        or_common = dict(common)
        or_common["retry"] = retry or RetryPolicy(
            attempts=6, base_s=4.0, max_s=90.0
        )
        return OpenRouterClient(
            spec,
            api_key=api_key.strip().strip('"').strip("'"),
            provider_order=order,
            quantizations=quants,
            timeout_s=timeout_s or 600.0,
            **or_common,
        )
    return CodexClient(
        spec, timeout_s=timeout_s or 600.0, reasoning_effort=reasoning_effort, **common
    )


async def smoke(
    models: Iterable[str] = DEFAULT_MODELS, prompt: str = "Reply with exactly: OK"
) -> dict[str, Any]:
    """One tiny completion per model; returns latencies and verbatim errors."""
    results: dict[str, Any] = {}
    for name in models:
        entry: dict[str, Any] = {"model": name}
        client: LLMClient | None = None
        t0 = time.monotonic()
        try:
            client = make_client(name, log_full_requests=False)
            entry["info"] = client.model_info()
            samples = await client.sample_detailed(
                [{"role": "user", "content": prompt}], n=1
            )
            s = samples[0]
            entry.update(
                ok=s.ok,
                latency_s=round(s.latency_s, 3),
                text=s.text[:200],
                prompt_tokens=s.prompt_tokens,
                completion_tokens=s.completion_tokens,
                reasoning_tokens=s.reasoning_tokens,
                error=s.error,
            )
        except BaseException as exc:  # noqa: BLE001 - verbatim error is the record
            entry.update(
                ok=False,
                latency_s=round(time.monotonic() - t0, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if client is not None:
                await client.aclose()
        results[name] = entry
    return results


if __name__ == "__main__":  # pragma: no cover - operational entry point
    import argparse
    import json

    ap = argparse.ArgumentParser(description="LLM client smoke test")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="bench/results/llm_smoke.json")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    res = asyncio.run(smoke(models))
    payload = {
        "ts": time.time(),
        # Per resolved provider, not one global claim: a smoke run of
        # openrouter/* models must not ship "no aggregator" in its own artifact.
        "routing": {
            p: routing_notes(p) for p in sorted({provider_of(m) for m in models})
        },
        "results": res,
    }
    atomic_write_json(args.out, payload, indent=2)
    print(json.dumps(payload, indent=2))
