"""Tier-1 orchestrator: runs the (arm x model x task x replicate) matrix.

Responsibilities that belong here and nowhere else:

* Expand the frozen pilot config into cells and schedule them concurrently
  under BOTH caps that Tier 0 produces: a run-count cap per node and a total
  sandbox-slot budget (peak concurrent sandboxes differ per arm).
* Keep all four arms of one (task, replicate) cell close together in time, so
  provider latency drifts hit every arm equally (cheap latency blocking).
* Journal everything per run and write partial results after every completion:
  an interrupted pilot must still be analysable.
* Reap on exit. TTL expiry HIBERNATES sandboxes, it does not clean up, so the
  prefix-scoped reaper is the only guarantee that nothing is left running.
* Pass the SEED snapshot through to every arm: ``--template-snap`` for a
  greenfield matrix, ``--from-snap`` (the very same option) when the cells must
  start from a baked checkpoint. A×K-from-S is A×K with
  ``template_snap=<checkpoint>`` and no other difference.
* Exit codes -- the block's contract with any wrapper script: 0 ONLY for a
  complete matrix, 2 when a provider died mid-block (partial results written,
  every affected cell on ``needs_rerun``), 1 for anything else incomplete --
  failed cells, cells the graceful stop never admitted, an interrupt, or an
  exception no cell accounted for.

It never invents its own endpoint: the number per run is the terminal fixed
60s-window probe produced by :mod:`bench.arms`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from bench.arms import (
    FakeRateLimit,
    EXP2_DEFAULT_M,
    EXP2_K,
    EXP3_K,
    EXP3_P_S,
    EXP3_WIDTH_FLOOR,
    LEASE_GUARD_SLACK_S,
    ArmConfig,
    FakeBridge,
    FakeLLM,
    claim_dry_journal_dir,
    fake_substrate,
    run_one,
)
from bench.common import RunJournal, atomic_write_json
from bench.llm import (
    PROVIDER_DEAD_CONSECUTIVE,
    PROVIDER_DEAD_WINDOW_S,
    ProviderDead,
    RetryPolicy,
    provider_health_snapshot,
    provider_of,
    reset_provider_health,
)
from bench.tier05 import Tier05Config, load_tier0_caps, peak_sandboxes

# v2.7: arm C (GameState-restore branching) disabled by default — GameState is
# FLE-specific plumbing; the benchmark targets Farplane capabilities that exist
# for ANY software. Re-enable explicitly via --arms A,AxK,B,C if ever needed.
DEFAULT_ARMS = ("A", "AxK", "B")

#: The greenfield template EVERY checkpoint descends from (Tier 0, settled
#: input; the same id as ``bench.exp1.TEMPLATE_SNAP``). It is NOT the run seed:
#: ``--from-snap`` sets the seed (Exp 2's S2), and the reaper's keep set must
#: hold BOTH -- a between-block sweep already ate the bake sandbox once because
#: substrate was not held out explicitly.
TEMPLATE_SNAP = "snapshot-5fa7769473a710b2"

#: Exp 2 runs every arm at the same hard wall clock. Raised 3600 -> 4200s when
#: admission moved to the fork p95: at 3600s (3510s of rollout) only ONE round
#: is guaranteed against the ~1781s p95 round estimate, and the design's
#: primary contrast needs a measured dose floor of >= 2 re-convergences.
#: 4200s - 90s reserve = 4110s -> 2 rounds guaranteed, ~3 at the p50 wave.
EXP2_T_S = 4200.0
#: Pre-registered validity floor: a B-iterated endpoint counts toward the
#: primary contrast only if its journal shows at least this many convergences.
EXP2_DOSE_FLOOR = 2

#: Exp 2's block, EXACTLY as the design pre-registers it: 8 runs, S2 + k3 only.
#: 3 x B-iterated and 3 x A×K-from-S interleaved as PAIRS FIRST (the primary
#: contrast is paired within checkpoint, so a truncated block still yields whole
#: pairs), then the two descriptive curve points. The order is also the drop
#: order read backwards: if the S2 clock runs short, truncating from the end
#: drops A-continue first, then B-once, and never a primary pair.
EXP2_BLOCK: tuple[tuple[str, int], ...] = (
    ("B", 1), ("AxK", 1),
    ("B", 2), ("AxK", 2),
    ("B", 3), ("AxK", 3),
    ("Bonce", 1),
    ("A", 1),
)

#: Exp 3's block, restructured (user-directed, pre-registered before round 1):
#: SIX runs = a THREE-ARM LADDER x two ROUNDS, one of each arm per round, ONE
#: checkpoint (S2B) and ONE model (codex). The ladder isolates one ingredient per
#: rung -- Control -> A×K-S is the value of forking wide, A×K-S -> Hybrid the
#: value of one convergence -- and the rounds exist so nothing commits 10h up
#: front: round 1 is ~7.7h, its results are reviewed, and round 2 runs only on an
#: explicit go (``--round 2``).
#: Within a round the order is Hybrid, A×K-S, Control: the decisive pair sits
#: adjacent in time, so time-correlated drift (provider latency, node contention)
#: cannot load onto one of the two arms being compared, and a mid-round cutoff
#: drops the rung that needs its neighbours least.
EXP3_BLOCK: tuple[tuple[str, int], ...] = (
    ("Hybrid", 1), ("AxK-S", 1), ("Control", 1),
    ("Hybrid", 2), ("AxK-S", 2), ("Control", 2),
)
#: The rungs of one Exp-3 round, bottom to top. Reporting order, not run order.
EXP3_LADDER = ("Control", "AxK-S", "Hybrid")
#: Exp 3's horizon: EVERY arm gets T_total = 2P of BUILD time (the pre-registered
#: halftime accounting -- Hybrid's regroup is extra wall clock, measured and
#: reported, never taken out of a leg). n=2 rounds are DIRECTIONAL/exploratory:
#: reported with per-seat distributions, never as a significance claim.
EXP3_T_S = 2 * EXP3_P_S

# --- parallel-round mode (pre-registered option) ----------------------------
#: Peak concurrent sandboxes when a whole round runs at once: A×K-S 8 + Hybrid 8
#: (its refork wave starts only after the 7 losers are deleted, so 8 is its
#: peak) + Control 1. The slot pool is sized for exactly this; anything smaller
#: would serialise the round it is meant to parallelise.
EXP3_PARALLEL_SLOTS = 2 * EXP3_K + 1
#: In-flight codex calls the 16 fan-out seats SHARE in parallel mode: the level
#: Exp 2 sustained (8 seats, one call each). It is a cap on the provider, not on
#: an arm, so both fan-out arms are throttled identically and the within-round
#: contrast stays fair.
EXP3_PROVIDER_CONCURRENCY = 8
#: Pre-registered EXEMPTION from that shared cap. A single redundancy-free line
#: loses far more from a halved step rate than a best-of-8 population does, so
#: throttling Control would inflate the very fork-value rung it exists to
#: measure. Its private gate costs ~6% extra provider load. Stated bias: the
#: exemption instead DEFLATES the Control -> A×K-S gap, which is the conservative
#: direction -- A×K-S beating an unthrottled Control is strong evidence for
#: forking; a Control win in parallel mode is weak evidence against it.
EXP3_PROVIDER_EXEMPT_ARMS = ("Control",)
#: Margin the sandbox lease keeps OVER the build clock. Round 1 died on this: the
#: default 7200s lease expired inside a ~8700s round, every seat hibernated, and
#: both surviving cells came back PARTIAL with no terminal probe.
#: Sized from what actually happens OUTSIDE T, on the FIRST seat created (whose
#: lease starts earliest and therefore expires soonest):
#:   provisioning of all K seats  ~8 x 60s p95 create     =  480s
#:   Hybrid's halftime regroup    10.1s snap + 7 x 151.6s fork p95 + 8s = 1090s
#:   ladder provisioning stagger  (parallel mode, widest rung)          =  240s
#:   terminal probes + drain + teardown                                 =  120s
#: ~1930s at the tail of every distribution at once, so the margin is set to a
#: round hour above it. TTL is never a cleanup mechanism -- deletion is always
#: explicit and the reaper is the guarantee -- so an over-long lease costs
#: nothing, while a short one costs the whole cell.
EXP3_TTL_MARGIN_S = 3600.0
#: Poll budget for ONE ``create_from_snapshot`` in this block. The wrapper's 300s
#: default is sized for a solo create; a parallel round asks for 17 at once and
#: they queue on the warm-slot lane, which is how round 1 lost Hybrid's 8th seat.
EXP3_CREATE_DEADLINE_S = 600.0
#: Per-arm provisioning offset in a parallel round, applied in LADDER order
#: (Control, then A×K-S, then Hybrid): the narrow cell claims its single slot
#: first and the widest cell meets an emptier queue. It delays CREATES only --
#: each cell's build clock starts after its own provisioning, so the arms stay
#: build-time-matched.
EXP3_PROVISION_STAGGER_S = 120.0


def exp3_ttl_s(t_total_s: float, *, margin_s: float = EXP3_TTL_MARGIN_S) -> int:
    """Sandbox lease for an Exp-3 cell: the whole round plus margin.

    DERIVED, never a literal: T is the only thing that changes between sizings,
    and a lease that has to be remembered separately is a lease that goes stale.
    """
    return int(t_total_s + margin_s)


#: Exp 3's checkpoint, secured by ``bench.exp3_prep`` (see
#: ``bench/results/exp3_prep.json``). Exp 1's S2 and TEMPLATE_SNAP both lapsed
#: at their 24h leases (``state: deleted``; warm boot returns HTTP 409 "not
#: published"), so the Exp-1 recipe was re-run end to end: fresh template from
#: ``debian-warm`` + the bench image, then ONE codex agent to the same milestone
#: (probe >= 2x quota). Pinned here, not typed on the command line, so no cell
#: of this block can start from the wrong -- or a dead -- checkpoint.
#: REBAKED 2026-08-12 (~11:52Z, 24h lease -- the API's max): the round-1 S2B
#: (snapshot-de94f03f2be50238) and its source S2 both lapsed. This checkpoint
#: was re-secured by ``bench.exp3_prep --how rebake`` with a deepseek milestone
#: agent (codex quota-dead), reference throughput 53.97 >= 2x quota. Round-1
#: Hybrid/Control ran from the OLD S2B; any cell run from this one carries a
#: checkpoint-substitution caveat against them (recorded in the results).
EXP3_S2B = "snapshot-4884dc542852ce8a"
#: The re-baked greenfield template S2B descends from. Held on the keep list for
#: the same reason TEMPLATE_SNAP is: a prefix sweep would otherwise eat the
#: substrate the block is built on.
EXP3_TEMPLATE_SNAP = "snapshot-32d170b01ce042f2"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Tier1Config:
    models: tuple[str, ...] = ("k3",)
    arms: tuple[str, ...] = DEFAULT_ARMS
    #: C runs on ONE mid model only: B-vs-C is about infra, not capability.
    c_model: str = ""
    tasks: tuple[str, ...] = ("iron_plate_throughput",)
    replicates: int = 1
    T_s: float = 1800.0
    K: int = 2
    #: Steps per round: probe cadence in every arm, convergence cadence in B.
    #: Default = Exp 2's sizing (:func:`bench.arms.exp2_round_sizing`) -- m x
    #: 19.9s realized step p50 >= 1.5 x the serial 7 x 62.3s fork wave. Override
    #: with --m (the Tier-1 pilot ran m=4); B needs a T of >= 2 rounds.
    m: int = EXP2_DEFAULT_M
    #: SEED snapshot every cell is created from: TEMPLATE_SNAP for a greenfield
    #: matrix, a baked checkpoint (Exp 2's S2/S3) for the -from-S variants,
    #: where --from-snap sets exactly this field. Held out of the reaper sweep.
    template_snap: str = ""
    ttl_s: int = 7200
    prefix: str = "flebench-"
    #: Substrate that outlives every run: TEMPLATE_SNAP and the run seed are
    #: added automatically (see :meth:`Tier1Runner.keep_ids`); list the bake
    #: sandbox id here so the reaper never treats it as residue. Operator flags
    #: can only ADD to the keep set, never shrink it.
    keep: tuple[str, ...] = ()
    #: The greenfield template, carried SEPARATELY from ``template_snap`` (the
    #: run seed) so a --from-snap run protects both ids, not just the seed.
    template_snap_id: str = TEMPLATE_SNAP
    run_cap: int = 0          # 0 -> from Tier-0 soak results
    max_sandboxes: int = 0    # 0 -> from Tier-0 soak results
    provider_concurrency: int = 16
    #: v2.4.1: stagger probe-cadence phase per run so fork waves do not align.
    stagger_s: float = 20.0
    #: Parallel-round PROVISIONING offset per ladder rung (0 = simultaneous).
    #: Delays creates only; the cell's build clock starts after its own
    #: provisioning, so the arms stay build-time-matched.
    provision_stagger_s: float = 0.0
    #: Poll budget for one create_from_snapshot (0 = the wrapper's 300s default).
    create_deadline_s: float = 0.0
    results_dir: str = "bench/results"
    journal_dir: str = "bench/journal/tier1"
    out: str = "bench/results/tier1_pilot.json"
    label: str = "TIER-1 PILOT (reduced T/tasks/replicates; not the full matrix)"
    dry: bool = False
    #: Expand Exp 2's pre-registered 8-cell manifest (:data:`EXP2_BLOCK`)
    #: instead of the uniform arm x model x task x replicate Cartesian product.
    exp2_block: bool = False
    #: Expand Exp 3's pre-registered 4-cell manifest (:data:`EXP3_BLOCK`):
    #: [Hybrid r1, A×K-2P r1, Hybrid r2, A×K-2P r2] at T = 2P from one
    #: checkpoint. Mutually exclusive with ``exp2_block``.
    exp3_block: bool = False
    #: Exp-3 leg length P (a Hybrid cell runs two legs of it; T_s = 2P).
    leg_s: float = 0.0
    #: Exp-3 round gate: run ONLY this round's rung (``--round 1`` -> the three
    #: r1 cells). 0 = every round in the manifest. The rounds are reviewed
    #: between launches by design, so this is the normal way to run the block --
    #: equivalent to ``--cells Hybrid:N,AxK-S:N,Control:N``.
    round: int = 0
    #: PARALLEL-ROUND mode (pre-registered option): run the selected round's
    #: three cells CONCURRENTLY (~2.7h instead of ~7.7h). Raises the run cap to
    #: 3, sizes the slot pool for the round's 17-sandbox peak, and switches the
    #: provider cap to the shared-8 + exempt-Control regime.
    parallel_round: bool = False
    #: Per-cell recovery selector, ``"arm:replicate"`` items (``"B:2,AxK:2"``,
    #: or a bare ``"B"`` for every replicate of that arm). Runs ONLY those cells
    #: and merges them into an existing ``out`` file, so a lost pair is
    #: re-runnable from S2 without repeating the whole block.
    cells: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class Cell:
    arm: str
    model: str
    task: str
    replicate: int

    @property
    def key(self) -> str:
        return f"{self.arm}|{self.model}|{self.task}|r{self.replicate}"


def exp2_block_cells(cfg: Tier1Config) -> list[Cell]:
    """Exp 2's 8 contract cells, in priority order (:data:`EXP2_BLOCK`)."""
    model = cfg.models[0] if cfg.models else ""
    task = cfg.tasks[0] if cfg.tasks else ""
    return [
        Cell(arm=arm, model=model, task=task, replicate=replicate)
        for arm, replicate in EXP2_BLOCK
    ]


def exp3_block_cells(cfg: Tier1Config) -> list[Cell]:
    """Exp 3's 4 contract cells, in run order (:data:`EXP3_BLOCK`)."""
    model = cfg.models[0] if cfg.models else ""
    task = cfg.tasks[0] if cfg.tasks else ""
    return [
        Cell(arm=arm, model=model, task=task, replicate=replicate)
        for arm, replicate in EXP3_BLOCK
    ]


def parse_cell_selector(spec: Sequence[str]) -> set[tuple[str, int | None]]:
    """``["B:2", "AxK"]`` -> ``{("B", 2), ("AxK", None)}`` (None = every replicate)."""
    selected: set[tuple[str, int | None]] = set()
    for item in spec:
        item = item.strip()
        if not item:
            continue
        arm, _, rep = item.partition(":")
        arm = arm.strip()
        rep = rep.strip().lstrip("rR")
        if not arm:
            raise ValueError(f"cell selector {item!r} has no arm")
        if rep and not rep.isdigit():
            raise ValueError(f"cell selector {item!r} has a non-numeric replicate")
        selected.add((arm, int(rep) if rep else None))
    return selected


def expand_cells(cfg: Tier1Config) -> list[Cell]:
    """The cells this invocation runs, in the order it runs them.

    Either a pre-registered manifest (``exp2_block`` / ``exp3_block``) or one
    cell per (arm, model, task, replicate) with C only for ``c_model``, ordered
    so that every arm of a given (task, replicate, model) block is adjacent.
    ``round`` then keeps only that round's rung (Exp 3 launches one round at a
    time, by design), and ``cells`` narrows further to the named
    (arm, replicate) pairs -- the per-cell recovery path. Neither filter ever
    reorders what survives.
    """
    cells: list[Cell] = []
    if cfg.exp2_block and cfg.exp3_block:
        raise ValueError("exp2_block and exp3_block are mutually exclusive")
    if cfg.exp2_block:
        cells = exp2_block_cells(cfg)
    elif cfg.exp3_block:
        cells = exp3_block_cells(cfg)
    else:
        # Dimensions are DEDUPED in the order given: a repeated --arms/--models/
        # --tasks entry would otherwise expand to two cells with the SAME key,
        # and that key is the identity every journal record, merge base and
        # rerun selector is written against.
        tasks = tuple(dict.fromkeys(cfg.tasks))
        models = tuple(dict.fromkeys(cfg.models))
        arms = tuple(dict.fromkeys(cfg.arms))
        for task in tasks:
            for replicate in range(1, cfg.replicates + 1):
                for model in models:
                    for arm in arms:
                        if arm == "C":
                            continue
                        cells.append(
                            Cell(arm=arm, model=model, task=task, replicate=replicate)
                        )
                if "C" in arms:
                    c_model = cfg.c_model or (models[0] if models else "")
                    if c_model:
                        cells.append(
                            Cell(arm="C", model=c_model, task=task,
                                 replicate=replicate)
                        )
    # Belt and braces: a duplicate key would overwrite a live task in
    # ``cell_tasks`` and merge two endpoints into one row. Neither manifest nor
    # the deduped product can produce one, so this is a bug, not a matrix.
    seen: set[str] = set()
    for cell in cells:
        if cell.key in seen:
            raise ValueError(
                f"duplicate cell {cell.key!r} in the expanded matrix: the cell "
                "key is the identity cell_tasks, --cells and every journal "
                "record use"
            )
        seen.add(cell.key)
    if cfg.round:
        cells = [c for c in cells if c.replicate == cfg.round]
    if cfg.cells:
        selected = parse_cell_selector(cfg.cells)
        cells = [
            c for c in cells
            if (c.arm, c.replicate) in selected or (c.arm, None) in selected
        ]
    return cells


def config_from_tier05(path: str) -> Tier1Config:
    """Build the Tier-1 config from Tier 0.5's frozen pilot config.

    Tier 0.5 may REFUSE to freeze a config (missing or errored calibration
    evidence, no ladder point that fits the budget); its payload then carries
    ``executable: false`` and the reasons. That is not a config with defaults --
    it is the absence of one, so it is an error here rather than a silent
    fallback to DEFAULT_ARMS with no models.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    frozen = payload["frozen_pilot_config"]
    if frozen.get("executable") is False or frozen.get("status") == "REFUSED":
        reasons = frozen.get("reasons") or []
        raise ValueError(
            f"{path}: Tier 0.5 REFUSED to freeze a pilot config "
            f"({frozen.get('error') or 'no error given'})"
            + (f"; reasons: {'; '.join(str(r) for r in reasons)}" if reasons else "")
            + " -- re-run Tier 0.5 instead of launching on defaults"
        )
    return Tier1Config(
        models=tuple(frozen.get("models") or ()),
        arms=tuple(frozen.get("arms") or DEFAULT_ARMS),
        c_model=frozen.get("c_model") or "",
        tasks=tuple(frozen.get("tasks") or ()),
        replicates=int(frozen.get("replicates", 1)),
        T_s=float(frozen.get("T_s", 1800.0)),
        K=int(frozen.get("K", 2)),
        m=int(frozen.get("m", 4)),
        run_cap=int(frozen.get("run_cap", 0) or 0),
        max_sandboxes=int(frozen.get("max_sandboxes", 0) or 0),
    )


#: Config fields that DEFINE the measurement. ``--cells`` recovery rewrites the
#: same results file, so the cells it preserves must have measured the same
#: thing: a cell re-run at another T/K/m, from another checkpoint or under
#: another template is not the same cell, and silently mixing the two would put
#: two experiments in one paired contrast. Scheduling knobs (caps, staggers,
#: prefixes, keep list, out paths) are deliberately absent -- they change how a
#: cell is admitted, never what it measures.
MERGE_FINGERPRINT_FIELDS = (
    "exp2_block", "exp3_block", "models", "tasks", "T_s", "leg_s", "K", "m",
    "template_snap", "template_snap_id",
)


def merge_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """The measurement-defining slice of a config, JSON-normalised.

    Tuples come back from ``json.load`` as lists, so both sides are compared as
    lists; a field the prior file never carried reads as ``None`` and therefore
    as drift -- an artifact that cannot prove compatibility is not compatible.
    """
    out: dict[str, Any] = {}
    for field in MERGE_FINGERPRINT_FIELDS:
        value = config.get(field)
        out[field] = list(value) if isinstance(value, (tuple, list)) else value
    return out


# ---------------------------------------------------------------------------
# Capacity scheduling
# ---------------------------------------------------------------------------


class SlotPool:
    """Counting pool for peak concurrent sandboxes across all live runs."""

    def __init__(self, total: int) -> None:
        self.total = max(1, total)
        self.free = self.total
        self.peak_used = 0
        self._cond = asyncio.Condition()

    def acquire(self, n: int):
        """Reserve ``n`` slots for one run (an async context manager).

        Validated EAGERLY and never clamped: a cell whose peak exceeds the pool
        cannot be admitted at all. Clamping (the old ``min(n, total)``) let an
        8-seat arm run under a 4-slot cap -- the pool then guarantees nothing,
        which is the one thing it exists to do.
        """
        need = max(1, int(n))
        if need > self.total:
            raise ValueError(
                f"a run needs {need} concurrent sandbox(es) but the pool holds "
                f"{self.total}: raise --max-sandboxes (or the Tier-0 cap) or "
                "lower K -- admitting it would break the cap it enforces"
            )
        return self._hold(need)

    @contextlib.asynccontextmanager
    async def _hold(self, need: int):
        async with self._cond:
            while self.free < need:
                await self._cond.wait()
            self.free -= need
            self.peak_used = max(self.peak_used, self.total - self.free)
        try:
            yield
        finally:
            async with self._cond:
                self.free += need
                self._cond.notify_all()


class Gate:
    """An in-flight cap that counts what passed through it.

    A ROOT gate owns the semaphore that actually blocks (one per provider: rate
    limits are per provider, not per run). A CHILD gate delegates to its root and
    counts only its own cell's calls, which is what lets a parallel round journal
    a per-cell high-water mark -- the evidence that the shared cap held live,
    rather than a claim that it was configured.

    Drop-in for :class:`asyncio.Semaphore` as ``LLMClient(semaphore=...)`` uses
    it (``async with``).
    """

    def __init__(self, limit: int, *, label: str, parent: "Gate | None" = None,
                 exempt: bool = False) -> None:
        self.limit = max(1, int(limit))
        self.label = label
        self.parent = parent
        self.exempt = exempt
        self._sem = asyncio.Semaphore(self.limit) if parent is None else None
        self.in_flight = 0
        self.high_water = 0
        self.acquisitions = 0

    @property
    def root(self) -> "Gate":
        return self if self.parent is None else self.parent.root

    async def acquire(self) -> bool:
        if self._sem is not None:
            await self._sem.acquire()
        else:
            assert self.parent is not None
            await self.parent.acquire()
        self.in_flight += 1
        self.acquisitions += 1
        self.high_water = max(self.high_water, self.in_flight)
        return True

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if self._sem is not None:
            self._sem.release()
        else:
            assert self.parent is not None
            self.parent.release()

    def locked(self) -> bool:
        return self.root._sem.locked() if self.root._sem is not None else False

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: Any) -> None:
        self.release()

    def to_dict(self) -> dict[str, Any]:
        root = self.root
        return {
            "label": self.label,
            "limit": self.limit,
            "exempt": self.exempt,
            "shared_gate": root.label,
            "shared_gate_id": id(root),
            "shared_limit": root.limit,
            "acquisitions": self.acquisitions,
            "in_flight_high_water": self.high_water,
            "shared_high_water": root.high_water,
            "in_flight_at_end": self.in_flight,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Tier1Runner:
    def __init__(self, cfg: Tier1Config) -> None:
        self.cfg = cfg
        os.makedirs(cfg.journal_dir, exist_ok=True)
        os.makedirs(cfg.results_dir, exist_ok=True)
        caps = load_tier0_caps(
            Tier05Config(
                results_dir=cfg.results_dir,
                run_cap=cfg.run_cap or 6,
                max_sandboxes=cfg.max_sandboxes or 24,
            )
        )
        self.caps = caps
        self.run_cap = self.resolve_cap("run_cap", cfg.run_cap, caps,
                                        flag="--node-cap", fallback=6)
        self.max_sandboxes = self.resolve_cap("max_sandboxes", cfg.max_sandboxes,
                                              caps, flag="--max-sandboxes",
                                              fallback=24)
        self.slots = SlotPool(self.max_sandboxes)
        self.run_sem = asyncio.Semaphore(self.run_cap)
        self.provider_sems: dict[str, Gate] = {}
        #: Per-cell provider gate, so a parallel round can journal each cell's
        #: own in-flight high-water mark and prove the shared cap held live.
        self.cell_gates: dict[str, Gate] = {}
        #: Live cell tasks by key, so a dead provider takes down exactly the
        #: cells that share its quota and nothing else.
        self.cell_tasks: dict[str, asyncio.Task] = {}
        #: FIRST dead provider (the trigger that stopped admission); the block
        #: exits nonzero with it. See ``dead_providers`` for the full set -- a
        #: second, DIFFERENT provider dying is its own abort, not a duplicate.
        self.provider_dead: ProviderDead | None = None
        #: ``{provider: ProviderDead}``: one abort per provider, and the
        #: authority for attributing a cancelled cell to the quota that died.
        self.dead_providers: dict[str, ProviderDead] = {}
        #: The provider each cell's CLIENT reports -- the authority for "who
        #: shares this quota", since a substrate stand-in may not be the model the
        #: cell names.
        self.cell_providers: dict[str, str] = {}
        self.results: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.skipped: list[str] = []
        #: Cells whose outcome is already recorded (result / failure /
        #: invalid_provider / skipped). Every exit path of :meth:`run_cell`
        #: claims this slot exactly once, so a cancellation in an admission wait
        #: can never leave a cell in no list at all -- and can never double-count
        #: one that already finished.
        self.accounted: set[str] = set()
        #: Exceptions ``asyncio.gather`` returned that no cell accounted for.
        #: Always empty in a healthy run; non-empty is a bug and exits nonzero.
        self.unaccounted: list[dict[str, Any]] = []
        self.stop = asyncio.Event()
        self.started_at = time.time()
        self.master = RunJournal(
            os.path.join(cfg.journal_dir, "tier1-master.jsonl"),
            run_id="tier1-master",
            meta={"config": cfg.to_dict(), "caps": caps, "run_cap": self.run_cap,
                  "max_sandboxes": self.max_sandboxes},
        )
        self.reaper_report: list[dict[str, Any]] = []
        self.needs_rerun: list[dict[str, Any]] = []
        self.merged_from: dict[str, Any] | None = None
        self.load_merge_base()

    # -- capacity ----------------------------------------------------------
    def resolve_cap(self, field: str, explicit: int, caps: dict[str, Any], *,
                    flag: str, fallback: int) -> int:
        """One capacity cap: explicit value > MEASURED Tier-0 evidence > refusal.

        Tier 0 now reports what it measured AND what it did not
        (``caps["evidence"]`` / ``caps["warnings"]``): a cap that was never
        measured is a DEFAULT, and a real block scheduled against a fabricated
        ceiling is exactly how a node gets over-admitted (null there means "never
        measured", 0 means "measured, and it cannot afford one run"). So a real
        run demands either the operator's flag or Tier-0 evidence. A dry run has
        no sandboxes to cap, so it keeps the default -- the gap is already
        journaled with ``caps`` in the master meta.
        """
        if explicit:
            if explicit < 1:
                raise ValueError(
                    f"{flag}={explicit} is not a usable {field}: a cap is at "
                    "least 1 run/sandbox"
                )
            return int(explicit)
        value = caps.get(field)
        measured = (
            field in (caps.get("evidence") or {})
            and isinstance(value, int) and not isinstance(value, bool)
            and value >= 1
        )
        if measured:
            return int(value)
        if self.cfg.dry:
            return fallback
        raise ValueError(
            f"Tier 0 never measured a usable {field} "
            f"(value={value!r}; {'; '.join(caps.get('warnings') or ['no evidence'])})"
            f" -- refusing to schedule a real block against a fabricated cap: "
            f"pass {flag} N explicitly or run Tier 0's soak first"
        )

    # -- dose floor --------------------------------------------------------
    def score_dose(self, cell: Cell, payload: dict[str, Any]) -> None:
        """Invalidate a convergent endpoint that never got its measured dose.

        The design scopes the null to the MEASURED dose, so a B-iterated run
        that converged fewer than :data:`EXP2_DOSE_FLOOR` times is not a weak
        result -- it is not a result. It must not reach the paired contrast; it
        goes on ``needs_rerun`` with a ``--cells`` selector so the pair can be
        recovered from S2 without repeating the block. B-once is exempt by
        construction (its dose IS one) and the A arms never converge.
        """
        if not self.cfg.exp2_block or cell.arm != "B":
            return
        dose = int(payload.get("branch_points") or 0)
        if dose >= EXP2_DOSE_FLOOR:
            return
        payload["status"] = "invalid_dose"
        payload["error"] = (
            f"{dose} convergence(s) < the pre-registered floor of "
            f"{EXP2_DOSE_FLOOR}: endpoint is INCONCLUSIVE, never counted toward "
            "either side of the primary contrast"
        )
        pair = [
            c.key for c in exp2_block_cells(self.cfg)
            if c.replicate == cell.replicate and c.arm in ("B", "AxK")
        ]
        entry = {
            "cell": cell.key,
            "arm": cell.arm,
            "replicate": cell.replicate,
            "dose": dose,
            "dose_floor": EXP2_DOSE_FLOOR,
            "reason": payload["error"],
            "selector": f"{cell.arm}:{cell.replicate}",
            "pair": pair,
            "pair_selector": ",".join(sorted({f"{c.split('|')[0]}:{cell.replicate}"
                                              for c in pair})),
        }
        self.needs_rerun.append(entry)
        self.master.incident(kind="invalid_dose", detail=payload["error"],
                             cell=cell.key, dose=dose, selector=entry["selector"])

    # -- width floor (Exp 3) -----------------------------------------------
    def score_width(self, cell: Cell, payload: dict[str, Any]) -> None:
        """Invalidate a hybrid endpoint whose refork wave was truncated.

        Exp 3's dose is structurally 1, so Exp 2's DOSE floor does not apply
        here; the failure mode that DOES apply is width. Judge-at-the-bell over
        five or fewer seats is a best-of-few, not the max-over-8 the arms are
        supposed to share, so such an endpoint never enters the paired read --
        it goes on ``needs_rerun`` with a ``--cells`` selector, exactly like an
        invalid dose. A×K-S and Control cannot truncate (they never fork).
        """
        if not self.cfg.exp3_block or cell.arm != "Hybrid":
            return
        validity = (payload.get("exp3") or {}).get("validity") or {}
        k_effective = int(validity.get("k_effective") or 0)
        if k_effective >= EXP3_WIDTH_FLOOR:
            return
        payload["status"] = "invalid_width"
        payload["error"] = (
            f"phase 2 judged {k_effective} seat(s) < the pre-registered Exp-3 "
            f"width floor of {EXP3_WIDTH_FLOOR}: the endpoint is a best-of-few, "
            "INCONCLUSIVE, and never counted toward either side of the contrast"
        )
        # Exp 3 is a LADDER, not a pair: an invalid rung invalidates the
        # comparison for its whole round, so the rerun entry names the round.
        round_cells = [
            c.key for c in exp3_block_cells(self.cfg)
            if c.replicate == cell.replicate
        ]
        entry = {
            "cell": cell.key,
            "arm": cell.arm,
            "replicate": cell.replicate,
            "round": cell.replicate,
            "k_effective": k_effective,
            "width_floor": EXP3_WIDTH_FLOOR,
            "reason": payload["error"],
            "selector": f"{cell.arm}:{cell.replicate}",
            "round_cells": round_cells,
            "round_selector": f"--round {cell.replicate}",
        }
        self.needs_rerun.append(entry)
        self.master.incident(kind="invalid_width", detail=payload["error"],
                             cell=cell.key, k_effective=k_effective,
                             selector=entry["selector"])

    # -- accounting --------------------------------------------------------
    def account(self, cell: Cell) -> bool:
        """Claim ``cell``'s ONE accounting slot; ``False`` if already recorded.

        Every terminal path of :meth:`run_cell` -- endpoint, failure,
        invalid_provider, skipped, cancelled -- goes through here, so a cell can
        neither vanish from every list (a cancellation in an admission wait used
        to) nor appear in two of them (an abort landing on a cell that had
        already finished scoring).
        """
        if cell.key in self.accounted:
            return False
        self.accounted.add(cell.key)
        return True

    def dead_provider_for(self, cell: Cell) -> ProviderDead | None:
        """The dead quota THIS cell was running on, if any.

        Keyed on the provider its own client reported (a substrate stand-in may
        not be the model the cell names), falling back to the model's provider
        for a cell cancelled before it ever built a client.
        """
        provider = self.cell_providers.get(cell.key, provider_of(cell.model))
        return self.dead_providers.get(provider)

    # -- provider death (online tripwire) ----------------------------------
    def score_provider(self, cell: Cell, run_id: str, dead: ProviderDead, *,
                       reason: str) -> None:
        """Mark one cell INVALID_PROVIDER with a rerun selector.

        Pre-registered rule: a cell that met a dead provider is NEVER a low
        endpoint. It carries no evidence at all, so it goes on ``needs_rerun``
        with the same ``--cells``/``--round`` selectors an invalid dose or width
        uses, and it stays out of every aggregate.
        """
        if not self.account(cell):
            self.master.event("provider_dead_observed", provider=dead.provider,
                              trigger=dead.trigger, cell=cell.key,
                              already_accounted=True, detail=reason)
            return
        entry = {
            "cell": cell.key,
            "arm": cell.arm,
            "replicate": cell.replicate,
            "round": cell.replicate,
            "status": "invalid_provider",
            "provider": dead.provider,
            "trigger": dead.trigger,
            "reason": f"{reason}: {dead}"[:2000],
            "stats": dead.stats,
            "selector": f"{cell.arm}:{cell.replicate}",
            "round_selector": f"--round {cell.replicate}",
        }
        self.needs_rerun.append(entry)
        self.failures.append({"cell": cell.key, "run_id": run_id,
                              "error": f"invalid_provider: {dead}"[:2000]})
        self.master.incident(kind="invalid_provider", detail=entry["reason"],
                             cell=cell.key, run_id=run_id, provider=dead.provider,
                             trigger=dead.trigger, selector=entry["selector"])

    def abort_provider(self, dead: ProviderDead, *, origin: Cell) -> None:
        """Take down every RUNNING cell on the dead provider, at once.

        Same mechanism as the second SIGINT: stop admitting new cells, then cancel
        the live ones so their arms tear down through their own ``finally`` and the
        block-level reaper sweeps whatever is left. Cells on a DIFFERENT provider
        are untouched -- the quota that died is not theirs.
        """
        prior = self.dead_providers.get(dead.provider)
        if prior is not None:
            # Concurrent cells on the SAME provider can each hit the wire before
            # the cancel lands. That provider's ABORT happens once -- one trigger
            # record, one cancel sweep -- and the extra sightings are journaled as
            # corroboration. A DIFFERENT provider dying is not a duplicate: it
            # owns its own victims, so it falls through to its own abort.
            self.master.event("provider_dead_observed", provider=dead.provider,
                              trigger=dead.trigger, cell=origin.key,
                              already_aborting=True)
            return
        self.dead_providers[dead.provider] = dead
        if self.provider_dead is None:
            self.provider_dead = dead
        self.stop.set()
        victims = [
            key for key, task in self.cell_tasks.items()
            if key != origin.key and not task.done()
            and self.cell_providers.get(key, provider_of(key.split("|")[1]))
            == dead.provider
        ]
        self.master.event(
            "provider_dead", provider=dead.provider, trigger=dead.trigger,
            origin=origin.key, aborted=victims, detail=str(dead),
            stats=dead.stats,
        )
        print(f"[tier1] PROVIDER DEAD ({dead.provider}): {dead}", flush=True)
        if victims:
            print(f"[tier1] aborting {len(victims)} cell(s) on {dead.provider}: "
                  f"{', '.join(victims)}", flush=True)
        for key in victims:
            self.cell_tasks[key].cancel()

    # -- keep list ---------------------------------------------------------
    def keep_ids(self) -> list[str]:
        """Ids the reaper must NEVER delete, in a fixed order.

        ALWAYS both snapshots: the greenfield TEMPLATE_SNAP (every checkpoint
        descends from it) and this run's SEED (``--from-snap`` = Exp 2's S2).
        The Tier-1 pilot's between-block sweep destroyed the bake sandbox
        precisely because ownership is a name prefix and substrate was not held
        out explicitly, so operator flags may only ADD to this set.
        """
        ids: list[str] = []
        for k in (self.cfg.template_snap_id, self.cfg.template_snap, *self.cfg.keep):
            if k and k not in ids:
                ids.append(k)
        return ids

    # -- per-cell recovery -------------------------------------------------
    def load_merge_base(self) -> None:
        """``--cells``: seed the results with the cells this pass is NOT re-running.

        Every later :meth:`write_partial` then rewrites the SAME file with the
        preserved cells plus the fresh ones, so re-running one lost pair from S2
        costs one pair, not the block.

        The prior file is only a merge base if it measured the SAME thing: its
        stored config is fingerprinted (:data:`MERGE_FINGERPRINT_FIELDS`) against
        this one and any drift is a hard refusal. A file that carries no config at
        all cannot prove compatibility, so it is refused too -- silently blending
        a 4200s endpoint with a 1800s one is the worst outcome available here.
        """
        path = self.cfg.out
        if not self.cfg.cells or not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            prior = json.load(fh)
        prior_config = prior.get("config")
        if not isinstance(prior_config, dict):
            raise ValueError(
                f"{path} carries no config block, so --cells cannot verify that "
                "its preserved cells measured what this pass measures; point "
                "--out at a fresh file or re-run the whole block"
            )
        mine = merge_fingerprint(self.cfg.to_dict())
        theirs = merge_fingerprint(prior_config)
        drift = {k: (theirs[k], mine[k]) for k in mine if theirs[k] != mine[k]}
        if drift:
            raise ValueError(
                f"refusing to merge into {path}: it was written under a "
                "different measurement config ("
                + "; ".join(f"{k} {was!r} -> {now!r}"
                            for k, (was, now) in sorted(drift.items()))
                + ") -- a cell re-run at a different T/K/m, from a different "
                "checkpoint or under a different template is not the same cell; "
                "use a fresh --out"
            )
        rerun = {c.key for c in expand_cells(self.cfg)}
        self.results = [r for r in prior.get("runs", []) if r.get("cell") not in rerun]
        self.failures = [
            f for f in prior.get("failures", []) if f.get("cell") not in rerun
        ]
        self.skipped = [s for s in prior.get("skipped", []) if s not in rerun]
        # A cell being re-run drops its old invalid-dose entry; if the rerun is
        # still short of the floor, score_dose puts a fresh one back.
        self.needs_rerun = [
            n for n in prior.get("needs_rerun", []) if n.get("cell") not in rerun
        ]
        self.started_at = float(prior.get("started_at") or self.started_at)
        self.merged_from = {
            "path": path,
            "preserved_runs": [r.get("cell") for r in self.results],
            "rerun_cells": sorted(rerun),
            "fingerprint": mine,
        }
        self.master.event("merge_base_loaded", **self.merged_from)

    # -- substrate ---------------------------------------------------------
    def provider_sem(self, model: str) -> Gate:
        """One ROOT gate per provider: rate limits are per provider, not per run."""
        provider = provider_of(model)
        if provider not in self.provider_sems:
            self.provider_sems[provider] = Gate(
                self.cfg.provider_concurrency, label=provider
            )
        return self.provider_sems[provider]

    def cell_gate(self, cell: Cell) -> Gate:
        """This cell's view of the provider cap -- shared, or a private exemption.

        Every cell of a provider normally hangs off that provider's single root
        gate. Exp 3's parallel round exempts ONE arm by pre-registration
        (:data:`EXP3_PROVIDER_EXEMPT_ARMS`): its single seat gets a PRIVATE gate
        of 1 so it steps at full rate, because a shared cap penalises a
        redundancy-free line far harder than a best-of-8 population and would
        inflate the fork-value rung it measures. The exemption is journaled and
        its bias direction is pre-registered, not discovered afterwards.
        """
        exempt = (
            self.cfg.exp3_block
            and self.cfg.parallel_round
            and cell.arm in EXP3_PROVIDER_EXEMPT_ARMS
        )
        if exempt:
            root = Gate(1, label=f"private:{cell.arm}", exempt=True)
        else:
            root = self.provider_sem(cell.model)
        gate = Gate(root.limit, label=cell.key, parent=root, exempt=exempt)
        self.cell_gates[cell.key] = gate
        return gate

    def make_substrate(self, run_id: str):
        if self.cfg.dry:
            world, fp, bridge_factory, template = fake_substrate(latency=0.004)
            if self.cfg.template_snap:
                # Honour an explicit seed id (a real S2/S3 checkpoint) in dry
                # runs too: register it in the fake world so the arms take the
                # same create-from-checkpoint path they take for real.
                world.snapshots[self.cfg.template_snap] = world.snapshots[
                    template
                ].clone()
                template = self.cfg.template_snap
            return fp, bridge_factory, template
        from bench.bridge_client import Bridge
        from bench.farplane import Farplane

        fp = Farplane(os.path.join(self.cfg.journal_dir, f"{run_id}-farplane.jsonl"))
        return fp, (lambda url: Bridge(url)), self.cfg.template_snap

    def make_llm(self, cell: Cell, journal: RunJournal):
        gate = self.cell_gate(cell)
        self.master.write("cell_provider_gate", cell=cell.key, arm=cell.arm,
                          model=cell.model, parallel_round=self.cfg.parallel_round,
                          **gate.to_dict())
        if self.cfg.dry:
            # The fakes go through the SAME gate: a dry parallel round has to
            # exercise the cap and the exemption, not just configure them.
            return FakeLLM(journal=journal, log_full_requests=False,
                           max_concurrency=self.cfg.K, semaphore=gate)
        from bench.llm import make_client

        return make_client(cell.model, journal=journal, semaphore=gate)

    # -- one cell ----------------------------------------------------------
    def run_id_of(self, cell: Cell) -> str:
        """The id this cell's journal, resources and records are named with."""
        return (
            f"{cell.arm}-{cell.model.replace('/', '-')}-{cell.task}-r{cell.replicate}"
        )

    def record_skipped(self, cell: Cell, *, reason: str) -> None:
        """A cell the graceful stop never admitted -- and the point it stopped at.

        The record names WHY the block stopped admitting, because "skipped" alone
        cannot distinguish an operator's interrupt from an aborted quota, and a
        cell that never started still has to be re-run.
        """
        if not self.account(cell):
            return
        self.skipped.append(cell.key)
        dead = self.dead_provider_for(cell) or self.provider_dead
        self.master.event("run_skipped", cell=cell.key, reason=reason,
                          provider_dead=None if dead is None else dead.provider)

    def record_cancelled(self, cell: Cell, run_id: str) -> None:
        """A cell cancelled ANYWHERE in its lifecycle, admission waits included.

        The old handler wrapped ``run_one`` only, so an abort that landed while a
        cell sat in the run-slot queue or slept out a 120-240s provisioning
        stagger left it in no list at all: ``gather(return_exceptions=True)``
        swallowed the ``CancelledError`` and the block reported a complete matrix
        minus a cell. Attribution is per PROVIDER -- cancelled with a dead quota
        is ``invalid_provider`` with a rerun selector, cancelled by an operator's
        second interrupt is a plain failure.
        """
        dead = self.dead_provider_for(cell)
        if dead is not None:
            self.score_provider(cell, run_id, dead,
                                reason="aborted with the provider")
        elif self.account(cell):
            self.failures.append({"cell": cell.key, "run_id": run_id,
                                  "error": "cancelled"})
        self.master.event("run_cancelled", cell=cell.key, run_id=run_id,
                          provider_dead=dead is not None,
                          provider=None if dead is None else dead.provider)

    def record_failure(self, cell: Cell, run_id: str, exc: BaseException, *,
                       kind: str = "run_failed") -> None:
        """One failing cell is a datum; losing the record of it is not."""
        if not self.account(cell):
            return
        self.failures.append(
            {"cell": cell.key, "run_id": run_id,
             "error": f"{type(exc).__name__}: {exc}"[:2000]}
        )
        self.master.incident(kind=kind, detail=str(exc)[:2000],
                             cell=cell.key, run_id=run_id)

    async def close_llm(self, cell: Cell, llm: Any) -> None:
        """Release one cell's client. A fake without ``aclose`` is a no-op.

        ``run_one`` only closes clients it created itself, so a client the
        orchestrator passes in is the orchestrator's to close -- on EVERY exit
        path, or a long block leaks one connection pool per cell.
        """
        aclose = getattr(llm, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a client that will not shut
            # down must never mask the cell's own outcome.
            self.master.incident(kind="llm_close_failed", detail=str(exc)[:2000],
                                 cell=cell.key)

    async def run_cell(self, cell: Cell, index: int) -> None:
        """Admit, run and ACCOUNT FOR one cell. Nothing leaves here unrecorded.

        The recording scope wraps the whole admitted-cell setup -- cell journal,
        substrate, client and the ArmConfig lease guard included. A refusal in any
        of them used to escape into ``gather(return_exceptions=True)``, which
        discarded it: no failure row, no rerun selector, and a leaked open journal.
        """
        run_id = self.run_id_of(cell)
        try:
            await self._admit_cell(cell, index, run_id)
        except asyncio.CancelledError:
            self.record_cancelled(cell, run_id)
            self.write_partial()
            raise
        except BaseException as exc:  # noqa: BLE001 - admission and setup are as
            # fallible as the run itself (slot pool, lease guard, substrate,
            # client); each is a failed cell, never a lost one.
            self.record_failure(cell, run_id, exc, kind="cell_setup_failed")
            self.write_partial()

    async def _admit_cell(self, cell: Cell, index: int, run_id: str) -> None:
        """Hold the run slot and this cell's sandbox slots, then run it.

        The graceful stop is re-checked after EVERY wait that can last minutes --
        run-slot queue, slot-pool queue, probe-cadence stagger, provisioning
        stagger -- because the check that matters is the one immediately before
        provisioning starts spending.
        """
        cfg = self.cfg
        if self.stop.is_set():
            self.record_skipped(cell, reason="graceful stop before admission")
            return
        peak = peak_sandboxes(cell.arm, cfg.K)
        async with self.run_sem:
            if self.stop.is_set():
                self.record_skipped(cell, reason="graceful stop in the run-slot queue")
                return
            async with self.slots.acquire(peak):
                if self.stop.is_set():
                    self.record_skipped(cell,
                                        reason="graceful stop in the slot-pool queue")
                    return
                if cfg.stagger_s:
                    # Phase-shift probe cadence between runs (v2.4.1) so their
                    # fork waves do not collide on the same warm slots.
                    await asyncio.sleep((index % max(1, self.run_cap)) * cfg.stagger_s)
                if cfg.provision_stagger_s:
                    # PROVISIONING stagger, in LADDER order: the narrow rung
                    # claims its slot first and the widest one meets an emptier
                    # create queue. Round 1 starved Hybrid's 8th seat because all
                    # 17 creates hit the warm-slot lane at once. This delays
                    # CREATES only -- the build clock starts inside run_one, after
                    # this cell's own provisioning -- so the arms stay
                    # build-time-matched.
                    rung = (EXP3_LADDER.index(cell.arm)
                            if cell.arm in EXP3_LADDER else 0)
                    delay_s = rung * cfg.provision_stagger_s
                    self.master.event("provision_stagger", cell=cell.key,
                                      arm=cell.arm, rung=rung,
                                      delay_s=round(delay_s, 3),
                                      ladder=list(EXP3_LADDER))
                    if delay_s:
                        await asyncio.sleep(delay_s)
                if self.stop.is_set():
                    # The last gate before anything is provisioned. A stagger is
                    # minutes long and a slot queue can be hours: an interrupt
                    # that lands inside one means "no new runs", not "one more".
                    self.record_skipped(
                        cell, reason="graceful stop before provisioning"
                    )
                    return
                await self._run_admitted(cell, run_id, peak)

    async def _run_admitted(self, cell: Cell, run_id: str, peak: int) -> None:
        """Provision, run, score and finalise ONE admitted cell."""
        cfg = self.cfg
        journal = RunJournal(
            os.path.join(cfg.journal_dir, f"{run_id}.jsonl"),
            run_id=run_id,
            meta={"cell": cell.key, "label": cfg.label},
        )
        fp: Any = None
        llm: Any = None
        try:
            fp, bridge_factory, template = self.make_substrate(run_id)
            llm = self.make_llm(cell, journal)
            self.cell_providers[cell.key] = getattr(
                llm, "provider", provider_of(cell.model)
            )
            # B/B-once refuse any round the remaining budget cannot cover,
            # using the MEASURED constants (snapshot 10.1s + 7 x 151.6s fork
            # p95 + m x 19.9s + K x 1s cleanup). A dry T is seconds, so
            # those constants would refuse every round and the dry block
            # would never exercise the fork path -- scale the whole cost
            # model to the fakes, keeping its SHAPE.
            dry_costs = (
                dict(snapshot_cost_estimate_s=0.05, fork_cost_estimate_s=0.05,
                     step_cost_estimate_s=0.05, probe_cost_estimate_s=0.05,
                     delete_cost_estimate_s=0.005)
                if cfg.dry else {}
            )
            arm_cfg = ArmConfig(
                arm=cell.arm, model=cell.model, task_key=cell.task,
                replicate=cell.replicate, T_s=cfg.T_s, K=cfg.K, m=cfg.m,
                template_snap=template, ttl_s=cfg.ttl_s, prefix=cfg.prefix,
                results_dir=cfg.results_dir, journal_dir=cfg.journal_dir,
                run_id=run_id, dry=cfg.dry, leg_s=cfg.leg_s,
                create_deadline_s=cfg.create_deadline_s,
                provision_stagger_s=cfg.provision_stagger_s, **dry_costs,
            )
            self.master.event("run_start", cell=cell.key, run_id=run_id,
                              peak_sandboxes=peak, slots_free=self.slots.free,
                              ttl_s=arm_cfg.ttl_s, T_s=arm_cfg.T_s,
                              create_deadline_s=arm_cfg.create_deadline_s)
            t0 = time.monotonic()
            result = await run_one(
                arm_cfg, farplane=fp, bridge_factory=bridge_factory,
                llm=llm, journal=journal,
            )
            payload = result.to_dict()
            payload["cell"] = cell.key
            # The cap is only real if it held: per-cell in-flight
            # high-water travels with the endpoint, so the analysis can
            # verify the parallel round's throttling instead of trusting
            # the config that asked for it.
            gate = self.cell_gates.get(cell.key)
            payload["provider_gate"] = gate.to_dict() if gate else {}
            self.score_dose(cell, payload)
            self.score_width(cell, payload)
            if self.account(cell):
                self.results.append(payload)
            # The journal carries the status the RESULTS file carries: the
            # validity floors run above, so a run_done that echoed the arm's
            # pre-scoring status would call an invalid_dose endpoint "ok" in
            # the one record the analysis reads first.
            self.master.event(
                "run_done", cell=cell.key, run_id=run_id,
                status=payload.get("status", result.status),
                arm_status=result.status,
                error=payload.get("error") or None,
                endpoint=result.endpoint_throughput,
                wall_s=round(time.monotonic() - t0, 2),
                slots_free=self.slots.free,
                peak_sandboxes_used=self.slots.peak_used,
                **{f"provider_{k}": v
                   for k, v in (gate.to_dict() if gate else {}).items()},
            )
        except asyncio.CancelledError:
            # Accounted by :meth:`run_cell`, which covers the whole lifecycle --
            # including the admission waits this scope never saw.
            raise
        except ProviderDead as exc:
            # The tripwire fired inside this cell. Mark it, journal the
            # trigger stats, then take down every other cell on the same
            # provider -- the SIGINT path, minus the human.
            self.score_provider(cell, run_id, exc,
                                reason="tripwire fired in this cell")
            self.abort_provider(exc, origin=cell)
            if fp is not None:
                await self.reap_cell(cell, run_id, fp)
        except BaseException as exc:  # noqa: BLE001 - one failing cell must
            # not take the pilot down; the failure itself is a datum.
            self.record_failure(cell, run_id, exc)
            # Failure hygiene: the arm's own teardown only knows the
            # seats it attached, so a sandbox that came up and then
            # failed its health poll survives as an orphan. Reap it HERE
            # -- still holding the run slot -- so the next cell's fork
            # wave meets an empty node instead of the last cell's
            # corpses. Only meaningful at run cap 1, where no sibling
            # cell's resources can be in the sweep's path.
            if fp is not None:
                await self.reap_cell(cell, run_id, fp)
        finally:
            try:
                await self.close_llm(cell, llm)
            finally:
                journal.close()
                self.write_partial()

    # -- output ------------------------------------------------------------
    def payload(self, *, interrupted: bool) -> dict[str, Any]:
        return {
            "label": self.cfg.label,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "wall_s": round(time.time() - self.started_at, 2),
            "config": self.cfg.to_dict(),
            "caps": {
                "source": self.caps.get("source"),
                "run_cap": self.run_cap,
                "max_sandboxes": self.max_sandboxes,
                "peak_sandboxes_used": self.slots.peak_used,
            },
            "interrupted": interrupted,
            "n_cells": len(self.results) + len(self.failures) + len(self.skipped),
            "runs": self.results,
            "failures": self.failures,
            "skipped": self.skipped,
            "needs_rerun": self.needs_rerun,
            "dose_floor": EXP2_DOSE_FLOOR if self.cfg.exp2_block else None,
            "width_floor": EXP3_WIDTH_FLOOR if self.cfg.exp3_block else None,
            "leg_s": self.cfg.leg_s if self.cfg.exp3_block else None,
            "round": self.cfg.round or None,
            "ladder": list(EXP3_LADDER) if self.cfg.exp3_block else None,
            "parallel_round": self.cfg.parallel_round,
            "provider": {
                "concurrency": self.cfg.provider_concurrency,
                "exempt_arms": (list(EXP3_PROVIDER_EXEMPT_ARMS)
                                if self.cfg.parallel_round else []),
                "shared_gates": {p: g.to_dict()
                                 for p, g in self.provider_sems.items()},
                "per_cell": {c: g.to_dict() for c, g in self.cell_gates.items()},
                "health": provider_health_snapshot(),
            },
            "provider_dead": (
                None if self.provider_dead is None else {
                    "provider": self.provider_dead.provider,
                    "trigger": self.provider_dead.trigger,
                    "detail": str(self.provider_dead),
                    "stats": self.provider_dead.stats,
                }
            ),
            #: Every provider that died, not just the first: a second one is its
            #: own abort with its own victims, never a duplicate of the first.
            "provider_dead_all": [
                {"provider": d.provider, "trigger": d.trigger,
                 "detail": str(d), "stats": d.stats}
                for d in self.dead_providers.values()
            ],
            #: Exceptions the gather returned that no cell accounted for. Empty
            #: in a healthy run; anything here is a bug AND a nonzero exit.
            "unaccounted": self.unaccounted,
            "summary": summarize(self.results),
            "paired": paired_differences(self.results),
            "reaper": self.reaper_report,
            "keep": self.keep_ids(),
            "merged_from": self.merged_from,
        }

    def write_partial(self, *, interrupted: bool = False) -> None:
        """Atomic rewrite after every completion: partial results are usable."""
        atomic_write_json(self.cfg.out, self.payload(interrupted=interrupted))

    # -- reaper ------------------------------------------------------------
    async def reap_cell(self, cell: Cell, run_id: str, fp: Any) -> None:
        """Sweep ONE dead cell's residue before the next cell starts.

        Uses the cell's own Farplane handle (its ledger already knows exactly
        what it created) and the run's full keep list, so the seed checkpoint
        and TEMPLATE_SNAP are never in the blast radius. Skipped above run cap 1
        because a prefix sweep there would also be inside a sibling cell's
        working set.
        """
        if self.run_cap != 1:
            self.master.event("cell_reap_skipped", cell=cell.key, run_id=run_id,
                              reason=f"run_cap={self.run_cap} > 1")
            return
        keep = self.keep_ids()
        try:
            swept = await asyncio.to_thread(
                lambda: fp.reaper(self.cfg.prefix, keep=keep)
            )
        except BaseException as exc:  # noqa: BLE001 - never mask the real failure
            self.master.incident(kind="cell_reaper_failed", detail=str(exc)[:2000],
                                 cell=cell.key, run_id=run_id)
            return
        self.reaper_report.extend(swept)
        self.master.event("cell_reaped", cell=cell.key, run_id=run_id,
                          n=len(swept), items=swept, kept=keep)

    async def reap(self) -> None:
        cfg = self.cfg
        # The reaper's ownership test is the `flebench-` name prefix, which the
        # BAKE sandbox and TEMPLATE_SNAP also carry -- they are the substrate
        # every run is created from, not run residue. They must be held out
        # explicitly or a between-block sweep destroys the pilot's own template.
        keep = self.keep_ids()
        try:
            if cfg.dry:
                world, fp, _bf, tpl = fake_substrate(latency=0.001)
                # Put the protected ids in the fake world so the dry sweep
                # exercises the keep list for real instead of trivially.
                for snap in keep:
                    world.snapshots.setdefault(snap, world.snapshots[tpl].clone())
            else:
                from bench.farplane import Farplane

                fp = Farplane(os.path.join(cfg.journal_dir, "reaper.jsonl"))
            swept = await asyncio.to_thread(
                lambda: fp.reaper(cfg.prefix, keep=keep)
            )
            # Accumulate: a per-cell sweep's record must survive into the final
            # payload, or the zero-residual audit loses everything the block
            # reaped between cells.
            self.reaper_report.extend(swept)
            self.master.event("reaped", n=len(swept), items=swept, kept=keep)
        except BaseException as exc:  # noqa: BLE001 - never mask the real result
            self.reaper_report.append(
                {"outcome": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            self.master.incident(kind="reaper_failed", detail=str(exc)[:2000])

    # -- accounting nets ---------------------------------------------------
    def preflight_capacity(self, cells: Sequence[Cell]) -> None:
        """Refuse the whole matrix if a cell cannot fit the slot pool.

        :class:`SlotPool` no longer clamps an over-wide request, so an 8-seat arm
        under a 4-slot cap would otherwise raise once per cell, mid-flight, after
        the block had already started spending. One error instead, before anything
        is provisioned, naming every offending arm.
        """
        pool = self.slots.total
        over = {
            cell.arm: peak_sandboxes(cell.arm, self.cfg.K)
            for cell in cells
            if peak_sandboxes(cell.arm, self.cfg.K) > pool
        }
        if not over:
            return
        detail = ", ".join(f"{arm} needs {n}" for arm, n in sorted(over.items()))
        raise ValueError(
            f"the {pool}-slot sandbox pool cannot hold every cell of this matrix "
            f"at K={self.cfg.K} ({detail} concurrent sandbox(es)): raise "
            "--max-sandboxes (or the Tier-0 cap) or lower K -- admitting them "
            "would break the cap the pool exists to enforce"
        )

    def account_gathered(self, cells: Sequence[Cell],
                         outcomes: Sequence[Any]) -> None:
        """Close the accounting net over whatever the gather returned.

        ``return_exceptions=True`` turns every escape into a value, so this is the
        last place a cell can still be lost. A task cancelled BEFORE its coroutine
        ever ran (an abort landing between ``create_task`` and the first step)
        never reached :meth:`run_cell`; anything else here is a bug in the
        recording scope. Both are recorded, and both exit nonzero.
        """
        for cell, outcome in zip(cells, outcomes):
            if not isinstance(outcome, BaseException):
                continue
            if cell.key in self.accounted:
                continue
            if isinstance(outcome, asyncio.CancelledError):
                self.record_cancelled(cell, self.run_id_of(cell))
                continue
            self.unaccounted.append(
                {"cell": cell.key,
                 "error": f"{type(outcome).__name__}: {outcome}"[:2000]}
            )
            self.record_failure(cell, self.run_id_of(cell), outcome,
                                kind="unaccounted_exception")

    # -- main --------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        cells = expand_cells(self.cfg)
        self.preflight_capacity(cells)
        self.master.event("pilot_start", n_cells=len(cells),
                          cells=[c.key for c in cells])
        loop = asyncio.get_running_loop()
        tasks = [
            asyncio.create_task(self.run_cell(cell, i), name=cell.key)
            for i, cell in enumerate(cells)
        ]
        # Registered by key so the provider tripwire can cancel exactly the cells
        # that share the dead quota (see :meth:`abort_provider`).
        self.cell_tasks = {cell.key: task for cell, task in zip(cells, tasks)}

        def on_signal() -> None:
            if self.stop.is_set():
                self.master.event("hard_stop")
                print("[tier1] cancelling running runs; teardown + reaper still run.",
                      flush=True)
                for t in tasks:
                    t.cancel()
            else:
                self.stop.set()
                self.master.event(
                    "graceful_stop",
                    note="no new runs; interrupt again to cancel running ones",
                )
                print("\n[tier1] graceful stop: no new runs will start. "
                      "Interrupt again to cancel running ones.", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, on_signal)
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            self.account_gathered(cells, outcomes)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError, ValueError):
                    loop.remove_signal_handler(sig)
            await self.reap()
            interrupted = self.stop.is_set()
            self.write_partial(interrupted=interrupted)
            payload = self.payload(interrupted=interrupted)
            self.master.event("pilot_end", interrupted=interrupted,
                              runs=len(self.results), failures=len(self.failures),
                              skipped=len(self.skipped))
            self.master.close()
        return payload


# ---------------------------------------------------------------------------
# Aggregation (primary endpoint: quota-normalised terminal probe, equal weights)
# ---------------------------------------------------------------------------


def _normalized(run: dict[str, Any]) -> float | None:
    """Quota-normalised endpoint, or None when the run is not a valid endpoint.

    ``invalid_dose`` (a convergent arm that never got its pre-registered dose)
    and ``partial`` (no terminal probe) are INCONCLUSIVE by pre-registration:
    they are reported and re-run, never averaged and never paired. Filtering
    here keeps that true for every consumer at once.
    """
    if run.get("status") not in (None, "", "ok"):
        return None
    quota = run.get("quota") or 0
    throughput = run.get("endpoint_throughput")
    if throughput is None or not quota:
        return None
    return throughput / quota


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[float]] = {}
    by_arm_model: dict[str, list[float]] = {}
    infra_fractions: dict[str, list[float]] = {}
    llm_fractions: dict[str, list[float]] = {}
    steps: dict[str, list[float]] = {}
    for run in runs:
        norm = _normalized(run)
        arm = run.get("arm", "?")
        if norm is not None:
            by_arm.setdefault(arm, []).append(norm)
            by_arm_model.setdefault(f"{arm}|{run.get('model')}", []).append(norm)
        timings = run.get("timings") or {}
        wall = timings.get("wall_s") or 0.0
        if wall:
            attributed = timings.get("attributed_s") or {}
            infra = sum(v for k, v in attributed.items() if k.startswith("infra_"))
            infra_fractions.setdefault(arm, []).append(infra / wall)
            llm_fractions.setdefault(arm, []).append(
                attributed.get("llm_wait", 0.0) / wall
            )
        steps.setdefault(arm, []).append(float(run.get("steps") or 0))
    return {
        "endpoint_definition": (
            "quota-normalised throughput from ONE fixed 60s-window probe at T on a "
            "disposable fork; equal task weights"
        ),
        "normalized_endpoint_by_arm": {a: _stats(v) for a, v in by_arm.items()},
        "normalized_endpoint_by_arm_model": {
            k: _stats(v) for k, v in by_arm_model.items()
        },
        "infra_fraction_by_arm": {a: _stats(v) for a, v in infra_fractions.items()},
        "llm_fraction_by_arm": {a: _stats(v) for a, v in llm_fractions.items()},
        "steps_by_arm": {a: _stats(v) for a, v in steps.items()},
        "end_to_end_s": _stats([float(r.get("end_to_end_s") or 0.0) for r in runs]),
        "n_runs": len(runs),
    }


#: Contrasts the design pre-registers. Read as paired per-(model, task,
#: replicate) differences; close calls stay inconclusive by construction.
#: Exp 3's ladder is read as two rungs -- ``AxK-S`` - ``Control`` is the value of
#: forking wide, ``Hybrid`` - ``AxK-S`` the value of one convergence -- both
#: DIRECTIONAL at n=2 rounds by pre-registration: reported with the per-seat
#: distributions, never as a significance claim.
CONTRASTS = (
    ("B", "AxK"), ("B", "A"), ("AxK", "A"),
    ("Hybrid", "AxK-S"), ("AxK-S", "Control"), ("Hybrid", "Control"),
)


def paired_differences(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    index: dict[tuple[str, str, str, int], float] = {}
    for run in runs:
        norm = _normalized(run)
        if norm is None:
            continue
        index[
            (
                run.get("arm", "?"),
                run.get("model", "?"),
                run.get("task_key", "?"),
                int(run.get("replicate") or 0),
            )
        ] = norm
    out: dict[str, Any] = {}
    for left, right in CONTRASTS:
        diffs: list[dict[str, Any]] = []
        for (arm, model, task, rep), value in index.items():
            if arm != left:
                continue
            other = index.get((right, model, task, rep))
            if other is None:
                continue
            diffs.append(
                {"model": model, "task": task, "replicate": rep,
                 "delta": round(value - other, 4),
                 left: round(value, 4), right: round(other, 4)}
            )
        deltas = [d["delta"] for d in diffs]
        out[f"{left}-{right}"] = {
            "n_pairs": len(diffs),
            "mean_delta": round(statistics.fmean(deltas), 4) if deltas else None,
            "median_delta": round(statistics.median(deltas), 4) if deltas else None,
            "wins": sum(1 for d in deltas if d > 0),
            "losses": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
            "pairs": diffs,
        }
    out["note"] = (
        "Paired per-(model, task, replicate) differences on the pre-registered "
        "endpoint. With pilot sample sizes, close calls are inconclusive; "
        "non-rejection is not equivalence."
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tier-1 pilot orchestrator")
    ap.add_argument("--config", default="", help="JSON file with a Tier1Config")
    ap.add_argument("--from-tier05", default="",
                    help="bench/results/tier05.json -> frozen_pilot_config")
    ap.add_argument("--arms", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--c-model", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--replicates", type=int, default=0)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--m", type=int, default=0)
    # --from-snap is this same knob under Exp 2's name for it: A×K-from-S is A×K
    # with template_snap=<checkpoint>, so there is nothing else to pass.
    ap.add_argument("--template-snap", "--from-snap",
                    default=os.environ.get("TEMPLATE_SNAP", ""),
                    help="seed snapshot id every cell is created from: "
                         "TEMPLATE_SNAP for greenfield arms, a baked checkpoint "
                         "(S2/S3) for the -from-S arms")
    ap.add_argument("--node-cap", type=int, default=0,
                    help="override the Tier-0 soak per-node run cap")
    ap.add_argument("--max-sandboxes", type=int, default=0)
    ap.add_argument("--provider-concurrency", type=int, default=0)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--keep", default="",
                    help="comma-separated resource ids the reaper must never "
                         "delete (the bake sandbox); ADDED to whatever the "
                         "config or block preset already protects -- the keep "
                         "set only ever grows -- and TEMPLATE_SNAP plus the "
                         "--from-snap seed are ALWAYS kept on top of that")
    ap.add_argument("--template-snap-id", default=None,
                    help="the greenfield template id, kept separately from the "
                         "run seed so both survive every sweep (default: the "
                         f"config's value, else {TEMPLATE_SNAP})")
    ap.add_argument("--exp2-block", action="store_true",
                    help="run Exp 2's pre-registered 8-cell manifest (3x "
                         "B-iterated + 3x AxK-from-S as pairs, then B-once, "
                         f"then A-continue) at K={EXP2_K}, T={EXP2_T_S:.0f}s, "
                         f"m={EXP2_DEFAULT_M}, dose floor {EXP2_DOSE_FLOOR}, "
                         "S2 + k3, run cap 1")
    ap.add_argument("--exp3-block", action="store_true",
                    help="run Exp 3's pre-registered ladder manifest (6 cells = "
                         "Hybrid + AxK-S + Control per round, 2 rounds) at "
                         f"K={EXP3_K}, P={EXP3_P_S:.0f}s, "
                         f"T=2P={EXP3_T_S:.0f}s, m={EXP2_DEFAULT_M}, width floor "
                         f"{EXP3_WIDTH_FLOOR}, one checkpoint + codex, run cap 1; "
                         "pair with --round N (rounds are reviewed between "
                         "launches -- round 2 needs an explicit go)")
    ap.add_argument("--round", type=int, default=0,
                    help="run ONLY this round of a rounds-based manifest "
                         "(--exp3-block: --round 1 = Hybrid r1, AxK-S r1, "
                         "Control r1; equivalent to --cells "
                         "Hybrid:1,AxK-S:1,Control:1)")
    ap.add_argument("--parallel-round", action="store_true",
                    help="run the SELECTED round's three cells concurrently "
                         f"(run cap 3, slot pool {EXP3_PARALLEL_SLOTS}, "
                         f"{EXP3_PROVIDER_CONCURRENCY}-in-flight shared codex "
                         f"cap with {'/'.join(EXP3_PROVIDER_EXEMPT_ARMS)} exempt); "
                         "requires --round N")
    ap.add_argument("--cells", default="",
                    help="per-cell recovery: run ONLY these 'arm:replicate' "
                         "cells (e.g. 'B:2,AxK:2') and merge them into the "
                         "existing --out file, preserving every other cell")
    ap.add_argument("--print-cells", action="store_true",
                    help="print the expanded cell manifest and exit")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true",
                    help="orchestrate against in-memory fakes (no network, no spend)")
    ap.add_argument("--dry-validate", action="store_true",
                    help="run the orchestrator's own dry assertions (manifest "
                         "order, per-cell merge, failure reaping, keep list)")
    return ap


def build_config(args: argparse.Namespace) -> Tier1Config:
    if args.from_tier05:
        cfg = config_from_tier05(args.from_tier05)
    elif args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = Tier1Config(
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()}
        )
    else:
        cfg = Tier1Config()
    if args.exp2_block:
        # The pre-registered block, applied BEFORE the explicit flags below so
        # an operator can still override any single knob on the command line.
        cfg.exp2_block = True
        cfg.models = ("k3",)
        cfg.tasks = ("iron_plate_throughput",)
        cfg.replicates = 3
        cfg.arms = tuple(dict.fromkeys(arm for arm, _ in EXP2_BLOCK))
        cfg.T_s = EXP2_T_S
        cfg.K = EXP2_K
        cfg.m = EXP2_DEFAULT_M
        cfg.run_cap = 1
        cfg.out = "bench/results/exp2_block.json"
        cfg.journal_dir = "bench/journal/exp2"
        cfg.label = (
            "EXPERIMENT 2 -- dose-response on convergence frequency "
            f"(8 cells, S2 + k3, K={EXP2_K}, T={EXP2_T_S:.0f}s, dose floor "
            f"{EXP2_DOSE_FLOOR})"
        )
    if args.exp3_block:
        # Same shape as Exp 2's block above: pre-registered knobs first, so a
        # single explicit flag can still override any one of them.
        cfg.exp3_block = True
        cfg.models = ("codex/gpt-5.6-sol",)
        cfg.tasks = ("iron_plate_throughput",)
        cfg.replicates = 2
        cfg.arms = tuple(dict.fromkeys(arm for arm, _ in EXP3_BLOCK))
        cfg.T_s = EXP3_T_S
        cfg.leg_s = EXP3_P_S
        cfg.K = EXP3_K
        cfg.m = EXP2_DEFAULT_M
        cfg.run_cap = 1
        cfg.provider_concurrency = EXP3_PROVIDER_CONCURRENCY
        # Round 1's post-mortem, in two numbers. The lease must outlive the whole
        # round (2P of build clock + provisioning + the halftime regroup +
        # terminal probes), DERIVED from T so a resizing cannot leave it behind;
        # and one create gets a poll budget sized for a queued burst, not for a
        # solo create.
        cfg.ttl_s = exp3_ttl_s(cfg.T_s)
        cfg.create_deadline_s = EXP3_CREATE_DEADLINE_S
        cfg.out = "bench/results/exp3_block.json"
        cfg.journal_dir = "bench/journal/exp3-block"
        # The checkpoint and its template are pinned, and the template joins the
        # keep list: Exp 1's ids are dead, so a stale --from-snap would be a
        # 409 four cells deep instead of an error at argument-parse time.
        cfg.template_snap = EXP3_S2B
        cfg.keep = tuple(dict.fromkeys(cfg.keep + (EXP3_TEMPLATE_SNAP, EXP3_S2B)))
        mode = "sequential"
        if args.parallel_round:
            # PARALLEL ROUND: the three rungs of one round at once. Cap 3 runs, a
            # pool sized for the round's real peak, no probe-cadence stagger (it
            # exists to keep concurrent fork waves apart, and only ONE arm forks
            # here) -- but a PROVISIONING stagger in ladder order, because 17
            # simultaneous creates is what starved round 1's last seat.
            cfg.parallel_round = True
            cfg.run_cap = 3
            cfg.max_sandboxes = EXP3_PARALLEL_SLOTS
            cfg.stagger_s = 0.0
            cfg.provision_stagger_s = EXP3_PROVISION_STAGGER_S
            mode = (
                f"PARALLEL round (3 cells at once, {EXP3_PARALLEL_SLOTS} slots, "
                f"creates staggered {EXP3_PROVISION_STAGGER_S:.0f}s per rung in "
                f"ladder order, {EXP3_PROVIDER_CONCURRENCY}-in-flight shared "
                f"codex cap, {'/'.join(EXP3_PROVIDER_EXEMPT_ARMS)} exempt -> "
                "fan-out seats step ~2x slower, which DEFLATES the "
                "Control->A×K-S gap: conservative for the fork-value claim, and "
                "read as such)"
            )
        cfg.label = (
            "EXPERIMENT 3 -- three-arm ladder in rounds: Control (1 seat, no "
            "persona) -> A×K-S (8 persona seats, never converged) -> Hybrid (one "
            f"halftime regroup), one checkpoint + codex, K={EXP3_K}, "
            f"P={EXP3_P_S:.0f}s, T=2P={EXP3_T_S:.0f}s build-time-matched, width "
            f"floor {EXP3_WIDTH_FLOOR}; DIRECTIONAL, 2 rounds, round 2 on "
            f"explicit go; {mode}"
        )
    if args.arms:
        cfg.arms = tuple(a for a in args.arms.split(",") if a)
    if args.models:
        cfg.models = tuple(m for m in args.models.split(",") if m)
    if args.c_model:
        cfg.c_model = args.c_model
    if args.tasks:
        cfg.tasks = tuple(t for t in args.tasks.split(",") if t)
    if args.replicates:
        cfg.replicates = args.replicates
    if args.T:
        cfg.T_s = args.T
    if args.K:
        cfg.K = args.K
    if args.m:
        cfg.m = args.m
    if args.template_snap:
        cfg.template_snap = args.template_snap
    if args.keep:
        # ADDITIVE by contract: operator flags may only ever grow the keep set
        # (the pilot's between-block sweep ate the bake sandbox once because a
        # flag replaced the substrate the block was built on). Order is preserved
        # and duplicates collapse.
        cfg.keep = tuple(dict.fromkeys(
            cfg.keep + tuple(k.strip() for k in args.keep.split(",") if k.strip())
        ))
    if args.template_snap_id is not None:
        # Only when SUPPLIED: a non-empty argparse default used to overwrite the
        # template a --config file (or a block preset) had already chosen.
        cfg.template_snap_id = args.template_snap_id
    if args.cells:
        cfg.cells = tuple(c for c in args.cells.split(",") if c.strip())
    if args.round:
        cfg.round = args.round
    if args.node_cap:
        cfg.run_cap = args.node_cap
    if args.max_sandboxes:
        cfg.max_sandboxes = args.max_sandboxes
    if args.provider_concurrency:
        cfg.provider_concurrency = args.provider_concurrency
    if args.prefix:
        cfg.prefix = args.prefix
    if args.out:
        cfg.out = args.out
    if args.dry:
        cfg.dry = True
    if args.T and cfg.exp3_block:
        # Exp 3's leg and lease are DERIVED from T (P = T/2, lease = T + margin)
        # and the preset derived them from the pinned horizon, before this
        # override landed. Re-derive them or the block runs a leg of the old
        # half-T and a lease that expires mid-round -- exactly the round-1
        # failure the derivation exists to prevent.
        cfg.leg_s = 0.5 * cfg.T_s
        cfg.ttl_s = exp3_ttl_s(cfg.T_s)
        print(
            f"[tier1] --T {cfg.T_s:.0f}s overrides the Exp-3 horizon: "
            f"P={cfg.leg_s:.0f}s, lease={cfg.ttl_s}s re-derived. This is NOT the "
            f"pre-registered T={EXP3_T_S:.0f}s -- the results carry the override.",
            flush=True,
        )
    return cfg


#: Exp 1's baked checkpoint. Used by the dry validator as a stand-in --from-snap
#: seed, so the keep-list assertion runs against the id Exp 2 will really pass.
S2_SNAP = "snapshot-50964b569b820c6a"


async def dry_validate(**kw: Any) -> dict[str, Any]:
    """Run the orchestrator gate under exclusive roots, always releasing them."""
    journal_root = kw.pop("journal_root", "bench/journal/dry_validate")
    try:
        return await _dry_validate(journal_root=journal_root, **kw)
    finally:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(journal_root, ".lock"))


async def _dry_validate(
    *, workdir: str = "bench/results/dry_validate",
    journal_root: str = "bench/journal/dry_validate",
) -> dict[str, Any]:
    """The orchestrator's own launch-blocker assertions, against the fakes.

    What a pre-registered block cannot start without, none of which the arm dry
    run can see, because they are scheduling, validity and hygiene properties:

    b) ``--exp2-block`` expands EXACTLY the 8 contract cells, in priority order,
       and ``--exp3-block`` EXACTLY its 4, alternating so a cutoff leaves pairs.
    c) the validity floors are enforced on RESULTS: a short-dose B (Exp 2) and a
       truncated-wave Hybrid (Exp 3) are invalidated, kept out of the headline
       stats, and surfaced as ``--cells``-recoverable reruns -- while a healthy
       cell of the same arm is left alone.
    d) ``--cells`` re-runs only the named cells and MERGES into the existing
       results file, leaving every completed cell byte-identical.
    e) a cell that dies while provisioning has its orphans reaped BEFORE the
       next cell starts.
    f) the keep set always holds both TEMPLATE_SNAP and the --from-snap seed.

    Both roots are wiped at start and asserted over afterwards, so concurrent
    invocations would corrupt each other's evidence: each run claims its roots
    exclusively and falls back to a private sibling when they are already in
    use (:func:`bench.arms.claim_dry_journal_dir`).
    """
    journal_root, owned = claim_dry_journal_dir(journal_root)
    if not owned:
        workdir = f"{workdir}-{os.getpid()}"
    for path in (workdir, journal_root):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    if owned:  # rmtree above took the lock with it
        claim_dry_journal_dir(journal_root)
    report: dict[str, Any] = {
        "workdir": workdir,
        "journal_root": journal_root,
        "roots_exclusive": owned,
    }

    # -- (b) the Exp-2 manifest -------------------------------------------
    block_cfg = build_config(_cli().parse_args(["--exp2-block", "--dry"]))
    block_cells = expand_cells(block_cfg)
    expected = [
        f"{arm}|k3|iron_plate_throughput|r{rep}" for arm, rep in EXP2_BLOCK
    ]
    assert [c.key for c in block_cells] == expected, (
        f"--exp2-block expanded {[c.key for c in block_cells]}, "
        f"expected {expected}"
    )
    per_arm = {arm: sum(1 for c in block_cells if c.arm == arm)
               for arm in {c.arm for c in block_cells}}
    assert per_arm == {"B": 3, "AxK": 3, "Bonce": 1, "A": 1}, (
        f"block composition is {per_arm}, not the contract 3/3/1/1"
    )
    assert [c.arm for c in block_cells[:6]] == ["B", "AxK"] * 3, (
        "the three primary pairs must come first, interleaved as pairs"
    )
    assert [c.arm for c in block_cells[6:]] == ["Bonce", "A"], (
        "B-once then A-continue must be last, in that drop order"
    )
    assert block_cfg.T_s == EXP2_T_S and block_cfg.K == EXP2_K, (
        f"block runs at T={block_cfg.T_s}s K={block_cfg.K}, contract is "
        f"T={EXP2_T_S}s K={EXP2_K}"
    )
    assert block_cfg.run_cap == 1, "the Exp-2 block is a run-cap-1 block"
    # -- (b0) EVERY live block preset clears the pre-flight lease guard -------
    # The guard itself raises at ArmConfig construction (the arm dry gate proves
    # that on a live-shaped config). What belongs here is the other direction: no
    # preset this orchestrator can launch may trip it, and a future preset change
    # that would hibernate its own seats fails the gate instead of the round.
    lease_presets: dict[str, Tier1Config] = {
        "exp2_block": build_config(_cli().parse_args(["--exp2-block"])),
        "exp3_block": build_config(_cli().parse_args(["--exp3-block"])),
        "exp3_parallel": build_config(
            _cli().parse_args(["--exp3-block", "--round", "1", "--parallel-round"])
        ),
        "pilot_default": replace(Tier1Config(), template_snap=S2_SNAP),
    }
    lease_report: dict[str, Any] = {}
    for name, preset in lease_presets.items():
        arms_of = (expand_cells(preset) or [Cell("A", "k3", "iron_plate_throughput", 1)])
        for cell in arms_of:
            probe_cfg = ArmConfig(
                arm=cell.arm, model=cell.model, task_key=cell.task,
                T_s=preset.T_s, K=preset.K, m=preset.m, leg_s=preset.leg_s,
                template_snap=preset.template_snap or S2_SNAP,
                ttl_s=preset.ttl_s, create_deadline_s=preset.create_deadline_s,
                provision_stagger_s=preset.provision_stagger_s,
                run_id=f"lease-guard-{name}-{cell.arm}",
            )
            need = (probe_cfg.T_s + probe_cfg.provision_stagger_s
                    + probe_cfg.terminal_reserve_s + LEASE_GUARD_SLACK_S)
            assert probe_cfg.lease_guard, f"{name}: the guard was disabled"
            assert probe_cfg.ttl_s >= need, (
                f"{name}/{cell.arm}: lease {probe_cfg.ttl_s}s < {need:.0f}s "
                "required -- this preset would hibernate its own seats"
            )
        lease_report[name] = {
            "T_s": preset.T_s, "ttl_s": preset.ttl_s,
            "provision_stagger_s": preset.provision_stagger_s,
            "slack_s": LEASE_GUARD_SLACK_S,
            "headroom_s": round(
                preset.ttl_s - preset.T_s - preset.provision_stagger_s
                - min(90.0, 0.2 * preset.T_s) - LEASE_GUARD_SLACK_S, 1
            ),
        }
    report["lease_guard"] = lease_report

    report["exp2_block"] = {
        "n_cells": len(block_cells),
        "cells": [c.key for c in block_cells],
        "T_s": block_cfg.T_s, "K": block_cfg.K, "m": block_cfg.m,
        "run_cap": block_cfg.run_cap, "out": block_cfg.out,
        "dose_floor": EXP2_DOSE_FLOOR,
    }

    # -- (b2) the Exp-3 ladder manifest, its round gate and its two modes ----
    # Launch blockers: the six cells in the pinned order (three rungs x two
    # rounds, Hybrid/A×K-S/Control within a round so the decisive pair is
    # adjacent in time), the pinned horizon, a --round gate that really isolates
    # one round (rounds are reviewed between launches -- a --round 1 that
    # expanded to six cells would be a 15h commitment), and the parallel-round
    # capacity/provider regime.
    block3_cfg = build_config(_cli().parse_args(["--exp3-block", "--dry"]))
    block3_cells = expand_cells(block3_cfg)
    expected3 = [
        f"{arm}|codex/gpt-5.6-sol|iron_plate_throughput|r{rep}"
        for arm, rep in EXP3_BLOCK
    ]
    assert [c.key for c in block3_cells] == expected3, (
        f"--exp3-block expanded {[c.key for c in block3_cells]}, "
        f"expected {expected3}"
    )
    assert len(block3_cells) == 6, f"{len(block3_cells)} cells, contract is 6"
    assert [c.arm for c in block3_cells] == ["Hybrid", "AxK-S", "Control"] * 2, (
        "each Exp-3 round must run Hybrid, A×K-S, Control in that order"
    )
    assert sorted({c.arm for c in block3_cells}) == sorted(EXP3_LADDER), (
        f"the ladder is {sorted({c.arm for c in block3_cells})}, expected "
        f"{sorted(EXP3_LADDER)}"
    )
    assert "AxK2P" not in {c.arm for c in block3_cells}, (
        "the middle rung still expands under its old label"
    )
    rounds3: dict[int, list[str]] = {}
    for n in (1, 2):
        round_cfg = build_config(
            _cli().parse_args(["--exp3-block", "--dry", "--round", str(n)])
        )
        round_cells = expand_cells(round_cfg)
        assert round_cfg.round == n
        assert [c.key for c in round_cells] == [
            f"{arm}|codex/gpt-5.6-sol|iron_plate_throughput|r{n}"
            for arm, rep in EXP3_BLOCK if rep == n
        ], f"--round {n} expanded {[c.key for c in round_cells]}"
        assert len(round_cells) == 3 and {c.replicate for c in round_cells} == {n}, (
            f"--round {n} did not isolate one round: "
            f"{[c.key for c in round_cells]}"
        )
        # The documented --cells equivalent must select exactly the same cells.
        equiv_cfg = replace(
            build_config(_cli().parse_args(["--exp3-block", "--dry"])),
            cells=tuple(f"{arm}:{n}" for arm in EXP3_LADDER),
        )
        assert [c.key for c in expand_cells(equiv_cfg)] == [
            c.key for c in round_cells
        ], "--round N and its --cells equivalent disagree"
        rounds3[n] = [c.key for c in round_cells]
    # --dry clamps T (and P with it); the UNCLAMPED config is the contract.
    live3_cfg = build_config(_cli().parse_args(["--exp3-block"]))
    assert live3_cfg.T_s == EXP3_T_S == 2 * EXP3_P_S, (
        f"Exp 3 runs at T={live3_cfg.T_s}s; the contract is 2P={2 * EXP3_P_S}s"
    )
    assert live3_cfg.leg_s == EXP3_P_S and live3_cfg.K == EXP3_K, (
        f"Exp 3 runs P={live3_cfg.leg_s}s K={live3_cfg.K}, contract is "
        f"P={EXP3_P_S}s K={EXP3_K}"
    )
    assert live3_cfg.models == ("codex/gpt-5.6-sol",), (
        f"Exp 3 is a one-model block; got {live3_cfg.models}"
    )
    # The k3 relaunch: --models overrides the preset's model and NOTHING else,
    # and the tripwire keys on k3's provider, not on the dead codex quota.
    k3_cfg = build_config(_cli().parse_args(["--exp3-block", "--models", "k3"]))
    k3_cells = expand_cells(k3_cfg)
    assert k3_cfg.models == ("k3",), f"--models did not override: {k3_cfg.models}"
    assert [c.key for c in k3_cells] == [
        f"{arm}|k3|iron_plate_throughput|r{rep}" for arm, rep in EXP3_BLOCK
    ], f"--models k3 expanded {[c.key for c in k3_cells]}"
    assert {provider_of(c.model) for c in k3_cells} == {"kimi"}, (
        "a k3 cell would be judged against another provider's health"
    )
    same = ("T_s", "leg_s", "K", "m", "ttl_s", "create_deadline_s", "run_cap",
            "template_snap", "keep", "provider_concurrency", "exp3_block")
    assert all(getattr(k3_cfg, f) == getattr(live3_cfg, f) for f in same), (
        "the model override changed more than the model: "
        f"{[f for f in same if getattr(k3_cfg, f) != getattr(live3_cfg, f)]}"
    )
    assert live3_cfg.provider_concurrency == EXP3_PROVIDER_CONCURRENCY, (
        f"the block's shared codex cap is {live3_cfg.provider_concurrency}, "
        f"contract is {EXP3_PROVIDER_CONCURRENCY} in flight"
    )
    assert not live3_cfg.parallel_round and live3_cfg.run_cap == 1, (
        "sequential mode is the default; parallel must be asked for"
    )
    assert block3_cfg.leg_s == 0.5 * block3_cfg.T_s, (
        "the dry clamp must scale P with T or leg 1 swallows the run"
    )
    # Every rung is build-time-matched at 2P, and only the strict control is
    # narrow: the block hands K=8 to all three, so ArmConfig must collapse it.
    rung_cfgs = {
        arm: ArmConfig(
            arm=arm, model=live3_cfg.models[0], task_key=live3_cfg.tasks[0],
            T_s=live3_cfg.T_s, K=live3_cfg.K, m=live3_cfg.m,
            leg_s=live3_cfg.leg_s, template_snap=live3_cfg.template_snap,
            ttl_s=live3_cfg.ttl_s, create_deadline_s=live3_cfg.create_deadline_s,
            provision_stagger_s=EXP3_PROVISION_STAGGER_S,
            run_id=f"exp3-{arm}-contract",
        )
        for arm in EXP3_LADDER
    }
    assert all(c.T_s == EXP3_T_S for c in rung_cfgs.values()), (
        "a rung does not get the matched 2P build clock"
    )
    assert rung_cfgs["Control"].K == 1 and rung_cfgs["Control"].diversify == "never", (
        f"Control is not strict: K={rung_cfgs['Control'].K} "
        f"diversify={rung_cfgs['Control'].diversify!r}"
    )
    assert peak_sandboxes("Control", live3_cfg.K) == 1, (
        "the scheduler reserves K slots for the one-agent control"
    )
    assert all(rung_cfgs[a].K == EXP3_K for a in ("Hybrid", "AxK-S")), (
        "a wide rung lost its width"
    )
    # Parallel-round capacity: cap 3 and a pool sized for the round's real peak.
    par_cfg = build_config(
        _cli().parse_args(["--exp3-block", "--round", "1", "--parallel-round"])
    )
    assert par_cfg.parallel_round and par_cfg.round == 1
    assert par_cfg.run_cap == 3, f"parallel run cap {par_cfg.run_cap}, expected 3"
    assert par_cfg.max_sandboxes == EXP3_PARALLEL_SLOTS == 17, (
        f"parallel slot pool {par_cfg.max_sandboxes}, expected "
        f"{EXP3_PARALLEL_SLOTS} (8 + 8 + 1)"
    )
    assert par_cfg.max_sandboxes == sum(
        peak_sandboxes(arm, EXP3_K) for arm in EXP3_LADDER
    ), "the pool is not the sum of the round's per-arm peaks"
    assert par_cfg.stagger_s == 0.0, (
        "a parallel round must start together; the stagger only exists to keep "
        "concurrent fork waves apart, and only one arm forks here"
    )
    # Round-1 post-mortem, asserted on the LIVE config: the lease outlives the
    # whole round and is DERIVED from T, one create gets a burst-sized poll
    # budget, and the parallel round staggers creates in ladder order.
    assert live3_cfg.ttl_s == exp3_ttl_s(live3_cfg.T_s) == int(
        EXP3_T_S + EXP3_TTL_MARGIN_S
    ), f"the block's lease is {live3_cfg.ttl_s}s, not T + {EXP3_TTL_MARGIN_S:.0f}s"
    assert live3_cfg.ttl_s >= live3_cfg.T_s + EXP3_TTL_MARGIN_S, (
        f"lease {live3_cfg.ttl_s}s does not cover T={live3_cfg.T_s:.0f}s plus "
        f"{EXP3_TTL_MARGIN_S:.0f}s of margin -- seats would hibernate mid-round"
    )
    assert live3_cfg.ttl_s > EXP3_T_S, "the lease must outlive the build clock"
    assert live3_cfg.create_deadline_s == EXP3_CREATE_DEADLINE_S > 300.0, (
        f"create deadline {live3_cfg.create_deadline_s}s is not the block's "
        f"{EXP3_CREATE_DEADLINE_S}s (the wrapper's 300s default starved round 1's "
        "8th create)"
    )
    # Every rung inherits both, and Infra derives ONE lease string for creates
    # and forks alike, so a halftime refork child cannot get a shorter one.
    assert all(
        ArmConfig(
            arm=arm, model=live3_cfg.models[0], task_key=live3_cfg.tasks[0],
            T_s=live3_cfg.T_s, K=live3_cfg.K, m=live3_cfg.m, leg_s=live3_cfg.leg_s,
            template_snap=live3_cfg.template_snap, ttl_s=live3_cfg.ttl_s,
            create_deadline_s=live3_cfg.create_deadline_s,
            run_id=f"exp3-{arm}-ttl",
        ).ttl_s == live3_cfg.ttl_s
        for arm in EXP3_LADDER
    ), "a rung did not inherit the round's lease"
    assert par_cfg.provision_stagger_s == EXP3_PROVISION_STAGGER_S, (
        f"the parallel round staggers creates by {par_cfg.provision_stagger_s}s, "
        f"contract is {EXP3_PROVISION_STAGGER_S}s per rung"
    )
    assert EXP3_LADDER[0] == "Control" and EXP3_LADDER[-1] == "Hybrid", (
        "the stagger order must put the narrow rung first and the widest last"
    )
    assert build_config(_cli().parse_args(["--exp3-block"])).provision_stagger_s == 0.0, (
        "a sequential round must not stagger provisioning"
    )
    assert "PARALLEL" in par_cfg.label and "DEFLAT" in par_cfg.label.upper(), (
        "the parallel label must carry the pre-registered bias direction"
    )
    try:
        main(["--exp3-block", "--parallel-round", "--dry"])
    except SystemExit as exc:
        assert "--round" in str(exc), f"unexpected refusal: {exc}"
    else:  # pragma: no cover - the guard must exist
        raise AssertionError("--parallel-round without --round was accepted")
    report["exp3_block"] = {
        "n_cells": len(block3_cells),
        "cells": [c.key for c in block3_cells],
        "ladder": list(EXP3_LADDER),
        "rounds": rounds3,
        "round_selector": "--round N (== --cells "
                          f"{','.join(f'{a}:N' for a in EXP3_LADDER)})",
        "T_s": live3_cfg.T_s, "leg_s": live3_cfg.leg_s, "K": live3_cfg.K,
        "m": live3_cfg.m, "run_cap": live3_cfg.run_cap, "out": live3_cfg.out,
        "width_floor": EXP3_WIDTH_FLOOR,
        "control_K": rung_cfgs["Control"].K,
        "control_diversify": rung_cfgs["Control"].diversify,
        "control_peak_sandboxes": peak_sandboxes("Control", live3_cfg.K),
        "provider_concurrency": live3_cfg.provider_concurrency,
        "ttl_s": live3_cfg.ttl_s,
        "ttl_margin_s": EXP3_TTL_MARGIN_S,
        "create_deadline_s": live3_cfg.create_deadline_s,
        "parallel": {
            "run_cap": par_cfg.run_cap,
            "max_sandboxes": par_cfg.max_sandboxes,
            "stagger_s": par_cfg.stagger_s,
            "provision_stagger_s": par_cfg.provision_stagger_s,
            "provision_order": list(EXP3_LADDER),
            "exempt_arms": list(EXP3_PROVIDER_EXEMPT_ARMS),
            "requires_round": True,
        },
        "dry_T_s": block3_cfg.T_s, "dry_leg_s": block3_cfg.leg_s,
        "model_override": {
            "flag": "--models k3",
            "cells": [c.key for c in k3_cells],
            "provider": sorted({provider_of(c.model) for c in k3_cells}),
        },
    }

    # -- (c) the dose floor is enforced on RESULTS, not just on paper -------
    # A B-iterated endpoint whose journal shows fewer than EXP2_DOSE_FLOOR
    # convergences is INCONCLUSIVE by pre-registration: it must never reach the
    # paired contrast, and it must come back as a --cells-recoverable rerun.
    def dose_block(name: str, t_s: float, k: int) -> tuple[Tier1Config, str]:
        out = os.path.join(workdir, f"{name}_dry.json")
        cfg = replace(
            build_config(_cli().parse_args(["--exp2-block", "--dry"])),
            T_s=t_s, K=k, m=2, cells=("B:1", "AxK:1"), run_cap=1,
            max_sandboxes=16, stagger_s=0.0, template_snap=S2_SNAP,
            results_dir=workdir, journal_dir=os.path.join(journal_root, name),
            out=out, label=f"dry-validate {name}",
        )
        return cfg, out

    # K=8 makes the round estimate a 7-fork wave (0.59s at dry costs) and T
    # leaves 0.64s of rollout budget: round 1 is admitted by arithmetic, and a
    # second admission would need the first round to have cost under 0.05s,
    # which a 7-fork wave cannot. Dose 1 is structural here, not timing luck.
    starved_cfg, starved_out = dose_block("dose_starved", 0.8, 8)
    await Tier1Runner(starved_cfg).run()
    with open(starved_out, encoding="utf-8") as fh:
        starved = json.load(fh)
    b_run = next(r for r in starved["runs"] if r["arm"] == "B")
    axk_run = next(r for r in starved["runs"] if r["arm"] == "AxK")
    assert b_run["branch_points"] < EXP2_DOSE_FLOOR, (
        f"the starved block still converged {b_run['branch_points']} times; the "
        "dose assertions below would be vacuous"
    )
    assert b_run["status"] == "invalid_dose", (
        f"a dose-{b_run['branch_points']} B endpoint was accepted as "
        f"{b_run['status']!r}"
    )
    assert axk_run["status"] == "ok", "the control arm was invalidated too"
    assert starved["paired"]["B-AxK"]["n_pairs"] == 0, (
        "an invalid-dose endpoint reached the primary contrast: "
        f"{starved['paired']['B-AxK']}"
    )
    assert "B" not in starved["summary"]["normalized_endpoint_by_arm"], (
        "an invalid-dose endpoint was averaged into the headline stats"
    )
    assert len(starved["needs_rerun"]) == 1, (
        f"needs_rerun is {starved['needs_rerun']}"
    )
    entry = starved["needs_rerun"][0]
    assert entry["cell"] == b_run["cell"] and entry["dose"] == b_run["branch_points"]
    assert parse_cell_selector([entry["selector"]]) == {("B", 1)}, (
        f"the rerun selector {entry['selector']!r} is not --cells-compatible"
    )
    assert sorted(entry["pair"]) == sorted(
        [b_run["cell"], axk_run["cell"]]
    ), f"the rerun entry does not name its pair: {entry['pair']}"
    assert starved["dose_floor"] == EXP2_DOSE_FLOOR

    # Control: a block that DOES get its dose is not invalidated, so the rule
    # discriminates instead of failing everything shut.
    ok_cfg, ok_out = dose_block("dose_ok", 1.5, 2)
    await Tier1Runner(ok_cfg).run()
    with open(ok_out, encoding="utf-8") as fh:
        healthy = json.load(fh)
    b_ok = next(r for r in healthy["runs"] if r["arm"] == "B")
    assert b_ok["branch_points"] >= EXP2_DOSE_FLOOR, (
        f"the control block only reached dose {b_ok['branch_points']}"
    )
    assert b_ok["status"] == "ok", f"a valid-dose B was marked {b_ok['status']!r}"
    assert not healthy["needs_rerun"], f"spurious rerun: {healthy['needs_rerun']}"
    assert healthy["paired"]["B-AxK"]["n_pairs"] == 1, (
        "a valid pair did not reach the primary contrast"
    )
    report["dose_floor"] = {
        "floor": EXP2_DOSE_FLOOR,
        "starved": {"dose": b_run["branch_points"], "status": b_run["status"],
                    "paired_pairs": starved["paired"]["B-AxK"]["n_pairs"],
                    "needs_rerun": starved["needs_rerun"]},
        "control": {"dose": b_ok["branch_points"], "status": b_ok["status"],
                    "paired_pairs": healthy["paired"]["B-AxK"]["n_pairs"]},
    }

    # -- (c2) the Exp-3 width floor is enforced on RESULTS ------------------
    # Exp 3's analogue of the dose floor. A Hybrid endpoint whose refork wave was
    # truncated below EXP3_WIDTH_FLOOR seats is a best-of-few, not the max-over-8
    # both arms are supposed to share: it must be invalidated, kept out of the
    # headline stats, and come back as a --cells-recoverable rerun.
    def width_block(name: str, t_s: float) -> tuple[Tier1Config, str]:
        out = os.path.join(workdir, f"{name}_dry.json")
        cfg = replace(
            build_config(_cli().parse_args(["--exp3-block", "--dry"])),
            T_s=t_s, leg_s=0.5 * t_s, K=EXP3_K, m=2,
            cells=("Hybrid:1", "AxK-S:1"), run_cap=1, max_sandboxes=16,
            stagger_s=0.0, template_snap=S2_SNAP, results_dir=workdir,
            journal_dir=os.path.join(journal_root, name), out=out,
            label=f"dry-validate {name}",
        )
        return cfg, out

    # Structural, not timing luck: leg 1 spends T/2 and the terminal reserve is
    # 0.2T, so the refork wave opens with ~0.3T left. At the dry cost model a
    # single fork needs 0.05s + K x 0.005s = 0.09s of cleanup headroom, so at
    # T=0.2s (0.06s left) the wave cannot admit even one child and phase 2 is
    # judged on the winner alone.
    starved3_cfg, starved3_out = width_block("width_starved", 0.2)
    assert (starved3_cfg.T_s - 0.2 * starved3_cfg.T_s - starved3_cfg.leg_s
            < 0.05 + EXP3_K * 0.005), (
        "the starved Exp-3 block can still afford a fork; the width assertions "
        "below would be vacuous"
    )
    await Tier1Runner(starved3_cfg).run()
    with open(starved3_out, encoding="utf-8") as fh:
        starved3 = json.load(fh)
    hyb_run = next(r for r in starved3["runs"] if r["arm"] == "Hybrid")
    axks_run = next(r for r in starved3["runs"] if r["arm"] == "AxK-S")
    hyb_width = (hyb_run.get("exp3") or {}).get("validity") or {}
    assert hyb_width.get("k_effective", 99) < EXP3_WIDTH_FLOOR, (
        f"the starved Hybrid still judged {hyb_width.get('k_effective')} seats"
    )
    assert hyb_run["status"] == "invalid_width", (
        f"a {hyb_width.get('k_effective')}-seat Hybrid endpoint was accepted as "
        f"{hyb_run['status']!r}"
    )
    assert axks_run["status"] == "ok", "the never-converge rung was invalidated too"
    assert "Hybrid" not in starved3["summary"]["normalized_endpoint_by_arm"], (
        "an invalid-width endpoint was averaged into the headline stats"
    )
    assert len(starved3["needs_rerun"]) == 1, (
        f"needs_rerun is {starved3['needs_rerun']}"
    )
    entry3 = starved3["needs_rerun"][0]
    assert entry3["cell"] == hyb_run["cell"]
    assert entry3["k_effective"] == hyb_width["k_effective"]
    assert parse_cell_selector([entry3["selector"]]) == {("Hybrid", 1)}, (
        f"the rerun selector {entry3['selector']!r} is not --cells-compatible"
    )
    assert sorted(entry3["round_cells"]) == sorted(
        f"{arm}|codex/gpt-5.6-sol|iron_plate_throughput|r1" for arm in EXP3_LADDER
    ), f"the rerun entry does not name its round: {entry3['round_cells']}"
    assert entry3["round_selector"] == "--round 1" and entry3["round"] == 1
    assert starved3["width_floor"] == EXP3_WIDTH_FLOOR

    # Control: a Hybrid whose wave DOES staff phase 2 keeps its endpoint, so the
    # rule discriminates instead of failing every hybrid cell shut.
    ok3_cfg, ok3_out = width_block("width_ok", 6.0)
    await Tier1Runner(ok3_cfg).run()
    with open(ok3_out, encoding="utf-8") as fh:
        healthy3 = json.load(fh)
    hyb_ok = next(r for r in healthy3["runs"] if r["arm"] == "Hybrid")
    ok_width = (hyb_ok.get("exp3") or {}).get("validity") or {}
    assert ok_width.get("k_effective") == EXP3_K, (
        f"the control Hybrid materialised {ok_width.get('k_effective')} seats"
    )
    assert hyb_ok["status"] == "ok", f"a full-width Hybrid was {hyb_ok['status']!r}"
    assert not healthy3["needs_rerun"], f"spurious rerun: {healthy3['needs_rerun']}"
    assert len(hyb_ok["seat_endpoints"]) == EXP3_K, (
        "the results file lost per-seat endpoints"
    )
    assert hyb_ok["endpoint_throughput"] == max(
        s["throughput"] for s in hyb_ok["seat_endpoints"]
    ), "the block's endpoint is not the max over seats"
    report["width_floor"] = {
        "floor": EXP3_WIDTH_FLOOR,
        "starved": {"k_effective": hyb_width.get("k_effective"),
                    "status": hyb_run["status"],
                    "needs_rerun": starved3["needs_rerun"]},
        "control": {"k_effective": ok_width.get("k_effective"),
                    "status": hyb_ok["status"],
                    "seat_endpoints": [s["throughput"]
                                       for s in hyb_ok["seat_endpoints"]],
                    "endpoint": hyb_ok["endpoint_throughput"]},
    }

    # -- (c3) parallel-round mode really runs in parallel, under the caps ----
    # The pre-registered option: all three rungs of ONE round at once. What has
    # to hold is not the config but the behaviour -- the two fan-out cells
    # sharing one provider gate, Control on its private exemption, and all three
    # cells overlapping in wall clock. Anything less and the round is either
    # secretly sequential or secretly throttling the control.
    par_out = os.path.join(workdir, "parallel_round_dry.json")
    par_run_cfg = replace(
        build_config(_cli().parse_args(
            ["--exp3-block", "--round", "1", "--parallel-round", "--dry"]
        )),
        T_s=6.0, leg_s=3.0, m=2, template_snap=S2_SNAP, results_dir=workdir,
        # The lease is derived from T, so a dry T of seconds gets a dry lease --
        # same relationship, testable magnitude (main's --dry clamp does this too).
        ttl_s=exp3_ttl_s(6.0, margin_s=3.0), provision_stagger_s=0.05,
        journal_dir=os.path.join(journal_root, "parallel_round"), out=par_out,
        label="dry-validate parallel round",
    )
    assert par_run_cfg.run_cap == 3 and par_run_cfg.max_sandboxes == \
        EXP3_PARALLEL_SLOTS
    par_runner = Tier1Runner(par_run_cfg)
    par_payload = await par_runner.run()
    par_tail = assert_exp3_dry_block(par_payload, round_only=1)
    # The gate objects themselves, not just their journaled ids: the two fan-out
    # clients must hold gates whose ROOT is one and the same semaphore.
    gates = par_runner.cell_gates
    fan_out_gates = [g for k, g in gates.items()
                     if k.split("|")[0] in ("Hybrid", "AxK-S")]
    control_gates = [g for k, g in gates.items() if k.split("|")[0] == "Control"]
    assert len(fan_out_gates) == 2 and len(control_gates) == 1
    assert fan_out_gates[0].root is fan_out_gates[1].root, (
        "the fan-out cells hold different provider gates"
    )
    assert control_gates[0].root is not fan_out_gates[0].root, (
        "Control shares the fan-out gate; its exemption is a no-op"
    )
    assert control_gates[0].root.limit == 1 and control_gates[0].exempt
    assert fan_out_gates[0].root.limit == EXP3_PROVIDER_CONCURRENCY
    assert fan_out_gates[0].root.high_water <= EXP3_PROVIDER_CONCURRENCY, (
        f"the shared gate peaked at {fan_out_gates[0].root.high_water} over its "
        f"{EXP3_PROVIDER_CONCURRENCY} cap"
    )
    spans = _cell_spans(par_payload)
    report["parallel_round"] = {
        "cells": [r["cell"] for r in par_payload["runs"]],
        "run_cap": par_payload["caps"]["run_cap"],
        "max_sandboxes": par_run_cfg.max_sandboxes,
        "peak_sandboxes_used": par_payload["caps"]["peak_sandboxes_used"],
        "provider": {
            "shared_limit": fan_out_gates[0].root.limit,
            "shared_high_water": fan_out_gates[0].root.high_water,
            "shared_cells": sorted(k for k in gates
                                   if k.split("|")[0] != "Control"),
            "exempt_cell": next(k for k in gates if k.split("|")[0] == "Control"),
            "exempt_limit": control_gates[0].root.limit,
            "per_cell_high_water": {
                k: g.high_water for k, g in gates.items()
            },
        },
        "spans_s": {
            k: round(v[1] - v[0], 3) for k, v in spans.items()
        },
        "overlap_s": round(
            min(v[1] for v in spans.values()) - max(v[0] for v in spans.values()), 3
        ),
        "tail": par_tail,
    }

    # -- (c4) the provider tripwire aborts the block, online ------------------
    # The failure this exists for: the codex round burned 2.6h on ~17k straight
    # 429s before a human noticed. A dead provider must take down every cell that
    # shares its quota, mark them INVALID_PROVIDER with rerun selectors, leave no
    # sandbox behind and exit nonzero -- all from inside the run, with no operator.
    reset_provider_health()
    trip_out = os.path.join(workdir, "provider_dead_dry.json")
    trip_cfg = replace(
        build_config(_cli().parse_args(
            ["--exp3-block", "--round", "1", "--parallel-round", "--dry"]
        )),
        T_s=60.0, leg_s=30.0, m=2, template_snap=S2_SNAP, results_dir=workdir,
        # The dry substrate's provider IS the fake, so the cells name it: the
        # tripwire's question is "who shares this quota", and here that is the
        # fake client every cell holds.
        models=("fake-model",),
        ttl_s=exp3_ttl_s(60.0, margin_s=30.0), provision_stagger_s=0.0,
        journal_dir=os.path.join(journal_root, "provider_dead"), out=trip_out,
        label="dry-validate provider tripwire",
    )
    trip_runner = Tier1Runner(trip_cfg)
    # Fail-after-N on every cell's client, with a retry policy that makes each
    # attempt terminal, so the 30-failure budget is reached in seconds. T=60s is
    # far longer than the tripwire needs: if the abort did not work the cells
    # would run their full horizon and this section would time out instead of
    # silently passing.
    trip_world: dict[str, Any] = {"clients": []}

    def trip_llm(cell: Cell, journal: RunJournal):
        gate = trip_runner.cell_gate(cell)
        client = FakeLLM(journal=journal, log_full_requests=False,
                         max_concurrency=trip_cfg.K, semaphore=gate,
                         latency=0.002, fail_after=2, empty_every=0,
                         retry=RetryPolicy(attempts=1, base_s=0.0, jitter=0.0))
        trip_world["clients"].append(client)
        return client

    trip_runner.make_llm = trip_llm  # type: ignore[method-assign]
    trip_t0 = time.monotonic()
    trip_payload = await trip_runner.run()
    trip_wall_s = time.monotonic() - trip_t0
    dead = trip_payload["provider_dead"]
    assert dead is not None, "the tripwire never fired"
    assert dead["trigger"] == "consecutive_failures", dead
    assert dead["stats"]["consecutive_failures"] >= PROVIDER_DEAD_CONSECUTIVE, dead
    # Aborted, not merely finished: the block gave up long before its horizon.
    assert trip_wall_s < trip_cfg.T_s, (
        f"the block ran {trip_wall_s:.1f}s of a {trip_cfg.T_s:.0f}s horizon; the "
        "abort did not take"
    )
    assert not trip_payload["runs"], (
        f"a cell produced an endpoint against a dead provider: "
        f"{[r['cell'] for r in trip_payload['runs']]}"
    )
    trip_cells = {c.key for c in expand_cells(trip_cfg)}
    marked = {e["cell"] for e in trip_payload["needs_rerun"]
              if e.get("status") == "invalid_provider"}
    assert marked == trip_cells, (
        f"invalid_provider marked {sorted(marked)}, expected every cell of the "
        f"round {sorted(trip_cells)}"
    )
    for entry in trip_payload["needs_rerun"]:
        assert entry["provider"] == "fake" and entry["trigger"]
        assert parse_cell_selector([entry["selector"]]), entry
        assert entry["round_selector"] == "--round 1"
    # The journal carries the trigger stats, once, with the aborted cells named.
    trip_master = [
        r for r in _all_master_records(trip_cfg.journal_dir)
        if r.get("name") == "provider_dead"
    ]
    assert len(trip_master) == 1, f"{len(trip_master)} provider_dead records"
    assert trip_master[0]["stats"]["consecutive_failures"] >= PROVIDER_DEAD_CONSECUTIVE
    assert trip_master[0]["aborted"], "no sibling cell was aborted"
    # Zero leaks: the arms' own teardown plus the block reaper own every sandbox.
    assert not trip_payload["reaper"] or all(
        r.get("outcome") in ("deleted", "failed") for r in trip_payload["reaper"]
    ), trip_payload["reaper"]
    trip_health = trip_payload["provider"]["health"]["fake"]
    assert trip_health["successes"] > 0, "the fake never succeeded at all"
    report["provider_tripwire"] = {
        "trigger": dead["trigger"],
        "detail": dead["detail"],
        "consecutive_limit": PROVIDER_DEAD_CONSECUTIVE,
        "silence_limit_s": PROVIDER_DEAD_WINDOW_S,
        "stats": dead["stats"],
        "wall_s": round(trip_wall_s, 2),
        "horizon_s": trip_cfg.T_s,
        "cells": sorted(trip_cells),
        "invalid_provider": sorted(marked),
        "aborted_by_origin": trip_master[0]["aborted"],
        "endpoints_produced": len(trip_payload["runs"]),
        "reaped": len(trip_payload["reaper"]),
        "health": trip_health,
    }
    reset_provider_health()

    # -- (c) per-cell recovery merges into an existing results file --------
    merge_out = os.path.join(workdir, "merge_block_dry.json")
    merge_cfg = Tier1Config(
        models=("k3",), arms=("A", "B"), tasks=("iron_plate_throughput",),
        replicates=2, T_s=2.0, K=2, m=2, run_cap=1, max_sandboxes=4,
        stagger_s=0.0, template_snap=S2_SNAP, results_dir=workdir,
        journal_dir=os.path.join(journal_root, "merge"), out=merge_out,
        label="dry-validate merge block", dry=True,
    )
    await Tier1Runner(merge_cfg).run()
    with open(merge_out, encoding="utf-8") as fh:
        before = json.load(fh)
    target = "B|k3|iron_plate_throughput|r2"
    assert len(before["runs"]) == 4, (
        f"merge base should hold 4 completed cells, has {len(before['runs'])}"
    )
    untouched = {r["cell"]: r for r in before["runs"] if r["cell"] != target}
    assert len(untouched) == 3, f"expected 3 other cells, got {sorted(untouched)}"

    recover_cfg = replace(merge_cfg, cells=("B:2",))
    recover = Tier1Runner(recover_cfg)
    assert [c.key for c in expand_cells(recover_cfg)] == [target], (
        f"--cells 'B:2' selected {[c.key for c in expand_cells(recover_cfg)]}"
    )
    await recover.run()
    with open(merge_out, encoding="utf-8") as fh:
        after = json.load(fh)
    assert len(after["runs"]) == 4, (
        f"merge produced {len(after['runs'])} runs, expected the same 4 cells"
    )
    assert {r["cell"] for r in after["runs"]} == {r["cell"] for r in before["runs"]}
    for key, original in untouched.items():
        merged = next(r for r in after["runs"] if r["cell"] == key)
        assert merged == original, f"recovery pass rewrote completed cell {key}"
    assert after["merged_from"]["rerun_cells"] == [target], (
        f"merged_from records {after['merged_from']}"
    )
    assert sorted(after["merged_from"]["preserved_runs"]) == sorted(untouched), (
        "the preserved-cell ledger does not match the untouched cells"
    )
    assert after["config"]["cells"] == ["B:2"]
    report["per_cell_recovery"] = {
        "out": merge_out,
        "rerun": target,
        "preserved": sorted(untouched),
        "runs_after_merge": len(after["runs"]),
    }

    # -- (d) a dead cell is reaped before the next cell starts -------------
    fail_out = os.path.join(workdir, "fail_block_dry.json")
    fail_journal = os.path.join(journal_root, "fail")
    fail_cfg = replace(
        merge_cfg, arms=("A", "AxK", "B"), replicates=1, out=fail_out,
        journal_dir=fail_journal, label="dry-validate failure hygiene",
    )
    fail_cells = [c.key for c in expand_cells(fail_cfg)]
    assert len(fail_cells) == 3, f"failure block is {fail_cells}"
    dead_cell, next_cell = fail_cells[1], fail_cells[2]
    dead_run_id = "AxK-k3-iron_plate_throughput-r1"

    class _DeadBridge(FakeBridge):
        """A sandbox that comes up on the control plane and never gets healthy.

        The realistic orphan: ``create_from_snapshot`` succeeded, the health
        poll did not, so the arm never registered a Node and its own teardown
        cannot see the sandbox. Only the per-cell reaper can.
        """

        def wait_healthy(self, deadline_s: float = 300.0) -> None:
            raise TimeoutError("simulated health-poll timeout during provisioning")

    original_substrate = Tier1Runner.make_substrate

    def failing_substrate(self: Tier1Runner, run_id: str):
        fp, bridge_factory, template = original_substrate(self, run_id)
        if run_id == dead_run_id:
            return fp, (lambda url: _DeadBridge(url, fp.world)), template
        return fp, bridge_factory, template

    Tier1Runner.make_substrate = failing_substrate  # type: ignore[method-assign]
    try:
        fail_runner = Tier1Runner(fail_cfg)
        await fail_runner.run()
    finally:
        Tier1Runner.make_substrate = original_substrate  # type: ignore[method-assign]

    master_path = os.path.join(fail_journal, "tier1-master.jsonl")
    with open(master_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    events = [r for r in records if r.get("kind") == "event"]
    reaps = [i for i, e in enumerate(events)
             if e.get("name") == "cell_reaped" and e.get("cell") == dead_cell]
    starts = [i for i, e in enumerate(events)
              if e.get("name") == "run_start" and e.get("cell") == next_cell]
    assert len(reaps) == 1, f"dead cell {dead_cell} was reaped {len(reaps)} times"
    assert starts, f"cell 3 ({next_cell}) never started"
    assert reaps[0] < starts[0], (
        f"cell 3 started before the dead cell was reaped "
        f"(reap at {reaps[0]}, start at {starts[0]})"
    )
    swept = events[reaps[0]]
    assert swept["n"] >= 1, (
        "the per-cell reaper found nothing; the simulated orphan was not created"
    )
    with open(fail_out, encoding="utf-8") as fh:
        fail_payload = json.load(fh)
    assert len(fail_payload["reaper"]) >= swept["n"], (
        "the per-cell sweep was dropped from the final reaper ledger "
        f"({fail_payload['reaper']})"
    )
    assert {c["cell"] for c in fail_runner.failures} == {dead_cell}, (
        f"unexpected failures: {fail_runner.failures}"
    )
    assert {r["cell"] for r in fail_runner.results} == {
        fail_cells[0], next_cell
    }, f"the surviving cells did not both complete: {fail_runner.results}"
    report["failure_hygiene"] = {
        "dead_cell": dead_cell,
        "next_cell": next_cell,
        "reaped_before_next_start": True,
        "swept": swept["items"],
        "surviving_cells": sorted(r["cell"] for r in fail_runner.results),
    }

    # -- (e) the keep list always holds BOTH snapshot ids ------------------
    keep_cfg = replace(
        build_config(_cli().parse_args([
            "--exp2-block", "--dry", "--from-snap", S2_SNAP,
            "--keep", "sandbox-5a221b67a6973991",
        ])),
        results_dir=workdir, journal_dir=os.path.join(journal_root, "keep"),
        out=os.path.join(workdir, "keep_dry.json"),
    )
    assert keep_cfg.template_snap == S2_SNAP, "--from-snap did not set the seed"
    assert keep_cfg.template_snap_id == TEMPLATE_SNAP, (
        "TEMPLATE_SNAP is not carried separately from the run seed"
    )
    keep_runner = Tier1Runner(keep_cfg)
    keep = keep_runner.keep_ids()
    assert TEMPLATE_SNAP in keep, f"keep list {keep} drops TEMPLATE_SNAP"
    assert S2_SNAP in keep, f"keep list {keep} drops the --from-snap seed"
    assert "sandbox-5a221b67a6973991" in keep, "operator --keep ids were dropped"
    # A greenfield run (no --from-snap) still protects the template.
    bare = Tier1Runner(replace(keep_cfg, template_snap="", keep=())).keep_ids()
    assert bare == [TEMPLATE_SNAP], f"greenfield keep list is {bare}"
    # And the sweep honours it rather than only recording it.
    await keep_runner.reap()
    swept_ids = {r.get("id") for r in keep_runner.reaper_report}
    assert keep_runner.reaper_report, "the dry sweep did nothing at all"
    assert not (swept_ids & set(keep)), (
        f"the sweep deleted protected ids: {sorted(swept_ids & set(keep))}"
    )
    report["keep_list"] = {
        "keep": keep,
        "greenfield_keep": bare,
        "swept": sorted(i for i in swept_ids if i),
    }

    report["assertions"] = {
        "exp2_block_is_the_8_contract_cells_in_order": True,
        "cells_selector_runs_only_named_cells": True,
        "cells_selector_merges_preserving_completed": True,
        "dead_cell_reaped_before_next_cell_starts": True,
        "keep_list_holds_template_and_seed": True,
        #: reviewer2-2: the dose floor is enforced on results, not on paper.
        "dose_floor_invalidates_short_dose_B": True,
        "dose_floor_excludes_it_from_paired_contrast": True,
        "dose_floor_surfaces_a_cells_recoverable_rerun": True,
        "dose_floor_leaves_a_valid_dose_alone": True,
        #: Exp 3.
        "exp3_block_is_the_4_contract_cells_as_pairs": True,
        "exp3_block_runs_at_2P_with_P_and_K_pinned": True,
        "width_floor_invalidates_a_truncated_hybrid": True,
        "width_floor_surfaces_a_cells_recoverable_rerun": True,
        "width_floor_leaves_a_full_width_hybrid_alone": True,
        "exp3_per_seat_endpoints_survive_into_results": True,
        #: Exp-3 restructure + parallel-round mode.
        "exp3_block_is_the_6_cell_ladder_in_round_order": True,
        "exp3_round_selector_isolates_one_round": True,
        "exp3_round_selector_matches_its_cells_equivalent": True,
        "exp3_control_is_one_seat_no_persona_one_slot": True,
        "exp3_middle_rung_reads_AxK_S_everywhere": True,
        "exp3_parallel_round_requires_a_round": True,
        "exp3_parallel_round_caps_are_3_runs_and_17_slots": True,
        "exp3_parallel_fanout_cells_share_one_provider_gate": True,
        "exp3_parallel_control_holds_a_private_gate_of_1": True,
        "exp3_parallel_shared_cap_held_live": True,
        "exp3_parallel_cells_ran_concurrently": True,
        "exp3_provider_high_water_journaled_per_cell": True,
        #: Round-1 post-mortem fixes.
        "exp3_lease_is_T_plus_margin_derived_never_a_literal": True,
        "exp3_lease_covers_seats_and_refork_children_alike": True,
        "exp3_create_deadline_sized_for_a_queued_burst": True,
        "exp3_parallel_creates_staggered_in_ladder_order": True,
        #: Online provider tripwire (auto-abort).
        "provider_tripwire_aborts_the_block_online": True,
        "provider_tripwire_marks_every_cell_invalid_provider": True,
        "provider_tripwire_surfaces_rerun_selectors": True,
        "provider_tripwire_leaves_no_endpoint_and_no_leak": True,
        "provider_tripwire_journals_the_trigger_stats": True,
        "exp3_model_override_keeps_the_rest_of_the_preset": True,
    }
    return report


def assert_exp3_dry_block(payload: dict[str, Any], *, round_only: int = 0) -> str:
    """Assert Exp 3's block-level contract on a ``--exp3-block --dry`` payload.

    The arm dry gate (``python -m bench.arms --dry``) owns the mechanism
    assertions -- persona placement, rotation, selection, truncation, leaks, the
    control's negatives. What only the ORCHESTRATED block can show is that the
    manifest expanded to the right rungs in the right order (optionally one
    round), that each cell ran the arm it claims to, that the WIDE arms judged
    every seat at T with the max as the endpoint while Control judged exactly
    one, and that the labels and per-seat distributions survived into the results
    file the analysis will read. Raises ``AssertionError`` on any violation;
    returns the one-line summary printed as the dry tail.
    """
    runs = {r["cell"]: r for r in payload["runs"]}
    # The model and task come from the payload's OWN config: --models k3 is a
    # documented relaunch of this block (the codex quota died mid-round), and a
    # hard-coded codex key would fail the gate on a run that was entirely
    # correct. Absent config = no contract to assert against, so it fails closed.
    config = payload.get("config")
    assert isinstance(config, dict), "the payload carries no config to assert against"
    models = list(config.get("models") or ())
    tasks = list(config.get("tasks") or ())
    assert models and tasks, (
        f"the payload's config names no model/task (models={models}, tasks={tasks})"
    )
    model, task = models[0], tasks[0]
    expected = [
        f"{arm}|{model}|{task}|r{rep}"
        for arm, rep in EXP3_BLOCK
        if not round_only or rep == round_only
    ]
    assert expected, f"round {round_only} selects no cell of EXP3_BLOCK"
    assert not payload["failures"], f"cells failed: {payload['failures']}"
    actual = [r["cell"] for r in payload["runs"]]
    if payload.get("parallel_round"):
        # A parallel round completes out of manifest order by construction, so
        # only the SET is pinned here; the ORDER assertion belongs to the
        # manifest gate in :func:`_dry_validate`.
        assert sorted(actual) == sorted(expected), (
            f"the dry parallel round ran {actual}, expected {expected}"
        )
    else:
        assert actual == expected, (
            f"the dry block ran {actual}, expected {expected}"
        )
    # The renamed rung must be what the results file SAYS, not just what the
    # code calls it: no artifact may still read "AxK2P".
    labels = {r["arm"] for r in payload["runs"]}
    assert "AxK-S" in labels and "AxK2P" not in labels, (
        f"the payload's arm labels are {sorted(labels)}; the middle rung must "
        "read AxK-S"
    )
    assert payload["ladder"] == list(EXP3_LADDER)
    lines: list[str] = []
    for key in expected:
        run = runs[key]
        arm, exp3 = run["arm"], run.get("exp3") or {}
        seats = run.get("seat_endpoints") or []
        values = [s["throughput"] for s in seats if s["throughput"] is not None]
        wide = arm in ("Hybrid", "AxK-S")
        want_seats = EXP3_K if wide else 1
        assert exp3.get("K") == want_seats, (
            f"{key}: K={exp3.get('K')}, expected {want_seats}"
        )
        assert exp3.get("T_total_s") == run["T_s"] == 2 * exp3.get("P_s"), (
            f"{key}: T_total {exp3.get('T_total_s')} is not 2P -- every rung is "
            "build-time-matched"
        )
        assert len(seats) == want_seats, (
            f"{key}: {len(seats)} per-seat endpoints, expected {want_seats}"
        )
        assert len(values) == want_seats, f"{key}: a seat was never probed"
        assert run["endpoint_throughput"] == max(values), (
            f"{key}: endpoint {run['endpoint_throughput']} != max over seats "
            f"{max(values)}"
        )
        assert exp3.get("endpoint_max") == max(values)
        phases = exp3.get("phases") or []
        assert len(phases) == {"Hybrid": 2, "AxK-S": 1, "Control": 0}[arm], (
            f"{key}: {len(phases)} persona phase record(s)"
        )
        for phase in phases:
            assigned = [s["persona"] for s in phase["seats"].values()]
            assert len(assigned) == EXP3_K and all(assigned), (
                f"{key}: phase {phase['phase']} left a seat without a persona"
            )
            assert len(set(assigned)) == EXP3_K, (
                f"{key}: phase {phase['phase']} reused a persona across seats"
            )
        if arm == "Hybrid":
            p1 = [phases[0]["seats"][f"L1s{i}"]["persona"] for i in range(EXP3_K)]
            p2 = [phases[1]["seats"][f"L2s{i}"]["persona"] for i in range(EXP3_K)]
            assert all(a != b for a, b in zip(p1, p2)) and set(p1) == set(p2), (
                f"{key}: phase 2 did not rotate the persona assignment"
            )
            assert run["branch_points"] == exp3.get("dose") == 1, (
                f"{key}: dose {run['branch_points']}, Exp 3's hybrid converges once"
            )
            assert exp3["refork"]["k_effective"] == len(seats)
            assert run["status"] in ("ok", "invalid_width")
            lines.append(
                f"{arm} r{run['replicate']}: k_eff="
                f"{exp3['refork']['k_effective']}, overhead "
                f"{exp3['refork']['overhead_s']}s, {len(values)} endpoints, max "
                f"{run['endpoint_throughput']} ({run['status']})"
            )
        else:
            assert run["branch_points"] == 0 and run["snapshots_created"] == 0, (
                f"{key}: a never-converge rung converged"
            )
            if arm == "Control":
                # The strict rung: one seat, no persona, its own endpoint.
                assert run["sandboxes_created"] == 1, (
                    f"{key}: {run['sandboxes_created']} sandboxes for a "
                    "one-agent control"
                )
                assert exp3.get("persona") is None
                assert run["endpoint_source"] == "Control"
                lines.append(
                    f"{arm} r{run['replicate']}: 1 seat, no persona, endpoint "
                    f"{run['endpoint_throughput']} ({run['status']})"
                )
            else:
                lines.append(
                    f"{arm} r{run['replicate']}: {len(values)} endpoints, max "
                    f"{run['endpoint_throughput']} ({run['status']})"
                )
        # LEASE: every create AND every fork of this cell -- Hybrid's halftime
        # refork children included -- must carry ONE lease that outlives the whole
        # round. Round 1's seats hibernated at 7200s inside a ~8700s round and
        # both surviving cells came back PARTIAL with no terminal probe, so this is
        # read from the cell's own journal rather than from the config that asked
        # for it. (The exact T + margin derivation is asserted on the LIVE config
        # in :func:`_dry_validate`; a dry T of seconds scales the margin with it.)
        provisioning = [
            r for r in _infra_ops(run["journal_path"])
            if r.get("op") in ("create_from_snapshot", "fork")
        ]
        assert provisioning, f"{key}: no create/fork records in its journal"
        leases = {r.get("ttl") for r in provisioning}
        assert len(leases) == 1 and None not in leases, (
            f"{key}: creates and forks carry different leases {sorted(leases)} -- "
            "one of those sandboxes will hibernate before the endpoint"
        )
        lease_s = float(str(leases.pop()).rstrip("s"))
        assert lease_s > run["T_s"], (
            f"{key}: lease {lease_s}s does not outlive T={run['T_s']}s"
        )
        if not payload["config"].get("dry"):
            assert lease_s >= run["T_s"] + EXP3_TTL_MARGIN_S, (
                f"{key}: lease {lease_s}s < T={run['T_s']}s + "
                f"{EXP3_TTL_MARGIN_S:.0f}s margin"
            )
        creates = [r for r in provisioning if r["op"] == "create_from_snapshot"]
        assert creates, f"{key}: no create_from_snapshot records"
        assert all(r.get("deadline_s") == EXP3_CREATE_DEADLINE_S for r in creates), (
            f"{key}: create deadlines {sorted({r.get('deadline_s') for r in creates})}"
            f" != {EXP3_CREATE_DEADLINE_S}s (a queued create would starve)"
        )
        if arm == "Hybrid":
            forks = [r for r in provisioning if r["op"] == "fork"]
            assert forks, f"{key}: the halftime refork left no fork records"
            assert all(float(str(r["ttl"]).rstrip("s")) == lease_s for r in forks), (
                f"{key}: a halftime refork child got a different lease"
            )
        # Timing partition: every bucket of every cell sums to its wall clock.
        timings = run["timings"]
        total = sum(timings["attributed_s"].values())
        assert abs(total - timings["wall_s"]) < 1e-5, (
            f"{key}: buckets {total} != wall {timings['wall_s']}"
        )
    scope = f"round {round_only}" if round_only else "all rounds"
    if not payload.get("parallel_round"):
        return f"EXP3 DRY BLOCK OK ({scope}, sequential): " + "; ".join(lines) + "."
    lines.append(assert_exp3_parallel_round(payload))
    return f"EXP3 DRY BLOCK OK ({scope}, PARALLEL): " + "; ".join(lines) + "."


def assert_exp3_parallel_round(payload: dict[str, Any]) -> str:
    """Assert parallel-round mode really ran in parallel, under the right caps.

    Three properties, all read from evidence the block wrote rather than from the
    config that asked for them:

    * the two FAN-OUT cells' clients hang off the SAME provider gate, and the
      exempt cell's off a PRIVATE one of 1 (the pre-registered exemption -- if it
      silently shared, Control would be throttled and the fork-value rung
      inflated; if the fan-out arms did NOT share, they would be throttled
      unequally and the within-round contrast would be worthless);
    * every cell's in-flight high-water stayed inside its gate, and the shared
      gate's own high-water stayed inside the cap;
    * the cells genuinely OVERLAPPED in wall clock (per-cell
      ``run_start``/``run_done`` intervals from the master journal, plus a slot
      pool that held more than one cell's seats at once).
    """
    provider = payload["provider"]
    per_cell = provider["per_cell"]
    runs = {r["cell"]: r for r in payload["runs"]}
    fan_out = [c for c, g in per_cell.items() if not g["exempt"]]
    exempt = [c for c, g in per_cell.items() if g["exempt"]]
    assert len(fan_out) == 2 and len(exempt) == 1, (
        f"expected 2 shared + 1 exempt cell, got {sorted(per_cell)} with "
        f"exempt={exempt}"
    )
    assert all(c.split("|")[0] in ("Hybrid", "AxK-S") for c in fan_out), (
        f"the shared-gate cells are {fan_out}; only the fan-out arms share"
    )
    assert exempt[0].split("|")[0] in EXP3_PROVIDER_EXEMPT_ARMS, (
        f"{exempt[0]} is exempt but is not a pre-registered exempt arm"
    )
    shared_ids = {per_cell[c]["shared_gate_id"] for c in fan_out}
    assert len(shared_ids) == 1, (
        f"the fan-out cells hold DIFFERENT provider gates ({shared_ids}); they "
        "must share one in-flight cap"
    )
    assert per_cell[exempt[0]]["shared_gate_id"] not in shared_ids, (
        "the exempt cell shares the fan-out gate; the exemption is a no-op"
    )
    assert per_cell[exempt[0]]["shared_limit"] == 1, (
        f"the exempt cell's private gate is {per_cell[exempt[0]]['shared_limit']}"
        ", expected 1"
    )
    assert all(per_cell[c]["shared_limit"] == EXP3_PROVIDER_CONCURRENCY
               for c in fan_out), (
        f"the shared cap is not {EXP3_PROVIDER_CONCURRENCY}: "
        f"{[per_cell[c]['shared_limit'] for c in fan_out]}"
    )
    for cell, gate in per_cell.items():
        assert gate["acquisitions"] > 0, f"{cell}: the gate was never used"
        assert 0 < gate["in_flight_high_water"] <= gate["shared_limit"], (
            f"{cell}: in-flight high-water {gate['in_flight_high_water']} vs cap "
            f"{gate['shared_limit']}"
        )
        assert gate["in_flight_at_end"] == 0, f"{cell}: a call never released"
        assert runs[cell]["provider_gate"]["in_flight_high_water"] == \
            gate["in_flight_high_water"], (
            f"{cell}: the results file and the runner disagree on the high-water"
        )
    for name, gate in provider["shared_gates"].items():
        assert gate["shared_high_water"] <= gate["limit"], (
            f"provider gate {name} peaked at {gate['shared_high_water']} over its "
            f"{gate['limit']} cap"
        )
    # Concurrency, from the intervals the master journal recorded.
    spans = _cell_spans(payload)
    assert len(spans) == len(per_cell), (
        f"the master journal has spans for {sorted(spans)}, cells are "
        f"{sorted(per_cell)}"
    )
    items = sorted(spans.items())
    for i, (cell_a, (a0, a1)) in enumerate(items):
        for cell_b, (b0, b1) in items[i + 1:]:
            assert a0 < b1 and b0 < a1, (
                f"{cell_a} and {cell_b} did not overlap: {a0}-{a1} vs {b0}-{b1} "
                "-- the round ran sequentially"
            )
    caps = payload["caps"]
    assert caps["run_cap"] == 3, f"run cap {caps['run_cap']}, expected 3"
    assert payload["config"]["max_sandboxes"] == EXP3_PARALLEL_SLOTS, (
        f"slot pool {payload['config']['max_sandboxes']}, expected "
        f"{EXP3_PARALLEL_SLOTS}"
    )
    assert caps["peak_sandboxes_used"] > EXP3_K, (
        f"peak slots used {caps['peak_sandboxes_used']} <= one fan-out cell's "
        f"{EXP3_K}: the cells did not hold sandboxes at the same time"
    )
    assert caps["peak_sandboxes_used"] <= EXP3_PARALLEL_SLOTS
    # Provisioning stagger: ladder order, monotonic delays, and the creates
    # really started in that order (round 1 starved its last seat with 17
    # simultaneous creates).
    stagger = _stagger_records(payload)
    assert len(stagger) == len(per_cell), (
        f"provision_stagger records for {sorted(stagger)}, cells are "
        f"{sorted(per_cell)}"
    )
    by_arm = {c.split("|")[0]: r for c, r in stagger.items()}
    assert [by_arm[a]["rung"] for a in EXP3_LADDER] == list(range(len(EXP3_LADDER))), (
        f"the stagger is not in ladder order: "
        f"{[(a, by_arm[a]['rung']) for a in EXP3_LADDER]}"
    )
    delays = [by_arm[a]["delay_s"] for a in EXP3_LADDER]
    assert delays[0] == 0.0 and delays == sorted(delays) and delays[-1] > 0.0, (
        f"the stagger delays are not increasing from the narrow rung: {delays}"
    )
    starts = {c: v[0] for c, v in spans.items()}
    ordered = [starts[c] for a in EXP3_LADDER
               for c in stagger if c.split("|")[0] == a]
    assert ordered == sorted(ordered), (
        f"the cells' creates did not start in ladder order: {ordered}"
    )
    hw = {c.split("|")[0]: g["in_flight_high_water"] for c, g in per_cell.items()}
    stagger_txt = "/".join(
        "{}+{:g}s".format(a, by_arm[a]["delay_s"]) for a in EXP3_LADDER
    )
    return (
        f"parallel: 3 cells overlapped, peak {caps['peak_sandboxes_used']}/"
        f"{EXP3_PARALLEL_SLOTS} slots, creates staggered {stagger_txt}, "
        f"shared codex gate {EXP3_PROVIDER_CONCURRENCY} in flight for "
        f"{'+'.join(sorted(c.split('|')[0] for c in fan_out))}, "
        f"{exempt[0].split('|')[0]} exempt on a private 1, per-cell high-water "
        f"{hw}"
    )


def _all_master_records(journal_dir: str) -> list[dict[str, Any]]:
    """Every record of a block's master journal (append-only across launches)."""
    out: list[dict[str, Any]] = []
    path = os.path.join(journal_dir, "tier1-master.jsonl")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _infra_ops(journal_path: str) -> list[dict[str, Any]]:
    """Every ``infra_op`` record of one cell's run journal."""
    out: list[dict[str, Any]] = []
    with open(journal_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "infra_op":
                out.append(rec)
    return out


def _stagger_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{cell: provision_stagger record}`` for THIS invocation."""
    path = os.path.join(payload["config"]["journal_dir"], "tier1-master.jsonl")
    since = float(payload.get("started_at") or 0.0)
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("name") == "provision_stagger" and rec.get("cell")
                    and float(rec.get("ts") or 0.0) >= since):
                out[str(rec["cell"])] = rec
    return out


def _cell_spans(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """``{cell: (run_start ts, run_done ts)}`` for THIS invocation's cells.

    The master journal is append-only across invocations (a relaunched block must
    keep its predecessor's evidence), so records are filtered to the window this
    payload describes -- otherwise a previous sequential dry run's spans would
    answer questions about this parallel one.
    """
    path = os.path.join(payload["config"]["journal_dir"], "tier1-master.jsonl")
    since = float(payload.get("started_at") or 0.0)
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name, cell = rec.get("name"), rec.get("cell")
            if not cell or float(rec.get("ts") or 0.0) < since:
                continue
            if name == "run_start":
                starts.setdefault(cell, float(rec["ts"]))
            elif name in ("run_done", "run_cancelled"):
                ends[cell] = float(rec["ts"])
    return {c: (starts[c], ends[c]) for c in starts if c in ends}


def _expand_or_exit(cfg: Tier1Config) -> list[Cell]:
    """Expand a config's matrix, turning a matrix error into a clean refusal."""
    try:
        return expand_cells(cfg)
    except ValueError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.dry_validate:
        report = asyncio.run(dry_validate())
        print(json.dumps(report, indent=2, default=str))
        block = report["exp2_block"]
        rec = report["per_cell_recovery"]
        hyg = report["failure_hygiene"]
        print(
            f"\nDRY VALIDATE OK: {block['n_cells']} contract cells in order "
            f"({', '.join(block['cells'])}) at T={block['T_s']}s K={block['K']} "
            f"m={block['m']} run_cap={block['run_cap']}."
        )
        print(
            f"Recovery: re-ran {rec['rerun']} into {rec['out']}, preserved "
            f"{len(rec['preserved'])} completed cells untouched."
        )
        print(
            f"Hygiene: {hyg['dead_cell']} died in provisioning, "
            f"{len(hyg['swept'])} resource(s) reaped before {hyg['next_cell']} "
            f"started."
        )
        print(f"Keep list: {', '.join(report['keep_list']['keep'])}.")
        block3, width = report["exp3_block"], report["width_floor"]
        par = report["parallel_round"]
        print(
            f"Exp-3 ladder: {block3['n_cells']} cells = "
            f"{'/'.join(block3['ladder'])} x {len(block3['rounds'])} rounds "
            f"({', '.join(block3['cells'])}) at P={block3['leg_s']:.0f}s, "
            f"T=2P={block3['T_s']:.0f}s, K={block3['K']} "
            f"(Control K={block3['control_K']}, "
            f"diversify={block3['control_diversify']!r}, "
            f"{block3['control_peak_sandboxes']} slot), m={block3['m']}, "
            f"width floor {block3['width_floor']}; round gate "
            f"{block3['round_selector']}."
        )
        print(
            f"Width floor: a wave truncated to k_eff="
            f"{width['starved']['k_effective']} came back "
            f"{width['starved']['status']} with a rerun selector; a "
            f"k_eff={width['control']['k_effective']} hybrid stayed "
            f"{width['control']['status']} and reported all "
            f"{len(width['control']['seat_endpoints'])} seat endpoints "
            f"(endpoint {width['control']['endpoint']} = max)."
        )
        trip = report["provider_tripwire"]
        print(
            f"Provider tripwire: {trip['trigger']} after "
            f"{trip['stats']['failures']} terminal failure(s) -> block aborted in "
            f"{trip['wall_s']}s of a {trip['horizon_s']:.0f}s horizon, "
            f"{len(trip['invalid_provider'])}/{len(trip['cells'])} cells "
            f"invalid_provider with rerun selectors, "
            f"{trip['endpoints_produced']} endpoints kept, "
            f"{trip['reaped']} resource(s) reaped."
        )
        print(
            f"Parallel round: {len(par['cells'])} cells overlapped for "
            f"{par['overlap_s']}s at run_cap={par['run_cap']}, peak "
            f"{par['peak_sandboxes_used']}/{par['max_sandboxes']} slots; "
            f"{len(par['provider']['shared_cells'])} fan-out cells shared one "
            f"{par['provider']['shared_limit']}-in-flight codex gate (peak "
            f"{par['provider']['shared_high_water']}), "
            f"{par['provider']['exempt_cell'].split('|')[0]} exempt on a private "
            f"{par['provider']['exempt_limit']}."
        )
        return 0
    cfg = build_config(args)
    if cfg.dry:
        cfg.journal_dir = cfg.journal_dir.rstrip("/") + "-dry"
        cfg.out = cfg.out.replace(".json", "_dry.json")
        cfg.T_s = min(cfg.T_s, 6.0)
        # Exp 3's legs are halves of T by definition; a dry T of seconds must
        # scale P with it or leg 1 would swallow the whole run. The LEASE is
        # re-derived from the clamped T for the same reason -- the invariant the
        # gate checks is ttl == T + margin, not a magnitude.
        if cfg.leg_s:
            cfg.leg_s = 0.5 * cfg.T_s
        if cfg.exp3_block:
            cfg.ttl_s = exp3_ttl_s(cfg.T_s, margin_s=0.5 * cfg.T_s)
        cfg.stagger_s = 0.0
        # The provisioning stagger keeps its ORDER (which is what the gate
        # asserts) at a dry magnitude, so a 3-cell dry round stays seconds long.
        if cfg.provision_stagger_s:
            cfg.provision_stagger_s = 0.05
    elif not cfg.template_snap:
        raise SystemExit(
            "refusing to run: a seed snapshot (--template-snap / --from-snap) is "
            "required for real runs; use --dry to exercise the orchestrator."
        )
    if not cfg.tasks or not cfg.models:
        raise SystemExit("refusing to run: config has no tasks or no models")
    if cfg.parallel_round and not cfg.round:
        raise SystemExit(
            "refusing to run: --parallel-round runs ONE round's three cells at "
            f"once and its slot pool is sized for that round's peak "
            f"({EXP3_PARALLEL_SLOTS}); pass --round N"
        )
    if cfg.cells:
        try:
            parse_cell_selector(cfg.cells)
        except ValueError as exc:
            raise SystemExit(f"refusing to run: {exc}") from exc
        if not _expand_or_exit(cfg):
            whole = [c.key for c in _expand_or_exit(replace(cfg, cells=()))]
            raise SystemExit(
                f"refusing to run: --cells {','.join(cfg.cells)} selected no "
                f"cell of this block ({whole})"
            )
    if cfg.round and not _expand_or_exit(cfg):
        whole = [c.key for c in _expand_or_exit(replace(cfg, round=0))]
        raise SystemExit(
            f"refusing to run: --round {cfg.round} selected no cell of this "
            f"block ({whole})"
        )

    if args.print_cells:
        cells = _expand_or_exit(cfg)
        print(json.dumps(
            {"n_cells": len(cells), "cells": [c.key for c in cells],
             "round": cfg.round or None, "T_s": cfg.T_s, "leg_s": cfg.leg_s,
             "K": cfg.K, "m": cfg.m, "run_cap": cfg.run_cap, "out": cfg.out},
            indent=2,
        ))
        return 0

    try:
        # Both the merge-base fingerprint and the capacity pre-flight refuse here,
        # before a single sandbox exists: a refusal is an exit code, never a
        # traceback over a half-provisioned block.
        runner = Tier1Runner(cfg)
        payload = asyncio.run(runner.run())
    except ValueError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc
    print(json.dumps(
        {
            "label": payload["label"],
            "interrupted": payload["interrupted"],
            "runs": len(payload["runs"]),
            "failures": payload["failures"],
            "skipped": payload["skipped"],
            "caps": payload["caps"],
            "summary": payload["summary"],
            "paired": {
                k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                for k, v in payload["paired"].items() if isinstance(v, dict)
            },
            "reaped": len(payload["reaper"]),
            "out": cfg.out,
        },
        indent=2, default=str,
    ))
    dead = payload.get("provider_dead")
    if dead:
        # One line a human can act on, and a nonzero exit so a wrapper script
        # cannot mistake an aborted block for a finished one. Partial results are
        # already written and every affected cell is on needs_rerun, so the block
        # assertions below (which expect a COMPLETE block) are skipped.
        rerun = payload["needs_rerun"]
        selector = rerun[0]["round_selector"] if rerun else "--round N"
        print(
            f"\nABORTED: {dead['detail']} -- {len(rerun)} cell(s) marked "
            f"invalid_provider, rerun with {selector}; partial results in "
            f"{cfg.out}",
            flush=True,
        )
        return 2
    if cfg.exp3_block and cfg.dry:
        print(assert_exp3_dry_block(payload, round_only=cfg.round))
    # Exit codes (C7): 0 means a COMPLETE matrix. A wrapper that reads 0 as "the
    # block ran" must never see it for a block that lost a cell -- failed cells,
    # cells the graceful stop never admitted, an interrupt, or an exception no
    # cell accounted for are each incomplete, and each exits 1.
    problems: list[str] = []
    if payload["failures"]:
        problems.append(f"{len(payload['failures'])} failed cell(s)")
    if payload["skipped"]:
        problems.append(f"{len(payload['skipped'])} skipped cell(s): "
                        f"{', '.join(payload['skipped'])}")
    if payload["interrupted"]:
        problems.append("interrupted before the matrix finished")
    if payload.get("unaccounted"):
        problems.append(
            f"{len(payload['unaccounted'])} unaccounted exception(s) -- a bug in "
            "the per-cell recording scope"
        )
    if problems:
        print(f"\nINCOMPLETE: {'; '.join(problems)}; results in {cfg.out}",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
