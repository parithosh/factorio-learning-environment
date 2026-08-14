"""The arm loops of the Farplane fan-out benchmark (design v2.5/v2.6).

Arms
----
``A``    sequential agent, one sandbox, until wall clock T.
``AxK``  K independent A trajectories in parallel; best terminal probe wins.
         The decisive control: if A×K >= B, convergent fan-out adds nothing.
``B``    Farplane fan-out: every m steps snapshot main, fork K-1 children
         (overlapped with sampling K candidates), run each branch m steps,
         score, promote the winner's sandbox to main, delete the rest.
``Bonce`` one seat to the first round boundary at or past T/2, then EXACTLY one
         fork wave + m-step branch rollout + selection; the winner continues to
         T. Exp 2's descriptive curve point between ``AxK`` (never converges)
         and ``B`` (converges every m steps).
``C``    same loop as B, but branches are made with GameState save/restore onto
         a pool of K-1 sandboxes.

Invariants this module is responsible for
-----------------------------------------
P3  no per-step ``verify()``; scoring is ONE fixed 60s window (bridge ``/probe``
    + ``FLE_BENCH_MODE``), run at a fixed cadence, never a plateau loop.
P4  conversation state is host-side: branch = deep-copied common prefix +
    candidate turn + its m turns + pending feedback; promotion adopts the
    winner's conversation wholesale; losers are journal artifacts only.
P5  per-branch baseline right after fork/restore, endpoint right after the m-th
    program and before any probe; tie-break = automated-production delta.
v2.3 wall clock T is the SOLE stopping rule: no quota termination, no step cap.
v2.6 EVERY arm gets the same fixed 60s probe every m steps, executed DIRECTLY on
    the sandbox that owns the line (zero measurement forks -- the Tier-0 gate
    showed the fork lane does not fit inside a sampling round), injected into
    the conversation in one fixed format, its full cost charged to T.

Experiment 2 (dose-response) additions
--------------------------------------
* SEED SNAPSHOT. ``ArmConfig.template_snap`` is only an id: TEMPLATE_SNAP for a
  greenfield run, a baked checkpoint (S2/S3) for the ``-from-S`` variants.
  A×K-from-S is A×K with ``template_snap=<checkpoint>`` and nothing else.
* HINT ROTATION. Arm B re-seeds K branches every round, so the divergent
  strategy set is rotated by round index (``ArmRun.hints_for(offset=...)``) and
  each branch carries its hint in its own first post-fork user turn.
* ROUND LENGTH. ``m`` defaults to Exp 2's sizing (:func:`exp2_round_sizing`),
  which hides B's serial fork wave inside a round with margin.

Experiment 3 (three-arm ladder, run in iterative rounds) additions
------------------------------------------------------------------
``Control`` ONE agent, NO persona, the same checkpoint, the full T = 2P. Arm A's
         loop verbatim: no forking benefit of any kind, endpoint = the terminal
         probe of its single factory. The bottom rung.
``AxK-S``  8 persona seats forked once from the checkpoint, never converged,
         T = 2P, endpoint = max over 8 at T. (Registered as "A×K-2P".)
``Hybrid`` 8 persona seats -> leg 1 of P -> ONE selection (the fixed probe on all
         8) -> snapshot the winner, delete the losers, refork 7 children beside
         it -> personas re-injected ROTATED -> leg 2 of P -> probe ALL 8, max.
The ladder isolates one ingredient per rung: Control -> A×K-S is the value of
forking wide, A×K-S -> Hybrid the value of one convergence. Build time is MATCHED
at 2P for every arm; Hybrid's regroup is EXTRA wall clock, measured and reported
as the price of convergence, never taken out of a leg.
* PERSONAS. :data:`PERSONAS` replaces :data:`DIVERSITY_HINTS` for the two WIDE
  Exp-3 arms (``ArmConfig.__post_init__``); Control is diversity-free by
  construction and gets the neutral prompt every other arm's seat 0 gets. A
  persona rides each seat's FIRST user turn through the same
  ``bench.llm.HINT_TEMPLATE`` channel, exactly once per phase, and Hybrid's
  phase 2 rotates the assignment by one seat.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence

from bench.common import (
    Budget,
    BudgetExhausted,
    Curve,
    RunJournal,
    ScoreRecord,
    TimingBuckets,
    atomic_write_json,
    new_run_id,
    resource_name,
)
from bench.llm import (
    HINT_TEMPLATE,
    LLMClient,
    ProviderDead,
    Sample,
    make_client,
)

ARMS = ("A", "AxK", "B", "Bonce", "C", "Hybrid", "AxK-S", "Control")

#: Experiment 3's ladder, bottom rung first. Every rung shares the checkpoint,
#: the horizon (T = 2P), the probe cadence and the token cap; what changes is one
#: ingredient per rung -- width, then convergence.
EXP3_ARMS = ("Control", "AxK-S", "Hybrid")
#: The WIDE Exp-3 arms: the ones whose seats carry personas. ``Control`` is
#: deliberately absent -- it is the no-persona, no-fork rung, so handing it a
#: persona would make it a 1-seat A×K-S instead of the strict control.
EXP3_PERSONA_ARMS = ("AxK-S", "Hybrid")

# ---------------------------------------------------------------------------
# Substrate protocols (bench.farplane / bench.bridge_client, or the fakes below)
# ---------------------------------------------------------------------------


class SBLike(Protocol):
    id: str
    name: str
    node: str
    base_url: str | None


class FarplaneLike(Protocol):
    timings: list

    def create_from_template(self, template: str, ttl: int, vcpu: int | None = ...,
                             mem: int | None = ..., name: str = ...) -> SBLike: ...
    def create_from_snapshot(self, snap_id: str, ttl: int, name: str = ...,
                             *, deadline: float | None = ...) -> SBLike: ...
    def snapshot(self, sb: SBLike) -> str: ...
    def fork(self, snap_id: str, ttl: int, name: str = ...,
             *, deadline: float | None = ...) -> SBLike: ...
    def expose(self, sb: SBLike, port: int) -> str: ...
    def exec(self, sb: SBLike, cmd: str) -> str: ...
    def delete_sandbox(self, sb: SBLike) -> None: ...
    def delete_snapshot(self, snap_id: str) -> None: ...
    def reaper(self, prefix: str | None = ..., *, dry_run: bool = ...,
               keep: Sequence[str] = ...) -> list: ...


class BridgeLike(Protocol):
    base_url: str

    def health(self) -> bool: ...
    def wait_healthy(self, deadline_s: float = ...) -> None: ...
    def execute(self, code: str) -> dict: ...
    def probe(self, entity: str) -> dict: ...
    def state_save(self) -> str: ...
    def state_restore(self, state: str) -> None: ...
    def system_prompt(self) -> str: ...
    def meta(self) -> dict: ...


BridgeFactory = Callable[[str], BridgeLike]


# ---------------------------------------------------------------------------
# Prompt / feedback formats (fixed across arms -- parity requirement)
# ---------------------------------------------------------------------------

GOAL_TEMPLATE = """## Task
{goal}

You act by writing Python programs against the Factorio API described above.
Each program you write is executed in the live game; its stdout/stderr and the
resulting production statistics come back as the next observation. Work
incrementally: build, verify, then extend. There is no step limit -- you are
stopped by a wall-clock budget, so prefer actions that compound.

Write the next program now, wrapped in a ```python code block."""

FEEDBACK_TEMPLATE = """## Step {step} Execution Results

**Program Output (STDOUT/STDERR):**
```
{output}
```

**Performance Results:**
- Production score: {production_score:.1f} (was {previous_score:.1f})
- Score change: {delta:+.1f}
- Automated production score: {automated_score:.1f}
- Elapsed game time: {elapsed}
- Ticks: {ticks} (cost +{ticks_cost})

Continue to step {next_step}."""

#: v2.6: identical format in every arm, so no arm gets a better instrument.
#: The probe now runs on the line's own sandbox, so the wording states the
#: real side effect (60 in-game seconds of advance) rather than v2.4's
#: disposable-copy claim.
PROBE_TEMPLATE = """## Objective Throughput Measurement (fixed 60s window)

Your factory ran unmodified for one fixed 60 in-game-second window while this
measurement was taken. Nothing else was changed.

- Target item: {entity}
- Measured throughput: {throughput:.2f} {entity} per 60 in-game seconds
- Quota for this task: {quota} per 60 in-game seconds
- Measurement window: ticks {start_tick} -> {end_tick}"""

PARSE_FAILURE_FEEDBACK = (
    "Could not parse a program from your response. Be sure to wrap your code in "
    "```python blocks. No program was executed; the game state is unchanged."
)

EXEC_ERROR_FEEDBACK = """## Step {step} Execution Error

The harness could not execute your program against the game:
```
{error}
```
The game state may be unchanged. Write the next program."""

#: Pre-registered diversification knob for temperature-locked models (DIVERSITY
#: GATE): the EIGHT divergent strategies Exp 1 measured branch decorrelation
#: with, verbatim (``bench.exp1.STRATEGY_HINTS`` keeps the frozen labelled copy
#: that Exp 1 ran). Eight is also Exp 2's K, so one B round assigns a full
#: permutation of the set. A×K takes them positionally -- seat i gets hint i,
#: protocol-identical to an Exp 1 wave; arm B rotates them by round index (see
#: :meth:`ArmRun.hints_for`), because a fixed assignment would hand every
#: re-seeded seat the same strategy round after round and the round-2+ branches
#: would be clones by construction.
DIVERSITY_HINTS: tuple[str, ...] = (
    "Expand the factory WIDE: add more mining drills and more furnaces "
    "together, in parallel, so ore supply and smelting capacity grow at the "
    "same time. Do not stop to optimise -- add capacity.",
    "Rebuild the line's RATIOS from scratch: work out how many drills feed "
    "how many furnaces and what the belt can carry, then lay the stages out "
    "matched to those numbers instead of extending what is there.",
    "Build an INDEPENDENT SECOND production cell somewhere else: its own "
    "miners, its own furnaces, its own output chest. Leave the existing "
    "line untouched so the two cells add up.",
    "Fix and extend POWER first -- offshore pump, boilers, steam engines, "
    "poles with real coverage -- and then ELECTRIFY mining so drills never "
    "stall for fuel.",
    "Optimise the LOGISTICS of the existing line only: inserters, belt "
    "routing, buffering and hand-off points. Add no new smelting capacity; "
    "make what exists move plates without stalling.",
    "Go VERTICAL on the ore you already deliver: keep the current patch and "
    "current miners, and stack many more furnaces onto that same ore flow.",
    "TEAR DOWN the current layout and re-lay it COMPACTLY: shortest belts, "
    "fewest transfer hops, furnaces packed next to the drills.",
    "DIVERSIFY the ore supply: prospect a different iron patch, build a new "
    "mining outpost there and run its ore into smelting.",
)

#: Experiment 3's diversity channel: EIGHT personas of well-known Factorio
#: players/styles, in the design's pre-registered order, replacing
#: :data:`DIVERSITY_HINTS` for the Exp-3 arms. Each is a behavioural
#: instruction, not a flavour label -- the model is temperature-locked, so the
#: only diversity available is the one written into the prompt, and a persona
#: that does not change what gets BUILT changes nothing. Eight is also Exp 3's
#: K, so one phase assigns a full permutation and phase 2 rotates it by one
#: seat (see :func:`run_arm_hybrid`).
PERSONAS: tuple[str, ...] = (
    "You are Nilaus, the master-class city-block builder. You lay out a grid of "
    "equally sized blocks BEFORE placing a single machine, standardise every "
    "build so it tiles and can be repeated, and move all materials on a main "
    "bus with balanced lanes. You never hand-wire one machine to another when a "
    "bus lane or a block interface can carry it.",
    "You are KatherineOfSky, the steady colony builder. You finish one reliable "
    "chain end to end -- ore to plate to product -- before you start the next, "
    "keep belts, inserters and power tidy, and scale by widening what already "
    "works. You never abandon a half-built chain to chase a new idea.",
    "You are Zisteau running 'meiosis'. You build one small self-contained cell "
    "that mines, smelts and assembles on its own, then grow the factory by "
    "CLONING that cell somewhere else rather than enlarging it. You never let "
    "two cells share a belt, an inserter or an ore patch.",
    "You are Xterminator, the ratio-perfect optimiser. You compute exact "
    "machine ratios before building -- drills per furnace, furnaces per belt -- "
    "and place only whole balanced sets, maximising throughput per tile of "
    "footprint. You never leave a machine starved or a belt backed up; you fix "
    "the ratio instead of bolting on another machine.",
    "You are Nefrums running any%. You build the MINIMUM viable factory that "
    "moves the target number and nothing else: hand-feed where hand-feeding is "
    "faster, no spare capacity, no beautification. You never build anything "
    "that does not raise the measured throughput within the next few steps.",
    "You are AntiElite, the speedrun-efficiency player. Every build must pay "
    "for itself inside this run, so you spend items and steps tightly, expand "
    "ONLY at the current bottleneck, and take a cheap fix now over a clean fix "
    "later. You never start a build whose payback is longer than the run.",
    "You are DoshDoshington, the chaotic challenge runner. You try layouts "
    "other players would refuse, and when a section underperforms you TEAR IT "
    "DOWN wholesale and rebuild it differently instead of patching it. You "
    "never keep a design out of sentiment, and every step carries one "
    "deliberately unconventional move.",
    "You are the Spaghetti Engineer of community legend. You fix every "
    "shortage the fastest opportunistic way -- one more belt threaded around "
    "whatever is in the way, an inserter bolted onto the nearest machine -- and "
    "you NEVER refactor or replan. The factory grows outward as tangled as it "
    "needs to be, as long as plates keep moving.",
)

#: Cheapest honest read of the cumulative score counters (P5 baseline). An
#: empty program advances no game ticks; it only returns the current scores.
BASELINE_CODE = "pass"


# ---------------------------------------------------------------------------
# Exp-2 round sizing (measured inputs only)
# ---------------------------------------------------------------------------

#: Fork p50 on this deployment, measured over Exp 1's 16 forks (1-20 attempts).
#: SIZING input only (see :func:`exp2_round_sizing`): it answers "how long is a
#: typical wave", which is what m has to cover.
EXP2_FORK_P50_S = 62.3
#: Fork p95, nearest-rank over the SAME 16 journaled ``infra_op`` fork durations
#: (5.4 / 31.6 / 36.9 / 45.0 / 45.7 / 53.6 / 54.3 / 61.2 / 61.6 / 62.2 / 62.3 /
#: 62.5 / 71.2 / 104.0 / 135.4 / 151.6 s -- p95 == max at n=16). ADMISSION input:
#: a round admitted on p50 whose wave then runs at the tail overruns T by
#: (K-1) x (p95 - p50) ~ 620s, which would silently break the equal-wall-clock
#: contrast. Rounds are therefore admitted on the tail, not the median, and the
#: wave itself carries an absolute deadline (:meth:`BranchingRun.
#: _materialize_children_fork`) for the case where even p95 is optimistic.
EXP2_FORK_P95_S = 151.6
#: k3's REALIZED median step latency, from ``bench/journal/exp1/exp1.jsonl``:
#: the median gap between consecutive ``step`` records of the same branch is
#: 19.89s over the 144 gaps that contain no probe record (21.32s over all 176
#: gaps, i.e. counting the probe-inflated ones). A B round pays for m plain
#: steps and one probe, so the probe-free figure is the right one -- and the
#: conservative one: a SHORTER step needs MORE steps to cover the fork wave, so
#: sizing against 19.9s keeps the round long enough even when steps come back
#: fast.
EXP2_STEP_P50_S = 19.9
#: Exp 2 runs K=8. Arm B forks K-1 children per round, SERIALLY (width-1 forks,
#: pinned to the source's node), so a round's wave is (K-1) x fork p50.
EXP2_K = 8
#: Design requirement: the serial fork wave hides inside the round WITH margin,
#: i.e. round wall clock >= 1.5x the wave.
EXP2_ROUND_MARGIN = 1.5


def exp2_round_sizing(
    K: int = EXP2_K,
    *,
    step_p50_s: float = EXP2_STEP_P50_S,
    fork_p50_s: float = EXP2_FORK_P50_S,
    margin: float = EXP2_ROUND_MARGIN,
) -> dict[str, Any]:
    """Exp-2 round length m, with the arithmetic that produced it.

    ``m x step_p50 >= margin x (K-1) x fork_p50``, rounded UP::

        m x 19.9s >= 1.5 x 7 forks x 62.3s = 654.15s  ->  m >= 32.91  ->  m = 33

    At 19.9s/step that is a ~11 min round per branch, so an Exp-2 B run needs a
    T comfortably above two rounds (pass ``--T``). m stays overridable
    everywhere: ``--m``, ``Tier1Config.m``, ``ArmConfig.m``.
    """
    forks = max(0, K - 1)
    wave_s = forks * fork_p50_s
    required_s = margin * wave_s
    m = max(1, math.ceil(required_s / step_p50_s))
    return {
        "m": m,
        "K": K,
        "forks_per_round": forks,
        "fork_p50_s": fork_p50_s,
        "fork_wave_s": round(wave_s, 2),
        "step_p50_s": step_p50_s,
        "margin": margin,
        "required_round_s": round(required_s, 2),
        "round_s_at_m": round(m * step_p50_s, 2),
        "arithmetic": (
            f"m x {step_p50_s}s >= {margin} x {forks} forks x {fork_p50_s}s = "
            f"{required_s:.2f}s -> m >= {required_s / step_p50_s:.2f} -> m = {m} "
            f"(round {m * step_p50_s:.0f}s vs {wave_s:.0f}s fork wave)"
        ),
    }


#: Exp-2 default round length (see :func:`exp2_round_sizing`).
EXP2_DEFAULT_M = exp2_round_sizing()["m"]


# ---------------------------------------------------------------------------
# Exp-3 phase sizing (the design's own numbers; nothing re-derived here)
# ---------------------------------------------------------------------------

#: Leg length P. The design fixes phases rather than T/2 splits: one leg is a
#: FULL Exp-2 horizon, so a hybrid walk is never shorter than the control's.
EXP3_P_S = 4200.0
#: Exp 3 keeps all 8 seats to the end in BOTH arms (max-over-8-at-T is the
#: endpoint), so K is the width of every phase, not a fan-out that collapses.
EXP3_K = 8
#: Phase-2 admission: the refork wave is charged to T, so leg 2 can start with
#: less than P left. Below this fraction of P the leg is journaled as SHORT and
#: runs whatever remains -- wall clock stays the sole stopping rule, and the
#: analysis is told the walk was clipped instead of guessing from timestamps.
EXP3_PHASE2_ADMISSION = 0.9
#: Pre-registered Exp-3 width floor: a hybrid endpoint that judged fewer than
#: this many phase-2 seats is ``invalid_width``. Exp 2's DOSE floor does not
#: apply here -- Exp 3's dose is structurally 1 -- but a truncated refork wave
#: would still quietly turn the judge-at-the-bell into a best-of-few.
EXP3_WIDTH_FLOOR = 5


def task_spec(task_key: str) -> tuple[str, str, int]:
    """(goal_description, throughput_entity, quota) from FLE's own registry."""
    from fle.eval.tasks.task_definitions.lab_play.throughput_tasks import (
        THROUGHPUT_TASKS,
    )

    cfg = THROUGHPUT_TASKS.get(task_key)
    if cfg is None:
        raise KeyError(f"unknown task {task_key!r}")
    entity = cfg.throughput_entity
    value = getattr(entity, "value", entity)
    if isinstance(value, tuple):
        value = value[0]
    return cfg.goal_description, str(value), int(cfg.quota)


def format_elapsed(ticks: int) -> str:
    total = int(ticks) // 60
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _tick_of(meta: dict[str, Any]) -> int:
    """Real Factorio tick from /meta, falling back to FLE's virtual counter."""
    for key in ("game_tick", "elapsed_ticks"):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


# ---------------------------------------------------------------------------
# Conversation (P4)
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    """Host-side conversation with the last action's feedback held separately.

    ``messages`` is the committed history. ``pending_feedback`` is the feedback
    of the most recent program, which is *not yet* a message: it becomes part of
    the next user turn. Branching deep-copies both, so a branch can be promoted
    or discarded atomically.
    """

    messages: list[dict[str, str]]
    pending_feedback: str | None = None
    pending_extras: list[str] = field(default_factory=list)

    def branch(self) -> "Conversation":
        return Conversation(
            messages=copy.deepcopy(self.messages),
            pending_feedback=self.pending_feedback,
            pending_extras=list(self.pending_extras),
        )

    def next_user_content(self, consume: bool = True) -> str:
        """The user turn that the next sample() must see."""
        parts = [p for p in ([self.pending_feedback] + self.pending_extras) if p]
        if not parts:
            parts = ["Continue. Write the next program."]
        if consume:
            self.pending_feedback = None
            self.pending_extras = []
        return "\n\n".join(parts)

    def prompt_with(self, user_content: str) -> list[dict[str, str]]:
        return self.messages + [{"role": "user", "content": user_content}]

    def append_turn(self, user_content: str, assistant_text: str) -> None:
        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": assistant_text})

    def inject(self, block: str) -> None:
        self.pending_extras.append(block)

    @property
    def n_turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "assistant")


# ---------------------------------------------------------------------------
# Config & results
# ---------------------------------------------------------------------------

#: Slack the sandbox lease must keep BEYOND the build clock, the cell's
#: provisioning stagger and the terminal-probe reserve: provisioning itself, a
#: halftime regroup, the terminal probes and teardown all happen inside the
#: lease. Sized from Exp 3's own measurements (a ~1080s convergence at the fork
#: p95 is the largest single item) and deliberately blunt -- it is a floor, not a
#: budget.
LEASE_GUARD_SLACK_S = 900.0


class LeaseTooShort(ValueError):
    """A run's sandbox lease cannot outlive the run.

    Raised at :class:`ArmConfig` construction -- BEFORE any sandbox exists -- so a
    preset that would hibernate its own seats fails loudly at launch instead of
    producing PARTIAL cells hours later (Exp-3 round 1).
    """



@dataclass
class ArmConfig:
    arm: str
    model: str
    task_key: str
    replicate: int = 1
    T_s: float = 2700.0
    K: int = 2
    #: Round length: parity-probe cadence in every arm, convergence cadence in
    #: B. Default = Exp 2's sizing (:func:`exp2_round_sizing`); every caller may
    #: override it (``--m``, ``Tier1Config.m``, Exp 1 passes its probe cadence).
    m: int = EXP2_DEFAULT_M
    #: SEED snapshot every sandbox of this run is created from: TEMPLATE_SNAP for
    #: the greenfield arms, a baked checkpoint (Exp 2's S2/S3) for the "-from-S"
    #: variants. A×K-from-S is literally A×K with template_snap=<checkpoint>:
    #: nothing in this module re-derives task setup, resets state or assumes the
    #: id is the baked template, so an arbitrary snapshot works for A, A×K and B.
    template_snap: str = ""
    #: Sandbox lease for every seat AND every fork child (:attr:`Infra.ttl`).
    #: NEVER a cleanup mechanism -- deletion is always explicit -- but it MUST
    #: outlive the run: TTL expiry hibernates a sandbox, and a hibernated seat
    #: cannot be probed. Callers that know T (the orchestrator's blocks) derive
    #: this from it rather than carrying a literal.
    ttl_s: int = 5400
    prefix: str = "flebench-"
    expose_port: int = 8730
    health_deadline_s: float = 300.0
    step_timeout_s: float = 420.0
    #: Held back from T so the mandatory terminal probe always fits. v2.6: a
    #: direct probe is ~6s warm / ~22s cold, so the reserve is small.
    terminal_reserve_s: float = 90.0
    #: A mid-run parity probe is skipped (and logged) if less than this remains.
    probe_cost_estimate_s: float = 45.0
    #: Measured mechanics constants the BRANCH-ROUND estimate is built from
    #: (settled inputs, this deployment). A round is refused -- and the run
    #: continues its canonical line to T instead -- when less than
    #: :meth:`BranchingRun.round_estimate_s` remains, so T is hard even for the
    #: convergent arms. Overridable so a dry run can scale them to the fakes.
    #: The fork term is the measured p95, NOT the p50 the round length was
    #: sized from: admission has to survive the tail it will actually meet,
    #: while m only has to cover a typical wave.
    snapshot_cost_estimate_s: float = 10.1
    fork_cost_estimate_s: float = EXP2_FORK_P95_S
    step_cost_estimate_s: float = EXP2_STEP_P50_S
    #: Settled mechanics constant: sandbox/snapshot delete ~1s. Reserved
    #: EXPLICITLY, so releasing a round's snapshot and its K-1 losers can never
    #: be the thing that pushes a run past T.
    delete_cost_estimate_s: float = 1.0
    diversify: str = "auto"  # auto | always | never
    #: The per-seat diversity set. Exp 2's strategy hints by default; the Exp-3
    #: arms swap in :data:`PERSONAS` below unless a caller passed its own set,
    #: so there is no way to launch Exp 3 with Exp 2's diversity channel by
    #: forgetting a flag.
    hints: tuple[str, ...] = DIVERSITY_HINTS
    #: Exp-3 leg length P (``Hybrid`` runs two legs of it; 0 -> T_s / 2, which
    #: is what T_total = 2P means).
    leg_s: float = 0.0
    results_dir: str = "bench/results"
    journal_dir: str = "bench/journal"
    run_id: str = ""
    dry: bool = False
    #: Pre-registered failed-child policy: retry once, then continue at K-1.
    child_retries: int = 1
    #: Poll budget for ONE ``create_from_snapshot`` (0 = the wrapper's 300s
    #: default). A burst of creates queues on the warm-slot lane, so a wide
    #: block raises it: Exp-3 round 1 lost Hybrid's 8th seat to a 300s timeout
    #: while 17 creates contended for the same lane.
    create_deadline_s: float = 0.0
    #: The orchestrator's per-cell provisioning delay, carried here ONLY so the
    #: lease guard below can account for it: a staggered cell's sandboxes are
    #: leased before its build clock starts.
    provision_stagger_s: float = 0.0
    #: Pre-flight lease guard (see :meth:`__post_init__`). Turned off ONLY by
    #: callers whose ``T_s`` is not one sandbox's horizon -- a 1e9 "no wall-clock
    #: stop" sentinel, or a whole-experiment budget spanning many short-lived
    #: sandboxes. Never turned off to make a real run start.
    lease_guard: bool = True

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.K < 1 or self.m < 1:
            raise ValueError("K and m must be >= 1")
        # The terminal-probe reserve must never eat the run: a reserve at or
        # above T would expire the budget before the first step and the run
        # would silently measure an empty factory.
        self.terminal_reserve_s = min(self.terminal_reserve_s, 0.2 * self.T_s)
        self.probe_cost_estimate_s = min(self.probe_cost_estimate_s, 0.25 * self.T_s)
        if not self.run_id:
            self.run_id = new_run_id(self.arm, self.model, self.task_key, self.replicate)
        if self.arm in EXP3_PERSONA_ARMS and tuple(self.hints) == DIVERSITY_HINTS:
            self.hints = PERSONAS
        if self.arm == "Control":
            # Enforced, not merely unused: the strict control is ONE agent with
            # the neutral prompt. A block passes K=8 to every cell, and a stray
            # persona set or an eight-seat slot reservation would quietly turn
            # the bottom rung into a narrow A×K-S.
            self.K = 1
            self.diversify = "never"
        if self.leg_s <= 0.0:
            self.leg_s = 0.5 * self.T_s
        # PRE-FLIGHT LEASE GUARD, before any caller can create a sandbox.
        # Exp-3 round 1 lost both surviving cells to a lease shorter than the
        # round: TTL expiry HIBERNATES a sandbox, a hibernated seat cannot be
        # probed, and the failure only surfaced at the terminal probe hours in.
        # The lease must cover the build clock plus everything wrapped around it
        # -- the cell's provisioning delay, the terminal-probe reserve, and slack
        # for provisioning, a halftime regroup and teardown. Skipped for dry runs
        # only, where T is clamped to seconds and the slack term is meaningless;
        # the dry gate asserts this guard against LIVE-shaped configs instead.
        if self.lease_guard and not self.dry:
            need_s = (self.T_s + self.provision_stagger_s
                      + self.terminal_reserve_s + LEASE_GUARD_SLACK_S)
            if self.ttl_s < need_s:
                raise LeaseTooShort(
                    f"{self.arm} cell {self.run_id!r}: sandbox lease "
                    f"ttl_s={self.ttl_s}s is shorter than the run it must "
                    f"outlive -- T_s={self.T_s:.0f}s + provisioning stagger "
                    f"{self.provision_stagger_s:.0f}s + terminal reserve "
                    f"{self.terminal_reserve_s:.0f}s + {LEASE_GUARD_SLACK_S:.0f}s "
                    f"slack = {need_s:.0f}s required. Raise ttl_s (blocks derive "
                    f"it from T) or lower T_s; TTL expiry hibernates seats "
                    f"mid-run and a hibernated seat has no endpoint."
                )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


@dataclass
class BranchOutcome:
    branch: str
    conv: Conversation
    node: "Node"
    score: ScoreRecord
    steps: int
    candidate_chars: int
    errors: int
    probe: dict[str, Any] | None = None
    rollout_s: float = 0.0
    incidents: list[str] = field(default_factory=list)
    #: Game tick counter the branch ENDED on, carried into promotion: the
    #: feedback template reports each step's tick cost as a delta against it,
    #: and the pre-round main's counter is not this line's.
    last_ticks: int = 0
    #: Non-empty -> this branch cannot be RANKED (no P5 baseline, or a sandbox
    #: holding a substrate call this run could not join). Journaled and
    #: archived as a diagnostic, never promoted.
    unscorable: str = ""


@dataclass
class RunResult:
    run_id: str
    arm: str
    model: str
    task_key: str
    replicate: int
    K: int
    m: int
    T_s: float
    status: str = "ok"
    error: str = ""
    endpoint_throughput: float | None = None
    endpoint_source: str = ""
    quota: int = 0
    entity: str = ""
    #: Total agent steps executed. For A×K this is the sum over the K
    #: trajectories (see ``steps_per_trajectory``); for B/C it is the promoted
    #: main line only -- branch steps that lost are counted in the journal.
    steps: int = 0
    steps_per_trajectory: list[int] = field(default_factory=list)
    branch_points: int = 0
    curve: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, Any] = field(default_factory=dict)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    provision_s: float = 0.0
    teardown_s: float = 0.0
    active_s: float = 0.0
    end_to_end_s: float = 0.0
    sandboxes_created: int = 0
    snapshots_created: int = 0
    #: Fork calls abandoned on their own deadline whose child may still have
    #: landed. Each names the source snapshot the reaper claims it through, and
    #: that snapshot is intentionally left alive for exactly that reason.
    orphan_forks: list[dict[str, Any]] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    journal_path: str = ""
    #: The :class:`~bench.common.RunJournal` session id every record of this run
    #: carries (contract R2C3). A journal path is an APPEND stream that may hold
    #: several sessions, so this id -- not the path -- is what binds a result row
    #: to the evidence that produced it; a consumer grading a verdict on journal
    #: digests must match it exactly.
    journal_session: str = ""
    #: EVERY seat's terminal probe, not just the winning one. Exp 3 judges
    #: max-over-8-at-T in both arms and reports per-seat distributions, so the
    #: losing endpoints are data, not noise, and must survive into the results
    #: file rather than living only in the journal. Empty for the Exp-2 arms.
    seat_endpoints: list[dict[str, Any]] = field(default_factory=list)
    #: Exp-3 run record: leg length, per-phase persona assignment, the selection
    #: and its measured overhead, phase-2 width and the width-floor verdict.
    exp3: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


# ---------------------------------------------------------------------------
# Timed / journaled substrate wrapper
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A live sandbox plus its bridge client.

    ``bridge`` is None for the window between the sandbox EXISTING and its
    health poll passing: :meth:`Infra._attach` owns the sandbox from creation,
    so a failed attach cannot leak it, and both teardown and the reaper reach
    it through ``Infra.live_sandboxes`` before there is anything to talk to.
    """

    sb: Any
    bridge: Any
    label: str

    @property
    def id(self) -> str:
        return getattr(self.sb, "id", "?")

    @property
    def name(self) -> str:
        return getattr(self.sb, "name", "?")


class Infra:
    """Every substrate call, timed into a bucket and journaled.

    Bucket mapping for arm C's branching mechanism is deliberate and reported:
    ``state_save`` -> ``infra_snapshot`` and ``state_restore`` -> ``infra_fork``,
    because those are C's snapshot and branch-creation primitives. The interval
    labels keep the underlying operation identifiable.
    """

    def __init__(
        self,
        farplane: FarplaneLike,
        bridge_factory: BridgeFactory,
        timings: TimingBuckets,
        journal: RunJournal,
        cfg: ArmConfig,
    ) -> None:
        self.fp = farplane
        self.bridge_factory = bridge_factory
        self.timings = timings
        self.journal = journal
        self.cfg = cfg
        self.sandboxes_created = 0
        self.snapshots_created = 0
        self.live_sandboxes: dict[str, Node] = {}
        self.live_snapshots: set[str] = set()
        #: Substrate calls whose awaiting coroutine was cancelled at the T
        #: deadline. ``asyncio.to_thread`` cannot be cancelled -- the worker
        #: thread is still talking to the sandbox -- so they stay here until
        #: :meth:`drain` joins them. See :meth:`ArmRun.terminal_probe`.
        self._inflight: dict[asyncio.Future, dict[str, Any]] = {}
        #: Sandboxes whose call a bounded drain gave up on. Kept on Infra, not
        #: on the joining caller: ANY line's drain joins whatever is in flight,
        #: so the seat that owns the abandoned call is usually not the one that
        #: learns about it. Reading such a sandbox again is never sound.
        self.abandoned_nodes: set[str] = set()
        #: Fork calls abandoned on their own deadline. The child may still land
        #: on the control plane, so its source snapshot stays owned (never
        #: deleted by the round or by teardown) and the reaper claims the child
        #: through it. See :meth:`register_orphan_fork`.
        self.orphan_forks: list[dict[str, Any]] = []
        self.orphan_sources: set[str] = set()
        #: Outcomes of substrate calls whose awaiting coroutine was cancelled at
        #: a deadline but which CONCLUDED anyway -- either before the
        #: cancellation reached us or during a later :meth:`drain`. The world
        #: moved and the result exists, so it is held here for the caller that
        #: has to commit it (:meth:`ArmRun.settle_node`): dropping it leaves a
        #: line whose transcript is one program behind its own sandbox.
        self.recovered: list[dict[str, Any]] = []

    # -- internals ---------------------------------------------------------
    def _finished(self, task: "asyncio.Future", rec: dict[str, Any]) -> None:
        """Done-callback: stamp the instant the worker actually finished.

        A call abandoned at a step deadline is settled later -- by
        :meth:`drain`, possibly minutes later -- and stamping it THEN charges
        every second between the deadline and the join to its bucket: a 6s
        ``/execute`` joined 300s later landed in ``rollout_exec`` as 300s and
        drowned the whole timing partition. The completion instant belongs to
        the call, so it is captured here, the moment the future resolves -- with
        the outcome, because whoever settles the record later may be a drain
        that never awaited this call at all.
        """
        if rec["t1"] is None:
            rec["t1"] = time.monotonic()
            rec["cancelled"] = task.cancelled()

    def _settle(self, task: "asyncio.Future", rec: dict[str, Any], *,
                exc: BaseException | None = None, outcome: str = "") -> None:
        """Record one substrate call's interval and journal it, exactly once."""
        if rec["logged"]:
            return
        rec["logged"] = True
        self._inflight.pop(task, None)
        # The completion instant :meth:`_finished` captured. Only a call that is
        # STILL RUNNING settles at "now", and that one is journaled as abandoned
        # by :meth:`drain` rather than measured.
        t1 = rec["t1"] if rec["t1"] is not None else time.monotonic()
        self.timings.record(rec["bucket"], rec["t0"], t1, f"{rec['op']}:{rec['target']}")
        self.journal.infra_op(
            op=rec["op"], bucket=rec["bucket"], duration_s=t1 - rec["t0"],
            outcome=outcome or ("error" if exc is not None else "ok"),
            target=rec["target"], branch=rec["branch"],
            error=f"{type(exc).__name__}: {exc}"[:1000] if exc is not None else "",
            **rec["extra"],
        )

    def _recover(self, task: "asyncio.Future", rec: dict[str, Any]) -> None:
        """Keep the outcome of a concluded call nobody is awaiting any more.

        A ``/execute`` abandoned at a step deadline still runs to completion in
        its worker thread. Its result is the only record of what the program did
        to the sandbox, so it is stashed rather than dropped and
        :meth:`ArmRun.settle_node` commits it into the line's bookkeeping.
        """
        if rec["cancelled"] or not task.done() or task.cancelled():
            return
        exc = task.exception()
        self.recovered.append({
            "op": rec["op"],
            "node_id": rec["node_id"],
            "branch": rec["branch"],
            "target": rec["target"],
            "duration_s": (rec["t1"] if rec["t1"] is not None else time.monotonic())
            - rec["t0"],
            "result": None if exc is not None else task.result(),
            "error": exc,
        })

    def take_recovered(self, op: str, node_id: str, *,
                       branch: str = "") -> dict[str, Any] | None:
        """Pop the most recent recovered outcome of ``op`` on one sandbox."""
        for i in range(len(self.recovered) - 1, -1, -1):
            entry = self.recovered[i]
            if entry["op"] != op or entry["node_id"] != node_id:
                continue
            if branch and entry["branch"] != branch:
                continue
            return self.recovered.pop(i)
        return None

    async def _timed(self, op: str, bucket: str, fn: Callable[[], Any], *,
                     target: str = "", branch: str = "", node_id: str = "",
                     **extra: Any) -> Any:
        rec = {"op": op, "bucket": bucket, "t0": time.monotonic(), "target": target,
               "branch": branch, "extra": extra, "logged": False, "t1": None,
               "cancelled": False, "node_id": node_id}
        task = asyncio.ensure_future(asyncio.to_thread(fn))
        # Attached before anything awaits the task, so the completion time is
        # captured even for a call nobody is waiting on any more.
        task.add_done_callback(lambda t, rec=rec: self._finished(t, rec))
        self._inflight[task] = rec
        try:
            # shield: a deadline cancellation must reach US, not the thread --
            # the thread cannot be stopped, and pretending it was is how a
            # terminal probe ends up racing a live /execute.
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._settle(task, rec,
                             exc=None if rec["cancelled"] else task.exception())
                # Concluded before the cancellation reached us: the result is
                # real and the caller still has to commit it.
                self._recover(task, rec)
            raise
        except BaseException as exc:  # noqa: BLE001 - journaled then re-raised
            self._settle(task, rec, exc=exc)
            raise
        self._settle(task, rec)
        return result

    async def drain(self, timeout_s: float, *,
                    node_id: str = "") -> list[dict[str, Any]]:
        """Join every substrate call abandoned at a step deadline.

        ``asyncio.wait_for`` hands control back at the deadline but the worker
        thread underneath keeps executing the program against the sandbox.
        Reading that sandbox before the call concludes measures a state that is
        still moving, so every arm drains here before its terminal probe. The
        join is bounded: a wedged call is journaled and the reaper owns it.

        ``node_id`` narrows the join to ONE sandbox's calls, which is what a
        mid-run step timeout needs: the line that timed out must not touch its
        node again while a program is still running on it, and waiting on a
        SIBLING seat's call would serialise the arms that run K lines at once.
        """
        pending = [t for t, rec in self._inflight.items()
                   if not node_id or rec["node_id"] == node_id]
        if not pending:
            return []
        self.journal.event("drain_start", n=len(pending),
                           ops=[self._inflight[t]["op"] for t in pending],
                           timeout_s=round(timeout_s, 3), node=node_id)
        done, still = await asyncio.wait(pending, timeout=max(0.0, timeout_s))
        out: list[dict[str, Any]] = []
        for task in done:
            rec = self._inflight.get(task)
            if rec is None:
                continue
            exc = None if rec["cancelled"] else task.exception()
            self._settle(task, rec, exc=exc, outcome="settled_after_deadline")
            self._recover(task, rec)
            out.append({"op": rec["op"], "target": rec["target"],
                        "node_id": rec["node_id"],
                        "outcome": "settled_after_deadline"})
        for task in still:
            rec = self._inflight.pop(task, None)
            if rec is None:
                continue
            rec["logged"] = True
            if rec["node_id"]:
                self.abandoned_nodes.add(rec["node_id"])
            self.journal.infra_op(
                op=rec["op"], bucket=rec["bucket"],
                duration_s=time.monotonic() - rec["t0"], outcome="abandoned",
                target=rec["target"], branch=rec["branch"],
                error=f"still running after a {timeout_s:.1f}s drain", **rec["extra"],
            )
            out.append({"op": rec["op"], "target": rec["target"],
                        "node_id": rec["node_id"], "outcome": "abandoned"})
        self.journal.event("drain_done", settled=len(done), abandoned=len(still),
                           node=node_id)
        return out

    def _name(self, role: str) -> str:
        return resource_name(self.cfg.prefix, self.cfg.run_id, role)

    @property
    def ttl(self) -> str:
        """Lease for EVERY sandbox this run creates -- seats and fork children.

        ``ArmConfig`` carries seconds; the ``panda`` CLI takes a Go duration
        string, so the conversion happens exactly here rather than in every
        call site. One property, one value: a lease shorter than the run
        HIBERNATES seats mid-flight (Exp-3 round 1 lost both surviving cells
        that way -- TTL 7200s under a ~8700s round), and the only way to be sure
        every path is covered is for every path to read this.
        """
        return f"{int(self.cfg.ttl_s)}s"

    # -- sandboxes ---------------------------------------------------------
    async def create_from_snapshot(self, snap_id: str, role: str) -> Node:
        """One seat from a snapshot, with an explicit poll budget.

        ``create_deadline_s`` exists because a BURST of creates queues on the
        warm-slot lane: Exp-3 round 1 lost Hybrid's 8th seat when it starved past
        the wrapper's 300s default while 17 creates contended. 0 keeps the
        wrapper's own default.
        """
        name = self._name(role)
        deadline_s = self.cfg.create_deadline_s or None
        sb = await self._timed(
            "create_from_snapshot", "infra_fork",
            lambda: self.fp.create_from_snapshot(
                snap_id, self.ttl, name, deadline=deadline_s
            ),
            target=name, source=snap_id, ttl=self.ttl,
            deadline_s=None if deadline_s is None else round(deadline_s, 3),
        )
        self.sandboxes_created += 1
        node = await self._attach(sb, role)
        return node

    async def fork(self, snap_id: str, role: str, *,
                   deadline_s: float | None = None) -> Node:
        """One width-1 fork, bounded by an ABSOLUTE deadline.

        ``deadline_s`` is passed straight to the wrapper's poll budget, so a
        fork that sits in the Farplane queue (cap 5m; the contended soak p95 was
        758s) is abandoned when T can no longer pay for it. Without it the call
        is shielded and uncancellable, and one admitted fork could run past T on
        its own. The child's lease is the run's own :attr:`ttl` -- a halftime
        refork child has to outlive the same T its parent does.
        """
        name = self._name(role)
        extra: dict[str, Any] = {"source": snap_id, "ttl": self.ttl}
        if deadline_s is not None:
            extra["deadline_s"] = round(deadline_s, 3)
        sb = await self._timed(
            "fork", "infra_fork",
            lambda: self.fp.fork(snap_id, self.ttl, name, deadline=deadline_s),
            target=name, **extra,
        )
        self.sandboxes_created += 1
        return await self._attach(sb, role)

    def register_orphan_fork(self, *, snap_id: str, label: str, detail: str) -> None:
        """A fork we stopped waiting for may still land. Keep it OWNED.

        The poll timed out before ``forks get`` named the child, so we never
        learn its id -- and fork children carry control-plane names, not our
        prefix. The only ownership handle left is the SOURCE SNAPSHOT: the
        reaper claims any sandbox whose ``sourceSnapshot`` is in our ledger.
        That handle survives only while the snapshot is still in the ledger, so
        a snapshot with orphans is deliberately NOT deleted by the round or by
        teardown; the reaper sweeps the child first and the snapshot after
        (:meth:`bench.farplane.Farplane.reaper` orders sandboxes before images).
        """
        self.orphan_sources.add(snap_id)
        rec = {"source_snapshot": snap_id, "label": label, "detail": detail}
        self.orphan_forks.append(rec)
        self.journal.write("fork_orphan", **rec)

    async def _attach(self, sb: Any, role: str) -> Node:
        """Expose and health-poll a fresh sandbox, OWNING it from the first line.

        Ownership is taken before anything can fail. An expose or health poll
        that raised used to leave the sandbox out of ``live_sandboxes``
        entirely, so neither this run's teardown nor :func:`_live_nodes` could
        see it and only the lease would ever reclaim it -- a leak that costs a
        warm slot for the whole TTL.
        """
        node = Node(sb=sb, bridge=None, label=role)
        self.live_sandboxes[node.id] = node
        try:
            base_url = getattr(sb, "base_url", None)
            if not base_url:
                base_url = await self._timed(
                    "expose", "infra_expose",
                    lambda: self.fp.expose(sb, self.cfg.expose_port),
                    target=node.name, node_id=node.id,
                )
            node.bridge = self.bridge_factory(base_url)
            await self._timed(
                "health", "infra_poll",
                lambda: node.bridge.wait_healthy(self.cfg.health_deadline_s),
                target=node.name, node_id=node.id,
            )
        except asyncio.CancelledError:
            # Still OWNED and deliberately not deleted here: the delete would
            # have to be awaited inside a cancelled scope. Teardown reads
            # live_sandboxes, and the block reaper claims it by prefix.
            self.journal.event("attach_abandoned", target=node.name, role=role,
                               reason="cancelled")
            raise
        except BaseException:
            await self._discard(node)
            raise
        return node

    async def _discard(self, node: Node) -> None:
        """Delete a sandbox that never finished attaching (best effort).

        A delete that itself fails leaves the node in ``live_sandboxes`` on
        purpose, so teardown retries it instead of the lease being the only
        cleanup left.
        """
        try:
            await self.delete(node)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - journaled; teardown retries
            self.journal.event("attach_cleanup_failed", target=node.name,
                               error=f"{type(exc).__name__}: {exc}"[:1000])

    async def delete(self, node: Node) -> None:
        await self._timed(
            "delete_sandbox", "infra_delete",
            lambda: self.fp.delete_sandbox(node.sb), target=node.name,
        )
        self.live_sandboxes.pop(node.id, None)

    # -- snapshots ---------------------------------------------------------
    async def snapshot(self, node: Node, *, branch: str = "") -> str:
        snap = await self._timed(
            "snapshot", "infra_snapshot",
            lambda: self.fp.snapshot(node.sb), target=node.name, branch=branch,
        )
        self.snapshots_created += 1
        self.live_snapshots.add(snap)
        return snap

    async def delete_snapshot(self, snap_id: str, *, branch: str = "") -> None:
        await self._timed(
            "delete_snapshot", "infra_delete",
            lambda: self.fp.delete_snapshot(snap_id), target=snap_id, branch=branch,
        )
        self.live_snapshots.discard(snap_id)

    # -- bridge ------------------------------------------------------------
    async def execute(self, node: Node, code: str, *, branch: str = "") -> dict:
        return await self._timed(
            "execute", "rollout_exec", lambda: node.bridge.execute(code),
            target=node.name, branch=branch, node_id=node.id, code_chars=len(code),
        )

    async def probe(self, node: Node, entity: str, *, branch: str = "") -> dict:
        return await self._timed(
            "probe", "probe", lambda: node.bridge.probe(entity),
            target=node.name, branch=branch, node_id=node.id, entity=entity,
        )

    async def state_save(self, node: Node, *, branch: str = "") -> str:
        return await self._timed(
            "state_save", "infra_snapshot", lambda: node.bridge.state_save(),
            target=node.name, branch=branch, node_id=node.id,
        )

    async def state_restore(self, node: Node, state: str, *, branch: str = "") -> None:
        await self._timed(
            "state_restore", "infra_fork",
            lambda: node.bridge.state_restore(state),
            target=node.name, branch=branch, node_id=node.id,
            state_chars=len(state),
        )

    async def meta(self, node: Node, *, branch: str = "") -> dict:
        return await self._timed(
            "meta", "infra_poll", lambda: node.bridge.meta(),
            target=node.name, branch=branch, node_id=node.id,
        )

    async def system_prompt(self, node: Node) -> str:
        return await self._timed(
            "system_prompt", "infra_poll", lambda: node.bridge.system_prompt(),
            target=node.name, node_id=node.id,
        )


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One continuing line of play (A has 1; A×K has K)."""

    tid: str
    node: Node
    conv: Conversation
    step: int = 0
    errors: int = 0
    last_production: float = 0.0
    last_automated: float = 0.0
    last_ticks: int = 0
    curve: Curve = field(default_factory=Curve)
    terminal_probe: dict[str, Any] | None = None
    #: Non-empty -> this line's transcript and its sandbox have come apart, so
    #: the line is PARTIAL: it stops stepping and is never probed again. Causes:
    #: a substrate call this run could not join (:meth:`ArmRun.settle_node`), a
    #: step result that could not be recovered, an AMBIGUOUS bridge call
    #: (contract R2C4), or a branch round that left no scorable state
    #: (:meth:`BranchingRun._round_unscorable`). PARTIAL always implies the
    #: sandbox is QUARANTINED (:meth:`ArmRun.fail_line`).
    partial: str = ""
    #: Set when a step's ``/execute`` was abandoned at its deadline and its
    #: result has NOT been committed yet: ``{"step": int, "code_chars": int}``.
    #: :meth:`ArmRun.settle_node` commits it from the recovered outcome (the
    #: program concluded, so the world moved) or ends the line.
    pending_exec: dict[str, Any] | None = None


class ArmRun:
    """Shared machinery: agent step, parity probe cycle, branch rounds."""

    def __init__(
        self,
        cfg: ArmConfig,
        *,
        farplane: FarplaneLike,
        bridge_factory: BridgeFactory,
        llm: LLMClient,
        journal: RunJournal,
        timings: TimingBuckets | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.journal = journal
        self.timings = timings or TimingBuckets()
        self.llm.timings = self.timings
        self.llm.journal = journal
        self.infra = Infra(farplane, bridge_factory, self.timings, journal, cfg)
        self.goal, self.entity, self.quota = task_spec(cfg.task_key)
        self.budget = Budget(total_s=cfg.T_s, reserve_s=cfg.terminal_reserve_s)
        self.incidents: list[dict[str, Any]] = []
        self.branch_points = 0
        self.system_prompt = ""
        self._probe_seq = 0
        #: Sandboxes already probed at least once: the first probe on a fresh
        #: microVM pays the cold-page tax, which v2.6 measures and reports.
        self._probed_nodes: set[str] = set()
        #: Sandboxes holding a substrate call this run could not join. Nothing
        #: steps, probes or re-seeds on them again (:meth:`settle_node`).
        self.quarantined: set[str] = set()
        #: The lines this run finished on, published by :func:`_finish` so a
        #: caller (the dry validator) can assert promotion invariants on the
        #: exact conversation that produced the endpoint.
        self.trajectories: list["Trajectory"] = []

    # -- helpers -----------------------------------------------------------
    def incident(self, kind: str, detail: str, **extra: Any) -> None:
        rec = {"kind": kind, "detail": detail, **extra}
        self.incidents.append(rec)
        self.journal.incident(kind=kind, detail=detail, **extra)

    def step_deadline_s(self) -> float:
        """Timeout for ONE agent step: the step cap, clipped to what T has left.

        Without the clip a step that starts just inside the deadline can run the
        full ``step_timeout_s`` past T; with it a run overruns T by at most the
        one step that was already in flight when the budget ran out.
        """
        return max(0.0, min(self.cfg.step_timeout_s, self.budget.remaining_s()))

    async def settle_node(self, traj: Trajectory, *, reason: str) -> bool:
        """Join whatever is still running on this line's sandbox; True if clean.

        ``asyncio.wait_for`` cancels the awaiting coroutine, not the worker
        thread under :meth:`Infra._timed`'s shield: a step abandoned at its
        deadline is STILL executing its program against the sandbox. Touching
        that node again -- another step, the boundary probe -- reads and mutates
        a state that is still moving, so the join happens here, bounded by the
        step cap (the same bound the terminal drain trusts).

        A call that outlives the join quarantines the sandbox and marks the line
        PARTIAL: it stops stepping and is never measured, because there is no
        honest way to measure a factory something else is still building.

        A call that SETTLED during the join is a different story: the program
        concluded, so the world moved and its result exists -- it just has no
        reader left. :meth:`commit_recovered_step` writes it into this line's
        bookkeeping (feedback, score counters, journal record) before the loop
        reuses the line, and ends the line when the outcome cannot be recovered
        at all: the next turn would otherwise say "Continue" to a factory the
        transcript has never seen.
        """
        drained = await self.infra.drain(self.cfg.step_timeout_s,
                                        node_id=traj.node.id)
        stuck = [d for d in drained if d["outcome"] == "abandoned"]
        if not stuck and traj.node.id not in self.infra.abandoned_nodes:
            return self.commit_recovered_step(traj, reason=reason)
        detail = (
            f"{reason}: "
            + (", ".join(f"{d['op']} on {d['target']}" for d in stuck)
               or f"an earlier call on {traj.node.name}")
            + f" outlived a {self.cfg.step_timeout_s:.0f}s join; line "
            f"{traj.tid} is partial and stops here"
        )
        self.quarantine(traj.node, detail, branch=traj.tid)
        traj.partial = detail
        traj.pending_exec = None
        return False

    def quarantine(self, node: Node, detail: str, *, branch: str = "") -> None:
        """Retire a sandbox this run can no longer account for."""
        if node.id in self.quarantined:
            return
        self.quarantined.add(node.id)
        self.incident("node_quarantined", detail, target=node.name, branch=branch)

    def fail_line(self, traj: Trajectory, detail: str, *, kind: str) -> None:
        """End a line whose sandbox this run can no longer account for.

        PARTIAL implies QUARANTINED, always: a line stops because its transcript
        and its world have come apart, and the only alternative to retiring the
        sandbox is publishing a probe of it as this line's endpoint.
        """
        self.incident(kind, detail, branch=traj.tid, target=traj.node.name)
        self.quarantine(traj.node, detail, branch=traj.tid)
        if not traj.partial:
            traj.partial = detail
        traj.pending_exec = None

    @staticmethod
    def is_ambiguous(exc: BaseException) -> bool:
        """A bridge failure whose effect on the game is UNKNOWN (contract R2C4).

        ``bench.bridge_client`` sets ``ambiguous=True`` on every non-idempotent
        call (``/execute``, ``/probe``, ``/state-restore``) that hit any 5xx, any
        transport failure not provably pre-send, or a 2xx it could not parse.
        Duck-typed on purpose: the fakes and the real client both answer it, and
        arms.py must not import the HTTP layer to ask.
        """
        return bool(getattr(exc, "ambiguous", False))

    def fail_node_ambiguous(self, node: Node, exc: BaseException, *, op: str,
                            branch: str = "", step: int = 0) -> str:
        """Retire a sandbox after an ambiguous mutating call (contract R2C4).

        Never retried and never reused: the call may have reached the game and
        never reported what it did, so nothing measured on this sandbox afterward
        can be reconciled with the transcript that asked for it.
        """
        detail = (
            f"{op} on {node.name} failed AMBIGUOUSLY "
            f"({type(exc).__name__}: {exc})"[:600]
            + f"; the call may have reached the game and never reported what it "
            f"did, so {node.name} is quarantined without retry"
        )
        self.incident("bridge_ambiguous", detail, target=node.name, branch=branch,
                      step=step, op=op)
        self.quarantine(node, detail, branch=branch)
        return detail

    def commit_recovered_step(self, traj: Trajectory, *, reason: str) -> bool:
        """Commit the step whose ``/execute`` settled after its own deadline.

        The join in :meth:`settle_node` proved nothing is running on this
        sandbox any more, which means the abandoned program CONCLUDED: it moved
        the world and produced a result nobody read. Committing it is what keeps
        the next turn from prompting "Continue" against a factory the transcript
        never saw -- and what keeps the score counters, the tick baseline and the
        journal's step record from skipping a step that really ran.

        Returns False (line ended) when the outcome cannot be recovered or the
        call was ambiguous; True when there was nothing pending, when the result
        was committed, or when the call failed unambiguously (which
        ``bench.bridge_client`` guarantees means the game was not touched).
        """
        pending = traj.pending_exec
        if pending is None:
            # Cancelled before any program was sent: agent_step already handed
            # the consumed feedback back to the next attempt.
            return True
        traj.pending_exec = None
        step = int(pending["step"])
        entry = self.infra.take_recovered("execute", traj.node.id, branch=traj.tid)
        if entry is None:
            self.fail_line(
                traj,
                f"{reason}: step {step}'s /execute was abandoned at its deadline "
                f"and its outcome could not be recovered; line {traj.tid} is "
                "partial and stops here",
                kind="step_result_lost",
            )
            return False
        exc = entry["error"]
        if exc is not None:
            if self.is_ambiguous(exc):
                detail = self.fail_node_ambiguous(
                    traj.node, exc, op="execute", branch=traj.tid, step=step,
                )
                if not traj.partial:
                    traj.partial = detail
                return False
            # Unambiguous failure: the program never reached the game, so this
            # is an ordinary environment error, told to the model as one.
            traj.errors += 1
            detail = f"{type(exc).__name__}: {exc}"
            traj.conv.pending_feedback = EXEC_ERROR_FEEDBACK.format(
                step=step, error=detail[:800]
            )
            self.incident("execute_failed",
                          f"{detail} (settled {entry['duration_s']:.1f}s after it "
                          f"was abandoned at the step deadline)",
                          branch=traj.tid, step=step)
            return True
        res = entry["result"]
        if not isinstance(res, dict):
            self.fail_line(
                traj,
                f"{reason}: step {step}'s /execute settled with "
                f"{type(res).__name__}, which is not a result this line can "
                f"continue from; line {traj.tid} is partial and stops here",
                kind="step_result_lost",
            )
            return False
        self.incident(
            "step_result_recovered",
            f"{reason}: step {step}'s /execute settled "
            f"{entry['duration_s']:.1f}s after it was abandoned; its result is "
            "committed to this line instead of being dropped",
            branch=traj.tid, step=step,
        )
        self.journal.event("step_result_recovered", step=step, branch=traj.tid,
                           duration_s=round(float(entry["duration_s"]), 3),
                           target=entry["target"])
        self._commit_execute(traj, res, step=step,
                             code_chars=int(pending["code_chars"]),
                             exec_s=float(entry["duration_s"]))
        return True

    def hints_for(self, n: int, *, offset: int = 0) -> list[str] | None:
        """The n per-seat strategy hints, or None when hinting is off.

        Positional by default: seat i gets hint i (one wave, reproducible).
        ``offset`` rotates the set, which is how arm B avoids repeating a
        hint-to-seat assignment across rounds: ``offset=round_idx-1`` shifts
        every seat by one strategy per round, so no seat sees the same hint
        twice in a row and a round is still a full sweep when len(hints) == n.
        """
        if n < 2 or self.cfg.diversify == "never":
            return None
        if self.cfg.diversify == "always" or self.llm.spec.temperature_locked:
            hints = self.cfg.hints
            return [hints[(i + offset) % len(hints)] for i in range(n)]
        return None

    def new_conversation(self) -> Conversation:
        messages = [{"role": "system", "content": self.system_prompt}]
        conv = Conversation(messages=messages)
        conv.pending_feedback = GOAL_TEMPLATE.format(goal=self.goal)
        return conv

    # -- agent step --------------------------------------------------------
    async def agent_step(
        self, traj: Trajectory, *, hint: str | None = None, tag: str = ""
    ) -> dict[str, Any]:
        """One model -> code -> /execute -> observation cycle."""
        step = traj.step + 1
        user_content = traj.conv.next_user_content()
        prompt = traj.conv.prompt_with(user_content)
        try:
            samples = await self.llm.sample_detailed(
                prompt, n=1, hints=[hint] if hint else None,
                branch=f"{traj.tid}{tag}", step=step,
            )
        except BaseException:
            # Cancelled on the T deadline, or the provider failed: no program
            # was ever sent, so this is NOT a step. Roll the counter back and
            # hand the consumed feedback to the next attempt -- otherwise the
            # matched-agent-steps read counts a step that never happened and
            # the line silently loses its last program's result.
            traj.conv.pending_feedback = user_content
            raise
        sample = samples[0]
        traj.step = step
        traj.conv.append_turn(user_content, sample.text or "(empty response)")
        return await self.apply_program(traj, sample, step=step)

    async def apply_program(
        self, traj: Trajectory, sample: Sample, *, step: int
    ) -> dict[str, Any]:
        """Execute a sampled program and set the pending feedback (P4)."""
        if not sample.code:
            traj.errors += 1
            traj.conv.pending_feedback = PARSE_FAILURE_FEEDBACK
            self.incident(
                "unparseable_response", sample.error or "no code block",
                branch=traj.tid, step=step,
            )
            return {"error": True, "parsed": False}
        t0 = time.monotonic()
        try:
            res = await self.infra.execute(traj.node, sample.code, branch=traj.tid)
        except asyncio.CancelledError:
            # NOT an environment error. The step deadline fired and the program
            # is STILL RUNNING in its worker thread; swallowing this would let
            # the caller march on to a terminal probe that races a live
            # mutation. Re-raise; Infra keeps the call for the pre-probe drain,
            # and the step is recorded as UNCOMMITTED so the join that settles
            # it can write its result into this line (:meth:`settle_node`)
            # instead of dropping a program the world has already seen.
            traj.pending_exec = {"step": step, "code_chars": len(sample.code)}
            self.incident(
                "step_deadline_cancelled",
                f"execute abandoned at the T deadline after "
                f"{time.monotonic() - t0:.1f}s; joined before the terminal probe",
                branch=traj.tid, step=step,
            )
            raise
        except BaseException as exc:  # noqa: BLE001 - env error, charged to T
            if self.is_ambiguous(exc):
                # Contract R2C4: the call may have run the program and never
                # reported what it did. Telling the model "your program errored"
                # would be a claim about a world nobody read, so the sandbox is
                # quarantined, the line is partial and nothing is retried.
                traj.errors += 1
                detail = self.fail_node_ambiguous(
                    traj.node, exc, op="execute", branch=traj.tid, step=step,
                )
                if not traj.partial:
                    traj.partial = detail
                return {"error": True, "parsed": True, "exception": detail,
                        "ambiguous": True}
            traj.errors += 1
            detail = f"{type(exc).__name__}: {exc}"
            traj.conv.pending_feedback = EXEC_ERROR_FEEDBACK.format(
                step=step, error=detail[:800]
            )
            self.incident("execute_failed", detail, branch=traj.tid, step=step)
            return {"error": True, "parsed": True, "exception": detail}
        return self._commit_execute(traj, res, step=step,
                                    code_chars=len(sample.code),
                                    exec_s=time.monotonic() - t0)

    def _commit_execute(self, traj: Trajectory, res: dict[str, Any], *, step: int,
                        code_chars: int, exec_s: float) -> dict[str, Any]:
        """Feedback, score counters and journal record for ONE concluded program.

        Shared by the normal path and by :meth:`commit_recovered_step`: a program
        that concluded after its step deadline moved the same world and produced
        the same kind of result, so it earns the same bookkeeping rather than
        being dropped on the floor.
        """
        production = float(res.get("production_score", 0.0) or 0.0)
        automated = float(res.get("automated_score", 0.0) or 0.0)
        ticks = int(res.get("ticks", 0) or 0)
        output = str(res.get("result", "") or "")
        traj.conv.pending_feedback = FEEDBACK_TEMPLATE.format(
            step=step,
            output=output[:6000] if output else "None",
            production_score=production,
            previous_score=traj.last_production,
            delta=production - traj.last_production,
            automated_score=automated,
            elapsed=format_elapsed(ticks),
            ticks=ticks,
            ticks_cost=max(0, ticks - traj.last_ticks),
            next_step=step + 1,
        )
        self.journal.step_result(
            step=step, branch=traj.tid, code_chars=code_chars,
            production_score=production, automated_score=automated, ticks=ticks,
            error=bool(res.get("error")), exec_s=exec_s, output_head=output[:400],
        )
        traj.last_production = production
        traj.last_automated = automated
        traj.last_ticks = ticks
        return {
            "error": bool(res.get("error")),
            "parsed": True,
            "production_score": production,
            "automated_score": automated,
            "ticks": ticks,
        }

    async def read_baseline(self, node: Node, branch: str) -> dict[str, Any] | None:
        """P5 baseline immediately after fork/restore (cumulative counters).

        ``None`` when the counters could not be read. A fabricated ``(0.0,
        0.0)`` is worse than no baseline at all: every score in this module is a
        DELTA from it, so a branch that failed its baseline read would be
        credited with the entire cumulative score of the checkpoint it
        inherited and would beat every sibling whose baseline did land. The
        caller marks such a branch unscorable instead of ranking it.
        """
        try:
            res = await self.infra.execute(node, BASELINE_CODE, branch=branch)
        except asyncio.CancelledError:
            raise  # deadline, not an env error -- Infra holds it for the drain
        except BaseException as exc:  # noqa: BLE001
            if self.is_ambiguous(exc):
                # Contract R2C4: the baseline program may have run. The branch is
                # unscorable either way (no P5 zero) and its sandbox is retired,
                # so nothing steps, probes or re-seeds on it again.
                self.fail_node_ambiguous(node, exc, op="execute", branch=branch)
                return None
            self.incident("baseline_read_failed", f"{type(exc).__name__}: {exc}",
                          branch=branch)
            return None
        if res.get("error"):
            # The bridge answered, but the program that reads the counters did
            # not run: the numbers in this response are whatever the sandbox
            # last held, not a measured baseline.
            self.incident(
                "baseline_read_failed",
                f"/execute reported an error: {str(res.get('result', ''))[:400]}",
                branch=branch,
            )
            return None
        return {
            "production": float(res.get("production_score", 0.0) or 0.0),
            "automated": float(res.get("automated_score", 0.0) or 0.0),
            # The tick counter the branch INHERITED (a fork keeps its source's,
            # C's restore resets it). The feedback template charges each step's
            # tick cost against it, so a branch left at 0 reports its first
            # step as having burned the whole history.
            "ticks": int(res.get("ticks", 0) or 0),
        }

    # -- parity probe (v2.6: DIRECT, zero measurement forks) ---------------
    async def probe_line(
        self, node: Node, *, branch: str, step: int, kind: str = "parity"
    ) -> dict[str, Any] | None:
        """ONE fixed 60s window (P3), executed on the sandbox owning this line.

        v2.6 replaces v2.4's measurement fork (snapshot -> fork -> health ->
        probe -> delete child -> delete snapshot; measured at 121-285s in
        Tier 0) with a direct ``/probe``. Parity is preserved exactly as
        before: every arm -- and every B/C branch -- gets one fixed-window
        probe per m steps from the same instrument, so the per-lineage side
        effect (60 in-game seconds of advance, nothing else) is identical
        everywhere and no arm holds a measurement oracle. The depletion
        objection of P3 targeted ``verify()``'s VARIABLE plateau loop, which
        is still disabled. Measurement forks: zero.

        The first probe on a given sandbox is flagged ``cold``: a freshly
        forked or freshly created microVM faults pages in during it, and that
        tax is part of B's treatment (v2.6 point 3), so it is measured rather
        than amortised away.
        """
        if node.id in self.quarantined:
            # Fail closed: a sandbox holding a substrate call this run could not
            # join is still being mutated by it, so anything measured here is a
            # reading off a moving target, not a probe.
            self.incident(
                "probe_skipped_unsettled",
                f"{node.name} holds a substrate call this run could not join; "
                "no probe of that sandbox can be honest",
                branch=branch, step=step,
            )
            return None
        self._probe_seq += 1
        cold = node.id not in self._probed_nodes
        self._probed_nodes.add(node.id)
        t0 = time.monotonic()
        try:
            res = await self.infra.probe(node, self.entity, branch=branch)
        except asyncio.CancelledError:
            raise  # deadline, not a probe failure
        except BaseException as exc:  # noqa: BLE001 - probe failure is survivable
            if self.is_ambiguous(exc):
                # Contract R2C4: /probe is a mutating call (it advances the game
                # by the measurement window). An ambiguous failure means an
                # unknown slice of that window may have been applied, so the
                # sandbox is quarantined instead of measured or reused.
                self.fail_node_ambiguous(node, exc, op="probe", branch=branch,
                                         step=step)
                return None
            self.incident(
                "probe_failed", f"{type(exc).__name__}: {exc}",
                branch=branch, step=step, cold=cold,
            )
            return None
        client_wall_s = round(time.monotonic() - t0, 3)
        extras = {
            k: res[k]
            for k in ("window_ticks", "speed", "start_count", "end_count",
                      "timed_out")
            if k in res
        }
        out = {
            "throughput": float(res.get("throughput", 0.0) or 0.0),
            "wall_s": float(res.get("wall_s", 0.0) or 0.0),
            "start_tick": int(res.get("start_tick", 0) or 0),
            "end_tick": int(res.get("end_tick", 0) or 0),
            "client_wall_s": client_wall_s,
            "cold": cold,
            **extras,
        }
        timed_out = bool(extras.get("timed_out"))
        if timed_out:
            self.incident("probe_window_timed_out",
                          f"probe window did not complete: {extras}", branch=branch,
                          step=step)
        self.journal.probe_result(
            entity=self.entity, throughput=out["throughput"], wall_s=out["wall_s"],
            start_tick=out["start_tick"], end_tick=out["end_tick"], branch=branch,
            step=step, kind=kind, sandbox=node.name, mode="direct", cold=cold,
            client_wall_s=client_wall_s, valid=not timed_out, **extras,
        )
        if timed_out:
            # Journaled raw above, with its incident, and returned to NOBODY: a
            # window that never closed is a partial count over an unknown span,
            # and every caller treats what it gets back as a measurement.
            return None
        return out

    def probe_block(self, probe: dict[str, Any]) -> str:
        return PROBE_TEMPLATE.format(
            entity=self.entity, throughput=probe["throughput"], quota=self.quota,
            start_tick=probe["start_tick"], end_tick=probe["end_tick"],
        )

    async def parity_probe(self, traj: Trajectory, *, kind: str = "parity") -> None:
        """Mid-run probe for A/A×K/(C main): result injected, cost charged to T."""
        if not self.budget.can_afford(self.cfg.probe_cost_estimate_s):
            self.incident(
                "probe_skipped_budget",
                f"{self.budget.remaining_s():.0f}s left < "
                f"{self.cfg.probe_cost_estimate_s:.0f}s estimate",
                branch=traj.tid, step=traj.step,
            )
            return
        probe = await self.probe_line(traj.node, branch=traj.tid, step=traj.step,
                                      kind=kind)
        if probe is None:
            if traj.node.id in self.quarantined and not traj.partial:
                # An ambiguous /probe (contract R2C4) retired this sandbox mid
                # line: there is no reconciled world left to step on, so the
                # line stops here instead of prompting against an unknown state.
                traj.partial = (
                    f"{traj.node.name} was quarantined during this line's {kind} "
                    f"probe; line {traj.tid} is partial and stops here"
                )
            return
        traj.conv.inject(self.probe_block(probe))
        traj.curve.add(t_s=self.budget.elapsed_s(), step=traj.step,
                       throughput=probe["throughput"], branch=traj.tid, kind=kind)

    # -- provisioning ------------------------------------------------------
    async def provision_main(self, role: str = "main") -> Node:
        """One seat, created from this run's SEED snapshot (``cfg.template_snap``).

        The seed is just an id: TEMPLATE_SNAP for a greenfield run, a baked
        checkpoint for the -from-S variants (A×K-from-S = A×K with
        ``template_snap=S2``). Nothing here re-derives task setup or resets the
        sandbox, so whatever state the snapshot holds is what the seat inherits.
        """
        if not self.cfg.template_snap:
            raise ValueError(
                "template_snap (seed snapshot id: TEMPLATE_SNAP or a baked "
                "checkpoint) is required"
            )
        node = await self.infra.create_from_snapshot(self.cfg.template_snap, role)
        if not self.system_prompt:
            self.system_prompt = await self.infra.system_prompt(node)
        return node

    # -- state dumps (post-hoc renders; see bench/blog_shots.py) -------------
    def _state_dumps_enabled(self) -> bool:
        return os.environ.get("FLE_BENCH_STATE_DUMPS", "") not in ("", "0")

    def _dump_state(self, label: str, state: str) -> None:
        """Persist one /state-save blob next to the journal.

        Host-side disk write ONLY -- the caller must already hold the blob;
        this never issues a bridge call of its own, so it can run inside the
        measured window without touching any timing bucket.
        """
        out_dir = self.journal.path.parent / f"{self.journal.run_id}.states"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{label}.state.json"
        path.write_text(state, encoding="utf-8")
        self.journal.event("state_dump", label=label, path=str(path),
                           chars=len(state))


    async def teardown(self, nodes: Sequence[Node]) -> None:
        if self._state_dumps_enabled():
            # Every driver calls teardown() AFTER run.timings.stop(), before
            # sandbox deletion. node.bridge is called directly (NOT through
            # Infra._timed) so the save lands in no timing bucket -- not even
            # the out-of-window accounting. Failures are journaled per node and
            # never block the deletes below.
            for i, node in enumerate(list(nodes)):
                if node.bridge is None:
                    # Owned but never attached (expose or the health poll
                    # failed): there is no bridge to save state through.
                    continue
                try:
                    state = await asyncio.to_thread(node.bridge.state_save)
                    self._dump_state(f"final-{i:02d}-{node.label}", state)
                except asyncio.CancelledError:
                    raise  # teardown is best effort, cancellation is not
                except BaseException as exc:  # noqa: BLE001
                    self.incident("state_dump_failed",
                                  f"{type(exc).__name__}: {exc}",
                                  target=node.name)
        # Deliberate best-effort teardown loops: one node's delete must not stop
        # the others, so every failure is journaled and the loop goes on. A
        # CANCELLATION is not such a failure -- the whole teardown is being torn
        # down -- so it is re-raised before the broad catch.
        for node in list(nodes):
            try:
                await self.infra.delete(node)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                self.incident("teardown_delete_failed", f"{type(exc).__name__}: {exc}",
                              target=node.name)
        for snap in list(self.infra.live_snapshots):
            if snap in self.infra.orphan_sources:
                # Deliberate residue: this snapshot is the reaper's only
                # ownership handle on a fork child that may still be landing.
                # Deleting it here would strand that child as unowned.
                self.incident(
                    "snapshot_retained_for_orphan",
                    f"{snap} kept so the reaper can claim the timed-out fork "
                    "child by source; both are swept by the block reaper",
                    target=snap,
                )
                continue
            try:
                await self.infra.delete_snapshot(snap)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                self.incident("teardown_snapshot_delete_failed",
                              f"{type(exc).__name__}: {exc}", target=snap)

    # -- terminal endpoint -------------------------------------------------
    async def terminal_probe(self, traj: Trajectory) -> dict[str, Any] | None:
        """The endpoint. Strictly ordered AFTER every substrate call concludes.

        A step abandoned on the T deadline leaves its ``/execute`` running in a
        worker thread; probing before it returns would measure a factory that is
        still being built. The join is bounded by the step cap so a wedged call
        cannot hold the endpoint hostage -- and when the join does time out, the
        sandbox is quarantined and this line gets NO endpoint, because the only
        alternative is publishing a number read off a moving factory.
        """
        drained = await self.infra.drain(self.cfg.step_timeout_s)
        for item in drained:
            if item["outcome"] == "abandoned":
                self.incident(
                    "drain_timeout",
                    f"{item['op']} on {item['target']} was still running after a "
                    f"{self.cfg.step_timeout_s:.0f}s join",
                    branch=traj.tid,
                )
        # Any line's drain may have been the one to give up on THIS line's call
        # (the join is global), so the verdict is read off Infra's ledger.
        if traj.node.id in self.infra.abandoned_nodes:
            detail = (f"{traj.node.name} holds a substrate call abandoned after a "
                      f"{self.cfg.step_timeout_s:.0f}s join; this line has no "
                      "honest endpoint")
            self.quarantine(traj.node, detail, branch=traj.tid)
            traj.partial = detail
            traj.pending_exec = None
        elif traj.pending_exec is not None:
            # The global drain above may have settled THIS line's abandoned
            # /execute. That program is the last thing that happened to this
            # factory, so its result is committed before the endpoint is read
            # rather than left out of the line's counters and journal.
            self.commit_recovered_step(traj, reason="terminal drain")
        probe = await self.probe_line(
            traj.node, branch=traj.tid, step=traj.step, kind="terminal"
        )
        traj.terminal_probe = probe
        if probe is not None:
            traj.curve.add(t_s=self.budget.elapsed_s(), step=traj.step,
                           throughput=probe["throughput"], branch=traj.tid,
                           kind="terminal")
        return probe


# ---------------------------------------------------------------------------
# Arm A / A×K
# ---------------------------------------------------------------------------


def _provider_dead(outcomes: Sequence[Any]) -> "ProviderDead | None":
    """The first :class:`~bench.llm.ProviderDead` among gathered outcomes, if any.

    Every parallel arm gathers its seats with ``return_exceptions=True``, which is
    right for a seat that broke and wrong for a provider that died: the run must
    not spend the rest of T sampling a quota that is gone. Callers journal the
    per-seat incidents as usual, then re-raise this AFTER teardown so the
    orchestrator sees the cause and can abort every cell on that provider.
    """
    for out in outcomes:
        if isinstance(out, ProviderDead):
            return out
    return None


def _live_nodes(run: ArmRun, *trajs: "Trajectory | None") -> list[Node]:
    """Every sandbox this run still owns, de-duplicated.

    Driven by ``infra.live_sandboxes`` (every seat that was attached and not
    yet deleted) rather than by the trajectories alone, so a run that DIES
    DURING PROVISIONING still tears down the seats it had already brought up
    instead of leaking them into the next cell of a run-cap-1 block.
    """
    nodes: list[Node] = [t.node for t in trajs if t is not None]
    nodes.extend(getattr(run, "pool", ()))
    nodes.extend(run.infra.live_sandboxes.values())
    return list({n.id: n for n in nodes}.values())


async def _sequential_loop(
    run: ArmRun, traj: Trajectory, *, hint_at_branch: str | None,
    stop_at_boundary: Callable[[], bool] | None = None,
    until_s: float | None = None,
) -> None:
    """One A trajectory: step until T, parity probe every m steps.

    ``stop_at_boundary`` is consulted at each m-step boundary, after that
    boundary's probe: it is how arm B-once finds its single convergence point
    (the first boundary at or past T/2) without perturbing the step or probe
    cadence, and how arm B resumes the canonical line after it stops
    converging. Left None, the loop simply runs to T.

    ``until_s`` is an EARLIER deadline on the same budget clock, measured from
    ``budget.start()``: Exp 3's leg 1 ends at P even though T is 2P. It clips
    the per-step timeout as well as the loop condition, because a step still
    mutating the sandbox when the leg ends would race the selection probe that
    decides the whole run. Left None, T is the only deadline (every Exp-2 arm).
    """
    cfg = run.cfg

    def leg_left_s() -> float:
        if until_s is None:
            return float("inf")
        return until_s - run.budget.elapsed_s()

    if traj.partial:
        # Handed in already broken (an ambiguous bridge call, a lost step result
        # or a round that could not be scored): a PARTIAL line never steps again.
        run.incident("line_partial_no_steps", traj.partial, branch=traj.tid,
                     step=traj.step)
        return
    while not run.budget.expired() and leg_left_s() > 0.0 and not traj.partial:
        is_branch_step = (traj.step % cfg.m) == 0
        hint = hint_at_branch if (is_branch_step and hint_at_branch) else None
        deadline_s = min(run.step_deadline_s(), leg_left_s())
        try:
            await asyncio.wait_for(
                run.agent_step(traj, hint=hint), timeout=deadline_s
            )
        except asyncio.TimeoutError:
            traj.errors += 1
            run.incident("step_timeout",
                         f"step {traj.step} exceeded {deadline_s:.1f}s "
                         f"(cap {cfg.step_timeout_s:.0f}s, clipped to T"
                         f"{'/leg' if until_s is not None else ''})",
                         branch=traj.tid)
            # wait_for cancelled the awaiting coroutine, NOT the worker thread:
            # the program is still running against this sandbox. Join it before
            # the next step or the boundary probe touches the same node.
            if not await run.settle_node(traj, reason=f"step {traj.step} timeout"):
                return
        except BudgetExhausted:
            break
        if traj.step % cfg.m == 0:
            # The leg's own probe is the SELECTION probe (Hybrid) or the
            # terminal one (leg 2), taken by the caller; a parity probe fired
            # after the leg boundary would only spend the next leg's clock.
            if not run.budget.expired() and leg_left_s() > 0.0:
                await run.parity_probe(traj)
            if traj.partial:
                return
            if stop_at_boundary is not None and stop_at_boundary():
                return


async def run_arm_a(run: ArmRun) -> RunResult:
    result = _new_result(run)
    traj: Trajectory | None = None
    t_provision = time.monotonic()
    try:
        # Provisioning sits INSIDE the teardown guard: a seat that comes up and
        # is then orphaned by a later failure must be reaped by this run, not
        # left for the next cell to trip over.
        main = await run.provision_main("main")
        result.provision_s = time.monotonic() - t_provision
        traj = Trajectory(tid="A", node=main, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm="A", run_id=run.cfg.run_id)
        await _sequential_loop(run, traj, hint_at_branch=None)
        await run.terminal_probe(traj)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, traj))
        result.teardown_s = time.monotonic() - t_teardown
    _finish(run, result, [traj])
    return result


async def run_arm_axk(run: ArmRun) -> RunResult:
    """K independent A trajectories in parallel; best terminal probe wins."""
    cfg = run.cfg
    result = _new_result(run)
    trajs: list[Trajectory] = []
    dead: ProviderDead | None = None
    t_provision = time.monotonic()
    try:
        # Under the teardown guard: A×K brings up K seats one at a time, so a
        # failure on seat i leaves i-1 live sandboxes that must die with it.
        nodes = [await run.provision_main(f"axk{i}") for i in range(cfg.K)]
        result.provision_s = time.monotonic() - t_provision
        trajs = [
            Trajectory(tid=f"AxK{i}", node=node, conv=run.new_conversation())
            for i, node in enumerate(nodes)
        ]
        hints = run.hints_for(cfg.K) or [None] * cfg.K
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm="AxK", run_id=cfg.run_id, k=cfg.K)

        async def one(traj: Trajectory, hint: str | None) -> None:
            await _sequential_loop(run, traj, hint_at_branch=hint)
            await run.terminal_probe(traj)

        outcomes = await asyncio.gather(
            *(one(t, h) for t, h in zip(trajs, hints)), return_exceptions=True
        )
        dead = _provider_dead(outcomes)
        for traj, out in zip(trajs, outcomes):
            if isinstance(out, BaseException):
                run.incident("trajectory_failed", f"{type(out).__name__}: {out}",
                             branch=traj.tid)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, *trajs))
        result.teardown_s = time.monotonic() - t_teardown
    _finish(run, result, trajs)
    if dead is not None:
        # Teardown and the endpoint bookkeeping are done; the CAUSE goes up so
        # the orchestrator can abort every other cell on this provider.
        raise dead
    return result


# ---------------------------------------------------------------------------
# Arm B / C (branch-and-converge)
# ---------------------------------------------------------------------------


class RoundAborted(RuntimeError):
    """The fork wave could not staff a round inside T.

    Distinct from a round FAILURE (survivable, retried as a sequential step):
    this says the budget is gone, so the loop must stop converging entirely and
    hand the line to the sequential loop for whatever is left of T.
    """


class BranchingRun(ArmRun):
    """Shared branch round for B (fork) and C (state save/restore)."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.pool: list[Node] = []  # arm C only

    # -- deadline model ----------------------------------------------------
    def rollout_estimate_s(self) -> float:
        """The part of a round AFTER the wave: m steps + the per-branch probe."""
        cfg = self.cfg
        return cfg.m * cfg.step_cost_estimate_s + cfg.probe_cost_estimate_s

    def cleanup_estimate_s(self) -> float:
        """Releasing the round's snapshot and its losers (measured: delete ~1s).

        Reserved explicitly rather than hoped for: at K=8 that is 8 deletes, and
        a round that fits only if cleanup is free is a round that ends past T.
        """
        return (1 + max(0, self.cfg.K - 1)) * self.cfg.delete_cost_estimate_s

    def round_estimate_s(self) -> float:
        """Wall clock ONE full branch round needs, from measured constants.

        The rollout and its cleanup only; subclasses add their branching
        mechanism's cost (see :meth:`ForkBranchingRun.round_estimate_s`).
        """
        return self.rollout_estimate_s() + self.cleanup_estimate_s()

    # -- child materialisation --------------------------------------------
    async def _materialize_children_fork(
        self, main: Node, want: int, round_idx: int
    ) -> tuple[list[Node], str | None, bool]:
        """B: snapshot main, then ``want`` sequential width-1 forks.

        The wave carries an ABSOLUTE deadline, at two levels. Between forks we
        re-check that what is left still covers the next fork PLUS the rollout
        PLUS cleanup PLUS the terminal-probe reserve. And each fork is itself
        submitted with that same remaining time as its poll budget, because a
        single admitted fork can otherwise sit in the Farplane queue (cap 5m;
        the contended soak p95 was 758s) long past T -- the call runs in a
        worker thread and cannot be cancelled from here.

        A fork that burns its whole deadline is failed FOR THIS ROUND (never
        retried -- there is no time), truncating the wave exactly like the
        budget check does. Its child may still land afterwards, so the source
        snapshot stays owned and un-deleted and the reaper claims the child
        through it (:meth:`Infra.register_orphan_fork`).
        """
        snap = await self.snapshot_for_round(main, round_idx)
        children: list[Node] = []
        truncated = ""
        tail_s = self.rollout_estimate_s() + self.cleanup_estimate_s()
        for i in range(want):
            label = f"r{round_idx}b{i + 1}"
            need_s = self.cfg.fork_cost_estimate_s + tail_s
            if not self.budget.can_afford(need_s):
                truncated = (
                    f"{self.budget.remaining_s():.1f}s left < {need_s:.1f}s "
                    f"(one fork + rollout + cleanup) after {len(children)} of "
                    f"{want} forks"
                )
                self.incident("fork_wave_truncated", truncated,
                              branch=f"r{round_idx}")
                break
            deadline_hit = ""
            for attempt in range(self.cfg.child_retries + 1):
                # Recomputed per attempt: the deadline is absolute, not a
                # per-call allowance that a retry gets to spend again.
                deadline_s = max(0.0, self.budget.remaining_s() - tail_s)
                t_fork = time.monotonic()
                try:
                    children.append(
                        await self.infra.fork(snap, label, deadline_s=deadline_s)
                    )
                    break
                except asyncio.CancelledError:
                    raise  # deadline, not a capacity failure: never retry it
                except BaseException as exc:  # noqa: BLE001
                    spent = time.monotonic() - t_fork
                    self.incident(
                        "child_fork_failed",
                        f"attempt {attempt + 1} after {spent:.1f}s of a "
                        f"{deadline_s:.1f}s deadline: {type(exc).__name__}: {exc}",
                        branch=label,
                    )
                    if deadline_s > 0.0 and spent >= 0.9 * deadline_s:
                        deadline_hit = (
                            f"fork {label} burned its {deadline_s:.1f}s deadline "
                            f"({type(exc).__name__}); no time to retry"
                        )
                        self.infra.register_orphan_fork(
                            snap_id=snap, label=label, detail=deadline_hit,
                        )
                        break
            if deadline_hit:
                truncated = (
                    f"{deadline_hit}; stopped at {len(children)} of {want} forks"
                )
                self.incident("fork_wave_truncated", truncated,
                              branch=f"r{round_idx}")
                break
        self.journal.event(
            "fork_wave", round=round_idx, wanted=want, materialized=len(children),
            k_effective=len(children) + 1, truncated=bool(truncated),
            reason=truncated, fork_estimate_s=self.cfg.fork_cost_estimate_s,
            orphans=len(self.infra.orphan_forks),
            remaining_s=round(self.budget.remaining_s(), 3),
        )
        # Fencing: forks are terminal (the wrapper polls), so the spent snapshot
        # is released immediately -- at most one branch snapshot per run. UNLESS
        # a timed-out fork may still produce a child from it: the snapshot is
        # then the only ownership handle the reaper has on that child, so it is
        # retained deliberately and swept with it.
        if snap in self.infra.orphan_sources:
            self.journal.event(
                "branch_snapshot_retained", round=round_idx, snapshot=snap,
                reason="a timed-out fork child may still land; the reaper claims "
                       "it by source snapshot, which requires the snapshot to "
                       "stay in our ledger",
            )
        else:
            try:
                await self.infra.delete_snapshot(snap, branch=f"r{round_idx}")
            except asyncio.CancelledError:
                raise  # torn down mid-cleanup; the reaper owns the residue
            except BaseException as exc:  # noqa: BLE001
                self.incident("branch_snapshot_delete_failed",
                              f"{type(exc).__name__}: {exc}", branch=f"r{round_idx}")
        return children, snap, bool(truncated)

    async def snapshot_for_round(self, main: Node, round_idx: int) -> str:
        return await self.infra.snapshot(main, branch=f"r{round_idx}")

    async def _materialize_children_restore(
        self, main: Node, want: int, round_idx: int
    ) -> tuple[list[Node], str | None, bool]:
        """C: /state-save on main, /state-restore onto ``want`` pool sandboxes."""
        state = await self.infra.state_save(main, branch=f"r{round_idx}")
        if self._state_dumps_enabled():
            # The blob is already in hand for this round's restores; persisting
            # it is a pure disk write with zero extra bridge traffic.
            self._dump_state(f"r{round_idx}", state)
        source_meta = await self.infra.meta(main, branch=f"r{round_idx}")
        children: list[Node] = []
        for i in range(min(want, len(self.pool))):
            node = self.pool[i]
            branch = f"r{round_idx}b{i + 1}"
            ok = False
            for attempt in range(self.cfg.child_retries + 1):
                try:
                    await self.infra.state_restore(node, state, branch=branch)
                    ok = True
                    break
                except asyncio.CancelledError:
                    raise  # the deadline, not a restore failure: never retried
                except BaseException as exc:  # noqa: BLE001
                    if self.is_ambiguous(exc):
                        # Contract R2C4: a restore that may have half-applied is
                        # NEVER retried -- a second restore onto an unknown state
                        # is not a recovery. The pool sandbox is quarantined
                        # (release() drops it for good) and the branch is skipped.
                        self.fail_node_ambiguous(node, exc, op="state_restore",
                                                 branch=branch)
                        break
                    self.incident(
                        "child_restore_failed",
                        f"attempt {attempt + 1}: {type(exc).__name__}: {exc}",
                        branch=branch,
                    )
            if not ok:
                continue
            # P7: restore is lossy (fluid_box dropped, ore replenished, counters
            # reset). Log the divergence per branch rather than assume parity.
            try:
                child_meta = await self.infra.meta(node, branch=branch)
                # game_tick is the real Factorio tick; elapsed_ticks is FLE's
                # virtual counter and only moves when agent code sleeps.
                src_tick = _tick_of(source_meta)
                child_tick = _tick_of(child_meta)
                self.journal.write(
                    "fidelity",
                    branch=branch,
                    source_tick=src_tick,
                    child_tick=child_tick,
                    source_elapsed_ticks=source_meta.get("elapsed_ticks"),
                    child_elapsed_ticks=child_meta.get("elapsed_ticks"),
                    source_entities=source_meta.get("entity_count"),
                    child_entities=child_meta.get("entity_count"),
                    tick_delta=child_tick - src_tick,
                    entity_delta=(child_meta.get("entity_count", 0) or 0)
                    - (source_meta.get("entity_count", 0) or 0),
                    same_pid=child_meta.get("factorio_pid")
                    == source_meta.get("factorio_pid"),
                )
                if (child_meta.get("entity_count", 0) or 0) != (
                    source_meta.get("entity_count", 0) or 0
                ):
                    self.incident(
                        "restore_entity_mismatch",
                        f"source {source_meta.get('entity_count')} -> child "
                        f"{child_meta.get('entity_count')}",
                        branch=branch,
                    )
            except asyncio.CancelledError:
                raise  # the deadline: the round is over, not merely unlogged
            except BaseException as exc:  # noqa: BLE001
                self.incident("fidelity_check_failed", f"{type(exc).__name__}: {exc}",
                              branch=branch)
            children.append(node)
        return children, None, False

    # -- one branch rollout -----------------------------------------------
    async def _run_branch(
        self,
        *,
        bid: str,
        node: Node,
        prefix: Conversation,
        user_content: str,
        candidate: Sample,
        round_idx: int,
        base_step: int,
    ) -> BranchOutcome:
        cfg = self.cfg
        conv = prefix.branch()
        traj = Trajectory(tid=bid, node=node, conv=conv, step=base_step)
        t0 = time.monotonic()
        baseline = await self.read_baseline(node, bid)
        unscorable = ""
        if baseline is None:
            # No P5 zero, so no delta from it means anything. The branch still
            # runs (its steps are real work, its transcript is archived) but it
            # is out of the ranking -- see :meth:`branch_round`.
            unscorable = (
                f"branch {bid} has no P5 baseline: the cumulative counters could "
                "not be read after materialisation"
            )
            self.incident("branch_unscorable", unscorable, branch=bid)
        traj.last_production = baseline["production"] if baseline else 0.0
        traj.last_automated = baseline["automated"] if baseline else 0.0
        traj.last_ticks = baseline["ticks"] if baseline else 0
        score = ScoreRecord(
            baseline_production=traj.last_production,
            baseline_automated=traj.last_automated,
        )
        # T is hard here too: the round-level estimate keeps a round from
        # STARTING past the deadline, and this keeps the seat that is already
        # in flight from overrunning it. The first candidate is not exempt --
        # an always-executes first step is exactly how a fork wave that landed
        # late used to push a B run a full step_timeout past T.
        if self.budget.expired():
            self.incident(
                "branch_step_skipped_budget",
                f"branch {bid} reached the deadline before its first candidate "
                f"({self.budget.remaining_s():.1f}s left); scored on the state "
                "it inherited",
                branch=bid,
            )
        else:
            traj.step += 1
            conv.append_turn(user_content, candidate.text or "(empty response)")
            deadline_s = self.step_deadline_s()
            try:
                await asyncio.wait_for(
                    self.apply_program(traj, candidate, step=traj.step),
                    timeout=deadline_s,
                )
            except asyncio.TimeoutError:
                traj.errors += 1
                self.incident("step_timeout",
                              f"branch {bid} candidate step {traj.step} exceeded "
                              f"{deadline_s:.1f}s", branch=bid)
                # The candidate is still executing on this branch's sandbox:
                # join it before another step or the branch probe reads it.
                await self.settle_node(traj, reason=f"branch {bid} candidate step")
        for _ in range(cfg.m - 1):
            if traj.partial or self.budget.expired():
                break
            deadline_s = self.step_deadline_s()
            try:
                await asyncio.wait_for(
                    self.agent_step(traj, tag=f"@r{round_idx}"),
                    timeout=deadline_s,
                )
            except asyncio.TimeoutError:
                traj.errors += 1
                self.incident("step_timeout",
                              f"branch {bid} step {traj.step} exceeded "
                              f"{deadline_s:.1f}s", branch=bid)
                if not await self.settle_node(traj, reason=f"branch {bid} step"):
                    break
        # P5: endpoint captured right after the m-th program, BEFORE any probe.
        score.endpoint_production = traj.last_production
        score.endpoint_automated = traj.last_automated
        rollout_s = time.monotonic() - t0
        # None for a quarantined sandbox, a /probe that failed and a window that
        # never closed -- in every case this branch HAS NO MEASUREMENT.
        probe = await self.probe_line(node, branch=bid, step=traj.step, kind="branch")
        if probe is not None:
            score.probe_throughput = probe["throughput"]
            traj.curve.add(t_s=self.budget.elapsed_s(), step=traj.step,
                           throughput=probe["throughput"], branch=bid, kind="branch")
        unscorable = unscorable or traj.partial
        if probe is None and not unscorable:
            # A branch with no probe cannot be RANKED: ``ScoreRecord.rank_key``
            # reads a missing throughput as 0.0, so the historical-flow tie-break
            # decides the round and can promote a seat nobody ever measured.
            unscorable = (
                f"branch {bid} has no branch probe: the fixed-window measurement "
                "this round ranks on failed, timed out or was skipped"
            )
            self.incident("branch_unscorable", unscorable, branch=bid,
                          step=traj.step)
        return BranchOutcome(
            branch=bid, conv=conv, node=node, score=score, steps=traj.step - base_step,
            candidate_chars=len(candidate.code or ""), errors=traj.errors,
            probe=probe, rollout_s=rollout_s, last_ticks=traj.last_ticks,
            unscorable=unscorable,
        )

    # -- the round ---------------------------------------------------------
    async def branch_round(self, main: Trajectory, round_idx: int) -> Trajectory:
        cfg = self.cfg
        self.journal.event(
            "round_start", round=round_idx,
            remaining_s=round(self.budget.remaining_s(), 3),
            round_estimate_s=round(self.round_estimate_s(), 3),
            elapsed_s=round(self.budget.elapsed_s(), 3), step=main.step,
        )
        want_children = cfg.K - 1
        # The shared user turn is identical for every branch: deep-copied common
        # prefix + pending feedback (P4). Consumed once, here.
        user_content = main.conv.next_user_content()
        prefix = main.conv.branch()
        prompt = prefix.prompt_with(user_content)
        # Hint rotation at re-seed: this round's K seats get the divergent set
        # rotated by round index, so no seat repeats its previous strategy and
        # the re-seeded branches cannot be clones by construction.
        hints = self.hints_for(cfg.K, offset=round_idx - 1)
        # Each seat's own first post-fork user turn = shared turn + its hint
        # (bench.llm.HINT_TEMPLATE -- the wave-1 mechanism), so the strategy is
        # committed to that branch's transcript and stays in context for all m
        # steps instead of only steering the first sample. The sampling call
        # below keeps the existing hints= path, which appends the same line as
        # its own trailing user turn: either way a prompt carries it exactly once.
        seat_content = [
            f"{user_content}\n\n{HINT_TEMPLATE.format(hint=h)}" if h else user_content
            for h in (hints or [None] * cfg.K)
        ]
        if hints:
            self.journal.write(
                "hint_assignment", round=round_idx,
                hints={f"r{round_idx}b{i}": hints[i] for i in range(cfg.K)},
            )

        sample_task = asyncio.create_task(
            self.llm.sample_detailed(
                prompt, n=cfg.K, hints=hints, branch=f"r{round_idx}",
                step=main.step + 1,
            )
        )
        infra_task = asyncio.create_task(
            self.materialize(main.node, want_children, round_idx)
        )
        sampled, materialized = await asyncio.gather(
            sample_task, infra_task, return_exceptions=True
        )
        if isinstance(sampled, BaseException):
            # Nothing executed, so the canonical line gets its consumed turn
            # back: the caller falls back to a sequential step, and without
            # this the last program's feedback is silently dropped.
            main.conv.pending_feedback = user_content
            if not isinstance(materialized, BaseException):
                await self.release(materialized[0], keep=None)
            raise sampled
        wave_truncated = False
        if isinstance(materialized, BaseException):
            self.incident("branch_materialization_failed",
                          f"{type(materialized).__name__}: {materialized}",
                          branch=f"r{round_idx}")
            children: list[Node] = []
        else:
            children, _spent_snap, wave_truncated = materialized

        candidates = list(sampled)
        branch_nodes = [main.node] + children
        if len(branch_nodes) < 2 and wave_truncated:
            # The deadline, not the substrate: there is no time for a wave, so
            # there is no time for ANY further convergence. Give the turn back
            # to the canonical line and tell the loop to stop converging --
            # running m steps here as a fake "round" would only add a snapshot
            # and a probe the budget does not have.
            main.conv.pending_feedback = user_content
            self.incident(
                "round_aborted_deadline",
                f"the fork wave put up 0 of {want_children} children inside T; "
                "no further convergence, the canonical line runs to T",
                branch=f"r{round_idx}",
            )
            raise RoundAborted(
                f"round {round_idx}: fork wave truncated to 0 children by the T "
                f"deadline ({self.budget.remaining_s():.1f}s left)"
            )
        self.branch_points += 1
        if len(branch_nodes) < 2:
            self.incident(
                "degenerate_round",
                f"only {len(branch_nodes)} branch(es) available; round runs as A",
                branch=f"r{round_idx}",
            )
        usable = min(len(branch_nodes), len(candidates))
        if usable < 1:
            # Pre-execution too: hand the turn back before failing the round.
            main.conv.pending_feedback = user_content
            raise RuntimeError(
                f"round {round_idx}: {len(candidates)} candidate(s) for "
                f"{len(branch_nodes)} node(s); nothing to roll out"
            )
        base_step = main.step
        outcomes = await asyncio.gather(
            *(
                self._run_branch(
                    bid=f"r{round_idx}b{i}", node=branch_nodes[i], prefix=prefix,
                    user_content=seat_content[i], candidate=candidates[i],
                    round_idx=round_idx, base_step=base_step,
                )
                for i in range(usable)
            ),
            return_exceptions=True,
        )
        good: list[BranchOutcome] = []
        for i, out in enumerate(outcomes):
            if isinstance(out, asyncio.CancelledError):
                # gather(return_exceptions=True) CAPTURES cancellation. The round
                # is being torn down, not failing, and nothing below may pretend
                # otherwise.
                raise out
            if isinstance(out, BaseException):
                self.incident("branch_failed", f"{type(out).__name__}: {out}",
                              branch=f"r{round_idx}b{i}")
            else:
                good.append(out)
        dead = _provider_dead(outcomes)
        if dead is not None:
            # The quota is gone. The caller's generic recovery would sample it
            # again with a fallback step and swallow the cause in its own broad
            # catch, so it goes up NOW -- before anything is promoted or adopted.
            self.incident("round_provider_dead", f"{type(dead).__name__}: {dead}",
                          branch=f"r{round_idx}")
            raise dead
        if not good:
            # No branch state survived, and the pre-round main is not a fallback:
            # its turn was consumed by this wave and its sandbox ran this round's
            # first candidate, so its transcript and its world have come apart.
            detail = (
                f"round {round_idx}: every branch failed, so no branch state "
                "survived; the canonical line cannot resume from the pre-round "
                "main, whose turn was consumed here and whose sandbox this round "
                "already mutated"
            )
            self.fail_line(main, detail, kind="line_partial_round_failed")
            await self.release(
                [n for n in branch_nodes if n.id != main.node.id], keep=None,
            )
            raise RuntimeError(detail)
        # Ranking is over branches that CAN be ranked: no P5 baseline, no branch
        # probe, or a sandbox left with a call this run could not join, means the
        # numbers are not comparable with their siblings'. Excluded branches stay
        # in the journal, and their transcripts are archived, as diagnostics.
        scorable = [o for o in good if not o.unscorable]
        excluded = {o.branch: o.unscorable for o in good if o.unscorable}
        if not scorable:
            raise RuntimeError(await self._round_unscorable(
                main, good, branch_nodes, prefix=prefix, base_step=base_step,
                round_idx=round_idx, excluded=excluded,
            ))

        winner = max(scorable, key=lambda o: o.score.rank_key())
        losers = [o for o in good if o is not winner]
        self.journal.write(
            "branch_selection",
            round=round_idx,
            winner=winner.branch,
            k_effective=len(scorable),
            excluded=excluded,
            scores={
                o.branch: {
                    **o.score.to_dict(),
                    "rollout_s": round(o.rollout_s, 3),
                    "steps": o.steps,
                    "errors": o.errors,
                }
                for o in good
            },
        )
        for loser in losers:
            # P4: loser transcripts are artifacts, never re-prompted.
            self.journal.archive_branch(
                branch=loser.branch, step=base_step + loser.steps,
                messages=loser.conv.messages[len(prefix.messages):],
                score=loser.score.to_dict(),
            )

        # Winner promotion: canonical history := prefix + winner turns + winner
        # pending feedback (the Conversation object already is exactly that).
        promoted = winner.conv
        if winner.probe is not None:
            promoted.inject(self.probe_block(winner.probe))
        new_main = Trajectory(
            tid="main",
            node=winner.node,
            conv=promoted,
            step=base_step + winner.steps,
            errors=main.errors + winner.errors,
            last_production=winner.score.endpoint_production,
            last_automated=winner.score.endpoint_automated,
            # The winner's OWN tick counter, not the pre-round main's: the
            # promoted line continues from the state the winner ended in.
            last_ticks=winner.last_ticks,
            curve=main.curve,
        )
        # SECONDARY curve: every branch probe of this round, winner and losers.
        for o in good:
            new_main.curve.points.append(
                {
                    "t_s": round(self.budget.elapsed_s(), 3),
                    "step": base_step + o.steps,
                    "throughput": (o.probe or {}).get("throughput", 0.0),
                    "branch": o.branch,
                    "kind": "branch",
                }
            )
        # Every node this round MATERIALIZED goes back except the winner's --
        # including the nodes of branches that raised, which used to hold a
        # warm slot until teardown, and the pre-round main when a child won.
        await self.release(branch_nodes, keep=winner.node, main_before=main.node)
        return new_main

    async def _round_unscorable(
        self, main: Trajectory, good: Sequence[BranchOutcome],
        branch_nodes: Sequence[Node], *, prefix: Conversation, base_step: int,
        round_idx: int, excluded: dict[str, str],
    ) -> str:
        """No branch of this round can be ranked: move the line, claim nothing.

        There is no winner -- the round produced no comparable measurement -- but
        there is also no way back. The pre-round main handed its turn to the wave
        and its sandbox ran this round's first candidate, so resuming it would
        prompt "Continue" against a world its transcript never saw. One branch is
        therefore ADOPTED for canonical continuity only, IN PLACE, so the
        caller's recovery runs on state that actually exists: journaled as an
        adoption and not a selection, with no probe block injected (there is no
        measurement to inject) and every other branch released and archived.

        When nothing is adoptable -- every survivor sits on a quarantined sandbox
        -- the line ends as PARTIAL instead. Returns the message the caller
        raises, so the round is still a failure to the loop above.
        """
        reason = ("; ".join(excluded.values()))[:1000]
        self.journal.write("round_unscorable", round=round_idx, excluded=excluded,
                           branches=len(good), step=base_step)
        main_before = main.node
        adoptable = [o for o in good if o.node.id not in self.quarantined]
        if not adoptable:
            detail = (
                f"round {round_idx}: none of the {len(good)} surviving branch(es) "
                f"is scorable and every one sits on a quarantined sandbox "
                f"({reason}); the canonical line ends here rather than resume the "
                "pre-round main, whose turn was consumed and whose sandbox this "
                "round mutated"
            )
            self.fail_line(main, detail, kind="line_partial_round_unscorable")
            await self.release(
                [n for n in branch_nodes if n.id != main_before.id], keep=None,
            )
            return detail
        adopted = adoptable[0]
        for o in good:
            if o is adopted:
                continue
            # P4: these transcripts are artifacts, never re-prompted.
            self.journal.archive_branch(
                branch=o.branch, step=base_step + o.steps,
                messages=o.conv.messages[len(prefix.messages):],
                score=o.score.to_dict(), reason="unscorable-round",
            )
        main.node = adopted.node
        main.conv = adopted.conv
        main.step = base_step + adopted.steps
        main.errors += adopted.errors
        main.last_production = adopted.score.endpoint_production
        main.last_automated = adopted.score.endpoint_automated
        main.last_ticks = adopted.last_ticks
        main.pending_exec = None
        detail = (
            f"round {round_idx}: none of the {len(good)} surviving branch(es) is "
            f"scorable ({reason}); {adopted.branch} was adopted for canonical "
            "continuity only -- it is not a selection and this round measured "
            "nothing"
        )
        self.incident("round_unscorable_adopted", detail, branch=adopted.branch)
        self.journal.write(
            "branch_adoption", round=round_idx, adopted=adopted.branch,
            selection=False, reason="round_unscorable", step=main.step,
            sandbox=adopted.node.name, excluded=excluded,
        )
        await self.release(branch_nodes, keep=adopted.node,
                           main_before=main_before)
        return detail

    # -- arm-specific hooks ------------------------------------------------
    async def materialize(
        self, main: Node, want: int, round_idx: int
    ) -> tuple[list[Node], str | None, bool]:
        """``(children, spent snapshot, wave truncated by the T deadline)``."""
        raise NotImplementedError

    async def release(
        self, nodes: Sequence[Node], *, keep: Node | None,
        main_before: Node | None = None,
    ) -> None:
        raise NotImplementedError


class ForkBranchingRun(BranchingRun):
    """Arm B."""

    def round_estimate_s(self) -> float:
        """Snapshot + the SERIAL (K-1)-fork train + the rollout.

        This is Exp 2's own sizing arithmetic read backwards: a p50 round is
        10.1s snapshot + 7 x 62.3s forks + m x 19.9s steps + one probe. If that
        does not fit in what is left of T, the round must not start -- the
        alternative is a fork wave that lands after the deadline and a run that
        blows past T.
        """
        cfg = self.cfg
        return (
            cfg.snapshot_cost_estimate_s
            + max(0, cfg.K - 1) * cfg.fork_cost_estimate_s
            + super().round_estimate_s()
        )

    async def materialize(
        self, main: Node, want: int, round_idx: int
    ) -> tuple[list[Node], str | None, bool]:
        return await self._materialize_children_fork(main, want, round_idx)

    async def release(
        self, nodes: Sequence[Node], *, keep: Node | None,
        main_before: Node | None = None,
    ) -> None:
        # Deliberate best-effort cleanup loop: one loser's delete must not keep
        # the round from releasing the rest, so failures are journaled and the
        # loop continues. A CANCELLATION is not one of those failures -- the run
        # itself is going away -- so it is re-raised before the broad catch.
        for node in nodes:
            if keep is not None and node.id == keep.id:
                continue
            try:
                await self.infra.delete(node)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                self.incident("loser_delete_failed", f"{type(exc).__name__}: {exc}",
                              target=node.name)


class RestoreBranchingRun(BranchingRun):
    """Arm C: branches by GameState restore onto a fixed pool (CPU parity with B)."""

    async def materialize(
        self, main: Node, want: int, round_idx: int
    ) -> tuple[list[Node], str | None, bool]:
        return await self._materialize_children_restore(main, want, round_idx)

    async def release(
        self, nodes: Sequence[Node], *, keep: Node | None,
        main_before: Node | None = None,
    ) -> None:
        """Pool sandboxes are reused, never deleted; the old main rejoins the pool."""
        keep_id = keep.id if keep else None
        pool: list[Node] = []
        seen: set[str] = set()
        for node in list(nodes) + ([main_before] if main_before else []) + self.pool:
            if node is None or node.id == keep_id or node.id in seen:
                continue
            seen.add(node.id)
            if node.id in self.quarantined:
                # Never restored onto again: a sandbox holding a call this run
                # could not join has an unknown state, and the pool is the
                # fixture every C branch is supposed to start from cleanly.
                continue
            pool.append(node)
        self.pool = pool


async def _branching_loop(run: BranchingRun, main: Trajectory) -> Trajectory:
    """Converge every m steps until T can no longer afford a WHOLE round.

    T is the sole stopping rule and it is HARD. A round that cannot finish
    inside the remaining budget must never start: its serial fork train would
    land after the deadline and the run would overrun T by most of a wave. When
    the next round no longer fits, the canonical line just continues -- same
    steps, same probe cadence, no further convergence -- and ``run_arm_b``
    still takes the terminal probe at T out of the reserve.
    """
    round_idx = 0
    while not run.budget.expired():
        need_s = run.round_estimate_s()
        if not run.budget.can_afford(need_s):
            run.incident(
                "round_skipped_budget",
                f"{run.budget.remaining_s():.1f}s left < {need_s:.1f}s round "
                f"estimate after {round_idx} round(s); continuing the canonical "
                "line to T without another convergence",
                branch=f"r{round_idx + 1}",
            )
            run.journal.event(
                "convergence_stopped", rounds_completed=round_idx,
                remaining_s=round(run.budget.remaining_s(), 3),
                round_estimate_s=round(need_s, 3), step=main.step,
            )
            await _sequential_loop(run, main, hint_at_branch=None)
            break
        round_idx += 1
        try:
            main = await run.branch_round(main, round_idx)
        except ProviderDead:
            # NOT a survivable round failure: a fallback step would sample the
            # same dead quota, and the loop would keep "recovering" until T.
            raise
        except BudgetExhausted:
            break
        except RoundAborted as exc:
            # The wave ran out of clock mid-flight. Stop converging for good
            # and spend what is left on the canonical line.
            run.journal.event(
                "convergence_stopped", rounds_completed=round_idx - 1,
                reason="fork_wave_truncated_to_zero",
                remaining_s=round(run.budget.remaining_s(), 3),
                round_estimate_s=round(run.round_estimate_s(), 3), step=main.step,
            )
            run.incident("convergence_stopped_mid_wave", str(exc),
                         branch=f"r{round_idx}")
            await _sequential_loop(run, main, hint_at_branch=None)
            break
        except asyncio.CancelledError:
            raise  # the run is being torn down; a fallback step would race it
        except BaseException as exc:  # noqa: BLE001 - round failure is survivable
            run.incident("round_failed", f"{type(exc).__name__}: {exc}",
                         branch=f"r{round_idx}")
            if run.budget.expired():
                break
            if main.partial:
                # The round could not leave a line this run can account for
                # (:meth:`BranchingRun._round_unscorable`). A fallback step would
                # prompt against a sandbox whose state nobody can reconcile.
                run.journal.event(
                    "convergence_stopped", rounds_completed=round_idx - 1,
                    reason="canonical_line_partial", detail=main.partial[:500],
                    remaining_s=round(run.budget.remaining_s(), 3), step=main.step,
                )
                break
            # Fall back to a plain sequential step so the run keeps compounding.
            try:
                await asyncio.wait_for(
                    run.agent_step(main, tag="@fallback"),
                    timeout=run.step_deadline_s(),
                )
            except asyncio.CancelledError:
                raise  # the T deadline, not a fallback-step failure
            except ProviderDead:
                # The quota is gone: the round's own recovery cannot be the thing
                # that hides it from the orchestrator.
                raise
            except BaseException as exc2:  # noqa: BLE001
                run.incident("fallback_step_failed", f"{type(exc2).__name__}: {exc2}")
                break
    return main


async def run_arm_b(run: BranchingRun) -> RunResult:
    result = _new_result(run)
    main: Trajectory | None = None
    t_provision = time.monotonic()
    try:
        # Provisioning under the teardown guard (see :func:`_live_nodes`).
        main_node = await run.provision_main("main")
        result.provision_s = time.monotonic() - t_provision
        main = Trajectory(tid="main", node=main_node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm=run.cfg.arm, run_id=run.cfg.run_id,
                          k=run.cfg.K, m=run.cfg.m,
                          round_estimate_s=round(run.round_estimate_s(), 3))
        main = await _branching_loop(run, main)
        await run.terminal_probe(main)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, main))
        result.teardown_s = time.monotonic() - t_teardown
    _finish(run, result, [main])
    return result


async def run_arm_b_once(run: BranchingRun) -> RunResult:
    """Exp 2's single-dose curve point: ONE convergence, then straight to T.

    Three phases on one budget: a single seat from S to the LAST m-step
    boundary that can still afford a whole round; exactly one fork wave +
    m-step branch rollout + selection there; then the promoted winner continues
    alone to T. Everything else -- K, m, hints, probe cadence, terminal probe
    -- is arm B's, so the only difference measured against B-iterated is the
    DOSE, which is the point of the curve.

    The boundary rule is affordability, NOT a literal T/2, because under p95
    admission the two disagree and T/2 loses: at T=4200s with m=33 the first
    boundary at or past T/2 is the 4th (~2627s), which leaves ~1483s against a
    ~1781s round estimate -- B-once would deterministically degenerate to
    A-continue and the curve point would never exist. The last affordable
    boundary is the 3rd (~1970s, 47% of T), so the descriptive "midpoint"
    reading survives while the convergence is guaranteed to happen. The
    realized point is journaled (``chosen_boundary``) and reported, never
    assumed.
    """
    cfg = run.cfg
    result = _new_result(run)
    main: Trajectory | None = None
    t_provision = time.monotonic()
    midpoint_s = 0.5 * cfg.T_s

    def at_last_affordable_boundary() -> bool:
        """True at the last m-boundary whose remaining budget fits a round."""
        need_s = run.round_estimate_s()
        if not run.budget.can_afford(need_s):
            # Already unaffordable, and remaining only shrinks: stop looking.
            return True
        # One more m-block of canonical stepping costs a rollout; if the round
        # would not fit after that, this boundary is the last one that works.
        return not run.budget.can_afford(need_s + run.rollout_estimate_s())

    try:
        main_node = await run.provision_main("main")
        result.provision_s = time.monotonic() - t_provision
        main = Trajectory(tid="main", node=main_node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm=cfg.arm, run_id=cfg.run_id, k=cfg.K,
                          m=cfg.m, midpoint_s=round(midpoint_s, 3),
                          round_estimate_s=round(run.round_estimate_s(), 3),
                          block_estimate_s=round(run.rollout_estimate_s(), 3))
        # Phase 1: one seat, S -> the last boundary that can still pay for a round.
        await _sequential_loop(
            run, main, hint_at_branch=None,
            stop_at_boundary=at_last_affordable_boundary,
        )
        # Phase 2: the single convergence -- subject to the same hard-T rule.
        boundary_s = run.budget.elapsed_s()
        boundary_step = main.step
        need_s = run.round_estimate_s()
        run.journal.event(
            "chosen_boundary", elapsed_s=round(boundary_s, 3),
            remaining_s=round(run.budget.remaining_s(), 3),
            estimate_s=round(need_s, 3), step=boundary_step,
            midpoint_s=round(midpoint_s, 3),
            fraction_of_T=round(boundary_s / cfg.T_s, 4) if cfg.T_s else 0.0,
            affordable=run.budget.can_afford(need_s),
        )
        if run.budget.expired() or not run.budget.can_afford(need_s):
            run.incident(
                "round_skipped_budget",
                f"last boundary reached with {run.budget.remaining_s():.1f}s left "
                f"< {need_s:.1f}s round estimate; B-once degenerates to "
                "A-continue",
                branch="r1",
            )
        else:
            try:
                main = await run.branch_round(main, 1)
            except ProviderDead:
                raise  # the quota is gone; nothing left to converge onto
            except RoundAborted as exc:
                run.incident("convergence_stopped_mid_wave", str(exc), branch="r1")
            except asyncio.CancelledError:
                raise  # torn down mid-convergence, not a survivable round failure
            except BaseException as exc:  # noqa: BLE001 - survivable, journaled
                run.incident("round_failed", f"{type(exc).__name__}: {exc}",
                             branch="r1")
            else:
                run.journal.event(
                    "bonce_convergence",
                    boundary_s=round(boundary_s, 3),
                    boundary_step=boundary_step,
                    elapsed_s=round(run.budget.elapsed_s(), 3),
                    midpoint_s=round(midpoint_s, 3), step=main.step,
                    winner_sandbox=main.node.name, k=cfg.K,
                )
        # Phase 3: the promoted winner IS the canonical line from here to T.
        await _sequential_loop(run, main, hint_at_branch=None)
        await run.terminal_probe(main)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, main))
        result.teardown_s = time.monotonic() - t_teardown
    _finish(run, result, [main])
    return result


async def run_arm_c(run: RestoreBranchingRun) -> RunResult:
    result = _new_result(run)
    main: Trajectory | None = None
    t_provision = time.monotonic()
    try:
        main_node = await run.provision_main("main")
        # Pool created at run start from TEMPLATE_SNAP (K-1 sandboxes, CPU parity).
        run.pool = [
            await run.infra.create_from_snapshot(run.cfg.template_snap, f"pool{i + 1}")
            for i in range(run.cfg.K - 1)
        ]
        result.provision_s = time.monotonic() - t_provision
        main = Trajectory(tid="main", node=main_node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm="C", run_id=run.cfg.run_id, k=run.cfg.K,
                          m=run.cfg.m, pool=len(run.pool))
        main = await _branching_loop(run, main)
        await run.terminal_probe(main)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, main))
        result.teardown_s = time.monotonic() - t_teardown
    _finish(run, result, [main])
    return result


# ---------------------------------------------------------------------------
# Experiment 3: Hybrid (one convergence + judge at the bell) and A×K-2P
# ---------------------------------------------------------------------------


def _persona_conversation(
    run: ArmRun, persona: str | None, *, prefix: Conversation | None = None
) -> Conversation:
    """A seat's conversation with its persona in that seat's FIRST user turn.

    Exactly the channel arm B commits a per-round strategy hint through
    (``bench.llm.HINT_TEMPLATE``, appended to the seat's own first user
    content): the persona becomes part of the seat's transcript and stays in
    context for the whole phase, instead of steering a single sample. It is
    injected ONCE per seat per phase -- ``pending_extras`` is consumed by the
    first ``next_user_content()`` -- which is what makes the phase-2
    re-injection a genuinely new conditioning event rather than a repeat.

    ``prefix`` re-seeds the seat from an existing line (Exp 3's leg 2 forks:
    every seat inherits the winner's history verbatim, then diverges).
    """
    conv = prefix.branch() if prefix is not None else run.new_conversation()
    if persona:
        conv.inject(HINT_TEMPLATE.format(hint=persona))
    return conv


def _journal_personas(
    run: ArmRun, *, phase: int, trajs: Sequence[Trajectory],
    personas: Sequence[str | None], offset: int,
) -> dict[str, Any]:
    """One ``persona_assignment`` record per phase: seat -> persona, verbatim."""
    n = len(run.cfg.hints) or 1
    seats = {
        t.tid: {
            "index": (i + offset) % n if personas[i] else None,
            "persona": personas[i] or "",
        }
        for i, t in enumerate(trajs)
    }
    rec = {"phase": phase, "offset": offset, "seats": seats}
    run.journal.write("persona_assignment", **rec)
    return rec


async def _hybrid_select(
    run: ArmRun, trajs: Sequence[Trajectory]
) -> tuple[Trajectory, dict[str, Any], dict[str, Any] | None]:
    """Exp 3's ONE selection: the fixed probe on every seat, then the best.

    Ordered exactly like a terminal probe: in-flight ``/execute`` calls are
    joined first, because a seat still building when it is measured would win or
    lose on a state that is still moving -- and this single measurement decides
    which line the whole second leg descends from. Every seat's score is
    journaled (``branch_selection``), losers included; that record IS the
    leg-1 distribution the analysis compares against leg 2's.
    """
    drained = await run.infra.drain(run.cfg.step_timeout_s)
    for item in drained:
        if item["outcome"] == "abandoned":
            run.incident(
                "drain_timeout",
                f"{item['op']} on {item['target']} was still running after a "
                f"{run.cfg.step_timeout_s:.0f}s join; the selection may race it",
            )
    for traj in trajs:
        # A seat whose call could not be joined is out of the running: its state
        # is still moving, so neither its selection probe nor its sandbox (which
        # leg 2 would descend from) can be trusted.
        if traj.node.id in run.infra.abandoned_nodes:
            detail = (f"{traj.node.name} holds a substrate call abandoned after a "
                      f"{run.cfg.step_timeout_s:.0f}s join")
            run.quarantine(traj.node, detail, branch=traj.tid)
            traj.partial = detail
            traj.pending_exec = None
        elif traj.pending_exec is not None:
            # The drain above settled this seat's abandoned /execute. Its result
            # is what the seat is about to be RANKED on, so it is committed
            # rather than dropped.
            run.commit_recovered_step(traj, reason="Exp-3 selection drain")
    probes = await asyncio.gather(
        *(
            run.probe_line(t.node, branch=t.tid, step=t.step, kind="selection")
            for t in trajs
        ),
        return_exceptions=True,
    )
    scored: list[tuple[Trajectory, ScoreRecord, dict[str, Any] | None]] = []
    for traj, probe in zip(trajs, probes):
        if isinstance(probe, BaseException):
            run.incident("selection_probe_failed",
                         f"{type(probe).__name__}: {probe}", branch=traj.tid)
            probe = None
        # Every seat started from the SAME checkpoint, so the cumulative
        # counters are directly comparable and the P5 baseline is the shared
        # zero: no per-seat baseline read (and no extra program) is needed for
        # the rank key to mean what it means in arm B.
        score = ScoreRecord(
            endpoint_production=traj.last_production,
            endpoint_automated=traj.last_automated,
            probe_throughput=None if probe is None else probe["throughput"],
        )
        if probe is not None:
            traj.curve.add(t_s=run.budget.elapsed_s(), step=traj.step,
                           throughput=probe["throughput"], branch=traj.tid,
                           kind="selection")
        scored.append((traj, score, probe))
    # Quarantined seats are journaled with the rest (that record IS the leg-1
    # distribution) but must never be promoted: leg 2 would run on a sandbox
    # this run cannot account for. An UNMEASURED seat is excluded for a second
    # reason: ``ScoreRecord.rank_key`` reads a missing probe as 0.0 throughput,
    # so the historical-flow tie-break would settle Exp 3's ONLY selection
    # between seats nobody measured.
    excluded: dict[str, str] = {}
    eligible: list[tuple[Trajectory, ScoreRecord, dict[str, Any] | None]] = []
    for traj, score, probe in scored:
        if traj.partial:
            excluded[traj.tid] = traj.partial
        elif probe is None:
            excluded[traj.tid] = (
                f"seat {traj.tid} has no selection probe: the fixed-window "
                "measurement this selection ranks on failed, timed out or was "
                "skipped"
            )
            run.incident("seat_unscorable", excluded[traj.tid], branch=traj.tid,
                         step=traj.step)
        else:
            eligible.append((traj, score, probe))
    if not eligible:
        raise RuntimeError(
            f"Exp-3 selection: none of the {len(scored)} leg-1 seat(s) is "
            f"scorable ({'; '.join(excluded.values())}); there is no measured "
            "state leg 2 could honestly descend from"
        )
    winner, winner_score, winner_probe = max(eligible, key=lambda s: s[1].rank_key())
    record = {
        "round": 1,
        "phase": 1,
        "winner": winner.tid,
        "k_effective": len(eligible),
        "seats_scored": len(scored),
        "excluded": excluded,
        "scores": {
            traj.tid: {
                **score.to_dict(),
                "steps": traj.step,
                "errors": traj.errors,
                "sandbox": traj.node.name,
            }
            for traj, score, _ in scored
        },
    }
    run.journal.write("branch_selection", **record)
    for traj, score, _ in scored:
        if traj is winner:
            continue
        # P4: loser transcripts are artifacts, never re-prompted. Archived
        # BEFORE the sandboxes go away, so a leg-1 line is fully reconstructable.
        run.journal.archive_branch(
            branch=traj.tid, step=traj.step, messages=traj.conv.messages,
            score=score.to_dict(), reason="exp3-leg1-loser",
        )
    return winner, record, winner_probe


class HybridRun(ForkBranchingRun):
    """Exp 3's Hybrid: arm B's fork wave, used exactly once, at full width.

    Only the admission TAIL differs from arm B, and it has to. B admits each
    fork against "one more fork + the rest of THIS ROUND (m steps + a probe) +
    cleanup", because a B round is a fixed, bounded unit of work. Exp 3's
    post-wave work is a whole leg of length P, and the wave is charged to T, so
    the same rule would demand P seconds after every fork and refuse the entire
    wave by construction -- manufacturing the truncation the width floor exists
    to detect. The leg is admitted separately and explicitly
    (:data:`EXP3_PHASE2_ADMISSION`, journaled as ``phase2_admission``), so here
    a fork only has to leave room for the deletes; the terminal-probe reserve is
    already held out of ``Budget.remaining_s``. Everything else -- p95 fork
    admission, absolute per-fork deadlines, wave truncation, orphan
    registration, snapshot fencing -- is inherited verbatim.
    """

    def rollout_estimate_s(self) -> float:
        return 0.0


async def run_arm_hybrid(run: HybridRun) -> RunResult:
    """Exp 3's treatment: two full-length legs joined by ONE convergence.

    Leg 1 is A×K-from-S at width K with a persona per seat. At P it stops, every
    seat is probed once (the selection), the winner's sandbox is snapshotted and
    K-1 children are forked beside it -- Exp 2's fork wave, used exactly once --
    the personas are re-injected ROTATED, and leg 2 runs another P. At T every
    seat is probed and the endpoint is the MAX, so the order statistic matches
    the control by construction.

    Budget: ``T_total = 2P + measured selection overhead``. The overhead (drain,
    8 probes, archival, 7 deletes, snapshot, fork wave) is measured and handed
    back to the budget, so neither leg pays for the convergence with its own
    walk length; everything else stays wall-clock-hard through the existing
    :class:`~bench.common.Budget` (per-seat step deadlines, p95 fork admission,
    terminal reserve). If leg 2 still starts with less than
    :data:`EXP3_PHASE2_ADMISSION` x P, that is journaled and the leg runs
    whatever remains -- clipped, labelled, never silently short.
    """
    cfg = run.cfg
    leg_s = cfg.leg_s
    result = _new_result(run)
    leg1: list[Trajectory] = []
    seats: list[Trajectory] = []
    dead: ProviderDead | None = None
    base_step = 0
    exp3: dict[str, Any] = {
        "arm": cfg.arm, "P_s": leg_s, "T_total_s": cfg.T_s, "K": cfg.K,
        "dose": 1, "phases": [], "refork": {}, "selection": {},
        "width_floor": EXP3_WIDTH_FLOOR,
    }
    t_provision = time.monotonic()
    try:
        # Under the teardown guard: K seats come up one at a time (see
        # :func:`_live_nodes`), and a failure on seat i must not leak i-1.
        nodes = [await run.provision_main(f"hyb{i}") for i in range(cfg.K)]
        result.provision_s = time.monotonic() - t_provision
        personas1 = run.hints_for(cfg.K, offset=0) or [None] * cfg.K
        leg1 = [
            Trajectory(tid=f"L1s{i}", node=node,
                       conv=_persona_conversation(run, personas1[i]))
            for i, node in enumerate(nodes)
        ]
        run.budget.start()
        run.timings.start()
        run.journal.event(
            "T_start", arm=cfg.arm, run_id=cfg.run_id, k=cfg.K, m=cfg.m,
            leg_s=leg_s, T_total_s=cfg.T_s,
            round_estimate_s=round(run.round_estimate_s(), 3),
        )
        exp3["phases"].append(
            _journal_personas(run, phase=1, trajs=leg1, personas=personas1, offset=0)
        )

        # ---- leg 1: K independent persona seats, stopped at P ---------------
        run.journal.event("leg_start", phase=1, leg_s=leg_s, seats=len(leg1),
                          elapsed_s=round(run.budget.elapsed_s(), 3))
        outcomes = await asyncio.gather(
            *(
                _sequential_loop(run, t, hint_at_branch=None, until_s=leg_s)
                for t in leg1
            ),
            return_exceptions=True,
        )
        dead = _provider_dead(outcomes)
        for traj, out in zip(leg1, outcomes):
            if isinstance(out, BaseException):
                run.incident("trajectory_failed", f"{type(out).__name__}: {out}",
                             branch=traj.tid)
        if dead is not None:
            # No point snapshotting a winner and paying for a refork wave when
            # the quota that would drive leg 2 is gone: unwind to teardown now.
            run.journal.event("leg_aborted", phase=1, reason="provider_dead",
                              provider=dead.provider, trigger=dead.trigger)
            raise dead
        run.journal.event("leg_done", phase=1,
                          elapsed_s=round(run.budget.elapsed_s(), 3),
                          steps=[t.step for t in leg1])

        # ---- the single convergence -----------------------------------------
        t_sel = time.monotonic()
        winner, selection, winner_probe = await _hybrid_select(run, leg1)
        run.branch_points += 1
        base_step = winner.step
        exp3["selection"] = {
            **selection,
            "winner_sandbox": winner.node.name,
            "winner_steps": winner.step,
            "at_s": round(run.budget.elapsed_s(), 3),
        }
        # The 7 losers die BEFORE the wave: they hold warm slots on the very
        # node the forks pin to, and their transcripts are already archived.
        losers = [t for t in leg1 if t is not winner]
        await run.release([t.node for t in losers], keep=winner.node)
        # Materialisation choice, journaled as a decision: the winner KEEPS its
        # sandbox and K-1 children are forked beside it. That is
        # ``_materialize_children_fork`` verbatim -- one snapshot of the winner,
        # a serial width-1 fork train under p95 admission with an absolute
        # deadline, wave truncation, orphan registration and snapshot fencing --
        # so Exp 3 inherits Exp 2's fork-wave rules instead of restating them.
        # Reforking all 8 from the snapshot would add one avoidable fork plus a
        # delete and change nothing about the state any seat starts from.
        run.journal.event(
            "refork_plan", phase=2, decision="winner continues + K-1 forks",
            wanted=cfg.K - 1, alternative="refork all K from the snapshot",
            rationale=("the winner's sandbox already holds the exact state the "
                       "snapshot would restore; forking it again would cost one "
                       "extra fork (p95 151.6s) and one extra delete for a "
                       "byte-identical seat"),
        )
        children, spent_snap, truncated = await run.materialize(
            winner.node, cfg.K - 1, 1
        )
        overhead_s = time.monotonic() - t_sel
        # T_total = 2P + MEASURED selection overhead: the convergence is the
        # treatment, not a tax on the walks, so it is added to the budget rather
        # than taken out of leg 2. Everything after this line is wall-clock-hard
        # against the extended total.
        run.budget.total_s += overhead_s
        k_effective = 1 + len(children)
        exp3["refork"] = {
            "wanted": cfg.K - 1,
            "materialized": len(children),
            "k_effective": k_effective,
            "truncated": bool(truncated),
            "snapshot": spent_snap,
            "overhead_s": round(overhead_s, 3),
        }
        run.journal.event(
            "selection_overhead", phase=2, overhead_s=round(overhead_s, 3),
            T_total_s=round(run.budget.total_s, 3), leg_s=leg_s,
            k_effective=k_effective, truncated=bool(truncated),
            note="drain + K probes + archival + K-1 deletes + snapshot + fork wave",
        )
        valid_width = k_effective >= EXP3_WIDTH_FLOOR
        if not valid_width:
            run.incident(
                "invalid_width",
                f"phase 2 judged {k_effective} seat(s) < the pre-registered "
                f"Exp-3 floor of {EXP3_WIDTH_FLOOR}: a truncated refork wave "
                "turns judge-at-the-bell into best-of-few, so this endpoint is "
                "not decision-grade",
                branch="phase2",
            )
        exp3["validity"] = {
            "k_effective": k_effective, "width_floor": EXP3_WIDTH_FLOOR,
            "valid_width": valid_width,
            "status": "ok" if valid_width else "invalid_width",
        }
        run.journal.event("exp3_width", phase=2, k_effective=k_effective,
                          width_floor=EXP3_WIDTH_FLOOR, valid=valid_width)

        # ---- leg 2: the same width, personas rotated by one seat ------------
        remaining_s = run.budget.remaining_s()
        required_s = EXP3_PHASE2_ADMISSION * leg_s
        admitted = remaining_s >= required_s
        run.journal.event(
            "phase2_admission", admitted=admitted,
            remaining_s=round(remaining_s, 3), required_s=round(required_s, 3),
            leg_s=leg_s, fraction_of_P=round(remaining_s / leg_s, 4) if leg_s else 0.0,
        )
        if not admitted:
            run.incident(
                "phase2_leg_short",
                f"leg 2 starts with {remaining_s:.1f}s < "
                f"{required_s:.1f}s ({EXP3_PHASE2_ADMISSION:.0%} of P); it runs "
                "whatever remains and the walk is labelled clipped",
                branch="phase2",
            )
        exp3["phase2_admission"] = {
            "admitted": admitted, "remaining_s": round(remaining_s, 3),
            "required_s": round(required_s, 3),
        }
        personas2 = run.hints_for(cfg.K, offset=1) or [None] * cfg.K
        promoted = winner.conv
        if winner_probe is not None:
            # Parity with arm B's promotion: the winner's own line sees its
            # selection score, exactly as every mid-run probe is injected.
            promoted.inject(run.probe_block(winner_probe))
        prefix = promoted.branch()
        seats = [
            Trajectory(
                tid=f"L2s{i}", node=node,
                conv=_persona_conversation(run, personas2[i], prefix=prefix),
                step=base_step,
                errors=winner.errors if i == 0 else 0,
                last_production=winner.last_production,
                last_automated=winner.last_automated,
                last_ticks=winner.last_ticks,
            )
            for i, node in enumerate([winner.node] + children)
        ]
        exp3["phases"].append(
            _journal_personas(run, phase=2, trajs=seats, personas=personas2,
                              offset=1)
        )
        run.journal.event("leg_start", phase=2, leg_s=leg_s, seats=len(seats),
                          base_step=base_step,
                          elapsed_s=round(run.budget.elapsed_s(), 3))

        async def one(traj: Trajectory) -> None:
            await _sequential_loop(run, traj, hint_at_branch=None)
            await run.terminal_probe(traj)

        outcomes2 = await asyncio.gather(
            *(one(t) for t in seats), return_exceptions=True
        )
        dead = _provider_dead(outcomes2)
        for traj, out in zip(seats, outcomes2):
            if isinstance(out, BaseException):
                run.incident("trajectory_failed", f"{type(out).__name__}: {out}",
                             branch=traj.tid)
        run.journal.event("leg_done", phase=2,
                          elapsed_s=round(run.budget.elapsed_s(), 3),
                          steps=[t.step - base_step for t in seats])
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, *leg1, *seats))
        result.teardown_s = time.monotonic() - t_teardown
    # If the run died before leg 2 existed, leg 1 IS the endpoint set: counting
    # it as prior work too would double every step it took.
    _finish_exp3(run, result, endpoint_trajs=seats or leg1,
                 prior_trajs=leg1 if seats else (), exp3=exp3,
                 base_step=base_step if seats else 0)
    if dead is not None:
        raise dead
    return result


async def run_arm_axk_s(run: ArmRun) -> RunResult:
    """Exp 3's middle rung, A×K-S: 8 persona seats at T = 2P, never converged.

    The thinnest variant of :func:`run_arm_axk` the design allows: same seats
    created from the same checkpoint, same probe cadence, same terminal probe on
    every seat, same max-over-K endpoint, and no snapshot or fork anywhere. Two
    deliberate differences: the per-seat diversity text is a PERSONA carried
    once in that seat's first user turn (both WIDE Exp-3 arms inject diversity
    through the same channel, so Hybrid's contrast is the convergence and
    nothing else), and EVERY seat's terminal probe is reported (Exp 3 reads
    per-seat distributions, not only the max).
    """
    cfg = run.cfg
    result = _new_result(run)
    trajs: list[Trajectory] = []
    dead: ProviderDead | None = None
    exp3: dict[str, Any] = {
        "arm": cfg.arm, "P_s": cfg.leg_s, "T_total_s": cfg.T_s, "K": cfg.K,
        "dose": 0, "phases": [],
    }
    t_provision = time.monotonic()
    try:
        nodes = [await run.provision_main(f"axks{i}") for i in range(cfg.K)]
        result.provision_s = time.monotonic() - t_provision
        personas = run.hints_for(cfg.K, offset=0) or [None] * cfg.K
        trajs = [
            Trajectory(tid=f"AxKS{i}", node=node,
                       conv=_persona_conversation(run, personas[i]))
            for i, node in enumerate(nodes)
        ]
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm=cfg.arm, run_id=cfg.run_id, k=cfg.K,
                          m=cfg.m, leg_s=cfg.leg_s, T_total_s=cfg.T_s)
        exp3["phases"].append(
            _journal_personas(run, phase=1, trajs=trajs, personas=personas,
                              offset=0)
        )

        async def one(traj: Trajectory) -> None:
            await _sequential_loop(run, traj, hint_at_branch=None)
            await run.terminal_probe(traj)

        outcomes = await asyncio.gather(
            *(one(t) for t in trajs), return_exceptions=True
        )
        dead = _provider_dead(outcomes)
        for traj, out in zip(trajs, outcomes):
            if isinstance(out, BaseException):
                run.incident("trajectory_failed", f"{type(out).__name__}: {out}",
                             branch=traj.tid)
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, *trajs))
        result.teardown_s = time.monotonic() - t_teardown
    _finish_exp3(run, result, endpoint_trajs=trajs, prior_trajs=(), exp3=exp3,
                 base_step=0)
    if dead is not None:
        raise dead
    return result


async def run_arm_control(run: ArmRun) -> RunResult:
    """Exp 3's bottom rung: ONE agent, no persona, no fork, the full T = 2P.

    Arm A's loop verbatim (:func:`run_arm_a`) -- one seat from the checkpoint,
    parity probes every m steps, the terminal probe on its own factory -- with
    Exp 3's horizon and Exp 3's bookkeeping. It exists so the ladder can price
    WIDTH: Control -> A×K-S is the value of forking wide, and without a
    no-persona single-seat rung that value is only ever inferred.

    ``ArmConfig`` pins K=1 and ``diversify="never"`` for this arm, so a block
    that hands every cell K=8 and a persona set still runs a strict control.
    """
    cfg = run.cfg
    assert cfg.K == 1 and cfg.diversify == "never", (
        f"Control must be one seat with no diversity, got K={cfg.K} "
        f"diversify={cfg.diversify!r}"
    )
    result = _new_result(run)
    traj: Trajectory | None = None
    dead: ProviderDead | None = None
    exp3: dict[str, Any] = {
        "arm": cfg.arm, "P_s": cfg.leg_s, "T_total_s": cfg.T_s, "K": 1,
        "dose": 0, "phases": [], "persona": None,
        "role": "strict control: one agent, neutral prompt, no fork",
    }
    t_provision = time.monotonic()
    try:
        node = await run.provision_main("control")
        result.provision_s = time.monotonic() - t_provision
        traj = Trajectory(tid="Control", node=node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        run.journal.event("T_start", arm=cfg.arm, run_id=cfg.run_id, k=1,
                          m=cfg.m, leg_s=cfg.leg_s, T_total_s=cfg.T_s,
                          persona=None)
        # No hint at any boundary and no persona in the conversation: the
        # transcript this arm produces is the neutral prompt every other arm's
        # seat starts from, which is exactly what makes it the control.
        await _sequential_loop(run, traj, hint_at_branch=None)
        await run.terminal_probe(traj)
    except ProviderDead as exc:
        # The one-seat rung has nothing to gather, so the tripwire surfaces here.
        # Tear down and finish the bookkeeping, then hand the CAUSE up.
        dead = exc
        run.incident("provider_dead", str(exc), branch="Control")
    finally:
        result.provision_s = result.provision_s or time.monotonic() - t_provision
        result.active_s = run.budget.elapsed_s()
        run.timings.stop()
        t_teardown = time.monotonic()
        await run.teardown(_live_nodes(run, traj))
        result.teardown_s = time.monotonic() - t_teardown
    _finish_exp3(run, result, endpoint_trajs=[traj] if traj else [],
                 prior_trajs=(), exp3=exp3, base_step=0)
    if dead is not None:
        raise dead
    return result


def _finish_exp3(
    run: ArmRun, result: RunResult, *, endpoint_trajs: Sequence[Trajectory],
    prior_trajs: Sequence[Trajectory], exp3: dict[str, Any], base_step: int,
) -> None:
    """:func:`_finish` plus Exp 3's per-seat endpoints and phase bookkeeping.

    The endpoint itself is still ``_finish``'s max over terminal probes -- there
    is exactly one endpoint rule in this module. What Exp 3 adds is the full
    per-seat distribution in the results file (``seat_endpoints``), honest step
    accounting across the two legs (leg-2 seats carry the winner's step counter
    for lineage continuity, so their OWN steps are counted here), and the phase
    record the analysis reads.
    """
    result.seat_endpoints = [
        {
            "seat": t.tid,
            "sandbox": t.node.name,
            "steps": t.step - base_step,
            "throughput": (t.terminal_probe or {}).get("throughput"),
            "cold": (t.terminal_probe or {}).get("cold"),
            "production": t.last_production,
            "automated": t.last_automated,
            "errors": t.errors,
        }
        for t in endpoint_trajs
    ]
    measured = [
        s["throughput"] for s in result.seat_endpoints if s["throughput"] is not None
    ]
    exp3["seat_endpoints"] = result.seat_endpoints
    exp3["endpoint_max"] = max(measured) if measured else None
    exp3["seats_probed"] = len(measured)
    result.exp3 = exp3
    leg1_steps = [t.step for t in prior_trajs]
    leg2_steps = [max(0, t.step - base_step) for t in endpoint_trajs]
    _finish(
        run, result, endpoint_trajs,
        steps=sum(leg1_steps) + sum(leg2_steps),
        steps_per_trajectory=leg1_steps + leg2_steps,
    )
    # Curve: leg-1 seats keep their own points (selection probes included), so
    # the plotted history is the whole run, not just the surviving lineage.
    result.curve = [p for t in prior_trajs for p in t.curve.points] + result.curve
    run.journal.write(
        "exp3_endpoints", arm=result.arm, seats=len(result.seat_endpoints),
        seats_probed=exp3["seats_probed"], endpoint=result.endpoint_throughput,
        endpoint_source=result.endpoint_source, max_over_seats=exp3["endpoint_max"],
        seat_endpoints=result.seat_endpoints, validity=exp3.get("validity"),
    )

# ---------------------------------------------------------------------------
# Result assembly & entry point
# ---------------------------------------------------------------------------


def _new_result(run: ArmRun) -> RunResult:
    cfg = run.cfg
    return RunResult(
        run_id=cfg.run_id, arm=cfg.arm, model=cfg.model, task_key=cfg.task_key,
        replicate=cfg.replicate, K=cfg.K, m=cfg.m, T_s=cfg.T_s,
        entity=run.entity, quota=run.quota, model_info=run.llm.model_info(),
        journal_path=str(run.journal.path),
        # Contract R2C3: the journal is an append stream of SESSIONS, so the
        # path alone does not identify this run's evidence -- the session id
        # does, and a consumer grading a verdict has to match it.
        journal_session=run.journal.session,
    )


def _finish(run: ArmRun, result: RunResult, trajs: Sequence[Trajectory], *,
            steps: int | None = None,
            steps_per_trajectory: Sequence[int] | None = None) -> None:
    """Endpoint, curve and totals. ONE endpoint rule for every arm: the max
    terminal probe over the lines handed in.

    The endpoint is only an ENDPOINT when every line was measured. A max over a
    subset is a max over the seats that happened to survive, which reads high
    exactly when a run went wrong, so partial coverage is reported as
    ``status='partial'`` with the unmeasured seats named in ``error``; the
    subset itself is kept as a diagnostic rather than thrown away.

    ``steps``/``steps_per_trajectory`` override the naive per-trajectory sum,
    which Exp 3's Hybrid needs: its leg-2 seats carry the winner's step counter
    (lineage continuity in the journal), so summing them raw would count the
    winner's leg-1 steps K times.
    """
    probed = [(t, t.terminal_probe) for t in trajs if t.terminal_probe]
    unmeasured = [t.tid for t in trajs if not t.terminal_probe]
    run.trajectories = list(trajs)
    if probed:
        best_traj, best_probe = max(probed, key=lambda p: p[1]["throughput"])
        result.endpoint_throughput = best_probe["throughput"]
        result.endpoint_source = best_traj.tid
    if not trajs:
        result.status = "partial"
        result.error = result.error or (
            "no line to measure: this run produced no trajectory at all"
        )
    elif unmeasured:
        # ONE coverage diagnostic, including the 0-of-N case: "no terminal probe"
        # said nothing about WHICH lines went unmeasured, and 0 of N is the case
        # where that list matters most.
        result.status = "partial"
        result.error = result.error or (
            f"partial seat coverage: {len(probed)} of {len(trajs)} line(s) "
            f"measured at T; unmeasured "
            f"({len(unmeasured)}): {', '.join(unmeasured)}"
        )
    result.steps = sum(t.step for t in trajs) if steps is None else int(steps)
    result.steps_per_trajectory = (
        [t.step for t in trajs] if steps_per_trajectory is None
        else [int(s) for s in steps_per_trajectory]
    )
    result.branch_points = run.branch_points
    result.curve = [p for t in trajs for p in t.curve.points]
    result.timings = run.timings.summary()
    result.tokens = run.llm.usage()
    result.incidents = run.incidents
    result.sandboxes_created = run.infra.sandboxes_created
    result.snapshots_created = run.infra.snapshots_created
    result.orphan_forks = list(run.infra.orphan_forks)
    result.end_to_end_s = result.provision_s + result.active_s + result.teardown_s
    run.journal.event("run_finished", **{
        k: v for k, v in result.to_dict().items() if k != "curve"
    })


def build_run(
    cfg: ArmConfig,
    *,
    farplane: FarplaneLike,
    bridge_factory: BridgeFactory,
    llm: LLMClient,
    journal: RunJournal,
) -> ArmRun:
    kw = dict(farplane=farplane, bridge_factory=bridge_factory, llm=llm, journal=journal)
    if cfg.arm == "Hybrid":
        return HybridRun(cfg, **kw)
    if cfg.arm in ("B", "Bonce"):
        return ForkBranchingRun(cfg, **kw)
    if cfg.arm == "C":
        return RestoreBranchingRun(cfg, **kw)
    return ArmRun(cfg, **kw)


async def execute_run(run: ArmRun) -> RunResult:
    arm = run.cfg.arm
    if arm == "A":
        return await run_arm_a(run)
    if arm == "AxK":
        return await run_arm_axk(run)
    if arm == "B":
        assert isinstance(run, ForkBranchingRun)
        return await run_arm_b(run)
    if arm == "Bonce":
        assert isinstance(run, ForkBranchingRun)
        return await run_arm_b_once(run)
    if arm == "Hybrid":
        assert isinstance(run, HybridRun)
        return await run_arm_hybrid(run)
    if arm == "AxK-S":
        return await run_arm_axk_s(run)
    if arm == "Control":
        return await run_arm_control(run)
    assert isinstance(run, RestoreBranchingRun)
    return await run_arm_c(run)


def default_bridge_factory() -> BridgeFactory:
    from bench.bridge_client import Bridge

    return lambda base_url: Bridge(base_url)


async def run_one(
    cfg: ArmConfig,
    *,
    farplane: FarplaneLike | None = None,
    bridge_factory: BridgeFactory | None = None,
    llm: LLMClient | None = None,
    journal: RunJournal | None = None,
) -> RunResult:
    """Run one (arm, model, task, replicate) cell against the real substrate."""
    own_journal = journal is None
    if journal is None:
        os.makedirs(cfg.journal_dir, exist_ok=True)
        journal = RunJournal(
            os.path.join(cfg.journal_dir, f"{cfg.run_id}.jsonl"),
            run_id=cfg.run_id,
            meta={"config": cfg.to_dict()},
        )
    if farplane is None:
        from bench.farplane import Farplane

        farplane = Farplane(os.path.join(cfg.journal_dir, f"{cfg.run_id}-farplane.jsonl"))
    if bridge_factory is None:
        bridge_factory = default_bridge_factory()
    own_llm = llm is None
    if llm is None:
        llm = make_client(cfg.model, journal=journal,
                          max_concurrency=max(4, cfg.K * 2))
    run = build_run(cfg, farplane=farplane, bridge_factory=bridge_factory,
                    llm=llm, journal=journal)
    try:
        return await execute_run(run)
    finally:
        if own_llm:
            await llm.aclose()
        if own_journal:
            journal.close()


# ---------------------------------------------------------------------------
# Mock substrate (--dry): exercises the full loop logic with no network at all
# ---------------------------------------------------------------------------


@dataclass
class FakeSB:
    id: str
    name: str
    node: str
    base_url: str | None = None


@dataclass
class FakeGame:
    """Just enough game to make branch selection and promotion meaningful."""

    ticks: int = 0
    entities: int = 0
    production: float = 0.0
    automated: float = 0.0
    pid: int = 4242
    history: list[str] = field(default_factory=list)

    def clone(self) -> "FakeGame":
        return FakeGame(self.ticks, self.entities, self.production, self.automated,
                        self.pid, list(self.history))


class FakeWorld:
    """Shared store: sandbox id -> game, snapshot id -> frozen game."""

    def __init__(self, *, latency: float = 0.01, seed: int = 7) -> None:
        self.latency = latency
        self.games: dict[str, FakeGame] = {}
        self.snapshots: dict[str, FakeGame] = {}
        self.deleted_sandboxes: list[str] = []
        self.deleted_snapshots: list[str] = []
        self.seq = 0
        self.seed = seed

    def nid(self, kind: str) -> str:
        self.seq += 1
        return f"{kind}-{self.seq:04d}"


class FakeOperationTimeout(RuntimeError):
    """Mirrors :class:`bench.farplane.OperationTimeout` for the fakes."""


class FakeFarplane:
    """In-memory stand-in with the exact bench.farplane surface."""

    def __init__(self, world: FakeWorld, *, fail_forks: int = 0,
                 fork_latency_mult: float = 2.0, slow_fork_at: int = 0,
                 slow_fork_mult: float = 200.0) -> None:
        self.world = world
        self.timings: list[dict[str, Any]] = []
        self.fail_forks = fail_forks
        #: Forks are the long-tailed op on this deployment (p50 62.3s, p95
        #: 151.6s). Raising this makes the wave overrun its own admission
        #: estimate, which is the case the wave deadline exists for.
        self.fork_latency_mult = fork_latency_mult
        #: 1-based index of ONE fork that runs pathologically long -- the
        #: Farplane queue case (cap 5m, contended soak p95 758s). It cannot
        #: finish inside any deadline it will be given, so it always times out
        #: and always leaves an orphan.
        self.slow_fork_at = slow_fork_at
        self.slow_fork_mult = slow_fork_mult
        self._fork_seq = 0
        self.ops: list[str] = []
        #: Children that landed after their fork call gave up on them.
        self.orphans: list[str] = []
        #: Every lease/deadline the SUBSTRATE was actually handed, per call. The
        #: journal records what the caller meant to pass; this records what
        #: arrived, which is what a hibernated-seat post-mortem needs.
        self.create_calls: list[dict[str, Any]] = []
        self.fork_calls: list[dict[str, Any]] = []

    def _sleep(self, mult: float = 1.0) -> None:
        time.sleep(self.world.latency * mult)

    def _record(self, op: str) -> None:
        self.ops.append(op)
        self.timings.append({"op": op, "ts": time.time()})

    def create_from_template(self, template: str, ttl: int, vcpu: int | None = None,
                            mem: int | None = None, name: str = "") -> FakeSB:
        self._sleep(3)
        self._record("create_from_template")
        sid = self.world.nid("sb")
        self.world.games[sid] = FakeGame()
        return FakeSB(id=sid, name=name or sid, node="node-a", base_url=f"fake://{sid}")

    def create_from_snapshot(self, snap_id: str, ttl: int, name: str = "", *,
                             deadline: float | None = None) -> FakeSB:
        self._sleep(3)
        self._record("create_from_snapshot")
        self.create_calls.append(
            {"snapshot": snap_id, "ttl": ttl, "name": name, "deadline": deadline}
        )
        sid = self.world.nid("sb")
        base = self.world.snapshots.get(snap_id) or FakeGame()
        self.world.games[sid] = base.clone()
        return FakeSB(id=sid, name=name or sid, node="node-a", base_url=f"fake://{sid}")

    def snapshot(self, sb: FakeSB) -> str:
        self._sleep(4)
        self._record("snapshot")
        snap = self.world.nid("snap")
        self.world.snapshots[snap] = self.world.games[sb.id].clone()
        return snap

    def fork(self, snap_id: str, ttl: int, name: str = "", *,
             deadline: float | None = None,
             queue_deadline: str = "5m") -> FakeSB:
        """Width-1 fork, honouring ``deadline`` the way the real wrapper does.

        When the fork cannot finish inside its poll budget the caller gets an
        ``OperationTimeout`` -- but the control plane does NOT stop: the child
        lands anyway, unnamed and unattached. That orphan is created here on
        purpose, because it is exactly what the reaper has to own.
        """
        self._fork_seq += 1
        self.fork_calls.append(
            {"snapshot": snap_id, "ttl": ttl, "name": name, "deadline": deadline}
        )
        is_slow = self._fork_seq == self.slow_fork_at
        mult = self.slow_fork_mult if is_slow else self.fork_latency_mult
        want_s = self.world.latency * mult
        # When a slow fork is DESIGNATED, only that one may miss its deadline:
        # the rest are the deterministic-fast control, so the orphan count is a
        # property of the configuration and not of scheduler jitter (a fast
        # fork handed a very short late-wave deadline would otherwise orphan
        # too, and the dry gate would flake).
        may_time_out = is_slow or not self.slow_fork_at
        if may_time_out and deadline is not None and deadline < want_s:
            time.sleep(max(0.0, deadline))
            self._record("fork")
            sid = self.world.nid("sb")
            self.world.games[sid] = self.world.snapshots[snap_id].clone()
            self.orphans.append(sid)
            raise FakeOperationTimeout(
                f"fork of {snap_id} produced no child in {deadline:.2f}s "
                f"(needed {want_s:.2f}s)"
            )
        self._sleep(mult)
        self._record("fork")
        if self.fail_forks > 0:
            self.fail_forks -= 1
            raise RuntimeError("fake fork failure (waiting_for_capacity)")
        sid = self.world.nid("sb")
        self.world.games[sid] = self.world.snapshots[snap_id].clone()
        return FakeSB(id=sid, name=name or sid, node="node-b", base_url=f"fake://{sid}")

    def expose(self, sb: FakeSB, port: int) -> str:
        self._sleep()
        self._record("expose")
        sb.base_url = f"fake://{sb.id}"
        return sb.base_url

    def exec(self, sb: FakeSB, cmd: str) -> str:
        self._sleep()
        self._record("exec")
        return "ok"

    def delete_sandbox(self, sb: FakeSB) -> None:
        self._sleep()
        self._record("delete_sandbox")
        self.world.games.pop(sb.id, None)
        self.world.deleted_sandboxes.append(sb.id)

    def delete_snapshot(self, snap_id: str) -> None:
        self._sleep()
        self._record("delete_snapshot")
        self.world.snapshots.pop(snap_id, None)
        self.world.deleted_snapshots.append(snap_id)

    def reaper(
        self,
        prefix: str | None = None,
        *,
        dry_run: bool = False,
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Same signature and keep semantics as :meth:`bench.farplane.Farplane.reaper`.

        The keep set is honoured for real here, so a dry sweep that would eat
        TEMPLATE_SNAP or the run's seed checkpoint fails the dry validator
        instead of only failing in production.
        """
        self._record("reaper")
        keep_set = {k for k in keep if k}
        outcome = "would-delete" if dry_run else "deleted"
        out: list[dict[str, Any]] = []
        for sid in [s for s in self.world.games if s not in keep_set]:
            out.append({"kind": "sandbox", "id": sid, "name": sid,
                        "reason": "ledger", "outcome": outcome})
            if not dry_run:
                self.world.games.pop(sid, None)
                self.world.deleted_sandboxes.append(sid)
        for snap in [s for s in self.world.snapshots if s not in keep_set]:
            out.append({"kind": "snapshot", "id": snap, "name": "",
                        "reason": "ledger", "outcome": outcome})
            if not dry_run:
                self.world.snapshots.pop(snap, None)
                self.world.deleted_snapshots.append(snap)
        return out


class FakeBridge:
    """Bridge HTTP API v1 semantics over :class:`FakeWorld`."""

    def __init__(self, base_url: str, world: FakeWorld) -> None:
        self.base_url = base_url
        self.sid = base_url.split("//", 1)[1]
        self.world = world

    @property
    def game(self) -> FakeGame:
        game = self.world.games.get(self.sid)
        if game is None:
            raise RuntimeError(f"sandbox {self.sid} is gone")
        return game

    def health(self) -> bool:
        return self.sid in self.world.games

    def wait_healthy(self, deadline_s: float = 300.0) -> None:
        time.sleep(self.world.latency)
        if not self.health():
            raise TimeoutError(f"sandbox {self.sid} never became healthy")

    def execute(self, code: str) -> dict:
        time.sleep(self.world.latency)
        game = self.game
        if code.strip() == BASELINE_CODE:
            return {"result": "", "production_score": game.production,
                    "automated_score": game.automated, "error": False,
                    "ticks": game.ticks}
        # "Quality" is deterministic in the program text so the winner of a
        # branch round is predictable in tests.
        quality = 1 + (sum(ord(c) for c in code[:64]) % 5)
        built = code.count("place_entity") + code.count("connect_entities") + 1
        game.entities += built
        game.production += quality * built
        game.automated += quality * built * 0.5
        game.ticks += 600
        game.history.append(code[:40])
        return {
            "result": f"built {built} entities (quality {quality})",
            "production_score": game.production,
            "automated_score": game.automated,
            "error": False,
            "ticks": game.ticks,
        }

    def probe(self, entity: str) -> dict:
        time.sleep(self.world.latency * 2)
        game = self.game
        start = game.ticks
        game.ticks += 3600  # exactly 60 in-game seconds
        count = round(game.automated / 10.0, 3)
        return {
            "throughput": count,
            "wall_s": self.world.latency * 2,
            "start_tick": start,
            "end_tick": game.ticks,
            "window_ticks": 3600,
            "speed": 10,
            "start_count": 0.0,
            "end_count": count,
            "timed_out": False,
        }

    def state_save(self) -> str:
        time.sleep(self.world.latency)
        g = self.game
        return json.dumps({"ticks": g.ticks, "entities": g.entities,
                           "production": g.production, "history": g.history})

    def state_restore(self, state: str) -> None:
        time.sleep(self.world.latency * 2)
        data = json.loads(state)
        g = self.game
        g.ticks = 0  # P7: restore resets tick/production counters
        g.entities = data["entities"]
        g.production = 0.0
        g.automated = 0.0
        g.history = list(data["history"])

    def system_prompt(self) -> str:
        return "FAKE FLE SYSTEM PROMPT: you have place_entity, connect_entities, ..."

    def meta(self) -> dict:
        g = self.game
        return {
            "factorio_pid": g.pid,
            "elapsed_ticks": g.ticks,
            "game_tick": g.ticks,
            "entity_count": g.entities,
            "speed": 10,
            "paused": False,
            "bench_mode": True,
        }


class FakeRateLimit(RuntimeError):
    """A retryable provider error, shaped like the 429 storm that killed round 1."""

    status_code = 429


class FakeLLM(LLMClient):
    """Deterministic-ish candidate generator; distinct program per call.

    Fault injection, for the provider tripwire:

    ``fail_after``
        after this many successful generations, EVERY call raises a retryable 429
        -- a provider that has gone away mid-block.
    ``empty_every``
        every nth generation raises :class:`~bench.llm.EmptyCompletion` ONCE and
        succeeds on the retry: k3's ~11% empty-200 noise, which must never count
        toward the tripwire.
    """

    def __init__(self, *, latency: float = 0.05, fail_after: int = 0,
                 empty_every: int = 0, **kw: Any) -> None:
        from bench.llm import ModelSpec

        spec = ModelSpec(key="fake-model", provider="fake", api_model="fake",
                         temperature=1.0, supports_n=False, max_tokens=1024,
                         notes="in-memory fake")
        super().__init__(spec, **kw)
        self.latency = latency
        self.fail_after = fail_after
        self.empty_every = empty_every
        self.n_generated = 0
        self.n_attempts = 0
        self.n_injected_429 = 0
        self.n_injected_empty = 0
        self._empty_pending: set[str] = set()

    async def _generate(self, messages, *, n, temperature, request_id):  # type: ignore[override]
        await asyncio.sleep(self.latency)
        self.n_attempts += 1
        if self.fail_after and self.n_generated >= self.fail_after:
            self.n_injected_429 += 1
            raise FakeRateLimit(
                f"429 rate_limit_exceeded (injected after {self.fail_after} calls)"
            )
        if (self.empty_every and self.n_attempts % self.empty_every == 0
                and request_id not in self._empty_pending):
            # Once per request: the retry then succeeds, which is exactly the
            # shape of the noise class the tripwire must ignore.
            self._empty_pending.add(request_id)
            self.n_injected_empty += 1
            from bench.llm import EmptyCompletion

            raise EmptyCompletion(
                "fake-model returned no content (injected empty-200)"
            )
        out: list[Sample] = []
        for _ in range(n):
            self.n_generated += 1
            idx = self.n_generated
            hint = ""
            for m in reversed(messages):
                if m["role"] == "user" and "[Branch strategy hint]" in str(m["content"]):
                    hint = str(m["content"]).split("]", 1)[1].strip()[:40]
                    break
            code = (
                f"# candidate {idx} hint={hint!r}\n"
                f"place_entity('drill', position={{'x': {idx}, 'y': 0}})\n"
                f"connect_entities({idx})\n"
                f"print('candidate {idx} done')\n"
            )
            out.append(
                Sample(
                    text=f"Reasoning for candidate {idx}.\n```python\n{code}```",
                    code=None, model=self.spec.key, provider="fake", latency_s=0.0,
                    prompt_tokens=len(messages) * 40, completion_tokens=120,
                    request_id=request_id,
                )
            )
        return out


def fake_substrate(*, latency: float = 0.01, fail_forks: int = 0,
                   fork_latency_mult: float = 2.0, slow_fork_at: int = 0):
    """Fake world + farplane + bridge factory + a baked TEMPLATE_SNAP id.

    The template is baked exactly as Tier 0 bakes the real one: bring a
    sandbox up, snapshot it, throw the sandbox away. Nothing is left live, so
    the dry run's leak assertions are meaningful. ``fork_latency_mult`` makes
    the serial fork wave slower than its admission estimate; ``slow_fork_at``
    makes ONE fork slower than any deadline it can be given, which is the
    queue-bound case that leaves an orphan child behind.
    """
    world = FakeWorld(latency=latency)
    fp = FakeFarplane(world, fail_forks=fail_forks,
                      fork_latency_mult=fork_latency_mult,
                      slow_fork_at=slow_fork_at)
    seed = fp.create_from_template("debian-warm", 600, name="flebench-template-bake")
    template = fp.snapshot(seed)
    fp.delete_sandbox(seed)
    world.deleted_sandboxes.clear()
    return world, fp, (lambda url: FakeBridge(url, world)), template


# ---------------------------------------------------------------------------
# Live smoke: real LLM + real bridge, one container, no Farplane
# ---------------------------------------------------------------------------


class LoopbackFarplane:
    """Smoke-test stand-in that resolves every 'fork' to the SAME live bridge.

    Exists only to exercise the real agent-step and probe code paths against a
    single container (e.g. a local ``docker run`` of the sandbox image) before
    any Farplane capacity is spent. It is NOT measurement-valid: the probe runs
    on the running factory and advances it by 60 in-game seconds, which is
    exactly what P3 forbids for a real run.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.timings: list[dict[str, Any]] = []
        self.ops: list[str] = []
        self._n = 0

    def _sb(self, name: str) -> FakeSB:
        self._n += 1
        return FakeSB(id=f"loopback-{self._n}", name=name or "loopback",
                      node="loopback", base_url=self.base_url)

    def create_from_template(self, template: str, ttl: int, vcpu: int | None = None,
                            mem: int | None = None, name: str = "") -> FakeSB:
        self.ops.append("create_from_template")
        return self._sb(name)

    def create_from_snapshot(self, snap_id: str, ttl: int, name: str = "", *,
                             deadline: float | None = None) -> FakeSB:
        # ``deadline`` is accepted and ignored: Infra always passes the create
        # poll budget, and a loopback create is instantaneous, so there is
        # nothing to bound -- but refusing the keyword made --live-url a
        # TypeError before the first step.
        self.ops.append("create_from_snapshot")
        return self._sb(name)

    def snapshot(self, sb: FakeSB) -> str:
        self.ops.append("snapshot")
        return "loopback-snap"

    def fork(self, snap_id: str, ttl: int, name: str = "", *,
             deadline: float | None = None, queue_deadline: str = "5m") -> FakeSB:
        self.ops.append("fork")
        return self._sb(name)

    def expose(self, sb: FakeSB, port: int) -> str:
        self.ops.append("expose")
        return self.base_url

    def exec(self, sb: FakeSB, cmd: str) -> str:
        self.ops.append("exec")
        return ""

    def delete_sandbox(self, sb: FakeSB) -> None:
        self.ops.append("delete_sandbox")

    def delete_snapshot(self, snap_id: str) -> None:
        self.ops.append("delete_snapshot")

    def reaper(self, prefix: str) -> list[dict[str, Any]]:
        return []


async def live_smoke(
    *,
    base_url: str,
    model: str = "k3",
    task: str = "iron_ore_throughput",
    steps: int = 2,
    journal_dir: str = "bench/journal/live",
) -> dict[str, Any]:
    """Real model -> real /execute -> real /probe against one running bridge."""
    from bench.bridge_client import Bridge

    os.makedirs(journal_dir, exist_ok=True)
    # Unique per invocation: two smokes of the same model and task must not
    # append into one journal, nor overwrite each other's results file.
    run_id = (f"live-{model.replace('/', '-')}-{task}-"
              f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}")
    journal = RunJournal(os.path.join(journal_dir, f"{run_id}.jsonl"), run_id=run_id,
                         meta={"mode": "live_smoke", "base_url": base_url})
    cfg = ArmConfig(arm="A", model=model, task_key=task, T_s=1e9, K=1, m=steps,
                    template_snap="loopback", terminal_reserve_s=0.0,
                    probe_cost_estimate_s=0.0, journal_dir=journal_dir,
                    run_id=run_id,
                    # No Farplane and no leases here: the smoke drives ONE
                    # already-running bridge, and T_s=1e9 is the "no wall-clock
                    # stop" sentinel, not a horizon a lease could cover.
                    lease_guard=False)
    llm = make_client(model, journal=journal)
    run = build_run(cfg, farplane=LoopbackFarplane(base_url),
                    bridge_factory=lambda url: Bridge(url), llm=llm, journal=journal)
    # ``run_id`` and the journal path are part of the REPORT: the caller writes
    # the results file, and a default path that does not carry this id
    # overwrites the previous smoke's evidence (see :func:`main`).
    out: dict[str, Any] = {"model": model, "task": task, "base_url": base_url,
                           "run_id": run_id, "journal_path": str(journal.path),
                           "journal_session": journal.session, "steps": []}
    try:
        node = await run.provision_main("live")
        out["system_prompt_chars"] = len(run.system_prompt)
        out["meta_before"] = await run.infra.meta(node)
        traj = Trajectory(tid="live", node=node, conv=run.new_conversation())
        run.budget.start()
        run.timings.start()
        for _ in range(steps):
            t0 = time.monotonic()
            res = await run.agent_step(traj)
            out["steps"].append(
                {
                    "step": traj.step,
                    "wall_s": round(time.monotonic() - t0, 2),
                    **{k: v for k, v in res.items() if k != "exception"},
                    "error_detail": res.get("exception", ""),
                }
            )
        probe = await run.probe_line(node, branch="live", step=traj.step,
                                     kind="live_smoke")
        run.timings.stop()
        out["probe"] = probe
        out["meta_after"] = await run.infra.meta(node)
        out["timings"] = run.timings.summary()
        out["tokens"] = llm.usage()
        out["incidents"] = run.incidents
        out["conversation_messages"] = len(traj.conv.messages)
        out["pending_feedback_chars"] = len(traj.conv.pending_feedback or "")
        # Fail closed: every requested step must have run, none of them may
        # carry the error flag (a parse failure or a bridge error is not a
        # smoke pass), and the probe must be a real measurement -- None means
        # it failed or its window never closed.
        out["ok"] = (
            len(out["steps"]) == steps
            and not any(s.get("error") for s in out["steps"])
            and probe is not None
        )
    finally:
        await llm.aclose()
        journal.close()
    return out

# ---------------------------------------------------------------------------
# Dry run: full loop logic against the fakes, with protocol assertions
# ---------------------------------------------------------------------------


#: Every journal the dry run writes, in the order the sections produce them.
DRY_JOURNALS = ("dry-A", "dry-B", "dry-Bhard", "dry-Bwave", "dry-Bforkdl",
                "dry-Adeadline", "dry-Bonce", "dry-AxK", "dry-AxKS",
                "dry-Hybrid", "dry-Hybridtrunc", "dry-Exp3AxKS",
                "dry-Exp3Control", "dry-C")


#: A dry gate that has been waiting this long is dead; steal its lock.
DRY_LOCK_STALE_S = 600.0


def claim_dry_journal_dir(base: str) -> tuple[str, bool]:
    """A journal directory THIS process owns exclusively.

    The dry gate wipes its journals at start and then asserts over their
    contents, so two concurrent invocations sharing a directory silently
    corrupt each other's evidence -- observed as flaky cross-journal counts
    when a reviewer and the author ran the gate at the same time. Take the
    canonical path when it is free, a private sibling otherwise, and say which.
    """
    os.makedirs(base, exist_ok=True)
    lock = os.path.join(base, ".lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return base, True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock)
            except OSError:
                continue
            if age <= DRY_LOCK_STALE_S:
                break
            with contextlib.suppress(OSError):
                os.remove(lock)  # stale: the owner died mid-run
    private = f"{base}-{os.getpid()}"
    os.makedirs(private, exist_ok=True)
    return private, False


async def dry_run(**kw: Any) -> dict[str, Any]:
    """Run the dry gate in a journal directory this process owns exclusively.

    Always releases the lock, so a failed gate does not block the next one.
    """
    base = kw.pop("journal_dir", "bench/journal/dry")
    journal_dir, owned = claim_dry_journal_dir(base)
    try:
        report = await _dry_run(journal_dir=journal_dir, **kw)
    finally:
        if owned:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(journal_dir, ".lock"))
    report["journal_dir"] = journal_dir
    report["journal_dir_exclusive"] = owned
    return report


async def _dry_run(*, T_s: float = 4.0, K: int = 2, m: int = 2, rounds: int = 3,
                   hard_T_s: float = 2.0, bonce_T_s: float = 2.0,
                   wave_T_s: float = 2.1,
                   journal_dir: str = "bench/journal/dry") -> dict[str, Any]:
    os.makedirs(journal_dir, exist_ok=True)
    # Journals append (a crashed real run must keep its evidence), so the dry
    # run starts from empty files or its record counts would accumulate.
    for name in DRY_JOURNALS:
        path = os.path.join(journal_dir, f"{name}.jsonl")
        if os.path.exists(path):
            os.remove(path)
    report: dict[str, Any] = {
        "params": {"T_s": T_s, "K": K, "m": m, "rounds": rounds,
                   "hard_T_s": hard_T_s, "bonce_T_s": bonce_T_s,
                   "wave_T_s": wave_T_s}
    }

    # Exp-2 round sizing: the default m every arm now carries, with the
    # measured arithmetic that produced it.
    sizing = exp2_round_sizing()
    assert sizing["m"] == EXP2_DEFAULT_M
    assert ArmConfig.m == EXP2_DEFAULT_M, (
        f"ArmConfig default m={ArmConfig.m} is not Exp 2's sized "
        f"m={EXP2_DEFAULT_M}"
    )
    assert sizing["round_s_at_m"] >= sizing["required_round_s"], (
        f"m={sizing['m']} rounds are {sizing['round_s_at_m']}s, short of the "
        f"{sizing['required_round_s']}s the fork wave needs"
    )
    report["exp2_sizing"] = {**sizing, "arm_config_default_m": ArmConfig.m}

    # The admission arithmetic the LIVE block will run on, spelled out: sizing
    # is a p50 question (how long is a typical round), admission is a p95 one
    # (will the round I am about to start still fit). T comes from the
    # orchestrator's pre-registration so the two can never drift apart.
    from bench.run_tier1 import EXP2_DOSE_FLOOR, EXP2_T_S

    live = ArmConfig(arm="B", model="k3", task_key="iron_plate_throughput",
                     T_s=EXP2_T_S, K=EXP2_K, m=EXP2_DEFAULT_M,
                     template_snap="s2", run_id="exp2-admission-arithmetic")
    live_estimate = (
        live.snapshot_cost_estimate_s
        + (EXP2_K - 1) * live.fork_cost_estimate_s
        + live.m * live.step_cost_estimate_s + live.probe_cost_estimate_s
        + EXP2_K * live.delete_cost_estimate_s
    )  # the B-slowWave section asserts the method returns exactly this
    live_typical = (
        live.snapshot_cost_estimate_s
        + (EXP2_K - 1) * EXP2_FORK_P50_S
        + live.m * live.step_cost_estimate_s + live.probe_cost_estimate_s
        + EXP2_K * live.delete_cost_estimate_s
    )
    live_budget = live.T_s - live.terminal_reserve_s
    live_block_s = live.m * live.step_cost_estimate_s
    assert live.fork_cost_estimate_s == EXP2_FORK_P95_S > EXP2_FORK_P50_S, (
        "rounds must be admitted on the fork tail, not its median"
    )
    assert exp2_round_sizing()["fork_p50_s"] == EXP2_FORK_P50_S, (
        "m is sized from the p50 wave; that input must not drift to the p95"
    )
    guaranteed = int(live_budget // live_estimate)
    assert guaranteed >= EXP2_DOSE_FLOOR, (
        f"T={live.T_s:.0f}s guarantees only {guaranteed} round(s) against the "
        f"{live_estimate:.0f}s p95 estimate, under the pre-registered dose floor "
        f"of {EXP2_DOSE_FLOOR}: the primary contrast would not be valid"
    )
    # B-once's boundary rule, evaluated in closed form on the LIVE constants
    # (step cadence only, the way the pre-registration states it): the last
    # m-boundary whose remaining budget still fits a round.
    boundaries = [
        (i, i * live_block_s) for i in range(1, int(live_budget // live_block_s) + 1)
    ]
    affordable = [(i, t) for i, t in boundaries if live_budget - t >= live_estimate]
    assert affordable, "no boundary can afford a round: B-once cannot exist"
    bonce_i, bonce_s = affordable[-1]
    first_past_half = next((t for _, t in boundaries if t >= 0.5 * live.T_s), None)
    assert first_past_half is None or live_budget - first_past_half < live_estimate, (
        "T/2 and affordability agree here, so the boundary rule change is "
        "unnecessary -- re-check the constants before keeping it"
    )
    report["exp2_admission"] = {
        "T_s": live.T_s,
        "rollout_budget_s": round(live_budget, 1),
        "snapshot_s": live.snapshot_cost_estimate_s,
        "fork_p50_s": EXP2_FORK_P50_S,
        "fork_p95_s": EXP2_FORK_P95_S,
        "wave_at_p95_s": round((EXP2_K - 1) * EXP2_FORK_P95_S, 1),
        "rollout_s": round(live.m * live.step_cost_estimate_s
                           + live.probe_cost_estimate_s, 1),
        "cleanup_s": round(EXP2_K * live.delete_cost_estimate_s, 1),
        "round_estimate_s": round(live_estimate, 1),
        "round_typical_s": round(live_typical, 1),
        "dose_floor": EXP2_DOSE_FLOOR,
        "rounds_that_fit_worst_case": guaranteed,
        "rounds_that_fit_typical": int(live_budget // live_typical),
        "bonce_boundary_index": bonce_i,
        "bonce_boundary_s": round(bonce_s, 1),
        "bonce_fraction_of_T": round(bonce_s / live.T_s, 4),
        "bonce_first_boundary_past_half_s": (
            round(first_past_half, 1) if first_past_half is not None else None
        ),
        "note": (
            "admission is p95-conservative on purpose; the measured dose is "
            "read from the round-boundary journal, never assumed. B-once "
            "converges at the last AFFORDABLE boundary -- the first boundary "
            "past T/2 cannot pay for a round at these constants"
        ),
    }

    hint_marker = HINT_TEMPLATE.split("{", 1)[0].strip()

    def leak_check(arm: str, world: FakeWorld, *keep: str) -> None:
        assert not world.games, f"arm {arm} leaked sandboxes: {list(world.games)}"
        leftover = set(world.snapshots) - {k for k in keep if k}
        assert not leftover, f"arm {arm} leaked snapshots: {sorted(leftover)}"

    def bake_checkpoint(world: FakeWorld, fp: FakeFarplane, template: str) -> str:
        """A deep checkpoint, baked the way Exp 1 bakes S2.

        Create a sandbox from the template, play it forward, snapshot it, throw
        the sandbox away. The resulting id is deliberately NOT the template's,
        so an arm seeded from it proves the seed id is honoured end to end.
        """
        sb = fp.create_from_snapshot(template, 600, "flebench-checkpoint-bake")
        FakeBridge(sb.base_url or "", world).execute(
            "place_entity('drill')\nconnect_entities(1)\n"
        )
        snap = fp.snapshot(sb)
        fp.delete_sandbox(sb)
        world.deleted_sandboxes.clear()
        return snap

    # ---- Arm A -----------------------------------------------------------
    world, fp, bridge_factory, template = fake_substrate()
    cfg_a = ArmConfig(arm="A", model="fake-model", task_key="iron_plate_throughput",
                      T_s=T_s, K=K, m=m, template_snap=template, dry=True,
                      terminal_reserve_s=0.5, probe_cost_estimate_s=0.1,
                      journal_dir=journal_dir, run_id="dry-A")
    journal_a = RunJournal(os.path.join(journal_dir, "dry-A.jsonl"), run_id="dry-A")
    llm_a = FakeLLM(journal=journal_a, log_full_requests=False)
    run_a = build_run(cfg_a, farplane=fp, bridge_factory=bridge_factory, llm=llm_a,
                      journal=journal_a)
    res_a = await execute_run(run_a)
    journal_a.close()

    assert res_a.steps >= 2 * m, f"arm A took only {res_a.steps} steps"
    assert res_a.endpoint_throughput is not None, "arm A produced no endpoint"
    run_a.timings.check_sums()
    a_summary = res_a.timings
    assert a_summary["attributed_s"]["probe"] > 0, "arm A never probed (v2.6 parity)"
    assert a_summary["attributed_s"]["llm_wait"] > 0, "arm A recorded no llm_wait"
    # A's conversation is one unbranched line: system + 2 messages per step.
    assert res_a.steps * 2 + 1 >= 3
    leak_check("A", world, template)
    a_probes = _count_journal(os.path.join(journal_dir, "dry-A.jsonl"), "probe")
    assert a_probes >= res_a.steps // m, "A missed parity probes"
    # v2.6: A owns exactly one sandbox and never touches the snapshot/fork
    # lane -- its probes are direct.
    assert res_a.sandboxes_created == 1, (
        f"arm A created {res_a.sandboxes_created} sandboxes; v2.6 allows exactly 1"
    )
    assert res_a.snapshots_created == 0, (
        f"arm A created {res_a.snapshots_created} snapshots; v2.6 allows none"
    )
    assert a_summary["attributed_s"]["infra_snapshot"] == 0.0, (
        "arm A spent wall clock in the snapshot lane"
    )
    report["A"] = {
        "steps": res_a.steps,
        "endpoint": res_a.endpoint_throughput,
        "timings": a_summary,
        "sandboxes_created": res_a.sandboxes_created,
        "snapshots_created": res_a.snapshots_created,
        "probes": a_probes,
        "curve_points": len(res_a.curve),
        "tokens": res_a.tokens,
        "incidents": res_a.incidents,
    }

    # ---- Arm B: R branch points, seeded from a CHECKPOINT (B-from-S) -----
    world_b, fp_b, bridge_b, template_b = fake_substrate()
    checkpoint_b = bake_checkpoint(world_b, fp_b, template_b)
    cfg_b = ArmConfig(arm="B", model="fake-model", task_key="iron_plate_throughput",
                      T_s=1e9, K=K, m=m, template_snap=checkpoint_b, dry=True,
                      terminal_reserve_s=0.0, probe_cost_estimate_s=0.1,
                      journal_dir=journal_dir, run_id="dry-B")
    journal_b = RunJournal(os.path.join(journal_dir, "dry-B.jsonl"), run_id="dry-B")
    llm_b = FakeLLM(journal=journal_b, log_full_requests=False)
    run_b = ForkBranchingRun(cfg_b, farplane=fp_b, bridge_factory=bridge_b,
                             llm=llm_b, journal=journal_b)

    res_b = _new_result(run_b)
    main_node = await run_b.provision_main("main")
    main = Trajectory(tid="main", node=main_node, conv=run_b.new_conversation())
    run_b.budget.start()
    run_b.timings.start()
    promotions: list[dict[str, Any]] = []
    for round_idx in range(1, rounds + 1):
        prefix_len = len(main.conv.messages)
        before_pending = main.conv.pending_feedback
        prefix_copy = copy.deepcopy(main.conv.messages)
        winner_node_before = main.node.id
        main = await run_b.branch_round(main, round_idx)
        # P4 promotion invariants.
        assert main.conv.messages[:prefix_len] == prefix_copy, (
            "promoted conversation does not extend the common prefix verbatim"
        )
        added = main.conv.messages[prefix_len:]
        assert len(added) == 2 * m, (
            f"round {round_idx}: expected {2 * m} promoted messages, got {len(added)}"
        )
        assert added[0]["role"] == "user" and added[1]["role"] == "assistant"
        # Hint rotation: the promoted branch carries its round's strategy hint
        # in its own first post-fork user turn (bench.llm.HINT_TEMPLATE).
        assert hint_marker in added[0]["content"], (
            f"round {round_idx}: promoted branch's first user turn carries no "
            f"{hint_marker} line"
        )
        promoted_hint = added[0]["content"].split(hint_marker, 1)[-1].strip()
        if before_pending:
            assert before_pending in added[0]["content"], (
                "pending feedback was not carried into the branch's first user turn"
            )
        assert main.conv.pending_feedback, "winner pending feedback lost"
        assert any("Objective Throughput Measurement" in e
                   for e in main.conv.pending_extras), (
            "winner probe result was not injected into the promoted conversation"
        )
        # Losers must not appear anywhere in the promoted history.
        promoted_text = "\n".join(msg["content"] for msg in added)
        candidates_in_history = promoted_text.count("# candidate ")
        assert candidates_in_history == m, (
            f"promoted history mixes branches: {candidates_in_history} candidate "
            f"markers for m={m} steps"
        )
        promotions.append(
            {
                "round": round_idx,
                "prefix_len": prefix_len,
                "added_messages": len(added),
                "winner_first_assistant": added[1]["content"][:48],
                "winner_hint": promoted_hint[:48],
                "winner_sandbox": main.node.id,
                "main_sandbox_before": winner_node_before,
                "promoted_step": main.step,
            }
        )
    await run_b.terminal_probe(main)
    res_b.active_s = run_b.budget.elapsed_s()
    run_b.timings.stop()
    await run_b.teardown(list(run_b.infra.live_sandboxes.values()))
    _finish(run_b, res_b, [main])
    journal_b.close()

    run_b.timings.check_sums()
    assert res_b.branch_points == rounds, (
        f"expected {rounds} branch points, got {res_b.branch_points}"
    )
    assert main.step == rounds * m, (
        f"expected {rounds * m} promoted steps, got {main.step}"
    )
    assert res_b.endpoint_throughput is not None, "arm B produced no endpoint"
    leak_check("B", world_b, template_b, checkpoint_b)

    b_journal = os.path.join(journal_dir, "dry-B.jsonl")
    archived = _count_journal(b_journal, "branch_archive")
    selections = _count_journal(b_journal, "branch_selection")
    assert selections == rounds, (
        f"expected {rounds} branch_selection records, got {selections}"
    )
    assert archived == rounds * (K - 1), (
        f"expected {rounds * (K - 1)} archived losers, got {archived}"
    )
    # Every branch of every round got its own fixed-window probe (P3/P5),
    # plus the terminal one.
    b_probes = _count_journal(b_journal, "probe")
    assert b_probes == rounds * K + 1, (
        f"expected {rounds * K + 1} probes, got {b_probes}"
    )
    # v2.6: the ONLY snapshot/fork traffic left is B's branching -- one
    # snapshot and K-1 forks per round, zero measurement forks.
    assert res_b.snapshots_created == rounds, (
        f"expected 1 branch snapshot per round ({rounds}), got "
        f"{res_b.snapshots_created}"
    )
    assert res_b.sandboxes_created == 1 + rounds * (K - 1), (
        f"expected main + {rounds * (K - 1)} branch forks, got "
        f"{res_b.sandboxes_created}"
    )
    # B never deletes the checkpoint it was re-seeded from.
    assert checkpoint_b in world_b.snapshots, "arm B deleted its own checkpoint"
    # Hint rotation at re-seed: every round assigns K DIVERGENT hints and no
    # seat is handed the same strategy two rounds in a row.
    assignments = [
        rec["hints"] for rec in _journal_records(b_journal, "hint_assignment")
    ]
    assert len(assignments) == rounds, (
        f"expected one hint_assignment record per round ({rounds}), got "
        f"{len(assignments)}"
    )
    for round_idx, seats in enumerate(assignments, start=1):
        assert len(seats) == K and all(seats.values()), (
            f"round {round_idx} left a seat without a hint: {seats}"
        )
        assert len(set(seats.values())) == K, (
            f"round {round_idx} hints are not divergent: {sorted(seats)}"
        )

    def by_seat(rec: dict[str, str]) -> dict[str, str]:
        return {seat.rsplit("b", 1)[-1]: hint for seat, hint in rec.items()}

    for prev, cur in zip(assignments, assignments[1:]):
        prev_seats, cur_seats = by_seat(prev), by_seat(cur)
        repeated = [s for s, h in cur_seats.items() if prev_seats.get(s) == h]
        assert not repeated, (
            "hint-to-seat assignment repeated in consecutive rounds for seats "
            f"{sorted(repeated)}"
        )
    # ALL K branches of a round -- the promoted one asserted above, the losers
    # archived -- carried their ASSIGNED hint in their own first post-fork user
    # turn, so the seat-to-strategy mapping is delivered, not just journaled.
    by_round = {
        rec["round"]: rec["hints"]
        for rec in _journal_records(b_journal, "hint_assignment")
    }
    for rec in _journal_records(b_journal, "branch_archive"):
        first = rec["messages"][0]
        assert first["role"] == "user" and hint_marker in first["content"], (
            f"archived branch {rec['branch']} has no hint in its first user turn"
        )
        seats = by_round[int(rec["branch"].split("b", 1)[0].lstrip("r"))]
        assert seats[rec["branch"]] in first["content"], (
            f"archived branch {rec['branch']} carried a hint it was not assigned"
        )
    report["B"] = {
        "seed_snapshot": checkpoint_b,
        "template_snapshot": template_b,
        "rounds": rounds,
        "branch_points": res_b.branch_points,
        "steps": main.step,
        "endpoint": res_b.endpoint_throughput,
        "timings": res_b.timings,
        "promotions": promotions,
        "hint_assignments": [
            {seat: hint[:40] for seat, hint in seats.items()}
            for seats in assignments
        ],
        "losers_archived": archived,
        "probes": b_probes,
        "sandboxes_created": res_b.sandboxes_created,
        "snapshots_created": res_b.snapshots_created,
        "tokens": res_b.tokens,
        "incidents": res_b.incidents,
    }

    # ---- Arm B under a HARD T: the late fork wave is REFUSED -------------
    # Same arithmetic as production, scaled to the fakes: a round costs
    # snapshot + (K-1) forks + m steps + a probe, and a round that does not fit
    # in what is left of T must not start -- the canonical line continues
    # instead and the endpoint is still measured at T.
    world_h, fp_h, bridge_h, template_h = fake_substrate()
    checkpoint_h = bake_checkpoint(world_h, fp_h, template_h)
    cfg_h = ArmConfig(
        arm="B", model="fake-model", task_key="iron_plate_throughput",
        T_s=hard_T_s, K=K, m=m, template_snap=checkpoint_h, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.02,
        snapshot_cost_estimate_s=0.05, fork_cost_estimate_s=0.05,
        step_cost_estimate_s=0.45, delete_cost_estimate_s=0.01, step_timeout_s=0.6,
        journal_dir=journal_dir, run_id="dry-Bhard",
    )
    journal_h = RunJournal(os.path.join(journal_dir, "dry-Bhard.jsonl"),
                           run_id="dry-Bhard")
    llm_h = FakeLLM(journal=journal_h, log_full_requests=False, max_concurrency=K)
    run_h = build_run(cfg_h, farplane=fp_h, bridge_factory=bridge_h, llm=llm_h,
                      journal=journal_h)
    res_h = await execute_run(run_h)
    journal_h.close()
    run_h.timings.check_sums()
    main_h = run_h.trajectories[0]
    h_journal = os.path.join(journal_dir, "dry-Bhard.jsonl")
    h_events = _journal_records(h_journal, "event")
    h_starts = [r for r in h_events if r.get("name") == "round_start"]
    h_stops = [r for r in h_events if r.get("name") == "convergence_stopped"]
    assert h_starts, (
        "hard-T B never started a round; the dry budget is mis-sized and the "
        "refusal assertion below would be vacuous"
    )
    for rec in h_starts:
        assert rec["remaining_s"] >= rec["round_estimate_s"], (
            f"round {rec['round']} started with {rec['remaining_s']}s left, "
            f"under its own {rec['round_estimate_s']}s estimate"
        )
    assert len(h_stops) == 1, (
        f"expected exactly one convergence_stopped record, got {len(h_stops)}"
    )
    assert h_stops[0]["remaining_s"] < h_stops[0]["round_estimate_s"], (
        "convergence stopped while a full round still fitted"
    )
    assert h_stops[0]["rounds_completed"] == len(h_starts), (
        f"{h_stops[0]['rounds_completed']} rounds counted vs {len(h_starts)} "
        "round_start records"
    )
    assert len(h_starts) == res_h.branch_points, (
        f"{len(h_starts)} rounds started but {res_h.branch_points} branch points"
    )
    # The canonical line kept compounding after the refusal, and T still ends
    # in a terminal probe rather than a truncated run.
    assert main_h.step > h_stops[0]["step"], (
        f"no canonical steps after the refusal: {main_h.step} <= "
        f"{h_stops[0]['step']}"
    )
    assert res_h.endpoint_throughput is not None, "hard-T B produced no endpoint"
    assert res_h.active_s <= cfg_h.T_s + cfg_h.step_timeout_s, (
        f"hard-T B ran {res_h.active_s:.2f}s, over T={cfg_h.T_s}s by more than "
        f"one {cfg_h.step_timeout_s}s step"
    )
    assert res_h.active_s >= 0.5 * cfg_h.T_s, (
        f"hard-T B stopped early at {res_h.active_s:.2f}s of T={cfg_h.T_s}s"
    )
    leak_check("B-hardT", world_h, template_h, checkpoint_h)
    report["B-hardT"] = {
        "T_s": cfg_h.T_s,
        "round_estimate_s": round(run_h.round_estimate_s(), 3),
        "rounds_started": len(h_starts),
        "rounds_at_start": [
            {"round": r["round"], "remaining_s": r["remaining_s"]} for r in h_starts
        ],
        "convergence_stopped": h_stops[0],
        "steps": main_h.step,
        "active_s": round(res_h.active_s, 3),
        "overrun_s": round(res_h.active_s - cfg_h.T_s, 3),
        "endpoint": res_h.endpoint_throughput,
        "timings": res_h.timings,
        "incidents": res_h.incidents,
    }

    # ---- Arm B with a SLOW fork wave: the wave stops at the deadline ------
    # Admission uses the fork p95; this substrate forks slower than that (the
    # measured max/p50 ratio is ~2.4x, so an over-p95 wave is a real case).
    # The round is admitted, the wave runs long, and the ABSOLUTE deadline
    # inside the wave stops it -- the round then converges at a shrunken K
    # instead of pushing the rollout and the terminal probe out of T.
    wave_K = 8
    world_w, fp_w, bridge_w, template_w = fake_substrate(fork_latency_mult=30.0)
    checkpoint_w = bake_checkpoint(world_w, fp_w, template_w)
    cfg_w = ArmConfig(
        arm="B", model="fake-model", task_key="iron_plate_throughput",
        T_s=wave_T_s, K=wave_K, m=m, template_snap=checkpoint_w, dry=True,
        terminal_reserve_s=0.15, probe_cost_estimate_s=0.03,
        snapshot_cost_estimate_s=0.02, fork_cost_estimate_s=0.2,
        step_cost_estimate_s=0.03, delete_cost_estimate_s=0.005,
        step_timeout_s=1.0, journal_dir=journal_dir, run_id="dry-Bwave",
    )
    journal_w = RunJournal(os.path.join(journal_dir, "dry-Bwave.jsonl"),
                           run_id="dry-Bwave")
    llm_w = FakeLLM(journal=journal_w, log_full_requests=False,
                    max_concurrency=wave_K)
    run_w = build_run(cfg_w, farplane=fp_w, bridge_factory=bridge_w, llm=llm_w,
                      journal=journal_w)
    # The admission estimate IS the p95 arithmetic, not the p50 one.
    expected_estimate = (
        cfg_w.snapshot_cost_estimate_s
        + (wave_K - 1) * cfg_w.fork_cost_estimate_s
        + m * cfg_w.step_cost_estimate_s + cfg_w.probe_cost_estimate_s
        + wave_K * cfg_w.delete_cost_estimate_s
    )
    assert abs(run_w.round_estimate_s() - expected_estimate) < 1e-9, (
        f"round estimate {run_w.round_estimate_s()} != snapshot + (K-1) forks + "
        f"rollout + cleanup = {expected_estimate}"
    )
    res_w = await execute_run(run_w)
    journal_w.close()
    run_w.timings.check_sums()
    main_w = run_w.trajectories[0]
    w_journal = os.path.join(journal_dir, "dry-Bwave.jsonl")
    w_events = _journal_records(w_journal, "event")
    waves = [r for r in w_events if r.get("name") == "fork_wave"]
    truncated = [r for r in waves if r["truncated"]]
    assert waves, "the slow-fork run never even started a wave"
    assert truncated, (
        "no wave hit its deadline; waves="
        f"{[(w['wanted'], w['materialized']) for w in waves]}"
    )
    for w in truncated:
        assert 0 <= w["materialized"] < w["wanted"], (
            f"a truncated wave reports {w['materialized']} of {w['wanted']} forks"
        )
        assert w["k_effective"] == w["materialized"] + 1, (
            "k_effective must count main plus the children that came up"
        )
        assert w["fork_estimate_s"] == cfg_w.fork_cost_estimate_s
    # Admission still never let a round start under its own estimate ...
    for rec in [r for r in w_events if r.get("name") == "round_start"]:
        assert rec["remaining_s"] >= rec["round_estimate_s"], (
            f"round {rec['round']} admitted with {rec['remaining_s']}s < "
            f"{rec['round_estimate_s']}s"
        )
    # ... and the shrunken round was really scored at the shrunken K (or the
    # wave staffed nobody, in which case convergence stopped instead).
    w_selections = _journal_records(w_journal, "branch_selection")
    aborted = [i for i in res_w.incidents if i["kind"] == "round_aborted_deadline"]
    assert w_selections or aborted, (
        "the truncated round neither converged at a shrunken K nor aborted"
    )
    for sel, wave in zip(w_selections, waves):
        assert sel["k_effective"] <= wave["k_effective"], (
            f"round {sel['round']} selected over {sel['k_effective']} branches but "
            f"only {wave['k_effective']} were up"
        )
        assert sel["k_effective"] < cfg_w.K or not wave["truncated"], (
            "a truncated wave still reported the full K at selection"
        )
    assert res_w.endpoint_throughput is not None, "slow-wave B produced no endpoint"
    assert res_w.active_s <= cfg_w.T_s + cfg_w.step_timeout_s, (
        f"slow-wave B ran {res_w.active_s:.2f}s, over T={cfg_w.T_s}s by more than "
        f"one {cfg_w.step_timeout_s}s step"
    )
    # The ONLY residue an arm may leave is a control-plane orphan it explicitly
    # owns; anything else is a sandbox it forgot.
    assert set(world_w.games) <= set(fp_w.orphans), (
        f"slow-wave B leaked non-orphan sandboxes: "
        f"{sorted(set(world_w.games) - set(fp_w.orphans))}"
    )
    w_sweep = fp_w.reaper(cfg_w.prefix, keep=[template_w, checkpoint_w])
    for orphan in fp_w.orphans:
        assert orphan in {r["id"] for r in w_sweep}, (
            f"the reaper did not sweep orphan {orphan}"
        )
    leak_check("B-slow-wave", world_w, template_w, checkpoint_w)
    report["B-slowWave"] = {
        "T_s": cfg_w.T_s,
        "K": cfg_w.K,
        "fork_estimate_s": cfg_w.fork_cost_estimate_s,
        "round_estimate_s": round(run_w.round_estimate_s(), 3),
        "waves": [
            {k: w[k] for k in ("round", "wanted", "materialized", "k_effective",
                               "truncated", "remaining_s")}
            for w in waves
        ],
        "selections_k_effective": [s["k_effective"] for s in w_selections],
        "rounds_aborted": len(aborted),
        "steps": main_w.step,
        "active_s": round(res_w.active_s, 3),
        "overrun_s": round(res_w.active_s - cfg_w.T_s, 3),
        "endpoint": res_w.endpoint_throughput,
        "timings": res_w.timings,
        "incidents": res_w.incidents,
        "orphans": len(res_w.orphan_forks),
        "orphans_swept": [r["id"] for r in w_sweep],
    }

    # ---- ONE fork slower than its own deadline: truncate + own the orphan --
    # The queue-bound case (Farplane queue cap 5m; contended soak p95 758s).
    # Fork #2 of the wave cannot finish inside any deadline it will be handed,
    # so the call must abandon it AT the deadline rather than after it lands,
    # truncate the wave, and leave the child OWNED: the source snapshot is
    # retained on purpose because it is the reaper's only handle on a fork
    # child (fork children carry control-plane names, not our prefix).
    fd_K = 4
    world_f, fp_f, bridge_f, template_f = fake_substrate(slow_fork_at=2)
    checkpoint_f = bake_checkpoint(world_f, fp_f, template_f)
    cfg_f = ArmConfig(
        arm="B", model="fake-model", task_key="iron_plate_throughput",
        T_s=1.6, K=fd_K, m=m, template_snap=checkpoint_f, dry=True,
        terminal_reserve_s=0.15, probe_cost_estimate_s=0.03,
        snapshot_cost_estimate_s=0.02, fork_cost_estimate_s=0.05,
        step_cost_estimate_s=0.03, delete_cost_estimate_s=0.005,
        step_timeout_s=1.0, journal_dir=journal_dir, run_id="dry-Bforkdl",
    )
    journal_f = RunJournal(os.path.join(journal_dir, "dry-Bforkdl.jsonl"),
                           run_id="dry-Bforkdl")
    llm_f = FakeLLM(journal=journal_f, log_full_requests=False, max_concurrency=fd_K)
    run_f = build_run(cfg_f, farplane=fp_f, bridge_factory=bridge_f, llm=llm_f,
                      journal=journal_f)
    res_f = await execute_run(run_f)
    journal_f.close()
    run_f.timings.check_sums()
    f_journal = os.path.join(journal_dir, "dry-Bforkdl.jsonl")
    f_recs = _all_journal_records(f_journal)
    f_orphans = [r for r in f_recs if r.get("kind") == "fork_orphan"]
    f_waves = [r for r in f_recs
               if r.get("kind") == "event" and r.get("name") == "fork_wave"]
    f_retained = [r for r in f_recs if r.get("kind") == "event"
                  and r.get("name") == "branch_snapshot_retained"]
    f_forks = [r for r in f_recs
               if r.get("kind") == "infra_op" and r.get("op") == "fork"]
    slow_want_s = world_f.latency * fp_f.slow_fork_mult
    # Jitter-proof: the fake designates exactly ONE fork that cannot meet its
    # deadline, so the count is configuration-driven -- and the properties that
    # actually matter (every orphan is journaled, and every orphan is swept)
    # hold for any count.
    assert f_orphans, "no orphaned fork was journaled"
    assert fp_f.orphans, "the fake control plane never landed the orphan child"
    assert len(f_orphans) == len(fp_f.orphans), (
        f"{len(fp_f.orphans)} child(ren) landed after their deadline but "
        f"{len(f_orphans)} were journaled: an orphan is unowned"
    )
    assert len(f_orphans) == 1, (
        f"the fake designates one un-meetable fork, so exactly one orphan is "
        f"expected; journaled {len(f_orphans)}"
    )
    # Stopped AT the deadline, NOT after the fork landed.
    timed_out = [r for r in f_forks if r.get("outcome") == "error"]
    assert timed_out, "no fork reported a deadline failure"
    assert all(r["duration_s"] < slow_want_s for r in timed_out), (
        f"a fork waited {max(r['duration_s'] for r in timed_out):.2f}s, i.e. past "
        f"its deadline and out to the {slow_want_s:.2f}s the fork actually needed"
    )
    assert all(r["duration_s"] <= r["deadline_s"] + 0.25 for r in timed_out), (
        "a timed-out fork overran the deadline it was given"
    )
    assert f_waves and f_waves[0]["truncated"], "the wave did not truncate"
    assert f_waves[0]["materialized"] < f_waves[0]["wanted"], (
        f"wave reports {f_waves[0]['materialized']} of {f_waves[0]['wanted']}"
    )
    assert f_waves[0]["orphans"] == 1
    # The source snapshot is retained ON PURPOSE, by the round and by teardown.
    assert f_retained, "the orphan's source snapshot was not retained"
    orphan_src = f_orphans[0]["source_snapshot"]
    assert orphan_src in world_f.snapshots, (
        "the orphan's source snapshot was deleted; the child is now unowned"
    )
    assert res_f.orphan_forks and res_f.orphan_forks[0]["source_snapshot"] == orphan_src
    assert res_f.endpoint_throughput is not None, "fork-deadline B has no endpoint"
    assert res_f.active_s <= cfg_f.T_s + cfg_f.step_timeout_s, (
        f"fork-deadline B ran {res_f.active_s:.2f}s, over T={cfg_f.T_s}s by more "
        f"than one {cfg_f.step_timeout_s}s step"
    )
    # The reaper -- not the arm -- owns every orphan and its source snapshot.
    residual_before = sorted(world_f.games)
    assert set(fp_f.orphans) <= set(residual_before), (
        "an orphan child is not among the residual sandboxes"
    )
    assert set(residual_before) <= set(fp_f.orphans), (
        f"the arm leaked non-orphan sandboxes: "
        f"{sorted(set(residual_before) - set(fp_f.orphans))}"
    )
    f_sweep = fp_f.reaper(cfg_f.prefix, keep=[template_f, checkpoint_f])
    swept_ids = {r["id"] for r in f_sweep}
    missed = [o for o in fp_f.orphans if o not in swept_ids]
    assert not missed, f"the reaper did not sweep orphan(s) {missed}: {f_sweep}"
    assert orphan_src in swept_ids, (
        "the retained source snapshot was not swept with its child"
    )
    assert checkpoint_f not in swept_ids and template_f not in swept_ids, (
        "the sweep ate substrate that was on the keep list"
    )
    leak_check("B-fork-deadline", world_f, template_f, checkpoint_f)
    report["B-forkDeadline"] = {
        "T_s": cfg_f.T_s,
        "K": cfg_f.K,
        "slow_fork_index": fp_f.slow_fork_at,
        "slow_fork_needed_s": round(slow_want_s, 3),
        "abandoned_after_s": [round(r["duration_s"], 3) for r in timed_out],
        "deadlines_given_s": [round(r["deadline_s"], 3) for r in timed_out],
        "wave": {k: f_waves[0][k] for k in ("wanted", "materialized", "k_effective",
                                            "truncated", "orphans")},
        "orphan": f_orphans[0],
        "snapshot_retained": orphan_src,
        "residual_before_sweep": residual_before,
        "reaper_swept": sorted(swept_ids),
        "active_s": round(res_f.active_s, 3),
        "overrun_s": round(res_f.active_s - cfg_f.T_s, 3),
        "endpoint": res_f.endpoint_throughput,
        "timings": res_f.timings,
        "incidents": res_f.incidents,
    }

    # ---- A step cancelled MID-EXECUTE: joined before the endpoint ---------
    # Deterministic by construction: /execute outlives the remaining budget, so
    # the step deadline fires while the program is still running in its worker
    # thread. That used to be swallowed as `execute_failed: CancelledError`
    # while the thread kept mutating the sandbox, which put the terminal probe
    # in a race with a live build.
    class SlowExecuteBridge(FakeBridge):
        """A bridge whose /execute cannot finish inside the step deadline."""

        exec_s = 0.4

        def execute(self, code: str) -> dict:
            if code.strip() != BASELINE_CODE:
                time.sleep(self.exec_s)
            return super().execute(code)

    world_d, fp_d, _bridge_d, template_d = fake_substrate()
    cfg_d = ArmConfig(
        arm="A", model="fake-model", task_key="iron_plate_throughput",
        T_s=0.9, K=1, m=2, template_snap=template_d, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.02,
        step_timeout_s=5.0, journal_dir=journal_dir, run_id="dry-Adeadline",
    )
    journal_d = RunJournal(os.path.join(journal_dir, "dry-Adeadline.jsonl"),
                           run_id="dry-Adeadline")
    llm_d = FakeLLM(journal=journal_d, log_full_requests=False)
    run_d = build_run(cfg_d, farplane=fp_d,
                      bridge_factory=lambda url: SlowExecuteBridge(url, world_d),
                      llm=llm_d, journal=journal_d)
    res_d = await execute_run(run_d)
    journal_d.close()
    run_d.timings.check_sums()
    d_journal = os.path.join(journal_dir, "dry-Adeadline.jsonl")
    d_recs = _all_journal_records(d_journal)
    d_cancelled = [r for r in d_recs if r.get("kind") == "incident"
                   and r.get("incident_kind") == "step_deadline_cancelled"]
    d_executes = [r for r in d_recs
                  if r.get("kind") == "infra_op" and r.get("op") == "execute"]
    d_settled = [r for r in d_executes if r.get("outcome") == "settled_after_deadline"]
    d_terminal = [r for r in d_recs if r.get("kind") == "probe"
                  and r.get("probe_kind") == "terminal"]
    assert d_cancelled, (
        "the slow-execute arm never hit a step deadline; the cancellation "
        "assertions below would be vacuous"
    )
    assert d_settled, (
        f"{len(d_cancelled)} step(s) were cancelled but the drain joined none of "
        "them -- the endpoint is not ordered after the substrate"
    )
    assert not [r for r in d_recs if r.get("incident_kind") == "execute_failed"
                and "CancelledError" in str(r.get("detail", ""))], (
        "a deadline cancellation was reported as a substrate error again"
    )
    assert d_terminal, "the slow-execute arm produced no terminal probe"
    assert max(e["seq"] for e in d_executes) < d_terminal[0]["seq"], (
        "an execute concluded AFTER the terminal probe: the endpoint measured a "
        "sandbox that was still being mutated"
    )
    assert not [r for r in d_executes if r.get("outcome") == "abandoned"], (
        "a substrate call outlived the bounded join"
    )
    assert res_d.endpoint_throughput is not None, "slow-execute arm has no endpoint"
    assert res_d.active_s <= cfg_d.T_s + SlowExecuteBridge.exec_s, (
        f"slow-execute arm ran {res_d.active_s:.2f}s, over T={cfg_d.T_s}s by more "
        f"than the one {SlowExecuteBridge.exec_s}s execute already in flight"
    )
    leak_check("A-deadline", world_d, template_d)
    report["A-deadline"] = {
        "T_s": cfg_d.T_s,
        "execute_s": SlowExecuteBridge.exec_s,
        "steps": res_d.steps,
        "deadline_cancellations": len(d_cancelled),
        "joined_by_drain": len(d_settled),
        "last_execute_seq": max(e["seq"] for e in d_executes),
        "terminal_probe_seq": d_terminal[0]["seq"],
        "active_s": round(res_d.active_s, 3),
        "overrun_s": round(res_d.active_s - cfg_d.T_s, 3),
        "endpoint": res_d.endpoint_throughput,
        "timings": res_d.timings,
        "incidents": res_d.incidents,
    }

    # ---- Arm B-once: ONE convergence, at the LAST affordable boundary -----
    world_o, fp_o, bridge_o, template_o = fake_substrate()
    checkpoint_o = bake_checkpoint(world_o, fp_o, template_o)
    cfg_o = ArmConfig(
        arm="Bonce", model="fake-model", task_key="iron_plate_throughput",
        T_s=bonce_T_s, K=K, m=m, template_snap=checkpoint_o, dry=True,
        terminal_reserve_s=0.1,
        # The boundary rule PREDICTS the next block from these estimates, so
        # here they are the fakes' real costs (a step is ~0.05s of LLM + 0.01s
        # of exec; a probe ~0.02s) -- a wrong predictor overshoots the last
        # affordable boundary, which is precisely the bug this guards. Scaled
        # so that boundary lands mid-run, as the live constants put it at ~47%.
        probe_cost_estimate_s=0.03, step_cost_estimate_s=0.07,
        snapshot_cost_estimate_s=0.3, fork_cost_estimate_s=0.3,
        delete_cost_estimate_s=0.015, step_timeout_s=1.0,
        journal_dir=journal_dir, run_id="dry-Bonce",
    )
    journal_o = RunJournal(os.path.join(journal_dir, "dry-Bonce.jsonl"),
                           run_id="dry-Bonce")
    llm_o = FakeLLM(journal=journal_o, log_full_requests=False, max_concurrency=K)
    run_o = build_run(cfg_o, farplane=fp_o, bridge_factory=bridge_o, llm=llm_o,
                      journal=journal_o)
    res_o = await execute_run(run_o)
    journal_o.close()
    run_o.timings.check_sums()
    main_o = run_o.trajectories[0]
    o_journal = os.path.join(journal_dir, "dry-Bonce.jsonl")
    o_conv = [
        r for r in _journal_records(o_journal, "event")
        if r.get("name") == "bonce_convergence"
    ]
    assert len(o_conv) == 1, (
        f"B-once must converge exactly once, journaled {len(o_conv)} times"
    )
    assert res_o.branch_points == 1, (
        f"B-once branch_points={res_o.branch_points}, must be 1"
    )
    assert _count_journal(o_journal, "branch_selection") == 1, (
        "B-once must select exactly once"
    )
    midpoint_o = 0.5 * cfg_o.T_s
    boundary_s = o_conv[0]["boundary_s"]
    # The contract is affordability, not a literal T/2: converge at the LAST
    # m-boundary whose remaining budget still fits a whole round.
    o_chosen = [r for r in _journal_records(o_journal, "event")
                if r.get("name") == "chosen_boundary"]
    assert len(o_chosen) == 1, (
        f"expected one chosen_boundary record, got {len(o_chosen)}"
    )
    chosen = o_chosen[0]
    o_block_s = cfg_o.m * cfg_o.step_cost_estimate_s + cfg_o.probe_cost_estimate_s
    assert chosen["remaining_s"] >= chosen["estimate_s"], (
        f"B-once converged at a boundary it could not afford: "
        f"{chosen['remaining_s']}s left < {chosen['estimate_s']}s estimate"
    )
    assert chosen["remaining_s"] < chosen["estimate_s"] + o_block_s, (
        f"B-once converged early: {chosen['remaining_s']}s left still covers "
        f"another {o_block_s}s block plus a {chosen['estimate_s']}s round, so "
        "this was not the LAST affordable boundary"
    )
    assert abs(chosen["elapsed_s"] - boundary_s) < 1e-6
    # ... and it still lands mid-run, so the curve point keeps its meaning.
    assert 0.2 * cfg_o.T_s <= boundary_s <= 0.8 * cfg_o.T_s, (
        f"B-once converged at {boundary_s}s of T={cfg_o.T_s}s "
        f"({chosen['fraction_of_T']:.0%}), not mid-run"
    )
    # Exactly ONE fork wave of infra traffic: main + (K-1) forks, one snapshot.
    assert res_o.snapshots_created == 1, (
        f"B-once took {res_o.snapshots_created} snapshots; exactly 1 wave allowed"
    )
    assert res_o.sandboxes_created == 1 + (K - 1), (
        f"B-once created {res_o.sandboxes_created} sandboxes; expected main + "
        f"{K - 1} forks"
    )
    # Hint assignment: one rotation, K divergent seats, delivered per branch.
    o_hints = [rec["hints"] for rec in _journal_records(o_journal, "hint_assignment")]
    assert len(o_hints) == 1 and len(o_hints[0]) == K, (
        f"B-once hint assignment is {o_hints}, expected one record of {K} seats"
    )
    assert len(set(o_hints[0].values())) == K, (
        f"B-once round hints are not divergent: {sorted(o_hints[0].values())}"
    )
    o_archived = _journal_records(o_journal, "branch_archive")
    assert len(o_archived) == K - 1, (
        f"B-once archived {len(o_archived)} losers, expected {K - 1}"
    )
    for rec in o_archived:
        first = rec["messages"][0]
        assert first["role"] == "user" and o_hints[0][rec["branch"]] in first["content"], (
            f"B-once loser {rec['branch']} did not carry its assigned hint"
        )
    # Promotion invariants on the line that actually produced the endpoint: the
    # winner's probe was injected, its hint is in the transcript, and NO loser
    # turn leaked in (one candidate marker and one assistant turn per step).
    o_messages = main_o.conv.messages
    o_text = "\n".join(msg["content"] for msg in o_messages)
    assert hint_marker in o_text, "B-once promoted line carries no strategy hint"
    assert "Objective Throughput Measurement" in o_text, (
        "B-once never injected a probe result into the promoted conversation"
    )
    o_assistants = sum(1 for msg in o_messages if msg["role"] == "assistant")
    assert o_assistants == main_o.step == o_text.count("# candidate "), (
        f"B-once transcript mixes branches: {o_assistants} assistant turns / "
        f"{o_text.count('# candidate ')} candidates for {main_o.step} steps"
    )
    # The winner continued to T rather than ending at the convergence.
    assert main_o.step > o_conv[0]["step"], (
        f"B-once winner did not continue past the convergence "
        f"({main_o.step} <= {o_conv[0]['step']})"
    )
    assert res_o.endpoint_throughput is not None, "B-once produced no endpoint"
    assert res_o.active_s <= cfg_o.T_s + cfg_o.step_timeout_s, (
        f"B-once ran {res_o.active_s:.2f}s, over T={cfg_o.T_s}s by more than one step"
    )
    assert checkpoint_o in world_o.snapshots, "B-once deleted its own checkpoint"
    leak_check("Bonce", world_o, template_o, checkpoint_o)
    report["Bonce"] = {
        "seed_snapshot": checkpoint_o,
        "T_s": cfg_o.T_s,
        "midpoint_s": midpoint_o,
        "convergence": o_conv[0],
        "steps_total": main_o.step,
        "steps_before_convergence": o_conv[0]["boundary_step"],
        "branch_points": res_o.branch_points,
        "hint_assignment": {seat: h[:40] for seat, h in o_hints[0].items()},
        "losers_archived": len(o_archived),
        "sandboxes_created": res_o.sandboxes_created,
        "snapshots_created": res_o.snapshots_created,
        "endpoint": res_o.endpoint_throughput,
        "active_s": round(res_o.active_s, 3),
        "timings": res_o.timings,
        "incidents": res_o.incidents,
    }

    # ---- A×K and C: same primitives, exercised end to end ---------------
    world_k, fp_k, bridge_k, template_k = fake_substrate()
    cfg_k = ArmConfig(arm="AxK", model="fake-model", task_key="iron_plate_throughput",
                      T_s=T_s / 2, K=K, m=m, template_snap=template_k, dry=True,
                      terminal_reserve_s=0.5, probe_cost_estimate_s=0.1,
                      journal_dir=journal_dir, run_id="dry-AxK")
    journal_k = RunJournal(os.path.join(journal_dir, "dry-AxK.jsonl"), run_id="dry-AxK")
    llm_k = FakeLLM(journal=journal_k, log_full_requests=False, max_concurrency=K)
    run_k = build_run(cfg_k, farplane=fp_k, bridge_factory=bridge_k, llm=llm_k,
                      journal=journal_k)
    res_k = await execute_run(run_k)
    journal_k.close()
    run_k.timings.check_sums()
    assert res_k.endpoint_source.startswith("AxK"), (
        f"A×K endpoint must come from a trajectory, got {res_k.endpoint_source!r}"
    )
    assert res_k.sandboxes_created == K, (
        f"A×K must provision exactly K={K} sandboxes and fork nothing, "
        f"got {res_k.sandboxes_created}"
    )
    assert res_k.snapshots_created == 0, "A×K created snapshots (v2.6 allows none)"
    leak_check("AxK", world_k, template_k)
    report["AxK"] = {
        "steps": res_k.steps,
        "endpoint": res_k.endpoint_throughput,
        "endpoint_source": res_k.endpoint_source,
        "timings": res_k.timings,
        "sandboxes_created": res_k.sandboxes_created,
        "incidents": res_k.incidents,
    }

    # ---- A×K-from-S: the SAME arm, seeded from a checkpoint ---------------
    # Exp 2's primary control. A×K-from-S is A×K with template_snap=<checkpoint>
    # and nothing else, so this asserts exactly that: the id reaches every
    # seat's create call and the checkpoint outlives the run.
    world_s, fp_s, bridge_s, template_s = fake_substrate()
    checkpoint_s = bake_checkpoint(world_s, fp_s, template_s)
    assert checkpoint_s != template_s, "checkpoint id must differ from the template"
    cfg_s = ArmConfig(arm="AxK", model="fake-model", task_key="iron_plate_throughput",
                      T_s=T_s / 2, K=K, m=m, template_snap=checkpoint_s, dry=True,
                      terminal_reserve_s=0.5, probe_cost_estimate_s=0.1,
                      journal_dir=journal_dir, run_id="dry-AxKS")
    journal_s = RunJournal(os.path.join(journal_dir, "dry-AxKS.jsonl"),
                           run_id="dry-AxKS")
    llm_s = FakeLLM(journal=journal_s, log_full_requests=False, max_concurrency=K)
    run_s = build_run(cfg_s, farplane=fp_s, bridge_factory=bridge_s, llm=llm_s,
                      journal=journal_s)
    res_s = await execute_run(run_s)
    journal_s.close()
    run_s.timings.check_sums()
    s_journal = os.path.join(journal_dir, "dry-AxKS.jsonl")
    seeds = [
        rec for rec in _journal_records(s_journal, "infra_op")
        if rec.get("op") == "create_from_snapshot"
    ]
    assert len(seeds) == K, (
        f"A×K-from-S must create exactly K={K} seats from the checkpoint, "
        f"got {len(seeds)}"
    )
    assert all(rec.get("source") == checkpoint_s for rec in seeds), (
        "a seat was created from something other than the checkpoint: "
        f"{sorted({rec.get('source') for rec in seeds})}"
    )
    assert res_s.endpoint_source.startswith("AxK"), (
        f"A×K-from-S endpoint must come from a trajectory, got "
        f"{res_s.endpoint_source!r}"
    )
    assert res_s.snapshots_created == 0, (
        "A×K-from-S created snapshots (v2.6 allows none)"
    )
    assert checkpoint_s in world_s.snapshots, (
        "A×K-from-S deleted the checkpoint it was seeded from"
    )
    leak_check("AxK-from-S", world_s, template_s, checkpoint_s)
    report["AxK-from-S"] = {
        "seed_snapshot": checkpoint_s,
        "template_snapshot": template_s,
        "seats_created_from_seed": len(seeds),
        "seat_seed_ids": sorted({str(rec.get("source")) for rec in seeds}),
        "steps": res_s.steps,
        "endpoint": res_s.endpoint_throughput,
        "endpoint_source": res_s.endpoint_source,
        "timings": res_s.timings,
        "sandboxes_created": res_s.sandboxes_created,
        "incidents": res_s.incidents,
    }

    # ---- Exp 3: the LIVE phase arithmetic, in closed form -----------------
    # Leg 2 must be admissible BY CONSTRUCTION at the pre-registered numbers:
    # T_total = 2P + measured overhead, leg 1 spends P, the overhead is handed
    # back, so leg 2 starts with P - reserve. If the reserve ever grew past
    # (1 - EXP3_PHASE2_ADMISSION) x P, every live Hybrid cell would report a
    # clipped leg 2 -- a launch blocker, caught here instead of at hour four.
    from bench.run_tier1 import (
        EXP3_CREATE_DEADLINE_S,
        EXP3_PROVISION_STAGGER_S,
        EXP3_T_S,
        EXP3_TTL_MARGIN_S,
        exp3_ttl_s,
    )

    # ---- PRE-FLIGHT LEASE GUARD: a short lease dies at construction --------
    # Exp-3 round 1's actual failure: seats leased for 7200s inside a ~8700s
    # round hibernated before their terminal probes and both surviving cells came
    # back PARTIAL. The guard must fire BEFORE any sandbox exists, on every
    # arm/block path, and it must name the numbers -- a future preset change is
    # exactly how this recurs.
    world_g, fp_g, bridge_g, template_g = fake_substrate()
    # Baseline: fake_substrate() bakes its own template, so "zero substrate
    # calls" means zero MORE than the world already had.
    ops_before_guard = list(fp_g.ops)
    sandboxes_before_guard = set(world_g.games)
    guard_slack = LEASE_GUARD_SLACK_S
    # An Exp-2-shaped horizon with a lease that only just covers T: the exact
    # shape of round 1's mistake, one step less obvious.
    guard_T_s = EXP2_T_S
    short_lease = int(guard_T_s)
    try:
        ArmConfig(arm="AxK", model="codex/gpt-5.6-sol",
                  task_key="iron_plate_throughput", T_s=guard_T_s, K=EXP3_K,
                  m=EXP2_DEFAULT_M, template_snap=template_g,
                  ttl_s=short_lease, run_id="guard-short-lease")
    except LeaseTooShort as exc:
        guard_msg = str(exc)
    else:  # pragma: no cover - the guard must exist
        raise AssertionError(
            f"a {short_lease}s lease was accepted for a {guard_T_s:.0f}s run"
        )
    for token in ("ttl_s", f"{guard_T_s:.0f}", f"{guard_slack:.0f}", "hibernate"):
        assert token in guard_msg, (
            f"the lease error does not name {token!r}: {guard_msg}"
        )
    # ZERO substrate calls: the config never reached a create.
    assert fp_g.ops == ops_before_guard, (
        "the rejected config still touched the substrate: "
        f"{fp_g.ops[len(ops_before_guard):]}"
    )
    assert not fp_g.create_calls and not fp_g.fork_calls, (
        "the rejected config created or forked a sandbox"
    )
    assert set(world_g.games) == sandboxes_before_guard, (
        f"the rejected config leaked sandboxes: "
        f"{sorted(set(world_g.games) - sandboxes_before_guard)}"
    )
    ok_lease = ArmConfig(
        arm="AxK", model="codex/gpt-5.6-sol", task_key="iron_plate_throughput",
        T_s=guard_T_s, K=EXP3_K, m=EXP2_DEFAULT_M, template_snap=template_g,
        ttl_s=int(guard_T_s + 90.0 + guard_slack), run_id="guard-ok-lease",
    )
    assert ok_lease.ttl_s >= ok_lease.T_s + ok_lease.terminal_reserve_s + guard_slack
    # A dry cell is exempt (T is clamped to seconds; the slack term is
    # meaningless there) -- which is why this section asserts the guard against
    # LIVE-shaped configs.
    ArmConfig(arm="AxK", model="fake-model", task_key="iron_plate_throughput",
              T_s=guard_T_s, K=2, m=2, template_snap=template_g, dry=True,
              ttl_s=1, run_id="guard-dry-exempt")
    report["lease_guard"] = {
        "slack_s": guard_slack,
        "rejected": {"T_s": guard_T_s, "ttl_s": short_lease,
                     "error": guard_msg},
        "accepted_ttl_s": ok_lease.ttl_s,
        "substrate_calls_on_rejection": len(fp_g.ops) - len(ops_before_guard),
        "substrate_calls_before": len(ops_before_guard),
        "dry_cells_exempt": True,
    }

    # ---- PROVIDER TRIPWIRE: what counts, and what is just noise ------------
    # k3 answers 200-with-no-content ~11% of the time and a retry fixes it. That
    # class must NEVER trip the wire, or the relaunch aborts itself on healthy
    # noise; a provider that fails AFTER classification and after every retry
    # must trip it, or the block burns hours on a dead quota (the codex round
    # spent 2.6h on ~17k straight 429s).
    from bench.llm import (
        PROVIDER_DEAD_CONSECUTIVE,
        PROVIDER_DEAD_WINDOW_S,
        RetryPolicy,
        provider_health,
        reset_provider_health,
    )

    reset_provider_health("fake")
    noise_llm = FakeLLM(log_full_requests=False, max_concurrency=1, latency=0.0,
                        empty_every=1,
                        retry=RetryPolicy(attempts=2, base_s=0.0, jitter=0.0))
    noise_calls = PROVIDER_DEAD_CONSECUTIVE + 5
    for _ in range(noise_calls):
        got = await noise_llm.sample_detailed([{"role": "user", "content": "hi"}], n=1)
        assert got[0].code, "the retry after an empty-200 did not produce a program"
    noise_health = provider_health("fake").snapshot()
    assert noise_llm.n_injected_empty >= noise_calls, (
        f"only {noise_llm.n_injected_empty} of {noise_calls} calls saw an empty-200"
    )
    assert noise_health["retry_noise"] >= noise_calls, (
        f"empty-200s were not recorded as retry noise: {noise_health}"
    )
    assert noise_health["failures"] == 0 and noise_health["consecutive_failures"] == 0, (
        f"an empty-200 that a retry absorbed counted toward the tripwire: "
        f"{noise_health}"
    )
    assert noise_health["successes"] == noise_calls

    # Now the real thing: terminal failures, and the wire trips EXACTLY at the
    # budget -- not before (a live block must survive a bad minute).
    reset_provider_health("fake")
    dead_llm = FakeLLM(log_full_requests=False, max_concurrency=1, latency=0.0,
                       fail_after=1,
                       retry=RetryPolicy(attempts=1, base_s=0.0, jitter=0.0))
    fired_at = 0
    for i in range(1, PROVIDER_DEAD_CONSECUTIVE + 3):
        try:
            await dead_llm.sample_detailed([{"role": "user", "content": "hi"}], n=1)
        except ProviderDead as exc:
            fired_at = i
            dead_msg = str(exc)
            dead_trigger = exc.trigger
            break
    assert fired_at, "the tripwire never fired on a dead provider"
    # One success, then N terminal failures: the wire fires on the call that
    # completes the budget, so the failure count is the budget exactly.
    assert fired_at == PROVIDER_DEAD_CONSECUTIVE + 1, (
        f"fired at call {fired_at}, expected the "
        f"{PROVIDER_DEAD_CONSECUTIVE}th terminal failure"
    )
    assert dead_trigger == "consecutive_failures"
    for token in ("fake", str(PROVIDER_DEAD_CONSECUTIVE), "429"):
        assert token in dead_msg, f"the cause does not name {token!r}: {dead_msg}"
    reset_provider_health("fake")
    report["provider_tripwire"] = {
        "consecutive_limit": PROVIDER_DEAD_CONSECUTIVE,
        "silence_limit_s": PROVIDER_DEAD_WINDOW_S,
        "noise": {
            "calls": noise_calls,
            "empty_200s": noise_llm.n_injected_empty,
            "retry_noise": noise_health["retry_noise"],
            "counted_failures": noise_health["failures"],
            "successes": noise_health["successes"],
        },
        "death": {"fired_at_call": fired_at, "trigger": dead_trigger,
                  "cause": dead_msg},
    }

    live3 = ArmConfig(arm="Hybrid", model="codex/gpt-5.6-sol",
                      task_key="iron_plate_throughput", T_s=EXP3_T_S, K=EXP3_K,
                      m=EXP2_DEFAULT_M, template_snap="s2b",
                      ttl_s=exp3_ttl_s(EXP3_T_S),
                      create_deadline_s=EXP3_CREATE_DEADLINE_S,
                      provision_stagger_s=EXP3_PROVISION_STAGGER_S,
                      run_id="exp3-phase-arithmetic")
    # The live Exp-3 preset clears the guard it just proved exists.
    assert live3.ttl_s >= (live3.T_s + live3.provision_stagger_s
                           + live3.terminal_reserve_s + guard_slack), (
        f"the Exp-3 preset would be rejected by its own lease guard: "
        f"ttl={live3.ttl_s}s"
    )
    assert live3.ttl_s == int(EXP3_T_S + EXP3_TTL_MARGIN_S)
    assert live3.leg_s == EXP3_P_S, (
        f"Exp 3's leg is {live3.leg_s}s, not the pre-registered P={EXP3_P_S}s"
    )
    assert live3.T_s == 2 * live3.leg_s, "T_total must be exactly 2P"
    assert tuple(live3.hints) == PERSONAS, (
        "an Exp-3 arm was configured with Exp 2's strategy hints"
    )
    live3_leg2_s = live3.T_s - live3.terminal_reserve_s - live3.leg_s
    live3_required_s = EXP3_PHASE2_ADMISSION * live3.leg_s
    assert live3_leg2_s >= live3_required_s, (
        f"leg 2 would start with {live3_leg2_s:.0f}s < the "
        f"{live3_required_s:.0f}s admission floor: every Hybrid cell would be "
        f"clipped at T={live3.T_s:.0f}s / P={live3.leg_s:.0f}s / reserve "
        f"{live3.terminal_reserve_s:.0f}s"
    )
    # The convergence itself, at the measured constants: one snapshot + the
    # serial (K-1) fork train at p95 + K deletes. It is charged to T and handed
    # back to the budget, so it is a COST, never a walk-length cut.
    live3_convergence_s = (
        live3.snapshot_cost_estimate_s
        + (EXP3_K - 1) * live3.fork_cost_estimate_s
        + EXP3_K * live3.delete_cost_estimate_s
    )
    assert live3_convergence_s < live3.leg_s, (
        "the convergence costs more than a leg; the design's phase split cannot "
        "hold"
    )
    report["exp3_arithmetic"] = {
        "P_s": live3.leg_s,
        "T_total_s": live3.T_s,
        "K": EXP3_K,
        "m": live3.m,
        "terminal_reserve_s": live3.terminal_reserve_s,
        "leg2_start_s": round(live3_leg2_s, 1),
        "leg2_required_s": round(live3_required_s, 1),
        "admission_fraction": EXP3_PHASE2_ADMISSION,
        "convergence_estimate_s": round(live3_convergence_s, 1),
        "width_floor": EXP3_WIDTH_FLOOR,
        "personas": len(PERSONAS),
        "note": (
            "T_total = 2P + MEASURED selection overhead: leg 1 spends P, the "
            "overhead is added back to the budget, so leg 2 starts with "
            "P - reserve and the 0.9P admission gate passes by construction"
        ),
    }

    def persona_turns(conv: Conversation) -> list[tuple[int, str]]:
        """(message index, persona text) for every hint line in a transcript."""
        out: list[tuple[int, str]] = []
        for idx, msg in enumerate(conv.messages):
            if msg.get("role") != "user":
                continue
            count = msg["content"].count(hint_marker)
            assert count <= 1, (
                f"message {idx} carries {count} persona lines; a seat must see "
                "its persona exactly once per phase"
            )
            if count:
                out.append((idx, msg["content"].split(hint_marker, 1)[-1].strip()))
        return out

    # ---- Exp 3 Hybrid: two legs, ONE convergence, 8 endpoints -------------
    h_K = EXP3_K
    world_y, fp_y, bridge_y, template_y = fake_substrate()
    checkpoint_y = bake_checkpoint(world_y, fp_y, template_y)
    cfg_y = ArmConfig(
        arm="Hybrid", model="fake-model", task_key="iron_plate_throughput",
        T_s=6.0, K=h_K, m=m, template_snap=checkpoint_y, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.03,
        snapshot_cost_estimate_s=0.02, fork_cost_estimate_s=0.05,
        step_cost_estimate_s=0.03, delete_cost_estimate_s=0.005,
        # Derived the way a block derives them: a lease that outlives the whole
        # run and a create budget bigger than one create's expected cost.
        ttl_s=int(6.0 + 3.0), create_deadline_s=0.5,
        journal_dir=journal_dir, run_id="dry-Hybrid",
    )
    assert cfg_y.ttl_s > cfg_y.T_s, (
        "the dry Hybrid's lease does not outlive its own run; the assertion "
        "below would not be testing the round-1 failure"
    )
    journal_y = RunJournal(os.path.join(journal_dir, "dry-Hybrid.jsonl"),
                           run_id="dry-Hybrid")
    llm_y = FakeLLM(journal=journal_y, log_full_requests=False, max_concurrency=h_K)
    run_y = build_run(cfg_y, farplane=fp_y, bridge_factory=bridge_y, llm=llm_y,
                      journal=journal_y)
    assert isinstance(run_y, HybridRun), "Hybrid must run on the fork substrate"
    res_y = await execute_run(run_y)
    journal_y.close()
    run_y.timings.check_sums()
    y_journal = os.path.join(journal_dir, "dry-Hybrid.jsonl")
    y_recs = _all_journal_records(y_journal)
    y_personas = [r for r in y_recs if r.get("kind") == "persona_assignment"]
    y_selection = [r for r in y_recs if r.get("kind") == "branch_selection"]
    y_waves = [r for r in y_recs
               if r.get("kind") == "event" and r.get("name") == "fork_wave"]
    y_admission = [r for r in y_recs
                   if r.get("kind") == "event" and r.get("name") == "phase2_admission"]
    y_overhead = [r for r in y_recs
                  if r.get("kind") == "event" and r.get("name") == "selection_overhead"]
    y_probes = _journal_records(y_journal, "probe")
    y_terminal = [r for r in y_probes if r.get("probe_kind") == "terminal"]
    y_select_probes = [r for r in y_probes if r.get("probe_kind") == "selection"]
    # ONE lease for every sandbox the arm touches, seats and halftime refork
    # children alike, and a create deadline sized for a queued burst. Round 1
    # lost both surviving cells to a lease shorter than the round, and Hybrid's
    # 8th seat to a create that starved past the wrapper's 300s default -- so the
    # substrate's OWN record of what it was handed is what gets asserted.
    want_ttl_y = f"{int(cfg_y.ttl_s)}s"
    # Only THIS run's resources: the checkpoint bake above shares the prefix.
    stem_y = f"{cfg_y.prefix}{cfg_y.run_id}"
    seat_calls = [c for c in fp_y.create_calls if c["name"].startswith(stem_y)]
    child_calls = [c for c in fp_y.fork_calls if c["name"].startswith(stem_y)]
    assert len(seat_calls) == h_K, (
        f"{len(seat_calls)} seat creates, expected {h_K}"
    )
    assert len(child_calls) == h_K - 1, (
        f"{len(child_calls)} refork children, expected {h_K - 1}"
    )
    assert {c["ttl"] for c in seat_calls + child_calls} == {want_ttl_y}, (
        "a seat or a refork child was handed a lease other than "
        f"{want_ttl_y}: creates {sorted({c['ttl'] for c in seat_calls})}, forks "
        f"{sorted({c['ttl'] for c in child_calls})}"
    )
    assert all(c["deadline"] == cfg_y.create_deadline_s for c in seat_calls), (
        f"a seat create did not carry the {cfg_y.create_deadline_s}s poll budget"
    )
    y_provision = [r for r in y_recs if r.get("kind") == "infra_op"
                   and r.get("op") in ("create_from_snapshot", "fork")]
    assert {r.get("ttl") for r in y_provision} == {want_ttl_y}, (
        "the journal does not record the lease on every create/fork"
    )

    # ONE convergence, at full width, with every seat's score on the record.
    assert len(y_selection) == 1, (
        f"Hybrid converged {len(y_selection)} time(s); the design's dose is 1"
    )
    assert len(y_select_probes) == h_K, (
        f"selection probed {len(y_select_probes)} of {h_K} seats; the judge must "
        "see them all"
    )
    assert len(y_selection[0]["scores"]) == h_K, (
        f"branch_selection carries {len(y_selection[0]['scores'])} scores, not "
        f"the {h_K} the leg-1 distribution needs"
    )
    assert res_y.branch_points == 1, f"dose {res_y.branch_points} != 1"
    assert len(y_waves) == 1 and not y_waves[0]["truncated"], (
        f"refork wave: {y_waves}"
    )
    assert y_waves[0]["k_effective"] == h_K, (
        f"phase 2 materialized k_effective={y_waves[0]['k_effective']}, not {h_K}"
    )
    assert res_y.exp3["validity"]["valid_width"], res_y.exp3["validity"]
    assert res_y.snapshots_created == 1, (
        f"Hybrid took {res_y.snapshots_created} snapshots; exactly one "
        "convergence means exactly one"
    )
    assert res_y.sandboxes_created == h_K + (h_K - 1), (
        f"Hybrid created {res_y.sandboxes_created} sandboxes; expected "
        f"{h_K} seats + {h_K - 1} forks"
    )
    # The losers die BEFORE the wave (they hold warm slots the forks need).
    y_deletes = [r for r in y_recs if r.get("kind") == "infra_op"
                 and r.get("op") == "delete_sandbox"]
    y_forks = [r for r in y_recs
               if r.get("kind") == "infra_op" and r.get("op") == "fork"]
    assert len(y_forks) == h_K - 1, f"{len(y_forks)} forks, expected {h_K - 1}"
    pre_wave_deletes = [d for d in y_deletes if d["seq"] < y_forks[0]["seq"]]
    assert len(pre_wave_deletes) == h_K - 1, (
        f"{len(pre_wave_deletes)} of {h_K - 1} losers were released before the "
        "refork wave; the wave competes with them for warm slots"
    )
    # Budget: T_total = 2P + MEASURED overhead, and leg 2 was admitted on it.
    assert y_overhead and y_admission, "the selection overhead was not journaled"
    assert y_overhead[0]["overhead_s"] > 0.0
    assert abs(y_overhead[0]["T_total_s"]
               - (cfg_y.T_s + y_overhead[0]["overhead_s"])) < 1e-6, (
        "T_total is not 2P + measured selection overhead"
    )
    assert y_admission[0]["admitted"], (
        f"leg 2 was refused its walk: {y_admission[0]}"
    )
    assert y_admission[0]["required_s"] == round(
        EXP3_PHASE2_ADMISSION * cfg_y.leg_s, 3
    )
    # Personas: one record per phase, a full permutation each, ROTATED by one.
    assert len(y_personas) == 2, (
        f"{len(y_personas)} persona_assignment record(s); expected one per phase"
    )
    phase1, phase2 = y_personas[0], y_personas[1]
    assert phase1["phase"] == 1 and phase2["phase"] == 2
    p1 = [phase1["seats"][f"L1s{i}"]["persona"] for i in range(h_K)]
    p2 = [phase2["seats"][f"L2s{i}"]["persona"] for i in range(h_K)]
    assert all(p1) and all(p2), "a seat ran without a persona"
    assert len(set(p1)) == h_K and len(set(p2)) == h_K, (
        "a persona was assigned to two seats in the same phase"
    )
    assert set(p1) == set(p2) == set(PERSONAS[:h_K]), (
        "a phase did not assign the full persona set"
    )
    assert all(a != b for a, b in zip(p1, p2)), (
        "phase 2 handed a seat its own phase-1 persona: the rotation is a no-op "
        "and leg 2's seats are conditioned exactly as before"
    )
    assert p2 == [PERSONAS[(i + 1) % len(PERSONAS)] for i in range(h_K)], (
        "phase 2 is not the pre-registered one-seat rotation"
    )
    # ... and each seat carries its persona exactly ONCE per phase, in its own
    # first user turn of that phase (the winner's leg-1 line is inherited, so a
    # leg-2 transcript holds exactly two: the winner's, then its own).
    winner_tid = y_selection[0]["winner"]
    winner_persona = phase1["seats"][winner_tid]["persona"]
    for i, traj in enumerate(run_y.trajectories):
        turns = persona_turns(traj.conv)
        assert len(turns) == 2, (
            f"seat {traj.tid} carries {len(turns)} persona line(s); expected the "
            "inherited leg-1 one plus its own leg-2 one"
        )
        assert turns[0][1] == winner_persona, (
            f"seat {traj.tid} did not inherit the winner's leg-1 persona"
        )
        assert turns[1][1] == p2[i], (
            f"seat {traj.tid} carries {turns[1][1][:40]!r}, not its rotated "
            f"persona {p2[i][:40]!r}"
        )
        assert turns[1][0] > turns[0][0], "leg 2's persona precedes leg 1's"
    # The endpoint: ALL K seats probed at T, endpoint = max, all of them reported.
    assert len(y_terminal) == h_K, (
        f"{len(y_terminal)} terminal probe(s) for {h_K} seats: judge-at-the-bell "
        "must measure every seat"
    )
    assert len(res_y.seat_endpoints) == h_K, (
        f"{len(res_y.seat_endpoints)} per-seat endpoints in the result; Exp 3 "
        "reports the distribution, not only the max"
    )
    y_seat_values = [s["throughput"] for s in res_y.seat_endpoints
                     if s["throughput"] is not None]
    assert len(y_seat_values) == h_K, "a seat endpoint is missing"
    assert res_y.endpoint_throughput == max(y_seat_values), (
        f"endpoint {res_y.endpoint_throughput} != max over seats "
        f"{max(y_seat_values)}"
    )
    assert res_y.exp3["endpoint_max"] == max(y_seat_values)
    assert res_y.endpoint_source.startswith("L2s"), (
        f"the endpoint came from {res_y.endpoint_source!r}, not a leg-2 seat"
    )
    assert res_y.active_s <= cfg_y.T_s + y_overhead[0]["overhead_s"] + \
        cfg_y.step_timeout_s, (
        f"Hybrid ran {res_y.active_s:.2f}s past 2P + overhead by more than one step"
    )
    assert checkpoint_y in world_y.snapshots, "Hybrid deleted its own checkpoint"
    leak_check("Hybrid", world_y, template_y, checkpoint_y)
    report["Hybrid"] = {
        "seed_snapshot": checkpoint_y,
        "T_total_s": cfg_y.T_s,
        "leg_s": cfg_y.leg_s,
        "K": h_K,
        "selection": {"winner": winner_tid,
                      "k_effective": y_selection[0]["k_effective"],
                      "scores": {k: v["probe_throughput"]
                                 for k, v in y_selection[0]["scores"].items()}},
        "refork": res_y.exp3["refork"],
        "overhead_s": y_overhead[0]["overhead_s"],
        "phase2_admission": {k: y_admission[0][k] for k in
                             ("admitted", "remaining_s", "required_s", "leg_s",
                              "fraction_of_P")},
        "personas": {"phase1": [p[:28] for p in p1], "phase2": [p[:28] for p in p2]},
        "selection_probes": len(y_select_probes),
        "terminal_probes": len(y_terminal),
        "seat_endpoints": [s["throughput"] for s in res_y.seat_endpoints],
        "endpoint": res_y.endpoint_throughput,
        "endpoint_source": res_y.endpoint_source,
        "steps": res_y.steps,
        "steps_per_seat": res_y.steps_per_trajectory,
        "sandboxes_created": res_y.sandboxes_created,
        "snapshots_created": res_y.snapshots_created,
        "validity": res_y.exp3["validity"],
        "active_s": round(res_y.active_s, 3),
        "timings": res_y.timings,
        "incidents": res_y.incidents,
    }

    # ---- Exp 3 Hybrid with a fork slower than any deadline: WIDTH floor ----
    # The refork wave is Exp 2's, so it truncates the same way -- and a
    # truncated wave is exactly how judge-at-the-bell would silently decay into
    # best-of-few. Exp 3's width floor must catch it, and the orphan child must
    # stay owned through its retained source snapshot.
    tr_K = 6
    world_t, fp_t, bridge_t, template_t = fake_substrate(slow_fork_at=2)
    checkpoint_t = bake_checkpoint(world_t, fp_t, template_t)
    cfg_t = ArmConfig(
        arm="Hybrid", model="fake-model", task_key="iron_plate_throughput",
        T_s=3.0, K=tr_K, m=m, template_snap=checkpoint_t, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.03,
        snapshot_cost_estimate_s=0.02, fork_cost_estimate_s=0.05,
        step_cost_estimate_s=0.03, delete_cost_estimate_s=0.005,
        step_timeout_s=1.0, journal_dir=journal_dir, run_id="dry-Hybridtrunc",
    )
    journal_t = RunJournal(os.path.join(journal_dir, "dry-Hybridtrunc.jsonl"),
                           run_id="dry-Hybridtrunc")
    llm_t = FakeLLM(journal=journal_t, log_full_requests=False, max_concurrency=tr_K)
    run_t = build_run(cfg_t, farplane=fp_t, bridge_factory=bridge_t, llm=llm_t,
                      journal=journal_t)
    res_t = await execute_run(run_t)
    journal_t.close()
    run_t.timings.check_sums()
    t_journal = os.path.join(journal_dir, "dry-Hybridtrunc.jsonl")
    t_recs = _all_journal_records(t_journal)
    t_waves = [r for r in t_recs
               if r.get("kind") == "event" and r.get("name") == "fork_wave"]
    t_orphans = [r for r in t_recs if r.get("kind") == "fork_orphan"]
    t_retained = [r for r in t_recs if r.get("kind") == "event"
                  and r.get("name") == "branch_snapshot_retained"]
    t_width = [r for r in t_recs
               if r.get("kind") == "event" and r.get("name") == "exp3_width"]
    t_terminal = [r for r in _journal_records(t_journal, "probe")
                  if r.get("probe_kind") == "terminal"]
    assert t_waves and t_waves[0]["truncated"], (
        f"the refork wave did not truncate: {t_waves}"
    )
    assert t_waves[0]["materialized"] < t_waves[0]["wanted"]
    k_eff_t = t_waves[0]["k_effective"]
    assert k_eff_t < EXP3_WIDTH_FLOOR, (
        f"the truncated wave still left k_effective={k_eff_t} >= the "
        f"{EXP3_WIDTH_FLOOR} floor, so this section asserts nothing"
    )
    assert t_width and t_width[0]["valid"] is False, (
        "a truncated refork wave was not marked invalid_width"
    )
    assert res_t.exp3["validity"]["status"] == "invalid_width"
    assert any(i["kind"] == "invalid_width" for i in res_t.incidents), (
        "invalid_width was not raised as an incident"
    )
    assert len(t_terminal) == k_eff_t, (
        f"{len(t_terminal)} terminal probes for {k_eff_t} surviving seats"
    )
    assert len(res_t.seat_endpoints) == k_eff_t
    # Wall clock stays hard even when the wave burns a fork deadline: the run
    # may spend 2P + the MEASURED overhead (which includes that abandoned fork)
    # and one in-flight step, and not a second more.
    t_overhead_s = (res_t.exp3.get("refork") or {}).get("overhead_s") or 0.0
    assert res_t.active_s <= cfg_t.T_s + t_overhead_s + cfg_t.step_timeout_s, (
        f"the truncated Hybrid ran {res_t.active_s:.2f}s, past 2P="
        f"{cfg_t.T_s}s + {t_overhead_s:.2f}s overhead by more than one step"
    )
    assert t_orphans and t_retained, "the orphaned fork child was not kept owned"
    orphan_src_t = t_orphans[0]["source_snapshot"]
    assert orphan_src_t in world_t.snapshots, (
        "the orphan's source snapshot was deleted; the child is unowned"
    )
    t_sweep = fp_t.reaper(cfg_t.prefix, keep=[template_t, checkpoint_t])
    t_swept = {r["id"] for r in t_sweep}
    assert all(o in t_swept for o in fp_t.orphans), (
        f"the reaper missed an orphan: {sorted(set(fp_t.orphans) - t_swept)}"
    )
    assert orphan_src_t in t_swept, "the retained snapshot was not swept"
    assert checkpoint_t not in t_swept and template_t not in t_swept
    leak_check("Hybrid-trunc", world_t, template_t, checkpoint_t)
    report["Hybrid-truncated"] = {
        "T_total_s": cfg_t.T_s,
        "leg_s": cfg_t.leg_s,
        "K": tr_K,
        "wave": {k: t_waves[0][k] for k in ("wanted", "materialized",
                                            "k_effective", "truncated", "orphans")},
        "width_floor": EXP3_WIDTH_FLOOR,
        "validity": res_t.exp3["validity"],
        "terminal_probes": len(t_terminal),
        "endpoint": res_t.endpoint_throughput,
        "orphan_snapshot_retained": orphan_src_t,
        "reaper_swept": sorted(t_swept),
        "active_s": round(res_t.active_s, 3),
        "overhead_s": round(
            (res_t.exp3.get("refork") or {}).get("overhead_s") or 0.0, 3),
        "timings": res_t.timings,
        "incidents": [i["kind"] for i in res_t.incidents],
    }

    # ---- Exp 3 A×K-S: the middle rung, personas, no snapshot, no fork ------
    world_p, fp_p, bridge_p, template_p = fake_substrate()
    checkpoint_p = bake_checkpoint(world_p, fp_p, template_p)
    cfg_p = ArmConfig(
        arm="AxK-S", model="fake-model", task_key="iron_plate_throughput",
        T_s=6.0, K=h_K, m=m, template_snap=checkpoint_p, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.03,
        journal_dir=journal_dir, run_id="dry-Exp3AxKS",
    )
    journal_p = RunJournal(os.path.join(journal_dir, "dry-Exp3AxKS.jsonl"),
                           run_id="dry-Exp3AxKS")
    llm_p = FakeLLM(journal=journal_p, log_full_requests=False, max_concurrency=h_K)
    run_p = build_run(cfg_p, farplane=fp_p, bridge_factory=bridge_p, llm=llm_p,
                      journal=journal_p)
    res_p = await execute_run(run_p)
    journal_p.close()
    run_p.timings.check_sums()
    p_journal = os.path.join(journal_dir, "dry-Exp3AxKS.jsonl")
    p_recs = _all_journal_records(p_journal)
    p_personas = [r for r in p_recs if r.get("kind") == "persona_assignment"]
    p_seeds = [r for r in p_recs if r.get("kind") == "infra_op"
               and r.get("op") == "create_from_snapshot"]
    p_terminal = [r for r in _journal_records(p_journal, "probe")
                  if r.get("probe_kind") == "terminal"]
    assert len(p_seeds) == h_K and all(r.get("source") == checkpoint_p
                                       for r in p_seeds), (
        "A×K-S must create exactly K seats from the checkpoint and nothing else"
    )
    assert res_p.snapshots_created == 0 and res_p.sandboxes_created == h_K, (
        "A×K-S touched the snapshot/fork lane; it is the never-converge rung"
    )
    assert not [r for r in p_recs if r.get("kind") == "branch_selection"], (
        "A×K-S converged; it must not"
    )
    assert res_p.branch_points == 0
    # One persona record, one persona per seat, each exactly once in its own
    # first user turn -- the same channel Hybrid uses, so the contrast between
    # the two arms is the convergence and nothing else.
    assert len(p_personas) == 1 and p_personas[0]["phase"] == 1
    p_assigned = [p_personas[0]["seats"][f"AxKS{i}"]["persona"] for i in range(h_K)]
    assert p_assigned == list(PERSONAS[:h_K]), (
        "A×K-S did not assign the persona set positionally"
    )
    for i, traj in enumerate(run_p.trajectories):
        turns = persona_turns(traj.conv)
        assert len(turns) == 1, (
            f"seat {traj.tid} carries {len(turns)} persona lines; A×K-S has one "
            "phase, so it must carry exactly one"
        )
        assert turns[0][1] == p_assigned[i]
    assert len(p_terminal) == h_K == len(res_p.seat_endpoints), (
        f"{len(p_terminal)} terminal probes / {len(res_p.seat_endpoints)} "
        f"reported endpoints for {h_K} seats"
    )
    p_seat_values = [s["throughput"] for s in res_p.seat_endpoints
                     if s["throughput"] is not None]
    assert res_p.endpoint_throughput == max(p_seat_values) == \
        res_p.exp3["endpoint_max"], (
        f"endpoint {res_p.endpoint_throughput} != max over seats"
    )
    assert checkpoint_p in world_p.snapshots, "A×K-S deleted its checkpoint"
    leak_check("AxK-S", world_p, template_p, checkpoint_p)
    report["AxK-S"] = {
        "seed_snapshot": checkpoint_p,
        "T_total_s": cfg_p.T_s,
        "leg_s": cfg_p.leg_s,
        "K": h_K,
        "seats_created_from_seed": len(p_seeds),
        "personas": [p[:28] for p in p_assigned],
        "terminal_probes": len(p_terminal),
        "seat_endpoints": [s["throughput"] for s in res_p.seat_endpoints],
        "endpoint": res_p.endpoint_throughput,
        "endpoint_source": res_p.endpoint_source,
        "steps": res_p.steps,
        "sandboxes_created": res_p.sandboxes_created,
        "snapshots_created": res_p.snapshots_created,
        "active_s": round(res_p.active_s, 3),
        "timings": res_p.timings,
        "incidents": res_p.incidents,
    }

    # ---- Exp 3 Control: ONE agent, NO persona, no fork ---------------------
    # The bottom rung has to be strict, and "strict" is a set of NEGATIVES: no
    # persona anywhere in its transcript, one seat, one create, zero snapshots,
    # zero forks, and an endpoint that is its own single terminal probe rather
    # than a max over anything. A block hands every cell K=8 and a persona set,
    # so the assertions below are what stops the control from quietly becoming a
    # narrow A×K-S.
    # Deferred: bench.tier05 imports THIS module, so the scheduler's slot rule
    # can only be checked from inside the gate.
    from bench.tier05 import peak_sandboxes

    world_n, fp_n, bridge_n, template_n = fake_substrate()
    checkpoint_n = bake_checkpoint(world_n, fp_n, template_n)
    cfg_n = ArmConfig(
        arm="Control", model="fake-model", task_key="iron_plate_throughput",
        T_s=6.0, K=h_K, m=m, template_snap=checkpoint_n, dry=True,
        terminal_reserve_s=0.1, probe_cost_estimate_s=0.03,
        journal_dir=journal_dir, run_id="dry-Exp3Control",
    )
    assert cfg_n.K == 1 and cfg_n.diversify == "never", (
        f"a K={h_K} persona block did not collapse to a strict control: "
        f"K={cfg_n.K} diversify={cfg_n.diversify!r}"
    )
    assert peak_sandboxes("Control", h_K) == 1, (
        "the scheduler would reserve K slots for a one-agent control"
    )
    journal_n = RunJournal(os.path.join(journal_dir, "dry-Exp3Control.jsonl"),
                           run_id="dry-Exp3Control")
    llm_n = FakeLLM(journal=journal_n, log_full_requests=False, max_concurrency=1)
    run_n = build_run(cfg_n, farplane=fp_n, bridge_factory=bridge_n, llm=llm_n,
                      journal=journal_n)
    assert run_n.hints_for(cfg_n.K) is None and run_n.hints_for(h_K) is None, (
        "the control was offered a diversity set"
    )
    res_n = await execute_run(run_n)
    journal_n.close()
    run_n.timings.check_sums()
    n_journal = os.path.join(journal_dir, "dry-Exp3Control.jsonl")
    n_recs = _all_journal_records(n_journal)
    n_seeds = [r for r in n_recs if r.get("kind") == "infra_op"
               and r.get("op") == "create_from_snapshot"]
    n_probes = _journal_records(n_journal, "probe")
    n_terminal = [r for r in n_probes if r.get("probe_kind") == "terminal"]
    n_parity = [r for r in n_probes if r.get("probe_kind") == "parity"]
    assert not [r for r in n_recs if r.get("kind") == "persona_assignment"], (
        "the control journaled a persona assignment"
    )
    assert not res_n.exp3["phases"] and res_n.exp3["persona"] is None
    assert len(n_seeds) == 1 and n_seeds[0].get("source") == checkpoint_n, (
        f"the control created {len(n_seeds)} seats from {checkpoint_n}"
    )
    assert res_n.sandboxes_created == 1 and res_n.snapshots_created == 0, (
        f"the control created {res_n.sandboxes_created} sandboxes / "
        f"{res_n.snapshots_created} snapshots; strict means 1 and 0"
    )
    assert res_n.branch_points == 0 and res_n.exp3["dose"] == 0
    for traj in run_n.trajectories:
        assert persona_turns(traj.conv) == [], (
            f"the control's transcript carries a persona line: {traj.tid}"
        )
    assert len(n_terminal) == 1 and len(res_n.seat_endpoints) == 1, (
        f"{len(n_terminal)} terminal probe(s) / {len(res_n.seat_endpoints)} "
        "endpoint(s); the control has exactly one factory"
    )
    assert n_parity, "the control skipped the parity probe cadence every arm shares"
    assert res_n.endpoint_throughput == res_n.seat_endpoints[0]["throughput"] == \
        res_n.exp3["endpoint_max"], (
        "the control's endpoint is not its own terminal probe"
    )
    assert res_n.endpoint_source == "Control"
    assert res_n.steps == res_n.steps_per_trajectory[0] == run_n.trajectories[0].step
    assert res_n.active_s <= cfg_n.T_s + cfg_n.step_timeout_s, (
        f"the control ran {res_n.active_s:.2f}s past T={cfg_n.T_s}s by more than "
        "one step"
    )
    assert checkpoint_n in world_n.snapshots, "the control deleted its checkpoint"
    leak_check("Control", world_n, template_n, checkpoint_n)
    report["Control"] = {
        "seed_snapshot": checkpoint_n,
        "T_total_s": cfg_n.T_s,
        "K": cfg_n.K,
        "diversify": cfg_n.diversify,
        "personas": [],
        "peak_sandboxes": peak_sandboxes("Control", h_K),
        "seats_created_from_seed": len(n_seeds),
        "parity_probes": len(n_parity),
        "terminal_probes": len(n_terminal),
        "seat_endpoints": [s["throughput"] for s in res_n.seat_endpoints],
        "endpoint": res_n.endpoint_throughput,
        "endpoint_source": res_n.endpoint_source,
        "steps": res_n.steps,
        "sandboxes_created": res_n.sandboxes_created,
        "snapshots_created": res_n.snapshots_created,
        "active_s": round(res_n.active_s, 3),
        "timings": res_n.timings,
        "incidents": res_n.incidents,
    }

    world_c, fp_c, bridge_c, template_c = fake_substrate()
    cfg_c = ArmConfig(arm="C", model="fake-model", task_key="iron_plate_throughput",
                      T_s=1e9, K=K, m=m, template_snap=template_c, dry=True,
                      terminal_reserve_s=0.0, probe_cost_estimate_s=0.1,
                      journal_dir=journal_dir, run_id="dry-C")
    journal_c = RunJournal(os.path.join(journal_dir, "dry-C.jsonl"), run_id="dry-C")
    llm_c = FakeLLM(journal=journal_c, log_full_requests=False, max_concurrency=K)
    run_c = RestoreBranchingRun(cfg_c, farplane=fp_c, bridge_factory=bridge_c,
                                llm=llm_c, journal=journal_c)
    res_c = _new_result(run_c)
    main_c_node = await run_c.provision_main("main")
    run_c.pool = [
        await run_c.infra.create_from_snapshot(template_c, f"pool{i + 1}")
        for i in range(K - 1)
    ]
    main_c = Trajectory(tid="main", node=main_c_node, conv=run_c.new_conversation())
    run_c.budget.start()
    run_c.timings.start()
    for round_idx in (1, 2):
        prefix_len = len(main_c.conv.messages)
        main_c = await run_c.branch_round(main_c, round_idx)
        assert len(main_c.conv.messages) - prefix_len == 2 * m
        assert len(run_c.pool) == K - 1, (
            f"arm C pool must stay at K-1={K - 1}, got {len(run_c.pool)}"
        )
    await run_c.terminal_probe(main_c)
    res_c.active_s = run_c.budget.elapsed_s()
    run_c.timings.stop()
    await run_c.teardown(list(run_c.infra.live_sandboxes.values()))
    _finish(run_c, res_c, [main_c])
    journal_c.close()
    run_c.timings.check_sums()
    c_journal = os.path.join(journal_dir, "dry-C.jsonl")
    assert _count_journal(c_journal, "fidelity") == 2 * (K - 1), (
        "arm C must log a P7 fidelity record per restored branch"
    )
    assert _count_journal(c_journal, "probe") == 2 * K + 1, (
        "arm C probes must be direct and per-branch (v2.6 parity)"
    )
    # C branches by /state-restore onto a fixed pool: main + (K-1) pool
    # sandboxes for the whole run, and not one Farplane snapshot or fork.
    assert res_c.sandboxes_created == K, (
        f"arm C must hold exactly main + {K - 1} pool sandboxes, "
        f"got {res_c.sandboxes_created}"
    )
    assert res_c.snapshots_created == 0, "arm C created snapshots (v2.6 allows none)"
    leak_check("C", world_c, template_c)
    report["C"] = {
        "branch_points": res_c.branch_points,
        "steps": main_c.step,
        "endpoint": res_c.endpoint_throughput,
        "timings": res_c.timings,
        "pool_size": len(run_c.pool),
        "fidelity_records": _count_journal(c_journal, "fidelity"),
        "sandboxes_created": res_c.sandboxes_created,
        "incidents": res_c.incidents,
    }

    for name in ("A", "B", "B-hardT", "B-slowWave", "B-forkDeadline",
                 "A-deadline", "Bonce", "AxK", "AxK-from-S", "Hybrid",
                 "Hybrid-truncated", "AxK-S", "Control", "C"):
        summary = report[name]["timings"]
        total = sum(summary["attributed_s"].values())
        # check_sums() already asserted the exact partition on raw values; this
        # only guards the reported (6-dp rounded) per-bucket numbers, whose sum
        # can differ from wall_s by up to one rounding step per bucket.
        assert abs(total - summary["wall_s"]) < 1e-5, (
            f"{name}: buckets {total} != wall {summary['wall_s']}"
        )
        assert abs(summary["attributed_total_s"] - summary["wall_s"]) < 1e-5
        report[name]["buckets_sum_s"] = round(total, 6)

    # ---- Cancellation is never swallowed; the endpoint never races --------
    # A step abandoned on the T deadline used to surface as
    # `execute_failed: CancelledError` while its worker thread kept mutating
    # the sandbox -- i.e. the terminal probe could measure a factory that was
    # still being built. Cancellation now propagates, and every arm joins its
    # in-flight calls before probing.
    ordering: dict[str, Any] = {}
    for name in DRY_JOURNALS:
        path = os.path.join(journal_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        recs = _all_journal_records(path)
        swallowed = [
            r for r in recs
            if r.get("kind") == "incident"
            and r.get("incident_kind") in ("execute_failed", "probe_failed",
                                           "baseline_read_failed")
            and "CancelledError" in str(r.get("detail", ""))
        ]
        assert not swallowed, (
            f"{name}: {len(swallowed)} cancellation(s) swallowed as substrate "
            f"errors, e.g. {swallowed[0].get('detail')!r}"
        )
        executes = [r for r in recs
                    if r.get("kind") == "infra_op" and r.get("op") == "execute"]
        terminals = [r for r in recs
                     if r.get("kind") == "probe" and r.get("probe_kind") == "terminal"]
        # Every execute record is written when the call CONCLUDES (settled
        # inline, or joined by the drain), so this is a real happens-before.
        for probe in terminals:
            later = [e for e in executes if e["seq"] > probe["seq"]]
            assert not later, (
                f"{name}: {len(later)} execute(s) concluded AFTER the terminal "
                f"probe on {probe.get('sandbox')} -- the endpoint raced a live "
                "mutation"
            )
        deadlines = [r for r in recs if r.get("kind") == "incident"
                     and r.get("incident_kind") == "step_deadline_cancelled"]
        settled = [e for e in executes
                   if e.get("outcome") == "settled_after_deadline"]
        abandoned = [e for e in executes if e.get("outcome") == "abandoned"]
        assert not abandoned, (
            f"{name}: {len(abandoned)} substrate call(s) outlived the bounded "
            "join; the endpoint is not safely ordered"
        )
        assert len(settled) <= len(deadlines), (
            f"{name}: {len(settled)} drained calls but only {len(deadlines)} "
            "deadline cancellations were journaled"
        )
        ordering[name] = {
            "executes": len(executes),
            "terminal_probes": len(terminals),
            "deadline_cancellations": len(deadlines),
            "joined_by_drain": len(settled),
        }
    assert any(v["deadline_cancellations"] for v in ordering.values()), (
        "no arm hit a step deadline; the cancellation assertions are vacuous"
    )
    report["cancellation_audit"] = ordering
    report["assertions"] = {
        "timing_partition_exact": True,
        "conversation_promotion": True,
        "pending_feedback_carried": True,
        "loser_archival_only": True,
        "probe_parity_all_arms": True,
        "zero_measurement_forks": True,
        "c_fidelity_logged": True,
        "no_resource_leak": True,
        #: Exp 2 additions.
        "seed_snapshot_propagated": True,
        "checkpoint_survives_run": True,
        "hint_per_branch_every_round": True,
        "hint_assignment_rotates": True,
        "exp2_m_covers_fork_wave": True,
        #: reviewer2-2 launch blockers.
        "b_refuses_round_it_cannot_afford": True,
        "b_continues_canonical_line_to_T": True,
        "b_overruns_T_by_at_most_one_step": True,
        "bonce_converges_exactly_once_at_midpoint": True,
        "bonce_winner_continues_to_T": True,
        #: reviewer2-2 residual blockers.
        "rounds_admitted_on_fork_p95": True,
        "fork_wave_stops_at_its_deadline": True,
        "cleanup_and_terminal_probe_reserved": True,
        "cancellation_never_swallowed": True,
        "endpoint_ordered_after_last_execute": True,
        #: Exp 3 additions.
        "exp3_persona_once_per_phase_per_seat": True,
        "exp3_personas_rotate_between_phases": True,
        "exp3_one_convergence_all_seats_scored": True,
        "exp3_losers_released_before_refork": True,
        "exp3_budget_is_2P_plus_measured_overhead": True,
        "exp3_leg2_admitted_on_09P": True,
        "exp3_all_seats_probed_at_T": True,
        "exp3_endpoint_is_max_over_seats": True,
        "exp3_per_seat_endpoints_reported": True,
        "exp3_truncated_refork_is_invalid_width": True,
        "exp3_axk_s_never_converges": True,
        #: Exp-3 restructure: the strict control rung.
        "exp3_control_is_one_seat_no_persona": True,
        "exp3_control_never_forks_or_snapshots": True,
        "exp3_control_endpoint_is_its_own_terminal_probe": True,
        "exp3_control_keeps_the_shared_probe_cadence": True,
        #: Round-1 post-mortem: the lease the SUBSTRATE was handed.
        "exp3_one_lease_for_seats_and_refork_children": True,
        "exp3_lease_outlives_the_run": True,
        "exp3_create_carries_the_configured_poll_budget": True,
        "exp3_lease_recorded_on_every_create_and_fork": True,
        #: Pre-flight guard, every arm/block path.
        "lease_guard_refuses_a_short_lease_at_construction": True,
        "lease_guard_names_every_number_in_its_error": True,
        "lease_guard_rejects_before_any_substrate_call": True,
        "lease_guard_accepts_the_live_presets": True,
        #: Online provider tripwire classification.
        "tripwire_ignores_the_empty_200_retry_class": True,
        "tripwire_counts_only_terminal_failures": True,
        "tripwire_fires_exactly_at_the_failure_budget": True,
        "tripwire_cause_names_provider_budget_and_error": True,
    }
    return report


def _all_journal_records(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _journal_records(path: str, kind: str) -> list[dict[str, Any]]:
    return [r for r in _all_journal_records(path) if r.get("kind") == kind]


def _count_journal(path: str, kind: str) -> int:
    return len(_journal_records(path, kind))


def _cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Farplane fan-out benchmark arms")
    ap.add_argument("--dry", action="store_true",
                    help="exercise the full loop against in-memory fakes")
    ap.add_argument("--arm", choices=ARMS, default="A")
    ap.add_argument("--model", default="k3")
    ap.add_argument("--task", default="iron_plate_throughput")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--T", type=float, default=2700.0)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--m", type=int, default=EXP2_DEFAULT_M,
                    help="steps per round / probe cadence (default: Exp 2's "
                         "sizing, see arms.exp2_round_sizing)")
    # --from-snap is the same knob under the name Exp 2 uses for it: A×K-from-S
    # is A×K seeded from a baked checkpoint (S2/S3) instead of TEMPLATE_SNAP.
    ap.add_argument("--template-snap", "--from-snap",
                    default=os.environ.get("TEMPLATE_SNAP", ""),
                    help="seed snapshot id: TEMPLATE_SNAP for greenfield arms, "
                         "a baked checkpoint for the -from-S variants")
    ap.add_argument("--live-url", default="",
                    help="smoke the real agent loop against one running bridge "
                         "(no Farplane; probes advance that container)")
    ap.add_argument("--live-steps", type=int, default=2)
    ap.add_argument("--out", default="")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.dry:
        report = asyncio.run(dry_run())
        payload = json.dumps(report, indent=2)
        if args.out:
            atomic_write_json(args.out, report)
        print(payload)
        sizing = report["exp2_sizing"]
        hard, once = report["B-hardT"], report["Bonce"]
        shared = "" if report["journal_dir_exclusive"] else (
            " (PRIVATE: the canonical dir was already claimed by another gate)"
        )
        print("\nDRY RUN OK: conversation promotion, loser archival, resource "
              "cleanup, exact timing partition, checkpoint seeding and B's "
              "per-round hint rotation all asserted.")
        print(f"Journals: {report['journal_dir']}{shared}")
        print(f"Exp-2 default m = {sizing['m']}: {sizing['arithmetic']}")
        print(
            f"Hard T: B started {hard['rounds_started']} round(s) at "
            f"{hard['round_estimate_s']}s estimate, refused the next with "
            f"{hard['convergence_stopped']['remaining_s']}s left, ran the "
            f"canonical line to {hard['active_s']}s of T={hard['T_s']}s "
            f"(overrun {hard['overrun_s']}s)."
        )
        chosen = once["convergence"]
        print(
            f"B-once: one convergence at {chosen['boundary_s']}s "
            f"({100 * chosen['boundary_s'] / once['T_s']:.0f}% of T; T/2 = "
            f"{once['midpoint_s']}s), the LAST boundary that could afford a "
            f"round -- step {once['steps_before_convergence']} -> "
            f"{once['steps_total']}, {once['snapshots_created']} snapshot / "
            f"{once['sandboxes_created']} sandboxes."
        )
        admit, wave = report["exp2_admission"], report["B-slowWave"]
        first = wave["waves"][0]
        print(
            f"Admission (LIVE): round = {admit['snapshot_s']}s snapshot + 7 x "
            f"{admit['fork_p95_s']}s fork p95 ({admit['wave_at_p95_s']}s wave) + "
            f"{admit['rollout_s']}s rollout + {admit['cleanup_s']}s cleanup = "
            f"{admit['round_estimate_s']}s, vs {admit['rollout_budget_s']}s of T "
            f"-> {admit['rounds_that_fit_worst_case']} round(s) guaranteed, more "
            f"as forks come in under p95 (dose is measured, not assumed)."
        )
        fdl = report["B-forkDeadline"]
        print(
            f"Fork deadline: fork #{fdl['slow_fork_index']} needed "
            f"{fdl['slow_fork_needed_s']}s, abandoned after "
            f"{fdl['abandoned_after_s']}s on a {fdl['deadlines_given_s']}s "
            f"deadline -> wave {fdl['wave']['materialized']}/"
            f"{fdl['wave']['wanted']}, K_eff={fdl['wave']['k_effective']}, "
            f"{fdl['wave']['orphans']} orphan kept owned via "
            f"{fdl['snapshot_retained']} and swept by the reaper "
            f"({len(fdl['reaper_swept'])} resources); ran to {fdl['active_s']}s "
            f"of T={fdl['T_s']}s (overrun {fdl['overrun_s']}s)."
        )
        print(
            f"B-once boundary (LIVE): block {admit['bonce_boundary_index']} at "
            f"{admit['bonce_boundary_s']}s "
            f"({admit['bonce_fraction_of_T']:.0%} of T); first boundary past T/2 "
            f"is {admit['bonce_first_boundary_past_half_s']}s, which leaves "
            f"{admit['rollout_budget_s'] - admit['bonce_first_boundary_past_half_s']:.1f}s "
            f"< the {admit['round_estimate_s']}s estimate -- T/2 would never "
            f"converge. Dose floor {admit['dose_floor']}, "
            f"{admit['rounds_that_fit_worst_case']} guaranteed / "
            f"{admit['rounds_that_fit_typical']} typical."
        )
        print(
            f"Slow wave: forks over p95 -> wave stopped at "
            f"{first['materialized']} of {first['wanted']}, round converged at "
            f"K_eff={first['k_effective']} (of {wave['K']}), "
            f"{wave['rounds_aborted']} round(s) aborted, ran to "
            f"{wave['active_s']}s of T={wave['T_s']}s "
            f"(overrun {wave['overrun_s']}s)."
        )
        audit = report["cancellation_audit"]
        print(
            "Cancellation: "
            f"{sum(a['deadline_cancellations'] for a in audit.values())} step "
            f"deadline(s) hit, "
            f"{sum(a['joined_by_drain'] for a in audit.values())} joined before "
            "the endpoint, 0 swallowed as execute_failed, 0 executes concluded "
            "after a terminal probe."
        )
        hyb, hyb_t = report["Hybrid"], report["Hybrid-truncated"]
        ctl, arith = report["AxK-S"], report["exp3_arithmetic"]
        strict = report["Control"]
        print(
            f"Exp-3 Hybrid: leg 1 = {hyb['leg_s']}s x {hyb['K']} personas -> ONE "
            f"selection over {hyb['selection']['k_effective']} seats (winner "
            f"{hyb['selection']['winner']}) -> refork "
            f"{hyb['refork']['materialized']}/{hyb['refork']['wanted']} "
            f"(k_eff={hyb['refork']['k_effective']}, "
            f"{hyb['overhead_s']}s measured overhead) -> leg 2 admitted with "
            f"{hyb['phase2_admission']['remaining_s']}s "
            f"({hyb['phase2_admission']['fraction_of_P']:.0%} of P, floor "
            f"{hyb['phase2_admission']['required_s']}s) -> "
            f"{hyb['terminal_probes']} terminal probes, endpoint "
            f"{hyb['endpoint']} = max over seats from {hyb['endpoint_source']}."
        )
        print(
            f"Exp-3 personas: 8 written, one per seat per phase, rotated by one "
            f"seat between phases (seat 0: {hyb['personas']['phase1'][0]!r} -> "
            f"{hyb['personas']['phase2'][0]!r}); A×K-S assigns the same set "
            f"positionally, converges never ({ctl['snapshots_created']} "
            f"snapshots, {ctl['sandboxes_created']} sandboxes) and reports all "
            f"{ctl['terminal_probes']} seat endpoints (endpoint {ctl['endpoint']})."
        )
        print(
            f"Exp-3 Control (strict): K={strict['K']} seat, "
            f"diversify={strict['diversify']!r}, {len(strict['personas'])} "
            f"personas, {strict['seats_created_from_seed']} create from the "
            f"checkpoint, {strict['snapshots_created']} snapshots, "
            f"{strict['parity_probes']} parity probes at cadence m, "
            f"{strict['terminal_probes']} terminal probe = its own endpoint "
            f"{strict['endpoint']} (peak slots {strict['peak_sandboxes']})."
        )
        print(
            f"Exp-3 width floor: a fork past its deadline truncated the refork "
            f"wave to {hyb_t['wave']['materialized']}/{hyb_t['wave']['wanted']} "
            f"(k_eff={hyb_t['wave']['k_effective']} < floor "
            f"{hyb_t['width_floor']}) -> {hyb_t['validity']['status']}, orphan "
            f"kept owned via {hyb_t['orphan_snapshot_retained']} and swept "
            f"({len(hyb_t['reaper_swept'])} resources)."
        )
        print(
            f"Exp-3 arithmetic (LIVE): P={arith['P_s']:.0f}s, "
            f"T_total=2P={arith['T_total_s']:.0f}s, K={arith['K']}, "
            f"m={arith['m']}; convergence costs ~"
            f"{arith['convergence_estimate_s']:.0f}s (snapshot + 7 forks at p95 "
            f"+ 8 deletes) and is added BACK to the budget, so leg 2 starts with "
            f"{arith['leg2_start_s']:.0f}s vs the "
            f"{arith['leg2_required_s']:.0f}s "
            f"({arith['admission_fraction']:.0%} of P) floor."
        )
        tw = report["provider_tripwire"]
        print(
            f"Provider tripwire: {tw['noise']['empty_200s']} empty-200s absorbed "
            f"by retries counted {tw['noise']['counted_failures']} failures "
            f"({tw['noise']['retry_noise']} recorded as noise, "
            f"{tw['noise']['successes']} successes); a dead provider fired at "
            f"call {tw['death']['fired_at_call']} on "
            f"{tw['consecutive_limit']} consecutive terminal failures "
            f"(silence window {tw['silence_limit_s']:.0f}s)."
        )
        lg = report["lease_guard"]
        print(
            f"Lease guard (pre-flight, every arm/block path): a "
            f"{lg['rejected']['ttl_s']}s lease for a "
            f"{lg['rejected']['T_s']:.0f}s run is REFUSED at ArmConfig "
            f"construction with {lg['substrate_calls_on_rejection']} new "
            f"substrate calls (T + stagger + reserve + {lg['slack_s']:.0f}s slack "
            f"required); {lg['accepted_ttl_s']}s passes; dry cells exempt "
            f"({lg['dry_cells_exempt']}) because T is clamped to seconds there."
        )
        return 0
    if args.live_url:
        report = asyncio.run(
            live_smoke(base_url=args.live_url, model=args.model, task=args.task,
                       steps=args.live_steps)
        )
        # The smoke's run id is unique per invocation, so its DEFAULT results
        # path is too: a fixed bench/results/live_smoke.json overwrote the
        # previous smoke's evidence, which is the only thing a smoke report is
        # for. An explicit --out still wins.
        run_id = str(report.get("run_id") or "live_smoke")
        out = args.out or os.path.join(ArmConfig.results_dir, f"{run_id}.json")
        report["results_path"] = out
        atomic_write_json(out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    cfg = ArmConfig(arm=args.arm, model=args.model, task_key=args.task,
                    replicate=args.replicate, T_s=args.T, K=args.K, m=args.m,
                    template_snap=args.template_snap)
    result = asyncio.run(run_one(cfg))
    payload = json.dumps(result.to_dict(), indent=2)
    out = args.out or os.path.join(cfg.results_dir, f"{cfg.run_id}.json")
    atomic_write_json(out, result.to_dict())
    print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
