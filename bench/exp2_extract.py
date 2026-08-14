"""Streaming extractor for Experiment 2 journals.

The run journals are up to 2.6 GB each (llm_call.request/response_text and
branch_archive.messages dominate) and single records reach hundreds of MB.
Nothing here holds a whole record: the file is read in fixed-size chunks, record
boundaries are found by scanning for newlines, and each record is retained only
up to the budget its kind earns (16 KiB for the two payload kinds, 4 MiB for
metadata records such as ``run_finished``).  Everything past that is discarded
as it streams past, so peak memory is ``CHUNK + PREFIX_MAX`` regardless of
record size or file size.

A record whose tail was discarded is parsed from its retained prefix: the parser
walks the prefix, finds the last position where the object can be closed after a
complete top-level ``"key": value`` pair, and parses that.  Every field this
extractor reads is a scalar that precedes the payload keys, so nothing needed is
lost -- and when a prefix yields nothing parseable the record is recorded as a
parse error instead of being silently dropped.

Session semantics follow ``bench.common.load_journal_records``: a journal may
contain several append sessions (one ``journal_open`` event each, every record
tagged with its ``session`` id).  A multi-session file is AMBIGUOUS and is
refused unless ``--session latest|all|<id>`` says which one to digest.

Emits one compact dict per journal into bench/results/exp2_extract.json so the
analysis (bench/analyze_exp2.py) never has to touch the raw journals again.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

from bench.common import atomic_write_json

KIND = re.compile(rb'"kind": "([A-Za-z_]+)"')
NAME = re.compile(rb'"name": "([A-Za-z_0-9]+)"')
SESSION = re.compile(rb'"session": "([0-9a-zA-Z]+)"')

CHUNK = 1 << 20         # bytes per read
PREFIX_MAX = 1 << 22    # bytes retained per metadata record (4 MiB)
PAYLOAD_PREFIX = 1 << 14  # ... but only 16 KiB of the two payload kinds
HEAD = 240              # bytes classified for kind/name/session
MAX_ERRORS = 200        # parse errors reported verbatim (count is always exact)

# fields that live BEFORE the fat payload keys of the two heavy record kinds
TRUNC_AT = {
    b"llm_call": b', "request":',
    b"branch_archive": b', "messages":',
}


class SessionAmbiguity(ValueError):
    """Raised when a journal holds several sessions and none was selected."""


# --------------------------------------------------------------------------
# bounded scanner
# --------------------------------------------------------------------------
def budget_for(head: bytes) -> int:
    """Bytes worth retaining for a record, decided from its first bytes.

    The two payload kinds carry the GB-scale fields and are cut at a known
    marker a few hundred bytes in, so keeping 16 KiB of them is already far more
    than this digest reads.  Every other kind is metadata -- ``run_finished``
    embeds the run's whole incident list and reaches ~2.5 MB -- and is kept
    whole up to ``PREFIX_MAX``.
    """
    m = KIND.search(head)
    if m is not None and m.group(1) in TRUNC_AT:
        return PAYLOAD_PREFIX
    return PREFIX_MAX


def iter_records(
    fh: BinaryIO, chunk_size: int = CHUNK, limit: Callable[[bytes], int] = budget_for,
) -> Iterator[tuple[int, int, bytes, int, bool, bool]]:
    """Yield ``(lineno, offset, prefix, size, truncated, terminated)`` per record.

    ``prefix`` holds the record's leading bytes up to the budget ``limit``
    returns for it (from its first ``HEAD`` bytes); ``size`` is the record's full
    length; ``truncated`` says bytes were discarded as they streamed past;
    ``terminated`` is False only for a record that hit EOF without a newline
    (necessarily the last one).  ``lineno`` is 1-based, ``offset`` is the byte
    offset of the record's first byte.  Peak memory is ``chunk_size`` plus the
    largest budget, whatever the record and file sizes are.
    """
    lineno = 0
    offset = 0
    parts: list[bytes] = []
    kept = 0
    size = 0
    budget = 0          # 0 until the head has been seen
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        pos = 0
        n = len(chunk)
        while pos < n:
            nl = chunk.find(b"\n", pos)
            end = n if nl < 0 else nl
            size += end - pos
            if not budget and kept + (end - pos) >= HEAD:
                # take the head first, then size the budget from it
                take = HEAD - kept
                parts.append(chunk[pos:pos + take])
                kept += take
                pos += take
                budget = limit(b"".join(parts))
            if kept < (budget or HEAD):
                take = min(end - pos, (budget or HEAD) - kept)
                parts.append(chunk[pos:pos + take])
                kept += take
            if nl < 0:
                break
            lineno += 1
            yield lineno, offset, b"".join(parts), size, kept < size, True
            offset += size + 1
            parts = []
            kept = 0
            size = 0
            budget = 0
            pos = nl + 1
    if parts or size:
        lineno += 1
        yield lineno, offset, b"".join(parts), size, kept < size, False


def _prefix_object(buf: bytes) -> dict:
    """Parse the complete top-level fields of a record whose tail is gone.

    Walks the retained prefix once tracking string/escape state and nesting
    depth, then closes the object after the last complete top-level pair.
    """
    if not buf.startswith(b"{"):
        raise ValueError("oversized record does not start with '{'")
    depth = 0
    in_str = False
    esc = False
    cut = -1
    for i, ch in enumerate(buf):
        if in_str:
            if esc:
                esc = False
            elif ch == 0x5C:        # backslash
                esc = True
            elif ch == 0x22:        # quote
                in_str = False
            continue
        if ch == 0x22:
            in_str = True
        elif ch in (0x7B, 0x5B):    # { [
            depth += 1
        elif ch in (0x7D, 0x5D):    # } ]
            depth -= 1
        elif ch == 0x2C and depth == 1:   # top-level comma
            cut = i
    if cut < 0:
        raise ValueError(f"oversized record: no complete top-level field in the "
                         f"first {len(buf)} retained bytes")
    try:
        rec = json.loads(buf[:cut] + b"}")
    except Exception as exc:
        raise ValueError(f"oversized record prefix unparseable: {exc}") from None
    if not isinstance(rec, dict):
        raise ValueError("oversized record prefix is not a JSON object")
    return rec


def _load(buf: bytes, kind: str, truncated: bool) -> dict:
    """Parse one record from its retained prefix.  Raises ValueError."""
    cut = TRUNC_AT.get(kind.encode())
    if cut is not None:
        i = buf.find(cut)
        if i > 0:
            buf, truncated = buf[:i] + b"}", False
    if truncated:
        return _prefix_object(buf)
    try:
        rec = json.loads(buf)
    except Exception as exc:
        raise ValueError(f"json: {exc}") from None
    if not isinstance(rec, dict):
        raise ValueError("record is not a JSON object")
    return rec


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[i], 3)


# --------------------------------------------------------------------------
# digest accumulator (one instance per session digested)
# --------------------------------------------------------------------------
class Digest:
    """Accumulates one session's digest.  ``feed`` returns an error string."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kc: Counter = Counter()
        self.events: list[dict] = []
        self.fork_waves: list[dict] = []
        self.branch_selections: list[dict] = []
        self.probes: list[dict] = []
        self.incidents: list[dict] = []
        self.hint_assignments: list[dict] = []
        self.step_trail: list[list] = []
        self.infra_raw: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.infra_fail: Counter = Counter()
        self.llm_lat: list[float] = []
        self.llm_out: Counter = Counter()
        self.llm_attempts: Counter = Counter()
        self.llm_tok = [0, 0, 0]        # prompt, completion, reasoning
        self.oversized: Counter = Counter()
        self.steps: dict[str, dict] = defaultdict(
            lambda: {"n": 0, "ticks": 0, "errors": 0, "exec_s": 0.0,
                     "prod_last": None, "auto_last": None, "noop": 0}
        )
        self.t_start_ts = None
        self.first_ts = None
        self.last_ts = None
        self.n_records = 0

    def feed(self, kind: str, prefix: bytes, truncated: bool) -> str | None:
        self.n_records += 1
        self.kc[kind] += 1
        if truncated:
            self.oversized[kind] += 1
        if kind == "branch_archive":
            return None                 # counted only; the payload IS the record
        try:
            rec = _load(prefix, kind, truncated)
        except ValueError as exc:
            return str(exc)
        if kind == "step":
            self._step(rec)
            return None
        if kind == "llm_call":
            self._llm(rec)
            return None
        self._other(kind, rec)
        return None

    # -- per-kind ----------------------------------------------------------
    def _step(self, rec: dict) -> None:
        b = rec.get("branch") or ""
        s = self.steps[b]
        s["n"] += 1
        s["ticks"] += int(rec.get("ticks") or 0)
        s["errors"] += 1 if rec.get("error") else 0
        s["exec_s"] += float(rec.get("exec_s") or 0.0)
        s["prod_last"] = rec.get("production_score")
        s["auto_last"] = rec.get("automated_score")
        if not (rec.get("ticks") or 0) and not (rec.get("code_chars") or 0):
            s["noop"] += 1
        self.step_trail.append(
            [rec.get("step"), b, round(float(rec.get("exec_s") or 0.0), 3),
             int(rec.get("ticks") or 0), bool(rec.get("error"))]
        )

    def _llm(self, rec: dict) -> None:
        self.llm_lat.append(float(rec.get("latency_s") or 0.0))
        self.llm_out[rec.get("outcome") or "?"] += 1
        self.llm_attempts[int(rec.get("attempt") or 0)] += 1
        self.llm_tok[0] += int(rec.get("prompt_tokens") or 0)
        self.llm_tok[1] += int(rec.get("completion_tokens") or 0)
        self.llm_tok[2] += int(rec.get("reasoning_tokens") or 0)

    def _other(self, kind: str, rec: dict) -> None:
        ts = rec.get("ts")
        if ts:
            if self.first_ts is None:
                self.first_ts = ts
            self.last_ts = ts
        if kind == "infra_op":
            key = (rec.get("bucket") or "?", rec.get("op") or "?")
            self.infra_raw[key].append(float(rec.get("duration_s") or 0.0))
            if (rec.get("outcome") or "ok") != "ok":
                self.infra_fail[key] += 1
        elif kind == "probe":
            self.probes.append({
                "ts": ts, "probe_kind": rec.get("probe_kind"),
                "branch": rec.get("branch"), "step": rec.get("step"),
                "throughput": rec.get("throughput"),
                "start_tick": rec.get("start_tick"),
                "end_tick": rec.get("end_tick"),
                "window_ticks": rec.get("window_ticks"),
                "start_count": rec.get("start_count"),
                "end_count": rec.get("end_count"),
                "wall_s": rec.get("wall_s"),
                "client_wall_s": rec.get("client_wall_s"),
                "sandbox": rec.get("sandbox"), "cold": rec.get("cold"),
                "speed": rec.get("speed"), "timed_out": rec.get("timed_out"),
            })
        elif kind == "branch_selection":
            self.branch_selections.append({
                "ts": ts, "round": rec.get("round"),
                "winner": rec.get("winner"),
                "k_effective": rec.get("k_effective"),
                "scores": rec.get("scores") or {},
            })
        elif kind == "incident":
            self.incidents.append({
                "ts": ts, "incident_kind": rec.get("incident_kind"),
                "detail": (rec.get("detail") or "")[:300],
                "branch": rec.get("branch"), "step": rec.get("step"),
            })
        elif kind == "hint_assignment":
            self.hint_assignments.append({
                "round": rec.get("round"),
                "seats": sorted((rec.get("hints") or {}).keys()),
                "hint_chars": {k: len(v) for k, v in (rec.get("hints") or {}).items()},
            })
        elif kind == "event":
            nm = rec.get("name")
            if nm == "fork_wave":
                self.fork_waves.append({
                    "ts": ts, "round": rec.get("round"),
                    "wanted": rec.get("wanted"),
                    "materialized": rec.get("materialized"),
                    "k_effective": rec.get("k_effective"),
                    "truncated": rec.get("truncated"),
                    "reason": rec.get("reason"),
                    "fork_estimate_s": rec.get("fork_estimate_s"),
                    "orphans": rec.get("orphans"),
                })
            else:
                slim = {k: v for k, v in rec.items()
                        if k not in ("mono", "run_id", "kind", "hints")}
                if isinstance(slim.get("label"), str):
                    slim.pop("label")
                self.events.append(slim)
            if nm == "T_start":
                self.t_start_ts = ts

    # -- output ------------------------------------------------------------
    def result(self) -> dict:
        rows = []
        for (bucket, op), vals in sorted(self.infra_raw.items()):
            rows.append({
                "bucket": bucket, "op": op, "n": len(vals),
                "total_s": round(sum(vals), 3), "p50": _pct(vals, 0.5),
                "p95": _pct(vals, 0.95), "max": round(max(vals), 3),
                "fails": self.infra_fail[(bucket, op)],
            })
        infra: dict[str, dict] = {}
        for r in rows:
            infra.setdefault(r["bucket"], {"n": 0, "total_s": 0.0})
            infra[r["bucket"]]["n"] += r["n"]
            infra[r["bucket"]]["total_s"] = round(
                infra[r["bucket"]]["total_s"] + r["total_s"], 3)
        lat = self.llm_lat
        return {
            "journal": str(self.path),
            "kind_counts": dict(self.kc),
            "events": self.events,
            "fork_waves": self.fork_waves,
            "branch_selections": self.branch_selections,
            "probes": self.probes,
            "incidents": self.incidents,
            "hint_assignments": self.hint_assignments,
            "infra": infra,
            "infra_ops": rows,
            "llm": {
                "n": len(lat), "total_s": round(sum(lat), 3),
                "mean_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
                "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                "max": round(max(lat), 3) if lat else 0.0,
                "outcomes": dict(self.llm_out),
                "attempts": {str(k): v for k, v in sorted(self.llm_attempts.items())},
                "prompt_tokens": self.llm_tok[0],
                "completion_tokens": self.llm_tok[1],
                "reasoning_tokens": self.llm_tok[2],
            },
            "steps": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()}
                      for k, v in sorted(self.steps.items())},
            "step_trail": self.step_trail,
            "t_start_ts": self.t_start_ts,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "oversized_records": dict(self.oversized),
        }


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def extract(path: Path, *, session: str | None = None) -> dict:
    """Digest one journal.

    ``session`` selects an append session: ``None`` (default) refuses a
    multi-session file, ``"latest"`` takes the last one, ``"all"`` merges
    everything, an explicit 12-hex id takes that session.  Parse failures are
    counted and reported in ``parse_errors``; a single unterminated FINAL
    record is tolerated (recorded in ``scan``), everything else is an error.
    """
    sel = (session or "strict").strip()
    dig = Digest(path)
    errors: list[dict] = []
    n_errors = 0
    n_records = 0
    n_bytes = 0
    n_oversized = 0
    n_foreign = 0
    n_blank = 0
    unterminated = False
    open_count = 0
    segments: list[str | None] = []     # session id per append segment, in order
    cur_sid: str | None = None

    def note(lineno: int, offset: int, msg: str) -> None:
        nonlocal n_errors
        n_errors += 1
        if len(errors) < MAX_ERRORS:
            errors.append({"lineno": lineno, "offset": offset, "error": msg})

    with open(path, "rb") as fh:
        for lineno, offset, prefix, size, truncated, terminated in iter_records(fh):
            n_records += 1
            n_bytes += size
            if not terminated:
                # a run killed mid-write leaves one torn record; tolerated,
                # reported, and never parsed.
                unterminated = True
                print(f"[extract] warning: {path.name}: unterminated final record at "
                      f"line {lineno} (offset {offset}, {size} bytes) -- dropped",
                      file=sys.stderr, flush=True)
                continue
            if not prefix.strip():
                n_blank += 1
                note(lineno, offset, "blank line")
                continue
            head = prefix[:HEAD]
            m = KIND.search(head)
            if m is None:
                note(lineno, offset,
                     f"no \"kind\" field in the first {HEAD} bytes")
                continue
            kind = m.group(1).decode()
            if truncated:
                n_oversized += 1
            sm = SESSION.search(head)
            sid = sm.group(1).decode() if sm else None
            is_open = False
            if kind == "event":
                nm = NAME.search(head)
                is_open = nm is not None and nm.group(1) == b"journal_open"
                if is_open:
                    open_count += 1
            if is_open or not segments:
                # a new append segment: an explicit journal_open, or the very
                # first record of a journal that was never opened with one
                # (legacy content counts as one implicit session).
                segments.append(sid)
                cur_sid = sid
                if len(segments) > 1:
                    if sel == "strict":
                        raise SessionAmbiguity(
                            f"{path}: {len(segments)} append sessions "
                            f"({[s or 'legacy' for s in segments]}, "
                            f"{open_count} journal_open records); pass "
                            f"--session latest|all|<id> to say which one to digest")
                    if sel == "latest":
                        dig = Digest(path)      # only the newest one is wanted
            if sel != "all":
                target = cur_sid if sel in ("strict", "latest") else sel
                if sid is None and target is not None:
                    # no session marker in the head: legacy record, or a
                    # malformed one.  Parse it rather than dropping it, and
                    # attribute it from the parsed record.
                    try:
                        sid = _load(prefix, kind, truncated).get("session")
                    except ValueError as exc:
                        note(lineno, offset, f"{kind}: {exc}")
                        continue
                if sid != target:
                    n_foreign += 1      # another session's records, interleaved
                    continue
            err = dig.feed(kind, prefix, truncated)
            if err is not None:
                note(lineno, offset, f"{kind}: {err}")

    if not n_records:
        # an empty journal is missing evidence, not a clean zero digest
        note(1, 0, "journal contains no records")
    if sel not in ("strict", "latest", "all") and dig.n_records == 0:
        raise ValueError(f"{path}: no records for session {sel!r} "
                         f"(sessions present: {[s or 'legacy' for s in segments]})")

    out = dig.result()
    out["session_selector"] = sel
    out["journal_open_count"] = open_count
    out["sessions"] = [s if s is not None else "legacy" for s in segments]
    out["session"] = ("all" if sel == "all" else
                      (cur_sid if sel in ("strict", "latest") else sel) or "legacy")
    out["parse_errors"] = errors
    out["scan"] = {
        "records": n_records,
        "bytes": n_bytes,
        "records_digested": dig.n_records,
        "oversized_record_count": n_oversized,
        "foreign_session_records": n_foreign,
        "blank_lines": n_blank,
        "unterminated_final_record": unterminated,
        "parse_error_count": n_errors,
        "parse_errors_truncated": n_errors > len(errors),
        "chunk_bytes": CHUNK,
        "prefix_bytes": PREFIX_MAX,
        "payload_prefix_bytes": PAYLOAD_PREFIX,
    }
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="exp2_extract",
        description="Bounded-memory digest of bench/journal/exp2/*.jsonl")
    ap.add_argument("patterns", nargs="*", default=None,
                    help="substrings selecting journals (default: codex k3)")
    ap.add_argument("--journal-dir", default="bench/journal/exp2")
    ap.add_argument("--out", default="bench/results/exp2_extract.json")
    ap.add_argument("--session", default=None,
                    help="latest | all | <12-hex session id>; default refuses "
                         "a multi-session journal")
    args = ap.parse_args(argv[1:])

    jdir = Path(args.journal_dir)
    pats = args.patterns or ["codex", "k3"]
    files = sorted(
        p for p in jdir.glob("*.jsonl")
        if not p.name.endswith("-farplane.jsonl")
        and p.name not in ("tier1-master.jsonl", "reaper.jsonl")
        and any(s in p.name for s in pats)
    )
    if not files:
        print(f"[extract] no journals under {jdir} matching {pats}", file=sys.stderr)
        return 1
    res = {}
    bad = 0
    for p in files:
        sz = p.stat().st_size
        print(f"[extract] {p.name} ({sz/1e6:.0f} MB)", flush=True)
        try:
            d = extract(p, session=args.session)
        except (SessionAmbiguity, ValueError) as exc:
            # nothing is written: a partial digest is worse evidence than none
            print(f"[extract] REFUSED {p.name}: {exc}", file=sys.stderr)
            return 1
        res[p.stem] = d
        print(f"          kinds={d['kind_counts']}", flush=True)
        scan = d["scan"]
        if scan["parse_error_count"]:
            bad += 1
            print(f"          PARSE ERRORS: {scan['parse_error_count']} "
                  f"(first: {d['parse_errors'][0]})", file=sys.stderr, flush=True)
        if scan["oversized_record_count"]:
            print(f"          oversized={scan['oversized_record_count']} "
                  f"(parsed from their retained prefix)", flush=True)
    outp = Path(args.out)
    atomic_write_json(outp, res, indent=1)
    print(f"[extract] wrote {outp} ({outp.stat().st_size/1e6:.1f} MB)")
    if bad:
        print(f"[extract] {bad}/{len(files)} journals had parse errors -- the "
              f"analysis will refuse this digest", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
