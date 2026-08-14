"""Streaming extractor for Experiment 2 journals.

The run journals are up to 2.4 GB each (llm_call.request/response_text and
branch_archive.messages dominate).  Nothing here parses those payloads: every
line is classified from its first 240 bytes and giant records are truncated
before ``json.loads``.  One pass per file, constant memory.

Emits one compact dict per journal into bench/results/exp2_extract.json so the
analysis (bench/analyze_exp2.py) never has to touch the raw journals again.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KIND = re.compile(rb'"kind": "([A-Za-z_]+)"')
NAME = re.compile(rb'"name": "([A-Za-z_0-9]+)"')

# fields that live BEFORE the fat payload keys of the two heavy record kinds
TRUNC_AT = {
    b"llm_call": b', "request":',
    b"branch_archive": b', "messages":',
}


def _load(line: bytes, kind: str) -> dict | None:
    cut = TRUNC_AT.get(kind.encode())
    if cut is not None:
        i = line.find(cut)
        if i > 0:
            line = line[:i] + b"}"
    try:
        return json.loads(line)
    except Exception:
        return None


def extract(path: Path) -> dict:
    out: dict = {
        "journal": str(path),
        "kind_counts": {},
        "events": [],
        "fork_waves": [],
        "branch_selections": [],
        "probes": [],
        "incidents": [],
        "hint_assignments": [],
        "infra": {},            # bucket -> {op -> [n, total_s, fails]}
        "infra_ops": [],        # per-op summary rows (op, bucket, n, p50, p95, max, fails)
        "llm": {},
        "steps": {},            # branch -> {n, ticks, errors, exec_s}
        "step_trail": [],       # (step, branch) in order, for matched-step reads
        "t_start_ts": None,
        "first_ts": None,
        "last_ts": None,
    }
    kc: Counter = Counter()
    infra_raw: dict[tuple[str, str], list] = defaultdict(list)
    infra_fail: Counter = Counter()
    llm_lat: list[float] = []
    llm_out: Counter = Counter()
    llm_tok = [0, 0, 0]  # prompt, completion, reasoning
    llm_attempts: Counter = Counter()
    steps: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "ticks": 0, "errors": 0, "exec_s": 0.0,
                 "prod_last": None, "auto_last": None, "noop": 0}
    )
    with open(path, "rb") as fh:
        for line in fh:
            m = KIND.search(line[:240])
            if not m:
                continue
            kind = m.group(1).decode()
            kc[kind] += 1
            if kind == "step":
                rec = _load(line, kind)
                if rec is None:
                    continue
                b = rec.get("branch") or ""
                s = steps[b]
                s["n"] += 1
                s["ticks"] += int(rec.get("ticks") or 0)
                s["errors"] += 1 if rec.get("error") else 0
                s["exec_s"] += float(rec.get("exec_s") or 0.0)
                s["prod_last"] = rec.get("production_score")
                s["auto_last"] = rec.get("automated_score")
                if not (rec.get("ticks") or 0) and not (rec.get("code_chars") or 0):
                    s["noop"] += 1
                out["step_trail"].append(
                    [rec.get("step"), b, round(float(rec.get("exec_s") or 0.0), 3),
                     int(rec.get("ticks") or 0), bool(rec.get("error"))]
                )
                continue
            if kind == "llm_call":
                rec = _load(line, kind)
                if rec is None:
                    continue
                llm_lat.append(float(rec.get("latency_s") or 0.0))
                llm_out[rec.get("outcome") or "?"] += 1
                llm_attempts[int(rec.get("attempt") or 0)] += 1
                llm_tok[0] += int(rec.get("prompt_tokens") or 0)
                llm_tok[1] += int(rec.get("completion_tokens") or 0)
                llm_tok[2] += int(rec.get("reasoning_tokens") or 0)
                continue
            if kind == "branch_archive":
                continue
            rec = _load(line, kind)
            if rec is None:
                continue
            ts = rec.get("ts")
            if ts:
                if out["first_ts"] is None:
                    out["first_ts"] = ts
                out["last_ts"] = ts
            if kind == "infra_op":
                key = (rec.get("bucket") or "?", rec.get("op") or "?")
                infra_raw[key].append(float(rec.get("duration_s") or 0.0))
                if (rec.get("outcome") or "ok") != "ok":
                    infra_fail[key] += 1
            elif kind == "probe":
                out["probes"].append({
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
                out["branch_selections"].append({
                    "ts": ts, "round": rec.get("round"),
                    "winner": rec.get("winner"),
                    "k_effective": rec.get("k_effective"),
                    "scores": rec.get("scores") or {},
                })
            elif kind == "incident":
                out["incidents"].append({
                    "ts": ts, "incident_kind": rec.get("incident_kind"),
                    "detail": (rec.get("detail") or "")[:300],
                    "branch": rec.get("branch"), "step": rec.get("step"),
                })
            elif kind == "hint_assignment":
                out["hint_assignments"].append({
                    "round": rec.get("round"),
                    "seats": sorted((rec.get("hints") or {}).keys()),
                    "hint_chars": {k: len(v) for k, v in (rec.get("hints") or {}).items()},
                })
            elif kind == "event":
                nm = rec.get("name")
                if nm == "fork_wave":
                    out["fork_waves"].append({
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
                    out["events"].append(slim)
                if nm == "T_start":
                    out["t_start_ts"] = ts

    def pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return round(s[i], 3)

    rows = []
    for (bucket, op), vals in sorted(infra_raw.items()):
        rows.append({
            "bucket": bucket, "op": op, "n": len(vals),
            "total_s": round(sum(vals), 3), "p50": pct(vals, 0.5),
            "p95": pct(vals, 0.95), "max": round(max(vals), 3),
            "fails": infra_fail[(bucket, op)],
        })
    out["infra_ops"] = rows
    out["infra"] = {}
    for r in rows:
        out["infra"].setdefault(r["bucket"], {"n": 0, "total_s": 0.0})
        out["infra"][r["bucket"]]["n"] += r["n"]
        out["infra"][r["bucket"]]["total_s"] = round(
            out["infra"][r["bucket"]]["total_s"] + r["total_s"], 3)
    out["llm"] = {
        "n": len(llm_lat), "total_s": round(sum(llm_lat), 3),
        "mean_s": round(sum(llm_lat) / len(llm_lat), 3) if llm_lat else 0.0,
        "p50": pct(llm_lat, 0.5), "p95": pct(llm_lat, 0.95),
        "max": round(max(llm_lat), 3) if llm_lat else 0.0,
        "outcomes": dict(llm_out), "attempts": {str(k): v for k, v in sorted(llm_attempts.items())},
        "prompt_tokens": llm_tok[0], "completion_tokens": llm_tok[1],
        "reasoning_tokens": llm_tok[2],
    }
    out["steps"] = {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()} for k, v in sorted(steps.items())}
    out["kind_counts"] = dict(kc)
    return out


def main(argv: list[str]) -> int:
    jdir = Path("bench/journal/exp2")
    pats = argv[1:] or ["codex", "k3"]
    files = sorted(
        p for p in jdir.glob("*.jsonl")
        if not p.name.endswith("-farplane.jsonl")
        and p.name not in ("tier1-master.jsonl", "reaper.jsonl")
        and any(s in p.name for s in pats)
    )
    res = {}
    for p in files:
        sz = p.stat().st_size
        print(f"[extract] {p.name} ({sz/1e6:.0f} MB)", flush=True)
        res[p.stem] = extract(p)
        print(f"          kinds={res[p.stem]['kind_counts']}", flush=True)
    outp = Path("bench/results/exp2_extract.json")
    outp.write_text(json.dumps(res, indent=1, default=str))
    print(f"[extract] wrote {outp} ({outp.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
