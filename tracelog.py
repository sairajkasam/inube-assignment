"""Run trace: every tool call (model-requested, harness-run and blocked), every
model turn, and a deterministic `what_changed` computed by diffing the store."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import traccia_store as ts

WRITE_TOOLS = {"propose_achievement", "update_profile_field", "overwrite_achievement"}


def _short(value: Any, limit: int = 600) -> Any:
    s = json.dumps(value, default=str)
    return value if len(s) <= limit else s[:limit] + "...<truncated>"


class Trace:
    def __init__(self, scenario: str, model: str, seed: Optional[str], force_fail: bool):
        self.scenario = scenario
        self.model = model
        self.seed = seed
        self.force_fail = force_fail
        self.calls: List[Dict[str, Any]] = []
        self.llm_turns: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.console: List[str] = []
        self.store_before = ts.dump_store()

    def add_call(self, tool, model_args, store_args, result, error, retries, attempt_errors,
                 by, blocked) -> Dict[str, Any]:
        entry = {
            "n": len(self.calls) + 1,
            "tool": tool,
            "by": by,                       # "model" or "harness" (approval / review plumbing)
            "arguments": model_args if model_args is not None else store_args,
            "store_arguments": store_args if store_args != model_args else "same",
            "executed": not blocked,
            "result": _short(result) if result is not None else None,
            "error": error,
            "retries": retries,
        }
        if attempt_errors:
            entry["attempt_errors"] = attempt_errors
        self.calls.append(entry)
        return entry

    def add_llm_turn(self, model: str, finish: str, n_calls: int, failed_attempts: List[str],
                     text: str) -> None:
        self.llm_turns.append({"turn": len(self.llm_turns) + 1, "model": model, "finish_reason": finish,
                               "tool_calls_requested": n_calls, "failed_attempts": failed_attempts,
                               "text": text[:400]})

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def to_dict(self, final_message: str, conversation_id: str, aborted: Optional[str] = None):
        after = ts.dump_store()
        writes = sum(1 for c in self.calls if c["tool"] in WRITE_TOOLS and c["executed"]
                     and not c["error"])
        return {
            "scenario": self.scenario,
            "conversation_id": conversation_id,
            "model": self.model,
            "seed": self.seed,
            "force_fail": self.force_fail,
            "summary": {
                "tool_calls": len(self.calls),
                "model_tool_calls": sum(1 for c in self.calls if c["by"] == "model"),
                "writes": writes,
                "errors": sum(1 for c in self.calls if c["error"]),
                "blocked_by_policy": sum(1 for c in self.calls if not c["executed"]),
                "retries": sum(c["retries"] for c in self.calls),
                "llm_turns": len(self.llm_turns),
                "what_changed": what_changed(self.store_before, after),
            },
            "aborted": aborted,
            "notes": self.notes,
            "calls": self.calls,
            "llm_turns": self.llm_turns,
            "final_message": final_message,
            "store_after": after,
        }


def what_changed(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    parts: List[str] = []
    for aid, rec in after["achievements"].items():
        old = before["achievements"].get(aid)
        if old is None:
            parts.append(f"added achievement {aid} '{rec['title']}'")
        elif old != rec:
            changed = sorted(k for k in rec if rec.get(k) != old.get(k) and k != "source")
            parts.append(f"enriched {aid} ({', '.join(changed)})")
    new_facts: Dict[str, List[str]] = {}
    for fid, f in after["profile_facts"].items():
        if fid not in before["profile_facts"]:
            new_facts.setdefault(f["fact_type"], []).append(f"{f['value']} [{fid}]")
    for ftype, vals in new_facts.items():
        parts.append(f"added {len(vals)} {ftype}(s): {', '.join(vals)}")
    new_reviews = after["review_queue"][len(before["review_queue"]):]
    if new_reviews:
        parts.append(f"queued {len(new_reviews)} item(s) for human review "
                     f"({'; '.join(r['reason'][:90] for r in new_reviews)})")
    new_q = after["followups"][len(before["followups"]):]
    for q in new_q:
        parts.append(f"asked follow-up: \"{q['question']}\"")
    return "; ".join(parts) if parts else "no changes"


# ---------------------------------------------------------------------------
# State persistence (S2/S3 start from S1's end state). load_profile() only
# reads the original profile format, so we restore the in-memory store
# directly. This sets up data; it does not change tool behaviour.
# ---------------------------------------------------------------------------


def save_state(path: str, transcripts: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"store": ts.dump_store(), "transcripts": transcripts}, fh, indent=2)


def load_state(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    s = data["store"]
    for name in ("professionals", "achievements", "profile_facts"):
        getattr(ts.STORE, name).clear()
        getattr(ts.STORE, name).update(s[name])
    ts.STORE.review_queue[:] = s["review_queue"]
    ts.STORE.followups[:] = s["followups"]
    return data.get("transcripts", [])
