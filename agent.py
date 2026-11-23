"""Traccia profile agent: reads a conversation and updates the profile via tools.

  python agent.py --profile data/profile.json --conversation data/conversation_C-204.json
  python agent.py --state traces/s1_state.json --conversation data/conversation_C-205.json
  python agent.py --chat [--approver interactive]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError

import traccia_store as ts
from llm import DEFAULT_MODEL, GeminiClient, LLMError
from prompt import build_batch_input, build_system
from tools import TOOL_DECLARATIONS, Approver, ToolExecutor, Transcript
from tracelog import WRITE_TOOLS, Trace, load_state, save_state, what_changed

MAX_STEPS = 12
FALLBACK_REPLY = ("Sorry, I could not finish processing this just now. Nothing beyond what is "
                  "listed in the log was changed on your profile.")


class Message(BaseModel):
    id: str
    speaker: Literal["professional", "assistant"]
    text: str


class Conversation(BaseModel):
    conversation_id: str
    professional_id: str
    messages: List[Message]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class Agent:
    def __init__(self, llm, executor: ToolExecutor, trace: Trace, system_fn: Callable[[], str],
                 out: Callable[[str], None] = print, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.ex = executor
        self.trace = trace
        self.system_fn = system_fn
        self.out = out
        self.max_steps = max_steps
        self.contents: List[Dict[str, Any]] = []

    def _generate(self, allow_tools: bool):
        turn = self.llm.generate(self.system_fn(), self.contents, TOOL_DECLARATIONS,
                                 allow_tools=allow_tools)
        self.trace.add_llm_turn(turn.model, turn.finish_reason, len(turn.tool_calls),
                                turn.attempts, turn.text)
        for a in turn.attempts:
            self.out(f"[llm] retry after: {a}")
        return turn

    def _grounded_reply(self, text: str, before: Dict[str, Any], first_call: int) -> str:
        """If any write failed or was blocked this turn, the model's reply may claim it was
        saved. Give it the authoritative record and have it rewrite the reply (one extra call)."""
        failed = [c for c in self.trace.calls[first_call:]
                  if c["by"] == "model" and c["tool"] in WRITE_TOOLS and c["error"]]
        if not failed:
            return text
        saved = what_changed(before, ts.dump_store())
        not_saved = "; ".join(f"{c['tool']}({json.dumps(c['arguments'])[:160]}) -> {c['error'][:120]}"
                              for c in failed)
        self.trace.note(f"{len(failed)} write(s) failed/blocked; asking model to re-ground its reply")
        self.contents.append({"role": "user", "parts": [{"text":
            "SYSTEM CHECK before your reply is sent. Saved this turn (authoritative): "
            f"{saved}. NOT saved: {not_saved}. Rewrite your final reply so it claims only what "
            "was saved and mentions anything pending review. Keep the same question if you asked one."}]})
        try:
            turn = self._generate(allow_tools=False)
        except LLMError:
            return text + f"\n(Note: only these changes were saved: {saved}.)"
        self.contents.append(turn.raw_content)
        return turn.text or text

    def run(self, user_text: str) -> Tuple[str, Optional[str]]:
        """One agent turn. Returns (final reply, abort reason or None)."""
        self.contents.append({"role": "user", "parts": [{"text": user_text}]})
        before, first_call = ts.dump_store(), len(self.trace.calls)
        empty = 0
        try:
            for _ in range(self.max_steps):
                turn = self._generate(allow_tools=not self.ex.asked)
                if not turn.tool_calls and not turn.text:
                    # Malformed function call, safety block, or empty output: nudge and retry.
                    empty += 1
                    if empty > 2:
                        return FALLBACK_REPLY, f"model gave no usable output ({turn.finish_reason})"
                    self.contents.append({"role": "model", "parts": [{"text": "(no output)"}]})
                    self.contents.append({"role": "user", "parts": [{"text":
                        f"Your last response was empty or malformed ({turn.finish_reason}). Call "
                        "tools with valid arguments, or reply in plain text."}]})
                    continue
                self.contents.append(turn.raw_content)
                if not turn.tool_calls:
                    return self._grounded_reply(turn.text, before, first_call), None
                parts = []
                for tc in turn.tool_calls:
                    result = self.ex.execute(tc.name, tc.args)
                    fr: Dict[str, Any] = {"name": tc.name, "response": result}
                    if tc.id:
                        fr["id"] = tc.id
                    parts.append({"functionResponse": fr})
                self.contents.append({"role": "user", "parts": parts})
            # Step budget used up: force a plain-text answer.
            self.trace.note(f"step budget ({self.max_steps}) exhausted; forcing final reply")
            turn = self._generate(allow_tools=False)
            self.contents.append(turn.raw_content)
            return turn.text or FALLBACK_REPLY, "step budget exhausted"
        except LLMError as e:
            self.trace.note(f"LLM failure: {e}")
            return FALLBACK_REPLY, f"LLM failure: {e}"


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def summarize_result(entry: Dict[str, Any]) -> str:
    if entry["error"]:
        return "BLOCKED: " + entry["error"] if not entry["executed"] else "ERROR: " + entry["error"]
    r = entry["result"]
    if isinstance(r, dict):
        if "result_count" in r:
            return f"{r['result_count']} results " + str([h["id"] for h in r["results"]])
        if "match_count" in r:
            return f"{r['match_count']} matches " + str([m["canonical_name"] for m in r["matches"]])
        for k in ("achievement_id", "fact_id", "review_id", "confirmation_token"):
            if k in r:
                return f"{k}={r[k]}" + (" (overwritten)" if r.get("overwritten") else "")
        if "asked" in r:
            return "asked"
    return str(r)[:120]


def make_printer(trace: Trace) -> Callable[[str], None]:
    def out(line: str) -> None:
        print(line, flush=True)
        trace.console.append(line)
    return out


def make_call_printer(out: Callable[[str], None]) -> Callable[[Dict[str, Any]], None]:
    def on_call(entry: Dict[str, Any]) -> None:
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (entry["arguments"] or {}).items())
        if len(args) > 220:
            args = args[:220] + "..."
        tag = "[tool]" if entry["by"] == "model" else "[harness]"
        retry = f" (retries={entry['retries']})" if entry["retries"] else ""
        out(f"{tag} {entry['tool']}({args}) -> {summarize_result(entry)}{retry}")
    return on_call


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def setup_store(args) -> List[Dict[str, Any]]:
    if args.state:
        return load_state(args.state)
    ts.load_profile(args.profile)
    return []


def make_llm(model: str):
    try:
        return GeminiClient(model=model)
    except LLMError as e:
        sys.exit(f"error: {e}")


def run_batch(args) -> int:
    try:
        with open(args.conversation, encoding="utf-8") as fh:
            conv = Conversation(**json.load(fh))
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        sys.exit(f"error: cannot read conversation {args.conversation}: {e}")
    history = setup_store(args)
    if conv.professional_id not in ts.STORE.professionals:
        sys.exit(f"error: professional {conv.professional_id} not in the loaded profile")
    llm = make_llm(args.model)

    transcript = Transcript()
    for h in history:
        transcript.add(h["messages"], current=False)
    conv_dict = conv.model_dump()
    transcript.add(conv_dict["messages"], current=True)

    scenario = args.scenario or conv.conversation_id
    trace = Trace(scenario, args.model, os.getenv("TRACCIA_SEED", "7"),
                  os.getenv("TRACCIA_FORCE_FAIL") == "1")
    out = make_printer(trace)
    ex = ToolExecutor(transcript, trace, Approver(args.approver), on_call=make_call_printer(out))
    system = build_system(ts.dump_store(), conv.professional_id)  # snapshot at session start
    agent = Agent(llm, ex, trace, lambda: system, out=out)

    out(f"== {scenario}: {conv.conversation_id} ({len(conv.messages)} messages), model={args.model}")
    final, aborted = agent.run(build_batch_input(conv_dict, history))
    out(f"agent> {final}")
    data = trace.to_dict(final, conv.conversation_id, aborted)
    out(f"== what changed: {data['summary']['what_changed']}")

    path = args.trace or os.path.join("traces", f"{conv.conversation_id}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(trace.console) + "\n")
    if args.save_state:
        save_state(args.save_state, history + [conv_dict])
    print(f"trace written to {path}")
    return 1 if aborted else 0


def run_chat(args) -> int:
    history = setup_store(args)
    pid = next(iter(ts.STORE.professionals))
    llm = make_llm(args.model)
    transcript = Transcript()
    for h in history:
        transcript.add(h["messages"], current=False)
    trace = Trace("chat", args.model, os.getenv("TRACCIA_SEED", "7"),
                  os.getenv("TRACCIA_FORCE_FAIL") == "1")
    trace.console.append("(chat session)")
    out = lambda line: (print(line, flush=True), trace.console.append(line))  # noqa: E731
    ex = ToolExecutor(transcript, trace, Approver(args.approver), on_call=make_call_printer(out))
    agent = Agent(llm, ex, trace, lambda: build_system(ts.dump_store(), pid), out=out)

    name = ts.STORE.professionals[pid]["name"]
    print(f"Chatting as {name} ({pid}). Commands: /changes, /store, /quit")
    n = 0
    first = True
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/changes":
            print(trace.to_dict("", "chat")["summary"]["what_changed"])
            continue
        if line == "/store":
            print(json.dumps(ts.dump_store(), indent=2))
            continue
        n += 1
        msg = {"id": f"U{n}", "speaker": "professional", "text": line}
        transcript.add([msg], current=True)
        trace.console.append(f"you> {line}")
        ex.reset_turn()
        user_text = f"[U{n}] professional: {line}"
        if first and history:
            user_text = build_batch_input({"conversation_id": "CHAT", "messages": [msg]}, history)
        first = False
        reply, aborted = agent.run(user_text)
        out(f"agent> {reply}")
        transcript.add([{"id": f"B{n}", "speaker": "assistant", "text": reply}], current=True)

    os.makedirs("traces", exist_ok=True)
    path = os.path.join("traces", f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trace.to_dict("(chat)", "chat"), fh, indent=2, default=str)
    print(f"trace written to {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="data/profile.json")
    p.add_argument("--state", help="start from a saved state (store + earlier transcripts) instead of --profile")
    p.add_argument("--conversation", help="conversation JSON to process (batch mode)")
    p.add_argument("--chat", action="store_true", help="interactive mode")
    p.add_argument("--scenario", help="label written into the trace (e.g. S1)")
    p.add_argument("--trace", help="where to write the trace JSON (default traces/<conversation_id>.json)")
    p.add_argument("--save-state", help="save store + transcripts after the run (to chain scenarios)")
    p.add_argument("--approver", choices=["additive", "never", "interactive"], default="additive",
                   help="who approves overwrite_achievement (default: auto-approve additive changes only)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)
    if args.chat:
        return run_chat(args)
    if not args.conversation:
        p.error("--conversation is required unless --chat is given")
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
