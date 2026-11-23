"""Model-facing tool definitions, argument validation and guardrails.

Everything the model asks for goes through ToolExecutor.execute(). It
  1. validates the arguments (pydantic, unknown fields rejected),
  2. enforces policy in code (evidence, duplicates, grounded numbers,
     certification checks, human approval for overwrites),
  3. calls traccia_store.call_tool() with retries where that makes sense,
  4. logs every call (including blocked ones) to the trace,
and returns a JSON-able dict that goes back to the model. Errors are returned
to the model as data so it can correct itself; they never crash the loop.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import traccia_store as ts

# ---------------------------------------------------------------------------
# What the model sees. Descriptions are written for the model, not copied
# from the store: they say when to use each tool and when not to.
# request_confirmation is not exposed: the harness runs the approval step
# itself when the model asks for overwrite_achievement.
# ---------------------------------------------------------------------------

_EVIDENCE = {"type": "array", "items": {"type": "string"},
             "description": "Ids of the professional's own messages (e.g. ['M4','M6']) that directly "
                            "state this. Never cite assistant messages."}

TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {"name": "search_profile",
     "description": "Keyword search over existing achievements and profile facts. Keyword-only (no "
                    "synonyms), so try 2-3 short phrasings of the same thing. Required before any write.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_achievement",
     "description": "Read one achievement in full. Use before enriching it or before claiming new work "
                    "is different from it.",
     "parameters": {"type": "object", "properties": {"achievement_id": {"type": "string"}},
                    "required": ["achievement_id"]}},
    {"name": "propose_achievement",
     "description": "Create a NEW achievement. Only for work not already on the profile; if a similar "
                    "achievement exists, use overwrite_achievement to enrich it instead.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string", "description": "Short factual title."},
         "description": {"type": "string", "description": "One or two sentences, in the professional's own terms."},
         "contribution": {"type": "string", "description": "What the professional personally did."},
         "skills": {"type": "array", "items": {"type": "string"}},
         "outcome": {"type": "string", "description": "Measured result, numbers exactly as stated."},
         "period": {"type": "string", "description": "When, as stated (e.g. 'November 2025')."},
         "evidence_message_ids": _EVIDENCE,
         "not_duplicate_of": {"type": "array", "items": {"type": "string"},
                              "description": "Ids of similar existing achievements you read with "
                                             "get_achievement and confirmed are different work."}},
         "required": ["title", "description", "evidence_message_ids"]}},
    {"name": "update_profile_field",
     "description": "Add one skill, role, certification, project or career_event. Only for specific, "
                    "confirmed facts not already on the profile in any wording. Certifications need an "
                    "exact name confirmed by verify_certification.",
     "parameters": {"type": "object", "properties": {
         "fact_type": {"type": "string", "enum": sorted(ts.VALID_FACT_TYPES)},
         "value": {"type": "string", "description": "Short canonical value, e.g. 'Process mapping'."},
         "evidence_message_ids": _EVIDENCE},
         "required": ["fact_type", "value", "evidence_message_ids"]}},
    {"name": "overwrite_achievement",
     "description": "Add detail to an EXISTING achievement. Send only the fields you are filling in or "
                    "extending. A human reviewer must approve: filling empty fields or extending existing "
                    "text is auto-approved; changing or removing existing content is queued for a person.",
     "parameters": {"type": "object", "properties": {
         "achievement_id": {"type": "string"},
         "title": {"type": "string"},
         "description": {"type": "string"},
         "contribution": {"type": "string"},
         "skills": {"type": "array", "items": {"type": "string"}},
         "outcome": {"type": "string"},
         "period": {"type": "string"},
         "evidence_message_ids": _EVIDENCE},
         "required": ["achievement_id", "evidence_message_ids"]}},
    {"name": "verify_certification",
     "description": "Look a certification up in an external registry. Unreliable (the harness retries). "
                    "Several matches means the name is ambiguous: ask, do not pick one.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "flag_for_human_review",
     "description": "Park a claim for a person instead of writing it: contradictions with earlier "
                    "evidence, overstatements, anything you are unsure about.",
     "parameters": {"type": "object", "properties": {
         "reason": {"type": "string", "description": "What the conflict is, quoting both sides briefly."},
         "refs": {"type": "array", "items": {"type": "string"},
                  "description": "Message ids and record ids involved (e.g. ['M13','M11','A-001'])."}},
         "required": ["reason", "refs"]}},
    {"name": "ask_followup",
     "description": "Ask the professional ONE question about the most important missing detail. Call it "
                    "last; after it no more tools run.",
     "parameters": {"type": "object", "properties": {
         "question": {"type": "string"}, "about": {"type": "string"}}, "required": ["question"]}},
]

WRITE_TOOLS = {"propose_achievement", "update_profile_field", "overwrite_achievement"}

# ---------------------------------------------------------------------------
# Argument models (extra fields rejected so invented parameters surface)
# ---------------------------------------------------------------------------


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchArgs(_Args):
    query: str = Field(min_length=1)


class GetArgs(_Args):
    achievement_id: str = Field(pattern=r"^A-\d{3}$")


class ProposeArgs(_Args):
    title: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=3)
    contribution: Optional[str] = None
    skills: List[str] = []
    outcome: Optional[str] = None
    period: Optional[str] = None
    evidence_message_ids: List[str] = Field(min_length=1)
    not_duplicate_of: List[str] = []


class FieldArgs(_Args):
    fact_type: Literal["skill", "role", "certification", "project", "career_event"]
    value: str = Field(min_length=2, max_length=100)
    evidence_message_ids: List[str] = Field(min_length=1)


class OverwriteArgs(_Args):
    achievement_id: str = Field(pattern=r"^A-\d{3}$")
    title: Optional[str] = None
    description: Optional[str] = None
    contribution: Optional[str] = None
    skills: Optional[List[str]] = None
    outcome: Optional[str] = None
    period: Optional[str] = None
    evidence_message_ids: List[str] = Field(min_length=1)


class VerifyArgs(_Args):
    name: str = Field(min_length=2)


class FlagArgs(_Args):
    reason: str = Field(min_length=5)
    refs: List[str] = Field(min_length=1)


class AskArgs(_Args):
    question: str = Field(min_length=5)
    about: str = ""

    @field_validator("question")
    @classmethod
    def one_question(cls, v: str) -> str:
        if v.count("?") > 2:
            raise ValueError("ask one focused question, not a list")
        return v


ARG_MODELS = {"search_profile": SearchArgs, "get_achievement": GetArgs,
              "propose_achievement": ProposeArgs, "update_profile_field": FieldArgs,
              "overwrite_achievement": OverwriteArgs, "verify_certification": VerifyArgs,
              "flag_for_human_review": FlagArgs, "ask_followup": AskArgs}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "i", "our", "my", "we",
         "was", "is", "it", "as", "by", "at", "from", "this", "that", "work", "process"}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def tokens(text: str) -> set:
    return {t for t in norm(text).split() if t not in _STOP and len(t) > 1}


def numbers_in(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", "")))


class PolicyBlock(Exception):
    """A guardrail refused the call. The message is returned to the model."""


class Transcript:
    """All messages the agent may cite: earlier conversations plus the current one."""

    def __init__(self) -> None:
        self.messages: Dict[str, Dict[str, Any]] = {}
        self.current_ids: List[str] = []

    def add(self, msgs: List[Dict[str, Any]], current: bool) -> None:
        for m in msgs:
            self.messages[m["id"]] = m
            if current:
                self.current_ids.append(m["id"])

    def text_of(self, ids: List[str]) -> str:
        return " ".join(self.messages[i]["text"] for i in ids if i in self.messages)


# ---------------------------------------------------------------------------
# Approval for destructive actions
# ---------------------------------------------------------------------------


class Approver:
    """Stands in for the human who approves overwrite_achievement.

    policy="additive"    approve only changes that fill empty fields or extend existing
                         text/lists without removing anything; everything else waits
                         for a person (queued for review, NOT written).
    policy="never"       never approve; every overwrite is queued.
    policy="interactive" ask on the terminal (chat demo).
    """

    def __init__(self, policy: str = "additive", ask: Callable[[str], str] = input):
        assert policy in {"additive", "never", "interactive"}
        self.policy = policy
        self.ask = ask

    @staticmethod
    def diff(existing: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        additive, changing = [], []
        for k, new in payload.items():
            if k == "evidence_message_ids":
                continue
            old = existing.get(k)
            if old in (None, "", []) or old == new:
                additive.append(k)
            elif isinstance(old, list) and isinstance(new, list) and \
                    {norm(x) for x in old} <= {norm(x) for x in new}:
                additive.append(k)
            elif isinstance(old, str) and isinstance(new, str) and norm(old) in norm(new):
                additive.append(k)
            else:
                changing.append(k)
        return additive, changing

    def decide(self, existing: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, str]:
        additive, changing = self.diff(existing, payload)
        if self.policy == "never":
            return False, "policy=never: all overwrites wait for a person"
        if self.policy == "interactive":
            lines = [f"  {k}: {existing.get(k)!r} -> {payload[k]!r}" for k in payload
                     if k != "evidence_message_ids"]
            ans = self.ask(f"[approval] overwrite {existing['achievement_id']}?\n" + "\n".join(lines)
                           + "\napprove? [y/N] ")
            return ans.strip().lower().startswith("y"), "interactive reviewer decision"
        if changing:
            return False, f"changes existing content in {changing}; needs a person"
        return True, f"additive only ({additive}); auto-approved by simulated reviewer"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    def __init__(self, transcript: Transcript, trace, approver: Approver,
                 on_call: Optional[Callable[[Dict[str, Any]], None]] = None,
                 verify_attempts: int = 3, max_tool_calls: int = 30):
        self.t = transcript
        self.trace = trace
        self.approver = approver
        self.on_call = on_call or (lambda entry: None)
        self.verify_attempts = verify_attempts
        self.max_tool_calls = max_tool_calls
        self.reset_turn()
        self.verified: Dict[str, Dict[str, Any]] = {}   # norm(name) -> registry result
        self.verify_failed: Dict[str, str] = {}
        self.read_ids: set = set()
        self.done_writes: set = set()

    def reset_turn(self) -> None:
        """Per-run (batch) or per-user-message (chat) counters."""
        self.searched = False
        self.asked = False
        self.calls_this_turn = 0
        self.cache: Dict[str, Dict[str, Any]] = {}

    # -- entry point -------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls_this_turn += 1
        if self.calls_this_turn > self.max_tool_calls:
            return self._log(name, args, None, error="BudgetExceeded: too many tool calls this turn; "
                                                      "stop calling tools and reply.", blocked=True)
        if self.asked:
            return self._log(name, args, None, error="Terminal: ask_followup already ended this "
                                                      "turn; no more tool calls.", blocked=True)
        if name not in ARG_MODELS:
            return self._log(name, args, None, error=f"UnknownTool: {name!r}", blocked=True)
        try:
            parsed = ARG_MODELS[name](**(args or {}))
        except (ValidationError, TypeError) as e:
            detail = e.errors(include_url=False) if isinstance(e, ValidationError) else str(e)
            return self._log(name, args, None, error=f"InvalidArguments: {detail}", blocked=True)

        key = name + json.dumps(args, sort_keys=True, default=str)
        if name in ("search_profile", "get_achievement") and key in self.cache:
            return {**self.cache[key], "note": "same call already made; cached result"}
        if name in WRITE_TOOLS | {"flag_for_human_review"} and key in self.done_writes:
            return self._log(name, args, None, error="Duplicate: identical write already done.",
                             blocked=True)
        try:
            result = getattr(self, "_" + name)(parsed, args)
        except PolicyBlock as e:
            return self._log(name, args, None, error=f"BlockedByPolicy: {e}", blocked=True)
        if name in ("search_profile", "get_achievement") and "error" not in result:
            self.cache[key] = result
        if name in WRITE_TOOLS | {"flag_for_human_review"} and "error" not in result:
            self.done_writes.add(key)
        return result

    # -- store access with logging ------------------------------------------
    def _store(self, tool: str, arguments: Dict[str, Any], model_args: Optional[Dict[str, Any]],
               by: str = "model", retries_allowed: int = 0) -> Dict[str, Any]:
        errors: List[str] = []
        for attempt in range(retries_allowed + 1):
            try:
                result = ts.call_tool(tool, arguments)
                return self._log(tool, model_args, arguments, result=result, retries=attempt,
                                 attempt_errors=errors, by=by)
            except ts.ExternalServiceError as e:
                errors.append(str(e))
                if attempt < retries_allowed:
                    time.sleep(0.3 * (2 ** attempt))
                    continue
                return self._log(tool, model_args, arguments, error=f"ExternalServiceError: {e}",
                                 retries=attempt, attempt_errors=errors, by=by)
            except ts.ToolError as e:
                return self._log(tool, model_args, arguments, error=f"{type(e).__name__}: {e}",
                                 retries=attempt, by=by)
            except Exception as e:  # bad kwargs etc. should not kill the run
                return self._log(tool, model_args, arguments, error=f"{type(e).__name__}: {e}",
                                 retries=attempt, by=by)
        raise AssertionError("unreachable")

    def _log(self, tool, model_args, store_args, result=None, error=None, retries=0,
             attempt_errors=None, by="model", blocked=False) -> Dict[str, Any]:
        entry = self.trace.add_call(tool=tool, model_args=model_args, store_args=store_args,
                                    result=result, error=error, retries=retries,
                                    attempt_errors=attempt_errors or [], by=by, blocked=blocked)
        self.on_call({**entry, "result": result})  # console gets the untruncated result
        return result if error is None else {"error": error}

    # -- guards ---------------------------------------------------------------
    def _check_evidence(self, ids: List[str]) -> None:
        bad = [i for i in ids if i not in self.t.messages]
        if bad:
            raise PolicyBlock(f"unknown message ids {bad}; cite ids from the conversation")
        not_prof = [i for i in ids if self.t.messages[i]["speaker"] != "professional"]
        if not_prof:
            raise PolicyBlock(f"{not_prof} are assistant messages; only the professional's own "
                              "messages count as evidence")
        if not set(ids) & set(self.t.current_ids):
            raise PolicyBlock("cites only earlier, already-processed conversations; writes must "
                              "be based on something said in the new conversation")

    def _check_numbers(self, fields: Dict[str, Any], ids: List[str], extra_text: str = "") -> None:
        source = numbers_in(self.t.text_of(ids) + " " + extra_text)
        for k, v in fields.items():
            text = " ".join(v) if isinstance(v, list) else str(v or "")
            missing = numbers_in(text) - source
            if missing:
                raise PolicyBlock(f"field {k!r} contains {sorted(missing)} which do not appear in the "
                                  f"cited messages {ids}. Use numbers exactly as stated; do not compute "
                                  "or round them.")

    def _require_search(self) -> None:
        if not self.searched:
            raise PolicyBlock("search_profile first to check this is not already on the profile")

    # -- handlers ------------------------------------------------------------
    def _search_profile(self, a: SearchArgs, raw) -> Dict[str, Any]:
        self.searched = True
        return self._store("search_profile", {"query": a.query}, raw)

    def _get_achievement(self, a: GetArgs, raw) -> Dict[str, Any]:
        res = self._store("get_achievement", {"achievement_id": a.achievement_id}, raw)
        if "error" not in res:
            self.read_ids.add(a.achievement_id)
        return res

    def _propose_achievement(self, a: ProposeArgs, raw) -> Dict[str, Any]:
        self._require_search()
        self._check_evidence(a.evidence_message_ids)
        fields = a.model_dump(exclude={"evidence_message_ids", "not_duplicate_of"}, exclude_none=True)
        self._check_numbers(fields, a.evidence_message_ids)
        new_tok = tokens(a.title + " " + a.description)
        for ex in ts.STORE.achievements.values():
            ex_tok = tokens(ex["title"] + " " + ex["description"])
            shared = new_tok & ex_tok
            if len(shared) >= 2 and len(shared) / max(1, min(len(new_tok), len(ex_tok))) >= 0.4:
                if ex["achievement_id"] in a.not_duplicate_of and ex["achievement_id"] in self.read_ids:
                    continue
                raise PolicyBlock(
                    f"looks like existing {ex['achievement_id']} '{ex['title']}' (shared words: "
                    f"{sorted(shared)}). If it is the same work, enrich it with overwrite_achievement. "
                    "If it is genuinely different, read it with get_achievement and pass "
                    f"not_duplicate_of=['{ex['achievement_id']}'].")
        payload = {**fields, "evidence_message_ids": a.evidence_message_ids, "status": "supported"}
        return self._store("propose_achievement", {"payload": payload}, raw)

    def _update_profile_field(self, a: FieldArgs, raw) -> Dict[str, Any]:
        self._require_search()
        self._check_evidence(a.evidence_message_ids)
        self._check_numbers({"value": a.value}, a.evidence_message_ids)
        nv = norm(a.value)
        for f in ts.STORE.profile_facts.values():
            if f["fact_type"] != a.fact_type:
                continue
            ev = norm(f["value"])
            if nv == ev or (len(ev) > 3 and ev in nv) or (len(nv) > 3 and nv in ev):
                raise PolicyBlock(f"already on the profile as {f['fact_id']} {f['value']!r}")
            shared = tokens(a.value) & tokens(f["value"])
            if a.fact_type == "skill" and shared:
                raise PolicyBlock(f"near-duplicate of existing skill {f['fact_id']} {f['value']!r} "
                                  f"(shared: {sorted(shared)}); skip it")
        if a.fact_type == "certification":
            hit = self.verified.get(nv)
            canonical = [m["canonical_name"] for r in self.verified.values() for m in r["matches"]
                         if r["match_count"] == 1]
            if not any(norm(c) == nv for c in canonical):
                why = ("the registry was unavailable" if self.verify_failed else
                       "it has not been verified to exactly one registry match" if not hit else
                       f"the registry returned {hit['match_count']} matches")
                raise PolicyBlock(f"cannot record certification {a.value!r}: {why}. Ask the "
                                  "professional for the exact certification name instead.")
        args = {"fact_type": a.fact_type, "value": a.value,
                "evidence_message_ids": a.evidence_message_ids, "status": "supported"}
        return self._store("update_profile_field", args, raw)

    def _overwrite_achievement(self, a: OverwriteArgs, raw) -> Dict[str, Any]:
        existing = ts.STORE.achievements.get(a.achievement_id)
        if existing is None:
            raise PolicyBlock(f"no achievement {a.achievement_id}")
        self._check_evidence(a.evidence_message_ids)
        fields = a.model_dump(exclude={"achievement_id", "evidence_message_ids"}, exclude_none=True)
        if not fields:
            raise PolicyBlock("nothing to change: pass at least one field")
        existing_text = json.dumps({k: existing.get(k) for k in fields})
        self._check_numbers(fields, a.evidence_message_ids, extra_text=existing_text)
        payload = dict(fields)
        payload["evidence_message_ids"] = list(dict.fromkeys(
            existing.get("evidence_message_ids", []) + a.evidence_message_ids))
        if "skills" in payload:  # never drop skills already listed
            payload["skills"] = list(dict.fromkeys((existing.get("skills") or []) + payload["skills"]))

        conf = self._store("request_confirmation",
                           {"action": "overwrite_achievement", "target_id": a.achievement_id,
                            "reason": f"set {sorted(fields)} from {a.evidence_message_ids}"},
                           None, by="harness")
        if "error" in conf:
            return conf
        token = conf["confirmation_token"]
        approved, why = self.approver.decide(existing, payload)
        if approved:
            ts.approve_token(token)
            self.trace.note(f"reviewer approved {token} for {a.achievement_id}: {why}")
            return self._store("overwrite_achievement", {"achievement_id": a.achievement_id,
                                                         "payload": payload,
                                                         "confirmation_token": token}, raw)
        self.trace.note(f"reviewer did not approve {token} for {a.achievement_id}: {why}")

        # The additive policy splits a mixed request: the purely additive fields are a
        # separate, approvable change; the fields that alter existing content wait for a person.
        additive, changing = self.approver.diff(existing, payload)
        applied = None
        if self.approver.policy == "additive" and additive:
            sub = {k: payload[k] for k in additive}
            sub["evidence_message_ids"] = payload["evidence_message_ids"]
            conf2 = self._store("request_confirmation",
                                {"action": "overwrite_achievement", "target_id": a.achievement_id,
                                 "reason": f"additive subset {sorted(additive)}"}, None, by="harness")
            if "error" not in conf2:
                ts.approve_token(conf2["confirmation_token"])
                self.trace.note(f"reviewer approved additive subset {sorted(additive)} "
                                f"({conf2['confirmation_token']}) for {a.achievement_id}")
                applied = self._store("overwrite_achievement",
                                      {"achievement_id": a.achievement_id, "payload": sub,
                                       "confirmation_token": conf2["confirmation_token"]}, raw)
        pending = {k: payload[k] for k in (changing if applied else payload)
                   if k != "evidence_message_ids"}
        flag = self._store("flag_for_human_review",
                           {"reason": f"Proposed change to {a.achievement_id} awaiting approval "
                                      f"(token {token}; {why}): {json.dumps(pending)}",
                            "refs": [a.achievement_id, token] + a.evidence_message_ids},
                           None, by="harness")
        if applied is None:
            # Record the model's request itself, unexecuted, so the trace shows it.
            self._log("overwrite_achievement", raw, None,
                      error=f"NotApproved: {why}; queued as {flag.get('review_id')}", blocked=True)
        return {"written": sorted(additive) if applied and "error" not in applied else [],
                "pending_review": sorted(pending), "queued_for_review": flag.get("review_id"),
                "reason": why,
                "guidance": "Do not retry the pending fields. Tell the professional they await review."}

    def _verify_certification(self, a: VerifyArgs, raw) -> Dict[str, Any]:
        key = norm(a.name)
        if key in self.verified:
            return {**self.verified[key], "note": "already verified this run; cached"}
        if key in self.verify_failed:
            return {"error": self.verify_failed[key], "note": "already failed this run; not retrying"}
        res = self._store("verify_certification", {"name": a.name}, raw,
                          retries_allowed=self.verify_attempts - 1)
        if "error" in res:
            self.verify_failed[key] = res["error"]
            return {**res, "guidance": f"Registry unavailable after {self.verify_attempts} attempts. "
                                       "Do not record the certification; ask for the exact name."}
        self.verified[key] = res
        if res["match_count"] != 1:
            res = {**res, "guidance": "Not exactly one match: do not record it. Ask which one."}
        return res

    def _flag_for_human_review(self, a: FlagArgs, raw) -> Dict[str, Any]:
        known = set(self.t.messages) | set(ts.STORE.achievements) | set(ts.STORE.profile_facts) \
            | {r["review_id"] for r in ts.STORE.review_queue}
        bad = [r for r in a.refs if r not in known]
        if bad:
            raise PolicyBlock(f"unknown refs {bad}; use message ids or record ids")
        return self._store("flag_for_human_review", {"reason": a.reason, "refs": a.refs}, raw)

    def _ask_followup(self, a: AskArgs, raw) -> Dict[str, Any]:
        if self.asked:
            raise PolicyBlock("only one follow-up question per turn; it was already asked")
        q = tokens(a.question)
        for prev in ts.STORE.followups:
            p = tokens(prev["question"])
            if q and p and len(q & p) / min(len(q), len(p)) >= 0.5:
                raise PolicyBlock(f"already asked earlier: {prev['question']!r}. Do not ask again "
                                  "until the professional answers it.")
        res = self._store("ask_followup", {"question": a.question, "about": a.about}, raw)
        if "error" not in res:
            self.asked = True
        return res
