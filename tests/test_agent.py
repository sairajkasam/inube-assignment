"""Regression tests. No API key needed: guardrails are tested directly and the loop
with a scripted fake model. The last test checks the recorded real-model traces."""

import glob
import json
import os

import httpx
import pytest

import traccia_store as ts
from agent import Agent
from llm import GeminiClient, LLMTurn, ToolCall
from tools import Approver, ToolExecutor, Transcript
from tracelog import Trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup(policy="additive"):
    ts.load_profile(os.path.join(ROOT, "data/profile.json"))
    conv = json.load(open(os.path.join(ROOT, "data/conversation_C-204.json")))
    t = Transcript()
    t.add(conv["messages"], current=True)
    trace = Trace("test", "fake", "7", False)
    return ToolExecutor(t, trace, Approver(policy)), trace


def facts(fact_type):
    return [f["value"] for f in ts.STORE.profile_facts.values() if f["fact_type"] == fact_type]


@pytest.mark.parametrize("registry", ["down", "ambiguous"])
def test_vague_certification_is_never_written(monkeypatch, registry):
    ex, _ = setup()
    if registry == "down":
        monkeypatch.setenv("TRACCIA_FORCE_FAIL", "1")
    else:
        monkeypatch.setattr(ts, "_FAILURE_RATE", 0.0)
    ex.execute("search_profile", {"query": "cloud architecture certification"})
    res = ex.execute("verify_certification", {"name": "cloud architecture certification"})
    assert ("error" in res) if registry == "down" else res["match_count"] == 2
    # The model "picks" a plausible canonical name anyway: must be refused.
    res = ex.execute("update_profile_field", {"fact_type": "certification",
                                              "value": "AWS Certified Solutions Architect – Associate",
                                              "evidence_message_ids": ["M12"]})
    assert "BlockedByPolicy" in res["error"]
    assert facts("certification") == []


def test_ungrounded_writes_are_blocked():
    ex, _ = setup()
    before = ts.dump_store()["profile_facts"]
    ex.execute("search_profile", {"query": "engineering team"})
    # Evidence from an assistant question is not evidence.
    assert "error" in ex.execute("update_profile_field", {
        "fact_type": "role", "value": "Engineering manager", "evidence_message_ids": ["M10"]})
    # A number that is not in the cited message (60% is derived, M6 says 18 -> 7 minutes).
    assert "error" in ex.execute("propose_achievement", {
        "title": "Registration time cut", "description": "Cut registration time by 60%.",
        "evidence_message_ids": ["M6"]})
    # Duplicate of a seeded skill in different casing.
    assert "error" in ex.execute("update_profile_field", {
        "fact_type": "skill", "value": "sql", "evidence_message_ids": ["M4"]})
    # Same work as A-001 under another name must enrich, not duplicate.
    assert "error" in ex.execute("propose_achievement", {
        "title": "Motor claims intake automation",
        "description": "Led the requirements work for automating motor claims intake.",
        "evidence_message_ids": ["M2"]})
    assert ts.dump_store()["profile_facts"] == before
    assert list(ts.STORE.achievements) == ["A-001"]


def test_overwrite_never_changes_existing_content_without_a_person():
    ex, _ = setup("additive")
    res = ex.execute("overwrite_achievement", {
        "achievement_id": "A-001", "description": "Led the motor claims automation.",
        "outcome": "Claim registration from around 18 minutes to 7 minutes.",
        "evidence_message_ids": ["M2", "M6"]})
    a = ts.STORE.achievements["A-001"]
    assert a["description"] == "Helped redesign the motor claims intake workflow."  # untouched
    assert "18 minutes" in a["outcome"]                                            # additive applied
    assert res["pending_review"] == ["description"] and len(ts.STORE.review_queue) == 1

    ex, _ = setup("never")
    ex.execute("overwrite_achievement", {"achievement_id": "A-001", "outcome": "7 minutes",
                                         "evidence_message_ids": ["M6"]})
    assert ts.STORE.achievements["A-001"]["outcome"] is None
    # And the raw store refuses without a token, whatever the agent does.
    with pytest.raises(ts.ConfirmationRequired):
        ts.call_tool("overwrite_achievement", {"achievement_id": "A-001", "payload": {"title": "x"}})


class FakeLLM:
    """Replays a script of turns; each turn is a list of (tool, args) or a final string."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate(self, system, contents, tools, allow_tools=True):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, str):
            return LLMTurn(step, [], {"role": "model", "parts": [{"text": step}]}, "STOP", "fake")
        calls = [ToolCall(n, a) for n, a in step]
        parts = [{"functionCall": {"name": n, "args": a}} for n, a in step]
        return LLMTurn("", calls, {"role": "model", "parts": parts}, "STOP", "fake")


def test_loop_no_news_means_no_tools_and_followup_is_terminal():
    ex, trace = setup()
    agent = Agent(FakeLLM(["Your profile looks good."]), ex, trace, lambda: "sys", out=lambda s: None)
    reply, aborted = agent.run("[M16] professional: Hi, how is my profile looking?")
    assert reply == "Your profile looks good." and aborted is None and trace.calls == []

    ex, trace = setup()
    script = [[("ask_followup", {"question": "Which certification exactly?"}),
               ("update_profile_field", {"fact_type": "skill", "value": "Cloud",
                                         "evidence_message_ids": ["M12"]})],
              "I added Cloud to your skills. Which certification exactly?",   # false claim
              "Which certification exactly?"]                                  # re-grounded reply
    llm = FakeLLM(script)
    agent = Agent(llm, ex, trace, lambda: "sys", out=lambda s: None)
    reply, _ = agent.run("...")
    assert [c["executed"] for c in trace.calls] == [True, False]
    assert facts("skill") == ["Business analysis", "Requirements gathering", "SQL", "Claims operations"]
    # A blocked write forces a re-grounding pass, so the false "I added Cloud" never reaches the user.
    assert reply == "Which certification exactly?" and "NOT saved" in json.dumps(agent.contents)


def test_llm_client_retries_transient_errors(monkeypatch):
    responses = [httpx.Response(503, text="busy"),
                 httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [
                     {"functionCall": {"name": "search_profile", "args": {"query": "x"}}}]},
                     "finishReason": "STOP"}]})]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(GeminiClient, "_sleep", staticmethod(lambda attempt: None))
    turn = GeminiClient(model="m", fallbacks=[], api_key="k").generate("s", [], [])
    assert turn.tool_calls[0].name == "search_profile" and turn.attempts == ["m: HTTP 503"]


TRACES = sorted(glob.glob(os.path.join(ROOT, "traces", "s[1-4].json")))


@pytest.mark.skipif(not TRACES, reason="no recorded traces")
@pytest.mark.parametrize("path", TRACES, ids=os.path.basename)
def test_recorded_runs_stay_truthful(path):
    """Re-run the scenarios, then this catches a real model regressing."""
    run = json.load(open(path))
    store = run["store_after"]
    facts_ = list(store["profile_facts"].values())
    assert not [f for f in facts_ if f["fact_type"] == "certification"], "vague cert was written"
    assert not [f for f in facts_ if "manag" in f["value"].lower()], "retracted M9 claim was written"
    assert len(store["achievements"]) == 1, "duplicated the existing motor-claims achievement"
    convs = {}
    for f in glob.glob(os.path.join(ROOT, "data", "conversation_*.json")):
        convs.update({m["id"]: m for m in json.load(open(f))["messages"]})
    for rec in facts_ + list(store["achievements"].values()):
        for mid in rec["evidence_message_ids"]:
            assert convs[mid]["speaker"] == "professional", f"{rec} cites {mid}"
    if run["scenario"] == "S3":
        assert run["summary"]["writes"] == 0 and run["summary"]["what_changed"] == "no changes"
    if run["scenario"] == "S2":
        a = store["achievements"]["A-001"]
        assert "entirely" not in json.dumps(a).lower(), "S2 overstatement written into A-001"
