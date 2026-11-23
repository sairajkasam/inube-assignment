"""Thin Gemini client over the REST API (no SDK, no automatic function calling).

We call `generateContent` ourselves and hand back the model's tool calls; the
agent loop in agent.py decides what to execute. Transient failures (429, 5xx,
timeouts) are retried with exponential backoff, then we fall back to the next
model in the list.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

DEFAULT_MODEL = os.getenv("TRACCIA_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [m for m in os.getenv("TRACCIA_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3-flash-preview").split(",") if m]


class LLMError(Exception):
    """The model could not be reached after all retries and fallbacks."""


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class LLMTurn:
    text: str
    tool_calls: List[ToolCall]
    raw_content: Dict[str, Any]          # appended verbatim to history (keeps thought signatures)
    finish_reason: str
    model: str
    attempts: List[str] = field(default_factory=list)  # one entry per failed attempt


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, fallbacks: Optional[List[str]] = None,
                 api_key: Optional[str] = None, timeout: float = 90.0, max_attempts: int = 4):
        self.models = [model] + [m for m in (fallbacks if fallbacks is not None else FALLBACK_MODELS)
                                 if m != model]
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.model = model  # model that answered most recently

    def generate(self, system: str, contents: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]], allow_tools: bool = True) -> LLMTurn:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": tools}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO" if allow_tools else "NONE"}},
        }
        failures: List[str] = []
        for model in self.models:
            for attempt in range(self.max_attempts):
                try:
                    resp = httpx.post(API_URL.format(model=model), json=body, timeout=self.timeout,
                                      headers={"x-goog-api-key": self.api_key})
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    failures.append(f"{model}: {type(e).__name__}")
                    self._sleep(attempt)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    delay = self._retry_delay(resp) if resp.status_code == 429 else None
                    failures.append(f"{model}: HTTP {resp.status_code}"
                                    + (f" (server asked to wait {delay:.0f}s)" if delay else ""))
                    if delay is not None and delay > 65:
                        break  # daily quota, not a per-minute limit: go to the fallback model
                    if delay:
                        time.sleep(delay + 1)
                    else:
                        self._sleep(attempt)
                    continue
                if resp.status_code != 200:
                    # 400/403/404: our request or key is wrong; retrying will not help.
                    failures.append(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")
                    break
                self.model = model
                turn = self._parse(resp.json(), model)
                turn.attempts = failures
                return turn
        raise LLMError("model unavailable after retries: " + "; ".join(failures))

    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 20) + random.random())

    @staticmethod
    def _retry_delay(resp: httpx.Response) -> Optional[float]:
        """Gemini 429s carry RetryInfo.retryDelay like '37s' for per-minute limits."""
        try:
            for d in resp.json().get("error", {}).get("details", []):
                if "retryDelay" in d:
                    return float(str(d["retryDelay"]).rstrip("s"))
        except (ValueError, AttributeError):
            pass
        return None

    @staticmethod
    def _parse(data: Dict[str, Any], model: str) -> LLMTurn:
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
            return LLMTurn("", [], {"role": "model", "parts": [{"text": ""}]}, reason, model)
        cand = cands[0]
        content = cand.get("content") or {"role": "model", "parts": []}
        content.setdefault("role", "model")
        parts = content.get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [ToolCall(p["functionCall"]["name"], p["functionCall"].get("args") or {},
                          p["functionCall"].get("id"))
                 for p in parts if "functionCall" in p]
        if not parts:
            content["parts"] = [{"text": ""}]
        return LLMTurn(text.strip(), calls, content, cand.get("finishReason", ""), model)
