#!/usr/bin/env python3
"""Tiny OpenAI-compatible chat client for the *agentic* RL-env-selection task.

The grader runs your ``solution.run(inputs_path)`` in a subprocess that INHERITS
the environment of whoever launched ``coral start``. So if you export the gateway
credentials before launching the run, ``solution.py`` can call the provided model
at grade time — letting you build an LLM-judge / multi-subagent **pipeline** that
scores each environment's learnability, instead of a hand-written feature.

Required env vars (set them BEFORE ``coral start``; they propagate to the grader
subprocess that runs your solution):

    CORAL_LLM_BASE_URL   OpenAI-compatible base, e.g. http://host:port/v1
                         (falls back to OPENAI_BASE_URL / OPENAI_API_BASE)
    CORAL_LLM_API_KEY    API key for the gateway
                         (falls back to OPENAI_API_KEY)
    CORAL_LLM_MODEL      model name the gateway serves, e.g. openai/qwen3-32b
                         (falls back to OPENAI_MODEL; default 'openai/qwen3-32b')

Zero third-party deps — uses only the stdlib (urllib + concurrent.futures), so it
runs unchanged inside the isolated grader venv. If no base_url/key is configured,
``configured()`` returns False and you should fall back to a cheap heuristic so the
solution still grades.

Typical pipeline usage (fan out one "subagent" call per environment in parallel):

    from llm_client import LLMClient
    cli = LLMClient()
    if cli.configured():
        prompts = [judge_prompt(rec) for rec in records]
        replies = cli.map(prompts, system=JUDGE_SYSTEM, max_workers=16)
        scores  = {rec["task"]: parse(r) for rec, r in zip(records, replies)}
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable


def _first_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


class LLMClient:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = (base_url or _first_env("CORAL_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")).rstrip("/")
        self.api_key = api_key or _first_env("CORAL_LLM_API_KEY", "OPENAI_API_KEY")
        self.model = model or _first_env("CORAL_LLM_MODEL", "OPENAI_MODEL", default="openai/qwen3-32b")
        self.timeout = timeout
        self.max_retries = max_retries

    def configured(self) -> bool:
        """True if a base_url (and, for most gateways, a key) is available."""
        return bool(self.base_url)

    # -- single call -------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **extra,
    ) -> str:
        """One chat completion. Returns the assistant message content (str)."""
        if not self.base_url:
            raise RuntimeError("LLMClient not configured: set CORAL_LLM_BASE_URL")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return (data["choices"][0]["message"]["content"] or "").strip()
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError, TimeoutError) as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"chat() failed after {self.max_retries} attempts: {last}")

    def complete(self, prompt: str, *, system: str | None = None, **kw) -> str:
        """Convenience single-turn completion."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw)

    # -- parallel fan-out (the "subagent" pipeline) ------------------------
    def map(
        self,
        prompts: Iterable[str],
        *,
        system: str | None = None,
        max_workers: int = 16,
        on_error: str = "",
        **kw,
    ) -> list[str]:
        """Run many prompts concurrently; return replies in input order.

        Each prompt becomes an independent "subagent" call. Failures yield
        ``on_error`` (default "") rather than aborting the whole batch, so one bad
        environment never sinks the run.
        """
        prompts = list(prompts)

        def _one(p: str) -> str:
            try:
                return self.complete(p, system=system, **kw)
            except Exception:
                return on_error

        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(prompts)))) as pool:
            return list(pool.map(_one, prompts))


if __name__ == "__main__":
    # Smoke test: `python llm_client.py "say hi in one word"`
    import sys

    cli = LLMClient()
    print(f"configured={cli.configured()} base_url={cli.base_url!r} model={cli.model!r}")
    if cli.configured():
        q = sys.argv[1] if len(sys.argv) > 1 else "Reply with exactly one word: ok"
        print("reply:", cli.complete(q, max_tokens=16))
