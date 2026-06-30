"""Agentic baseline for the RL environment selection task.

Goal: score each SWE RL environment by how useful it is to KEEP for training.
Higher score = more likely to keep. You are evaluated by mSR (mean Signal
Retention) against a hidden ground-truth value v_i; see CORAL.md / BENCHMARK.md.

This variant gives you a **callable model** (an OpenAI-compatible gateway) so you
can build an LLM-judge / multi-subagent *pipeline* instead of a hand-written
feature. `llm_client.LLMClient` fans out one "subagent" call per environment in
parallel (`.map(...)`) — each call rates how *learnable* an environment looks for
the base model (you want p≈0.5: neither trivially solved nor impossible, since
v_i = 1 - p^8 - (1-p)^8 peaks at p=0.5).

Contract:
    run(inputs_path: str) -> dict[str, float]
        inputs_path : JSONL; each line is ONE environment's INPUT record (no
                      answers): task, repo, commit_hash, docker_image,
                      data_source, ability, problem_statement, prompt
        returns     : {task: score}  (higher = keep; omitted tasks rank last)

The model is configured via env vars (CORAL_LLM_BASE_URL / _API_KEY / _MODEL),
which the grader subprocess inherits from your `coral start` shell. If no model is
configured, this falls back to a cheap heuristic so the solution still grades.

You may NOT use any ground-truth pass rate / reward / test oracle (all hidden) —
predict learnability from the inputs (and, optionally, from exploring the docker
image via env_explore.py). This LLM-judge is one pipeline; improve it.
"""

from __future__ import annotations

import json
import re

from llm_client import LLMClient

JUDGE_SYSTEM = (
    "You are rating SWE bug-fix tasks for their value as RL training environments. "
    "A task is most valuable when it is *learnable* for a mid-size code model: not "
    "trivially easy (the model would always pass) and not impossible (it would always "
    "fail), but right in the middle where the model sometimes succeeds. Output ONLY a "
    "single integer 0-100 = estimated probability (in %) that the base model solves it. "
    "0 = clearly impossible, 100 = clearly trivial, 50 = a genuine coin-flip."
)


def _judge_prompt(rec: dict) -> str:
    ps = (rec.get("problem_statement") or "").strip()
    repo = rec.get("repo", "?")
    # Cap the issue text to keep calls cheap and within context.
    if len(ps) > 4000:
        ps = ps[:4000] + " …[truncated]"
    return (
        f"Repository: {repo}\n"
        f"GitHub issue / problem statement:\n\"\"\"\n{ps}\n\"\"\"\n\n"
        f"Estimate the base model's pass probability (0-100). Answer with ONLY the integer."
    )


def _parse_pass_pct(reply: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", reply or "")
    if not m:
        return None
    val = float(m.group())
    return max(0.0, min(100.0, val)) / 100.0


def _learnability(p: float) -> float:
    """v_i = 1 - p^8 - (1-p)^8 — the GRPO learning-signal value (peaks at p=0.5)."""
    return 1.0 - p**8 - (1.0 - p) ** 8


def run(inputs_path: str) -> dict[str, float]:
    records: list[dict] = []
    with open(inputs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    cli = LLMClient()
    if not cli.configured():
        # Fallback: heuristic placeholder (problem_statement length) so the run
        # still grades when no model is wired up. Beat this with the LLM pipeline.
        return {r["task"]: float(len(r.get("problem_statement") or "")) for r in records}

    # Pipeline: one subagent call per environment, fanned out in parallel.
    prompts = [_judge_prompt(r) for r in records]
    replies = cli.map(prompts, system=JUDGE_SYSTEM, max_workers=16, max_tokens=8, temperature=0.0)

    scores: dict[str, float] = {}
    for rec, reply in zip(records, replies):
        p = _parse_pass_pct(reply)
        if p is None:
            # Unparseable / failed call → mid-rank fallback by issue length.
            scores[rec["task"]] = float(len(rec.get("problem_statement") or "")) * 1e-6
        else:
            scores[rec["task"]] = _learnability(p)
    return scores
