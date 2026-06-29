"""Baseline solution for the RL environment selection task.

Goal: score each SWE RL environment by how useful it is to KEEP for training.
Higher score = more likely to keep. You are evaluated by mSR (mean Signal
Retention) against a hidden ground-truth value v_i; see CORAL.md / BENCHMARK.md.

Contract:
    run(inputs_path: str) -> dict[str, float]
        inputs_path : path to a JSONL file; each line is ONE environment's
                      INPUT record (no answers), with keys:
                        task, repo, commit_hash, docker_image,
                        data_source, ability, problem_statement, prompt
        returns     : {task: score}  (higher = keep). Score any real number;
                      only the induced ranking matters. Tasks you omit are
                      ranked last.

This baseline just uses problem_statement length as a trivial feature — it is a
PLACEHOLDER to beat. Random ≈ 0.60 mSR, repo-prior ≈ 0.65. Find a signal that
predicts which environments yield non-degenerate GRPO groups (neither all-pass
nor all-fail under the base model).

You MAY (optionally) use the base model / run rollouts in the docker_image to
estimate difficulty — but you must NOT use any ground-truth pass rate / reward
(it is hidden). Cheap, rollout-free features are encouraged first.
"""

import json


def run(inputs_path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(inputs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Placeholder feature — replace with a real learnability predictor.
            scores[rec["task"]] = float(len(rec.get("problem_statement") or ""))
    return scores
