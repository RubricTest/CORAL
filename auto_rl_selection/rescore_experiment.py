#!/usr/bin/env python3
"""Re-score a finished rl-env-selection run with the NEW metrics.

The original run (results/rl-environment-selection/<ts>) was graded with the old
metric (mSR over v = 1 - p^8 - (1-p)^8). This script recomputes every attempt with
the NEW scheme used by examples/rl-env-selection-agentic:

    primary   : AUROC of the score vs the keep/drop label (keep = 0<p<1)
    secondary : AP (avg precision), mSR over v = p(1-p), Spearman(score, v)

Attempt JSONs only store the final scalar, not the per-task {task: score}, so we
re-run each attempt's solution.py: git-archive the commit's solution.py + cached
feature JSONs (NOT the 43MB inputs) into a temp dir, run run(dev500_inputs), then
apply the agentic grader's metric functions (imported, so this stays in lockstep
with the real grader). GT = the run's own .coral/private/gt (== the agentic gt).

Usage:
    uv run python auto_rl_selection/rescore_experiment.py \
        [--run results/rl-environment-selection/2026-06-26_033844] \
        [--split dev500] [--limit N] [--out auto_rl_selection/rescore_dev500.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRADER_SRC = REPO_ROOT / "examples/rl-env-selection-agentic/grader/src"
SEED_INPUTS = REPO_ROOT / "examples/rl-env-selection-agentic/seed/data/task_env_inputs.jsonl"

sys.path.insert(0, str(GRADER_SRC))
from rl_env_selection_grader.grader import _load_gt, _run_solution, _score_ranking  # noqa: E402


def build_split_inputs(gt: dict, out_path: str) -> int:
    """Write the eval-split input subset (the exact records the grader feeds run())."""
    eval_tasks = set(gt)
    n = 0
    with open(SEED_INPUTS) as f, open(out_path, "w") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task") in eval_tasks:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def rescore_commit(run_repo: Path, commit: str, inputs_path: str, gt: dict, timeout: int) -> dict | None:
    """git-archive the commit's code, run its solution, return new metrics (or None)."""
    with tempfile.TemporaryDirectory(prefix="rescore_") as tmp:
        # Extract only solution.py + cached feature JSONs (skip the 43MB inputs file).
        archive = subprocess.run(
            ["git", "-C", str(run_repo), "archive", commit, "solution.py", "data"],
            capture_output=True,
        )
        if archive.returncode != 0:
            return {"error": "archive: " + archive.stderr.decode()[-200:]}
        untar = subprocess.run(
            ["tar", "-x", "-C", tmp, "--exclude=data/task_env_inputs.jsonl"],
            input=archive.stdout,
            capture_output=True,
        )
        if untar.returncode != 0:
            return {"error": "untar: " + untar.stderr.decode()[-200:]}
        prog = os.path.join(tmp, "solution.py")
        if not os.path.exists(prog):
            return {"error": "no solution.py at commit"}
        try:
            scores = _run_solution(prog, inputs_path, timeout, [sys.executable])
        except Exception as e:
            return {"error": f"run: {str(e)[-200:]}"}
        scores = {str(k): float(v) for k, v in scores.items() if v is not None}
        return _score_ranking(gt, scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/rl-environment-selection/2026-06-26_033844")
    ap.add_argument("--split", default="dev500")
    ap.add_argument("--limit", type=int, default=0, help="rescore only first N attempts (0 = all)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="auto_rl_selection/rescore_dev500.csv")
    args = ap.parse_args()

    run_dir = (REPO_ROOT / args.run).resolve()
    run_repo = run_dir / "repo"
    gt_dir = run_dir / ".coral/private/gt"
    attempts_dir = run_dir / ".coral/public/attempts"

    gt = _load_gt(args.split, gt_dir)
    print(f"GT: split={args.split}  N={len(gt)}  keep={sum(g['keep'] for g in gt.values())}", flush=True)

    inputs_tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False).name
    n_in = build_split_inputs(gt, inputs_tmp)
    print(f"built split inputs: {n_in} records -> {inputs_tmp}", flush=True)

    # Load attempts (each = one commit + its old score).
    attempts = []
    for jf in sorted(attempts_dir.glob("*.json")):
        a = json.loads(jf.read_text())
        if a.get("commit_hash"):
            attempts.append(a)
    if args.limit:
        attempts = attempts[: args.limit]
    print(f"rescoring {len(attempts)} attempts...\n", flush=True)

    rows = []
    for i, a in enumerate(attempts, 1):
        commit = a["commit_hash"]
        res = rescore_commit(run_repo, commit, inputs_tmp, gt, args.timeout)
        row = {
            "commit": commit[:12],
            "agent": a.get("agent_id", ""),
            "old_mSR": round(a.get("score", float("nan")), 4) if a.get("score") is not None else None,
            "status": a.get("status", ""),
        }
        if res and "error" not in res:
            row.update(
                AUROC=round(res["auroc"], 4),
                AP=round(res["ap"], 4),
                new_mSR=round(res["mSR"], 4),
                Spearman=round(res["spearman"], 4),
                error="",
            )
        else:
            row.update(AUROC=None, AP=None, new_mSR=None, Spearman=None,
                       error=(res or {}).get("error", "unknown"))
        rows.append(row)
        if i % 10 == 0 or i == len(attempts):
            ok = sum(1 for r in rows if r["error"] == "")
            print(f"  {i}/{len(attempts)}  (ok={ok})", flush=True)

    os.unlink(inputs_tmp)

    out_path = REPO_ROOT / args.out
    fields = ["commit", "agent", "status", "old_mSR", "AUROC", "AP", "new_mSR", "Spearman", "error"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}", flush=True)

    _summarize(rows)
    return 0


def _summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r["error"] == ""]
    print(f"\n===== summary ({len(ok)}/{len(rows)} rescored) =====")
    if not ok:
        errs = {}
        for r in rows:
            errs[r["error"][:40]] = errs.get(r["error"][:40], 0) + 1
        print("all failed; error histogram:", errs)
        return

    def topk(key, k=10, rev=True):
        return sorted(ok, key=lambda r: (r[key] is None, r[key]), reverse=rev)[:k]

    print("\nTop 10 by OLD mSR (the run's own ranking):")
    print(f"  {'commit':14}{'old_mSR':>9}{'AUROC':>9}{'AP':>9}{'new_mSR':>9}")
    for r in topk("old_mSR"):
        print(f"  {r['commit']:14}{_f(r['old_mSR']):>9}{_f(r['AUROC']):>9}{_f(r['AP']):>9}{_f(r['new_mSR']):>9}")

    print("\nTop 10 by NEW AUROC (primary metric):")
    print(f"  {'commit':14}{'AUROC':>9}{'AP':>9}{'new_mSR':>9}{'old_mSR':>9}")
    for r in topk("AUROC"):
        print(f"  {r['commit']:14}{_f(r['AUROC']):>9}{_f(r['AP']):>9}{_f(r['new_mSR']):>9}{_f(r['old_mSR']):>9}")

    # Rank-correlation between old mSR and each new metric.
    import math

    def spearman(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) < 3:
            return float("nan")
        xs, ys = zip(*pairs)

        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for pos, idx in enumerate(order):
                rk[idx] = pos
            return rk

        rx, ry = ranks(xs), ranks(ys)
        n = len(rx)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n)))
        return num / den if den else float("nan")

    old = [r["old_mSR"] for r in ok]
    print("\nSpearman rank-corr vs OLD mSR ranking:")
    for key in ("AUROC", "AP", "new_mSR"):
        print(f"  old_mSR vs {key:8}: {spearman(old, [r[key] for r in ok]):+.3f}")

    best_old = max(ok, key=lambda r: r["old_mSR"])
    best_auroc = max(ok, key=lambda r: r["AUROC"])
    print(f"\nbest by old_mSR : {best_old['commit']}  old={best_old['old_mSR']}  AUROC={best_old['AUROC']}")
    print(f"best by AUROC   : {best_auroc['commit']}  AUROC={best_auroc['AUROC']}  old={best_auroc['old_mSR']}")
    print(f"  -> same attempt? {best_old['commit'] == best_auroc['commit']}")


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
