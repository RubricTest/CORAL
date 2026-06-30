"""RL environment selection grader.

The agent writes ``solution.py`` defining ``run(inputs_path) -> dict[str, float]``
that maps each environment ``task`` (``repo:commit_hash``) to a *keep score*
(higher = should keep for RL training). The grader runs it on a fixed eval split
(input-only records) and scores it against the hidden empirical pass rate ``p_i``.

The task splits into two sub-problems; we score them separately:

  * **Primary — keep/drop separation (AUROC).** ``keep_i = 1 iff 0 < p_i < 1``
    (a dead env at p=0 or a trivial env at p=1 yields no GRPO signal). AUROC of
    the score against this binary label is robust to the coarse p estimate
    (n=8..24 samples/env). Random = 0.50, perfect separation = 1.0. **This is the
    returned score.** Reported alongside: **AP** (average precision, area under PR;
    random baseline ≈ keep prevalence ≈ 0.56).

  * **Secondary — continuous learnability ranking (mSR).** Learnability value is
    the GRPO reward variance ``v_i = p_i (1 - p_i)`` (smooth, peaks at p=0.5),
    NOT the old saturated ``1 - p_i^8 - (1-p_i)^8``. mSR = mean over budgets N of
    ``SR@N = Σ_{top-N by method} v_i / Σ_{top-N by v} v_i``. Spearman(score, v) too.

The ground truth (v / p / keep) is delivered via ``grader.private`` into
``.coral/private/gt/`` (read here through ``self.private_dir``) — NOT shipped
inside this package, so a `cat` of the (editable-installed) grader source reveals
no answers. ``.coral/private/`` is covered by the agent's Read deny rule and the
Bash filesystem sandbox. The agent only sees input fields (problem_statement,
docker_image, prompt, ...) in ``seed/data/task_env_inputs.jsonl``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import textwrap
from pathlib import Path

from coral.grader import TaskGrader
from coral.types import ScoreBundle

_SPLITS = {
    "full": "task_gt.csv",
    "dev1000": "task_gt_dev1000.csv",
    "dev500": "task_gt_dev500.csv",
    "dev300": "task_gt_dev300.csv",
    "dev100": "task_gt_dev100.csv",
}


def _load_gt(split: str, gt_dir: Path) -> dict[str, dict]:
    path = gt_dir / _SPLITS[split]
    gt: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            # Learnability value = reward variance p(1-p) (smooth, peaks at p=0.5).
            # Derived from the empirical pass rate p; the stale `v` column in the
            # CSV (the old 1-p^8-(1-p)^8) is intentionally ignored.
            p = float(r["p"])
            gt[r["task"]] = {"v": p * (1.0 - p), "p": p, "keep": int(r["keep"])}
    return gt


def _spearman(xs: list[float], ys: list[float]) -> float:
    import math

    def ranks(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den else 0.0


def _auroc(labels: list[int], xs: list[float]) -> float:
    """AUROC for the keep/drop binary label via the Mann-Whitney U statistic.

    Tie-aware (average ranks), so it does not depend on any tie-break ordering.
    Returns nan if a class is empty. Random=0.5, perfect separation=1.0.
    """
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            rk[order[k]] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = sum(rk[i] for i in range(n) if labels[i] == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(method: list[str], gt: dict[str, dict]) -> float:
    """Average Precision (area under PR) for keep=1, walking the ranked list.

    AP = mean over positives of precision@(rank of that positive). The unbiased
    estimator (no trapezoidal PR interpolation). Random baseline ≈ prevalence.
    """
    n_pos = sum(gt[t]["keep"] for t in method)
    if n_pos == 0:
        return float("nan")
    tp = 0
    ap = 0.0
    for rank, t in enumerate(method, 1):
        if gt[t]["keep"] == 1:
            tp += 1
            ap += tp / rank
    return ap / n_pos


def _score_ranking(gt: dict[str, dict], scores: dict[str, float]) -> dict:
    """Compute AUROC (primary) + AP + mSR(v=p(1-p)) + Spearman vs hidden GT."""
    tasks = list(gt)
    # GT-independent deterministic tie-break: a constant-score method can't shortcut.
    def tb(t):
        return hashlib.md5(t.encode()).hexdigest()

    method = sorted(tasks, key=lambda t: (-scores.get(t, float("-inf")), tb(t)))
    n = len(tasks)
    K = sum(gt[t]["keep"] for t in tasks)

    # --- Primary: keep/drop separation (AUROC), secondary AP --------------
    xs = [scores.get(t, float("-inf")) for t in tasks]
    labels = [gt[t]["keep"] for t in tasks]
    auroc = _auroc(labels, xs)
    ap = _average_precision(method, gt)

    # --- Secondary: continuous learnability ranking, v = p(1-p) ------------
    v_sorted = sorted((gt[t]["v"] for t in tasks), reverse=True)
    opt_cum = [0.0]
    for v in v_sorted:
        opt_cum.append(opt_cum[-1] + v)
    m_cum = [0.0]
    for t in method:
        m_cum.append(m_cum[-1] + gt[t]["v"])

    def SR(k):
        return m_cum[k] / opt_cum[k] if opt_cum[k] > 0 else 1.0

    mSR = sum(SR(k) for k in range(1, n + 1)) / n
    sr_q = {q: SR(max(1, round(n * q))) for q in (0.25, 0.5, 0.75)}
    rho = _spearman(xs, [gt[t]["v"] for t in tasks])

    return {
        "auroc": auroc,
        "ap": ap,
        "mSR": mSR,
        "sr_q": sr_q,
        "spearman": rho,
        "keep_prevalence": K / n if n else 0.0,
        "n": n,
        "missing": sum(1 for t in tasks if t not in scores),
    }


class Grader(TaskGrader):
    """Grader for the RL environment selection task."""

    def evaluate(self) -> ScoreBundle:
        program_file = self.args.get("program_file", "solution.py")
        inputs_file = self.args.get("inputs_file", "data/task_env_inputs.jsonl")
        split = self.args.get("split", "dev500")
        if split not in _SPLITS:
            return self.fail(f"Unknown split '{split}' (choices: {list(_SPLITS)})")

        program_path = os.path.join(self.codebase_path, program_file)
        inputs_path = os.path.join(self.codebase_path, inputs_file)
        if not os.path.exists(program_path):
            return self.fail(f"Program file ({program_file}) not found")
        if not os.path.exists(inputs_path):
            return self.fail(f"Inputs file ({inputs_file}) not found")

        # Hidden answers live in .coral/private/gt/ (delivered via grader.private).
        gt_dir = Path(self.private_dir) / "gt"
        if not (gt_dir / _SPLITS[split]).exists():
            return self.fail(f"Ground-truth not found at {gt_dir / _SPLITS[split]} (check grader.private)")
        gt = _load_gt(split, gt_dir)
        eval_tasks = set(gt)

        # Build the eval-split inputs (input-only records, no answers) to feed run().
        import tempfile

        split_records = []
        with open(inputs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("task") in eval_tasks:
                    split_records.append(rec)
        if len(split_records) != len(eval_tasks):
            return self.fail(
                f"Inputs cover {len(split_records)}/{len(eval_tasks)} eval tasks for split '{split}' "
                f"— seed/data/task_env_inputs.jsonl is incomplete."
            )

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            for rec in split_records:
                tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            split_inputs_path = tf.name

        try:
            scores = _run_solution(program_path, split_inputs_path, self.timeout, self.get_python_command())
        except TimeoutError:
            return self.fail(f"Solution timed out after {self.timeout}s")
        except Exception as e:
            return self.fail(f"Solution failed: {e}")
        finally:
            try:
                os.unlink(split_inputs_path)
            except OSError:
                pass

        if not isinstance(scores, dict):
            return self.fail("run() must return a dict {task: score}")
        scores = {str(k): float(v) for k, v in scores.items() if v is not None}

        res = _score_ranking(gt, scores)
        q = res["sr_q"]
        explanation = (
            f"AUROC={res['auroc']:.4f} on split={split} (N={res['n']}, keep_prev={res['keep_prevalence']:.3f}) | "
            f"AP={res['ap']:.4f} | "
            f"mSR={res['mSR']:.4f} [v=p(1-p)] SR@25%={q[0.25]:.3f} SR@50%={q[0.5]:.3f} SR@75%={q[0.75]:.3f} | "
            f"Spearman={res['spearman']:.3f} | "
            f"baselines — AUROC: random=0.50 oracle=1.0; AP: random≈keep_prev"
        )
        if res["missing"]:
            explanation += f" | [warn] {res['missing']} eval tasks unscored (treated as last)"
        return self.score(res["auroc"], explanation)


def _run_solution(program_path: str, inputs_path: str, timeout: int, python_cmd: list[str]) -> dict:
    """Run the agent's solution.run(inputs_path) in a subprocess; parse {task: score} JSON."""
    import subprocess

    script = textwrap.dedent(
        f"""\
        import json, sys, os
        sys.path.insert(0, os.path.dirname({os.path.abspath(program_path)!r}))
        program = __import__({os.path.splitext(os.path.basename(program_path))[0]!r})
        scores = program.run({inputs_path!r})
        print("__SCORES_BEGIN__")
        print(json.dumps({{str(k): float(v) for k, v in dict(scores).items()}}))
        print("__SCORES_END__")
        """
    )
    result = subprocess.run([*python_cmd, "-c", script], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-2000:])
    out = result.stdout
    if "__SCORES_BEGIN__" not in out or "__SCORES_END__" not in out:
        raise RuntimeError(f"No scores marker in output.\nstdout tail: {out[-500:]}\nstderr: {result.stderr.strip()[-500:]}")
    payload = out.split("__SCORES_BEGIN__", 1)[1].split("__SCORES_END__", 1)[0].strip()
    return json.loads(payload)
