"""RL environment selection grader.

The agent writes ``solution.py`` defining ``run(inputs_path) -> dict[str, float]``
that maps each environment ``task`` (``repo:commit_hash``) to a *keep score*
(higher = should keep for RL training). The grader runs it on a fixed eval split
(input-only records) and scores the ranking against the hidden ground-truth value
``v_i = 1 - p_i^8 - (1-p_i)^8`` (expected per-step GRPO learning-signal probability).

Primary metric: **mSR** (mean Signal Retention over all budgets N), in [0, 1].
    SR@N = Σ_{top-N by method} v_i  /  Σ_{top-N by v} v_i
    mSR  = (1/T) Σ_{N=1..T} SR@N
Honest baselines: random ≈ 0.60, repo-prior ≈ 0.65, oracle (= GT order) = 1.0.

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
            gt[r["task"]] = {"v": float(r["v"]), "keep": int(r["keep"])}
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


def _score_ranking(gt: dict[str, dict], scores: dict[str, float]) -> dict:
    """Compute mSR + auxiliaries from a {task: score} mapping vs hidden GT."""
    tasks = list(gt)
    # GT-independent deterministic tie-break: a constant-score method can't shortcut.
    def tb(t):
        return hashlib.md5(t.encode()).hexdigest()

    method = sorted(tasks, key=lambda t: (-scores.get(t, float("-inf")), tb(t)))

    v_sorted = sorted((gt[t]["v"] for t in tasks), reverse=True)
    opt_cum = [0.0]
    for v in v_sorted:
        opt_cum.append(opt_cum[-1] + v)
    m_cum = [0.0]
    for t in method:
        m_cum.append(m_cum[-1] + gt[t]["v"])
    n = len(tasks)

    def SR(k):
        return m_cum[k] / opt_cum[k] if opt_cum[k] > 0 else 1.0

    mSR = sum(SR(k) for k in range(1, n + 1)) / n
    sr_q = {q: SR(max(1, round(n * q))) for q in (0.25, 0.5, 0.75)}

    xs = [scores.get(t, float("-inf")) for t in tasks]
    ys = [gt[t]["v"] for t in tasks]
    rho = _spearman(xs, ys)

    K = sum(gt[t]["keep"] for t in tasks)
    pred_keep = set(method[:K])
    tp = sum(1 for t in pred_keep if gt[t]["keep"] == 1)
    f1 = tp / K if K else 0.0  # precision == recall == F1 at fixed K

    return {"mSR": mSR, "sr_q": sr_q, "spearman": rho, "keepF1": f1, "n": n, "missing": sum(1 for t in tasks if t not in scores)}


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
            f"mSR={res['mSR']:.4f} on split={split} (N={res['n']}) | "
            f"SR@25%={q[0.25]:.3f} SR@50%={q[0.5]:.3f} SR@75%={q[0.75]:.3f} | "
            f"Spearman={res['spearman']:.3f} keepF1={res['keepF1']:.3f} | "
            f"baselines: random≈0.60 repo-prior≈0.65 oracle=1.0"
        )
        if res["missing"]:
            explanation += f" | [warn] {res['missing']} eval tasks unscored (treated as last)"
        return self.score(res["mSR"], explanation)


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
