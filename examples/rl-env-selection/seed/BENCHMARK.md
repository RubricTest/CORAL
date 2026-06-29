# RL Environment Selection — Benchmark Spec

Score each SWE RL environment by how useful it is to **keep** for RL training.
Static selection (no per-step dynamics). You are the *method*; an agent iterates
`solution.py` to maximize **mSR** against a hidden ground truth.

## Inputs you get (no answers)

`data/task_env_inputs.jsonl` — one line per environment (4,459 total):

| field | meaning |
|---|---|
| `task` | unique id `repo:commit_hash` |
| `repo` | one of 10 repos (pandas, numpy, pillow, orange3, aiohttp, tornado, scrapy, pyramid, datalad, coveragepy) |
| `commit_hash` | buggy commit |
| `docker_image` | `namanjain12/<repo>_final:<commit_hash>` — runnable testbed (optional, for rollouts) |
| `problem_statement` | the github issue text (the agent prompt's core) |
| `prompt` | `[{role:system,...},{role:user,...}]` — the exact base-model input |
| `data_source`,`ability` | both `swe` |

The grader feeds your `run()` only the **eval-split** subset (input-only).

## Contract

```python
def run(inputs_path: str) -> dict[str, float]:
    # inputs_path: JSONL of eval-split input records (keys above)
    # return {task: keep_score}; higher = keep; omitted tasks rank last
```

## Ground truth (hidden) & metric

From a full 181-step GRPO run each env has an empirical base pass rate `p_i`.
GRPO group size = 8 ⇒ a group gives non-zero advantage only if not all-pass and
not all-fail, so the per-step expected learning-signal value is:

```
v_i = 1 - p_i^8 - (1 - p_i)^8        # 0 at p∈{0,1}, ≈1 at p≈0.5
```

**Primary metric — mSR (maximize), in [0,1]:**

```
SR@N = Σ_{top-N by your score} v_i  /  Σ_{top-N by v} v_i      # signal retained at budget N
mSR  = (1/T) Σ_{N=1..T} SR@N                                   # averaged over all budgets
```

Reported alongside: `SR@25/50/75%`, `Spearman(score, v)`, `keepF1` (at the
keep/drop budget). Honest baselines: **random ≈ 0.60, repo-prior ≈ 0.65,
oracle = 1.0**. repo-level features cap near 0.65 (Spearman≈0.12) — the win is in
distinguishing environments **within** a repo.

## Eval splits (nested, distribution-matched)

`dev100 ⊂ dev300 ⊂ dev500 ⊂ dev1000 ⊂ full`. Default eval = `dev500`. Each is
stratified by `repo × difficulty` so mSR on a split tracks full. dev100/dev300 are
noisier (smoke tests); report on dev500+ / full. The eval split is fixed by the
grader (`task.yaml: grader.args.split`); you cannot pick your own subset.

## Rules (anti-leak)

- Predict `v` from the inputs and from **exploring** the environment. You may
  inspect the repo, its tests, code structure, the issue context, etc.
- Do NOT use any ground-truth pass rate, reward, or test oracle (all hidden). The
  goal is to PREDICT learnability cheaply, not to measure it by solving the task.
- Think **learnability** (p≈0.5), not raw difficulty: both impossible (p=0) and
  trivial (p=1) environments have v=0 and should be dropped.

## Exploring an environment

Each `docker_image` is a runnable SWE testbed (the repo checked out at the buggy
commit at `/testbed`). Spin it up via OpenSandbox to **gather signal about the
repo/task** — file layout, size, the affected module, test density, how localized
the issue is, etc. — and turn that into features that predict `v`. (This is about
*exploration*, not attempting to solve.)

Helper provided: **`env_explore.py`** — `EnvSandbox(image)` → `start()/exec()/read_file()/stop()`.
It auto-prefixes the internal registry (`10.10.110.20:5000`) and points at the
deployed server (`10.10.110.50:30080`) by default, so just pass the
`docker_image` straight from the inputs.

Smoke-test one image (verified working: create → exec in /testbed → kill, ~45s):
```bash
uv run --with "opensandbox>=0.1.6" --python 3.12 env_explore.py \
    namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6
```

Direct exploration (simplest — no model needed):
```python
import asyncio
from env_explore import EnvSandbox

async def explore(rec):
    async with EnvSandbox(rec["docker_image"]) as env:
        # poke around the repo to build features (examples — design your own)
        _, tree, _   = await env.exec("cd /testbed && git ls-files | wc -l")
        _, target, _ = await env.exec("cd /testbed && git show --stat HEAD | tail -20")
        code = await env.read_file("/testbed/setup.py")
        return {"n_files": tree, "head_stat": target}  # -> derive a v feature
```

### Optional: agentic exploration with mini-swe-agent

If you want the model to explore *autonomously* (instead of hand-written commands),
`EnvSandbox.rollout()` runs **mini-swe-agent** inside the container with whatever
instruction you pass as `task`. Give it an **exploration** instruction (not "fix the
bug") and read back what it found:

```python
async with EnvSandbox(rec["docker_image"]) as env:
    r = await env.rollout(
        task="Explore /testbed and the issue. Report: which files/functions the "
             "problem touches, how localized it is, and how hard it looks. Do NOT fix it.",
        model="openai/qwen3-32b",   # your vLLM-served Qwen (litellm name)
        api_base="<BASE_URL>/v1", api_key="<API_KEY>",
        max_turns=15,
    )
    # r["trajectory"] / r["stdout"] = the model's exploration notes -> features
```
(Needs the service model; see `qwen3_vllm_calling.md`. Install of mini-swe-agent
inside the container is automatic via `uv tool install --python 3.12`.)

> Caveats: sandboxes are billable/limited — explore a **sample**, cache results, and
> always `stop()` (the context manager does). Images resolve from the internal
> registry; a bare Docker-Hub path will NOT pull. None of this touches the hidden
> GT — you are extracting *predictive features*, not measuring the true reward.
