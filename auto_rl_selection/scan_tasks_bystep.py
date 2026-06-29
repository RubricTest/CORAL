import os, json, glob, collections
import torch
from multiprocessing import Pool

ROOT = "/shared_workspace_mfs/yanruo/github/auto_rl_selection/data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c"
RD = os.path.join(ROOT, "rollout_data")

def proc(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    step = int(os.path.basename(path)[:-3])
    # key -> [repo, n, npass]   (within this single step)
    local = {}
    for s in o["samples"]:
        md = s.get("metadata") or {}
        repo = md.get("repo_name", "?")
        commit = md.get("commit_hash", "?")
        key = f"{repo}:{commit}"
        rw = s.get("reward") or {}
        val = rw.get("value", 0.0) if isinstance(rw, dict) else 0.0
        d = local.setdefault(key, [repo, 0, 0])
        d[1] += 1
        if val >= 1.0:
            d[2] += 1
    return step, local

def main():
    files = sorted(glob.glob(os.path.join(RD, "*.pt")), key=lambda p: int(os.path.basename(p)[:-3]))
    print(f"scanning {len(files)} files (per task x step)...", flush=True)
    # task -> {step: [n, npass]}, plus repo
    tasks = {}
    repo_of = {}
    with Pool(24) as pool:
        for i, (step, local) in enumerate(pool.imap_unordered(proc, files)):
            for key, (repo, n, npass) in local.items():
                repo_of[key] = repo
                tasks.setdefault(key, {})[step] = [n, npass]
            if (i+1) % 30 == 0:
                print(f"  done {i+1}/{len(files)}", flush=True)

    rows = []
    for key, bystep in tasks.items():
        steps = sorted(bystep)
        n = sum(bystep[s][0] for s in steps)
        npass = sum(bystep[s][1] for s in steps)
        pr = npass / n
        first_step, last_step = steps[0], steps[-1]
        mean_step = sum(steps) / len(steps)
        first_solved = next((s for s in steps if bystep[s][1] > 0), -1)
        # first vs second half of THIS task's appearances
        half = len(steps) // 2
        early = steps[:half] if half else steps[:1]
        late = steps[half:] if half else steps[-1:]
        e_n = sum(bystep[s][0] for s in early); e_p = sum(bystep[s][1] for s in early)
        l_n = sum(bystep[s][0] for s in late); l_p = sum(bystep[s][1] for s in late)
        early_pr = e_p / e_n if e_n else 0.0
        late_pr = l_p / l_n if l_n else 0.0
        # compact per-step string: "12:0/8;45:2/8"
        pbs = ";".join(f"{s}:{bystep[s][1]}/{bystep[s][0]}" for s in steps)
        # classify trend among dead-by-overall? generic trend tag
        if npass == 0:
            base = "dead_hard"
        elif pr == 1.0:
            base = "trivial"
        elif pr < 0.25:
            base = "hard"
        elif pr < 0.75:
            base = "learnable"
        else:
            base = "easy"
        # dynamics: solved only late / only early / mixed / never
        if npass == 0:
            dyn = "never_solved"
        elif first_solved >= 0 and first_solved >= mean_step and early_pr == 0 and late_pr > 0:
            dyn = "emerging_late"   # 早期全错, 后段开始解出 -> 不该剔除
        elif late_pr == 0 and early_pr > 0:
            dyn = "lost_late"       # 早期能解, 后段不行
        else:
            dyn = "steady"
        rows.append(dict(task=key, repo=repo_of[key], n=n, npass=npass,
                         pass_rate=round(pr, 4), n_steps=len(steps),
                         first_step=first_step, last_step=last_step,
                         mean_step=round(mean_step, 1), first_solved_step=first_solved,
                         early_pass_rate=round(early_pr, 4), late_pass_rate=round(late_pr, 4),
                         bucket=base, dynamics=dyn, pass_by_step=pbs))

    rows.sort(key=lambda r: (r["repo"], r["pass_rate"], r["task"]))
    import csv
    cols = ["task","repo","n","npass","pass_rate","n_steps","first_step","last_step",
            "mean_step","first_solved_step","early_pass_rate","late_pass_rate",
            "bucket","dynamics","pass_by_step"]
    with open(os.path.join(os.path.dirname(__file__), "task_difficulty_bystep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "task_bystep.json"), "w"))
    print(f"wrote task_difficulty_bystep.csv : {len(rows)} tasks", flush=True)

    # quick summary
    bdyn = collections.Counter(r["dynamics"] for r in rows)
    print("dynamics:", dict(bdyn))

if __name__ == "__main__":
    main()
