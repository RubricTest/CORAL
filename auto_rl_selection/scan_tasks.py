import os, json, glob, collections
import torch
from multiprocessing import Pool

ROOT = "/shared_workspace_mfs/yanruo/github/auto_rl_selection/data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c"
RD = os.path.join(ROOT, "rollout_data")

def proc(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    step = int(os.path.basename(path)[:-3])
    # per-task within this file: key -> dict(repo, n, npass, steps set via step)
    local = {}
    for s in o["samples"]:
        md = s.get("metadata") or {}
        repo = md.get("repo_name", "?")
        commit = md.get("commit_hash", "?")
        key = f"{repo}:{commit}"
        rw = s.get("reward") or {}
        val = rw.get("value", 0.0) if isinstance(rw, dict) else 0.0
        d = local.setdefault(key, dict(repo=repo, n=0, npass=0, steps=set()))
        d["n"] += 1
        d["steps"].add(step)
        if val >= 1.0:
            d["npass"] += 1
    # make steps json-serializable
    for d in local.values():
        d["steps"] = sorted(d["steps"])
    return local

def main():
    files = sorted(glob.glob(os.path.join(RD, "*.pt")), key=lambda p: int(os.path.basename(p)[:-3]))
    print(f"scanning {len(files)} files for per-task stats...", flush=True)
    tasks = {}
    with Pool(24) as pool:
        for i, local in enumerate(pool.imap_unordered(proc, files)):
            for key, d in local.items():
                t = tasks.setdefault(key, dict(repo=d["repo"], n=0, npass=0, steps=set()))
                t["n"] += d["n"]
                t["npass"] += d["npass"]
                t["steps"].update(d["steps"])
            if (i+1) % 30 == 0:
                print(f"  done {i+1}/{len(files)}", flush=True)
    for t in tasks.values():
        t["nsteps"] = len(t["steps"])
        del t["steps"]
        t["pass_rate"] = t["npass"] / t["n"]
    out = [dict(task=k, **v) for k, v in tasks.items()]
    out.sort(key=lambda r: (r["repo"], -r["pass_rate"]))
    with open(os.path.join(os.path.dirname(__file__), "task_scan.json"), "w") as f:
        json.dump(out, f)
    print(f"saved task_scan.json : {len(out)} tasks", flush=True)

if __name__ == "__main__":
    main()
