import os, json, glob, collections, sys
import torch
from multiprocessing import Pool

ROOT = "/shared_workspace_mfs/yanruo/github/auto_rl_selection/data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c"
RD = os.path.join(ROOT, "rollout_data")

def proc(path):
    step = int(os.path.basename(path)[:-3])
    o = torch.load(path, map_location="cpu", weights_only=False)
    samples = o["samples"]
    n = len(samples)
    cat = collections.Counter()
    repo_tot = collections.Counter()
    repo_pass = collections.Counter()
    status_c = collections.Counter()
    npass = 0
    rsum = 0.0
    grp = collections.defaultdict(list)  # group_index -> [reward values]
    for s in samples:
        rw = s.get("reward") or {}
        val = rw.get("value", 0.0) if isinstance(rw, dict) else 0.0
        c = rw.get("category", "?") if isinstance(rw, dict) else "?"
        st = s.get("status", "?")
        md = s.get("metadata") or {}
        repo = md.get("repo") or md.get("docker_image", "?")
        # normalize repo: docker image like namanjain12/numpy_final:hash -> numpy
        if "/" in str(repo):
            repo = str(repo).split("/")[-1].split(":")[0].replace("_final", "")
        cat[c] += 1
        status_c[st] += 1
        repo_tot[repo] += 1
        if val >= 1.0:
            npass += 1
            repo_pass[repo] += 1
        rsum += float(val)
        grp[s.get("group_index", -1)].append(float(val))
    # group-level: fraction of groups with >=1 solve and all-solve / all-fail
    ngroups = len(grp)
    g_any = sum(1 for v in grp.values() if any(x >= 1.0 for x in v))
    g_all = sum(1 for v in grp.values() if all(x >= 1.0 for x in v))
    g_none = sum(1 for v in grp.values() if all(x < 1.0 for x in v))
    return dict(step=step, n=n, npass=npass, rmean=rsum/max(n,1),
                cat=dict(cat), status=dict(status_c),
                repo_tot=dict(repo_tot), repo_pass=dict(repo_pass),
                ngroups=ngroups, g_any=g_any, g_all=g_all, g_none=g_none)

def main():
    files = sorted(glob.glob(os.path.join(RD, "*.pt")), key=lambda p: int(os.path.basename(p)[:-3]))
    print(f"scanning {len(files)} files...", flush=True)
    results = []
    with Pool(24) as pool:
        for i, r in enumerate(pool.imap_unordered(proc, files)):
            results.append(r)
            if (i+1) % 20 == 0:
                print(f"  done {i+1}/{len(files)}", flush=True)
    results.sort(key=lambda r: r["step"])
    with open(os.path.join(os.path.dirname(__file__), "rollout_scan.json"), "w") as f:
        json.dump(results, f)
    print("saved rollout_scan.json", flush=True)

if __name__ == "__main__":
    main()
