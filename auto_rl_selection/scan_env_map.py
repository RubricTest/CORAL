import os, glob, csv, collections
import torch
from multiprocessing import Pool

ROOT = "/shared_workspace_mfs/yanruo/github/auto_rl_selection/data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c"
RD = os.path.join(ROOT, "rollout_data")

FIELDS = ["docker_image", "repo_name", "commit_hash", "data_source", "ability"]

def proc(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    local = {}
    for s in o["samples"]:
        md = s.get("metadata") or {}
        repo = md.get("repo_name", "?"); commit = md.get("commit_hash", "?")
        key = f"{repo}:{commit}"
        if key in local: continue
        local[key] = {f: md.get(f) for f in FIELDS}
        # 是否还能拿到定义 env/reward 的字段
        local[key]["has_problem"] = bool(md.get("problem_statement"))
        local[key]["has_tests"] = bool(md.get("expected_output_json"))
    return local

def main():
    files = sorted(glob.glob(os.path.join(RD, "*.pt")), key=lambda p: int(os.path.basename(p)[:-3]))
    env = {}
    with Pool(24) as pool:
        for local in pool.imap_unordered(proc, files):
            for k, v in local.items():
                env.setdefault(k, v)
    rows = [dict(task=k, **v) for k, v in env.items()]
    rows.sort(key=lambda r: (r["repo_name"] or "", r["task"]))
    with open(os.path.join(os.path.dirname(__file__), "doc/task_env_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "repo", "commit_hash", "docker_image", "data_source", "ability", "has_problem", "has_tests"])
        for r in rows:
            w.writerow([r["task"], r["repo_name"], r["commit_hash"], r["docker_image"],
                        r["data_source"], r["ability"], r["has_problem"], r["has_tests"]])
    print(f"tasks={len(rows)}, wrote doc/task_env_map.csv")
    # 验证命名规律: docker_image 是否 = <prefix>/<repo>_final:<commit>
    derivable = sum(1 for r in rows if r["docker_image"] and r["commit_hash"] in (r["docker_image"] or ""))
    print(f"docker_image 含 commit_hash 的: {derivable}/{len(rows)}")
    # 每个 repo 的 image 前缀样式
    pat = collections.Counter()
    for r in rows:
        di = r["docker_image"] or ""
        name = di.split(":")[0] if di else "?"
        pat[name] += 1
    print("image 名(去tag) 分布:")
    for k, v in pat.most_common():
        print(f"  {v:5d}  {k}")
    print("has_problem all True:", all(r["has_problem"] for r in rows),
          " has_tests all True:", all(r["has_tests"] for r in rows))

if __name__ == "__main__":
    main()
