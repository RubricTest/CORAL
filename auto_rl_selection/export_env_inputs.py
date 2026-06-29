import os, glob, csv, json
import torch
from multiprocessing import Pool

BASE = "/shared_workspace_mfs/yanruo/github/auto_rl_selection"
ROOT = os.path.join(BASE, "data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c")
RD = os.path.join(ROOT, "rollout_data")
OUT = os.path.join(BASE, "doc/task_env_inputs.jsonl")

# 只保留输入相关字段; 显式排除答案/奖励/结果类
INPUT_FIELDS = ["docker_image", "repo_name", "commit_hash", "data_source",
                "ability", "problem_statement", "prompt"]

def proc(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    local = {}
    for s in o["samples"]:
        md = s.get("metadata") or {}
        repo = md.get("repo_name", "?"); commit = md.get("commit_hash", "?")
        key = f"{repo}:{commit}"
        if key in local:
            continue
        local[key] = {f: md.get(f) for f in INPUT_FIELDS}
    return local

def split_tier_map():
    """嵌套档位 -> 每个 task 的最小所属档 (dev100<dev300<dev500<dev1000<full)。"""
    tiers = ["dev100", "dev300", "dev500", "dev1000"]
    tier_of = {}
    for t in tiers:
        with open(os.path.join(BASE, f"doc/task_gt_{t}.csv")) as f:
            for r in csv.DictReader(f):
                tier_of.setdefault(r["task"], t)   # 第一次(最小档)命中即定
    # 全集里其余的归 full
    with open(os.path.join(BASE, "doc/task_gt.csv")) as f:
        for r in csv.DictReader(f):
            tier_of.setdefault(r["task"], "full")
    return tier_of

def main():
    files = sorted(glob.glob(os.path.join(RD, "*.pt")), key=lambda p: int(os.path.basename(p)[:-3]))
    env = {}
    with Pool(24) as pool:
        for local in pool.imap_unordered(proc, files):
            for k, v in local.items():
                env.setdefault(k, v)
    tier = split_tier_map()

    rows = []
    for key, v in env.items():
        rows.append({
            "task": key,
            "split": tier.get(key, "full"),
            "repo": v["repo_name"],
            "commit_hash": v["commit_hash"],
            "docker_image": v["docker_image"],
            "data_source": v["data_source"],
            "ability": v["ability"],
            "problem_statement": v["problem_statement"],
            "prompt": v["prompt"],          # [{role:system,...},{role:user,...}]
        })
    rows.sort(key=lambda r: (r["repo"], r["task"]))
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections
    by = collections.Counter(r["split"] for r in rows)
    print(f"wrote {OUT}  ({len(rows)} tasks, {os.path.getsize(OUT)/1e6:.1f} MB)")
    print("tier(最小所属档) 计数:", dict(by))
    # 验证: 取某档 = tier ∈ 该档及更小档, 数量应与 dev csv 行数一致
    order = ["dev100", "dev300", "dev500", "dev1000", "full"]
    cum = 0
    for t in order:
        cum += by.get(t, 0)
        print(f"  split<= {t:8s}: {cum} tasks")

if __name__ == "__main__":
    main()
