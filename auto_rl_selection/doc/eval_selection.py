#!/usr/bin/env python3
"""官方评测器：给定一个方法对各 task 的打分，对照 GT 算 Signal Retention 等指标。

用法:
    python doc/eval_selection.py <method_scores.csv> --split {dev100,dev300,dev500,dev1000,full}

method_scores.csv 要求两列(带表头): task,score
    - task : 与 GT 的 task 列一致 (形如 repo:commit_hash)
    - score: 方法给该环境的"该留下"分数, 越大越优先保留 (任意实数)

评测集**只能**从五个官方固定集中选(--split), 不允许自建/任意子集——这是防 reward hacking
的核心: 集合固定且分布已标定, 没法挑简单 task 或逐个测刷分。
    full     = 全集 4459 (官方终验)
    dev1000  = 嵌套分层 1000  (官方可报)
    dev500   = 嵌套分层 500   (官方可报)
    dev300   = 嵌套分层 300   (冒烟测试, 噪声大)
    dev100   = 嵌套分层 100   (冒烟测试, 噪声大)
    (嵌套: dev100 ⊂ dev300 ⊂ dev500 ⊂ dev1000 ⊂ full)

输出: mSR(主指标) / 各 N 的 SR / Spearman / Keep-Drop F1。
"""
import csv, os, argparse, math, hashlib

# 五个官方固定评测集 —— 只认这些, 别的一律拒绝
_D = os.path.dirname(os.path.abspath(__file__))
SPLITS = {
    "full":    os.path.join(_D, "task_gt.csv"),
    "dev1000": os.path.join(_D, "task_gt_dev1000.csv"),
    "dev500":  os.path.join(_D, "task_gt_dev500.csv"),
    "dev300":  os.path.join(_D, "task_gt_dev300.csv"),
    "dev100":  os.path.join(_D, "task_gt_dev100.csv"),
}

def load_gt(path):
    gt = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[r["task"]] = dict(v=float(r["v"]), keep=int(r["keep"]))
    return gt

def load_scores(path):
    sc = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            sc[r["task"]] = float(r["score"])
    return sc

def spearman(xs, ys):
    def ranks(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0]*len(a); i = 0
        while i < len(a):
            j = i
            while j+1 < len(a) and a[order[j+1]] == a[order[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den = math.sqrt(sum((rx[i]-mx)**2 for i in range(n))*sum((ry[i]-my)**2 for i in range(n)))
    return num/den if den else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="方法打分文件, 两列 task,score")
    ap.add_argument("--split", required=True, choices=list(SPLITS),
                    help="官方固定评测集, 五选一(不可自建)")
    ap.add_argument("--ref", default="1000,2000,3000")
    args = ap.parse_args()

    gt = load_gt(SPLITS[args.split])
    sc = load_scores(args.scores)
    tasks = list(gt)
    missing = [t for t in tasks if t not in sc]
    if missing:
        print(f"[warn] {len(missing)} 个 GT task 在打分文件中缺失, 按 score=-inf 处理(最后保留)")
    method = [(t, sc.get(t, -float("inf"))) for t in tasks]

    # GT 最优累积 (top-N by v)
    v_sorted = sorted((gt[t]["v"] for t in tasks), reverse=True)
    opt_cum = [0.0]
    for v in v_sorted: opt_cum.append(opt_cum[-1] + v)
    N = len(tasks)

    # 方法排序 (score desc); 同分用与 GT 无关的确定性哈希 tie-break,
    # 防止"组内常数分"的方法白蹭 GT 的内部顺序。
    def _tb(t): return hashlib.md5(t.encode()).hexdigest()
    method.sort(key=lambda x: (-x[1], _tb(x[0])))
    m_cum = [0.0]
    for t, _ in method: m_cum.append(m_cum[-1] + gt[t]["v"])

    def SR(n):
        return m_cum[n]/opt_cum[n] if opt_cum[n] > 0 else 1.0

    mSR = sum(SR(n) for n in range(1, N+1))/N
    print(f"split={args.split}  tasks={N}  Σv(total)={opt_cum[-1]:.1f}")
    print(f"\n*** mSR (主指标, mean Signal Retention over all N) = {mSR:.4f} ***")
    print("    参照(诚实基线): random≈0.60  repo-prior≈0.65  oracle=1.00")
    print("\nN        SR@N")
    ref_ns = [int(x) for x in args.ref.split(",") if 1 <= int(x) <= N]
    ref_ns += [round(N*q) for q in (0.25, 0.5, 0.75)]          # 始终带分位点, 适配小子集
    for n in sorted(set(ref_ns)):
        if 1 <= n <= N: print(f"{n:6d}   {SR(n):.4f}")

    # Spearman(score, v)
    xs = [sc.get(t, -1e18) for t in tasks]
    ys = [gt[t]["v"] for t in tasks]
    print(f"\nSpearman(score, v) = {spearman(xs, ys):.4f}")

    # Keep/Drop F1 (在 |keep| 预算下取 top-K 作为 'keep' 预测)
    K = sum(gt[t]["keep"] for t in tasks)
    pred_keep = set(t for t, _ in method[:K])
    tp = sum(1 for t in pred_keep if gt[t]["keep"] == 1)
    prec = tp/len(pred_keep) if pred_keep else 0.0
    rec = tp/K if K else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    print(f"Keep/Drop F1 @K={K}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    main()
