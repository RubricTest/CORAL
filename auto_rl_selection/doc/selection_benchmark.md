# RL 环境选择 Benchmark（autoresearch 靶子）

> 目标：从 4,459 个 SWE RL 环境（task）中，**用 base model + 环境信息**挑出一个子集用于训练，使保留的「学习信号」最大。
> 本文件定义 **GT** 和 **score**，给 autoresearch 一个固定、可迭代优化的目标。
> 数据背景见 `doc/data_overview.md`。暂只做**静态选择**（不考虑 RL 不同 step 的动态选择）。

---

## 1. 问题设定

- **候选池**：4,459 个环境，每个由 `task = repo:commit_hash` 唯一标识（见 `doc/task_gt.csv`）。
- **task → env 映射（确定性）**：`docker_image = namanjain12/<repo>_final:<commit_hash>`。映射表 `doc/task_env_map.csv`。要真正实例化一个 env 需三样（均在 dump 的 per-task metadata）：`docker_image`（测试床）+ `problem_statement`（issue→prompt）+ `expected_output_json`（测试 oracle→reward）。方法的在线 rollout 通过率与 GT 的 `p_i` 同源可比。
- **方法 M 的输入**：
  - 环境信息：`problem_statement`、`repo_name`、`commit_hash`、测试清单 `expected_output_json`、`docker_image` 等（均可从 dump 的 sample metadata 取，或从 docker 环境实跑）。
  - **base model**（RL 起点 policy）：可对环境采样 rollout、读 logprob/entropy。
  - 一个 **cost 预算**（base-model token / rollout 次数）。
- **方法 M 的输出**：对全部 4,459 个 task 各给一个分数 `s_i`（越大=越该保留）。**给全集打分，不固定子集大小**——预算 N 在评测时扫描。
- **铁律（防作弊）**：M **不得**使用本 dump 的 run 统计或 GT（`p_i / v_i / pass_by_step / reward` 等都是答案）。只能用 base-model 信号 + 静态环境特征。

---

## 2. GT（标准答案）：每个环境的学习信号价值

从完整 181 步 run 算每个 task 的经验通过率 `p_i = npass_i / n_i`（n_i = 该 task 全程被采样数，8~24）。GRPO 一组 `G=8` 个样本，只有**非全对非全错**才产生非零 advantage。定义环境价值：

```
v_i = 1 − p_i^G − (1−p_i)^G        # G=8；该环境每步产生学习信号的期望概率
```

| p_i | v_i | 含义 |
|---|---|---|
| 0（全错）| 0 | 无信号，该剔 |
| 1（全对）| 0 | 无信号，该剔 |
| 0.5 | ≈1.0 | 黄金环境 |

- **GT 是全集上的连续打分** `{v_i}`，不预设 N。最优选择 = 任意预算 N 下取 `v_i` 最大的 N 个。
- 衍生二分标签：`keep_i = 1 if 0 < p_i < 1 else 0`（全集中 keep=2,507，dead=1,759，trivial=193）。
- 全局量：`Σv_i = 1917.3`（理论可得总信号），`mean v = 0.430`。
- 文件：**`doc/task_gt.csv`**，列 `task, repo, n, npass, p, v, keep, gt_class, gt_rank`。

---

## 3. Score（评测方法好坏）

设方法输出排序后，预算 N 下选中集 `S_N` = 方法 top-N；GT 最优 = `v` top-N。

### 主指标 — Signal Retention（信号保留率）
```
SR@N = Σ_{i∈S_N} v_i  /  Σ_{top-N by v} v_i      ∈ [0,1]
```
"在预算 N 下保住了理论上限多少比例的学习信号"。

**汇总主标量 mSR**（与预算无关，对所有 N 求平均）：
```
mSR = (1/T) Σ_{N=1..T} SR@N            T = 4459
```
完美方法（=GT 排序）mSR=1.0。

### 辅助指标
- **Spearman ρ(s, v)** —— 整体排序质量。
- **Keep/Drop F1 @K**（K=2,507）—— 对「该剔 vs 该留」判得准不准。
- **cost**（base-model token 数）—— 与 SR 一起画 Pareto；**autoresearch 的真正目标 = 推 SR–cost 前沿**（同样 SR 用更少算力，或同样算力更高 SR）。

### 基线参照（诚实值，全集）
| 方法 | mSR |
|---|---|
| 随机 | ~0.60 |
| repo 先验（按 repo 平均 v 排，组内 GT-无关 tie-break）| ~0.65 |
| **Oracle = GT** | **1.00** |

→ 可优化空间在 **0.65 → 1.0**。repo 这种粗特征天花板就 ~0.65（区分不了 repo 内部好坏，Spearman≈0.12）；要往上走必须用细信号。强基线预期是「小 K（1–2 次）base rollout 估 p_i」，autoresearch 要在更低 cost 下逼近它、或叠加特征超过它。

> 注：评测器对同分 task 用**与 GT 无关的哈希 tie-break**——否则"组内常数分"的方法会白蹭 `task_gt.csv` 的 `(repo,-v)` 内部排序，把 repo 先验虚高到 0.745（已修正）。

---

## 4. 怎么评测（官方脚本）

方法产出 `method_scores.csv`（两列 `task,score`），然后：
```bash
python doc/eval_selection.py method_scores.csv --gt doc/task_gt.csv
```
输出 mSR / 各 N 的 SR / Spearman / F1。评测器源码 `doc/eval_selection.py`（GT 在内部加载，方法看不到）。

---

## 4b. 快速验证 dev 集 & 防 reward hacking

方法在全集 4,459 个环境上跑 base-model rollout 很贵，迭代期可只在**同分布小集**上验证，最后才上全集。

**官方分层 dev 集**（按 `repo × 难度类别` 分层抽样；**嵌套** dev100 ⊂ dev300 ⊂ dev500 ⊂ dev1000 ⊂ full，换档集合一致、提升可单调追踪；分布与全集对齐）：

| 文件 | N | mean_v | dead/learn/triv | random mSR | repo-prior mSR |
|---|---|---|---|---|---|
| `doc/task_gt_dev100.csv` | 100 | 0.394 | 39/57/4% | 0.57 | 0.65 |
| `doc/task_gt_dev300.csv` | 300 | 0.412 | 39/57/4% | 0.61 | 0.66 |
| `doc/task_gt_dev500.csv` | 500 | 0.416 | 40/56/4% | 0.59 | 0.66 |
| `doc/task_gt_dev1000.csv` | 1000 | 0.425 | 39/56/4% | 0.60 | 0.67 |
| `doc/task_gt.csv`（全集）| 4459 | 0.430 | 39/56/4% | 0.60 | 0.65 |

> 基线跨档稳定在 random≈0.60 / repo-prior≈0.65，说明分层集标定良好。**dev100/dev300 仅供冒烟测试（噪声大），官方分数请在 dev500+ 或全集上报。**

用法：`python doc/eval_selection.py scores.csv --split dev500`

**防 hack 的核心约束**：评测集**只能从这五个官方固定集中选**（`--split` 五选一），**不允许自建/任意子集**。集合固定且分布已标定，因此无法靠"挑简单 task"或"逐个 task 测（N=1 时 SR 恒为 1 → mSR 虚高到 1.0）"刷分。

> 规范：**官方分数在 `dev500 / dev1000 / full` 上报**（dev100/dev300 噪声大，仅冒烟测试）。为防过拟合，dev 用于迭代、full 用于终验。

## 5. 给 autoresearch 的候选方法方向（非穷举）
- **小样本通过率估计**：base model 对每个环境采 k 次（k=1,2,4…），用 `p̂` 估 `v̂`；研究"最省 k / 最优估计量（Beta 后验、early-stop）"。
- **零 rollout 特征预测**：problem_statement 长度、测试数、patch 行数、repo、代码库规模、issue 类型 → 回归/分类预测 v。
- **base-model 内省信号**：首步 token entropy、logprob、自评置信度，单次前向就能拿。
- **embedding/相似度**：与已知 learnable 环境的相似度迁移。
- **混合**：零成本特征先粗筛，再对边界环境补少量 rollout。

---

## 6. 诚实的 caveat（必须知道）
1. **GT 是 run-specific**：`p_i` 是在那条 run 不断变化的 policy 下测的，不是 policy 无关的环境属性；且每 task 只有 8~24 样本，`p_i` 粒度粗（`0/24` 很可信，`1/8` 边界噪声大）。因此 mSR 的**粗分辨**（keep/drop、dead/learnable）可信，**精细排序**偏噪声——score 设计已让主信号落在 keep/drop 上。
2. **更干净的 v2 GT（可选，需算力）**：用 base ckpt(step0) 对全部 4,459 个环境各跑 K≥16 次 rollout，得 policy 无关的 `p_base`，重算 `v`。pipeline 与 score 完全复用，只换 `task_gt.csv`。
3. **终极验证（贵）**：拿方法选出的子集真跑一段 RL，对比「全 4k / 随机同量」的最终模型——SR 只是它的廉价代理。

---

## 7. 文件清单
| 文件 | 作用 |
|---|---|
| `doc/task_gt.csv` | ⭐ GT 靶子（全集 4459）：每个环境的 `p / v / keep / gt_class / gt_rank` |
| `doc/task_gt_dev{100,300,500,1000}.csv` | ⭐ 嵌套分层 dev 集，快速验证用（同分布） |
| `doc/task_env_map.csv` | task → docker_image 映射（方法实例化 env 用） |
| `doc/task_env_inputs.jsonl` | ⭐ 每 task 一行的 env **输入规格**（4459，43MB）：`task,split,repo,commit_hash,docker_image,data_source,ability,problem_statement,prompt`。**只含输入，已排除 expected_output_json/patch/reward 等答案**。`split` 标最小所属档（嵌套），取某档 = 取 split ≤ 该档的行 |
| `doc/eval_selection.py` | ⭐ 官方评测器：吃 `method_scores.csv`(列 `task,score`) 出 mSR；内置最小规模+代表性防 hack |
| `doc/data_overview.md` | 数据背景与基础分析 |
| `doc/task_difficulty_bystep.csv` | 逐 step 明细（仅供分析/造 v2 GT 参考，**方法禁用**）|
