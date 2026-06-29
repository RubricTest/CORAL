# 数据总览：miles RL 轨迹 dump (260401_231502_cd7fb81c)

> 来源：`/shared_workspace_mfs/fenglin/miles_dump_details/260401_231502_cd7fb81c.tar.zst`（43G 压缩）
> 解压位置：`./data/miles_qwen3_32b_multi_r2e_gym_slime_deepswe_epoch0_260401_231502_cd7fb81c/`（389G）
> 训练：Qwen3-32B，r2e-gym / slime deepswe，agentic SWE RL（GRPO），共 **181 步**
> 生成时间：2026-06-24

---

## 1. 目录构成

| 子项 | 数量 | 大小 | 说明 |
|---|---|---|---|
| 顶层 `*.txt` | 101,076 | ~100G | 人类可读的逐条 trajectory dump |
| `rollout_data/*.pt` | 181 | 33G | 每 step 一个，含 512 条原始采样（**结构化、含 reward/status**） |
| `train_data/*.pt` | 11,584 | 254G | `step_rank.pt`（181 × 64），含已算好的 advantages/returns |
| `tokenizer/` | 7 | 18M | HF Qwen3 分词器 |

### 1.1 `.txt` 轨迹格式
文件名：`...-swe-<repo>-<taskhash>-<ts>-R<rolloutID>.txt`
- 头部 11 行元信息：`Status / Total/Prompt/Response/Trainable/Masked tokens / session_id / num_records / context_overflow / timing`
- PROMPT 段（masked，不训练）：system prompt（4 工具 `file_editor`/`execute_bash`/`search`/`finish`）+ issue + repo 文件树
- trajectory 段：`<think>…</think>`、`<function=…><parameter=…>…</parameter></function>` 工具调用、`[-----] output of [tool]:` 工具返回

### 1.2 `rollout_data/N.pt` 结构（筛选最该用这套）
`dict{ rollout_id, samples[512] }`，每条 sample 关键字段：
`prompt(=docker image)`, `tokens`, `response`, `response_length`, `reward{value,category}`, `status`, `loss_mask`, `rollout_log_probs`, `metadata{docker_image,repo}`, `group_index`（每组 8 条 → 64 组 × 8 = 512，即 GRPO group）。

### 1.3 `train_data/N_R.pt` 结构
`dict{ rollout_id, rank, rollout_data{ tokens, response_lengths, rewards, truncated, loss_masks, rollout_log_probs, advantages, returns, raw_reward, total_lengths, correct_entropy, … } }`，已是可直接喂 trainer 的张量。

---

## 2. Repo / Task 分布

数据有两个层级：**repo（仓库）= 10 个**，**task（具体 issue，用 `<repo>-<taskhash>` 唯一标识）= 4,576 个**。一个 repo 下含多个 task；每个 task 在 181 步训练中被多次采样，平均约 22 条轨迹/task。

| repo | task 数 | 轨迹数 | 占比 | 平均 rollout/task |
|---|---|---|---|---|
| pandas | 1,444 | 31,879 | 31.5% | 22.1 |
| numpy | 781 | 17,352 | 17.2% | 22.2 |
| pillow | 618 | 13,625 | 13.5% | 22.0 |
| orange3 | 482 | 10,582 | 10.5% | 22.0 |
| aiohttp | 299 | 6,537 | 6.5% | 21.9 |
| tornado | 261 | 5,836 | 5.8% | 22.4 |
| scrapy | 215 | 4,841 | 4.8% | 22.5 |
| pyramid | 189 | 4,154 | 4.1% | 22.0 |
| datalad | 179 | 3,928 | 3.9% | 21.9 |
| coveragepy | 108 | 2,342 | 2.3% | 21.7 |
| **合计** | **4,576** | **101,076** | **100%** | **~22** |

> task 是 `auto_rl_selection` 做筛选的最小单位（保留/降采样/剔除），数量大致与 repo 体量成正比。

---

## 3. 训练结果分析（扫 181 个 rollout_data，92,672 条采样）

### 3.1 总体
- **整体 pass 率 30.4%**（28,210 / 92,672）
- status：completed 69.6% / truncated 30.4%

**reward.category 分布**

| 类别 | 占比 | 含义 |
|---|---|---|
| tests_failed | 58.7% | 改了但测试没过 |
| **pass** | **30.4%** | 解决 ✅ |
| test_count_mismatch | 9.1% | 测试数对不上 |
| eval_error | 1.6% | 评测崩 |
| timeout / test_run_error | 0.2% | 极少 |

### 3.2 各 repo pass 率（难度差异明显）

| repo | tot | pass | pass率 |
|---|---|---|---|
| scrapy | 4,448 | 2,055 | 46.2% （最易） |
| pyramid | 3,800 | 1,566 | 41.2% |
| pillow | 12,584 | 4,997 | 39.7% |
| aiohttp | 5,960 | 2,189 | 36.7% |
| tornado | 5,296 | 1,734 | 32.7% |
| numpy | 15,872 | 4,780 | 30.1% |
| datalad | 3,624 | 975 | 26.9% |
| pandas | 29,344 | 7,245 | 24.7% （占比最大且偏难）|
| orange3 | 9,624 | 2,300 | 23.9% |
| coveragepy | 2,120 | 369 | 17.4% （最难）|

### 3.3 训练曲线
- pass 率：step 1 **23.4%** → step 181 **35.4%**；前 ~40 步快速爬升（24%→32%），之后进入平台期在 30–34% 震荡。
- reward 均值同步 0.238 → 0.354。
- 结论：**确实在学，但后 ~140 步收益递减**。

| step区间 | pass率 | 有效组率 | 全错组率 |
|---|---|---|---|
| 1-10 | 23.8% | 46.1% | 50.0% |
| 31-40 | 32.2% | 44.2% | 45.0% |
| 91-100 | 34.3% | 43.0% | 44.2% |
| 171-180 | 28.6% | 40.9% | 49.4% |
| 181 | 35.4% | 57.8% | 32.8% |

### 3.4 ⭐ GRPO group 信号（数据筛选核心依据）
每组 8 条同 task 采样，共 11,584 组：

| 组类型 | 占比 | 价值 |
|---|---|---|
| **全错 (8/8 fail)** | **46.2%** | ❌ advantage 全 0，无梯度信号（task 太难） |
| 全对 (8/8 pass) | 9.1% | ❌ advantage 全 0，无信号（task 太易） |
| **混合** | **44.7%** | ✅ 唯一产生有效 advantage |

**含义**：约 **55% 的采样组对 GRPO 没有学习信号**。这正是 `auto_rl_selection` 的切入点——提前识别/降采样注定全错（太难）或全对（太易）的 task，可把有效信号密度从 ~45% 推向接近 100%，显著省算力。

---

## 4. 逐 task 难度（筛选直接依据）

按 task 标识 `repo:commit_hash` 聚合全程 181 步采样：rollout_data 实际覆盖 **4,459 个 task**（每 task 采样 8~24 次，中位数 24），整体 pass 率 30.4%。

### 4.1 难度分桶（按全程 pass 率）

| 桶 | pass率 | task 数 | 占 task | 占样本 | 价值 |
|---|---|---|---|---|---|
| **全错** | 0% | 1,759 | 39.4% | 38.9% | ⛔ 全程没解出，零梯度信号（太难/可能不可解）|
| 难 | 0–25% | 731 | 16.4% | 16.9% | 信号稀疏 |
| **可学** | 25–75% | 1,108 | 24.8% | 24.9% | ⭐ 黄金区，advantage 信号最足 |
| 易 | 75–100% | 668 | 15.0% | 15.3% | 信号渐弱 |
| **全对** | 100% | 193 | 4.3% | 4.0% | ⛔ 已学会，零梯度信号 |

> **无信号 task（全错+全对）= 1,952 个（43.8%），消耗 42.9% 采样预算却不产生任何 GRPO 学习信号。** 这是 `auto_rl_selection` 最直接的优化空间：剔除/降采样这批，把预算让给「可学」区。

### 4.2 各 repo 难度构成

| repo | task | 全错 | 难 | 可学 | 易 | 全对 | pass率 |
|---|---|---|---|---|---|---|---|
| pandas | 1,408 | 585 | 279 | 373 | 151 | 20 | 24.7% |
| numpy | 765 | 278 | 131 | 233 | 95 | 28 | 30.1% |
| pillow | 602 | 184 | 83 | 166 | 139 | 30 | 39.7% |
| orange3 | 466 | 234 | 75 | 81 | 61 | 15 | 23.9% |
| aiohttp | 291 | 90 | 54 | 65 | 55 | 27 | 36.7% |
| tornado | 253 | 110 | 29 | 60 | 39 | 15 | 32.7% |
| scrapy | 213 | 70 | 16 | 46 | 56 | 25 | 46.2% |
| pyramid | 183 | 60 | 23 | 41 | 38 | 21 | 41.2% |
| datalad | 175 | 81 | 29 | 32 | 24 | 9 | 26.9% |
| coveragepy | 103 | 67 | 12 | 11 | 10 | 3 | 17.4% |

> pandas/orange3/coveragepy 的「全错」task 占比最高（~40–65%），是冗余采样的重灾区。

> ⚠️ 注意：此处 pass 率是**全程（181 步）累计**，会把早期/后期混在一起。「全错」基于该 task 全部 8–24 次采样都没解出，可信度较高；但若要做课程式调度，需结合 step 维度看动态变化（见 §5）。

## 5. Step 维度动态（判断「全错」task 是否真该剔除）

每个 task 在 181 步里只被采样 ~3 次（中位 24 样本 = 3 次 × 8），按它出现的各 step 上是否解出，给每个 task 打一个 **dynamics 标签**：

| dynamics | task 数 | 占比 | 含义 / 处置 |
|---|---|---|---|
| steady | 2,301 | 51.6% | 有时解出有时不解 —— 正常保留 |
| **never_solved** | 1,759 | 39.4% | 全程从未解出 —— 剔除候选（见下） |
| **emerging_late** | 266 | 6.0% | ⭐ 早期全错、**后段开始解出** —— **绝不能剔除** |
| lost_late | 133 | 3.0% | 早期能解、后段反而不行 —— 关注（可能退化） |

### 5.1 「全错」task 是否真不可解？看它最后一次被采样的 step

| last_step 区间 | never_solved task 数 |
|---|---|
| 0–30 | 17 |
| 30–60 | 30 |
| 60–90 | 50 |
| 90–120 | 99 |
| 120–150 | 790 |
| 150–181 | 773 |

- **1,563 / 1,759（89%）的全错 task 在后期（step≥120）仍被采到且依旧全错** → 高置信度该剔除/降采样（模型练到很后面也解不出）。
- 仅 **47 个**只在前期（last_step<60）出现过就没再采样 → 证据不足，不宜直接判死。
- 各 repo 这个「后期仍全错」比例都很高（82–94%），说明全错 task 普遍是真·难，而非采样早。

### 5.2 emerging_late 是反例警示
266 个 task 早期 8/8 全错、后段却能解出 50–70%（如 `pandas 11:0/8;74:7/8;136:4/8`）。**纯按早期 pass 率筛会误杀这批正在被学会的 task** —— 所以筛选必须用「全程是否解出 + 后期表现」，不能只看前期。

### 5.3 筛选建议（基于以上）
- **可安全剔除**：`never_solved` 且 `last_step≥120`（1,563 个，约 35% 的 task）。
- **剔除/降采样**：`trivial`（全对，193 个）。
- **保留并优先**：`learnable`、`emerging_late`。
- 合计可省下约 1,563 + 193 ≈ **1,756 个 task 的采样预算（~38%）**，几乎不损失有效梯度信号。

## 6. 产物（CSV 可直接看）
- **`doc/task_difficulty_bystep.csv`** ⭐ 最终交付，4,459 行，列：
  `task, repo, n, npass, pass_rate, n_steps, first_step, last_step, mean_step, first_solved_step, early_pass_rate, late_pass_rate, bucket, dynamics, pass_by_step`
  其中 `pass_by_step` 形如 `11:0/8;74:7/8;136:4/8`（每个 step 的 解出/采样 数），`dynamics` 列直接给出 never_solved / emerging_late / steady / lost_late。
- `doc/task_difficulty.csv`：精简版（无 step 维度）。
- 脚本与中间数据（项目根目录）：`scan_rollouts.py`→`rollout_scan.json`（181 步全局明细）、`scan_tasks.py`→`task_scan.json`、`scan_tasks_bystep.py`→`task_bystep.json`。

## 7. 后续可做
1. 用 `task_difficulty_bystep.csv` 落地：剔除 `dynamics=never_solved & last_step≥120` + `bucket=trivial`。
2. 跟踪「有效组率」随 step 变化（step 171-180 已降到 40.9%），评估动态重采样。
3. 把筛选策略落成 RL pipeline 里的 data selector。
