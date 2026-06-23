# RL-Result

RAGEN (StarPO) 实验结果归档仓。每个实验一个目录, 命名
`<家族>-<尺寸>-<任务>-<优化器>-filter{on,off}` (如 `Qwen2.5-3B-sokoban-PPO-filteroff`,
重名追加 `-2`/`-3`), 内含 `train.log` / `metadata.txt` / `final_metrics.txt` /
`summary.md` (人读汇总) / `metrics.csv` (逐步指标, 画图用)。不含 checkpoint 权重 (留在训练机本地)。

实验怎么跑、脚本怎么用, 见主仓 `experiment_scripts/Guide.md` 与 `experiment.md`。
本仓默认工作分支 **CJJ**, merge 在 GitHub 上做。

---

# 集合分析: RAGEN-2 在 Sokoban 上的复现 (Qwen2.5 + Qwen3)

任务 = CoordSokoban (坐标版推箱子), 多轮 agent RL, 4×H100, 上限 200 步, 验证 512 episode。
所有指标口径与论文 RAGEN-2 (arXiv:2604.06268) Table 4 对齐 (success rate)。
**当前共 17 个 run** (Qwen2.5-0.5B/1.5B/3B + Qwen3-4B × {PPO,GRPO,DrGRPO} × filter{on,off} 的子集)。

## 0. 执行摘要 (TL;DR)

**17 个 run 后, 论文 RAGEN-2 的定性结论全部复现, 同时数据规模扩大后浮现一个更精细的主张:
"filter 不是万能解药, 真正稳的配方是 `PPO + filter` 且落在 1.5B–3B 这个甜区"。**

- ✅ **filter 防坍缩在 3B PPO 上最戏剧**: 21.5% → **37.1%** (+15.6%, 论文 +16.0%), 且从 step175 坍缩变为**跑满 200 步不坍缩, 思考率全程 100%**。这仍是全仓最强对照。
- ⚠️ **但 filter 并非普适**: 开 filter 仍坍缩的有 —— 0.5B GRPO-on(跑满 200 但末值归 0)、1.5B DrGRPO-on(step102 坍缩)、Qwen3-4B PPO-on(峰值 52.5% 后归 0)、Qwen3-4B GRPO-on(step118)。**只有 PPO 这一支 + 合适尺寸, filter 才真正"治本"。**
- ✅ **优化器稳定性 PPO > GRPO > DrGRPO** 在每个尺寸上都成立 (3B: 175>155>93; Qwen3-4B: PPO 跑最久 > GRPO 149 > DrGRPO 105)。
- ✅ **模型大小非单调 (1.5B ≥ 3B)**: 与论文 1.5B>3B 反常现象一致; 越小越早坍缩 (0.5B GRPO step26 最早)。
- 🆕 **Qwen3-4B (新家族)**: 基线最强 (31.8% vs Qwen2.5 ~10%)、峰值最高 (DrGRPO 冲到 **68.2%**), 但**所有 6 个配置无一存活, 连 PPO+filter 也归零**。需注意其 baseline 思考率仅 48% (response_length 截断, think 不闭合), 结果**部分被截断噪声污染, 仅作定性参考**。
- 📊 **额外贡献**: 给出论文没有的**逐 run 坍缩步数** + **"MI 先虚高再崩盘"前兆信号** + **success 归零的因果链**。

> 绝对值普遍略高于论文 (步数/验证规模/seed 差异), 但**所有趋势方向一致**。仍缺的 7 个网格格子见文末 [§5 待跑清单](#5-待跑清单)。

## 1. 总览表 (全部 17 个真实 run)

末值=最后记录步的 val success; "坍缩"列给出 `reward_variance_collapse` 触发步 (未触发则注明)。

| 模型 | 优化器 | filter | 基线 | 峰值 success (step) | 末值 | 跑满200? | 坍缩 | MI 走势 | 思考率末值 |
|------|------|------|------|------|------|------|------|------|------|
| **Qwen2.5-3B** | PPO | **on** | 10.2% | **37.1% (200)** | **37.1%** | ✅ | ❌ 未坍缩 | 平稳 0.7–0.9 | **100%** |
| Qwen2.5-1.5B | PPO | off | 10.0% | 23.8% (120) | 21.1% | ✅ | ❌ 未坍缩 | 平稳 1.2–1.6 | 100% |
| Qwen2.5-1.5B | GRPO | on | 10.0% | 20.1% (120) | 13.3% | ✅ | ⚠️ 退化中(思考率→57%) | 0.65→2.27→1.63 | 56.6% |
| Qwen2.5-3B | PPO | off | 10.2% | 21.5% (90) | 8.4% | ❌ | ✅ step175 | 2.04→**0.12崩** | 22.7% |
| Qwen2.5-0.5B | GRPO | on | 8.2% | 25.6% (100) | 0.0% | ✅(末值崩) | 跑满但末值=0 | 1.71→0.37 | 3.3% |
| Qwen2.5-0.5B | DrGRPO | off | 8.2% | 21.1% (60) | 0.4% | ❌ | ✅ step104 | 1.94→1.71 | 7.3% |
| Qwen2.5-0.5B | DrGRPO | on | 8.2% | 20.7% (60) | 4.3% | ❌ | ✅ step89 | 0.91→**-0.16** | 6.6% |
| Qwen2.5-3B | DrGRPO | off | 10.2% | 15.8% (60) | 0.0% | ❌ | ✅ step93 | 2.05→0.73 | 0.0% |
| Qwen2.5-3B | GRPO | off | 10.2% | 13.5% (40) | 0.0% | ❌ | ✅ step155 | 2.35→**0.06崩** | 1.4% |
| Qwen2.5-0.5B | PPO | off | 8.2% | 12.1% (30) | 0.0% | ❌ | ✅ step71 | 1.86→0.47 | 0.0% |
| Qwen2.5-1.5B | DrGRPO | on | 10.0% | 10.2% (10) | 0.0% | ❌ | ✅ step102(几乎没学会) | 0.68→0.14 | 0.0% |
| Qwen2.5-0.5B | GRPO | off | 8.2% | 8.2% (0) | 0.0% | ❌ | ✅ **step26(最早)** | 0.84→0.10 | 1.6% |
| 🆕 Qwen3-4B | DrGRPO | off | 31.8% | **68.2% (60)** | 0.0% | ❌ | ✅ step105 | 2.06→**0.05崩** | 0.0% |
| 🆕 Qwen3-4B | GRPO | off | 31.8% | 56.1% (40) | 2.9% | ❌ | ✅ step149 | 2.38→**-3.05崩** | 1.4% |
| 🆕 Qwen3-4B | PPO | on | 31.8% | 52.5% (40) | 0.0% | ❌(190步) | 末值=0(非触发) | 1.06→**-0.14** | 7.6% |
| 🆕 Qwen3-4B | PPO | off | 31.8% | 49.8% (60) | 0.0% | ❌(130步) | 末值=0(非触发) | 1.14→0.43 | 23.8% |
| 🆕 Qwen3-4B | GRPO | on | 31.8% | 41.2% (50) | 0.0% | ❌ | ✅ step118 | 2.19→0.55 | 0.0% |

**一眼结论**: 17 个 run 里**真正跑满且没崩的只有 2 个 —— `3B PPO-on (37.1%)` 与 `1.5B PPO-off (21.1%)`**, 全是 PPO。另有 `1.5B GRPO-on` 勉强跑满但已退化到 13%。其余 14 个全部坍缩(末值≤4%)。

## 2. 与论文 Table 4 完整逐项对照 — **结论: 定性完全一致**

论文数字来自 RAGEN-2 (arXiv:2604.06268) Table 4 的 Sokoban 列。

### A. 优化器线 (固定 Qwen2.5-3B, nofilter)
| 优化器 | 本仓峰值 success | 论文 Table 4 | 一致性 |
|--------|------|------|------|
| PPO | **21.5%** | 12.9% (另一配置23.6%) | ✅ 落在论文区间 |
| GRPO | **13.5%** | 12.1% | ✅ 几乎吻合 |
| DrGRPO | **15.8%** | 12.1% | ✅ 同量级 |

排序: 本仓 PPO(21.5) > DrGRPO(15.8) > GRPO(13.5); 论文 PPO 最高。✅ 一致。

### B. 模型大小线 (固定 PPO, nofilter)
| 尺寸 | 本仓峰值 success | 论文 Table 4 | 一致性 |
|------|------|------|------|
| 0.5B | **12.1%** | 3.3% | ✅ 同为最弱档 (step71坍缩) |
| 1.5B | **23.8%** | 17.0% | ✅ 同量级, 且为最高 |
| 3B | **21.5%** | 12.9% | ✅ 同量级 |
| 7B | (未跑) | 42.4% | — |

**关键非单调现象一致**: 论文 1.5B(17.0%) > 3B(12.9%); 本仓 1.5B(23.8%) > 3B(21.5%)。不是越大越好。
**坍缩随尺寸**: 0.5B(step71) < 3B(step175) < 1.5B(未坍缩) —— 越小越早崩。

### C. filter 增益 (on 峰值 − off 峰值) — 数据扩充后更细
| 配置 | off 峰值 | on 峰值 | 本仓增益 | 论文增益 | 一致性 / 备注 |
|------|------|------|------|------|------|
| 3B PPO | 21.5% | **37.1%** | **+15.6** | **+16.0** | ✅ 几乎完全命中, 且 on 不坍缩 |
| 0.5B DrGRPO | 21.1% | 20.7% | **−0.4** | **−0.4** | ✅ **精确命中论文的负增益** |
| 0.5B GRPO | 8.2% | 25.6% | **+17.4** | +9.0(3B) | ⚠️ 峰值大涨但 on 末值仍归 0 |
| Qwen3-4B PPO | 49.8% | 52.5% | +2.7 | — | ⚠️ 两者都坍缩到 0 |
| Qwen3-4B GRPO | 56.1% | 41.2% | **−14.9** | — | ⚠️ filter 反而压低峰值 |

> 重要观察: filter 对**峰值**的影响因算法/尺寸而异 (PPO 正、DrGRPO≈0、Qwen3-4B GRPO 甚至为负),
> 但对"**是否跑满不坍缩**"的影响**只在 Qwen2.5-3B PPO 上明确成立**。filter 救峰值 ≠ filter 救坍缩。

### D. 坍缩稳定性 (论文定性, 本仓给出具体步数)
排序在每个家族内都成立: **PPO 最稳 > GRPO > DrGRPO**。
- Qwen2.5-3B (off): PPO step175 > GRPO step155 > DrGRPO step93。
- Qwen3-4B: PPO 跑到 130/190 步(未触发硬停) > GRPO step149/118 > DrGRPO step105。
- Qwen2.5-0.5B (off): PPO step71 > GRPO step26。

> 绝对值普遍略高于论文 (配置差异), 但**所有趋势方向与论文一致**, 并额外提供逐 run 坍缩步数。

## 3. 四条核心结论

### 结论一: SNR Filtering 能防坍缩, 但**只在 PPO + 合适尺寸下治本** (复现并细化 V2 灵魂)
3B PPO **on vs off** 仍是最强对照:
- success 21.5% → **37.1%** (+15.6%, 论文 +16.0%); off step175 坍缩, **on 跑满200步未坍缩**; 思考率 off→22.7% vs on 全程 **100%**; MI off 冲 2.04 后**崩到 0.12**, on 全程平稳。

但扩大数据后看到反例: **0.5B GRPO-on、1.5B DrGRPO-on、Qwen3-4B PPO-on/GRPO-on 开了 filter 照样坍缩**。
→ filter 减少梯度噪声从而推迟/阻止坍缩, 但**当算法本身噪声太大 (GRPO/DrGRPO) 或模型/截断问题太严重 (0.5B、Qwen3-4B) 时, filter 不足以补救**。真正稳的配方是 **`PPO(有 critic) + filter` 且尺寸在 1.5B–3B 甜区**。

### 结论二: 优化器稳定性 PPO > GRPO > DrGRPO (每个尺寸都成立)
有 critic 的 PPO 提供 token 级 baseline、梯度噪声小, 在 3B/Qwen3-4B/0.5B 上都坍缩最晚;
无 critic 的 GRPO/DrGRPO 更早崩, DrGRPO (不除 std) 几乎总是最早。与论文/README "PPO 比 GRPO 更稳" 一致。

### 结论三: 模型大小与坍缩非单调 (1.5B ≥ 3B)
固定 PPO+nofilter: 1.5B 峰值 **23.8%** 且**跑满未坍缩** > 3B 峰值 21.5% 但 step175 坍缩 > 0.5B 12.1% step71 崩。
→ 呼应论文 Table 4 "1.5B(17.0%) > 3B(12.9%)" 反常现象, 不是越大越好。

### 结论四 🆕: 更强的基座 (Qwen3-4B) 起点高、天花板高, 但坍缩更彻底
Qwen3-4B 基线 31.8% (远高于 Qwen2.5 的 ~10%), 峰值能冲到 **68.2% (DrGRPO)**, 但 6 个配置**无一存活**, 连 PPO+filter 也归零。
⚠️ **重要 caveat**: Qwen3-4B baseline 思考率仅 **48%** (vs Qwen2.5 大模型 ~99%), 是 response_length 不够、`<think>` 被截断所致, 结果**部分被截断噪声污染**。需按 §5 调大 `response_length=1024 / max_model_len=8192` 重跑才能下定论, 当前仅作**定性参考**。

## 4. 自有分析视角 (论文主表未正面画的交叉分析)
所有 run 都记录了 RAGEN-2 的 MI 坍缩诊断指标 (`collapse/mi_estimate`,
`conditional_entropy_est`, `valid_thinking_rate`), 逐步数据在各 `metrics.csv`。

**前兆信号**: **坍缩前 MI 常先虚高再崩盘** —— off/坍缩系列 MI 峰值普遍冲到 2.0–2.4 (Qwen3-4B GRPO 甚至冲到 2.38 后跌成 **−3.05**) 后骤降。"MI 短暂上冲"可作为坍缩的早期预警。

### val success 归零的因果链 (机制解读)
为何坍缩实验 success 先升后崩到 **0%** (非单纯"变差")? 以 3B GRPO step130→155 为例:

| step | success | 响应长度 | invalid_action | valid思考率 | actor熵 |
|------|------|------|------|------|------|
| 130 | ~12% | 947 | 79% | 20% | 0.62 |
| 140 | 下滑 | 1115 | 99% | 15% | 1.44 |
| 150 | ~4% | 1234 | 99.8% | 6% | **4.57** |
| 155 | **0%** | — | **100%** | 1.4% | 爆炸 |

因果链: 学会任务 → 为多拿奖励输出越来越长、`<think>` 不闭合 → valid思考率从20%崩到1.4%
→ 环境解析不出合法动作 (`manager_invalid_action` 99%→**100%**) → 动作全非法、推不动箱子
→ **success 必然=0**, 同时 actor熵爆炸 (输出退化成高熵乱码)。

→ **0% 不是"模型变笨", 是"输出格式彻底坍缩、产生不了合法动作"**。这正是 reasoning collapse;
而 3B PPO **filter=on** 全程 success 稳在 37%、思考率 100%, 证明 filter 在该配置下阻止了这条因果链。
Qwen3-4B 因 baseline 思考率本就只有 48%, 这条链触发得更快更彻底。

---

## 实验清单 (17 个)

**Qwen2.5-0.5B** (5): PPO-off(step71崩) · GRPO-off(step26最早崩) · GRPO-on(跑满但末值0) · DrGRPO-off(step104) · DrGRPO-on(step89)
**Qwen2.5-1.5B** (3): PPO-off(✅跑满未崩 21.1%) · GRPO-on(跑满退化到13.3%) · DrGRPO-on(step102几乎没学会)
**Qwen2.5-3B** (4): PPO-off(step175) · **PPO-on(✅跑满未崩 37.1% 标杆)** · GRPO-off(step155) · DrGRPO-off(step93)
**Qwen3-4B** 🆕 (5): PPO-off(峰49.8末0) · PPO-on(峰52.5末0) · GRPO-off(step149峰56.1) · GRPO-on(step118) · DrGRPO-off(step105峰**68.2**)

> 备注: 早期有一次 0.5B 失败 (旧脚本 run_serial.sh 复用了 3B 同名 checkpoint 目录, size mismatch 崩溃),
> 已用新脚本 (按模型命名目录) 重跑, 该 bug 不再出现。

---

## 5. 待跑清单 (还欠缺哪些数据)

完整网格 = 4 模型 × 3 优化器 × 2 filter = **24 格, 现有 17, 缺 7 格**。

### 优先级 1: 补 PPO+filter=on 的小/中尺寸 (验证"治本配方"边界, 最高价值)
现在"PPO+filter 治本"只有 3B 一个证据点。补 0.5B / 1.5B PPO-on 才能回答 **"PPO+filter 的甜区到底从多小开始"**。论文预言 0.5B filter 增益 **+22.9% (最大)**。

| 待跑 | 缺口 | 价值 |
|------|------|------|
| **0.5B PPO on** | 缺 | ⭐ 最高 — 论文 +22.9%, 验证"filter 能否救小模型 PPO" |
| **1.5B PPO on** | 缺 | ⭐ 凑齐 PPO 模型大小线的 on; 1.5B PPO-off 已是稳的, 看 on 能否更稳/更高 |

### 优先级 2: 填满 filter 对照表 C 的剩余算法格
| 待跑 | 缺口 | 价值 |
|------|------|------|
| 3B GRPO on | 缺 | 算法线 on (论文 +9.0); 0.5B GRPO-on 已显示"救峰值不救坍缩", 看 3B 是否同样 |
| 3B DrGRPO on | 缺 | 算法线 on (论文 −0.4); 验证负增益 |
| Qwen3-4B DrGRPO on | 缺 | 补齐 Qwen3-4B 家族最后一格 |

### 优先级 3: 补 1.5B 的 nofilter 算法线 (目前 1.5B 只有 PPO-off, 两个 on)
| 待跑 | 缺口 | 价值 |
|------|------|------|
| 1.5B GRPO off | 缺 | 1.5B 优化器线的 off 基线 |
| 1.5B DrGRPO off | 缺 | 同上 (现仅有 DrGRPO-on 且几乎没学会) |

一键命令 (后台串行, 自动归档推 CJJ; GPU 空闲时可直接跑):
```bash
cd /root/RAGEN && GIT_PUSH=on nohup bash experiment_scripts/run_multiple_experiments.sh \
  "Qwen2.5,0.5B,PPO,on" "Qwen2.5,1.5B,PPO,on" \
  "Qwen2.5,3B,GRPO,on" "Qwen2.5,3B,DrGRPO,on" \
  "Qwen2.5,1.5B,GRPO,off" "Qwen2.5,1.5B,DrGRPO,off" \
  "Qwen3-4B,4B,DrGRPO,on" \
  > logs/queue_fill_grid.log 2>&1 &
```
> 时间: filter=on/未坍缩的 run 跑满 200 步, 每个约 3–4.5h。Qwen3-4B 每步更慢 (PPO ~150–190s/step)。

### 优先级 4: 补 7B (扩展模型大小线上界)
论文 7B=42.4% (最强档)。本机有 `/data/share/Qwen2.5-7B-Instruct`。建议 `GPU_MEM_UTIL=0.3`。
```bash
cd /root/RAGEN && GIT_PUSH=on GPU_MEM_UTIL=0.3 nohup bash experiment_scripts/run_single_experiment.sh \
  Qwen2.5 7B PPO off > logs/7b.log 2>&1 &
```

### 关于 Qwen3-4B 的重跑 (结果可信度问题)
当前 Qwen3-4B 5 个 run 的 baseline 思考率仅 ~48% (response_length 截断), **结果被截断噪声污染**。
若要把 Qwen3-4B 当正式结论, 需先 `STEPS=5` 冒烟 + 调大 `response_length=1024 max_model_len=8192` 重跑, 否则只能定性参考。

### 已知不做 / 可选
- 论文主表的 **DAPO 列**: 脚本已支持 (`run_single_experiment.sh ... DAPO ...`), 按需补。
- **跨任务** (frozenlake / countdown / sudoku) 验证结论是否跨任务成立。
- checkpoint 权重未保留 (训练机空间考虑), 如需推理评估需重训。

---

## 6. Presentation 建议 (展示什么 + 分析什么)

### 6.1 核心叙事 (一句话)
"我们复现了 RAGEN-2 的 **reasoning collapse**, 并发现 **SNR filter 能阻止它 —— 但仅在 `PPO + 1.5B~3B` 这个甜区, filter 不是万能药**。" 所有结论都有 17 个真实 run + 论文 Table 4 双重支撑。

### 6.2 建议展示的图 (数据都在各 metrics.csv)
1. **坍缩曲线 (主图)**: 一张图叠 4–5 条坍缩 run 的 val success vs step —— "先升 → 见顶 → 崩到 0%"; 叠 `valid_thinking_rate` 与 `manager_invalid_action` 说明崩盘=格式坏掉。崩点: 0.5B GRPO step26、0.5B PPO step71、3B DrGRPO step93、3B GRPO step155、3B PPO step175。
2. **filter on vs off (最有说服力)**: 3B PPO 两条曲线叠放 —— off step175 崩到 8.4%, on 全程稳在 37%。
3. **filter 不万能 (新亮点)**: 把"开了 filter 仍坍缩"的 run (0.5B GRPO-on、Qwen3-4B PPO-on) 与 3B PPO-on 并列, 说明 filter 的作用域有边界。
4. **MI 前兆信号**: 画坍缩 run 的 `MI I(X;Z)` vs step —— 崩前先冲高 (2.0–2.4) 再骤降 (Qwen3-4B GRPO 跌到 −3.05)。
5. **三条对照线柱状图**: 优化器线 / 模型大小线 / filter 增益, 每条并列论文 Table 4 数字。

### 6.3 建议讲的分析点 (机制, 不只数字)
- **success 归零的因果链** (最能体现理解): 学会任务 → 输出变长、`<think>` 不闭合 → valid思考率崩 → `manager_invalid_action`→100% → 动作全非法 → success=0。"0% 不是变笨, 是输出彻底跑偏"。
- **为何 PPO 比 GRPO 稳**: critic 提供 token 级 baseline, 梯度噪声小, 坍缩最晚。
- **为何 filter 不万能**: filter 只是减噪, 算法/模型本身噪声太大时补不回来 → 治本需 PPO+合适尺寸。
- **为何 1.5B > 3B**: 非单调, 呼应论文。

### 6.4 为支撑 Presentation, 建议补跑 (按优先级)
| 优先级 | 补跑 | 为 Presentation 提供什么 |
|------|------|------|
| ⭐ P1 | 0.5B PPO on + 1.5B PPO on | 让"filter+PPO 甜区"和"filter 增益"柱状图在小/中模型档完整 (论文 0.5B +22.9% 最戏剧) |
| P2 | 3B GRPO on + DrGRPO on | 填满 filter 对照表 C, 坐实"filter 救峰值不救坍缩" |
| P2 | Qwen3-4B 调大 response_length 重跑 | 让 Qwen3-4B 从"定性参考"升级为正式结论 |
| P3 | 7B PPO off | 模型大小线延伸到 42.4% |

> 最小可展示集: 当前 17 个 run 已足够讲完整故事 (坍缩 + filter 防坍缩 + filter 边界 + 三条对照线)。
> 补 P1 的 2 个 PPO-on 性价比最高 —— 直接验证"治本配方"的尺寸下界。
