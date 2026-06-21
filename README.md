# RL-Result

RAGEN (StarPO) 实验结果归档仓。每个实验一个目录, 命名
`<模型名>-<任务>-<优化器>-filter{on,off}`, 内含 `train.log` / `metadata.txt` /
`final_metrics.txt` (不含 checkpoint 权重, 权重留在训练机本地)。

实验怎么跑、脚本怎么用, 见主仓 `experiment_scripts/Guide.md` 与 `experiment.md`。
本仓默认工作分支 **CJJ** (个人分支), merge 在 GitHub 上手动做。

---

## 当前进度 (2026-06-21)

### ✅ Qwen2.5-3B-sokoban-PPO-filteroff — 有效真实数据
第一个跑通的实验 (旧脚本 run_serial.sh, 已人为对齐成新归档格式)。

- 训练 4 小时, 跑到 **step 175 / 200** 后触发**提前停止** `reward_variance_collapse`。
- val 平均奖励轨迹: 基线 0.47 → **见顶 ~1.67 (约 step 90-100)** → 坍缩下行至 0.36。
- 这是一条清晰的 **"学习→坍缩 (reasoning collapse)"** 曲线, 正是论文 V2 研究的核心现象。
- checkpoint 存到 `global_step_100` (留在训练机 `checkpoints/ragen/`)。

> **提前停止不是 bug**: RAGEN 内置机制, 当一个 batch 内几乎所有轨迹奖励相同
> (reward variance 坍缩、无学习信号) 时主动停止。论文 V2 的 **SNR-Adaptive Filtering
> (filter=on)** 正是为缓解这种坍缩设计的 —— 因此 filter on/off 对比预期会改变坍缩发生的步数。

### ❌ Qwen2.5-0.5B — 失败 (未归档, 仅记录教训)
0.5B 紧接 3B 启动后**立即崩溃**: `size mismatch (2048 vs 896)` —— 旧脚本 `run_serial.sh`
让 0.5B 复用了 3B 的同名 checkpoint 目录 (`sokoban-PPO-0_1_2_3gpu-nofilter`), 试图把 3B 权重
加载进 0.5B 模型。**0.5B 没产生任何有效训练数据**, 故不归档 (不伪造成结果)。

> **这次失败直接催生了新脚本设计**: 新的 `run_single_experiment.sh` 按
> `<模型>-<任务>-<优化器>-filter{on/off}` 命名 checkpoint 目录, 不同模型不会撞目录,
> 这个 bug 在新体系下不存在。0.5B 待用新脚本重跑。

---

## 待办
- [ ] 用新脚本重跑 0.5B (`run_single_experiment.sh Qwen2.5 0.5B PPO off`)
- [ ] 补 1.5B (模型大小线)
- [ ] 3B filter on/off 对比 (验证 filter 能否推迟/缓解坍缩)
