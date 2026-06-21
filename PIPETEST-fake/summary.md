# 实验汇总: PIPETEST-fake

- 总步数(有 metrics 的最后一步): **20**

## 1. 任务表现
- val success:    基线 10.00% → 峰值 20.00%(step 20) → 末值 20.00%
- val pass@1:     —
- val reward均值: 基线 0.50 → 峰值 0.50(step 0) → 末值 0.50
- train success:  —

## 2. 坍缩诊断 (RAGEN-2 核心)
- MI I(X;Z) 轨迹:   基线 0.90 → 峰值 1.50(step 20) → 末值 1.50  (越高=推理越能区分输入; 坍缩时下降)
- 条件熵 H(Z|X):    —  (within-input 多样性)
- MI 首轮:          —
- valid思考率:      基线 100.00% → 峰值 100.00%(step 10) → 末值 95.00%  (坍缩时骤降)

## 3. 训练动力学
- actor entropy:  基线 0.40 → 峰值 0.40(step 10) → 末值 0.20
- pg_loss:        —
- actor grad_norm:—
- critic vf_loss: —
- reward 均值:    —

## 4. 耗时
- 平均每步: 81.0s | 共记录 2 步

## 关键指标逐步表 (每 ~10 步抽样)
| step | val_success | MI_traj | valid_think | actor_entropy |
|------|-------------|---------|-------------|---------------|
| 0 | 10.000% | — | — | — |
| 10 | 15.000% | 0.900 | 100.000% | 0.400 |
| 20 | 20.000% | 1.500 | 95.000% | 0.200 |

> 完整逐步数据见 `metrics.csv`；原始日志见 `train.log`。
