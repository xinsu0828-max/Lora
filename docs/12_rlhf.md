# 阶段 12 · RLHF 学习笔记

## 完整 RLHF 三步流水线
1. **SFT**：监督微调，让模型"会说话"。(本项目阶段 05)
2. **Reward Model (RM)**：用偏好数据训练打分模型，输入"问题+回答"→输出标量分数；chosen 高分、rejected 低分。
3. **PPO**：强化学习优化策略模型，最大化 RM 分数，同时用 **KL 惩罚**约束不偏离 SFT 太远(防 reward hacking)。

## RLHF vs DPO
| | RLHF (RM+PPO) | DPO |
|---|---|---|
| 步骤 | 两阶段(训RM + PPO) | 一步 |
| 奖励模型 | 需要 | 不需要(隐式) |
| 强化学习 | PPO | 不用，转监督损失 |
| 稳定性 | 难调、吃资源 | 简单稳定 |
| 数据 | 偏好数据 | 相同 |

**核心**：DPO 在数学上等价于"RM+PPO"的化简，直接用监督损失对齐。免费 GPU 上首选 DPO。

## 关键概念
- **奖励模型**：把人类偏好蒸馏成可自动打分的模型(AutoModelForSequenceClassification, num_labels=1)。
- **PPO**：生成→打分→更新，循环；KL 惩罚是稳定关键。
- **reward hacking**：模型为刷高分而走捷径/胡说，KL 约束用于缓解。

## 2026 趋势
- **GRPO**(DeepSeek)：PPO 高效变体，省价值网络，推理模型常用。
- **在线/迭代 DPO**：边生成边对齐。
- 主线不变：预训练 → SFT → 偏好对齐。

## 本项目为什么用 DPO 而非完整 RLHF
PPO 流水线复杂、不稳定、显存需求高，免费 T4 难以稳定跑通。DPO 用相同偏好数据、一步监督式优化，稳定且资源友好，是学习与中小规模对齐的首选。

## 可选实操
`src/train/train_reward.py` 演示 RLHF 第 2 步(训练奖励模型)。完整 PPO 因资源限制仅作概念理解，trl 提供 `PPOTrainer` / `GRPOTrainer` 可进一步探索。
