# 实验记录 · 奖励模型 (Reward Model, RLHF 第2步)

- **日期**: 2026-06-16
- **阶段**: 12 (RLHF, 可选实操)
- **环境**: Google Colab, Tesla T4 (14.56 GB), Transformers 5.5.0, Torch 2.10.0+cu128, bitsandbytes 4bit

## 配置 (configs/reward.yaml + lora.yaml)
| 项 | 值 |
|----|----|
| 基座 | Qwen/Qwen3-1.7B → Qwen3ForSequenceClassification (num_labels=1) |
| 偏好数据 | datasets/preference.jsonl (14 条) |
| epochs | 3 (total steps = 12) |
| batch / grad_accum | 1 / 4 |
| learning_rate | 5e-5 |
| LoRA | r=16, alpha=32, dropout=0.05, task_type=SEQ_CLS |
| 量化 | 4bit nf4, compute_dtype fp16 |

## 结果
| 指标 | 起点 | 终点 |
|------|------|------|
| loss | 0.6638 | 0.0057 |
| accuracy | 0.5 | 1.0 |
| margin (chosen-rejected 分差) | 0.13 | ~6.7 |
| train_runtime | - | 42.7s |

## 现象记录
- 加载报告: `lm_head.weight UNEXPECTED` + `score.weight MISSING` → 正常。
  Causal LM 改造为序列分类时丢弃原输出头、新建打分头(score), 需训练。
- `Skipping import of cpp extensions ... torch >= 2.11.0` → 警告, 不影响训练。
- `use_reentrant` / `return_dict` 弃用警告 → 无害。

## 结论
- 奖励模型成功: accuracy 稳定 1.0、margin 持续拉大到 ~6.7、loss 趋近 0。
- 产物: `outputs/reward_model`, 能对"问题+回答"输出标量分数。
- 这是 RLHF 第 2 步; 第 3 步 PPO 因资源限制仅作概念理解(本项目用 DPO 替代)。

## 复现
```bash
python src/train/train_reward.py
```
