# 实验记录 · DPO 偏好对齐 (identity)

- **日期**: 2026-06-16
- **阶段**: 11 (DPO)
- **环境**: Google Colab, Tesla T4 (14.56 GB), Unsloth 2026.6.7, Transformers 5.5.0, Torch 2.10.0+cu128

## 配置 (configs/dpo.yaml)
| 项 | 值 |
|----|----|
| start_from | unsloth/qwen3-1.7b-unsloth-bnb-4bit |
| 偏好数据 | datasets/preference.jsonl (14 条) |
| beta | 0.1 |
| epochs | 3 (total steps = 12) |
| batch / grad_accum | 1 / 4 (等效 batch = 4) |
| learning_rate | 5e-5 (cosine) |
| LoRA | r=16, alpha=32, dropout=0.05, 7 个 target_modules |
| 可训练参数 | 17,432,576 / 1,738,007,552 (1.00%) |

## 结果
| 指标 | 起点 | 终点 |
|------|------|------|
| loss | 0.6931 | 0.0049 |
| rewards/margins | 0 | 5.758 |
| rewards/accuracies | 0 | 1.0 |
| rewards/chosen | 0 | ~3.1 |
| rewards/rejected | 0 | ~-2.6 |
| train_runtime | - | 37.2s |

## 结论
- DPO 收敛良好: margins 持续拉大、accuracies 稳定为 1.0, loss 从理论初值 ln2≈0.693 降至接近 0。
- 产物: `outputs/dpo_adapter` (LoRA adapter)。
- 现象记录: dropout=0.05 触发 Unsloth "performance hit" 提示(非错误); warmup_ratio 弃用警告(改用 warmup_steps 可消除)。

## 复现
```bash
python src/train/train_dpo.py
```
