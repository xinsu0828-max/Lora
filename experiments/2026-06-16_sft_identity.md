# 实验记录 · SFT 监督微调 (identity)

- **日期**: 2026-06-16
- **阶段**: 05 (SFT)
- **环境**: Google Colab, Tesla T4 (14.56 GB), Unsloth 2026.6.7, Transformers 5.5.0, Torch 2.10.0+cu128

## 配置 (configs/train.yaml + lora.yaml)
| 项 | 值 |
|----|----|
| 基座 | Qwen/Qwen3-1.7B (4bit, QLoRA) |
| 数据 | datasets/processed/train.jsonl (27 条) |
| epochs | 10 (total steps = 40) |
| batch / grad_accum | 2 / 4 (等效 batch = 8) |
| learning_rate | 2e-4 (cosine) |
| LoRA | r=16, alpha=32, dropout=0.05, 7 个 target_modules |
| 掩码 | train_on_responses_only (只对 assistant 算 loss) |
| 可训练参数 | 17,432,576 / 1,738,007,552 (1.00%) |

## 结果
| 指标 | 起点 | 终点 |
|------|------|------|
| loss | ~5.67 | ~0.05 |
| train_runtime | - | ~130s |

## 评测 (阶段 06)
微调后对"你是谁/你叫什么/你是谁开发的/你是ChatGPT吗/介绍自己"5 问, 均稳定自称"小鲸, 鲸鱼实验室", 对比基座(通义千问/阿里巴巴)改写成功。

## 产物
- `outputs/lora_adapter` (LoRA adapter, ~77.5 MB)
- 合并模型: `models/xiaojing-merged` (16bit 完整模型, ~3.4 GB)

## 复现
```bash
python src/build_dataset.py
python src/train/train_lora.py
```
