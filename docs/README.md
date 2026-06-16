# LoRA 微调学习项目 · 总览

从 0 到 1 系统学习 LoRA 微调，基座模型 **Qwen3-1.7B**，本地(Cursor)开发 + Google Colab(免费 T4)训练。

## 学习路线（12 阶段）
| 阶段 | 主题 | 核心产物 |
|------|------|----------|
| 01 | 环境搭建 | requirements / check_env.py |
| 02 | 数据集制作 | datasets/identity.jsonl, build_dataset.py |
| 03 | Tokenizer 与 chat_template | inspect_tokenizer.py |
| 04 | LoraConfig 与 PEFT | configs/lora.yaml, lora_config.py |
| 05 | LoRA 微调(SFT) | train_lora.py → outputs/lora_adapter |
| 06 | 模型测试 | evaluate/compare.py |
| 07 | 保存 adapter | inspect_adapter.py |
| 08 | merge_and_unload | merge/merge_model.py → models/xiaojing-merged |
| 09 | 推理 | inference/infer.py |
| 10 | OpenAI API 服务化 | deploy/openai_server.py |
| 11 | DPO 微调 | preference.jsonl, train_dpo.py → outputs/dpo_adapter |
| 12 | RLHF 学习 | docs/12_rlhf.md, train_reward.py(可选) |

## 工作流
- **代码**：本地 Cursor 编写 → `git push` → GitHub → Colab `git pull`。
- **权重**：训练产物(adapter/合并模型)不进 Git，持久化到 Google Drive 或 HuggingFace Hub。
- **配置/数据/代码三分离**：`configs/` `datasets/` `src/`。

## 目录结构
```
LoRA/
├── docs/          学习笔记与总览
├── experiments/   实验记录(配置+结果+复现)
├── datasets/      原始与处理后数据 + 偏好数据
├── notebooks/     各阶段 Colab notebook
├── configs/       model/lora/dataset/train/dpo/reward 配置
├── src/
│   ├── check_env.py / build_dataset.py / inspect_adapter.py
│   ├── tokenizer/   inspect_tokenizer.py
│   ├── train/       lora_config.py / train_lora.py / train_dpo.py / train_reward.py
│   ├── evaluate/    compare.py
│   ├── merge/       merge_model.py
│   ├── inference/   infer.py
│   └── deploy/      openai_server.py
├── outputs/       adapter 等训练产物(gitignore)
├── models/        合并后完整模型(gitignore)
├── requirements.txt / requirements-colab.txt
```

## 关键认知
- LoRA：冻结基座，只训练低秩旁路(~1% 参数)，省显存。
- QLoRA：LoRA + 4bit 量化，进一步省显存。
- chat_template：对话→ChatML 固定格式，训练只对 assistant 算 loss。
- SFT 学"怎么说"，DPO 学"哪种更好"，RLHF(RM+PPO) 是 DPO 的完整原型。

## 复现顺序(Colab)
```bash
git clone <repo> && cd Lora
pip install unsloth && pip install --no-deps --force-reinstall unsloth unsloth_zoo  # 重启会话
python src/build_dataset.py
python src/train/train_lora.py      # SFT
python src/evaluate/compare.py      # 评测
python src/train/train_dpo.py       # DPO
```
