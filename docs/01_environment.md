# 阶段 01 · 环境搭建 学习笔记

## 目标
搭建一个**本地（Cursor）+ Colab（免费 GPU）双环境**、可复现的 LoRA 微调工程骨架。

## 为什么环境要分离
- 本地 Windows 通常无 NVIDIA GPU / 显存不足，无法真正训练。
- Colab 免费提供 Tesla T4（16GB 显存），适合训练 1.7B 级别模型。
- 设计原则：**代码 / 数据 / 配置分离**，同一份 `src/` 代码本地写、Colab 跑，无需改路径。

## 核心概念速记
| 概念 | 说明 |
|------|------|
| CUDA / VRAM | GPU 计算与显存，训练时存放参数、梯度、优化器状态、激活值 |
| Qwen3-1.7B | 选它因为小、4-bit 量化后仅占 ~2-3GB 显存，免费 T4 够用 |
| LoRA | 冻结原模型，只训练少量低秩适配器参数（<1%），省显存 |
| QLoRA | LoRA + 4-bit 量化，进一步省显存 |

## 各库分工
- `transformers`：加载模型 / tokenizer
- `datasets`：数据集加载与处理
- `peft`：LoRA 实现（`LoraConfig` / `get_peft_model`）
- `trl`：高层训练器（`SFTTrainer` / `DPOTrainer`）
- `unsloth`：训练加速、省显存（仅 Linux GPU）
- `bitsandbytes`：4/8-bit 量化（QLoRA 底层，仅 Linux GPU）

## 文件作用
- `requirements.txt`：本地轻量依赖（CPU 可装）
- `requirements-colab.txt`：Colab 完整 GPU 依赖（含 unsloth/bitsandbytes）
- `src/check_env.py`：通用环境自检脚本
- `notebooks/01_environment.ipynb`：Colab 上的环境检查流程

## 验证清单
- [ ] 本地 `python src/check_env.py` 显示核心库版本号
- [ ] Colab `nvidia-smi` 显示 Tesla T4
- [ ] Colab `check_env.py` 显示 GPU + unsloth/bitsandbytes 版本

## 常见错误
| 现象 | 原因 | 解决 |
|------|------|------|
| 本地装 unsloth 失败 | Windows 不支持 | 正常，本地不装，只在 Colab 装 |
| Colab `nvidia-smi` 报错 | 没开 GPU 运行时 | 更改运行时类型 → T4 GPU |
| `torch.cuda.is_available()` 为 False（本地） | 无 GPU / 装的是 CPU 版 torch | 本地正常，训练去 Colab |
| 库版本冲突 | 旧版本残留 | 新建虚拟环境重装 |
