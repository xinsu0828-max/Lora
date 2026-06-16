"""
Tokenizer 与 chat_template 探查脚本 (本地 CPU 可运行)
----------------------------------------------------
作用: 直观理解一段对话是如何被处理成模型输入的。
流程:
    1. 读取 configs/model.yaml, 加载 Qwen3 的 tokenizer (只下 tokenizer, 不下权重);
    2. 打印 tokenizer 基本信息与特殊 token;
    3. 用 apply_chat_template 把 messages 渲染成训练文本(看到 <|im_start|> 等标记);
    4. 演示 add_generation_prompt 的区别(训练用 vs 推理用);
    5. 真正切成 token id, 并逐个解码展示。

用法:
    python src/tokenizer/inspect_tokenizer.py
"""

import argparse

import yaml
from transformers import AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="探查 tokenizer 与 chat_template")
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model_id"]
    print(f"[加载] tokenizer: {model_id} (首次运行会从 HuggingFace 下载, 仅几 MB)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ---- 1. 基本信息 ----
    print("\n[1] Tokenizer 基本信息")
    print(f"  类型          : {type(tokenizer).__name__}")
    print(f"  词表大小       : {tokenizer.vocab_size}")
    print(f"  eos_token     : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token     : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # ---- 2. 一段示例对话 ----
    messages = [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是小鲸，一个由鲸鱼实验室训练的 AI 助手。"},
    ]

    # ---- 3. 渲染为训练文本 (完整对话, 不加生成提示) ----
    enable_thinking = cfg.get("enable_thinking", False)
    train_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    print("\n[2] apply_chat_template 渲染结果 (训练用, add_generation_prompt=False)")
    print("-" * 56)
    print(train_text)
    print("-" * 56)

    # ---- 4. 推理用提示 (只给到 user, 让模型续写 assistant) ----
    infer_text = tokenizer.apply_chat_template(
        messages[:1],  # 只保留 user 这一轮
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    print("\n[3] 推理用提示 (add_generation_prompt=True, 末尾会留出 assistant 起始标记)")
    print("-" * 56)
    print(infer_text)
    print("-" * 56)

    # ---- 5. 真正切成 token id ----
    # 注意: chat template 渲染出的 train_text 里已经包含特殊标记,
    # 所以这里 add_special_tokens=False, 避免重复添加 BOS 等。
    token_ids = tokenizer(train_text, add_special_tokens=False)["input_ids"]
    print(f"\n[4] 切词结果: 共 {len(token_ids)} 个 token")
    print(f"  前 20 个 token id: {token_ids[:20]}")
    print("\n  逐个解码 (前 20 个):")
    for tid in token_ids[:20]:
        piece = tokenizer.decode([tid])
        print(f"    {tid:>7}  ->  {piece!r}")


if __name__ == "__main__":
    main()
