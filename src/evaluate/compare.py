"""
微调前后对比评测 (Google Colab T4 GPU 运行)
-------------------------------------------
作用: 用同一批问题分别问"基座原模型"和"微调后模型", 并排打印回答,
      直观验证 LoRA 是否生效(模型是否自称"小鲸")。

为省显存: 先加载基座 -> 回答 -> 释放显存 -> 再加载微调模型 -> 回答。

用法 (Colab 项目根目录):
    python src/evaluate/compare.py
    python src/evaluate/compare.py --adapter outputs/lora_adapter
"""

import argparse
import gc

import yaml


# 用来检验身份的测试问题
QUESTIONS = [
    "你是谁？",
    "你叫什么名字？",
    "你是谁开发的？",
    "你是 ChatGPT 吗？",
    "介绍一下你自己。",
]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_answer(model, tokenizer, question: str, enable_thinking: bool) -> str:
    """对单个问题生成回答。"""
    import torch

    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    # 只解码新生成的部分
    new_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_on_model(model_path: str, model_cfg: dict, label: str) -> dict:
    """加载一个模型, 回答所有问题, 返回 {问题: 回答}, 用完释放显存。"""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n{'='*56}\n[加载] {label}: {model_path}\n{'='*56}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)  # 切换到推理模式(更快)

    enable_thinking = model_cfg.get("enable_thinking", False)
    answers = {}
    for q in QUESTIONS:
        answers[q] = generate_answer(model, tokenizer, q, enable_thinking)
        print(f"  Q: {q}\n  A: {answers[q]}\n")

    # 释放显存, 为加载下一个模型腾空间
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="微调前后对比评测")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default="outputs/lora_adapter")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)

    base_answers = run_on_model(model_cfg["model_id"], model_cfg, "基座原模型 (微调前)")
    tuned_answers = run_on_model(args.adapter, model_cfg, "微调后模型 (小鲸)")

    # 并排汇总
    print(f"\n{'#'*56}\n# 对比汇总\n{'#'*56}")
    for q in QUESTIONS:
        print(f"\n问题: {q}")
        print(f"  [微调前] {base_answers[q]}")
        print(f"  [微调后] {tuned_answers[q]}")


if __name__ == "__main__":
    main()
