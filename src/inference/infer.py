"""
推理脚本 (Colab T4 GPU 运行)
---------------------------
对 adapter 目录或合并后的完整模型目录均可加载。

两种用法:
    1. 单轮问答:
        python src/inference/infer.py --prompt "你是谁？"
    2. 多轮交互聊天 (不带 --prompt, 输入 exit 退出):
        python src/inference/infer.py
    指定模型:
        python src/inference/infer.py --model models/xiaojing-merged
"""

import argparse

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 模型推理")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model", default="outputs/lora_adapter",
                        help="adapter 目录 或 合并模型目录")
    parser.add_argument("--prompt", default=None, help="单轮提问; 不填则进入多轮交互")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    enable_thinking = model_cfg.get("enable_thinking", False)

    import torch
    from unsloth import FastLanguageModel

    print(f"[加载] {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)  # 开启推理优化(含 KV cache)

    def generate(messages: list) -> str:
        """给定完整对话历史, 生成下一句 assistant 回答。"""
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
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # --- 单轮模式 ---
    if args.prompt:
        answer = generate([{"role": "user", "content": args.prompt}])
        print(f"\n用户: {args.prompt}\n小鲸: {answer}")
        return

    # --- 多轮交互模式 (维护对话历史) ---
    print("\n进入多轮对话 (输入 exit / quit 退出)\n" + "-" * 40)
    messages = []
    while True:
        try:
            user_input = input("你 > ").strip()
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        answer = generate(messages)
        messages.append({"role": "assistant", "content": answer})  # 回答存回历史, 保留上下文
        print(f"小鲸 > {answer}")


if __name__ == "__main__":
    main()
