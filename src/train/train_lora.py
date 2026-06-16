"""
LoRA / QLoRA 微调主脚本 (Google Colab T4 GPU 运行)
--------------------------------------------------
依赖 Unsloth + trl, 需要 GPU, 本地(CPU)无法运行。

流程:
    1. 读取 configs 下的 model / lora / train 配置;
    2. 用 Unsloth 以 4-bit 加载 Qwen3 基座 (QLoRA);
    3. 套上 LoRA 旁路 (get_peft_model);
    4. 加载阶段02产出的 train/val 数据, 用 chat_template 渲染成文本;
    5. 用 trl SFTTrainer 训练, 并用 train_on_responses_only 做 completion-only 掩码;
    6. 训练完成后只保存 LoRA adapter 到 output_dir。

用法 (在 Colab 项目根目录):
    python src/train/train_lora.py
"""

import argparse

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 微调 Qwen3-1.7B")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    lora_cfg = load_yaml(args.lora_config)
    train_cfg = load_yaml(args.train_config)

    # ---- 1. 用 Unsloth 4-bit 加载基座 (QLoRA) ----
    from unsloth import FastLanguageModel

    max_seq_length = model_cfg["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_id"],
        max_seq_length=max_seq_length,
        load_in_4bit=train_cfg.get("load_in_4bit", True),
        dtype=None,  # None = 自动按 GPU 选 (T4 用 fp16)
    )

    # ---- 2. 套上 LoRA 旁路 ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        use_gradient_checkpointing="unsloth",  # 省显存
        random_state=train_cfg["seed"],
    )

    # ---- 3. 加载数据并渲染为训练文本 ----
    from datasets import load_dataset

    enable_thinking = model_cfg.get("enable_thinking", False)

    def format_to_text(example: dict) -> dict:
        # 把 messages 渲染成完整对话文本 (含特殊标记, 不加生成提示)
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        return {"text": text}

    train_ds = load_dataset("json", data_files=train_cfg["train_file"], split="train")
    train_ds = train_ds.map(format_to_text, remove_columns=train_ds.column_names)
    print(f"[数据] 训练样本数: {len(train_ds)}")
    print(f"[数据] 样本预览:\n{train_ds[0]['text']}")

    # ---- 4. 配置并构建 SFTTrainer ----
    from trl import SFTTrainer, SFTConfig

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        seed=train_cfg["seed"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_config,
    )

    # ---- 5. completion-only 掩码: 只对 assistant 回答算 loss ----
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ---- 6. 开始训练 ----
    print("[训练] 开始 ...")
    trainer.train()

    # ---- 7. 只保存 LoRA adapter ----
    out_dir = train_cfg["output_dir"]
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[完成] LoRA adapter 已保存到: {out_dir}")


if __name__ == "__main__":
    main()
