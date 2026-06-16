"""
奖励模型 (Reward Model) 训练 — RLHF 第 2 步 (可选实操, Colab T4 GPU)
-------------------------------------------------------------------
学习目的: 直观理解 RLHF 里"给回答打分"的模型是怎么来的。

要点:
    - 用序列分类模型(num_labels=1)输出标量分数, 而非生成文本;
    - 用偏好数据(chosen/rejected)训练, 让 chosen 得分高于 rejected;
    - 用 4bit + LoRA 省显存; 用 trl 的 RewardTrainer。

说明: 这是 RLHF 的第 2 步。完整 PPO(第 3 步)资源消耗大, 本项目仅作概念理解,
      trl 提供 PPOTrainer / GRPOTrainer 可进一步探索。

用法 (Colab 项目根目录):
    python src/train/train_reward.py
"""

import argparse

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练奖励模型 (RLHF 第2步)")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    lora_cfg = load_yaml(args.lora_config)
    rw_cfg = load_yaml(args.reward_config)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from datasets import load_dataset
    from trl import RewardTrainer, RewardConfig

    model_id = model_cfg["model_id"]

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4bit 加载序列分类模型 (输出 1 个分数) ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, quantization_config=bnb,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- LoRA (注意 task_type = SEQ_CLS) ----
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type="SEQ_CLS",
    )

    # ---- 偏好数据: 拼成 chosen/rejected 全文本 (prompt + 回答) ----
    ds = load_dataset("json", data_files=rw_cfg["pref_file"], split="train")

    def to_text(example: dict) -> dict:
        chosen = tokenizer.apply_chat_template(
            example["prompt"] + example["chosen"], tokenize=False
        )
        rejected = tokenizer.apply_chat_template(
            example["prompt"] + example["rejected"], tokenize=False
        )
        return {"chosen": chosen, "rejected": rejected}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    # ---- 训练 ----
    reward_config = RewardConfig(
        output_dir=rw_cfg["output_dir"],
        num_train_epochs=rw_cfg["num_train_epochs"],
        per_device_train_batch_size=rw_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=rw_cfg["gradient_accumulation_steps"],
        learning_rate=rw_cfg["learning_rate"],
        logging_steps=rw_cfg["logging_steps"],
        seed=rw_cfg["seed"],
        max_length=rw_cfg["max_length"],
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("[训练] 奖励模型开始 ...")
    trainer.train()

    trainer.save_model(rw_cfg["output_dir"])
    print(f"[完成] 奖励模型已保存到: {rw_cfg['output_dir']}")
    print("       它能对'问题+回答'打分; 这就是 RLHF 第 2 步的产物。")


if __name__ == "__main__":
    main()
