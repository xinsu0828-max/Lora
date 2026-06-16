"""
DPO (Direct Preference Optimization) 训练脚本 (Colab T4 GPU 运行)
----------------------------------------------------------------
用偏好数据(prompt/chosen/rejected)做对齐, 让模型倾向更好的回答。

要点:
    - 用 LoRA 做 DPO, 参考模型 = 关闭 LoRA 旁路的同一模型 (ref_model=None), 省显存;
    - 数据是对话格式三元组, trl 会自动套 chat_template。

用法 (Colab 项目根目录):
    python src/train/train_dpo.py
"""

import argparse

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO 微调")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    parser.add_argument("--dpo-config", default="configs/dpo.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    lora_cfg = load_yaml(args.lora_config)
    dpo_cfg = load_yaml(args.dpo_config)

    # ---- 1. 加载起点模型 + 套 LoRA ----
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dpo_cfg["start_from"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=dpo_cfg["seed"],
    )

    # ---- 2. 加载偏好数据 ----
    from datasets import load_dataset

    train_ds = load_dataset("json", data_files=dpo_cfg["pref_file"], split="train")
    print(f"[数据] 偏好样本数: {len(train_ds)}")

    # ---- 3. 配置并训练 DPO ----
    from trl import DPOTrainer, DPOConfig

    dpo_config = DPOConfig(
        output_dir=dpo_cfg["output_dir"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
        optim=dpo_cfg["optim"],
        logging_steps=dpo_cfg["logging_steps"],
        seed=dpo_cfg["seed"],
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # PEFT: 关闭 LoRA 旁路即为参考模型, 无需单独加载
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("[训练] DPO 开始 ...")
    trainer.train()

    out_dir = dpo_cfg["output_dir"]
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[完成] DPO adapter 已保存到: {out_dir}")


if __name__ == "__main__":
    main()
