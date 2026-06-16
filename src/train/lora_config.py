"""
LoRA 配置构建 (本地可验证)
--------------------------
作用:
    1. 提供 build_lora_config(): 从 configs/lora.yaml 构建 peft.LoraConfig 对象,
       供训练阶段(05)复用;
    2. __main__ 默认只打印 LoraConfig (秒出, 不下模型);
    3. 加 --with-model 时, 才真正加载 Qwen3 并套上 LoRA, 打印可训练参数占比
       (会下载约 3.4GB 权重, 仅用于直观感受 LoRA 省了多少参数)。

用法:
    python src/train/lora_config.py
    python src/train/lora_config.py --with-model   # 会下载模型权重
"""

import argparse

import yaml
from peft import LoraConfig


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lora_config(lora_cfg_path: str = "configs/lora.yaml") -> LoraConfig:
    """从 yaml 构建 peft.LoraConfig 对象。"""
    cfg = load_yaml(lora_cfg_path)
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        task_type=cfg["task_type"],
        target_modules=cfg["target_modules"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建并查看 LoRA 配置")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="真正加载基座模型并套上 LoRA, 打印可训练参数占比 (会下载权重)",
    )
    args = parser.parse_args()

    lora_config = build_lora_config(args.lora_config)
    print("[LoRA 配置] 构建成功:")
    print(f"  r              = {lora_config.r}")
    print(f"  lora_alpha     = {lora_config.lora_alpha}  (缩放 = alpha/r = {lora_config.lora_alpha / lora_config.r})")
    print(f"  lora_dropout   = {lora_config.lora_dropout}")
    print(f"  bias           = {lora_config.bias}")
    print(f"  task_type      = {lora_config.task_type}")
    print(f"  target_modules = {lora_config.target_modules}")

    if not args.with_model:
        print("\n(未加 --with-model, 跳过模型加载。加该参数可查看可训练参数占比。)")
        return

    # 仅在显式要求时才下载并加载模型
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model

    model_cfg = load_yaml(args.model_config)
    model_id = model_cfg["model_id"]
    print(f"\n[加载基座] {model_id} (下载约 3.4GB, 请耐心等待) ...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    print("[套上 LoRA] get_peft_model ...")
    peft_model = get_peft_model(base_model, lora_config)

    print("\n[可训练参数占比]")
    peft_model.print_trainable_parameters()


if __name__ == "__main__":
    main()
