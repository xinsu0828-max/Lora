"""
合并 LoRA adapter 到基座, 导出完整独立模型 (Colab T4 GPU 运行)
-------------------------------------------------------------
作用:
    把训练好的 LoRA 旁路永久焊进基座, 得到一个标准的 Qwen3 模型,
    之后无需 peft 即可加载, 也能被 vLLM / Ollama 等直接部署。

关键: 基座是 4-bit 量化版, 不能直接合并; 用 Unsloth 的
      save_pretrained_merged(save_method="merged_16bit") 还原成 16-bit 再合并。

用法 (Colab 项目根目录):
    python src/merge/merge_model.py
    python src/merge/merge_model.py --adapter outputs/lora_adapter --out models/xiaojing-merged
"""

import argparse

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA 到基座导出完整模型")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default="outputs/lora_adapter")
    parser.add_argument("--out", default="models/xiaojing-merged")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)

    from unsloth import FastLanguageModel

    # 以 16-bit 加载基座 + adapter (load_in_4bit=False, 合并更干净)
    print(f"[加载] adapter: {args.adapter} (16-bit 基座)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=False,
        dtype=None,
    )

    # 合并并保存为标准 16-bit 完整模型
    print(f"[合并] 导出 16-bit 完整模型到: {args.out}")
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")

    print(f"[完成] 完整模型已保存到: {args.out}")
    print("       该目录是标准 Qwen3 模型, 无需 peft 即可加载/部署。")


if __name__ == "__main__":
    main()
