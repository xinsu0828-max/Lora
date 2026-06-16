"""
查看 LoRA adapter 内容 (纯 Python, 无需 torch, 本地/Colab 均可)
-------------------------------------------------------------
作用:
    1. 列出 adapter 目录下的文件及大小;
    2. 解析 adapter_config.json 的关键字段(基座模型/r/alpha/target_modules);
    3. 统计 adapter 总大小, 并区分"最终 adapter"与"中间 checkpoint"。

用法:
    python src/inspect_adapter.py
    python src/inspect_adapter.py --adapter outputs/lora_adapter
"""

import argparse
import json
from pathlib import Path


def human_size(num_bytes: int) -> str:
    """把字节数转成易读单位。"""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 LoRA adapter 内容")
    parser.add_argument("--adapter", default="outputs/lora_adapter")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        print(f"[错误] 目录不存在: {adapter_dir}")
        return

    # ---- 1. 解析 adapter_config.json ----
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print("[adapter_config.json 关键字段]")
        print(f"  peft_type            = {cfg.get('peft_type')}")
        print(f"  base_model           = {cfg.get('base_model_name_or_path')}")
        print(f"  r                    = {cfg.get('r')}")
        print(f"  lora_alpha           = {cfg.get('lora_alpha')}")
        print(f"  target_modules       = {cfg.get('target_modules')}")
    else:
        print("[警告] 未找到 adapter_config.json (可能不是有效的 adapter 目录)")

    # ---- 2. 列出根目录文件 (区分最终文件与 checkpoint) ----
    print("\n[根目录文件]")
    total_root = 0
    for p in sorted(adapter_dir.iterdir()):
        if p.is_file():
            total_root += p.stat().st_size
            print(f"  {p.name:<32} {human_size(p.stat().st_size)}")

    # ---- 3. 统计 checkpoint 子目录 ----
    checkpoints = [d for d in adapter_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    if checkpoints:
        ckpt_total = sum(f.stat().st_size for d in checkpoints for f in d.rglob("*") if f.is_file())
        print(f"\n[中间 checkpoint] 共 {len(checkpoints)} 个, 合计 {human_size(ckpt_total)}")
        print("  提示: 最终推理只需根目录的 adapter, checkpoint-* 可安全删除以省空间。")

    print(f"\n[最终 adapter 大小(仅根目录文件)] {human_size(total_root)}")


if __name__ == "__main__":
    main()
