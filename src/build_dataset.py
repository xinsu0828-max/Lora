"""
数据集构建脚本 (本地 CPU 可运行)
--------------------------------
流程:
    1. 读取 configs/dataset.yaml 配置;
    2. 用 datasets 库加载原始 JSONL;
    3. 校验每条样本的 messages 格式是否合法;
    4. 按 val_ratio 划分 train / val 并保存到 output_dir;
    5. 打印统计信息与一条样本预览。

用法:
    python src/build_dataset.py
    python src/build_dataset.py --config configs/dataset.yaml
"""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset


def load_config(config_path: str) -> dict:
    """读取 yaml 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_sample(sample: dict, idx: int) -> None:
    """校验单条样本: 必须含 messages, 且 role/content 合法、user 与 assistant 成对出现。"""
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError(f"第 {idx} 条: 缺少有效的 messages 列表")

    valid_roles = {"system", "user", "assistant"}
    has_user = has_assistant = False
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role not in valid_roles:
            raise ValueError(f"第 {idx} 条: 非法 role '{role}'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"第 {idx} 条: role '{role}' 的 content 为空")
        has_user = has_user or role == "user"
        has_assistant = has_assistant or role == "assistant"

    if not (has_user and has_assistant):
        raise ValueError(f"第 {idx} 条: 必须同时包含 user 和 assistant 消息")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建并划分 SFT 数据集")
    parser.add_argument("--config", default="configs/dataset.yaml", help="数据集配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[配置] 数据集: {cfg['name']} | 原始文件: {cfg['raw_path']}")

    # 用 datasets 库加载 JSONL (split='train' 表示把整个文件当作一个 split 读入)
    ds = load_dataset("json", data_files=cfg["raw_path"], split="train")
    print(f"[加载] 共 {len(ds)} 条样本")

    # 逐条校验格式
    for i, sample in enumerate(ds):
        validate_sample(sample, i)
    print("[校验] 全部样本格式合法 ✓")

    # 按比例划分 train / val
    split = ds.train_test_split(test_size=cfg["val_ratio"], seed=cfg["seed"])
    train_ds, val_ds = split["train"], split["test"]
    print(f"[划分] train={len(train_ds)}  val={len(val_ds)}  (val_ratio={cfg['val_ratio']})")

    # 保存为 JSONL, 供后续训练阶段使用
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    train_ds.to_json(str(train_path), force_ascii=False)
    val_ds.to_json(str(val_path), force_ascii=False)
    print(f"[保存] {train_path}\n[保存] {val_path}")

    # 预览一条训练样本
    print("\n[预览] 一条训练样本:")
    for msg in train_ds[0]["messages"]:
        print(f"  {msg['role']:>9} | {msg['content']}")


if __name__ == "__main__":
    main()
