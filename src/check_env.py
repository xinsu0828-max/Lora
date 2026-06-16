"""
环境自检脚本 (本地 Cursor 与 Google Colab 通用)
------------------------------------------------
作用:
    在写训练代码之前, 确认运行环境是否就绪:
      1. 当前运行在本地还是 Colab;
      2. 是否有可用的 NVIDIA GPU 及显存大小;
      3. 关键依赖库是否安装成功及其版本;
      4. (可选) unsloth / bitsandbytes 是否可用 (仅 GPU 环境关心)。

用法:
    本地:  python src/check_env.py
    Colab: !python src/check_env.py   或在 notebook 中 import 调用 main()
"""

from importlib import import_module


def _detect_runtime() -> str:
    """判断当前运行环境: 'colab' 或 'local'。"""
    try:
        import google.colab  # noqa: F401  (只在 Colab 存在)
        return "colab"
    except ImportError:
        return "local"


def _check_package(pip_name: str, import_name: str | None = None) -> str:
    """尝试导入一个包并返回版本字符串, 失败则返回 '未安装'。"""
    module_name = import_name or pip_name
    try:
        module = import_module(module_name)
        return getattr(module, "__version__", "已安装(无版本号)")
    except Exception as exc:  # noqa: BLE001
        return f"未安装 ({type(exc).__name__})"


def _check_gpu() -> None:
    """检查 PyTorch 是否能看到 GPU 及显存。"""
    try:
        import torch
    except ImportError:
        print("  [!] 未安装 torch, 跳过 GPU 检查")
        return

    print(f"  torch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  [OK] 检测到 GPU: {name}")
        print(f"  [OK] 显存总量: {total_gb:.1f} GB")
    else:
        print("  [i] 未检测到可用 GPU (本地开发正常; Colab 请确认已开启 GPU 运行时)")


def main() -> None:
    runtime = _detect_runtime()
    print("=" * 56)
    print(f"运行环境: {'Google Colab' if runtime == 'colab' else '本地 (Local)'}")
    print("=" * 56)

    print("\n[1] GPU 检查")
    _check_gpu()

    print("\n[2] 核心依赖版本")
    core = {
        "transformers": "transformers",
        "datasets": "datasets",
        "peft": "peft",
        "trl": "trl",
        "accelerate": "accelerate",
    }
    for pip_name, import_name in core.items():
        print(f"  {pip_name:<14}: {_check_package(pip_name, import_name)}")

    if runtime == "colab":
        print("\n[3] GPU 训练专用依赖 (仅 Colab 关心)")
        gpu_only = {"unsloth": "unsloth", "bitsandbytes": "bitsandbytes"}
        for pip_name, import_name in gpu_only.items():
            print(f"  {pip_name:<14}: {_check_package(pip_name, import_name)}")

    print("\n检查完成。若核心依赖均显示版本号, 即表示环境就绪。")


if __name__ == "__main__":
    main()
