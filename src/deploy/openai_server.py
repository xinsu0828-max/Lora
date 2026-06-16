"""
OpenAI 兼容 API 服务 (Colab T4 GPU 运行)
----------------------------------------
用 FastAPI 实现 OpenAI 风格的 /v1/chat/completions 与 /v1/models 接口,
背后是我们微调的"小鲸"模型。任何 OpenAI 客户端都能直接调用。

启动:
    python src/deploy/openai_server.py --model outputs/lora_adapter --port 8000

测试 (另一个 cell):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key")
    r = client.chat.completions.create(model="xiaojing",
            messages=[{"role":"user","content":"你是谁?"}])
    print(r.choices[0].message.content)
"""

import argparse
import time
import uuid
from typing import List, Optional

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Xiaojing OpenAI-Compatible API")

# 启动时填充的全局对象
STATE = {"model": None, "tokenizer": None, "enable_thinking": False, "model_name": "xiaojing"}


# ---- OpenAI 风格的请求体 ----
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "xiaojing"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


@app.get("/v1/models")
def list_models():
    """返回可用模型列表 (OpenAI 格式)。"""
    return {
        "object": "list",
        "data": [{"id": STATE["model_name"], "object": "model", "owned_by": "whale-lab"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """核心接口: 接收对话, 返回 OpenAI 格式的回答。"""
    import torch

    model, tokenizer = STATE["model"], STATE["tokenizer"]
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=STATE["enable_thinking"],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    prompt_tokens = inputs.shape[1]
    completion_tokens = len(new_tokens)

    # 组装成 OpenAI 标准响应
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or STATE["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def load_model(model_path: str, model_config: str) -> None:
    """启动时加载模型到全局 STATE。"""
    from unsloth import FastLanguageModel

    cfg = yaml.safe_load(open(model_config, "r", encoding="utf-8"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    STATE.update(
        model=model,
        tokenizer=tokenizer,
        enable_thinking=cfg.get("enable_thinking", False),
    )
    print("[服务] 模型加载完成, 接口就绪。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 API 服务")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model", default="outputs/lora_adapter")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model, args.model_config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
