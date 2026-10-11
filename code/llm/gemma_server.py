#!/usr/bin/env python3
"""Minimal local OpenAI-compatible server for GameDirector's Gemma planner."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "GEMMA_MODEL_PATH", REPOSITORY_ROOT / "external_models" / "gemma-4-E2B-it"
    )
).expanduser().resolve()
MODEL_NAME = os.environ.get("GEMMA_MODEL", "gemma-4-E2B-it")
DEVICE = os.environ.get("GEMMA_DEVICE", "cuda:0")
PORT = int(os.environ.get("GEMMA_PORT", "8018"))
GENERATE_LOCK = threading.Lock()

print(f"[gemma_server] loading {MODEL_PATH} on {DEVICE}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model_type = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True).model_type
if model_type == "gemma4":
    from transformers import Gemma4ForConditionalGeneration as ModelClass
elif model_type == "gemma3n":
    from transformers import Gemma3nForConditionalGeneration as ModelClass
elif model_type == "gemma3":
    from transformers import Gemma3ForConditionalGeneration as ModelClass
else:
    raise ValueError(f"unsupported Gemma model_type: {model_type!r}")

model = ModelClass.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
).eval()
print(f"[gemma_server] loaded {MODEL_NAME} ({model_type})", flush=True)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def generate(messages, temperature: float, max_new_tokens: int) -> str:
    system = "\n".join(
        _text(row.get("content", "")) for row in messages if row.get("role") == "system"
    )
    turns = []
    for row in messages:
        if row.get("role") == "system":
            continue
        content = _text(row.get("content", ""))
        if row.get("role") == "user" and system:
            content = f"{system}\n\n{content}"
            system = ""
        turns.append(
            {"role": row.get("role", "user"), "content": [{"type": "text", "text": content}]}
        )
    inputs = processor.apply_chat_template(
        turns,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    with GENERATE_LOCK, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.rstrip("/") in {"/health", "/v1/models"}:
            self.send_json(
                200, {"status": "ok", "model": MODEL_NAME, "model_path": str(MODEL_PATH)}
            )
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            requested = str(body.get("model", ""))
            if requested.lower().split("/")[-1] != MODEL_NAME.lower().split("/")[-1]:
                raise ValueError(f"requested {requested!r}, but server hosts {MODEL_NAME!r}")
            result = generate(
                body["messages"],
                float(body.get("temperature", 0.0)),
                int(body.get("max_tokens", 256)),
            )
            self.send_json(
                200,
                {
                    "id": "gamedirector-gemma",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": result},
                        }
                    ],
                },
            )
        except Exception as error:
            self.send_json(500, {"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    print(f"[gemma_server] serving http://127.0.0.1:{PORT}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
