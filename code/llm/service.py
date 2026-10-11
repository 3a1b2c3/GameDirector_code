"""Health checking and automatic startup for the bundled Gemma service."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "GAMEDIRECTOR_LLM_MODEL",
        REPOSITORY_ROOT / "external_models" / "gemma-4-E2B-it",
    )
)
DEFAULT_LOG = REPOSITORY_ROOT / "output" / "llm" / "gemma_server.log"
DEFAULT_PID = REPOSITORY_ROOT / "output" / "llm" / "gemma_server.pid"


def _models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def server_health(base_url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(_models_url(base_url), timeout=timeout) as response:
            payload = json.loads(response.read())
        # Support both our compact health payload and standard OpenAI/vLLM
        # /v1/models responses.
        return payload if response.status == 200 and isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve_python(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("GAMEDIRECTOR_GEMMA_PYTHON"),
        str(PROJECT_ROOT / ".venv-gemma" / "bin" / "python"),
        sys.executable,
        shutil.which("python3"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            # Do not resolve a venv's bin/python symlink: invoking the resolved
            # base interpreter discards that venv's site-packages.
            return Path(candidate).expanduser().absolute()
    raise RuntimeError(
        "no Python executable is available for Gemma; create code/.venv-gemma "
        "from code/llm/requirements.txt or pass --llm-server-python"
    )


def ensure_llm_server(
    *,
    base_url: str,
    model: str,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    python: str | None = None,
    device: str = "cuda:0",
    auto_start: bool = True,
    startup_timeout: float = 300.0,
    log_path: str | Path = DEFAULT_LOG,
) -> dict:
    health = server_health(base_url)
    if health is not None:
        return {"started": False, "health": health}
    if not auto_start:
        raise RuntimeError(f"Gemma service is not reachable at {_models_url(base_url)}")

    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(f"refusing to auto-start a server for non-local URL: {base_url}")
    port = parsed.port or 80
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (checkpoint_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete Gemma checkpoint at {checkpoint_path}; missing {missing}; "
            "download it as described in README.md or pass --llm-checkpoint"
        )

    python_path = _resolve_python(python)
    destination = Path(log_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "GEMMA_MODEL_PATH": str(checkpoint_path),
            "GEMMA_MODEL": model,
            "GEMMA_DEVICE": device,
            "GEMMA_PORT": str(port),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with destination.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [str(python_path), str(HERE / "gemma_server.py")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    DEFAULT_PID.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PID.write_text(f"{process.pid}\n", encoding="utf-8")

    deadline = time.monotonic() + float(startup_timeout)
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            # Concurrent rollout queues may race to start the shared server;
            # the process that lost the bind race should reuse the winner.
            health = server_health(base_url)
            if health is not None:
                return {"started": False, "health": health}
            tail = destination.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Gemma service exited with code {return_code}; log tail:\n{tail}"
            )
        health = server_health(base_url)
        if health is not None:
            return {
                "started": True,
                "pid": process.pid,
                "log": str(destination),
                "health": health,
            }
        time.sleep(0.5)

    process.terminate()
    raise TimeoutError(
        f"Gemma did not become ready within {startup_timeout:.0f}s; see {destination}"
    )
