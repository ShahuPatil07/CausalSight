"""Run paired CausalSight examples through a local or routed VLM."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Adapter(Protocol):
    def infer(self, image_path: Path, prompt: str) -> dict[str, Any]: ...


class OllamaAdapter:
    def __init__(self, model: str, endpoint: str = "http://localhost:11434/api/chat", max_tokens: int = 32) -> None:
        self.model = model
        self.endpoint = endpoint
        self.max_tokens = max_tokens

    def infer(self, image_path: Path, prompt: str) -> dict[str, Any]:
        image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }
        request = Request(self.endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            raw = json.loads(response.read())
        return {
            "text": raw.get("message", {}).get("content", ""),
            "raw": raw,
            "usage": {"prompt_tokens": raw.get("prompt_eval_count"), "completion_tokens": raw.get("eval_count")},
        }


class HuggingFaceAdapter:
    """OpenAI-compatible Hugging Face Inference Providers chat route."""

    def __init__(self, model: str, token: str, endpoint: str = "https://router.huggingface.co/v1/chat/completions", max_tokens: int = 32) -> None:
        self.model = model
        self.token = token
        self.max_tokens = max_tokens
        self.endpoint = endpoint

    def infer(self, image_path: Path, prompt: str) -> dict[str, Any]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = Request(self.endpoint, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
        })
        with urlopen(request, timeout=300) as response:
            raw = json.loads(response.read())
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        return {"text": message.get("content", ""), "raw": raw, "usage": raw.get("usage")}


def make_adapter(provider: str, model: str, endpoint: str | None = None, max_tokens: int = 32) -> Adapter:
    if provider == "ollama":
        return OllamaAdapter(model, endpoint or "http://localhost:11434/api/chat", max_tokens)
    if provider == "huggingface":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN in the environment before using Hugging Face inference.")
        return HuggingFaceAdapter(model, token, endpoint or "https://router.huggingface.co/v1/chat/completions", max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


def run(dataset: Path, output: Path, provider: str, model: str, endpoint: str | None = None, pair_id: str | None = None, max_tokens: int = 32) -> int:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    adapter = make_adapter(provider, model, endpoint, max_tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    completed = 0
    with dataset.open(encoding="utf-8") as source, output.open("x", encoding="utf-8", newline="\n") as destination:
        for line in source:
            if not line.strip():
                continue
            example = json.loads(line)
            if pair_id is not None and str(example.get("pair_id")) != pair_id:
                continue
            image_path = dataset.parent / example["image"]
            prompt = example["question"]
            started = time.perf_counter()
            result: dict[str, Any]
            try:
                inference = adapter.infer(image_path, prompt)
                result = {"status": "ok", "raw_response": inference["text"], "provider_response": inference["raw"], "usage": inference["usage"]}
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
                result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                if isinstance(error, HTTPError):
                    try:
                        result["provider_error_body"] = error.read().decode("utf-8", errors="replace")
                    except OSError:
                        pass
            result.update({
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "example_id": example.get("example_id"),
                "pair_id": example.get("pair_id"),
                "scene_id": example.get("scene_id"),
                "split": example.get("split"),
                "original_answer": example.get("original_answer"),
                "seed": example.get("seed"),
                "generator_version": example.get("generator_version"),
                "variant": example.get("variant"),
                "image": example.get("image"),
                "question": prompt,
                "expected_answer": example.get("expected_answer"),
                "intervention": example.get("intervention"),
                "latency_seconds": round(time.perf_counter() - started, 4),
                "inference_config": {"temperature": 0, "max_tokens": max_tokens},
            })
            destination.write(json.dumps(result, sort_keys=True) + "\n")
            destination.flush()
            completed += 1
            print(json.dumps({"example_id": result["example_id"], "status": result["status"], "latency_seconds": result["latency_seconds"]}), flush=True)
    print(json.dumps({"run_id": run_id, "rows": completed, "output": str(output)}, indent=2))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to examples.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output path; existing files are never overwritten")
    parser.add_argument("--provider", choices=("ollama", "huggingface"), required=True)
    parser.add_argument("--model", required=True, help="Provider model tag / identifier")
    parser.add_argument("--endpoint", help="Optional provider-compatible endpoint override")
    parser.add_argument("--pair-id", help="Run only the complete intervention pair with this ID")
    parser.add_argument("--max-tokens", type=int, default=32, help="Generation budget, including reasoning tokens where applicable (default: 32)")
    args = parser.parse_args()
    run(args.dataset, args.output, args.provider, args.model, args.endpoint, args.pair_id, args.max_tokens)


if __name__ == "__main__":
    main()

