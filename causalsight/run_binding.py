"""Run the free-compute binding experiment with immutable, resumable artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from .evaluate_binding import evaluate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TransformersAdapter:
    def __init__(self, model_id: str, max_tokens: int, revision: str | None = None):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No free GPU runtime available. Use the free Colab notebook; do not purchase compute.")
        import transformers
        import accelerate
        from huggingface_hub import HfApi
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        self.max_tokens = max_tokens
        self.revision = revision or HfApi().model_info(model_id).sha
        properties = torch.cuda.get_device_properties(0)
        if properties.total_memory < 6*2**30:
            raise RuntimeError("Need at least 6 GiB GPU memory for this FP16 configuration")
        self.metadata = {"model": model_id, "revision": self.revision, "torch": torch.__version__,
                         "transformers": transformers.__version__, "accelerate": accelerate.__version__,
                         "gpu": properties.name, "vram_bytes": properties.total_memory,
                         "cuda": torch.version.cuda, "dtype": "float16", "attention": "sdpa"}
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=self.revision, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa").eval()

    def infer(self, image_path: Path | None, prompt: str) -> dict:
        from PIL import Image
        torch = self.torch
        content = []
        if image_path is not None:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        inputs = generated = tokens = None
        try:
            torch.cuda.reset_peak_memory_stats()
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_tokens)
            torch.cuda.synchronize()
            tokens = generated[0, inputs["input_ids"].shape[1]:]
            return {"text": self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "raw": {"generated_token_ids": tokens.tolist(), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()},
                    "usage": {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(tokens)},
                    "budget_reached": len(tokens) >= self.max_tokens}
        finally:
            del inputs, generated, tokens


class LocalAdapter:
    """Optional Ollama diagnostic: never silently substitute for the Instruct FP16 study."""
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        with urlopen("http://localhost:11434/api/tags") as response:
            tags = json.load(response)
        matches = [m for m in tags["models"] if m["name"] == model]
        if not matches:
            raise ValueError(f"Model not installed: {model}; no automatic download")
        with urlopen("http://localhost:11434/api/version") as response:
            version = json.load(response)
        self.metadata = {"model": matches[0], "runtime": version, "scope": "separate local diagnostic"}

    def infer(self, image_path: Path | None, prompt: str) -> dict:
        import base64
        message = {"role": "user", "content": prompt}
        if image_path is not None:
            message["images"] = [base64.b64encode(image_path.read_bytes()).decode("ascii")]
        payload = {"model": self.model, "messages": [message], "stream": False,
                   "options": {"temperature": 0, "num_predict": self.max_tokens, "num_ctx": 4096}, "keep_alive": "2m"}
        request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            raw = json.load(response)
        return {"text": raw.get("message", {}).get("content", ""), "raw": raw,
                "usage": {"prompt_tokens": raw.get("prompt_eval_count"), "completion_tokens": raw.get("eval_count")},
                "budget_reached": raw.get("done_reason") == "length"}


def invoke(adapter, image: Path | None, prompt: str) -> dict:
    started = time.perf_counter()
    try:
        inference = adapter.infer(image, prompt)
        return {"status": "ok", "raw_response": inference["text"], "provider_response": inference["raw"],
                "usage": inference["usage"], "budget_reached": inference.get("budget_reached", False),
                "latency_seconds": round(time.perf_counter()-started, 4)}
    except (Exception, KeyboardInterrupt) as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}",
                "latency_seconds": round(time.perf_counter()-started, 4)}


def run(config_path: Path, output: Path, split: str, adapter=None, resume: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    manifest = root / config["dataset"]
    if sha256(manifest) != config["dataset_sha256"]:
        raise ValueError("Dataset hash differs from experiment configuration")
    from .generate_binding import validate
    validate(manifest)
    examples = [r for r in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()) if r["split"] == split]
    if len(examples) != config["expected_rows"][split]:
        raise ValueError("Unexpected number of selected examples")
    freeze = output.parent / "development" / "freeze.json"
    if split == "evaluation" and not freeze.exists():
        raise ValueError("Complete development and its operational gate before evaluation")
    frozen = json.loads(freeze.read_text()) if split == "evaluation" else None
    if frozen and frozen["config_sha256"] != sha256(config_path):
        raise ValueError("Configuration changed after development freeze")
    environment_path = output / "environment.json"
    previous = None
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        previous = json.loads(environment_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != sha256(config_path) or previous["split"] != split:
            raise ValueError("Resume configuration or split mismatch")
    elif resume:
        raise ValueError("Cannot resume a missing run")
    if adapter is None:
        if config["provider"] == "transformers":
            revision = (previous or frozen or {}).get("model_revision")
            adapter = TransformersAdapter(config["model"], config["max_tokens"], revision)
        elif config["provider"] == "ollama":
            adapter = LocalAdapter(config["model"], config["max_tokens"])
        else:
            raise ValueError("Only free local execution providers are permitted")
    metadata = adapter.metadata
    if previous and metadata != previous["adapter"]:
        raise ValueError("Runtime/model changed; cannot append to existing run")
    if frozen and metadata != frozen["adapter"]:
        raise ValueError("Runtime/model differs from development freeze")
    run_id = previous["run_id"] if previous else str(uuid.uuid4())
    environment = {"run_id": run_id, "split": split, "config": config, "config_sha256": sha256(config_path),
                   "model_revision": metadata.get("revision"), "adapter": metadata, "python": platform.python_version(),
                   "image_hashes": {r["image"]: sha256(manifest.parent / r["image"]) for r in examples}}
    if previous and environment["image_hashes"] != previous["image_hashes"]:
        raise ValueError("Image bytes changed since original run")
    if not previous:
        output.mkdir(parents=True)
        environment_path.write_text(json.dumps(environment, indent=2)+"\n", encoding="utf-8")
    raw = output / "visual.jsonl"
    completed = {}
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["example_id"] in completed:
                raise ValueError("Duplicate saved row")
            completed[r["example_id"]] = r
        if not set(completed) <= {e["example_id"] for e in examples}:
            raise ValueError("Unexpected saved example")
    stop = False
    with raw.open("a" if previous else "x", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            if example["example_id"] in completed:
                continue  # Even failed requests are immutable; no selective retries.
            # Local diagnostic must stop when memory headroom becomes unsafe.
            if config["provider"] == "ollama" and platform.system() == "Windows":
                import ctypes
                class Memory(ctypes.Structure):
                    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(f"value{i}", ctypes.c_ulonglong) for i in range(7)]
                memory = Memory(); memory.length = ctypes.sizeof(memory)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
                if memory.value1 < 1*2**30:
                    stop = True
                    print("Stopped: less than 1 GiB available RAM", flush=True)
                    break
            result = invoke(adapter, manifest.parent / example["image"], example["question"])
            row = {**example, **result, "run_id": run_id, "provider": config["provider"], "model": config["model"],
                   "inference_config": {"do_sample": False, "max_tokens": config["max_tokens"]}}
            handle.write(json.dumps(row, sort_keys=True)+"\n"); handle.flush()
            completed[example["example_id"]] = row
            print(json.dumps({"example_id": example["example_id"], "status": row["status"], "answer": row.get("raw_response"),
                              "seconds": row["latency_seconds"], "completed": len(completed), "total": len(examples)}), flush=True)
            if row["status"] != "ok":
                stop = True
                break
    evaluation_path = output / "metrics"
    if not evaluation_path.exists():
        metrics = evaluate(manifest, raw, evaluation_path, split)
    else:
        # A resumed run gets a new analysis directory; never overwrite a partial report.
        evaluation_path = output / ("metrics-"+uuid.uuid4().hex[:8])
        metrics = evaluate(manifest, raw, evaluation_path, split)
    if stop or len(completed) != len(examples):
        return {"state": "incomplete", "saved": len(completed), "expected": len(examples)}
    # Two actual text-only requests; projected rows are explicitly linked to their source.
    text_path = output / "text-only-requests.jsonl"
    text_requests = {}
    if text_path.exists():
        text_requests = {r["question"]: r for r in (json.loads(line) for line in text_path.read_text().splitlines())}
    with text_path.open("a" if text_path.exists() else "x", encoding="utf-8", newline="\n") as handle:
        for prompt in sorted({e["question"] for e in examples}):
            if prompt not in text_requests:
                result = {**invoke(adapter, None, prompt), "question": prompt, "source_request_id": str(uuid.uuid4())}
                handle.write(json.dumps(result)+"\n"); handle.flush()
                text_requests[prompt] = result
                if result["status"] != "ok":
                    return {"state": "text-only-error", "saved": len(completed)}
    projected = output / "text-only-projected.jsonl"
    if not projected.exists():
        with projected.open("x", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                row = {**example, **text_requests[example["question"]], "run_id": run_id+"-text-only",
                       "provider": config["provider"], "model": config["model"], "expanded_text_only": True}
                handle.write(json.dumps(row, sort_keys=True)+"\n")
    if not (output / "text-only-metrics").exists():
        evaluate(manifest, projected, output / "text-only-metrics", split)
    # Freeze an operationally valid development run, regardless of its accuracy.
    requests = metrics["requests"]
    gate = all(requests[key] == 0 for key in ("error", "missing", "empty_final_answer", "budget_reached", "parse_failure"))
    if split == "development" and gate and not (output / "freeze.json").exists():
        (output / "freeze.json").write_text(json.dumps({"config_sha256": sha256(config_path), "adapter": metadata,
                                                       "model_revision": metadata.get("revision"),
                                                       "rule": "No request errors, missing rows, empty answers, truncation or parse failures; no accuracy threshold."}, indent=2)+"\n")
    return {"state": "complete" if split == "evaluation" or gate else "development-gate-failed", "saved": len(completed), "metrics": str(evaluation_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, args.split, resume=args.resume), indent=2))
