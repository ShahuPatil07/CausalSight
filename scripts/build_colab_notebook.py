"""Package the existing pilot and evaluator into a free-GPU Colab notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def build() -> Path:
    cells = []

    def cell(kind: str, source: str) -> None:
        entry = {"cell_type": kind, "metadata": {}, "source": dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell("markdown", """
    # CausalSight: free Colab GPU smoke check

    Upload this notebook at https://colab.research.google.com/ using **File → Upload notebook**.
    Choose **Runtime → Change runtime type → GPU** on Colab's free tier, then run cells in order.
    No paid API, subscription, API token or rented endpoint is used by this notebook.
    If no free GPU is allocated, stop and try another time; GPU availability and quotas vary.

    The original pilot-v0 manifest and nine PNGs are embedded, without regeneration.
    Only `pilot-00000` (three images) runs by default. Download the results ZIP before
    disconnecting: Colab storage is temporary.

    This uses **Qwen/Qwen3-VL-2B-Instruct in FP16**, a different model variant/runtime
    from the local Ollama thinking run. It is an infrastructure check, not a controlled
    model comparison or research finding. Dataset, task, controls and sample size still
    need research design work. This notebook has been checked locally for syntax and
    embedded-data integrity; GPU execution has not been verified.

    Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
    [official model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
    """)
    cell("code", """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU allocated. Select a free GPU runtime or retry later. Do not buy compute.")
    gpu = torch.cuda.get_device_properties(0)
    print(gpu.name, round(gpu.total_memory / 2**30, 2), "GiB VRAM")
    if gpu.total_memory < 6 * 2**30:
        raise RuntimeError("Less than 6 GiB VRAM: stop this smoke check before downloading weights.")
    """)
    cell("code", """
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers==4.57.1", "accelerate>=1.0,<2", "Pillow>=10,<12"], check=True)
    """)
    files = [ROOT / "data/pilot-v0/examples.jsonl", *sorted((ROOT / "data/pilot-v0/images").glob("*.png"))]
    assets = {
        p.relative_to(ROOT).as_posix(): {
            "base64": base64.b64encode(p.read_bytes()).decode("ascii"),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        } for p in files
    }
    cell("code", dedent("""
    import base64, hashlib, json, uuid, time, platform
    from pathlib import Path
    WORK = Path("/content/causalsight")
    WORK.mkdir(parents=True, exist_ok=True)
    """) + "\nASSETS = " + repr(assets) + "\n" + dedent("""
    for relative, asset in ASSETS.items():
        payload = base64.b64decode(asset["base64"])
        assert hashlib.sha256(payload).hexdigest() == asset["sha256"]
        target = WORK / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f"Different existing dataset file: {target}")
        target.write_bytes(payload)
    DATASET = WORK / "data/pilot-v0/examples.jsonl"
    PAIR_ID = "pilot-00000"
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    RUN_ID = str(uuid.uuid4())
    EXPERIMENT = "EXP-004-colab-qwen3vl-2b-instruct-smoke-" + RUN_ID[:8]
    RUN_DIR = WORK / "results" / EXPERIMENT
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    RAW = RUN_DIR / "raw.jsonl"
    examples = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    examples = [row for row in examples if row["pair_id"] == PAIR_ID]
    assert len(examples) == 3 and {r["variant"] for r in examples} == {"original", "causal", "nuisance"}
    print(EXPERIMENT, "—", len(examples), "images; hashes verified")
    """))
    cell("code", """
    import transformers, accelerate
    from huggingface_hub import HfApi
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    # Resolve once, then use the exact commit for both weights and processor.
    REVISION = HfApi().model_info(MODEL_ID).sha
    config = {"do_sample": False, "max_new_tokens": 256}
    environment = {
        "experiment": EXPERIMENT, "run_id": RUN_ID, "model": MODEL_ID,
        "model_revision": REVISION, "provider": "transformers-colab",
        "dtype": "float16", "attention": "sdpa", "inference_config": config,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "accelerate": accelerate.__version__,
        "cuda": torch.version.cuda, "gpu": gpu.name, "vram_bytes": gpu.total_memory,
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "assets_sha256": {path: asset["sha256"] for path, asset in ASSETS.items()},
        "pair_id": PAIR_ID, "scope": "one-triplet infrastructure smoke check",
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(environment, indent=2))
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    print("Loaded", MODEL_ID, "revision", REVISION)
    """)
    cell("code", """
    with RAW.open("x", encoding="utf-8") as destination:
        for example in examples:
            row = {key: value for key, value in example.items() if key != "scene"}
            row.update(run_id=RUN_ID, experiment=EXPERIMENT, provider="transformers-colab",
                       model=MODEL_ID, model_revision=REVISION, inference_config=config)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            stop_after_error = False
            inputs = generated = new_tokens = None
            try:
                with Image.open(DATASET.parent / example["image"]) as source:
                    image = source.convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example["question"]},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **config)
                new_tokens = generated[0, inputs["input_ids"].shape[1]:]
                raw_text = processor.decode(new_tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                row.update(status="ok", raw_response=raw_text,
                           generated_token_ids=new_tokens.tolist(),
                           usage={"prompt_tokens": inputs["input_ids"].shape[1],
                                  "completion_tokens": len(new_tokens)},
                           budget_reached=len(new_tokens) >= config["max_new_tokens"])
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                # Preserve the failure and stop; never repeatedly retry an OOM.
                stop_after_error = True
            finally:
                torch.cuda.synchronize()
                row["latency_seconds"] = round(time.perf_counter() - started, 4)
                row["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                destination.write(json.dumps(row, sort_keys=True) + "\\n")
                destination.flush()
                print(row["variant"], row["status"], repr(row.get("raw_response")), row["latency_seconds"])
                del inputs, generated, new_tokens
            if stop_after_error:
                break
    """)
    cell("markdown", """
    Evaluate only a complete triplet. Errors and parse failures remain failures.
    The following cell embeds the repository evaluator as it existed when this notebook
    was built. Download raw artifacts even if evaluation cannot run.
    """)
    evaluator = (ROOT / "causalsight/evaluate.py").read_text(encoding="utf-8")
    cell("code", evaluator.split('if __name__ == "__main__":')[0])
    cell("code", """
    saved = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    if len(saved) == 3 and {r["variant"] for r in saved} == {"original", "causal", "nuisance"}:
        print(json.dumps(evaluate(RAW, RUN_DIR / "processed", "development"), indent=2))
    else:
        print("Incomplete triplet; no metrics. Download the preserved raw rows and environment.")
    """)
    cell("code", """
    import zipfile
    from google.colab import files
    archive = WORK / (EXPERIMENT + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(RUN_DIR.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(RUN_DIR.parent))
    files.download(str(archive))
    """)
    cell("code", """
    import gc
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded. Use Runtime → Disconnect and delete runtime when finished.")
    """)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"colab": {"name": "CausalSight_free_colab.ipynb"},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
    }
    for index, entry in enumerate(cells):
        entry["id"] = f"causalsight-{index:02d}"
    output = ROOT / "notebooks/CausalSight_free_colab.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
