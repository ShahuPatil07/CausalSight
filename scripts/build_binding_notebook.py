"""Bundle binding-v1 and its runner for a free Colab GPU."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def build() -> Path:
    cells = []
    def cell(kind: str, source: str):
        row = {"cell_type": kind, "metadata": {}, "id": f"binding-{len(cells):02d}",
               "source": dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            row.update(execution_count=None, outputs=[])
        cells.append(row)
    cell("markdown", """
    # CausalSight EXP-005: object-color binding

    Upload this notebook at https://colab.research.google.com/ and choose a **free GPU**
    under Runtime → Change runtime type. No paid API, subscription or API token is used.
    If no free GPU is available, retry later. This notebook has not yet been GPU-executed.

    The exact versioned dataset, configuration and Python modules are embedded and hashed.
    Development: 16 families / 64 visual requests. Evaluation: 128 balanced families /
    512 visual requests. Each split also makes two actual text-only requests and explicitly
    projects their answers onto the repeated questions; these are not independent trials.

    Each request has one image/question and no conversation history. FP16 Instruct weights
    are distinct from the previous local quantized thinking smoke model.
    Inspect development results before evaluation. An operational failure prevents freezing.
    Original accuracy is not used to exclude families or decide whether to run evaluation.

    Raw rows flush immediately. On an interrupted runtime, download available artifacts
    with the final download cell. The command-line runner supports --resume on unchanged
    artifacts/model/runtime; failed requests are preserved and never selectively retried.
    Colab storage is temporary: download the ZIP before disconnecting.
    """)
    cell("code", """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No free GPU allocated. Retry later; do not purchase compute.")
    print(torch.cuda.get_device_name(0))
    if torch.cuda.get_device_properties(0).total_memory < 6*2**30:
        raise RuntimeError("Less than 6 GiB GPU memory: stop before downloading weights.")
    """)
    cell("code", """
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers==4.57.1", "accelerate>=1.0,<2", "Pillow>=10,<12"], check=True)
    """)
    paths = [ROOT / "configs/experiments/EXP-005-binding-v1.json"]
    paths += sorted((ROOT / "data/binding-v1").rglob("*"))
    paths += [ROOT / "causalsight" / name for name in ("__init__.py", "generate_pilot.py", "generate_binding.py", "evaluate_binding.py", "run_binding.py", "binding_baselines.py")]
    assets = {p.relative_to(ROOT).as_posix(): {"base64": base64.b64encode(p.read_bytes()).decode("ascii"),
                                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in paths if p.is_file()}
    cell("code", dedent("""
    import base64, hashlib, json, uuid, sys
    from pathlib import Path
    WORK = Path("/content/causalsight-binding")
    WORK.mkdir(parents=True, exist_ok=True)
    """) + "\nASSETS = " + repr(assets) + "\n" + dedent("""
    for relative, asset in ASSETS.items():
        data = base64.b64decode(asset["base64"])
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
        target = WORK / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Changed existing bundled file: {target}")
        target.write_bytes(data)
    sys.path.insert(0, str(WORK))
    CONFIG = WORK / "configs/experiments/EXP-005-binding-v1.json"
    RUN_DIR = WORK / "results" / ("EXP-005-colab-" + uuid.uuid4().hex[:8])
    from causalsight.generate_binding import validate
    from causalsight.run_binding import run, TransformersAdapter
    from causalsight.binding_baselines import run as run_baselines
    audit = validate(WORK / "data/binding-v1/examples.jsonl")
    print(audit["families_by_split"], "Dataset audit passed")
    print("Artifacts:", RUN_DIR)
    """))
    cell("code", """
    config = json.loads(CONFIG.read_text())
    adapter = TransformersAdapter(config["model"], config["max_tokens"])
    print("Pinned model revision:", adapter.revision)
    development = run(CONFIG, RUN_DIR / "development", "development", adapter=adapter)
    print(json.dumps(development, indent=2))
    print((RUN_DIR / "development/metrics/report.md").read_text())
    """)
    cell("markdown", """
    Review the development results above. If transport, empty-answer, truncation or parser
    failures occurred, do not change settings and silently continue on evaluation. Record a
    new development/configuration version instead. The next cell requires freeze.json.
    """)
    cell("code", """
    if not (RUN_DIR / "development/freeze.json").exists():
        raise RuntimeError("Development operational gate failed. Download artifacts and investigate.")
    evaluation = run(CONFIG, RUN_DIR / "evaluation", "evaluation", adapter=adapter)
    print(json.dumps(evaluation, indent=2))
    print((RUN_DIR / "evaluation/metrics/report.md").read_text())
    baselines = run_baselines(WORK / "data/binding-v1/examples.jsonl", RUN_DIR / "baselines")
    print((RUN_DIR / "baselines/report.md").read_text())
    """)
    cell("code", """
    import zipfile
    from google.colab import files
    if not RUN_DIR.exists():
        raise RuntimeError("No run artifacts yet")
    archive = WORK / (RUN_DIR.name+"-"+uuid.uuid4().hex[:8]+".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(RUN_DIR.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(RUN_DIR.parent))
        bundle.write(CONFIG, "experiment-config.json")
        bundle.writestr("bundle-hashes.json", json.dumps({p:a["sha256"] for p,a in ASSETS.items()}, indent=2))
    files.download(str(archive))
    """)
    cell("code", """
    import gc
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    print("Use Runtime → Disconnect and delete runtime after downloading results.")
    """)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"colab": {"name": "CausalSight_binding_v1_free_colab.ipynb"}, "accelerator": "GPU",
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    output = ROOT / "notebooks/CausalSight_binding_v1_free_colab.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2)+"\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
