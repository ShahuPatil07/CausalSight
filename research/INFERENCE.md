# Inference setup

**Current constraint:** All project compute must be free. Use local Ollama or a free Colab GPU; do not invoke paid APIs, rent GPUs or purchase subscriptions.

## EXP-005 balanced binding experiment

**Completed run:** EXP-005-colab-d4f38207 was imported from the user's ZIP and validated against dataset/source/config/image hashes and the development freeze. All 512 evaluation visual answers were correct. Updated model/text-only/shortcut comparison: `results/processed/EXP-005-binding-v1-baselines/report.md`. Raw source artifacts remain preserved. `scripts/import_binding_results.py` recomputes metrics and refuses to overwrite an existing imported run. Text-only projections share two source requests; one prose response fails the frozen parser.

Use `notebooks/CausalSight_binding_v1_free_colab.ipynb` for the full experiment. The earlier `CausalSight_free_colab.ipynb` remains the three-image pilot smoke notebook. Upload the new notebook to Colab, select a free GPU, run development, inspect its report, then run evaluation if `freeze.json` exists. Download the results ZIP before disconnecting; the download cell also works for partial runs. The model/configuration differs from the old Ollama thinking smoke check.

On a suitable free GPU with notebook dependencies installed:

```powershell
python -m causalsight.run_binding --config configs/experiments/EXP-005-binding-v1.json --output-dir results/raw/EXP-005-binding-v1/development --split development
python -m causalsight.run_binding --config configs/experiments/EXP-005-binding-v1.json --output-dir results/raw/EXP-005-binding-v1/evaluation --split evaluation
```

Add `--resume` only to continue the same artifacts/config/model/runtime. Saved failures remain immutable. The evaluator retains missing expected rows in denominators; partial reports are not complete experiments. Do not manually create a development freeze to bypass failures. Full protocol and baseline results: `research/EXP-005.md`.

## Free cloud GPU notebook

Upload `notebooks/CausalSight_free_colab.ipynb` at [Google Colab](https://colab.research.google.com/), select **Runtime → Change runtime type → GPU** on the free tier, and run the cells in order. The original pilot-v0 images and manifest are embedded; no separate dataset upload or HF token is needed. Download the results ZIP before disconnecting.

The notebook checks for a GPU before downloading public model weights and uses `Qwen/Qwen3-VL-2B-Instruct` in FP16. This differs from the local quantized thinking model and must be logged as a separate configuration. It saves resolved model revision, raw answers/errors/token IDs, decoding settings, hardware/library versions, hashes and paired metrics. Rebuild after evaluator/data changes with `python scripts/build_colab_notebook.py`.

Local validation covered syntax and embedded-file integrity; GPU execution is still pending. [Colab's free GPU allocation and limits vary](https://research.google.com/colaboratory/faq.html). If no free GPU is available, stop and retry later.

The inference runner preserves one output row per input, including raw provider response, paired example identifiers, expected answer, intervention metadata, usage when available, latency, provider, model, and decoding settings. Errors are saved as rows so a failed request is not silently dropped.

No request is sent until the runner is explicitly invoked. Choose the provider and model for each run; never overwrite raw run output.

## Local Ollama

Ollama has `qwen3-vl:2b` installed (ID prefix `0635d9d857d4`, approximately 1.9 GB). The previous models were removed at the user's request. The machine has 15.6 GiB RAM. EXP-001 attempted one CPU triplet and was stopped at 96% memory usage, with no completed rows saved. The model is unloaded; available RAM subsequently recovered to 6.87 GiB. Check sustained headroom before retrying. Use a new output path and complete one triplet before scaling:

```powershell
python -m causalsight.run_inference --dataset data/pilot-v0/examples.jsonl --output results/raw/EXP-002-qwen3vl-2b-smoke.jsonl --provider ollama --model qwen3-vl:2b --pair-id pilot-00000
```

The runner flushes completed rows immediately and prints their status and latency. An interrupted in-flight request may have no saved row. Preserve the empty EXP-001 file as an interrupted attempt; never evaluate it as a model result or overwrite it.

**Latest local outcome:** EXP-002 completed safely but its 32-token budget yielded three empty final answers. EXP-003 used `--max-tokens 256` and produced `yellow`, `blue`, `yellow`, all expected for one triplet. Its approximately five-second requests used cached prompt tokens. The current observed runtime is Ollama 0.40.2 with model digest prefix `6e72c4c9d31b`, which differs from EXP-001's handoff. See the log and saved environment metadata. The model is unloaded. For another run use a new path and explicitly set the generation budget; reasoning tokens can consume it. No broad model claim follows from this smoke check.

## Hugging Face Inference Providers

The OpenAI-compatible routed endpoint can accept text plus a base64 image. Set `HF_TOKEN` in the environment and provide a model that currently has a compatible VLM inference provider:

```powershell
$env:HF_TOKEN = "hf_..."
python -m causalsight.run_inference --dataset data/pilot-v0/examples.jsonl --output results/raw/EXP-001-hf.jsonl --provider huggingface --model "Qwen/Qwen2.5-VL-3B-Instruct:cheapest"
```

Provider/model support and billing are account-dependent. The [Hugging Face pricing page](https://huggingface.co/docs/inference-providers/pricing) currently says free accounts have no included Inference Provider credits; Pro users and Team/Enterprise organizations receive monthly credits. Check current availability and account usage before running a batch. A Hugging Face token is read from the environment and is never written to the output.

## Interpretation caveat

The pilot has one simple synthetic task and fixed wording. It is useful for checking whether pair behavior is measurable; it is not a broad benchmark and is not evidence of real-world visual grounding. Preserve prompts, provider/model revisions where available, and raw responses for every actual run.

## Evaluate a saved run

After inference, summarize a single run without modifying its raw JSONL:

```powershell
python -m causalsight.evaluate --run results/raw/EXP-001-ollama.jsonl --output-dir results/processed/EXP-001-ollama --split development
```

The parser accepts exactly one unambiguous color from the pilot's closed answer set. The report keeps ordinary accuracy, causal answer-update success, nuisance answer-preservation, and joint success conditioned on original correctness separate. Ambiguous responses remain unparsed. Metrics are descriptive pilot measures, not a validated single measure of visual grounding. The result directory must be empty or new.
