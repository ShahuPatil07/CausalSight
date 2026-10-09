# Research log

## EXP-000 — Repository and pilot framing

**Date:** 2026-10-09  
**Question:** What is the research target and what is the smallest responsible first build?  
**Hypothesis:** A controlled synthetic paired-data pilot can validate intervention construction and reveal whether the inference/evaluation loop is feasible; it is not evidence of broad visual grounding.  
**Setup:** Read the repository research charter in `README.md`. The repository initially contained only that charter. Proposed a shape-color scene pilot with causal target-color changes and nuisance distractor-color changes.  
**Result:** No model data or experimental result yet. A data-generation foundation is being added.  
**Interpretation:** Keep scientific claims pending; select the local inference stack only after checking available hardware and model feasibility.  
**Limitations / open questions:** Synthetic images, one simple attribute task, unknown runtime constraints, no literature review yet.  
**Next:** Inspect generated pairs and schema; identify hardware/runtime and choose a feasible model; review related work before fixing evaluation design.

## Inference environment update

**Date:** 2026-10-09  
**Question:** Which inference path is feasible without assuming dedicated GPU access or paid API usage?  
**Observations:** User reports CPU-only hardware and a free Hugging Face account. Local environment has Ollama 0.32.9 with `qwen2.5vl:7b` cached (Q4_K_M, ~6 GB), 15.6 GiB RAM total and ~2.3 GiB available at inspection; `ollama ps` showed no loaded model. HF's current pricing page states free accounts have no included Inference Provider credits; Pro and Team/Enterprise receive credits. Ollama's catalog lists Qwen3-VL 2B at ~1.9 GB, but it is not cached.
**Decision:** Keep both Ollama and Hugging Face adapters available. Do not send remote requests, download weights, or select an HF model assuming zero-cost access. The first likely reproducible route is the already-cached local model after enough RAM is available; verify feasibility with one controlled call before a batch.  
**Limitations:** Current RAM headroom may be too low for the cached 7B model; hosted provider availability and billing depend on account/model/provider.  
**Implementation update:** Added deterministic pilot image/metadata generation, Ollama and Hugging Face-compatible inference adapters, and a separate evaluator for ordinary accuracy, causal answer-update behavior, nuisance preservation, and joint behavior conditional on original correctness. Raw inference output is no-overwrite; request failures stay in the run file. The evaluator retains parse failures as metric failures and writes aggregate plus pair-level summaries.
**Next:** Choose the first actual run after checking available resources and access; do not download weights or use a paid provider without a clear user choice.

## Resource update — model removal

**Date:** 2026-10-09  
**Action:** At the user's explicit request, removed all Ollama models that were installed: `qwen3-coder:30b` and `qwen2.5vl:7b`. `ollama list` is empty; `ollama ps` is empty.  
**Observation:** Available RAM moved only from 2.4 GiB to 2.6 GiB, confirming downloaded model files were not what occupied RAM. Top processes include `Code.exe` (16 processes, ~3.55 GiB RSS) and `msedge.exe` (20 processes, ~1.91 GiB RSS). Closing either may discard unsaved work, so no other processes were stopped.  
**Next:** User identifies which applications are safe to close; after RAM is available, choose and download a VLM before the paired smoke run.

## Resource update — Edge closed

**Date:** 2026-10-09  
**Action:** User authorized closing Edge and unused Code windows. Closed Edge processes; available RAM increased to 4.4 GiB.  
**Observation:** No visible top-level Edge or Code window titles/handles are exposed to this shell, so I could not identify which Code windows were unused. `Code.exe` processes account for ~3.55 GiB RSS across 16 processes. No Code processes were stopped to avoid ending the active workspace or losing unsaved work.  
**Next:** Close unused Code windows manually or identify which Code session is safe to close; after reaching sufficient headroom, choose and download the VLM.

## EXP-001 — Qwen3-VL 2B smoke attempt interrupted

**Date:** 2026-10-09  
**Question:** Can the CPU-only machine complete one triplet without excessive memory pressure?  
**Hypothesis:** The 2B model may validate the inference loop; this is an infrastructure check only.  
**Setup:** Download completed: `qwen3-vl:2b`, local model ID prefix `0635d9d857d4`, weights 1.9 GB. Command: `python -m causalsight.run_inference --dataset data/pilot-v0/examples.jsonl --output results/raw/EXP-001-qwen3vl-2b-smoke.jsonl --provider ollama --model qwen3-vl:2b --pair-id pilot-00000`. Temperature 0, generation budget 32 tokens. Ollama reported context 4096, 100% CPU and loaded size 2.0 GB.  
**Result:** Available RAM sampled at 4.23 GiB before inference, then 1.59, 1.41 and 0.60 GiB during inference; memory usage reached 96%. Interrupted the runner and stopped the model. No completed response rows were saved. The empty raw JSONL is preserved; no evaluation was run. After unloading, `ollama ps` was empty and available RAM was 6.87 GiB. These are samples, not continuous monitoring or a measured peak.  
**Interpretation:** Feasibility was not validated under the initial resource conditions. No accuracy or intervention-behavior evidence exists. The first request's server-side outcome is unknown; an empty file does not establish that the model produced no tokens.  
**Data checks:** All three triplets have the expected variants. Each of six interventions changes exactly one object's color in metadata; labels match the target color, and pixel-difference bounding boxes stay inside the changed object. All three targets are on the right, so this tiny pilot does not counterbalance position. The dataset was not regenerated.  
**Implementation update:** The runner now flushes every completed row immediately and prints its example ID, status and latency. Previously, buffering hid progress and could lose completed rows during process termination. This does not preserve an in-flight request. Python compilation passed.  
**Limitations:** No completed triplet or response/latency results; changing RAM headroom; no controlled runtime comparison. CIM memory inspection was denied; samples used the read-only Windows `GlobalMemoryStatusEx` API.  
**Next:** Keep the model unloaded. Check sustained headroom before retrying and document any context/runtime changes. Use a new run path; do not overwrite EXP-001. Complete one triplet before scaling, inspect raw responses, and evaluate only a complete run. Literature, controls, counterbalancing and a capable feasible runtime remain prerequisites for meaningful research.

## EXP-002 — Complete triplet, truncated final answers

**Date:** 2026-10-09  
**Question:** Does the original smoke configuration complete with improved RAM headroom?  
**Setup:** Same pilot-00000 triplet, Ollama `qwen3-vl:2b`, temperature 0, 32-token budget. Started with 7.04 GiB available RAM; samples while loaded were 4.19, 4.08 and 4.29 GiB. Raw file: `results/raw/EXP-002-qwen3vl-2b-smoke.jsonl`.  
**Result:** Three successful HTTP responses, each with an empty final answer, a separate thinking field and `done_reason: length`; all generated 32 tokens. Latencies: 42.2665, 41.6961 and 41.6942 seconds. Evaluated all three rows; original accuracy, causal-target correctness and nuisance preservation each 0/1, with no original-correct conditional denominator. These are truncation/parse failures, not evidence of inability to recognize colors.  
**Environment observation:** After this run, the local API reported Ollama 0.40.2 and model digest prefix `6e72c4c9d31b`, differing from the earlier handoff's version/model ID. This continuation did not update Ollama or pull/modify the model. The cause of the external change is unknown. Do not treat EXP-001 versus EXP-002 as a controlled comparison.  
**Decision:** Add `--max-tokens` to both adapters and recorded configuration. Preserve the 32-token default and all previous outputs. For the next infrastructure check, explicitly increase the budget to 256; do not parse the thinking field as a final answer. The documented API separates `message.thinking` and `message.content`: https://docs.ollama.com/capabilities/thinking.  
**Validation:** Offline mocked HTTP checks verified both adapters forward a 256-token budget, preserve full provider responses and return final content separately from thinking. Compilation passed.

## EXP-003 — First completed and answerable smoke triplet

**Date:** 2026-10-09  
**Question:** Does a larger generation budget allow the same triplet to produce final answers?  
**Setup:** `python -m causalsight.run_inference --dataset data/pilot-v0/examples.jsonl --output results/raw/EXP-003-qwen3vl-2b-smoke.jsonl --provider ollama --model qwen3-vl:2b --pair-id pilot-00000 --max-tokens 256`. Temperature 0, unchanged questions/images, default model reasoning behavior. Pre-run model/runtime metadata and dataset SHA-256 saved in `results/raw/EXP-003-qwen3vl-2b-smoke-environment.json`: Ollama 0.40.2, model digest `6e72c4c9d31b349bf4068fd1eab8f22b7912cb727f29d7df94be375185d6a5c5`, Q4_K_M.  
**Result:** Final answers original `yellow`, causal `blue`, nuisance `yellow`, all matching expected answers. All stopped normally; completion counts 60, 54 and 61 tokens. Original accuracy, causal-target correctness, nuisance preservation and conditional joint success each 1/1. Processed outputs saved under `results/processed/EXP-003-qwen3vl-2b-smoke`. Latencies 4.9873, 4.7459 and 5.0393 seconds. Each response reports 1102 cached prompt tokens, so these are warm-cache timings, not cold inference performance. Available RAM sampled at 4.00 GiB after the run. Model unloaded afterward; `ollama ps` empty.  
**Interpretation:** One triplet validates that inference, raw preservation, final-answer parsing and paired evaluation can complete. This does not establish general visual evidence reliance. The budget change is an exploratory infrastructure repair, not a confirmed model-performance improvement.  
**Next:** Review literature and design controls/splits/counterbalancing before scaling. Preserve both the failed-budget run and successful smoke run.

## Free-only cloud preparation

**Date:** 2026-10-09  
**User constraint:** All project compute must be completely free. No paid inference requests, GPU rentals or subscriptions.  
**Action:** Built `notebooks/CausalSight_free_colab.ipynb` with reproducible packager `scripts/build_colab_notebook.py`. It embeds the exact pilot-v0 manifest and nine images, checks GPU availability before downloading weights, and defaults to one complete triplet. It uses public `Qwen/Qwen3-VL-2B-Instruct` weights in FP16 with Transformers 4.57.1; it resolves and records the model commit, saves raw text/token IDs/errors, hardware/library versions, dataset/image hashes, latency and peak allocated GPU memory, evaluates a complete triplet and downloads a results ZIP. It stops after an inference error rather than repeatedly retrying.  
**Validation:** Every notebook code cell compiled locally; restored manifest/PNG bytes match all ten original files; selection contains exactly one complete triplet. No Colab GPU execution has been performed.  
**Limitations:** User must upload the notebook and select a free GPU in their Google account. Colab free GPU allocation/quotas are variable, not guaranteed. The Instruct FP16 model differs from the Ollama thinking/quantized configuration; do not merge runs or claim a controlled comparison. Notebook evaluator is a source snapshot; rerun the packager after relevant changes.  
**Sources:** https://research.google.com/colaboratory/faq.html and https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct. HF's current pricing page was opened directly and still reports no included Inference Provider credits for free users (search snippets gave conflicting older text): https://huggingface.co/docs/inference-providers/pricing. Downloading public weights is separate from paid hosted inference.  
**Next:** Run the notebook on an allocated free GPU, download artifacts before runtime deletion, then inspect/import them. If no free GPU is available, retry later or use the now-working local smoke route; do not purchase compute.
