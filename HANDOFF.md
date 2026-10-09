# CausalSight handoff

## Research direction

Read README.md, PLAN.md and research/LOG.md. The shape/color task is a pipeline pilot, not a validated benchmark. Preserve raw outputs, paired structure and negative findings. The README was left as found during this continuation.

## Current state (2026-10-09)

- Dataset data/pilot-v0: seed 17, three scenes, nine PNGs. Do not regenerate this version.
- Read-only checks confirmed all six interventions change only the intended color field; pixel differences remain within the intended object. All three targets happen to be on the right, so this sample does not counterbalance position.
- Ollama and Hugging Face adapters preserve paired metadata, raw responses, errors and latency. Output files refuse overwrite. --pair-id selects a scene triplet.
- Runner now flushes each completed row and prints its status/latency. In-flight requests are not preserved on interruption. Python compilation passed; no formal test suite was run.
- Evaluator reports separate descriptive pilot metrics. No scientific evaluation is complete.

## EXP-001 outcome

qwen3-vl:2b finished downloading (1.9 GB, ID prefix 0635d9d857d4). One CPU triplet was attempted using the command in research/LOG.md. Available RAM fell from 4.23 GiB to 0.60 GiB (96% usage). The runner was interrupted and the model unloaded. results/raw/EXP-001-qwen3vl-2b-smoke.jsonl is empty and must remain preserved; no metrics exist. No model-behavior results or completed-response latencies exist. The first request's server-side outcome is unknown.

No active inference or pull session remains. ollama ps was empty after stopping. Last RAM sample: 6.87 GiB available of 15.62 GiB. This is not a sustained-headroom measurement.

## Constraints

- CPU-only machine. Leave active VS Code alone; helper processes are not separate windows.
- Edge was previously closed at user request. No unrelated applications were closed during this continuation.
- Previous Ollama models were deleted at the user's request; only the replacement 2B model is installed.
- Free Hugging Face account: verify current access/billing before remote requests; do not assume inference is free.
- Do not scale after memory pressure without establishing safe feasibility.

## Next work

1. Check sustained available RAM and determine whether another local attempt is feasible. Document context/runtime changes explicitly.
2. If retrying, use a new path such as results/raw/EXP-002-qwen3vl-2b-smoke.jsonl. Never overwrite EXP-001. Keep the run to pilot-00000 and monitor memory.
3. Inspect all three raw responses and latency before evaluating a complete triplet. Stop/unload if memory pressure recurs.
4. Before meaningful experiments, review literature, choose a capable feasible runtime, and revisit task, controls, counterbalancing, splits, sample size and uncertainty.

## Latest continuation: EXP-002, EXP-003 and free cloud route

This section supersedes the earlier next-work list and installed model identity above.

- User requires completely free compute: no paid APIs, rentals or subscriptions.
- EXP-002 completed with improved RAM headroom but all three final answers were empty: the 32-token budget was exhausted by reasoning. All failures were preserved and evaluated.
- Runner supports --max-tokens for both providers and records the chosen budget. Offline adapter checks and compilation passed.
- EXP-003 used the same triplet and prompts with --max-tokens 256. Final answers yellow / blue / yellow all match expectations. One paired smoke check passes; no research conclusion follows.
- EXP-003 latencies were approximately five seconds, with 1102 cached prompt tokens each. Do not report these as cold performance.
- Runtime/model observed now: Ollama 0.40.2 and qwen3-vl:2b digest prefix 6e72c4c9d31b. These differ from earlier handoff metadata; this agent did not update or pull them. Full pre-EXP-003 metadata is in results/raw/EXP-003-qwen3vl-2b-smoke-environment.json.
- Results for EXP-002 and EXP-003 are saved in raw and processed directories. Model unloaded afterward. No active exec session remains.
- notebooks/CausalSight_free_colab.ipynb embeds the exact pilot manifest and nine images. It uses public Qwen/Qwen3-VL-2B-Instruct FP16 weights on a free Colab GPU, saves reproducibility metadata/raw outputs/errors, evaluates one triplet and downloads a ZIP. It does not call a hosted inference API.
- Rebuild the notebook with python scripts/build_colab_notebook.py. All code cells compile and all ten embedded files match original bytes. No actual cloud GPU run has been verified.

Next: user uploads the notebook to colab.research.google.com, selects a free GPU runtime and runs cells in order. Download results before disconnecting. Free GPU allocation is variable; retry later if unavailable and never buy compute. Keep Instruct/FP16 cloud results separate from thinking/quantized local results. Review literature, controls, splits and counterbalancing before scaling into a meaningful experiment.
