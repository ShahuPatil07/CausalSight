# EXP-005 baseline and model comparison

Verified Colab run: `EXP-005-colab-d4f38207`. 128 evaluation families, 512 visual requests.

| Predictor | Original accuracy | Correct causal update | Nuisance preservation | Correct triplet | Question switch (different colors) |
|---|---:|---:|---:|---:|---:|
| constant_red | 32/128 (25.0%) | 0/128 (0.0%) | 128/128 (100.0%) | 0/128 (0.0%) | 0/96 (0.0%) |
| leftmost_object | 80/128 (62.5%) | 64/128 (50.0%) | 64/128 (50.0%) | 64/128 (50.0%) | 0/96 (0.0%) |
| distractor_object | 32/128 (25.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/96 (0.0%) |
| Qwen3-VL-2B-Instruct_visual | 128/128 (100.0%) | 128/128 (100.0%) | 128/128 (100.0%) | 128/128 (100.0%) | 96/96 (100.0%) |
| Qwen3-VL-2B-Instruct_text_only | 16/128 (12.5%) | 0/128 (0.0%) | 64/128 (50.0%) | 0/128 (0.0%) | 0/96 (0.0%) |

Visual model: all 512 evaluation answers correct, with no missing rows, request errors, parser failures, empty answers or truncation. Development also passed all 64 visual requests.

Text-only: two actual evaluation requests, projected onto repeated questions. Circle response: `Red` (parsed); square response: `The square is blue.` (unparsed under the frozen strict parser). Original accuracy is 16/128 (12.5%); 256/512 projected answers are format failures. These are not 512 independent requests, and the score partly reflects answer format. Do not change the parser after evaluation or interpret 12.5% as pure semantic accuracy.

Constant red's 100% invariance is compatible with wrong answers. Correctness and paired update remain separate.

Interpretation: this model reaches the ceiling on this simple synthetic shape-color grid. This supports success on the tested controls, not broad visual grounding or a novel causal mechanism. A harder task is needed to expose informative failures.

Raw imported artifacts: `results/raw/EXP-005-colab-d4f38207/`. Recomputed results and provenance: `results/processed/EXP-005-colab-d4f38207/`.
