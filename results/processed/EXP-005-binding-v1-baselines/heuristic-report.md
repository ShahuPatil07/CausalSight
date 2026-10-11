# Shortcut baseline results

Metadata heuristics, not model inference.

| Baseline | Original accuracy | Correct causal update | Nuisance preservation | Correct triplet | Question switch (different colors) |
|---|---:|---:|---:|---:|---:|
| constant_red | 32/128 (25.0%) | 0/128 (0.0%) | 128/128 (100.0%) | 0/128 (0.0%) | 0/96 (0.0%) |
| leftmost_object | 80/128 (62.5%) | 64/128 (50.0%) | 64/128 (50.0%) | 64/128 (50.0%) | 0/96 (0.0%) |
| distractor_object | 32/128 (25.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 0/96 (0.0%) |

Nuisance preservation can be high for a consistently wrong predictor. Inspect correctness jointly.
