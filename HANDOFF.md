# CausalSight handoff ? 2026-10-11

Read README.md, PLAN.md, research/LOG.md and research/EXP-005.md. Preserve paired data, raw responses and negative/ceiling findings. All compute must remain free. Leave active VS Code alone. Do not substitute local thinking weights for Instruct FP16 without a separately documented configuration.

## Current outcome: EXP-005 complete

User ran the full notebook on Colab and supplied notebooks/EXP-005-colab-d4f38207-10d95700.zip. Results were imported and independently recomputed. No further inference or parser changes occurred during import.

- Model: Qwen/Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203, FP16/SDPA, greedy decoding, 256-token budget, Tesla T4, Transformers 4.57.1, PyTorch 2.11.0+cu130, Accelerate 1.15.0.
- Development: 64/64 visual answers correct, operational gate passed, real freeze exists in imported artifacts.
- Evaluation: 512/512 visual answers correct; original accuracy, correct causal update, nuisance preservation and complete-triplet correctness each 128/128. Different-color question switches 96/96; same-color switches 32/32. No missing/error/empty/parse/truncation failures.
- Text-only: two actual requests per split, projected onto repeated prompts. Circle answer Red is parsed. Square answer The square is blue. is rejected as prose by the frozen strict parser. Evaluation original accuracy 16/128 (12.5%), zero correct updates/triplets, 256/512 projected parse failures. This is a format-sensitive protocol score, not independent repeated trials or pure semantic accuracy. Keep the parser unchanged.
- Metadata shortcuts: constant-red 25%, leftmost-object 62.5%, distractor-object 25% original accuracy. Fixed-red has 100% nuisance invariance but zero correct triplets/updates.
- Conclusion: this simple synthetic grid reaches a ceiling for the tested model. No broad grounding or internal causal-mechanism claim; increase task difficulty in a new experiment instead of tuning against EXP-005.

## Artifacts and provenance

- Source ZIP preserved under notebooks. SHA-256: 11d7f351e6cf29d18f9ba60122467c81d35c8422dfc20193829ccaaea53999ec.
- Original imported snapshots: results/raw/EXP-005-colab-d4f38207/. Includes development/evaluation raw visual answers, text-only source requests/projections, environment, freeze and original metrics.
- Recomputed metrics and provenance: results/processed/EXP-005-colab-d4f38207/. Import validated 441 bundled source/data hashes, config/manifest/image hashes, frozen model/runtime identities, complete raw rows, text-only source links and metric equality for both splits.
- Combined comparison: results/processed/EXP-005-binding-v1-baselines/report.md and comparison.json. Original heuristic-only report preserved as heuristic-report.md; summary.json remains the original shortcut-only summary.
- Reproducible importer: scripts/import_binding_results.py. It refuses to overwrite imported run directories. Do not rerun against existing output directories or delete immutable outputs to force it.

## Implementation

- data/binding-v1: immutable seed 11017, 16 development families, 128 evaluation families, four inputs per family, 576 rows and 432 PNGs. Evaluation has two renderings per 64 factorial cells; retain 32 same-color originals. Dataset audit passed; no regeneration.
- generate_binding.py generates/audits; evaluate_binding.py preserves full manifest denominators, strict parsing, separate paired metrics, missing/error diagnostics and strata.
- run_binding.py records hashes/revision/runtime, flushes raw rows, gates evaluation on development, and supports failure-preserving --resume with unchanged model/runtime/config. Failed requests are never selectively retried.
- configs/experiments/EXP-005-binding-v1.json is the original protocol. No adjustments were made after evaluation outcomes.
- Nine offline regression tests passed during implementation. Test model responses were synthetic fixtures in temporary directories, separate from actual Colab results.
- notebooks/CausalSight_binding_v1_free_colab.ipynb is the full experiment. scripts/build_binding_notebook.py packages it. Avoid rebuilding an executed notebook if outputs need preservation; the ZIP remains the primary run artifact.

## Earlier preserved history

pilot-v0 (seed 17, three triplets) remains untouched. EXP-001 aborted for RAM pressure; EXP-002 exhausted 32 tokens with empty final answers; EXP-003 completed one triplet at 256 tokens (yellow / blue / yellow), validating only the loop. The original three-image Colab smoke notebook remains distinct from the full EXP-005 notebook.

## Next research work

Choose a harder diagnostic (counting or spatial/relational evidence are candidates), document hypothesis and new controls/templates before inference, and retain free-only compute. EXP-005 evaluation is now observed development evidence; do not call it an unseen confirmatory test or claim population uncertainty from its fixed grid.

## Git

Prior pipeline commit bc036af was pushed to main on 2026-10-09. This handoff accompanies the EXP-005 implementation and validated-result progress commit requested by the user. The user plans to revisit the next experiment tomorrow; no further inference or task expansion is pending in this session.
