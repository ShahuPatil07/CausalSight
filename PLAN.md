# CausalSight plan

## Research north star

Do vision-language models change their predictions when answer-determining visual evidence changes, while preserving predictions when irrelevant visual information changes?

The README is the research charter. This plan turns it into a small sequence of falsifiable steps; it does not lock the project into one benchmark or metric.

## Current stage: validate the experiment loop

**Latest progress (2026-10-11):** EXP-005 completed on the user's Colab Tesla T4. Imported run EXP-005-colab-d4f38207 matches all 441 bundled file hashes; locally recomputed metrics match captured metrics. Qwen3-VL-2B-Instruct answered all 512 evaluation visual inputs correctly, including 128/128 correct triplets and 96/96 different-color question switches. Text-only original accuracy is 12.5% under the frozen parser: one of two unique responses is prose and rejected. Metadata shortcuts score 25%, 62.5%, and 25% original accuracy. This is a ceiling result on the simple synthetic task. See `research/EXP-005.md`; next design a harder diagnostic with new templates and controls, rather than claiming broad grounding.

**Progress (2026-10-09):** One local triplet completed and evaluated in EXP-003 after increasing the generation budget to 256. EXP-001 was interrupted for memory pressure; EXP-002 produced truncated, empty final answers. These outcomes are preserved. The local inference/evaluation loop now has a successful smoke check, but scientific model/task adequacy remains unestablished. A self-contained free Colab GPU notebook is prepared and locally validated; cloud execution is pending. All compute must be completely free, per the user.

**Pilot candidate (provisional):** controlled shape-color scenes with a question about the color of one named shape. For each scene, generate:

1. an original image;
2. a causal pair member where the queried object's color changes, so the answer should change;
3. a nuisance pair member where a non-queried object's color changes, so the answer should stay fixed.

Keep geometry, question wording, and all other attributes fixed within each pair. Save the scene seed, complete object state, intervention type, changed field, expected answers, image paths, and split. The pilot tests whether generation and pairing are correct; it cannot establish real-world visual grounding.

### Pilot questions and controls

- Can the generator guarantee that only the intended attribute changes?
- Does a local open-weight VLM answer the original task reliably enough for intervention behavior to be interpretable?
- Are causal and nuisance outcomes separable from ordinary accuracy and simple answer priors?
- Do conclusions survive held-out scene seeds and held-out prompt templates?
- Does rendering create systematic cues (object position, palette, order, or answer frequency)?

Counterbalance colors and positions; preserve paired scene identity; split by scene/template before inference. Keep malformed or ambiguous samples visible in raw data and report them rather than silently dropping them.

## Sequence and decision gates

1. **Build data contract and generator** — deterministic paired examples with inspectable PNGs and JSONL metadata. Gate: automated structural checks can establish the intended changed fields and labels. (Checks are part of the implementation; no model claim.)
2. **Select a feasible local model** — inspect hardware and choose one open-weight model plus an inference runtime that is supported by that hardware. Gate: original-image task performance is adequate for analyzing pairs. Do not install or download large model weights until the hardware/runtime choice is clear.
3. **Pilot inference** — retain raw response, parsed answer, prompt, model/revision, decoding settings, seed, latency, pair IDs, and errors. Gate: inspect failures and identify confounds before scaling.
4. **Freeze a modest evaluation design** — choose task coverage, sample count, prompts, splits, baselines, and uncertainty analysis based on pilot behavior and relevant literature. This is a researcher decision, not inferred from a single pilot.
5. **Run a confirmatory experiment** — new versioned config and immutable outputs; analyze paired behavior and failure modes; narrow claims to evidence.

## Initial measurements (descriptive, not a single grounding score)

Report ordinary accuracy on original images; causal answer-update rate; nuisance answer-preservation rate; joint behavior conditioned on original correctness; parse/error rate; and per-example paired outcomes. Keep the full contingency table and raw responses. Do not collapse these into a single “grounding” number without later justification.

## Decisions still open

- Which local model and runtime fit the available GPU/VRAM or CPU budget?
- What minimum original-task accuracy makes the pilot interpretable?
- How many scenes and templates are feasible within the ~40-hour research budget?
- Which additional controls are required after literature review (e.g. text-only priors, random interventions, alternate prompts)?
- Which later task family provides the strongest follow-up evidence: counting, relations, or transformed natural images?

## Scope guardrails

No frontend, deployment, authentication, database, broad model matrix, or “grounding score” at this stage. Add infrastructure only when it answers a research question or makes results reproducible. Preserve raw outputs and version every dataset/config/run.


## Inference reality check (2026-10-09)

The user is CPU-only and has a free Hugging Face account. Earlier pricing checks reported no included Inference Provider credits for free accounts; verify access and billing before hosted inference. The previous Ollama models were removed at the user's request. Subsequently, `qwen3-vl:2b` was downloaded for an infrastructure smoke check. EXP-001 was interrupted when available RAM fell from 4.23 GiB to 0.60 GiB (96% usage); no completed response rows were saved. The model is unloaded. Available RAM recovered to 6.87 GiB, but this sample does not establish sustained headroom. See `research/LOG.md`. Complete one triplet safely before scaling or making model-behavior claims.
