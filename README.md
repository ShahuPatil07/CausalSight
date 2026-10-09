# CausalSight

> Investigating whether vision-language models actually use the visual evidence that should determine their answers.

## 0. Purpose of This Document

This document is the persistent research charter for CausalSight.

It should be read by any coding or research agent working on this repository before making substantial architectural, experimental, or methodological decisions.

This is intentionally **not a rigid implementation specification**.

CausalSight is a research project. The exact hypotheses, datasets, interventions, models, metrics, and methods should evolve as evidence is collected.

Agents should therefore:

- preserve room for research exploration;
- question assumptions in this document when evidence warrants it;
- avoid prematurely optimizing around one benchmark, model, or technique;
- keep experimental infrastructure modular;
- distinguish exploratory experiments from confirmed findings;
- record negative results and unexpected observations;
- prioritize scientific validity over feature count or demo quality;
- never manufacture a positive result simply to fit the initial hypothesis.

The goal is not to prove that the original idea was correct.

The goal is to discover something defensible about how multimodal models use visual evidence.

---

# 1. Project Motivation

Vision-language models can achieve impressive performance on visual question answering, reasoning, OCR, counting, spatial reasoning, and multimodal benchmarks.

However, final-answer accuracy does not necessarily tell us **why** a model produced an answer.

A model may answer correctly because it:

- identified the relevant visual evidence;
- exploited correlations or dataset priors;
- relied disproportionately on language priors;
- attended to an irrelevant but correlated visual feature;
- guessed successfully;
- produced an answer that remains unchanged even when answer-determining evidence changes.

This creates an important distinction:

> A model being correct is not necessarily equivalent to the model being visually grounded in the evidence that should determine the answer.

CausalSight investigates this distinction experimentally.

The broad research theme is:

**Visual evidence reliance in multimodal models under controlled interventions.**

---

# 2. Core Research Question

The initial research question is:

> **Do vision-language models change their predictions appropriately when answer-determining visual evidence changes, while remaining invariant when irrelevant visual information changes?**

This should be treated as the starting question rather than the permanent boundary of the project.

Potential follow-up questions may emerge naturally:

- Are some reasoning capabilities more causally grounded than others?
- Do larger models exhibit stronger visual reliance?
- Does explicit reasoning improve or worsen evidence sensitivity?
- Can models recognize when visual evidence is insufficient?
- Does confidence correlate with genuine visual reliance?
- Can evidence sensitivity predict model errors?
- Are certain visual features systematically ignored?
- Do models remain correct for the wrong reasons?
- Are incorrect predictions sometimes surprisingly well grounded?
- Does repeated inference reveal unstable evidence use?
- Can a model explain which visual evidence caused its prediction?
- Do model explanations correspond to behavioral evidence?
- Can interventions reveal language-prior dependence?
- Can evidence-based diagnostics improve selective prediction or model routing?

These are possible directions, not mandatory experiments.

---

# 3. Research Philosophy

CausalSight should be developed as a **research project first and software project second**.

A polished repository is useful.

A sophisticated frontend is optional.

A scientifically interesting result is essential.

The project should optimize for:

1. clear hypotheses;
2. controlled experiments;
3. reproducibility;
4. strong baselines;
5. interpretable results;
6. useful negative findings;
7. careful claims;
8. high-quality failure analysis.

Avoid optimizing primarily for:

- number of models supported;
- number of benchmark questions;
- frontend complexity;
- repository size;
- agent-generated code volume;
- flashy demos;
- arbitrary benchmark scores.

A 500-example experiment that reveals a convincing phenomenon can be more valuable than evaluating 100,000 examples without understanding what happened.

---

# 4. Central Experimental Idea

The core methodology is based on **controlled visual interventions**.

Suppose an image contains visual evidence \(E\) and the correct answer is \(Y\).

We want to create related images where specific properties are deliberately modified.

Two broad intervention families are particularly important.

## 4.1 Causal / Answer-Changing Intervention

Modify visual evidence that should determine the answer.

Example:

Original:

- red circle;
- blue triangle;
- question: "What color is the circle?"
- answer: red.

Intervention:

- circle becomes green.

Expected behavior:

- answer should change from red to green.

A model that continues answering "red" may not be responding correctly to the modified evidence.

---

## 4.2 Nuisance / Answer-Preserving Intervention

Modify something that should not affect the answer.

Example:

Original:

- red circle;
- blue triangle;
- question: "What color is the circle?"
- answer: red.

Intervention:

- triangle becomes yellow.

Expected behavior:

- answer remains red.

A model whose answer changes may be sensitive to irrelevant evidence.

---

## 4.3 Beyond Binary Intervention Types

Do not assume that all useful interventions will fit neatly into these two categories.

Possible extensions include:

- evidence removal;
- evidence addition;
- distractor insertion;
- partial occlusion;
- spatial relocation;
- object substitution;
- attribute swapping;
- counterfactual scene construction;
- ambiguity introduction;
- evidence-strength manipulation;
- conflicting evidence;
- compositional interventions;
- multiple simultaneous interventions.

The intervention framework should remain extensible.

---

# 5. What CausalSight Should Ultimately Measure

Ordinary accuracy remains useful but is insufficient.

The project should investigate behavioral properties such as:

### Correctness

Does the model produce the correct answer?

### Causal sensitivity

When answer-determining evidence changes, does the model appropriately update its prediction?

### Nuisance invariance

When irrelevant information changes, does the model preserve its answer?

### Consistency

Does the model behave similarly across semantically equivalent inputs?

### Error recoverability

When the model initially fails, can additional evidence or controlled transformations cause it to recover?

### Confidence alignment

Does model confidence correspond to correctness or evidence reliance?

### Evidence dependence

How strongly does prediction behavior depend on the evidence expected to determine the answer?

### Selective reliability

Can behavioral signals derived from interventions identify predictions that should not be trusted?

Not every metric must survive into the final project.

Metrics should be retained only when they correspond to meaningful scientific questions.

---

# 6. Dataset Strategy

The initial benchmark should favor **controllability over realism**.

Procedurally generated scenes are attractive because the project can know exactly:

- which objects exist;
- their attributes;
- their positions;
- their relationships;
- which evidence determines each answer;
- which interventions should change the answer;
- which interventions should preserve it.

This makes causal experimentation significantly cleaner.

However, synthetic data should not automatically become the entire project.

The research may eventually include:

- synthetic scenes;
- transformed natural images;
- existing public datasets;
- manually curated examples;
- generated images;
- hybrid synthetic-real experiments.

The benchmark strategy should evolve according to findings.

---

# 7. Potential Capability Families

The benchmark should eventually cover multiple visual reasoning behaviors rather than one narrow task.

Candidate families include:

### Attribute recognition

- color;
- shape;
- size;
- texture;
- orientation.

### Counting

- number of objects;
- number satisfying an attribute;
- comparative counting.

### Spatial relationships

- left/right;
- above/below;
- inside/outside;
- nearest/farthest;
- relative ordering.

### Compositional reasoning

Questions requiring multiple attributes or relationships.

### Distractor robustness

Relevant evidence surrounded by irrelevant objects.

### Occlusion

Partial visibility of relevant evidence.

### Relational reasoning

Questions whose answers depend on relationships rather than isolated objects.

### Evidence sufficiency

Questions where the image intentionally does not contain enough evidence.

These categories are candidates.

Do not force equal coverage if early experiments show that particular categories contain substantially more interesting behavior.

---

# 8. Models

The architecture should support evaluating multiple multimodal models without coupling the benchmark to one model family.

Initial development should prioritize models that are:

- open-weight;
- reproducible;
- accessible under available compute;
- reasonably capable;
- easy to run through a stable inference stack.

Start small.

Scale model diversity only after the experiment pipeline is trustworthy.

A useful comparison might eventually include differences in:

- model size;
- architecture;
- instruction tuning;
- reasoning behavior;
- quantization;
- prompting strategy.

Do not add models merely to increase the size of a results table.

Every additional model should answer a research question.

---

# 9. Baselines

Strong baselines are mandatory.

Possible baselines include:

- direct prediction;
- text-only prediction where meaningful;
- image-only variants where meaningful;
- reasoning-before-answer prompting;
- repeated inference / self-consistency;
- self-reported confidence;
- perturbation consistency;
- random intervention controls;
- simple heuristic predictors.

Additional baselines should be identified during literature review.

For every proposed method, ask:

> What is the simplest explanation for why this appears to work?

Then design a baseline that tests that explanation.

---

# 10. Evaluation Principles

## 10.1 Separate development from final evaluation

Do not repeatedly tune ideas against the final test set.

Where possible maintain:

- development scenes;
- development templates;
- held-out scene configurations;
- held-out generation templates.

Template-level splitting may be more meaningful than random image splitting.

---

## 10.2 Preserve paired structure

Intervention experiments are inherently paired.

Keep explicit links between:

- original image;
- intervention image;
- intervention type;
- changed variables;
- expected answer before intervention;
- expected answer after intervention.

Never lose this structure during preprocessing.

---

## 10.3 Save raw model behavior

Do not save only aggregate metrics.

For every inference, preserve where practical:

- model identifier;
- model/version information;
- prompt;
- image identifier;
- intervention identifier;
- raw model response;
- parsed answer;
- expected answer;
- inference configuration;
- random seed where applicable;
- latency;
- token counts where available;
- experimental run identifier.

Interesting research often begins with examining individual failures.

---

## 10.4 Quantify uncertainty

When comparing methods, use appropriate statistical uncertainty.

Possible techniques include:

- paired bootstrap confidence intervals;
- permutation tests;
- effect sizes;
- confidence intervals over accuracy differences.

Do not treat a tiny score difference as meaningful simply because one number is larger.

---

# 11. Failure Analysis

Failure analysis is a first-class component of CausalSight.

Potential failure categories may include:

- answer remains unchanged after causal intervention;
- answer changes after nuisance intervention;
- correct answer with suspicious intervention behavior;
- incorrect answer with otherwise consistent grounding;
- contradictory reasoning and final answer;
- excessive reliance on language priors;
- counting instability;
- spatial reasoning instability;
- sensitivity to irrelevant object attributes;
- prompt sensitivity;
- parsing failures;
- ambiguous generated scenes.

The taxonomy should evolve based on observed behavior.

Do not force failures into categories designed before seeing the data.

---

# 12. Research Log

Maintain a persistent research log, for example:

`research/LOG.md`

Every meaningful experiment should record:

- date;
- question;
- hypothesis;
- experiment configuration;
- result;
- interpretation;
- surprising observations;
- limitations;
- next experiment.

Example:

```text
Experiment: EXP-014

Question:
Does nuisance sensitivity increase with scene clutter?

Hypothesis:
Models will become less invariant as irrelevant object count increases.

Setup:
...

Result:
...

Interpretation:
...

Unexpected observation:
...

Next:
...
```

This log is extremely important.

Coding agents should consult previous experiments before proposing new ones.

---

# 13. Experiment Registry

Every serious experiment should receive an identifier.

Example:

```text
EXP-001
EXP-002
EXP-003
...
```

Experiments should ideally have configuration files specifying:

- dataset version;
- model;
- prompts;
- generation parameters;
- evaluation parameters;
- seed;
- intervention set.

An experiment should be reproducible without reconstructing parameters from chat history.

---

# 14. Reproducibility

The repository should aim toward commands conceptually similar to:

```bash
python generate_dataset.py --config configs/dataset/base.yaml

python run_inference.py \
    --model <model> \
    --dataset <dataset>

python evaluate.py \
    --run <run_id>

python generate_report.py \
    --run <run_id>
```

Exact interfaces may change.

The important principle is:

> Research results should be reproducible from configuration rather than hidden notebook state.

Notebooks are encouraged for exploration and analysis.

They should not become the only way to reproduce final results.

---

# 15. Suggested Repository Structure

This structure is illustrative rather than mandatory.

```text
causalsight/
│
├── README.md
├── PLAN.md
├── LICENSE
│
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── causalsight/
│   ├── data/
│   ├── interventions/
│   ├── models/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── generate_dataset.py
│   ├── run_inference.py
│   └── evaluate.py
│
├── research/
│   ├── LOG.md
│   ├── hypotheses.md
│   ├── literature.md
│   └── experiment_notes/
│
├── notebooks/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── tests/
│
└── report/
```

Change this structure if the research demands it.

Do not reorganize the repository repeatedly for aesthetic reasons.

---

# 16. Coding-Agent Instructions

Any coding agent working on CausalSight should follow these rules.

## Before implementing a substantial feature

Read:

1. `PLAN.md`;
2. relevant sections of `research/LOG.md`;
3. existing experiment configurations;
4. related code and tests.

Do not assume the user's latest idea automatically replaces previous experimental evidence.

---

## When asked to implement an experiment

First identify:

- the research question;
- hypothesis;
- independent variables;
- dependent variables;
- controls;
- potential confounders;
- required outputs.

If these are unclear, surface the ambiguity rather than silently making major scientific assumptions.

---

## Do not silently make research decisions

Implementation decisions can be autonomous.

Scientific decisions should be explicit.

For example, an agent may independently choose a reasonable serialization utility.

It should **not** silently decide that:

- one metric represents grounding;
- one prompt represents all prompting strategies;
- one dataset represents real-world behavior;
- one model family is sufficient;
- failed examples should be excluded;
- an outlier should be removed.

Such choices can materially affect conclusions and must be documented.

---

## Prefer modularity where it enables experiments

The following should ideally be replaceable:

- model adapters;
- dataset generators;
- intervention operators;
- prompting strategies;
- answer parsers;
- metrics.

However, avoid unnecessary abstraction before multiple implementations actually exist.

Research velocity matters.

---

## Never optimize away useful information

Do not delete raw predictions simply because metrics have already been computed.

Do not overwrite previous experimental results.

Do not silently regenerate datasets under an existing version identifier.

Do not alter experiment configurations after results have been produced.

Create a new experiment/version instead.

---

# 17. Use of Coding Agents

CausalSight is intentionally being developed in an era where coding agents can implement substantial software quickly.

The project should embrace this.

Agents may assist heavily with:

- implementation;
- refactoring;
- tests;
- experiment orchestration;
- visualization;
- documentation;
- debugging;
- literature organization;
- repetitive analysis.

However, the project's value should not come from how much code was produced.

The human researcher should retain ownership of:

- research questions;
- interpretation;
- experimental design;
- decisions about evidence;
- scientific claims;
- prioritization of follow-up experiments.

When an agent proposes an interpretation, treat it as a hypothesis—not ground truth.

---

# 18. Literature Review

Maintain:

`research/literature.md`

For every closely related work, record:

```text
Paper / Project:
Year:
Problem:
Method:
Dataset:
Key result:
Relationship to CausalSight:
What it already solves:
What remains unanswered:
```

Literature review should continue throughout the project rather than occurring only at the beginning.

Search particularly around concepts such as:

- VLM grounding;
- visual counterfactual evaluation;
- multimodal hallucination;
- causal evaluation of multimodal models;
- visual reliance;
- image perturbation robustness;
- multimodal shortcut learning;
- counterfactual VQA;
- visual attribution;
- multimodal calibration;
- VLM uncertainty;
- intervention-based evaluation.

Do not claim novelty until relevant literature has been investigated.

---

# 19. Development Phases

The project has approximately 20 working days at roughly two hours per day.

These phases are intentionally broad.

## Phase I — Research framing

Approx. Days 1–3

Objectives:

- understand related literature;
- formalize initial hypotheses;
- define terminology;
- identify closest existing work;
- decide what minimum experiment could falsify the initial hypothesis.

Deliverable:

A clear initial research question and minimal experimental design.

---

## Phase II — Experimental infrastructure

Approx. Days 4–7

Objectives:

- build initial scene generation;
- build intervention system;
- establish metadata representation;
- implement one model adapter;
- establish reproducible inference;
- validate ground truth.

Deliverable:

A complete original → intervention → inference → evaluation loop.

Do this before expanding the benchmark.

---

## Phase III — Baseline experiments

Approx. Days 8–11

Objectives:

- run initial model behavior experiments;
- establish ordinary accuracy;
- measure initial intervention behavior;
- inspect failures manually;
- determine whether the phenomenon is interesting.

Deliverable:

First meaningful experimental results.

At this point, reassess the project.

Do not blindly continue the original plan if the evidence suggests a better question.

---

## Phase IV — Research exploration

Approx. Days 12–16

This is deliberately the least constrained phase.

Potential directions might include:

- new interventions;
- different model scales;
- prompting effects;
- reasoning versus direct answering;
- confidence;
- error prediction;
- task-specific effects;
- compositional complexity;
- evidence removal;
- ambiguity;
- real-image validation.

Choose experiments according to what Phase III reveals.

The objective is not breadth.

The objective is finding the **most interesting defensible phenomenon**.

---

## Phase V — Consolidation

Approx. Days 17–20

Objectives:

- freeze the central experiment;
- rerun important results cleanly;
- add appropriate baselines;
- run statistical analysis;
- generate final figures;
- document negative results;
- write limitations;
- prepare reproducibility instructions;
- write the research report;
- clean the repository.

Avoid adding major new features during this phase unless they are necessary to validate the central claim.

---

# 20. Decision Gates

Several explicit decision gates should prevent sunk-cost continuation.

## Gate 1 — Infrastructure works

Can we reliably generate interventions with known expected outcomes?

If no:

Fix this before scaling anything.

---

## Gate 2 — Model is capable enough

Does the baseline model perform sufficiently well on original examples for intervention analysis to be meaningful?

If no:

Change task difficulty, model, or experiment.

---

## Gate 3 — Interesting behavior exists

Do interventions reveal systematic behavior beyond ordinary prediction errors?

If no:

Investigate alternative interventions or research questions.

---

## Gate 4 — Result survives controls

Does the interesting effect remain after reasonable baselines and confounders are considered?

If no:

Do not market the original result.

Investigate why it disappeared.

That itself may produce a useful finding.

---

## Gate 5 — Claim matches evidence

Before finalizing:

> Could a skeptical researcher reproduce this experiment and reasonably reach the same conclusion?

If not, narrow the claim.

---

# 21. What Success Looks Like

Success does NOT necessarily mean inventing a new state-of-the-art algorithm.

CausalSight succeeds if it produces at least one strong, reproducible insight.

Examples of the *form* a successful result might take:

> Model family X achieves high task accuracy but frequently fails to update predictions when answer-determining evidence changes.

or:

> Explicit reasoning improves ordinary accuracy while unexpectedly reducing invariance to irrelevant visual changes.

or:

> Intervention consistency predicts model errors substantially better than self-reported confidence under these conditions.

or:

> The initially proposed grounding metric does not outperform simple prediction consistency, revealing that the more complicated diagnostic is unnecessary.

These are examples only.

**Do not optimize experiments toward obtaining any of these results.**

A credible negative result is preferable to a manufactured positive one.

---

# 22. Final Deliverables

The ideal finished project contains:

### 1. Open-source repository

Clean and reproducible.

### 2. Controlled benchmark

Dataset or generator plus intervention metadata.

### 3. Evaluation framework

Reusable across compatible VLMs.

### 4. Experimental results

Raw outputs plus processed metrics.

### 5. Research report

Approximately paper-style:

```text
Abstract
Introduction
Related Work
Research Questions
Methodology
Experimental Setup
Results
Analysis
Ablations
Failure Analysis
Limitations
Conclusion
```

### 6. Strong figures

Prefer figures that communicate scientific findings rather than decorative visualizations.

Examples:

- causal sensitivity versus ordinary accuracy;
- nuisance invariance by reasoning category;
- model-size comparisons;
- intervention effect distributions;
- selective-risk curves;
- capability-specific failure heatmaps.

Only produce figures justified by the final experiments.

### 7. Example explorer

Optional but valuable.

A lightweight interface could show:

```text
Original Image
       ↓
Model Prediction

Counterfactual Image
       ↓
Model Prediction

Expected Behavioral Change
       ↓
Observed Behavioral Change
```

This should help people understand the research.

It should not consume a substantial fraction of project time.

---

# 23. README Philosophy

The eventual public `README.md` should be substantially shorter than this document.

Its opening should communicate:

1. the research question;
2. the central finding;
3. one compelling visual example;
4. quantitative results;
5. reproduction instructions.

Avoid opening with:

> "CausalSight is an innovative AI-powered framework..."

Prefer something scientifically concrete:

> **Do VLMs actually change their answers when the visual evidence changes?**

Then show the experiment.

---

# 24. Resume Philosophy

Do not write the final resume bullet until results exist.

The eventual bullet should contain:

**Problem + technical contribution + scale + measured result.**

General structure:

> Built CausalSight, a counterfactual evaluation framework for measuring visual evidence reliance in VLMs across [N] controlled interventions; discovered [measured finding] across [models/tasks], with [metric/effect] relative to [baseline].

Only include numbers actually supported by final experiments.

---

# 25. Scope Protection

This project has approximately **40 hours of human research time**.

Protect that time aggressively.

Before implementing something, ask:

> Does this increase our ability to answer the research question?

If not, postpone it.

Likely unnecessary during early research:

- authentication;
- production deployment;
- complex frontend;
- distributed infrastructure;
- elaborate databases;
- dozens of model integrations;
- large-scale cloud pipelines;
- excessive framework abstraction.

The project should feel scientifically deep, not artificially large.

---

# 26. Open Questions

Maintain unresolved questions here or in `research/hypotheses.md`.

Initial questions include:

- What exactly constitutes evidence reliance behaviorally?
- Which intervention families provide the cleanest causal interpretation?
- How should causal sensitivity be normalized against ordinary accuracy?
- How should ambiguous model responses be handled?
- How much synthetic complexity is required before results become meaningful?
- Which capabilities expose the largest differences?
- Are intervention-derived reliability signals useful for predicting errors?
- Does explicit reasoning change evidence reliance?
- How stable are findings across models?
- Can any results transfer from synthetic to natural images?
- How should multiple simultaneously relevant pieces of evidence be treated?
- Can models identify the evidence responsible for their own answer?
- How do visual and linguistic priors interact under intervention?

These should evolve continuously.

---

# 27. Research North Star

When uncertain about what to work on next, return to this question:

> **What experiment would most increase our understanding of whether the model's answer actually depends on the visual evidence that should determine it?**

Build that experiment.

Not the feature that produces the most code.

Not the feature that looks best in a demo.

Not the experiment most likely to produce a positive result.

The one that provides the most information.

---

# 28. Final Instruction to Agents

CausalSight is not a predetermined benchmark implementation.

It is an evolving research investigation.

Preserve optionality early.

Be rigorous with experiments.

Challenge hypotheses.

Search for confounders.

Record failures.

Prefer simple explanations.

Do not mistake correlation for causal evidence.

Do not mistake model explanations for evidence of model reasoning.

Do not mistake benchmark accuracy for understanding.

And when the data contradicts the plan:

**follow the data.**