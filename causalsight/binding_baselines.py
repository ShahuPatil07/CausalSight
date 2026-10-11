"""Run deterministic shortcut baselines with the same manifest and evaluator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .evaluate_binding import evaluate


def run(manifest: Path, output: Path, split: str = "evaluation") -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    examples = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = {}
    for baseline in ("constant_red", "leftmost_object", "distractor_object"):
        raw = output / (baseline+".jsonl")
        with raw.open("x", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                if example["split"] != split:
                    continue
                objects = example["scene"]["objects"]
                if baseline == "constant_red":
                    answer = "red"
                elif baseline == "leftmost_object":
                    answer = min(objects, key=lambda o: o["center_x"])["color"]
                else:
                    answer = next(o["color"] for o in objects if o["name"] != example["queried_shape"])
                row = {**example, "status": "ok", "raw_response": answer, "run_id": f"baseline-{baseline}-{split}",
                       "provider": "metadata-heuristic", "model": baseline}
                handle.write(json.dumps(row, sort_keys=True)+"\n")
        results[baseline] = evaluate(manifest, raw, output / baseline, split)
    (output / "summary.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    lines = ["# Shortcut baseline results", "", "Metadata heuristics, not model inference.", "",
             "| Baseline | Original accuracy | Correct causal update | Nuisance preservation | Correct triplet | Question switch (different colors) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        cells = [f"{metrics[key]['numerator']}/{metrics[key]['denominator']} ({metrics[key]['rate']:.1%})"
                 for key in ("original_accuracy", "correct_causal_update", "nuisance_answer_preservation", "complete_triplet_correctness", "question_switch_correct_different_colors")]
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "Nuisance preservation can be high for a consistently wrong predictor. Inspect correctness jointly."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    args = parser.parse_args()
    results = run(args.dataset, args.output_dir, args.split)
    print(json.dumps({name: data["original_accuracy"] for name, data in results.items()}, indent=2))
