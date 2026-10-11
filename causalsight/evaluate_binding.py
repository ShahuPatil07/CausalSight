"""Evaluate binding-v1 against its manifest, retaining missing and failed requests."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ("original", "causal", "nuisance", "question_switch")


def parse_answer(text: str | None) -> str | None:
    # Do not extract colors from refusals, negations, or explanations.
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*(red|blue|green|yellow)[.!]?\s*", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def rate(success: int, total: int) -> dict:
    return {"numerator": success, "denominator": total, "rate": success/total if total else None}


def summarize(pairs: list[dict]) -> dict:
    n = len(pairs)
    correct_originals = [p for p in pairs if p["original_correct"]]
    different = [p for p in pairs if not p["same_color"]]
    same = [p for p in pairs if p["same_color"]]
    def metric(key: str, population: list[dict] = pairs) -> dict:
        return rate(sum(p[key] for p in population), len(population))
    return {
        "families": n, "original_accuracy": metric("original_correct"),
        "causal_target_accuracy": metric("causal_correct"),
        "correct_causal_update": metric("correct_causal_update"),
        "nuisance_answer_preservation": metric("nuisance_preserved"),
        "correct_nuisance_preservation": metric("correct_nuisance_preservation"),
        "consistently_wrong_nuisance_preservation": metric("wrong_nuisance_preservation"),
        "complete_triplet_correctness": metric("triplet_correct"),
        "question_switch_correct_different_colors": metric("question_switch_correct", different),
        "question_switch_correct_same_colors": metric("question_switch_correct", same),
        "conditioned_on_original_correct": {
            "correct_causal_update": metric("correct_causal_update", correct_originals),
            "nuisance_answer_preservation": metric("nuisance_preserved", correct_originals),
            "complete_triplet_correctness": metric("triplet_correct", correct_originals),
        },
    }


def evaluate(manifest: Path, raw_path: Path, output_dir: Path, split: str = "evaluation") -> dict:
    expected = {r["example_id"]: r for r in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
                if r["split"] == split}
    if not expected:
        raise ValueError(f"No manifest examples for split {split}")
    saved = {}
    identities = set()
    diagnostics = Counter()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["example_id"]
        if key not in expected:
            raise ValueError(f"Unexpected example for selected split: {key}")
        if key in saved:
            raise ValueError(f"Duplicate prediction: {key}")
        for field in ("pair_id", "variant", "split", "expected_answer", "question", "image"):
            if row.get(field) != expected[key][field]:
                raise ValueError(f"Prediction metadata disagrees with manifest: {key}, {field}")
        identities.add((row.get("run_id"), row.get("provider"), row.get("model")))
        saved[key] = row
    if len(identities) > 1:
        raise ValueError("Cannot combine different runs/providers/models")
    groups = defaultdict(dict)
    for key, example in expected.items():
        row = saved.get(key, {"status": "missing"})
        status = row.get("status", "error")
        parsed = parse_answer(row.get("raw_response")) if status == "ok" else None
        diagnostics[status] += 1
        if status == "ok":
            diagnostics["parse_failure"] += parsed is None
            diagnostics["empty_final_answer"] += not row.get("raw_response", "").strip()
            provider = row.get("provider_response", {})
            diagnostics["budget_reached"] += bool(row.get("budget_reached") or provider.get("done_reason") == "length")
        groups[example["pair_id"]][example["variant"]] = {"example": example, "answer": parsed, "status": status}
    pairs = []
    for pair_id, members in sorted(groups.items()):
        if set(members) != set(VARIANTS):
            raise ValueError(f"Manifest family incomplete: {pair_id}")
        prediction = {v: members[v]["answer"] for v in VARIANTS}
        correct = {v: prediction[v] is not None and prediction[v] == members[v]["example"]["expected_answer"] for v in VARIANTS}
        original = members["original"]["example"]
        objects = original["scene"]["objects"]
        invariant = prediction["original"] is not None and prediction["original"] == prediction["nuisance"]
        changed = prediction["original"] is not None and prediction["causal"] is not None and prediction["original"] != prediction["causal"]
        pairs.append({"pair_id": pair_id, "predictions": prediction,
                      "expected": {v: members[v]["example"]["expected_answer"] for v in VARIANTS},
                      "statuses": {v: members[v]["status"] for v in VARIANTS},
                      "target_shape": original["target_shape"], "target_side": original["target_side"],
                      "target_color": original["expected_answer"], "same_color": objects[0]["color"] == objects[1]["color"],
                      "original_correct": correct["original"], "causal_correct": correct["causal"],
                      "correct_causal_update": correct["original"] and correct["causal"] and changed,
                      "nuisance_preserved": invariant,
                      "correct_nuisance_preservation": correct["original"] and correct["nuisance"] and invariant,
                      "wrong_nuisance_preservation": invariant and not correct["original"],
                      "triplet_correct": all(correct[v] for v in ("original", "causal", "nuisance")),
                      "question_switch_correct": correct["original"] and correct["question_switch"]})
    metrics = {"metric_version": "binding-1.0.0", "split": split, **summarize(pairs),
               "requests": {"expected": len(expected), "saved": len(saved),
                            **{key: diagnostics[key] for key in ("ok", "error", "missing", "parse_failure", "empty_final_answer", "budget_reached")}},
               "parser": "Exactly one color word, optionally followed by . or !; no prose or thinking-field extraction.",
               "interpretation": "Descriptive results for a fixed balanced synthetic grid; no population confidence intervals or general grounding claim.",
               "strata": {field: {str(value): summarize([p for p in pairs if p[field] == value])
                                  for value in sorted({p[field] for p in pairs})}
                          for field in ("target_shape", "target_side", "target_color", "same_color")}}
    if any(r.get("expanded_text_only") for r in saved.values()):
        metrics["text_only_projection"] = {
            "actual_requests": len({r["source_request_id"] for r in saved.values()}),
            "projected_prediction_rows": len(saved),
            "note": "Repeated identical questions share a text-only response; projected rows are not independent trials.",
        }
    if output_dir.exists():
        raise FileExistsError(f"Results already exist: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    with (output_dir / "pairs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, sort_keys=True)+"\n")
    lines = [f"# Binding experiment: {split}", "", f"Families: {len(pairs)}. Saved requests: {len(saved)}/{len(expected)}.", "",
             "| Measurement | Success / total | Rate |", "|---|---:|---:|"]
    for key, value in metrics.items():
        if isinstance(value, dict) and "numerator" in value:
            percent = f"{value['rate']:.1%}" if value["rate"] is not None else "undefined"
            lines.append(f"| {key} | {value['numerator']}/{value['denominator']} | {percent} |")
    lines += ["", metrics["interpretation"], "", "Missing, failed and unparseable requests remain in denominators."]
    (output_dir / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.run, args.output_dir, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "strata"}, indent=2))


if __name__ == "__main__":
    main()
