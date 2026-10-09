"""Summarize paired pilot behavior without collapsing it into one score."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

COLORS = ("red", "blue", "green", "yellow")
COLOR_PATTERN = re.compile(r"(?<![a-z])(" + "|".join(COLORS) + r")(?![a-z])", re.IGNORECASE)


def parse_color(response: str | None) -> str | None:
    """Parse only a single unambiguous color token; keep other answers unparsed."""
    if not response:
        return None
    found = {match.group(1).lower() for match in COLOR_PATTERN.finditer(response)}
    return next(iter(found)) if len(found) == 1 else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(input_path: Path, output_dir: Path, split: str = "development") -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty results directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                continue
            row["parsed_answer"] = parse_color(row.get("raw_response")) if row.get("status") == "ok" else None
            groups[str(row.get("pair_id"))][str(row.get("variant"))] = row

    pair_rows: list[dict[str, Any]] = []
    original_n = original_correct = causal_n = causal_success = nuisance_n = nuisance_invariant = 0
    conditional_n = conditional_causal = conditional_nuisance = joint = 0
    for pair_id, variants in sorted(groups.items()):
        original = variants.get("original", {})
        causal = variants.get("causal", {})
        nuisance = variants.get("nuisance", {})
        op, cp, np = (original.get("parsed_answer"), causal.get("parsed_answer"), nuisance.get("parsed_answer"))
        o_ok = op is not None and op == original.get("expected_answer")
        c_ok = cp is not None and cp == causal.get("expected_answer")
        n_same = op is not None and np is not None and op == np
        if "original" in variants:
            original_n += 1
            original_correct += int(o_ok)
        if "causal" in variants:
            causal_n += 1
            causal_success += int(c_ok)
        if "nuisance" in variants and "original" in variants:
            nuisance_n += 1
            nuisance_invariant += int(n_same)
        if o_ok and "causal" in variants and "nuisance" in variants:
            conditional_n += 1
            conditional_causal += int(c_ok)
            conditional_nuisance += int(n_same)
            joint += int(c_ok and n_same)
        pair_rows.append({
            "pair_id": pair_id,
            "split": original.get("split"),
            "predictions": {"original": op, "causal": cp, "nuisance": np},
            "expected": {key: variants.get(key, {}).get("expected_answer") for key in ("original", "causal", "nuisance")},
            "original_correct": o_ok,
            "causal_answer_update_success": c_ok,
            "nuisance_answer_preserved": n_same,
            "statuses": {key: variants.get(key, {}).get("status", "missing") for key in ("original", "causal", "nuisance")},
        })

    metrics = {
        "split": split,
        "pairs": len(pair_rows),
        "original_accuracy": {"numerator": original_correct, "denominator": original_n, "rate": _rate(original_correct, original_n)},
        "causal_answer_update_success": {"numerator": causal_success, "denominator": causal_n, "rate": _rate(causal_success, causal_n)},
        "nuisance_answer_preservation": {"numerator": nuisance_invariant, "denominator": nuisance_n, "rate": _rate(nuisance_invariant, nuisance_n)},
        "conditioned_on_original_correct": {
            "pairs": conditional_n,
            "causal_answer_update_success": _rate(conditional_causal, conditional_n),
            "nuisance_answer_preservation": _rate(conditional_nuisance, conditional_n),
            "joint_success": _rate(joint, conditional_n),
        },
        "parser": "one unambiguous occurrence from the closed color set {red, blue, green, yellow}; otherwise unparsed",
        "interpretation_note": "Descriptive pilot metrics; these are not a validated single measure of visual grounding.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "pairs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Raw inference JSONL from one run")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty results directory")
    parser.add_argument("--split", default="development", help="Split name or 'all' (default: development)")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.output_dir, args.split), indent=2))


if __name__ == "__main__":
    main()
