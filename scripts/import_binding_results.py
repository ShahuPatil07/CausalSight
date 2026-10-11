"""Validate and import an EXP-005 Colab archive, recomputing its metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causalsight.evaluate_binding import evaluate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_run(archive: Path) -> dict:
    manifest = ROOT / "data/binding-v1/examples.jsonl"
    examples = {r["example_id"]: r for r in (json.loads(line) for line in manifest.read_text().splitlines())}
    config = json.loads((ROOT / "configs/experiments/EXP-005-binding-v1.json").read_text())
    with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(archive) as bundle:
        staging = Path(directory)
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        roots = {PurePosixPath(n).parts[0] for n in names if n.startswith("EXP-005-colab-")}
        if len(roots) != 1:
            raise ValueError("Expected exactly one Colab run")
        run_name = roots.pop()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe archive path")
            if path.parts[0] != run_name and name not in ("experiment-config.json", "bundle-hashes.json"):
                raise ValueError(f"Unexpected archive member: {name}")
            target = staging.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
        imported_config = staging / "experiment-config.json"
        if json.loads(imported_config.read_text()) != config:
            raise ValueError("Experiment configuration differs from local protocol")
        hashes = json.loads((staging / "bundle-hashes.json").read_text())
        for relative, expected_hash in hashes.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or digest(path) != expected_hash:
                raise ValueError(f"Bundled dataset/source differs from local file: {relative}")
        source = staging / run_name
        frozen = json.loads((source / "development/freeze.json").read_text())
        processed = staging / "recomputed"
        metrics = {}
        environments = {}
        for split in ("development", "evaluation"):
            environment = json.loads((source / split / "environment.json").read_text())
            environments[split] = environment
            if environment["config"] != config or environment["config_sha256"] != digest(imported_config):
                raise ValueError("Environment configuration hash mismatch")
            if config["dataset_sha256"] != digest(manifest):
                raise ValueError("Manifest hash mismatch")
            if environment["adapter"] != frozen["adapter"] or environment["config_sha256"] != frozen["config_sha256"]:
                raise ValueError("Evaluation/development identity differs from freeze")
            selected = {key: row for key, row in examples.items() if row["split"] == split}
            expected_image_hashes = {r["image"]: digest(manifest.parent / r["image"]) for r in selected.values()}
            if environment["image_hashes"] != expected_image_hashes:
                raise ValueError("Recorded image hashes differ from local dataset")
            visual = [json.loads(line) for line in (source / split / "visual.jsonl").read_text().splitlines()]
            if len(visual) != config["expected_rows"][split]:
                raise ValueError("Incomplete visual run")
            for row in visual:
                expected = selected[row["example_id"]]
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError("Raw row disagrees with dataset metadata")
                if row["run_id"] != environment["run_id"] or row["model"] != config["model"]:
                    raise ValueError("Raw run identity mismatch")
            requests = [json.loads(line) for line in (source / split / "text-only-requests.jsonl").read_text().splitlines()]
            by_source = {r["source_request_id"]: r for r in requests}
            if len(requests) != 2 or len(by_source) != 2 or {r["question"] for r in requests} != {r["question"] for r in selected.values()}:
                raise ValueError("Expected two distinct text-only source requests")
            for row in (json.loads(line) for line in (source / split / "text-only-projected.jsonl").read_text().splitlines()):
                request = by_source[row["source_request_id"]]
                if any(row.get(key) != value for key, value in request.items()) or not row.get("expanded_text_only"):
                    raise ValueError("Text-only projection differs from source response")
            for label, filename, snapshot in [("visual", "visual.jsonl", "metrics"),
                                               ("text_only", "text-only-projected.jsonl", "text-only-metrics")]:
                result = evaluate(manifest, source / split / filename, processed / split / label, split)
                captured = json.loads((source / split / snapshot / "metrics.json").read_text())
                if result != captured:
                    raise ValueError(f"Recomputed metrics differ from captured metrics: {split}/{label}")
                metrics[split+"_"+label] = result
        gate = metrics["development_visual"]["requests"]
        if any(gate[key] for key in ("error", "missing", "empty_final_answer", "budget_reached", "parse_failure")):
            raise ValueError("Captured development does not satisfy the operational freeze gate")
        destination = ROOT / "results/raw" / run_name
        analysis = ROOT / "results/processed" / run_name
        if destination.exists() or analysis.exists():
            raise FileExistsError("Run was already imported; refusing overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        analysis.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        shutil.copy2(imported_config, destination / "experiment-config.json")
        shutil.copy2(staging / "bundle-hashes.json", destination / "bundle-hashes.json")
        shutil.copytree(processed, analysis)
        provenance = {"source_archive": archive.relative_to(ROOT).as_posix(), "source_archive_sha256": digest(archive),
                      "run": run_name, "bundled_files_verified": len(hashes), "metrics_recomputed_and_equal": True,
                      "development_gate_verified": True, "model_revision": frozen["model_revision"],
                      "adapter": frozen["adapter"], "config_sha256": frozen["config_sha256"]}
        (analysis / "import-validation.json").write_text(json.dumps(provenance, indent=2)+"\n")
        baseline_dir = ROOT / "results/processed/EXP-005-binding-v1-baselines"
        heuristics = json.loads((baseline_dir / "summary.json").read_text())
        comparison = {**heuristics, "Qwen3-VL-2B-Instruct_visual": metrics["evaluation_visual"],
                      "Qwen3-VL-2B-Instruct_text_only": metrics["evaluation_text_only"]}
        (analysis / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
        lines = ["# EXP-005 baseline and model comparison", "", f"Verified Colab run: `{run_name}`. 128 evaluation families, 512 visual requests.", "",
                 "| Predictor | Original accuracy | Correct causal update | Nuisance preservation | Correct triplet | Question switch (different colors) |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name, result in comparison.items():
            values = [result[key] for key in ("original_accuracy", "correct_causal_update", "nuisance_answer_preservation", "complete_triplet_correctness", "question_switch_correct_different_colors")]
            cells = [f"{v['numerator']}/{v['denominator']} ({v['rate']:.1%})" for v in values]
            lines.append("| "+name+" | "+" | ".join(cells)+" |")
        lines += ["", "Visual model: all 512 evaluation answers correct, with no missing rows, request errors, parser failures, empty answers or truncation. Development also passed all 64 visual requests.",
                  "", "Text-only: two actual evaluation requests, projected onto repeated questions. Circle response: `Red` (parsed); square response: `The square is blue.` (unparsed under the frozen strict parser). Original accuracy is 16/128 (12.5%); 256/512 projected answers are format failures. These are not 512 independent requests, and the score partly reflects answer format. Do not change the parser after evaluation or interpret 12.5% as pure semantic accuracy.",
                  "", "Constant red's 100% invariance is compatible with wrong answers. Correctness and paired update remain separate.",
                  "", "Interpretation: this model reaches the ceiling on this simple synthetic shape-color grid. This supports success on the tested controls, not broad visual grounding or a novel causal mechanism. A harder task is needed to expose informative failures.",
                  "", f"Raw imported artifacts: `results/raw/{run_name}/`. Recomputed results and provenance: `results/processed/{run_name}/`."]
        report = "\n".join(lines)+"\n"
        (analysis / "comparison.md").write_text(report, encoding="utf-8")
        backup = baseline_dir / "heuristic-report.md"
        if not backup.exists():
            shutil.copy2(baseline_dir / "report.md", backup)
        (baseline_dir / "report.md").write_text(report, encoding="utf-8")
        (baseline_dir / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
        return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_run(args.archive.resolve()), indent=2))
