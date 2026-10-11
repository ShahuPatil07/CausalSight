"""Scientific-contract regression tests; no model downloads or GPU required."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from causalsight.generate_binding import generate, validate
from causalsight.evaluate_binding import evaluate, parse_answer
from causalsight.binding_baselines import run as baselines
from causalsight.run_binding import run


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = cls.root / "data/binding-v1"
        cls.audit = generate(cls.dataset)
        cls.manifest = cls.dataset / "examples.jsonl"
        cls.rows = [json.loads(line) for line in cls.manifest.read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def target(self):
        return self.root / uuid.uuid4().hex

    def write_run(self, rows):
        path = self.target()
        path.write_text("".join(json.dumps({**r, "run_id": "test", "provider": "fake", "model": "fake"})+"\n" for r in rows))
        return path

    def test_balance_and_determinism(self):
        self.assertEqual(self.audit["families_by_split"], {"development":16, "evaluation":128})
        second = self.target()
        generate(second)
        self.assertEqual(self.manifest.read_bytes(), (second / "examples.jsonl").read_bytes())
        for image in (self.dataset / "images").glob("*.png"):
            self.assertEqual(image.read_bytes(), (second / "images" / image.name).read_bytes())
        with self.assertRaises(FileExistsError):
            generate(self.dataset)

    def test_audit_rejects_unintended_geometry_change(self):
        changed = copy.deepcopy(self.rows)
        changed[1]["scene"]["objects"][1]["center_x"] += 1
        path = self.dataset / "corrupted.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in changed))
        with self.assertRaises(ValueError):
            validate(path)

    def test_parser_rejects_negation_and_prose(self):
        for text in ("not red", "red or blue", "The circle is green.", "", None):
            self.assertIsNone(parse_answer(text))
        self.assertEqual(parse_answer(" Yellow.\n"), "yellow")

    def test_perfect_and_missing_results(self):
        rows = [{**r, "status":"ok", "raw_response":r["expected_answer"]} for r in self.rows if r["split"] == "evaluation"]
        result = evaluate(self.manifest, self.write_run(rows), self.target())
        self.assertEqual(result["correct_causal_update"], {"numerator":128, "denominator":128, "rate":1.0})
        self.assertEqual(result["question_switch_correct_different_colors"]["denominator"], 96)
        missing = evaluate(self.manifest, self.write_run(rows[:1]), self.target())
        self.assertEqual(missing["requests"]["missing"], 511)
        self.assertEqual(missing["complete_triplet_correctness"]["denominator"], 128)

    def test_edited_correct_is_not_correct_update(self):
        family = [r for r in self.rows if r["split"] == "evaluation"][:4]
        rows = [{**r, "status":"ok", "raw_response":r["expected_answer"]} for r in family]
        rows[0]["raw_response"] = next(c for c in ("red","blue","green","yellow") if c != rows[0]["expected_answer"])
        rows[2]["raw_response"] = rows[0]["raw_response"]
        result = evaluate(self.manifest, self.write_run(rows), self.target())
        self.assertEqual(result["causal_target_accuracy"]["numerator"], 1)
        self.assertEqual(result["correct_causal_update"]["numerator"], 0)
        self.assertEqual(result["consistently_wrong_nuisance_preservation"]["numerator"], 1)

    def test_duplicate_and_metadata_mismatch_rejected(self):
        r = next(r for r in self.rows if r["split"] == "evaluation")
        with self.assertRaises(ValueError):
            evaluate(self.manifest, self.write_run([r,r]), self.target())
        with self.assertRaises(ValueError):
            evaluate(self.manifest, self.write_run([{**r,"expected_answer":"invalid"}]), self.target())

    def test_baseline_expected_rates(self):
        results = baselines(self.manifest, self.target())
        self.assertEqual(results["constant_red"]["original_accuracy"]["rate"], .25)
        self.assertEqual(results["leftmost_object"]["original_accuracy"]["rate"], .625)
        self.assertEqual(results["distractor_object"]["original_accuracy"]["rate"], .25)
        self.assertEqual(results["constant_red"]["nuisance_answer_preservation"]["rate"], 1.)
        self.assertEqual(results["constant_red"]["complete_triplet_correctness"]["rate"], 0.)

    def test_runner_gate_and_text_only_projection(self):
        path = self.root / "configs/experiments/test.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"dataset":"data/binding-v1/examples.jsonl", "dataset_sha256":self.audit["manifest_sha256"],
                                   "expected_rows":{"development":64,"evaluation":512},"provider":"transformers","model":"fake","max_tokens":256}))
        answers = {(str((self.manifest.parent/r["image"]).resolve()),r["question"]):r["expected_answer"] for r in self.rows}
        class Fake:
            metadata = {"revision":"fake"}
            calls = 0
            def infer(self, image, prompt):
                self.calls += 1
                return {"text":answers[(str(image.resolve()),prompt)] if image else "Cannot determine without an image", "raw":{}, "usage":{}}
        adapter = Fake()
        out = self.target()
        with self.assertRaises(ValueError):
            run(path, out / "evaluation", "evaluation", adapter)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(path, out / "development", "development", adapter)
            self.assertEqual(result["state"], "complete", (out / "development/visual.jsonl").read_text()[:2000])
            self.assertEqual(adapter.calls, 66)
            result = run(path, out / "evaluation", "evaluation", adapter)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(adapter.calls, 580)
        self.assertEqual(len((out / "evaluation/text-only-requests.jsonl").read_text().splitlines()),2)
        with self.assertRaises(FileExistsError):
            run(path, out / "evaluation", "evaluation", adapter)

    def test_runner_preserves_failure_on_resume(self):
        path = self.root / "configs/experiments/resume.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"dataset":"data/binding-v1/examples.jsonl", "dataset_sha256":self.audit["manifest_sha256"],
                                   "expected_rows":{"development":64,"evaluation":512},"provider":"transformers","model":"fake","max_tokens":256}))
        answers = {(str((self.manifest.parent/r["image"]).resolve()),r["question"]):r["expected_answer"] for r in self.rows}
        class Fake:
            metadata = {"revision":"fake"}
            first = True
            def infer(self, image, prompt):
                if self.first:
                    self.first = False
                    raise RuntimeError("Simulated OOM; not model data")
                return {"text":answers[(str(image.resolve()),prompt)] if image else "red", "raw":{}, "usage":{}}
        out = self.target() / "development"
        adapter = Fake()
        with contextlib.redirect_stdout(io.StringIO()):
            first = run(path,out,"development",adapter)
            original = (out / "visual.jsonl").read_text()
            second = run(path,out,"development",adapter,resume=True)
        self.assertEqual(first["state"],"incomplete")
        self.assertEqual(second["state"],"development-gate-failed")
        self.assertTrue((out / "visual.jsonl").read_text().startswith(original))
        self.assertEqual(len((out / "visual.jsonl").read_text().splitlines()),64)
        self.assertEqual(json.loads((Path(second["metrics"]) / "metrics.json").read_text())["requests"]["error"],1)
        self.assertFalse((out / "freeze.json").exists())


if __name__ == "__main__":
    unittest.main()
