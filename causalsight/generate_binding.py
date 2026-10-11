"""Generate and audit the balanced object-color binding experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw
from .generate_pilot import BACKGROUND, COLORS, INK, RGB, WIDTH, HEIGHT

VARIANTS = ("original", "causal", "nuisance", "question_switch")


def render(scene: dict) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for obj in scene["objects"]:
        x, y, r = obj["center_x"], obj["center_y"], obj["radius"]
        bounds = (x-r, y-r, x+r, y+r)
        draw_fn = draw.ellipse if obj["shape"] == "circle" else draw.rectangle
        draw_fn(bounds, fill=RGB[obj["color"]], outline=INK, width=3)
    return image


def generate(output: Path, seed: int = 11017) -> dict:
    if output.exists():
        raise FileExistsError(f"Dataset version already exists: {output}")
    (output / "images").mkdir(parents=True)
    rows = []
    combinations = list(itertools.product(range(2), range(2), range(4), range(4)))
    # Development covers shape, side and target color; evaluation covers the full factorial twice.
    development = [(s, p, t, (t+2*s+p) % 4) for s, p, t in itertools.product(range(2), range(2), range(4))]
    for split, specs in [("development", [(0, c) for c in development]),
                         ("evaluation", [(rep, c) for rep in range(2) for c in combinations])]:
        for index, (rep, (s, p, t, d)) in enumerate(specs):
            pair_id = f"binding-{split}-{index:04d}"
            family_seed = int.from_bytes(hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()[:8], "big")
            rng = random.Random(family_seed)
            target = ("circle", "square")[s]
            distractor = ("square", "circle")[s]
            xs = [112+rng.randint(-14, 14), 272+rng.randint(-14, 14)]
            scene = {"objects": [{"name": name, "shape": name, "color": COLORS[color],
                                   "center_x": xs[side], "center_y": 128+rng.randint(-12, 12),
                                   "radius": rng.randint(30, 38)}
                                  for name, color, side in [(target, t, p), (distractor, d, 1-p)]]}
            for variant in VARIANTS:
                state = copy.deepcopy(scene)
                queried = target
                intervention = None
                if variant in ("causal", "nuisance"):
                    changed_index = 0 if variant == "causal" else 1
                    old_index = t if changed_index == 0 else d
                    offset = 1 + ((rep+s+p+(d if changed_index == 0 else t)) % 3)
                    after = COLORS[(old_index+offset) % 4]
                    state["objects"][changed_index]["color"] = after
                    intervention = {"type": variant, "changed_object": state["objects"][changed_index]["name"],
                                    "changed_attribute": "color", "before": COLORS[old_index], "after": after}
                if variant == "question_switch":
                    queried = distractor
                    intervention = {"type": variant, "changed_attribute": "queried_shape", "before": target, "after": distractor}
                image_rel = f"images/{pair_id}-{'original' if variant == 'question_switch' else variant}.png"
                if variant != "question_switch":
                    render(state).save(output / image_rel)
                answer = next(o["color"] for o in state["objects"] if o["name"] == queried)
                rows.append({"example_id": f"{pair_id}-{variant}", "pair_id": pair_id, "scene_id": pair_id,
                             "split": split, "variant": variant, "image": image_rel, "scene": state,
                             "question": f"What color is the {queried}? Answer with one color.",
                             "queried_shape": queried, "target_shape": target, "target_side": ("left", "right")[p],
                             "expected_answer": answer, "original_answer": COLORS[t], "intervention": intervention,
                             "seed": seed, "family_seed": family_seed, "generator_version": "binding-1.0.0"})
    with (output / "examples.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True)+"\n")
    audit = validate(output / "examples.jsonl")
    (output / "audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    return audit


def validate(manifest: Path) -> dict:
    groups = defaultdict(dict)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    original_inputs = set()
    original_images_by_split = defaultdict(set)
    for row in rows:
        if row["example_id"] in ids or row["variant"] in groups[row["pair_id"]]:
            raise ValueError("Duplicate example or family variant")
        ids.add(row["example_id"])
        groups[row["pair_id"]][row["variant"]] = row
        image_path = (manifest.parent / row["image"]).resolve()
        if not image_path.is_relative_to(manifest.parent.resolve()):
            raise ValueError("Image path escapes dataset")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if ImageChops.difference(image, render(row["scene"])).getbbox():
                raise ValueError(f"Rendered pixels disagree with metadata: {row['example_id']}")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
        queried = next(o for o in row["scene"]["objects"] if o["name"] == row["queried_shape"])
        if queried["color"] != row["expected_answer"] or row["question"] != f"What color is the {row['queried_shape']}? Answer with one color.":
            raise ValueError("Question/label mismatch")
        if row["variant"] == "original":
            identity = (digest, row["question"])
            if identity in original_inputs:
                raise ValueError("Repeated original input")
            original_inputs.add(identity)
            original_images_by_split[row["split"]].add(digest)
    counts, transitions = defaultdict(Counter), defaultdict(Counter)
    for family, variants in groups.items():
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Incomplete family: {family}")
        original = variants["original"]
        target, distractor = original["scene"]["objects"]
        left = target["center_x"] < distractor["center_x"]
        if original["target_side"] != ("left" if left else "right"):
            raise ValueError("Target-side metadata mismatch")
        counts[original["split"]][(target["name"], original["target_side"], target["color"], distractor["color"])] += 1
        for variant, changed_index in [("causal", 0), ("nuisance", 1)]:
            row = variants[variant]
            differences = [(i, key) for i, (a, b) in enumerate(zip(original["scene"]["objects"], row["scene"]["objects"]))
                           for key in a if a[key] != b[key]]
            changed = row["scene"]["objects"][changed_index]
            before = original["scene"]["objects"][changed_index]
            expected_edit = {"type": variant, "changed_object": changed["name"], "changed_attribute": "color",
                             "before": before["color"], "after": changed["color"]}
            if differences != [(changed_index, "color")] or row["intervention"] != expected_edit:
                raise ValueError("Intervention changes the wrong attribute or has invalid metadata")
            if row["question"] != original["question"]:
                raise ValueError("Visual intervention changes question")
            transitions[original["split"]+"_"+variant][before["color"]+"->"+changed["color"]] += 1
        switched = variants["question_switch"]
        if switched["scene"] != original["scene"] or switched["image"] != original["image"] or switched["queried_shape"] != distractor["name"]:
            raise ValueError("Question switch must preserve the image and query the other object")
        if len({r["split"] for r in variants.values()}) != 1:
            raise ValueError("Family spans splits")
    if original_images_by_split["development"] & original_images_by_split["evaluation"]:
        raise ValueError("Image overlap between development and evaluation")
    if len(counts["evaluation"]) != 64 or set(counts["evaluation"].values()) != {2}:
        raise ValueError("Evaluation must have two families per factorial cell")
    if sum(counts["development"].values()) != 16:
        raise ValueError("Development must contain 16 families")
    return {"families": len(groups), "rows": len(rows), "images": len(list((manifest.parent / "images").glob("*.png"))),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "families_by_split": {s: sum(c.values()) for s, c in counts.items()},
            "evaluation_cells": len(counts["evaluation"]), "families_per_evaluation_cell": 2,
            "transition_counts": {key: dict(sorted(value.items())) for key, value in transitions.items()},
            "checks": "Exact rendered pixels, single color changes, question switches, labels, unique original inputs, split separation and factorial balance passed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/binding-v1"))
    parser.add_argument("--seed", type=int, default=11017)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.validate) if args.validate else generate(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
