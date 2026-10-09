"""Generate small, paired synthetic scenes for intervention-pipeline pilots."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

COLORS = ("red", "blue", "green", "yellow")
RGB = {
    "red": (220, 45, 45),
    "blue": (45, 95, 210),
    "green": (35, 155, 85),
    "yellow": (235, 190, 35),
}
BACKGROUND = (248, 248, 244)
INK = (40, 40, 40)
WIDTH, HEIGHT = 384, 256


def _render(path: Path, target_color: str, distractor_color: str, target_x: int, distractor_x: int) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    y = HEIGHT // 2
    radius = 38
    # The circle is the queried object. The square is an irrelevant distractor.
    draw.ellipse((target_x-radius, y-radius, target_x+radius, y+radius), fill=RGB[target_color], outline=INK, width=3)
    side = 62
    draw.rectangle((distractor_x-side//2, y-side//2, distractor_x+side//2, y+side//2), fill=RGB[distractor_color], outline=INK, width=3)
    image.save(path, format="PNG")


def generate(output: Path, count: int = 100, seed: int = 17) -> dict[str, Any]:
    """Write deterministic original/causal/nuisance PNGs and paired JSONL rows."""
    if count < 1:
        raise ValueError("count must be at least 1")
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        scene_id = f"pilot-{index:05d}"
        colors = rng.sample(COLORS, 2)
        target_color, distractor_color = colors
        # Mirror positions to prevent position from identifying the queried answer.
        if rng.random() < 0.5:
            target_x, distractor_x = 112, 272
        else:
            target_x, distractor_x = 272, 112
        question = "What color is the circle? Answer with one color."
        states = {
            "original": (target_color, distractor_color),
            "causal": (rng.choice([c for c in COLORS if c != target_color]), distractor_color),
            "nuisance": (target_color, next(c for c in COLORS if c not in (target_color, distractor_color))),
        }
        for kind, (target_after, distractor_after) in states.items():
            image_rel = f"images/{scene_id}-{kind}.png"
            _render(output / image_rel, target_after, distractor_after, target_x, distractor_x)
            if kind == "original":
                intervention = None
                answer = target_color
            elif kind == "causal":
                intervention = {"type": "causal", "changed_object": "circle", "changed_attribute": "color", "before": target_color, "after": target_after}
                answer = target_after
            else:
                intervention = {"type": "nuisance", "changed_object": "square", "changed_attribute": "color", "before": distractor_color, "after": distractor_after}
                answer = target_color
            rows.append({
                "example_id": f"{scene_id}-{kind}",
                "pair_id": scene_id,
                "scene_id": scene_id,
                "split": "development" if index % 5 == 0 else "train",
                "variant": kind,
                "image": image_rel,
                "question": question,
                "expected_answer": answer,
                "original_answer": target_color,
                "intervention": intervention,
                "scene": {
                    "objects": [
                        {"name": "circle", "shape": "circle", "color": target_after, "center_x": target_x, "center_y": HEIGHT//2},
                        {"name": "square", "shape": "square", "color": distractor_after, "center_x": distractor_x, "center_y": HEIGHT//2},
                    ]
                },
                "seed": seed,
                "generator_version": "0.1.0",
            })
    manifest = output / "examples.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"examples": len(rows), "scenes": count, "manifest": str(manifest), "images": str(images)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pilot-v0"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()

