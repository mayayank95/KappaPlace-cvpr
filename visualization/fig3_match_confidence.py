#!/usr/bin/env python3
"""Pitts30k match-confidence figure: top-3 retrievals at high, median, and low joint κ.

Queries are chosen from the top-1 joint κ on Pitts30k:

    most certain    maximum joint κ. Its top-3 are all correct.
    most uncertain  minimum joint κ. Its top-3 are all wrong.
    middle          closest to the median joint κ among queries with exactly
                    one correct match in the top-3.

Each match score is the cross-term C = κ_Q κ_R cos θ, min-max normalized over
every top-20 query-reference pair on Pitts30k.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_utils import read_images_paths
from eval_metrics.scores.joint_kappa import compute_l2_joint_kappa

DATA = Path("/home/shared/datasets/pitts30k/images/test")
CACHE = ROOT / (
    "logs/CosPlace/2026-04-09_09-51-14/resume_model_2026-04-10_18-03-29/"
    "best_model_ece_recall_01/descriptors_cache/pitts30k"
)
SCORES = ROOT / "analysis/score_combine/cosplace_r50_cache/pitts30k.npz"
POSITIVES = ROOT / "analysis/score_combine/cosplace_r50_cache/pitts30k.positives.pkl"
OUT = ROOT / "reports/cosplace_r50_jt_pt/fig3_cross_term_minmax.png"

ROW_LABELS = ("Most certain", "Middle", "Most uncertain")
TOP_K = 20
BATCH = 512

FONT = "/usr/share/fonts/liberation-serif/LiberationSerif-Bold.ttf"
IMG_Y = (55, 319, 583)
IMG_X = (184, 397, 611, 824)
SIDE = 200
BORDER = 6
GREEN = (3, 197, 5)
RED = (218, 32, 32)


def _top3_hits(preds: np.ndarray, positives) -> np.ndarray:
    """How many of the top-3 references are GPS positives, one count per query."""
    counts = np.empty(len(preds), dtype=np.int32)
    for query in range(len(preds)):
        hits = {int(index) for index in positives[query]}
        counts[query] = sum(int(index) in hits for index in preds[query, :3])
    return counts


def choose_queries(joint: np.ndarray, preds: np.ndarray, positives) -> tuple[int, int, int]:
    """Most certain, middle, and most uncertain query indexes."""
    most_certain = int(np.argmax(joint))
    most_uncertain = int(np.argmin(joint))
    one_correct = np.flatnonzero(_top3_hits(preds, positives) == 1)
    distance = np.abs(joint[one_correct] - np.median(joint))
    middle = int(one_correct[int(np.argmin(distance))])
    return most_certain, middle, most_uncertain


def _kappa(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(len(values), -1).mean(axis=1)


def _cross_term_range(preds, q_var, db_var, q_desc, db_desc) -> tuple[float, float]:
    """Minimum and maximum C over every query's top-20 references."""
    c_min, c_max = np.inf, -np.inf
    for start in range(0, len(q_desc), BATCH):
        end = min(len(q_desc), start + BATCH)
        scores = compute_l2_joint_kappa(
            preds[start:end, :TOP_K],
            q_var[start:end],
            db_var,
            q_desc[start:end],
            db_desc,
        )
        c_min = min(c_min, float(scores.min()))
        c_max = max(c_max, float(scores.max()))
    return c_min, c_max


def _framed(path: str, color: tuple[int, int, int] | None) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    if color is None:
        return image
    framed = Image.new("RGB", (SIDE, SIDE), color)
    inner = image.resize((SIDE - 2 * BORDER, SIDE - 2 * BORDER), Image.Resampling.LANCZOS)
    framed.paste(inner, (BORDER, BORDER))
    return framed


def _text_width(font: ImageFont.ImageFont, text: str) -> tuple[int, tuple[int, int, int, int]]:
    box = font.getbbox(text)
    return box[2] - box[0], box


def _draw_bottom(draw, cx: float, bottom: float, text: str, font) -> None:
    width, box = _text_width(font, text)
    draw.text((cx - width / 2, bottom - box[3]), text, font=font, fill=(0, 0, 0))


def render(paths, correct, kappa, scores) -> Image.Image:
    canvas = Image.new("RGB", (1024, 781), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_row = ImageFont.truetype(FONT, 24)
    font_header = ImageFont.truetype(FONT, 20)
    font = ImageFont.truetype(FONT, 25)
    font_sub = ImageFont.truetype(FONT, 16)

    for row, (photos, hits, query_kappa, row_scores) in enumerate(
        zip(paths, correct, kappa, scores)
    ):
        y = IMG_Y[row]
        _, label_box = _text_width(font_row, ROW_LABELS[row])
        label_y = y + (SIDE - (label_box[3] - label_box[1])) / 2 - label_box[1]
        draw.text((6, label_y), ROW_LABELS[row], font=font_row, fill=(0, 0, 0))
        for col, path in enumerate(photos):
            color = None if col == 0 else (GREEN if hits[col - 1] else RED)
            x = IMG_X[col]
            canvas.paste(_framed(path, color), (x, y))
            cx = x + SIDE / 2
            score_bottom = y - 5
            header_bottom = score_bottom - 22
            if col == 0:
                _draw_bottom(draw, cx, header_bottom, "Query", font_header)
                _draw_bottom(draw, cx, score_bottom, f"κ = {query_kappa:.1f}", font)
                continue
            _draw_bottom(draw, cx, header_bottom, f"R{col}", font_header)
            subscript = f"Q→R{col}"
            number = f": {row_scores[col - 1]:.3f}"
            width_c, box_c = _text_width(font, "C")
            width_sub, box_sub = _text_width(font_sub, subscript)
            width_num, box_num = _text_width(font, number)
            x0 = cx - (width_c + width_sub + width_num) / 2
            draw.text((x0, score_bottom - box_c[3]), "C", font=font, fill=(0, 0, 0))
            draw.text((x0 + width_c, score_bottom - box_sub[3]), subscript, font=font_sub, fill=(0, 0, 0))
            draw.text((x0 + width_c + width_sub, score_bottom - box_num[3]), number, font=font, fill=(0, 0, 0))
    return canvas


def main() -> None:
    q_paths = read_images_paths(str(DATA / "queries"))
    db_paths = read_images_paths(str(DATA / "database"))
    stored = np.load(SCORES)
    preds = stored["preds"]
    joint = stored["joint"]
    q_desc = np.ascontiguousarray(np.load(CACHE / "queries_descriptors.npy"), dtype=np.float32)
    db_desc = np.ascontiguousarray(np.load(CACHE / "database_descriptors.npy"), dtype=np.float32)
    q_var = np.load(CACHE / "queries_variances.npy")
    db_var = np.load(CACHE / "database_variances.npy")
    with POSITIVES.open("rb") as handle:
        positives = pickle.load(handle)

    queries = np.asarray(choose_queries(joint, preds, positives))
    print("queries", queries.tolist())
    raw = compute_l2_joint_kappa(preds[queries, :3], q_var[queries], db_var, q_desc[queries], db_desc)
    c_min, c_max = _cross_term_range(preds, q_var, db_var, q_desc, db_desc)
    scores = (raw - c_min) / (c_max - c_min)

    paths = []
    correct = []
    for query, retrieved in zip(queries, preds[queries, :3]):
        hits = {int(index) for index in positives[int(query)]}
        paths.append([q_paths[int(query)]] + [db_paths[int(index)] for index in retrieved])
        correct.append([int(index) in hits for index in retrieved])

    image = render(paths, correct, _kappa(q_var)[queries], scores)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(f"C min {c_min:.2f}  C max {c_max:.2f}")
    print(np.array2string(scores, precision=3, suppress_small=True))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
