"""Query-corruption catalog for the κ study and its report (no torch/FAISS).

Levels are clean → hardest.
"""
from typing import Any, Dict

CORRUPTIONS: Dict[str, Dict[str, Any]] = {
    "blur": {
        "label": "Gaussian blur",
        "param": "radius",
        "levels": [0.0, 2.0, 4.0, 8.0],
        "sentence": "Smooths the query (defocus or motion blur) and removes fine texture.",
    },
    "noise": {
        "label": "Gaussian noise",
        "param": "sigma",
        "levels": [0.0, 15.0, 30.0, 50.0],
        "sentence": "Adds random grain on each pixel, like a high-ISO sensor.",
    },
    "jpeg": {
        "label": "JPEG compression",
        "param": "quality",
        "levels": [100, 40, 20, 10],
        "sentence": "Re-encodes the query at low quality, which creates blocking artifacts.",
    },
    "lowlight": {
        "label": "Low light",
        "param": "gain",
        "levels": [1.0, 0.5, 0.3, 0.15],
        "sentence": "Darkens the query and adds mild grain, like a nighttime capture.",
    },
    "occlusion": {
        "label": "Occlusion (black box)",
        "param": "frac",
        "levels": [0.0, 0.15, 0.30, 0.45],
        "sentence": "Hides a random rectangular patch, as if a vehicle blocked the view.",
    },
    "downsample": {
        "label": "Downsample → upsample",
        "param": "scale",
        "levels": [1.0, 0.5, 0.25, 0.125],
        "sentence": "Shrinks then resizes back, losing spatial detail like a low-res camera.",
    },
}
