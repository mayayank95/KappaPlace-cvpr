"""Transformer backbones for CosPlace.

ViT follows the paper: CLS → MLP → tanh → 768-D @ 224.
DINOv2 uses the same head on the pretrained CLS token. Input spatial size
must be a multiple of the patch size (14).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models import VisionTransformer

logger = logging.getLogger(__name__)

# hub entrypoint, embed dim. No register tokens: CLS is index 0.
DINOV2_BACKBONES: Dict[str, Tuple[str, int]] = {
    "DINOv2-S": ("dinov2_vits14", 384),
    "DINOv2-B": ("dinov2_vitb14", 768),
    "DINOv2-L": ("dinov2_vitl14", 1024),
    "DINOv2-G": ("dinov2_vitg14", 1536),
}
DINOV2_PATCH = 14
DINOV2_IMAGE_SIZE = 224  # 16 * 14
# Last dinov2 main commit that still imports on Python 3.9.
# Newer hub code uses `float | None` and fails on this project's interpreter.
DINOV2_HUB = "facebookresearch/dinov2:b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"

TRANSFORMER_BACKBONES = ("ViT",) + tuple(DINOV2_BACKBONES)


def is_transformer_backbone(name: str) -> bool:
    return name in TRANSFORMER_BACKBONES


class ViTClsExtractor(nn.Module):
    """Torchvision ViT-B/16 that returns the CLS token (before classification head)."""

    def __init__(self, vit: VisionTransformer):
        super().__init__()
        self.vit = vit
        # Drop classification head; CosPlace uses MLP+tanh on CLS instead.
        self.vit.heads = nn.Identity()

    @property
    def embed_dim(self) -> int:
        return int(self.vit.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # heads is Identity, so the native forward returns the CLS token.
        return self.vit(x)


def _descriptor_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"descriptor_activation must be 'tanh' or 'gelu', got {name!r}")


def _mlp_tanh_head(
    in_dim: int,
    out_dim: int,
    activation: str = "tanh",
    extra_linear: bool = False,
) -> nn.Module:
    """CLS → Linear → activation, then an optional Linear to ``out_dim``.

    The second linear is always added when the descriptor size differs from the
    CLS size. ``extra_linear`` also adds it when those sizes match
    (Linear → activation → Linear, both widths equal).
    """
    layers: list[nn.Module] = [
        nn.Linear(in_dim, in_dim),
        _descriptor_activation(activation),
    ]
    if extra_linear or out_dim != in_dim:
        layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class DINOv2ClsExtractor(nn.Module):
    """DINOv2 that returns the normalized CLS token (no classification head)."""

    def __init__(self, model: nn.Module, embed_dim: int, patch_size: int = DINOV2_PATCH):
        super().__init__()
        self.model = model
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv2 patch size is {self.patch_size}, but the input is {height}x{width}. "
                f"--image_size must be a multiple of {self.patch_size} "
                f"(for example {DINOV2_IMAGE_SIZE})."
            )
        features = self.model.forward_features(x)
        return features["x_norm_clstoken"]


def _freeze_dinov2_early(wrap: DINOv2ClsExtractor, train_all_layers: bool) -> None:
    model = wrap.model
    if train_all_layers:
        logger.debug("Train all layers of DINOv2")
        return
    for name in ("patch_embed", "cls_token", "pos_embed", "mask_token"):
        obj = getattr(model, name, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Parameter):
            obj.requires_grad = False
        else:
            for parameter in obj.parameters():
                parameter.requires_grad = False
    blocks = model.blocks
    n_freeze = max(1, len(blocks) // 2)
    for index in range(n_freeze):
        for parameter in blocks[index].parameters():
            parameter.requires_grad = False
    logger.debug(
        "Train only the last %d DINOv2 blocks; freeze earlier ones",
        len(blocks) - n_freeze,
    )


def _load_dinov2(hub_name: str) -> nn.Module:
    logger.info("Loading %s from torch.hub (%s)", hub_name, DINOV2_HUB)
    # skip_validation: this ref is a commit, not a branch or tag.
    return torch.hub.load(DINOV2_HUB, hub_name, trust_repo=True, skip_validation=True)


def _freeze_vit_early(vit_wrap: ViTClsExtractor, train_all_layers: bool) -> None:
    if train_all_layers:
        logger.debug("Train all layers of ViT")
        return
    # Freeze patch embed + first half of encoder blocks (analogous to early CNN freeze).
    for p in vit_wrap.vit.conv_proj.parameters():
        p.requires_grad = False
    vit_wrap.vit.class_token.requires_grad = False
    if vit_wrap.vit.encoder.pos_embedding is not None:
        vit_wrap.vit.encoder.pos_embedding.requires_grad = False
    layers = vit_wrap.vit.encoder.layers
    n_freeze = max(1, len(layers) // 2)
    for i in range(n_freeze):
        for p in layers[i].parameters():
            p.requires_grad = False
    logger.debug("Train only the last %d ViT encoder blocks; freeze earlier ones", len(layers) - n_freeze)


def _descriptor_head_label(activation: str, extra_linear: bool) -> str:
    head = f"Linear+{activation}"
    if extra_linear:
        head += "+Linear"
    return head


def get_transformer_backbone_and_aggregation(
    backbone_name: str,
    fc_output_dim: int,
    train_all_layers: bool = False,
    descriptor_activation: str = "tanh",
    descriptor_extra_linear: bool = False,
) -> Tuple[nn.Module, nn.Module, int]:
    """Return (backbone, aggregation, features_dim_before_optional_proj).

    ``aggregation`` maps backbone output → descriptor of size ``fc_output_dim``.
    """
    if backbone_name == "ViT":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        vit = torchvision.models.vit_b_16(weights=weights)
        backbone = ViTClsExtractor(vit)
        _freeze_vit_early(backbone, train_all_layers)
        embed = backbone.embed_dim  # 768
        aggregation = _mlp_tanh_head(
            embed, fc_output_dim, descriptor_activation, descriptor_extra_linear
        )
        logger.info(
            "CosPlace ViT: CLS → %s → %d-D (paper uses tanh, 768-D @ 224px)",
            _descriptor_head_label(descriptor_activation, descriptor_extra_linear),
            fc_output_dim,
        )
        return backbone, aggregation, embed

    if backbone_name in DINOV2_BACKBONES:
        hub_name, embed = DINOV2_BACKBONES[backbone_name]
        backbone = DINOv2ClsExtractor(_load_dinov2(hub_name), embed)
        _freeze_dinov2_early(backbone, train_all_layers)
        aggregation = _mlp_tanh_head(
            embed, fc_output_dim, descriptor_activation, descriptor_extra_linear
        )
        logger.info(
            "CosPlace %s (%s): CLS → %s → %d-D (input multiple of %d, default %dpx)",
            backbone_name,
            hub_name,
            _descriptor_head_label(descriptor_activation, descriptor_extra_linear),
            fc_output_dim,
            DINOV2_PATCH,
            DINOV2_IMAGE_SIZE,
        )
        return backbone, aggregation, embed

    raise ValueError(
        f"Unknown transformer backbone {backbone_name!r}; "
        f"expected one of {TRANSFORMER_BACKBONES}"
    )


