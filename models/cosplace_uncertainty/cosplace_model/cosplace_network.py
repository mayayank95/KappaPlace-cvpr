import logging
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.cosplace_uncertainty.cosplace_model.layers import Flatten, L2Norm, GeM
from models.cosplace_uncertainty.cosplace_model.transformer_backbones import (
    TRANSFORMER_BACKBONES,
    get_transformer_backbone_and_aggregation,
    is_transformer_backbone,
)

logger = logging.getLogger(__name__)

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "EfficientNet_B0": 1280,
    "EfficientNet_B1": 1280,
    "EfficientNet_B2": 1408,
    "EfficientNet_B3": 1536,
    "EfficientNet_B4": 1792,
    "EfficientNet_B5": 2048,
    "EfficientNet_B6": 2304,
    "EfficientNet_B7": 2560,
}

SUPPORTED_BACKBONES = tuple(CHANNELS_NUM_IN_LAST_CONV.keys()) + TRANSFORMER_BACKBONES


class GeoLocalizationNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        fc_output_dim: int,
        train_all_layers: bool = False,
        descriptor_activation: str = "tanh",
        descriptor_extra_linear: bool = False,
    ):
        """Base model for GeoLocalization: backbone + aggregation.

        CNNs use L2Norm → GeM → Flatten → Linear (original CosPlace).
        Transformers follow the CosPlace paper (CLS → MLP → tanh):
          ViT, and DINOv2-S/B/L/G.

        Args:
            backbone: torchvision CNN name, ViT, or a DINOv2-* name.
            fc_output_dim: output descriptor dimension.
            train_all_layers: whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone in SUPPORTED_BACKBONES, (
            f"backbone must be one of {list(SUPPORTED_BACKBONES)}"
        )
        self.backbone_name = backbone
        if is_transformer_backbone(backbone):
            self.backbone, self.aggregation, _ = get_transformer_backbone_and_aggregation(
                backbone,
                fc_output_dim,
                train_all_layers,
                descriptor_activation,
                descriptor_extra_linear,
            )
        else:
            self.backbone, features_dim = get_backbone(backbone, train_all_layers)
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
            )

    def freeze_base(self):
        """Freeze backbone and aggregation parameters. Returns list of remaining trainable params."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.aggregation.parameters():
            p.requires_grad = False
        trainable = [p for p in self.parameters() if p.requires_grad]
        return trainable

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(
            __import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]),
            f"{backbone_name}_Weights",
        )
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name: str, train_all_layers: bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        if train_all_layers:
            logger.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logger.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")

        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        if train_all_layers:
            logger.debug("Train all layers of the VGG-16")
        else:
            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logger.debug("Train last layers of the VGG-16, freeze the previous ones")

    elif backbone_name.startswith("EfficientNet"):
        if train_all_layers:
            logger.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.features.named_children():
                if name == "5":  # Freeze layers before block 5
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logger.debug(f"Train only the last three blocks of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    else:
        raise ValueError(f"Unsupported CNN backbone: {backbone_name}")

    backbone = torch.nn.Sequential(*layers)
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim
