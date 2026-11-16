# Model from "MixVPR: Feature Mixing for Visual Place Recognition" - https://arxiv.org/abs/2303.02190
# Parts of this code are from https://github.com/amaralibey/MixVPR

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.mixvpr_uncertainty.resnet import ResNet

# Architecture settings per output descriptor dimension.
MIXVPR_ARCH = {
    128: {"out_channels": 64, "out_rows": 2},
    512: {"out_channels": 256, "out_rows": 2},
    4096: {"out_channels": 1024, "out_rows": 4},
}


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ) -> None:
        super().__init__()

        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.out_rows = out_rows

        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio

        hw = in_h * in_w
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return x.flatten(1)


class MixVPRModel(torch.nn.Module):
    def __init__(self, agg_config=None, pretrained=True):
        super().__init__()
        self.backbone = ResNet(pretrained=pretrained)
        self.aggregator = MixVPR(**(agg_config or {}))

    def forward(self, x):
        x = transforms.Resize([320, 320], antialias=True)(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def freeze_base(self):
        """Freeze backbone and aggregator. Returns remaining trainable params."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.aggregator.parameters():
            p.requires_grad = False
        return [p for p in self.parameters() if p.requires_grad]


def mixvpr_agg_config(descriptors_dimension):
    if descriptors_dimension not in MIXVPR_ARCH:
        raise ValueError(
            f"Unsupported MixVPR descriptors_dimension={descriptors_dimension}. "
            f"Expected one of {sorted(MIXVPR_ARCH)}."
        )
    arch = MIXVPR_ARCH[descriptors_dimension]
    return {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": arch["out_channels"],
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": arch["out_rows"],
    }
