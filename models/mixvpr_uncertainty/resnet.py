# ResNet50 backbone used by MixVPR (Ali-bey, Chaib-draa, Giguère, WACV 2023).
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    """ResNet50 through layer3; layer4 removed (1024-d maps), as in Lightning/GSV MixVPR training."""

    def __init__(self, pretrained=True, layers_to_freeze=2):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)

        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None

        out_channels = 2048
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x
