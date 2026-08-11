"""Vector heads (MLP, CNN1D) + image backbones (reused from v32)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .compat import build_image_backbone  # ResNet50 / ConvNeXt-Tiny / etc. (v32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """Length-agnostic 1D CNN (AdaptiveMaxPool handles max/avg/raw vector lengths)."""

    def __init__(self, num_classes: int, output_length: int = 256, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.adapt = nn.AdaptiveMaxPool1d(output_length)
        self.fc1 = nn.Linear(64 * output_length, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.adapt(x).reshape(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


def build_model(cfg: dict, input_dim: int, n_classes: int) -> nn.Module:
    model = cfg["model"]
    if model == "mlp":
        return MLP(input_dim, n_classes)
    if model == "cnn1d":
        return CNN1D(n_classes)
    # image backbones (resnet50 / convnext_tiny / ...) — reused from v32
    return build_image_backbone(model, n_classes, unfreeze=cfg["unfreeze"],
                                pretrained=bool(cfg["pretrained"]))
