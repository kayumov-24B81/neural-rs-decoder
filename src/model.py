"""Neural models for Rs error-position prediction, and the training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePredictor(nn.Module):
    def __init__(
        self, input_size: int = 511, hidden_size: int = 512, output_size: int = 255
    ) -> None:
        """Plain 4-layer MLP, no regularization (ablation baseline).

        Kept as an ablation reference against PositionPredictor, which adds
        BatchNorm and dropout. Not used in the main benchmark.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 511,
        hidden_size: int = 512,
        output_size: int = 255,
        dropout: float = 0.1,
    ) -> None:
        """4-layer MLP with BatchNorm and dropout for error-position prediction.

        Input is the syndrome+zero-mask feature vector; output is per-symbol
        logits over the 255 codeword positions.
        """
        super().__init__()

        layers = []
        in_features = input_size

        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, gamma: float = 1.5) -> None:
        """Focal loss for class-imbalanced binary targets.

        alpha weights the positive class; gamma down-weights easy examples.
        Operates on raw logits (sigmoid is applied internally).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
