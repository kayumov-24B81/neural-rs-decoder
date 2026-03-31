import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePredictor(nn.Module):
    def __init__(self, input_size=511, hidden_size=512, output_size=255):
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

    def forward(self, x):
        return self.net(x)


class PositionPredictor(nn.Module):
    def __init__(self, input_size=511, hidden_size=512, output_size=255, dropout=0.1):
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

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
