"""Validation metrics: loss and hybrid-decoding FER over a dataset."""

import numpy as np
import torch

from .codec import NSYM, ClassicDecoder


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: str) -> float:
    """Compute average loss without gradient updates."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_fer(
    model: torch.nn.Module, val_dataset, threshold: float, device: str, batch_size: int = 256
) -> float:
    """Frame Error Rate on a fixed val_dataset using hybrid (model + RS) decoding.

    val_dataset must be in fixed mode (provides .get_raw(idx)).
    """
    model.eval()
    decoder = ClassicDecoder(nsym=NSYM)
    failures = 0
    n = len(val_dataset)

    all_predictions = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            inputs = np.stack([val_dataset.get_raw(i)[0] for i in range(start, end)])
            x = torch.from_numpy(inputs).to(device)
            probs = torch.sigmoid(model(x))
            mask = (probs > threshold).cpu().numpy()
            all_predictions.append(mask)
    predictions = np.concatenate(all_predictions, axis=0)

    for i in range(n):
        _, _, noisy, msg = val_dataset.get_raw(i)
        erase_pos = [j for j in range(len(noisy)) if predictions[i, j]]
        decoded = decoder.decode(noisy, erase_pos=erase_pos)
        if decoded != msg:
            failures += 1

    return failures / n
