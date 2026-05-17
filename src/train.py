import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluate import evaluate_fer


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion, device):
    """Compute average loss without gradient updates."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_dataset,
    val_datasets,
    criterion,
    epochs=500,
    batch_size=256,
    lr=1e-3,
    device="cpu",
    verbose=True,
    log_every=50,
    patience=None,
    checkpoint_dir=None,
    selection_metric="fer",  # fer or loss
    threshold=0.3,  # only when metrics is fer
):
    """Train model. If patience is set, stop when val_loss does not improve for 'patience'
    epochs and restore the best weights."""
    if selection_metric not in {"loss", "fer"}:
        raise ValueError(f"selection_metric must be 'loss' or 'fer', got {selection_metric!r}")

    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loaders = {name: DataLoader(ds, batch_size=batch_size) for name, ds in val_datasets.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": {name: [] for name in val_datasets},
        "val_fer": {name: [] for name in val_datasets},
    }
    best_metric = float("inf")  # lower is better for both loss and FER
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)

        val_losses = {}
        val_fers = {}
        for name, loader in val_loaders.items():
            val_losses[name] = evaluate_loss(model, loader, criterion, device)
            history["val_loss"][name].append(val_losses[name])
            if selection_metric == "fer":
                val_fers[name] = evaluate_fer(
                    model,
                    val_datasets[name],
                    threshold=threshold,
                    device=device,
                    batch_size=batch_size,
                )
                history["val_fer"][name].append(val_fers[name])

        mean_loss = float(np.mean(list(val_losses.values())))
        mean_fer = float(np.mean(list(val_fers.values()))) if val_fers else None
        current_metric = mean_fer if selection_metric == "fer" else mean_loss

        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_dir is not None:
                torch.save(best_state, checkpoint_dir / "best.pth")
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % log_every == 0:
            msg = f"Epoch {epoch+1:3d}/{epochs}, Train: {train_loss:.4f}"
            msg += f", Val loss: {mean_loss:.4f}"
            if mean_fer is not None:
                msg += f", Val FER: {mean_fer:.4f}"
            print(msg)

        if patience is not None and epochs_no_improve >= patience:
            if verbose:
                metric_name = "val_fer" if selection_metric == "fer" else "val_loss"
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best: {best_epoch}, {metric_name}={best_metric:.4f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_metric"] = best_metric
    history["selection_metric"] = selection_metric

    if checkpoint_dir is not None:
        torch.save(model.state_dict(), checkpoint_dir / "last.pth")
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history
