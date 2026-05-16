import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


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
    val_dataset,
    criterion,
    epochs=500,
    batch_size=256,
    lr=1e-3,
    device="cpu",
    verbose=True,
    log_every=50,
    patience=None,
    checkpoint_dir=None,
):
    """Train model. If patience is set, stop when val_loss does not improve for 'patience'
    epochs and restore the best weights."""
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_dir is not None:
                torch.save(best_state, checkpoint_dir / "best.pth")
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if patience is not None and epochs_no_improve >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch+1} (best: {best_epoch}, \
                    val_loss={best_val_loss:.4f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    if checkpoint_dir is not None:
        torch.save(model.state_dict(), checkpoint_dir / "last.pth")
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history
