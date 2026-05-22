"""Shared helpers: bit/byte conversions, model I/O, config loading, seeding."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from reedsolo import rs_calc_syndromes


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert a byte sequence into a flat uint8 bit vector (MSB-firts)."""
    arr = np.array(list(data), dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(data: np.ndarray) -> bytes:
    """Convert a bit vector back into a byte sequence (MSB-first)."""
    return bytes(np.packbits(data))


def get_zero_mask(data: bytes) -> np.ndarray:
    """Create a mask indicating zero-valued symbols."""
    return np.array([1.0 if b == 0 else 0.0 for b in data], dtype=np.float32)


def compute_syndrome_bits(data: bytes, nsym: int = 32) -> np.ndarray:
    """Compute the RS syndrome of a received word as a bit vector."""
    syndrome = rs_calc_syndromes(data, nsym)[1:]
    return bytes_to_bits(bytes(syndrome))


def build_input(noisy: bytes, nsym: int = 32) -> np.ndarray:
    """Build input for model (syndrome + zero mask)."""
    syndrome_bits = compute_syndrome_bits(noisy, nsym)
    zero_mask = get_zero_mask(noisy)
    return np.concatenate([syndrome_bits, zero_mask])


# TODO: build input with soft information: syndrome + mask + LLR


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    """Save model weights (state_dict) to path."""
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Load weights from path into model and move it to device."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def set_seed(seed: int) -> None:
    """Capture all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file into a dict.

    If path is not found as-is, retry relative to the repository root
    (parent of this file's directory).
    """
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / path
    with open(p) as f:
        return yaml.safe_load(f)
