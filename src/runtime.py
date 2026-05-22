"""Runtime/environment helpers shared between training and benchmarking"""

import subprocess
import sys

import torch


def git_info() -> dict:
    """Return current git commit hash and dirty-tree flag.

    Returns a dict with keys 'commit' (str) and 'dirty' (bool).
    Both are None if the git command fails (e.g. not a repository).
    """
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def env_info(device: str) -> dict:
    """Return Python/PyTorch/device info for run reproducibility context.

    Includes CUDA version and GPU name when device is 'cuda'.
    """
    info = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "device": device,
    }
    if device == "cuda":
        info["cuda"] = str(torch.version.cuda)
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
