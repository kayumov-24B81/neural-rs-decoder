import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.channel import erasure_channel, make_ge_channel
from src.codec import NSYM, ClassicDecoder, HybridDecoder, K, N, OracleDecoder, encode
from src.metrics import DecodeResult, finalize_stats, init_stats, update_stats
from src.model import PositionPredictor
from src.utils import build_input, load_config, load_model, set_seed

# ARGUMENT PARSING


def parse_args():
    p = argparse.ArgumentParser(description="RS(255,223) decoder benchmark.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--model", default=None, help="Override model.path from config.")
    p.add_argument(
        "--channel", default=None, help="Override channel.preset (light|moderate|heavy)."
    )
    p.add_argument("--samples", type=int, default=None, help="Override benchmark.num_samples.")
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Override device. 'auto' uses CUDA if avialable.",
    )
    p.add_argument("--tag", default=None, help="Override benchmark.tag (used in output filename).")
    p.add_argument("--output", default=None, help="Override output.dir.")
    p.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    p.add_argument("--no-verbose", dest="verbose", action="store_false")
    return p.parse_args()


def apply_overrides(config, args):
    if args.model is not None:
        config["model"]["path"] = args.model
    if args.channel is not None:
        config["channel"]["preset"] = args.channel
    if args.samples is not None:
        config["benchmark"]["num_samples"] = args.samples
    if args.device is not None:
        config["device"] = args.device
    if args.tag is not None:
        config["benchmark"]["tag"] = args.tag
    if args.output is not None:
        config["output"]["dir"] = args.output
    if args.verbose is not None:
        config["output"]["verbose"] = args.verbose
    return config


# SETUP


def resolve_device(device_spec):
    if device_spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device='cuda' but CUDA is not available.")
    return device_spec


def build_channel(cfg):
    ch = cfg["channel"]
    if ch["type"] == "gilbert_elliott":
        preset = ch["preset"]
        if preset == "custom":
            params = ch.get("custom_params") or {}
            return make_ge_channel("custom", **params)
        return make_ge_channel(preset)
    if ch["type"] == "erasure":
        p_erase = ch["erasure"]["symbol_erase_prob"]
        return lambda cw: erasure_channel(cw, p_erase=p_erase)
    raise ValueError(f"Unknown channel type: {ch['type']}")


def build_model(cfg, device):
    m = cfg["model"]
    path = m["path"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. "
            f"Set decoders.neural=false or provide a valid --model path."
        )
    model = PositionPredictor(
        input_size=m["input_size"],
        hidden_size=m["hidden_size"],
        dropout=m["dropout"],
    )
    return load_model(model, path, device=device)


def build_decoders(cfg, device):
    dec_cfg = cfg["configs"]
    decoders = {}
    if dec_cfg.get("classic", False):
        decoders["classic"] = ClassicDecoder(nsym=NSYM)
    if dec_cfg.get("oracle", False):
        decoders["oracle"] = OracleDecoder(nsym=NSYM)
    if dec_cfg.get("neural", False):
        model = build_model(cfg, device)
        decoders["neural"] = HybridDecoder(
            model,
            threshold=cfg["model"]["threshold"],
            nsym=NSYM,
            device=device,
        )
    if not decoders:
        raise ValueError("No decoders enabled in config.")
    return decoders


# PASSES


def run_metrics_pass(decoders, channel_fn, num_samples, nsym, verbose):
    stats = init_stats(decoders.keys())
    iterator = range(num_samples)
    if verbose:
        iterator = tqdm(iterator, desc="metrics", unit="block")

    for _ in iterator:
        msg = bytes(os.urandom(K))
        codeword = encode(msg)
        noisy, _, _ = channel_fn(codeword)
        true_errors = {i for i in range(N) if noisy[i] != codeword[i]}
        features = build_input(noisy, nsym) if decoders else None

        for name, decoder in decoders.items():
            if name == "classic":
                decoded = decoder.decode(noisy)
                predicted = None
            elif name == "oracle":
                decoded = decoder.decode(noisy, original=codeword)
                predicted = None
            elif name == "neural":
                predicted_list = decoder.predict_positions(features)
                predicted = set(predicted_list)
                decoded = decoder.decode(noisy, predicted_positions=predicted_list)
            else:
                raise ValueError(f"Unknown decoder: {name}")

            result = DecodeResult(
                decoded=decoded,
                original=bytes(msg),
                true_errors=true_errors,
                predicted_positions=predicted,
            )
            update_stats(stats, name, result)

    return finalize_stats(stats, num_samples)


def run_timing_pass(decoders, channel_fn, num_samples, warmup, nsym, verbose):
    test_data = []
    for _ in range(num_samples + warmup):
        msg = bytes(os.urandom(K))
        codeword = encode(msg)
        noisy, _, _ = channel_fn(codeword)
        features = build_input(noisy, nsym) if "neural" in decoders else None
        test_data.append((noisy, codeword, msg, features))

    results = {}
    for name, decoder in decoders.items():
        for noisy, codeword, _, features in test_data[:warmup]:
            if name == "classic":
                decoder.decode(noisy)
            elif name == "oracle":
                decoder.decode(noisy, original=codeword)
            else:
                decoder.decode(noisy, features=features)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for noisy, codeword, _, features in test_data[warmup:]:
            if name == "classic":
                decoder.decode(noisy)
            elif name == "oracle":
                decoder.decode(noisy, original=codeword)
            else:
                decoder.decode(noisy, features=features)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results[name] = {
            "total_sec": elapsed,
            "per_frame_ms": elapsed / num_samples * 1000,
        }
        if verbose:
            print(f"  {name}: {results[name]['per_frame_ms']:.3f} ms/frame")

    return results


# OUTPUT


def _git_info():
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


def _env_info(device):
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
    }
    if device == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def _make_run_id(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = cfg["benchmark"].get("tag")
    if tag is None:
        preset = cfg["channel"].get("preset", cfg["channel"]["type"])
        model_name = (
            Path(cfg["model"]["path"]).stem if cfg["decoders"].get("neural") else "no model"
        )
        tag = f"{preset}_{model_name}"
    return f"{timestamp}_{tag}"


def save_results(metrics, timing, cfg, device, out_dir):
    run_id = _make_run_id(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}.csv"
    yaml_path = out_dir / f"{run_id}.yaml"

    columns = [
        "decoder",
        "fer",
        "ber",
        "dfr",
        "precision",
        "recall",
        "mask_covers_all",
        "num_erasures_mean",
        "num_erasures_max",
        "overflow_rate",
        "per_frame_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, m in metrics.items():
            row = {"decoder": name, **m}
            row["per_frame_ms"] = timing[name]["per_frame_ms"]
            writer.writerow(row)

    context = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "environment": _env_info(device),
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(context, f, sort_keys=False, default_flow_style=False)

    return csv_path, yaml_path


# MAIN


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = resolve_device(cfg.get("device", "auto"))
    verbose = cfg["output"].get("verbose", True)

    if verbose:
        print(f"Device: {device}")
        print(f"Channel: {cfg['channel']['type']}/{cfg['channel'].get('preset', '-')}")
        print(f"Seed: {seed}")

    channel_fn = build_channel(cfg)
    decoders = build_decoders(cfg, device)

    if verbose:
        print(f"Decoders: {list(decoders.keys())}")
        print("Pass 1/2: metrics")
    metrics = run_metrics_pass(
        decoders,
        channel_fn,
        num_samples=cfg["benchmark"]["num_samples"],
        nsym=cfg["code"]["nsym"],
        verbose=verbose,
    )

    if verbose:
        print("Pass 2/2: timing")
    timing = run_timing_pass(
        decoders,
        channel_fn,
        num_samples=cfg["timing"]["num_samples"],
        warmup=cfg["timing"]["warmup"],
        nsym=cfg["code"]["nsym"],
        verbose=verbose,
    )

    csv_path, yaml_path = save_results(
        metrics,
        timing,
        cfg,
        device,
        cfg["output"]["dir"],
    )
    print(f"Results: {csv_path}")
    print(f"Context: {yaml_path}")


if __name__ == "__main__":
    main()
