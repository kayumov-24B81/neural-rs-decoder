import argparse
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.channel import erasure_channel, make_ge_channel
from src.codec import NSYM, ClassicDecoder, HybridDecoder, K, N, OracleDecoder, encode
from src.metrics import DecodeResult, finalize_stats, init_stats, update_stats
from src.model import PositionPredictor
from src.utils import build_input, load_model

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
