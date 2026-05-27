"""RS(255,223) decoder benchmark: runs metrics, timing, and encoding passes."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.channel import erasure_channel, make_ge_channel
from src.codec import NSYM, ClassicDecoder, HybridDecoder, K, N, OracleDecoder, T, encode
from src.metrics import DecodeResult, finalize_stats, init_stats, update_stats
from src.model import PositionPredictor
from src.pcap_source import load_pcap_messages
from src.runtime import env_info, git_info
from src.utils import build_input, load_config, load_model, set_seed

# ARGUMENT PARSING


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
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
        help="Override device. 'auto' uses CUDA if available.",
    )
    p.add_argument("--tag", default=None, help="Override benchmark.tag (used in output filename).")
    p.add_argument("--output", default=None, help="Override output.dir.")
    p.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    p.add_argument("--no-verbose", dest="verbose", action="store_false")
    p.add_argument(
        "--decoders",
        default=None,
        help="Comma-separated list of decoders to enable (classic,oracle,neural). "
        "Overrides decoders.* from config.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override seed.")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override model.threshold (neural decoder mask threshold).",
    )
    p.add_argument(
        "--message-source",
        default=None,
        choices=["random", "pcap"],
        help="Override message_source (random|pcap).",
    )
    return p.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of the loaded config dict."""
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
    if args.seed is not None:
        config["seed"] = args.seed
    if args.threshold is not None:
        config["model"]["threshold"] = args.threshold
    if args.message_source is not None:
        config["message_source"] = args.message_source
    if args.decoders is not None:
        valid = {"classic", "oracle", "neural"}
        requested = {d.strip() for d in args.decoders.split(",") if d.strip()}
        unknown = requested - valid
        if unknown:
            raise ValueError(f"Unknown decoders: {sorted(unknown)}. Valid: {sorted(valid)}.")
        for d in valid:
            config["decoders"][d] = d in requested
    return config


# SETUP


def resolve_device(device_spec: str) -> str:
    """Resolve a device spec ('auto'|'cpu'|'cuda') to a concrete device."""
    if device_spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device='cuda' but CUDA is not available.")
    return device_spec


def make_message_source(cfg: dict):
    """Return a zero-arg callable yielding one K-byte message per call.

    With message_sources='pcap', messages are real payloads from a pcap
    file, cycled if exhausted. With 'random', messages are random bytes.
    """
    src = cfg.get("message_source", "random")
    if src == "random":
        return lambda: np.random.bytes(K)
    if src == "pcap":
        messages = load_pcap_messages(cfg["pcap"]["path"], msg_len=K)
        state = {"i": 0}

        def next_message():
            msg = messages[state["i"] % len(messages)]
            state["i"] += 1
            return msg

        return next_message
    raise ValueError(f"Unknown message_source: {src!r}")


def build_channel(cfg: dict):
    """Build a channel callable from the config's channel section."""
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


def build_model(cfg: dict, device: str) -> torch.nn.Module:
    """Load the neural model with weights from the configured path."""
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


def build_decoders(cfg: dict, device: str) -> dict:
    """Build the set of enabled decoders (classic/oracle/neural)."""
    dec_cfg = cfg["decoders"]
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


def run_metrics_pass(
    decoders: dict, channel_fn, next_message, num_samples: int, nsym: int, verbose: bool
) -> dict:
    """Run the quality pass: decode num_samples blocks, accumulate metrics.

    Metrics are accumulated into three buckets: all blocks, block with
    <=T errors (within BM correction capability), and blocks with > T
    errors. Returns a dict with finalized metrics and block counts per bucket.
    """
    stats_all = init_stats(decoders.keys())
    stats_le16 = init_stats(decoders.keys())
    stats_gt16 = init_stats(decoders.keys())
    n_le16 = 0
    n_gt16 = 0

    iterator = range(num_samples)
    if verbose:
        iterator = tqdm(iterator, desc="metrics", unit="block")

    for _ in iterator:
        msg = msg = next_message()
        codeword = encode(msg)
        noisy, _, _ = channel_fn(codeword)
        true_errors = {i for i in range(N) if noisy[i] != codeword[i]}
        features = build_input(noisy, nsym) if "neural" in decoders else None

        is_le16 = len(true_errors) <= T
        if is_le16:
            n_le16 += 1
        else:
            n_gt16 += 1

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
            update_stats(stats_all, name, result)
            if is_le16:
                update_stats(stats_le16, name, result)
            else:
                update_stats(stats_gt16, name, result)

    return {
        "all": finalize_stats(stats_all, num_samples),
        "le16": finalize_stats(stats_le16, n_le16) if n_le16 else None,
        "gt16": finalize_stats(stats_gt16, n_gt16) if n_gt16 else None,
        "counts": {"all": num_samples, "le16": n_le16, "gt16": n_gt16},
    }


def run_timing_pass(
    decoders: dict,
    channel_fn,
    next_message,
    num_samples: int,
    warmup: int,
    nsym: int,
    verbose: bool,
) -> dict:
    """Run the decoding-timing pass: measure per-frame decode time per decoder."""
    test_data = []
    for _ in range(num_samples + warmup):
        msg = next_message()
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


def run_encoding_pass(next_message, num_samples: int, warmup: int, verbose: bool) -> dict:
    """Run the encoding-timing pass: measure per-frame RS encode time."""
    test_msgs = [next_message() for _ in range(num_samples + warmup)]

    for msg in test_msgs[:warmup]:
        encode(msg)

    start = time.perf_counter()
    for msg in test_msgs[warmup:]:
        encode(msg)
    elapsed = time.perf_counter() - start

    result = {
        "total_sec": elapsed,
        "per_frame_ms": elapsed / num_samples * 1000,
    }
    if verbose:
        print(f"  encoding: {result['per_frame_ms']:.3f} ms/frame")

    return result


# OUTPUT


def _make_run_id(cfg: dict) -> str:
    """Build a unique run id: timestamp plus a tag derived from config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = cfg["benchmark"].get("tag")
    if tag is None:
        preset = cfg["channel"].get("preset", cfg["channel"]["type"])
        model_name = (
            Path(cfg["model"]["path"]).stem if cfg["decoders"].get("neural") else "no-model"
        )
        tag = f"{preset}_{model_name}"
    return f"{timestamp}_{tag}"


def save_results(
    metrics: dict, timing: dict, encoding: dict, cfg: dict, device: str, out_dir: str | Path
) -> tuple:
    """Write the results CSV and the run-context YAML; return their paths."""
    run_id = _make_run_id(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}.csv"
    yaml_path = out_dir / f"{run_id}.yaml"

    columns = [
        "decoder",
        "error_bucket",
        "n_blocks",
        "fer",
        "fer_ci_low",
        "fer_ci_high",
        "ber",
        "dfr",
        "dfr_ci_low",
        "dfr_ci_high",
        "precision",
        "recall",
        "mask_covers_all",
        "mask_covers_all_ci_low",
        "mask_covers_all_ci_high",
        "num_erasures_mean",
        "num_erasures_max",
        "overflow_rate",
        "overflow_rate_ci_low",
        "overflow_rate_ci_high",
        "per_frame_ms",
        "encoding_time_ms",
        "decode_to_encode_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for bucket in ("all", "le16", "gt16"):
            bucket_metrics = metrics[bucket]
            if bucket_metrics is None:
                continue
            n_blocks = metrics["counts"][bucket]
            for name, m in bucket_metrics.items():
                row = {"decoder": name, "error_bucket": bucket, "n_blocks": n_blocks, **m}
                # timing/encoding are bucket-independent: attach only to 'all'
                if bucket == "all":
                    row["per_frame_ms"] = timing[name]["per_frame_ms"]
                    row["encoding_time_ms"] = encoding["per_frame_ms"]
                    row["decode_to_encode_ratio"] = (
                        timing[name]["per_frame_ms"] / encoding["per_frame_ms"]
                    )
                writer.writerow(row)

    context = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "environment": env_info(device),
        "config": cfg,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(context, f, sort_keys=False, default_flow_style=False)

    return csv_path, yaml_path


# SUMMARY


def print_summary(metrics: dict) -> None:
    """Print a compact human-readable summary of the metrics pass."""
    counts = metrics["counts"]
    print()
    print("=" * 64)
    print(
        f"Blocks: {counts['all']} total  |  <={T} err: {counts['le16']}  "
        f">{T} err: {counts['gt16']}"
    )
    print("=" * 64)
    for name in metrics["all"]:
        m_all = metrics["all"][name]
        fer = m_all["fer"]
        lo, hi = m_all["fer_ci_low"], m_all["fer_ci_high"]
        fer_le = metrics["le16"][name]["fer"] if metrics["le16"] else float("nan")
        fer_gt = metrics["gt16"][name]["fer"] if metrics["gt16"] else float("nan")
        print(
            f"  {name:8s} FER={fer:.4f} [{lo:.4f}, {hi:.4f}]   "
            f"(<={T}: {fer_le:.4f}  >{T}: {fer_gt:.4f})"
        )
    print("=" * 64)


# MAIN


def main() -> None:
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
    next_message = make_message_source(cfg)

    if verbose:
        print(f"Decoders: {list(decoders.keys())}")
        print(f"Messages: {cfg.get('message_source', 'random')}")
        print("Pass 1/3: metrics")
    metrics = run_metrics_pass(
        decoders,
        channel_fn,
        next_message,
        num_samples=cfg["benchmark"]["num_samples"],
        nsym=cfg["code"]["nsym"],
        verbose=verbose,
    )

    if verbose:
        print("Pass 2/3: decoding timing")
    timing = run_timing_pass(
        decoders,
        channel_fn,
        next_message,
        num_samples=cfg["timing"]["num_samples"],
        warmup=cfg["timing"]["warmup"],
        nsym=cfg["code"]["nsym"],
        verbose=verbose,
    )

    if verbose:
        print("Pass 3/3: encoding timing")
    encoding = run_encoding_pass(
        next_message,
        num_samples=cfg["timing"]["num_samples"],
        warmup=cfg["timing"]["warmup"],
        verbose=verbose,
    )

    csv_path, yaml_path = save_results(
        metrics,
        timing,
        encoding,
        cfg,
        device,
        cfg["output"]["dir"],
    )

    print_summary(metrics)

    print(f"Results: {csv_path}")
    print(f"Context: {yaml_path}")


if __name__ == "__main__":
    main()
