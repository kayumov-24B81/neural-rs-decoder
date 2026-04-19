import argparse

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
