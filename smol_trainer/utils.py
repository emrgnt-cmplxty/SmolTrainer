"""Utilities for the smol_trainer package."""
import argparse
import logging
import os
from dataclasses import fields, is_dataclass


def get_root_py_fpath() -> str:
    """Get the path to the root of the python code directory."""

    return os.path.dirname(os.path.realpath(__file__))


def get_root_fpath() -> str:
    """Get the path to the root of the smol_trainer directory."""

    return os.path.join(get_root_py_fpath(), "..")


def get_configured_logger(name: str, log_level: str) -> logging.Logger:
    """Get a configured logger."""

    log_level = getattr(logging, log_level.upper(), "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


# TODO - Break this into multiple functions and/or configs
# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for the GPT model"
    )

    # I/O arguments
    parser.add_argument(
        "--config-file", default="", type=str, help="Configuration file"
    )
    parser.add_argument(
        "--out-dir", default="results", type=str, help="Output directory"
    )
    parser.add_argument(
        "--eval-interval", default=2000, type=int, help="Evaluation interval"
    )
    parser.add_argument(
        "--log-interval", default=1, type=int, help="Log interval"
    )
    parser.add_argument(
        "--log-level", default="INFO", type=str, help="Log level"
    )
    parser.add_argument(
        "--eval-iters", default=200, type=int, help="Evaluation iterations"
    )
    parser.add_argument(
        "--always-save-checkpoint",
        default=True,
        action="store_true",
        help="Always save checkpoint after each evaluation",
    )

    # Run parameters
    parser.add_argument(
        "--init-from",
        default="scratch",
        type=str,
        choices=["scratch", "resume"],
        help="Initialization mode: scratch, resume or gpt2*",
    )
    parser.add_argument(
        "--ckpt-path-override",
        default=None,
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--iter-num",
        default=0,
        type=int,
        help="Iteration number, used when resuming training",
    )
    parser.add_argument(
        "--run-name", default="run_0", type=str, help="Specify the run name."
    )

    # WandB logging
    parser.add_argument(
        "--wandb-log",
        default=False,
        action="store_true",
        help="Enable W&B logging",
    )

    # Data arguments
    parser.add_argument(
        "--dataset", default="openwebtext", type=str, help="Dataset name"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=5 * 8,
        type=int,
        help="Steps for gradient accumulation",
    )
    parser.add_argument(
        "--batch-size", default=12, type=int, help="Batch size"
    )
    parser.add_argument(
        "--block-size", default=1024, type=int, help="Block size"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        default="pythia-70m",
        type=str,
        help="The name of the model to run with.",
    )

    # Optimizer arguments
    parser.add_argument(
        "--initial-lr",
        default=6e-4,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-iters",
        default=600000,
        type=int,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-1,
        type=float,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Beta1 for optimizer"
    )
    parser.add_argument(
        "--beta2", default=0.95, type=float, help="Beta2 for optimizer"
    )
    parser.add_argument(
        "--grad-clip", default=1.0, type=float, help="Gradient clipping value"
    )
    parser.add_argument(
        "--do-flash-v2",
        default=False,
        action="store_true",
        help="Use flash v2 calculation (Requires A100 or better).",
    )

    # Learning rate decay settings
    parser.add_argument(
        "--decay-lr",
        default=True,
        action="store_true",
        help="Enable learning rate decay",
    )
    parser.add_argument(
        "--warmup-iters",
        default=2000,
        type=int,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--lr-decay-iters",
        default=600000,
        type=int,
        help="Learning rate decay iterations",
    )
    parser.add_argument(
        "--min-lr", default=6e-5, type=float, help="Minimum learning rate"
    )
    parser.add_argument(
        "--max-checkpoints",
        default=5,
        type=int,
        help="Maximum checkpoints to hold.",
    )

    # DDP settings
    parser.add_argument(
        "--backend",
        default="nccl",
        type=str,
        choices=["nccl", "gloo"],
        help="Backend for DDP",
    )

    # System arguments
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Other arguments
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to use for training"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        help="Data type for training",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=True,
        help="Use PyTorch 2.0 to compile the model to be faster.",
    )

    return parser.parse_args()


def custom_asdict(obj) -> dict:
    import _thread

    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            # Check if value is a thread lock or any other non-pickleable type
            # Modify this check as per your needs
            if isinstance(value, _thread.RLock):
                result[f.name] = "Skipped due to non-pickleable type"
            else:
                result[f.name] = custom_asdict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [custom_asdict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: custom_asdict(v) for k, v in obj.items()}
    else:
        return obj
