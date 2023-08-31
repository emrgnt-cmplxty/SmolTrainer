import argparse
import logging
import os


def get_root_py_fpath() -> str:
    """Get the path to the root of the python code directory."""

    return os.path.dirname(os.path.realpath(__file__))


def get_root_fpath() -> str:
    """Get the path to the root of the baby_moe directory."""

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
        "--out-dir", default="out", type=str, help="Output directory"
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
        "--eval-only",
        default=False,
        action="store_true",
        help="Only evaluate the model",
    )
    parser.add_argument(
        "--always-save-checkpoint",
        default=True,
        action="store_true",
        help="Always save checkpoint after each evaluation",
    )
    parser.add_argument(
        "--init-from",
        default="scratch",
        type=str,
        choices=["scratch", "resume", "gpt2*"],
        help="Initialization mode: scratch, resume or gpt2*",
    )
    # Wandb logging
    parser.add_argument(
        "--wandb-log",
        default=False,
        action="store_true",
        help="Enable W&B logging",
    )
    parser.add_argument(
        "--wandb-project", default="owt", type=str, help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name", default="gpt2", type=str, help="W&B run name"
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
        "--n-layer",
        default=12,
        type=int,
        help="Number of layers in the GPT model",
    )
    parser.add_argument(
        "--n-head",
        default=12,
        type=int,
        help="Number of heads in the GPT model",
    )
    parser.add_argument(
        "--n-embd",
        default=768,
        type=int,
        help="Embedding dimension in the GPT model",
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Dropout rate"
    )
    parser.add_argument(
        "--bias",
        default=False,
        action="store_true",
        help="Use bias inside LayerNorm and Linear layers",
    )

    # MoE Settings
    parser.add_argument(
        "--n-experts",
        default=12,
        type=int,
        help="Number of experts for the MoE model.",
    )
    parser.add_argument(
        "--top-k-experts",
        default=6,
        type=int,
        help="Top k to consider for gating.",
    )

    # Optimizer arguments
    parser.add_argument(
        "--learning-rate", default=6e-4, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--max-iters",
        default=600000,
        type=int,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--mode", default="gpt", type=str, help="Mode to run the model"
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
