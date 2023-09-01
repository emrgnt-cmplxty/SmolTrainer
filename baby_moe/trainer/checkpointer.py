"""A module for checkpointing and saving models during training."""
import argparse
import datetime
import glob
import os

import torch
from torch.nn import Module
from torch.optim import Optimizer

from baby_moe.utils import get_configured_logger


def _get_checkpoint_name(args: argparse.Namespace, iter_num: str) -> str:
    """Returns the name of the checkpoint file"""
    return f"checkpoint__n_layer_{args.n_layer}__n_head_{args.n_head}__n_embd_{args.n_embd}__n_experts_{args.n_experts}__top_k_experts_{args.top_k_experts}__iter_{iter_num}.pt"


def manage_checkpoints(args: argparse.Namespace) -> None:
    """Manage the checkpoints: save, delete old ones"""
    # List all checkpoints
    file_name = _get_checkpoint_name(args, "*")
    print("file_name = ", file_name)
    checkpoints = sorted(glob.glob(os.path.join(args.out_dir, file_name)))

    # Remove older checkpoints
    for ckpt in checkpoints[: -args.max_checkpoints]:
        os.remove(ckpt)


def save_checkpoint(
    args: argparse.Namespace,
    model: Module,
    optimizer: Optimizer,
    iter_num: int,
    best_val_loss: float,
    running_mfu: float,
) -> None:
    """Saves the checkpoint to the designated output location"""
    logger = get_configured_logger(__name__, args.log_level)

    checkpoint = {
        # Model & Optimizer state
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # Model hyperparameters
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "bias": args.bias,
        "n_experts": args.n_experts,
        "top_k_experts": args.top_k_experts,
        # Training state & hyperparameters
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "running_mfu": running_mfu,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        # Other details
        "config": vars(args),
        "timestamp": str(datetime.datetime.now()),
        "pytorch_version": torch.__version__,
    }

    checkpoint_path = os.path.join(
        args.out_dir, _get_checkpoint_name(args, str(iter_num))
    )
    temp_checkpoint_path = f"{checkpoint_path}.temp"

    # Save to a temporary file to avoid data corruption
    torch.save(checkpoint, temp_checkpoint_path)
    os.rename(temp_checkpoint_path, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
