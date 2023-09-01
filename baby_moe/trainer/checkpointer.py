"""A module for checkpointing and saving models during training."""

import argparse
import datetime
import glob
import os

import torch
from torch.nn import Module
from torch.optim import Optimizer

from baby_moe.utils import get_configured_logger


def get_project_identifier(args: argparse.Namespace) -> str:
    """Returns the name of the checkpoint file"""
    mode = "gpt" if args.mode == "gpt" else "moe"
    return f"mode_{mode}__n_layer_{args.n_layer}__n_head_{args.n_head}__n_embd_{args.n_embd}__n_experts_{args.n_experts}__top_k_experts_{args.top_k_experts}"


def get_checkpoint_prefix(args: argparse.Namespace) -> str:
    """Returns the name of the checkpoint file"""
    return f"{args.run_name}_checkpoint__{get_project_identifier(args)}"


def manage_checkpoints(args: argparse.Namespace) -> None:
    """Manage the checkpoints: save, delete old ones"""
    # List all checkpoints
    prefix = get_checkpoint_prefix(args)
    file_name = f"{prefix}__iter_num_*.pt"
    checkpoints = sorted(
        glob.glob(os.path.join(args.out_dir, prefix, file_name)),
        key=lambda x: int(x.split("__iter_num_")[-1].split(".pt")[0]),
    )

    # Remove older checkpoints
    for ckpt in checkpoints[: -args.max_checkpoints]:
        os.remove(ckpt)


def save_checkpoint(
    args: argparse.Namespace,
    model: Module,
    optimizer: Optimizer,
    iter_num: int,
    best_val_loss: float,
    train_loss: float,
    running_mfu: float,
) -> None:
    """Saves the checkpoint to the designated output location"""
    logger = get_configured_logger(__name__, args.log_level)

    checkpoint = {
        # Model & Optimizer state
        "mode": args.mode,
        "model_args": args.model_args,
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
        "train_loss": train_loss,
        "running_mfu": running_mfu,
        "learning_rate": args.initial_lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        # Other details
        "config": vars(args),
        "timestamp": str(datetime.datetime.now()),
        "pytorch_version": torch.__version__,
    }

    prefix = get_checkpoint_prefix(args)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    temp_checkpoint_path = os.path.join(
        args.checkpoint_dir, f"{prefix}__iter_num_{iter_num}.temp"
    )
    checkpoint_path = temp_checkpoint_path.replace(".temp", ".pt")

    # Save to a temporary file to avoid data corruption
    torch.save(checkpoint, temp_checkpoint_path)
    os.rename(temp_checkpoint_path, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
