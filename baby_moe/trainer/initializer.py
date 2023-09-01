"""Contains the logic for initializing a model."""
import argparse
import logging
import os
from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from baby_moe.model import MoEGPT
from baby_moe.nano_gpt.model import GPT, GPTConfig


def initialize_model_from_scratch(
    args: argparse.Namespace,
    meta_vocab_size: Optional[int],
    logger: logging.Logger,
):
    """Initialize a new model from scratch."""
    logger.info("Initializing a new model from scratch")

    if meta_vocab_size is None:
        logger.info(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    args.model_args["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    logger.info(f"Model is initializing with args:\n{args.model_args}")
    gptconf = GPTConfig(**args.model_args)
    return (
        GPT(gptconf)
        if args.mode == "gpt"
        else MoEGPT(gptconf, args.n_experts, args.top_k_experts)
    )


def initialize_model_from_checkpoint(
    args: argparse.Namespace, logger: logging.Logger
) -> Tuple[GPT, Any]:  # TODO - Find a correct type for the checkpoint.
    """Resume training from a checkpoint."""
    logger.info(f"Resuming training from {args.out_dir}")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    checkpoint_model_args = checkpoint["model_args"]

    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
    ]:
        args.model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**args.model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model, checkpoint


def initialize_model_from_gpt2(
    args: argparse.Namespace, logger: logging.Logger
) -> GPT:
    """Initialize from OpenAI GPT-2 weights."""
    logger.info(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")

    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)

    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
    ]:
        args.model_args[k] = getattr(model.config, k)

    return model


def crop_and_move_model(args: argparse.Namespace, model: GPT) -> Module:
    """Handle model cropping and device transfer."""
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        args.model_args[
            "block_size"
        ] = args.block_size  # so that the checkpoint will have the right value

    model.to(args.device)
    return model
