"""Contains the logic for initializing a model."""

import argparse
import logging
import os
from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from smol_trainer.config import Mode
from smol_trainer.model import MoEGPT
from smol_trainer.nano_gpt.model import GPT, GPTConfig


def configure_optimizers(model: Module, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # --- DISABLING use_fused due to issue in nightly ---
    # Create AdamW optimizer and use the fused version if it is available
    # fused_available = (
    #     "fused" in inspect.signature(torch.optim.AdamW).parameters
    # )
    # use_fused = fused_available and device_type == "cuda"
    # extra_args = dict(fused=True) if use_fused else dict()
    # optimizer = torch.optim.AdamW(
    #     optim_groups, fused=False, lr=learning_rate, betas=betas, **extra_args
    # )
    # print(f"using fused AdamW: {use_fused}")
    # optimizer = torch.optim.AdamW(
    #     optim_groups, fused=False, lr=learning_rate, betas=betas, **extra_args
    # )

    return torch.optim.AdamW(
        optim_groups,
        fused=False,
        lr=learning_rate,
        betas=betas,
    )


def initialize_optimizer(
    args: argparse.Namespace, model: Module, checkpoint: Any = None
):
    """Initialize optimizer and load its state if resuming from a checkpoint."""

    optimizer = configure_optimizers(
        model,
        args.weight_decay,
        args.initial_lr,
        (args.beta1, args.beta2),
    )
    if checkpoint and args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def initialize_model_from_scratch(
    args: argparse.Namespace,
    meta_vocab_size: Optional[int],
    logger: logging.Logger,
) -> Module:
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
    logger.info(f"Running in architecture mode = {args.mode}")
    return (
        GPT(gptconf)
        if args.mode == Mode.GPT.value
        else MoEGPT(gptconf, args.n_experts, args.top_k_experts)
    )


def initialize_model_from_checkpoint(
    args: argparse.Namespace, logger: logging.Logger
) -> Tuple[Module, Any]:  # TODO - Find a correct type for the checkpoint.
    """Resume training from a checkpoint."""

    logger.info(f"Resuming training from {args.out_dir}")

    # TODO - Fix this to dynamically write the correct path.
    ckpt_path = os.path.join(args.out_dir, args.model_path)
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")

    checkpoint = torch.load(ckpt_path, map_location=args.device)

    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
    ]:
        args.model_args[k] = checkpoint[k]

    args.model_args["vocab_size"] = checkpoint.get("vocab_size", 50304)

    gptconf = GPTConfig(**args.model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # Update args to reflect latest numbers from checkpoint
    args.iter_num = checkpoint["iter_num"]
    args.best_val_loss = checkpoint["best_val_loss"]

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


def crop_and_move_model(args: argparse.Namespace, model: Module) -> Module:
    """Handle model cropping and device transfer."""

    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        args.model_args[
            "block_size"
        ] = args.block_size  # so that the checkpoint will have the right value

    model.to(args.device)
    return model
