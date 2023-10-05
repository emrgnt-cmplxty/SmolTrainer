"""Contains the logic for initializing a model."""

import argparse
import logging
import os
from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from smol_trainer.model import GPT
from smol_trainer.trainer.checkpointer import get_checkpoint_prefix


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
    logger: logging.Logger,
) -> Module:
    """Initialize a new model from scratch."""

    logger.info(f"Running model {args.model_name}")

    if args.iter_num != 0:
        raise ValueError("iter_num must be 0 to initialize from scratch")

    return GPT.from_name(args.model_name)


def initialize_model_from_checkpoint(
    args: argparse.Namespace, logger: logging.Logger
) -> Tuple[Module, Any]:  # TODO - Find a correct type for the checkpoint.
    """Resume training from a checkpoint."""

    logger.info(f"Resuming training from {args.out_dir}")

    if args.ckpt_path_override:
        checkpoint = torch.load(
            args.ckpt_path_override, map_location=args.device
        )
    else:
        try:
            checkpoint_prefix = get_checkpoint_prefix(
                {"run_name": args.run_name, "model_name": args.model_name}
            )
            model_path = os.path.join(
                args.out_dir,
                checkpoint_prefix,
                f"{checkpoint_prefix}__iter_num_{args.iter_num}.pt",
            )
            if not os.path.exists(model_path):
                raise ValueError(
                    f"Checkpoint path {model_path} does not exist"
                )

            checkpoint = torch.load(model_path, map_location=args.device)
            args.iter_num = checkpoint["iter_num"]
            assert (
                args.iter_num == checkpoint["iter_num"]
            ), "Iteration numbers do not match!"

        except Exception as e:
            logger.error(
                "Encountered an  error {e} while attempting to load model checkpoint"
            )
            raise e

    model = GPT.from_name(args.model_name, block_size=args.block_size)
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
