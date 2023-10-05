"""Base training loop for SmolTrainer."""

import math
import threading
import time

# from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer

from smol_trainer.config.train import LearningConfig, TrainConfig
from smol_trainer.trainer.checkpointer import (
    manage_checkpoints,
    save_checkpoint,
)
from smol_trainer.trainer.data_loader import get_batch
from smol_trainer.utils import custom_asdict

# ========================== Learning Rate Logic ==========================


def linear_warmup_lr(lr_config: LearningConfig, it):
    """Calculate learning rate during the warmup phase."""
    return lr_config.initial_lr * it / lr_config.warmup_iters


def cosine_decay_lr(lr_config: LearningConfig, it):
    """Calculate learning rate using cosine decay."""
    decay_ratio = (it - lr_config.warmup_iters) / (
        lr_config.lr_decay_iters - lr_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_config.min_lr + coeff * (lr_config.initial_lr - lr_config.min_lr)


def get_lr(lr_config: LearningConfig, it: int):
    """Get the learning rate for the given iteration."""
    if it < lr_config.warmup_iters:
        return linear_warmup_lr(lr_config, it)
    elif it > lr_config.lr_decay_iters:
        return lr_config.min_lr
    else:
        return cosine_decay_lr(lr_config, it)


# ========================== Logging Logic ==========================


def log_metrics(
    config: TrainConfig,
    lossf: float,
    dt: float,
):
    """Log metrics during training."""
    config.logger.info(
        f"iter {config.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {config.running_mfu*100:.2f}%, total_tokens_processed {config.total_tokens_processed}"
    )


# ========================== Evaluation Logic ==========================


@torch.no_grad()
def estimate_loss(
    config: TrainConfig,
    amp_context: Any,
    model: Module,
    train_data: np.memmap,
    val_data: np.memmap,
) -> dict:
    """Estimate the loss on the training and validation sets."""

    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(
                config, train_data if split == "train" else val_data
            )  # fetch the very first batch

            with amp_context:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def perform_evaluation(
    config: TrainConfig,
    optimizer: Optimizer,
    model: Module,
    raw_model: Module,
    train_data: np.memmap,
    val_data: np.memmap,
    amp_context: Any,
) -> None:
    """Evaluate the model and save checkpoints if necessary."""
    losses = estimate_loss(config, amp_context, model, train_data, val_data)

    config.total_time = time.time() - config.initial_time
    config.logger.info(
        f"Eval @ iter = {config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, total time {config.total_time:.2f}"
    )

    # Logging with WandB
    if config.wandb_log:
        wandb.log(
            {
                "iter": config.iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": config.lr_config.lr,
                "mfu": config.running_mfu * 100,  # convert to percentage
                "tokens_processed": config.total_tokens_processed,
            }
        )

    # Save checkpoint and manage old checkpoints
    if losses["val"] < config.best_val_loss or config.always_save_checkpoint:
        config.best_val_loss, training_loss = losses["val"], losses["train"]
        config.training_loss = training_loss
        if config.iter_num > 0:
            output_config = custom_asdict(config)
            output_config.pop("logger")
            save_checkpoint(
                output_config,
                raw_model,
                optimizer,
            )
            thread = threading.Thread(
                target=manage_checkpoints, args=(output_config,)
            )
            thread.start()


def train_model(
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    train_data: np.memmap,
    val_data: np.memmap,
    # TODO - Track down the type for amp_context
    # Why does amp_context: AbstractContextManager fail?
    config: TrainConfig,
    amp_context: Any,
    raw_model: Module,
) -> None:
    """Train the model."""
    # TODO - Break this function up into smaller functions

    # initial setup
    lr_config = config.lr_config

    # Fetch the very first batch
    X, Y = get_batch(config, train_data)
    t0 = time.time()

    for local_iter_num, iter_num in enumerate(
        range(config.iter_num, config.max_iters + 1)
    ):
        config.iter_num = iter_num

        # determine and set the learning rate for this iteration
        lr = (
            get_lr(lr_config, iter_num)
            if lr_config.decay_lr
            else lr_config.initial_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        lr_config.lr = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config.eval_interval == 0 and config.master_process:
            perform_evaluation(
                config,
                optimizer,
                model,
                raw_model,
                train_data,
                val_data,
                amp_context,
            )

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(lr_config.gradient_accumulation_steps):
            config.total_tokens_processed += X.numel()

            if config.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == lr_config.gradient_accumulation_steps - 1
                )
            with amp_context:
                _, loss = model(X, Y)
                loss = (
                    loss / lr_config.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(config, train_data)  # fetch the very first batch
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if lr_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), lr_config.grad_clip
            )
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if iter_num % config.log_interval == 0 and config.master_process:
            # Calculate elapsed time
            dt = time.time() - t0
            t0 = time.time()

            # Get loss and update metrics
            lossf = loss.item() * lr_config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    config.batch_size * lr_config.gradient_accumulation_steps,
                    dt,
                )
                config.running_mfu = (
                    mfu
                    if config.running_mfu == -1.0
                    else 0.9 * config.running_mfu + 0.1 * mfu
                )

            log_metrics(config, lossf, dt)
    return
