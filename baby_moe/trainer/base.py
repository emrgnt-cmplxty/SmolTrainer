"""Base training loop for Baby MoE."""
import argparse
import logging
import math
import threading
import time

# from contextlib import AbstractContextManager
from typing import Any, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer

from baby_moe.trainer.checkpointer import manage_checkpoints, save_checkpoint
from baby_moe.trainer.data_loader import get_batch


# learning rate decay scheduler (cosine with warmup)
def get_lr(args: argparse.Namespace, it: int):
    """Get the learning rate for the given iteration."""

    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (
        args.lr_decay_iters - args.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(
    args: argparse.Namespace,
    ctx: Any,
    model: Module,
    train_data: np.memmap,
    val_data: np.memmap,
):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(
                args, train_data if split == "train" else val_data
            )  # fetch the very first batch

            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(
    logger: logging.Logger,
    args: argparse.Namespace,
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    train_data: np.memmap,
    val_data: np.memmap,
    # TODO - Track down the types for these
    # Why does ctx: AbstractContextManager fail?
    ctx: Any,
    raw_model: Module,
) -> Tuple[int, float]:
    """Train the model."""
    # TODO - Break this function up into smaller functions

    # logging
    if args.wandb_log and args.master_process:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    X, Y = get_batch(args, train_data)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    iter_num = 0
    best_val_loss = 1e9

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(args, iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and args.master_process:
            losses = estimate_loss(args, ctx, model, train_data, val_data)
            logger.info(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if args.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }
                )
            if losses["val"] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    save_checkpoint(
                        args,
                        raw_model,
                        optimizer,
                        iter_num,
                        running_mfu,
                        best_val_loss,
                    )

                    # Manage old checkpoints asynchronously
                    thread = threading.Thread(
                        target=manage_checkpoints, args=(args,)
                    )
                    thread.start()

                    if iter_num == 0 and args.eval_only:
                        break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(args.gradient_accumulation_steps):
            if args.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(X, Y)
                loss = (
                    loss / args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(args, train_data)  # fetch the very first batch
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and args.master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * args.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    args.batch_size * args.gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu
                    if running_mfu == -1.0
                    else 0.9 * running_mfu + 0.1 * mfu
                )
            logger.info(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, 100*mfu {running_mfu*100*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > args.max_iters:
            break
    return iter_num, best_val_loss
