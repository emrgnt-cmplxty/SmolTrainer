"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import logging
import math
import os
import pickle
import sys
import time

# from contextlib import AbstractContextManager
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

sys.path.append("/Users/ocolegrove/babyMoE/baby_moe/nano_gpt")

from moe import MoEGPT
from nano_gpt.model import GPT, GPTConfig
from utils import get_configured_logger, get_root_py_fpath, parse_args


def load_config_and_overwrite_args(
    logger: logging.Logger, args: argparse.Namespace
) -> None:
    local_vars_before = locals().copy()
    config_load = open(args.config_file).read()
    logger.info(f"Reading config from {args.config_file}:\n{config_load}")

    local_namespace: dict = {}
    exec(config_load, globals(), local_namespace)

    new_vars = set(local_namespace.keys()) - set(local_vars_before.keys())

    for var in new_vars:
        setattr(args, var, local_namespace[var])


def load_data(
    logger: logging.Logger,
    args: argparse.Namespace,
) -> Tuple[np.memmap, np.memmap, Optional[int]]:
    """Load training and validation data."""
    data_dir = os.path.join(
        get_root_py_fpath(), "nano_gpt", "data", args.dataset
    )
    train_data = np.memmap(
        os.path.join(get_root_py_fpath(), data_dir, "train.bin"),
        dtype=np.uint16,
        mode="r",
    )
    val_data = np.memmap(
        os.path.join(get_root_py_fpath(), data_dir, "val.bin"),
        dtype=np.uint16,
        mode="r",
    )
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        logger.info(
            f"Found vocab_size = {meta_vocab_size} (inside {meta_path})"
        )

    return (train_data, val_data, meta_vocab_size)


def get_batch(data: np.memmap) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from either the training or validation set."""
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + args.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + args.block_size]).astype(np.int64)
            )
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(
            args.device, non_blocking=True
        ), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y


def setup_run_args(logger: logging.Logger, args: argparse.Namespace) -> None:
    """Setup the arguments for the run."""
    # sourcery skip: extract-method

    # various inits, derived attributes, I/O setup
    args.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if args.ddp:
        init_process_group(backend=args.backend)
        args.ddp_rank = int(os.environ["RANK"])
        args.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        args.ddp_world_size = int(os.environ["WORLD_SIZE"])
        args.device = f"cuda:{args.ddp_local_rank}"
        torch.cuda.set_device(args.device)
        args.master_process = (
            args.ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        args.seed_offset = args.ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.gradient_accumulation_steps % args.ddp_world_size == 0
        args.gradient_accumulation_steps //= args.ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        args.master_process = True
        args.seed_offset = 0
        args.ddp_world_size = 1
    args.tokens_per_iter = (
        args.gradient_accumulation_steps
        * args.ddp_world_size
        * args.batch_size
        * args.block_size
    )
    logger.info(f"tokens per iteration will be: {args.tokens_per_iter:,}")


def train_model(
    logger: logging.Logger,
    args: argparse.Namespace,
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    train_data: np.memmap,
    # TODO - Why does this fail type checks? ctx: AbstractContextManager,
    ctx: Any,
    ddp: bool = False,
    raw_model: Optional[Module] = None,
) -> Tuple[int, float]:
    X, Y = get_batch(train_data)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    iter_num = 0
    best_val_loss = 1e9

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and args.master_process:
            losses = estimate_loss()
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
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    logger.info(f"saving checkpoint to {args.out_dir}")
                    torch.save(
                        checkpoint, os.path.join(args.out_dir, "ckpt.pt")
                    )
        if iter_num == 0 and args.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(train_data)  # fetch the very first batch
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
                if not raw_model:
                    raise ValueError("raw_model is None")
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


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    logger = get_configured_logger(__name__, args.log_level)

    logger.info(f"Running with passed in args:\n{args}")

    if args.config_file:
        load_config_and_overwrite_args(logger, args)

    setup_run_args(logger, args)

    if args.master_process:
        os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337 + args.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    logger.info(f"Running over dataset = {args.dataset}")
    train_data, val_data, meta_vocab_size = load_data(logger, args)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=None,
        dropout=args.dropout,
    )  # start with model_args from command line
    if args.init_from == "scratch":
        # init a new model from scratch
        logger.info("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            logger.info(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        logger.info(f"Model is initializing with args:\n{model_args}")
        gptconf = GPTConfig(**model_args)
        model = (
            GPT(gptconf)
            if args.mode == "gpt"
            else MoEGPT(gptconf, args.n_experts)  # , args.top_k_experts)
        )
    elif args.init_from == "resume":
        logger.info(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
        ]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif args.init_from.startswith("gpt2"):
        logger.info(
            f"Initializing from OpenAI GPT-2 weights: {args.init_from}"
        )
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=args.dropout)
        model = GPT.from_pretrained(args.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
        ]:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        model_args[
            "block_size"
        ] = args.block_size  # so that the checkpoint will have the right value
    model.to(args.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type,
    )
    if args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if args.compile:
        logger.info("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if args.ddp:
        model = DDP(model, device_ids=[args.ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(
                    train_data if split == "train" else val_data
                )  # fetch the very first batch

                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
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
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff ranges 0..1
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # logging
    if args.wandb_log and args.master_process:
        import wandb

        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=config
        )

    raw_model = (
        model.module if args.ddp else model
    )  # unwrap DDP container if needed

    iter_num, best_val_loss = train_model(
        logger,
        args,
        model,
        optimizer,
        scaler,
        train_data,
        ctx,
        args.ddp,
        raw_model,
    )

    if args.ddp:
        destroy_process_group()
