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
import os
import sys

# from contextlib import AbstractContextManager
from contextlib import nullcontext

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("/Users/ocolegrove/babyMoE/baby_moe/nano_gpt")

from baby_moe.trainer import (
    crop_and_move_model,
    initialize_model_from_checkpoint,
    initialize_model_from_gpt2,
    initialize_model_from_scratch,
    load_data,
    train_model,
)
from baby_moe.utils import get_configured_logger, parse_args


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
    args.device_type = (
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
        if args.device_type == "cpu"
        else torch.amp.autocast(device_type=args.device_type, dtype=ptdtype)
    )

    logger.info(f"Running over dataset = {args.dataset}")
    train_data, val_data, meta_vocab_size = load_data(logger, args)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # # model init
    args.model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=None,
        dropout=args.dropout,
    )  # start with model_args from command line

    checkpoint = None
    if args.init_from == "scratch":
        model = initialize_model_from_scratch(args, meta_vocab_size, logger)
    elif args.init_from == "resume":
        model, checkpoint = initialize_model_from_checkpoint(args, logger)
    elif args.init_from.startswith("gpt2"):
        model = initialize_model_from_gpt2(args, logger)

    model = crop_and_move_model(args, model)

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # Configure the model optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        args.device_type,
    )

    if checkpoint:
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
        val_data,
        ctx,
        raw_model,
    )

    if args.ddp:
        destroy_process_group()
