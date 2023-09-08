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

# from contextlib import AbstractContextManager
from contextlib import nullcontext
from typing import Any, Union

import torch
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from smol_trainer.config import LearningConfig, Mode, TrainConfig
from smol_trainer.trainer import (
    crop_and_move_model,
    get_checkpoint_prefix,
    get_project_identifier,
    initialize_model_from_checkpoint,
    initialize_model_from_gpt2,
    initialize_model_from_scratch,
    initialize_optimizer,
    load_data,
    train_model,
)
from smol_trainer.utils import get_configured_logger, parse_args


def load_config_and_overwrite_args(
    logger: logging.Logger, args: argparse.Namespace
) -> None:
    """Load config from file and overwrite args."""
    if args.config_file:
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
    args.device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast

    # Initialize here so we can override if init_from='resume' (i.e. from a checkpoint)
    args.iter_num = 0
    args.best_val_loss = 1e9

    prefix = get_checkpoint_prefix(vars(args))
    args.tensorboard_path = os.path.join(
        args.out_dir, f"{prefix}__tensorboard"
    )
    args.checkpoint_dir = os.path.join(
        args.out_dir,
        prefix,
    )

    logger.info(f"tokens per iteration will be: {args.tokens_per_iter:,}")


def setup_amp_context(args: argparse.Namespace) -> Any:
    """Sets up the autocast context."""
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    return (
        nullcontext()
        if args.device_type == "cpu"
        else torch.amp.autocast(device_type=args.device_type, dtype=ptdtype)
    )


def setup_ddp(args: argparse.Namespace, model: Module) -> Union[Module, Any]:
    # TODO - What is the appropriate type for DDP?
    """Sets up the DDP model."""
    if args.ddp:
        model = DDP(model, device_ids=[args.ddp_local_rank])
    return model


def setup_training_environment(args: argparse.Namespace) -> Any:
    """Setup the training environment"""

    load_config_and_overwrite_args(logger, args)
    setup_run_args(logger, args)
    torch.manual_seed(1337 + args.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if (
        (
            os.path.exists(args.checkpoint_dir)
            or os.path.exists(args.tensorboard_path)
        )
        and args.init_from == "scratch"
        and args.master_process
    ):
        raise ValueError(
            f"Checkpoint directory {args.checkpoint_dir} or {args.tensorboard_path} already exists, please move before re-running."
        )
    return setup_amp_context(args)


def initialize_run_performance_logging(
    args: argparse.Namespace,
) -> SummaryWriter:
    """Initialize logging with WandB and TensorBoard."""
    if args.wandb_log and args.master_process:
        config_dict = vars(args)
        wandb.init(
            project=get_project_identifier(config_dict),
            name=args.run_name,
            config=config_dict,
        )
    return SummaryWriter(log_dir=args.tensorboard_path)


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    logger = get_configured_logger(__name__, args.log_level)
    logger.info(f"Running with passed in args:\n{args}")

    amp_context = setup_training_environment(args)

    # Setting model arguments  init
    args.model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=None,
        dropout=args.dropout,
        do_flash_v2=args.do_flash_v2,
    )  # start with model_args from command line

    logger.info(f"Running over dataset = {args.dataset}")
    train_data, val_data, meta_vocab_size = load_data(logger, args.dataset)

    checkpoint = None
    if args.init_from == "scratch":
        model: Module = initialize_model_from_scratch(
            args, meta_vocab_size, logger
        )
    elif args.init_from == "resume":
        model, checkpoint = initialize_model_from_checkpoint(args, logger)
    elif args.init_from.startswith("gpt2"):
        model = initialize_model_from_gpt2(args, logger)

    model = crop_and_move_model(args, model)

    optimizer = initialize_optimizer(args, model, checkpoint)

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    # We must comment out this code which appears in nanoGPT
    # This is to avoid explosions of gradients when using Torch 2.0.x
    # The authors in nanoGPT claim this is a fault of Torch
    # TODO - Investigate this further
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.dtype == "float16"), growth_interval=0
    )
    # Compile the model
    if args.compile:
        logger.info("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Wrap the model into DDP container
    model = setup_ddp(args, model)

    if args.master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    raw_model = (
        model.module if args.ddp else model
    )  # unwrap DDP container if needed
    logger.info(f"Running with the following model:\m{model}")

    lr_config = LearningConfig(
        # Learning rate settings
        lr=args.initial_lr,
        initial_lr=args.initial_lr,
        decay_lr=args.decay_lr,
        min_lr=args.min_lr,
        # Optimizer settings
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        do_flash_v2=args.do_flash_v2,
        # Iteration variables
        lr_decay_iters=args.lr_decay_iters,
        warmup_iters=args.warmup_iters,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.mode not in [member.value for member in Mode.__members__.values()]:
        raise ValueError(
            f"Invalid mode specified {args.mode} {Mode.__members__}"
        )

    # Initialize the training config
    train_config = TrainConfig(
        # Logging support
        tb_writer=initialize_run_performance_logging(args),
        logger=logger,
        lr_config=lr_config,
        master_process=args.master_process,
        log_interval=args.log_interval,
        wandb_log=args.wandb_log,
        # Architecture
        bias=args.bias,
        mode=args.mode,
        dropout=args.dropout,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_experts=args.n_experts,
        n_embd=args.n_embd,
        top_k_experts=args.top_k_experts,
        # Training params
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters=args.max_iters,
        eval_iters=args.eval_iters,
        max_checkpoints=args.max_checkpoints,
        # Run information
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        ddp=args.ddp,
        device=args.device,
        device_type=args.device_type,
        always_save_checkpoint=args.always_save_checkpoint,
        iter_num=checkpoint["iter_num"] if checkpoint else 0,
    )

    train_model(
        model,
        optimizer,
        scaler,
        train_data,
        val_data,
        train_config,
        amp_context,
        raw_model,
    )

    if args.ddp:
        destroy_process_group()
