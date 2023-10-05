"""Configurations for training."""
import logging
import time
from dataclasses import dataclass
from enum import Enum


@dataclass
class LearningConfig:
    """Learning rate and optimizer configuration."""

    # Learning rate arguments
    initial_lr: float
    decay_lr: float
    lr: float
    min_lr: float

    # Optimizer arguments
    grad_clip: float
    weight_decay: float
    beta1: float
    beta2: float
    do_flash_v2: bool

    # Iteration variables
    warmup_iters: int
    lr_decay_iters: int
    gradient_accumulation_steps: int


@dataclass
class TrainConfig:
    """Training configuration."""

    # Logging support
    logger: logging.Logger
    lr_config: LearningConfig
    master_process: bool
    log_interval: int
    wandb_log: bool

    # Arhitecture
    model_name: str

    # Training Params
    eval_interval: int
    batch_size: int
    block_size: int
    max_iters: int
    eval_iters: int
    max_checkpoints: int

    # Run variables
    out_dir: str
    checkpoint_dir: str
    run_name: str
    ddp: bool
    device: str
    device_type: str
    always_save_checkpoint: bool

    # Run information
    iter_num: int = 0
    total_tokens_processed: int = 0
    best_val_loss: float = 1e9
    running_mfu: float = -1.0
    initial_time: float = time.time()
    total_time: float = 0.0
    training_loss: float = -1
