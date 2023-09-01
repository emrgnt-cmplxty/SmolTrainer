"""Data loading utilities for training and validation."""

import argparse
import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from utils import get_root_py_fpath


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


def get_batch(
    args: argparse.Namespace, data: np.memmap
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if args.device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(
            args.device, non_blocking=True
        ), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y
