"""A module for checkpointing and saving models during training."""
import datetime
import glob
import os

import torch
from torch.nn import Module
from torch.optim import Optimizer


def get_project_identifier(output_config: dict) -> str:
    """Returns the name of the checkpoint file"""

    common_prefix = f"n_layer_{output_config['n_layer']}__n_head_{output_config['n_head']}__n_embd_{output_config['n_embd']}"
    if output_config["mode"].value == "gpt":
        return f"mode_gpt__{common_prefix}"
    elif output_config["mode"].value == "moe":
        return f"mode_moe__{common_prefix}__n_experts_{output_config['n_experts']}__top_k_experts_{output_config['top_k_experts']}"
    else:
        raise NotImplementedError("This mode is not supported yet.")


def get_checkpoint_prefix(output_config: dict) -> str:
    """Returns the name of the checkpoint file"""
    return f"{output_config['run_name']}_checkpoint__{get_project_identifier(output_config)}"


def manage_checkpoints(output_config: dict) -> None:
    """Manage the checkpoints: save, delete old ones"""

    # List all checkpoints
    prefix = get_checkpoint_prefix(output_config)
    file_name = f"{prefix}__iter_num_*.pt"
    checkpoints = sorted(
        glob.glob(os.path.join(output_config["out_dir"], prefix, file_name)),
        key=lambda x: int(x.split("__iter_num_")[-1].split(".pt")[0]),
    )

    # Remove older checkpoints
    for ckpt in checkpoints[: -output_config["max_checkpoints"]]:
        os.remove(ckpt)


def save_checkpoint(
    output_config: dict,
    model: Module,
    optimizer: Optimizer,
) -> None:
    """Saves the checkpoint to the designated output location"""

    checkpoint = {
        **output_config,
        "model": model.state_dict(),
        "num_params": model.get_num_params(),
        "optimizer": optimizer.state_dict(),
        "timestamp": str(datetime.datetime.now()),
        "pytorch_version": torch.__version__,
    }

    prefix = get_checkpoint_prefix(output_config)
    os.makedirs(output_config["checkpoint_dir"], exist_ok=True)

    temp_checkpoint_path = os.path.join(
        output_config["checkpoint_dir"],
        f"{prefix}__iter_num_{output_config['iter_num']}.temp",
    )
    checkpoint_path = temp_checkpoint_path.replace(".temp", ".pt")

    # Save to a temporary file to avoid data corruption
    torch.save(checkpoint, temp_checkpoint_path)
    os.rename(temp_checkpoint_path, checkpoint_path)
