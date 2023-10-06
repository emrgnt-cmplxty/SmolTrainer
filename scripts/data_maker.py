import concurrent.futures
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import fire as fire
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_config import datasets_config
from smol_trainer.utils import get_root_py_fpath

# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TokenizationManager:
    """Class for managing tokenization of datasets."""

    def __init__(self, encoding, num_proc=None):
        self.tokenizer = AutoTokenizer.from_pretrained(encoding)
        self.num_proc = num_proc or multiprocessing.cpu_count()

    def tokenize_dataset(self, dataset, formatters, raw_tokenize_function):
        tokenize_function = lambda x: raw_tokenize_function(x, formatters, self.tokenizer)
        tokenized_dataset = dataset.map(tokenize_function, num_proc=self.num_proc)
        return [token for sublist in tokenized_dataset["input_ids"] for token in sublist[0]]

    def split_and_save(self, flattened_tokens, val_frac, dataset_name):
        train_ids = flattened_tokens[: int(len(flattened_tokens) * (1 - val_frac))]
        val_ids = flattened_tokens[int(len(flattened_tokens) * (1 - val_frac)) :]

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

        if not os.path.exists(os.path.join(os.path.dirname(__file__), dataset_name)):
            os.mkdir(os.path.join(os.path.dirname(__file__), dataset_name))

        train_ids.tofile(
            os.path.join(
                os.path.dirname(__file__),
                dataset_name,
                "train.bin",
            )
        )
        val_ids.tofile(
            os.path.join(
                os.path.dirname(__file__),
                dataset_name,
                "val.bin",
            )
        )


class DatasetLoader:
    def __init__(self, split, token):
        self.datasets = {}
        self.split = split
        self.token = token

    def add_dataset(self, name, path, mappings, tokenizer):
        self.datasets[name] = {
            "path": path,
            "mappings": mappings,
            "tokenizer": tokenizer,
        }

    def load_all_datasets(self):
        loaded_data = {}
        for name, data in self.datasets.items():
            loaded_data[name] = (
                load_dataset(data["path"], split=self.split, token=self.token),
                data["mappings"],
                data["tokenizer"],
            )
        return loaded_data


def simple_tokenizer(batch, formatters, tokenizer, add_bos_eos=True):
    combined_text = []
    for column, prefix in formatters.items():
        prefix = formatters[column]
        combined_text.append(f"{prefix}:\n{batch[column]}")
    if add_bos_eos:
        combined_text = [tokenizer.bos_token] + combined_text + [tokenizer.eos_token]
    return tokenizer.encode("\n\n".join(combined_text), return_tensors="np")


def read_binary_dataset(dataset_name, split):
    file_path = os.path.join(os.path.dirname(__file__), dataset_name, f"{split}.bin")
    return np.fromfile(file_path, dtype=np.uint16)


def load_weights_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        dataset_weights = yaml.safe_load(file)
    return dataset_weights


def tokenizer(hf_token=None, encoding="mistralai/Mistral-7B-v0.1", val_frac=0.01):
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    logging.info(f"Loading datasets with token: {hf_token}")

    if not hf_token:
        raise ValueError(
            "No HuggingFace token provided and HF_TOKEN environment variable is not set."
        )

    loader = DatasetLoader(split="train", token=hf_token)

    for dataset_name, config in datasets_config.items():
        loader.add_dataset(
            dataset_name,
            config["path"],
            config["mappings"],
            globals()[config["tokenizer"]],
        )

    datasets = loader.load_all_datasets()
    manager = TokenizationManager(encoding)

    for dataset_name, payload in datasets.items():
        logging.info(f"Tokenizing dataset: {dataset_name}")

        dataset, formatters, raw_tokenize_function = payload
        if "train" in dataset:
            dataset = dataset["train"]

        flattened_tokens = manager.tokenize_dataset(dataset, formatters, raw_tokenize_function)
        manager.split_and_save(flattened_tokens, val_frac, dataset_name)


def process_dataset(dataset_name, fraction, all_data, chunk_size):
    # TODO - Move parallelism downstream to here.
    logging.info(f"Loading and remixing dataset: {dataset_name} with fraction: {fraction}")

    local_data = {"train": [], "val": []}

    for dataset_type in all_data.keys():
        data = read_binary_dataset(dataset_name, dataset_type)
        subset_length = int(len(data) * fraction)
        local_data[dataset_type].extend(data[:subset_length])

    for dataset_type, data in local_data.items():
        # Chunking
        chunks = [
            data[i : i + chunk_size]
            for i in tqdm(
                range(0, len(data), chunk_size),
                desc=f"Chunking {dataset_type}",
            )
        ]

        # Shuffling merged chunks
        np.random.shuffle(chunks)

        # Unchunking
        local_data[dataset_type] = [token for chunk in chunks for token in chunk]

    return local_data


def remixer(config_name, chunk_size=2_048, num_proc=None):
    config_path = os.path.join(
        get_root_py_fpath(),
        "data",
        "data_configurations",
        f"{config_name}.yaml" if ".yaml" not in config_name else config_name,
    )

    logging.info(f"Loading dataset weights from: {config_path}")
    dataset_weights = load_weights_from_yaml(config_path)

    # Identify the highest weight
    max_weight = max(dataset_weights.values())

    # Normalize weights
    fractions = {k: v / max_weight for k, v in dataset_weights.items()}

    remixed_name = config_name.replace(".yaml", "")
    logging.info(f"Saving datasets now to remixed dataset name = {remixed_name}")

    all_data = {"train": [], "val": []}

    num_proc = num_proc or multiprocessing.cpu_count()

    # Parallelizing dataset processing
    with ThreadPoolExecutor(max_workers=num_proc) as executor:  # Adjust max_workers as necessary
        futures = []
        for dataset_name, fraction in fractions.items():
            future = executor.submit(process_dataset, dataset_name, fraction, all_data, chunk_size)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            # merge processed data back to all_data
            processed_data = future.result()
            for dataset_type in all_data.keys():
                all_data[dataset_type].extend(processed_data[dataset_type])

    # Construct path for saving
    save_path = os.path.join(get_root_py_fpath(), "data", remixed_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    logging.info(f"Saving remixed datasets to {save_path}")

    # Save the remixed datasets
    for dataset_type, data in all_data.items():
        np.array(data, dtype=np.uint16).tofile(os.path.join(save_path, f"{dataset_type}.bin"))


if __name__ == "__main__":
    fire.Fire({"tokenizer": tokenizer, "remixer": remixer})
