# TODO - Implement GPT
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os

import numpy as np
import tiktoken
from datasets import concatenate_datasets, load_dataset  # huggingface datasets
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Input args

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

tokenizer = "gpt2"
seed = 2357
test_size = 0.01

# Configurations for the dataset and processing

config = {
    "open-orca": {
        "dataset_name": "Open-Orca/OpenOrca",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "system_prompt": "### System Prompt:\n",
            "question": "\n### Instruction:\n",
            "response": "\n### Response:\n",
        },
    },
    "tiny-codes": {
        "dataset_name": "nampdn-ai/tiny-codes",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "prompt": "You are a helpful coding agent.\n### Instruction:\n",
            "response": "\n### Response:\n",
        },
    },
    "tiny-cot": {
        "dataset_name": "nampdn-ai/tiny-cot",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "source": "You are a helpful assistant that thinks step-by-step to solve instructions.\n### Instruction:\n",
            "rationale": "\n### Response:\n### Thought:\n",
            "target": "\n### Answer:\n",
        },
    },
    "wizardlm-orca": {
        "dataset_name": "psmathur/WizardLM_Orca",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "system": "",
            "instruction": "### Instruction:\n",
            "output": "\n### Response:\n",
        },
    },
    "cot-submix": {
        "dataset_name": "conceptofmind/cot_submix_original",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "inputs": "",
            "targets": "",
        },
    },
    "gpt-teacher": {
        "dataset_name": "teknium/GPTeacher-General-Instruct",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "input": "\n### Input:\n",
            "response": "\n### Response:\n",
        },
    },
    "alpaca-cleaned": {
        "dataset_name": "yahma/alpaca-cleaned",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "input": "\n### Input:\n",
            "output": "\n### Response:\n",
        },
    },
    "codealpaca-20k": {
        "dataset_name": "sahil2801/CodeAlpaca-20k",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "input": "\n",
            "output": "### Response:\n",
        },
    },
    "unnatural": {
        "dataset_name": "TokenBender/unnatural_code_instructions_20M",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "text": "",
        },
    },
    "evol-instruct-code-v1": {
        "dataset_name": "nickrosh/Evol-Instruct-Code-80k-v1",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "output": "\n### Response:\n",
        },
    },
    "open-platypus": {
        "dataset_name": "garage-bAInd/Open-Platypus",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "output": "\n### Response:\n",
        },
    },
    "evol-codealpaca-v1": {
        "dataset_name": "theblackcat102/evol-codealpaca-v1",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "output": "\n### Response:\n",
        },
    },
    "vicuna_70k": {
        "dataset_name": "ehartford/wizard_vicuna_70k_unfiltered",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "conversations": "Below is a conversation between a human and a helpful agent, in JSON format.\n",
        },
    },
    "chat_arena": {
        "dataset_name": "lmsys/chatbot_arena_conversations",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "conversation_a": "Below is a conversation between a human and a helpful agent, in JSON format.\n",
            "conversation_b": "The following is a separate version of the same conversation, in JSON format.\n",
        },
    },
    "llava_150k": {
        "dataset_name": "liuhaotian/LLaVA-Instruct-150K",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "conversations": "Below is a conversation between a human and a helpful agent, in JSON format.\n",
        },
    },
    "kaist_ai_cot": {
        "dataset_name": "kaist-ai/CoT-Collection",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "source": "You are a helpful assistant that thinks step-by-step to solve instructions.\n### Instruction:\n",
            "rationale": "\n### Response:\n### Thought:\n",
            "target": "\n### Answer:\n",
        },
    },
    "leetcode_with_solutions": {
        "dataset_name": "mhhmm/leetcode-solutions-python",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "code_with_data": "Below is a coding problem with the correct solution:\n",
            "explanation_only": "Here is an explanation for the solution:\n",
        },
    },
    "tiny_stories": {
        "dataset_name": "skeskinen/TinyStories-GPT4",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "prompt": "",
            "story": "\n",
            "summary": "\n ### Summary:\n",
        },
    },
    "claude_instruct": {
        "dataset_name": "Norquinal/claude_evol_instruct_210k",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### Instruction:\n",
            "output": "\n### Response:\n",
        },
    },
    "airoboros": {
        "dataset_name": "jondurbin/airoboros-2.1",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "system": "",
            "instruction": "### Instruction:\n",
            "response": "\n### Response:\n",
        },
    },
    "databricks-dolly": {
        "dataset_name": "databricks/databricks-dolly-15k",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "instruction": "### User:\n",
            "response": "\n### Assistant:\n",
        },
    },
    "claude_chats": {
        "dataset_name": "Norquinal/claude_multiround_chat_30k",
        "test_size": test_size,
        "shuffle": True,
        "format": {
            "conversations": "Below is a conversation between a human and a helpful assistant, in JSON format.\n",
        },
    },
}
# Short List For Addition
# https://huggingface.co/datasets/kaist-ai/CoT-Collection/viewer/en/train?row=1
# https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations
# https://huggingface.co/datasets/Muennighoff/natural-instructions/viewer/default/train?row=1
# https://huggingface.co/datasets/laion/OIG
# https://huggingface.co/datasets/roneneldan/TinyStories
# https://huggingface.co/datasets/SirNeural/flan_v2
# https://huggingface.co/datasets/stingning/ultrachat
# https://huggingface.co/datasets/laion/OIG
# https://huggingface.co/datasets/Open-Orca/FLAN
# https://huggingface.co/datasets/MBZUAI/LaMini-instruction/viewer/default/train?p=2&row=204
# https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval
# https://huggingface.co/datasets/camel-ai/math
# https://huggingface.co/datasets/eli5_category/viewer/default/train?row=31
# https://huggingface.co/datasets/ehartford/dolphin/viewer/default/train?row=10
# https://huggingface.co/datasets/StudentLLM/Open-Wyvern-74k


def load_and_split_dataset(config):
    """Load the dataset and split into train and validation sets."""
    dataset = load_dataset(
        config["dataset_name"],
        num_proc=num_proc_load_dataset,
        token=os.getenv("HF_TOKEN", None),
    )

    # Only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(
        test_size=config["test_size"],
        seed=seed,
        shuffle=config["shuffle"],
    )
    split_dataset["val"] = split_dataset.pop("test")

    return split_dataset


def tokenize_dataset(split_dataset, config):
    """Tokenize the dataset."""
    columns = config["format"].keys()
    format_dict = config["format"]

    def process(example):
        # Use the formatter to format each column
        text = "\n".join(
            [f"{format_dict.get(col, '{}')} {example[col]}" for col in columns]
        )
        enc = tiktoken.get_encoding(tokenizer)

        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=columns,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    return tokenized


if __name__ == "__main__":
    # Load and split dataset based on the configuration
    print("Splitting now...")
    results_train = None
    results_val = None
    total_tokens = 0
    for key in config.keys():
        print(f"Handling config key = {key}")

        print("Loading dataset now...")
        selected_config = config[key]
        split_data = load_and_split_dataset(selected_config)

        # Get the tokenizer encoding based on the configuration
        print("Tokenizing now...")
        # Tokenize the data based on the configuration
        tokenized_data = tokenize_dataset(split_data, selected_config)
        tokens = np.sum(tokenized_data["train"]["len"]) / 1e6
        total_tokens += tokens
        print(f"Total Tokens = {total_tokens}M")
        if not results_train:
            results_train = tokenized_data["train"]
            results_val = tokenized_data["val"]
        else:
            results_train = concatenate_datasets(
                [results_train, tokenized_data["train"]]
            ).shuffle(seed=seed)
            results_val = concatenate_datasets(
                [results_val, tokenized_data["val"]]
            ).shuffle(seed=seed)
    tokenized = {
        "train": results_train,
        "val": results_val,
    }
    print("Creating the test-train split")
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = (
            np.uint16
        )  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(
            range(total_batches), desc=f"writing {filename}"
        ):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
