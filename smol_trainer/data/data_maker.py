import os

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


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


def simple_tokenize(batch, formatters, tokenizer):
    combined_text = []
    for column, prefix in formatters.items():
        prefix = formatters[column]
        combined_text.append(f"{prefix}:\n{batch[column]}")
    return tokenizer("\n\n".join(combined_text), return_tensors="np")


HF_TOKEN = os.environ.get("HF_TOKEN")
TRAIN_SPLIT = "train"
MULTIPLIER = 1
ENCODING = "mistralai/Mistral-7B-v0.1"
VAL_FRAC = 0.01
NUM_PROC = 8

if __name__ == "__main__":
    # Example of usage
    loader = DatasetLoader(split=TRAIN_SPLIT, token=HF_TOKEN)

    loader.add_dataset(
        "sciphi-python-textbook",
        "emrgnt-cmplxty/sciphi-python-textbook",
        {
            "formatted_prompt": "### Instruction:",
            "completion": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "sciphi-textbooks-are-all-you-need",
        "emrgnt-cmplxty/sciphi-textbooks-are-all-you-need",
        {
            "formatted_prompt": "### Instruction:",
            "completion": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "open-phi-textbooks",
        "open-phi/textbooks",
        {"markdown": ""},
        simple_tokenize,
    )
    loader.add_dataset(
        "programming-books-llama",
        "open-phi/programming_books_llama",
        {"markdown": ""},
        simple_tokenize,
    )
    loader.add_dataset(
        "open-orca",
        "Open-Orca/OpenOrca",
        {
            "system_prompt": "### System:",
            "question": "### Question:",
            "response": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "tiny-stories", "roneneldan/TinyStories", {"text": ""}, simple_tokenize
    )
    loader.add_dataset(
        "tiny-codes",
        "nampdn-ai/tiny-codes",
        {
            "prompt": "### Instruction:",
            "response": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "tiny-orca",
        "nampdn-ai/tiny-orca-textbooks",
        {
            "prompt": "### System:",
            "question": "### Question:",
            "textbook": "### Context:",
            "response": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "tiny-textbooks",
        "nampdn-ai/tiny-textbooks",
        {
            "text": "### Instruction:\nWrite a lesson based on the following content:",
            "textbook": "### Response:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "meta-math",
        "meta-math/MetaMathQA",
        {
            "query": "### Question:",
            "response": "### Answer:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "evol-instruct",
        "nickrosh/Evol-Instruct-Code-80k-v1",
        {
            "instruction": "### Instruction:",
            "output": "### Output:",
        },
        simple_tokenize,
    )
    loader.add_dataset(
        "open-platypus",
        "garage-bAInd/Open-Platypus",
        {
            "instruction": "### Instruction:",
            "output": "### Output:",
        },
        simple_tokenize,
    )

    datasets = loader.load_all_datasets()
    tokenizer = AutoTokenizer.from_pretrained(ENCODING)

    for dataset_name, payload in datasets.items():
        dataset, formatters, raw_tokenize_function = payload
        if "train" in dataset:
            dataset = dataset["train"]

        tokenize_function = lambda x: raw_tokenize_function(
            x, formatters, tokenizer
        )
        dataset = dataset.shuffle(seed=42)

        # Tokenize the subset
        tokenized_dataset = dataset.map(tokenize_function, num_proc=NUM_PROC)

        flattened_tokens = [
            token
            for sublist in tokenized_dataset["input_ids"]
            for token in sublist[0]
        ]

        train_ids = flattened_tokens[
            : int(len(flattened_tokens) * (1 - VAL_FRAC))
        ]
        val_ids = flattened_tokens[
            int(len(flattened_tokens) * (1 - VAL_FRAC)) :
        ]

        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), dataset_name)
        ):
            os.mkdir(os.path.join(os.path.dirname(__file__), dataset_name))

        train_ids.tofile(
            os.path.join(
                os.path.dirname(__file__),
                dataset_name,
                f"{dataset_name}_train.bin",
            )
        )
        val_ids.tofile(
            os.path.join(
                os.path.dirname(__file__),
                dataset_name,
                f"{dataset_name}_val.bin",
            )
        )
