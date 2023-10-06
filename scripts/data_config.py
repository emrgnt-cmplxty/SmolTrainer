# datasets_config.py

datasets_config = {
    "textbook_quality_programming": {
        "path": "vikp/textbook_quality_programming",
        "mappings": {
            "markdown": "",
        },
        "tokenizer": "simple_tokenize",
    },
    "open-platypus": {
        "path": "garage-bAInd/Open-Platypus",
        "mappings": {
            "instruction": "### Instruction:",
            "output": "### Output:",
        },
        "tokenizer": "simple_tokenize",
    },
    "sciphi-python-textbook": {
        "path": "emrgnt-cmplxty/sciphi-python-textbook",
        "mappings": {
            "formatted_prompt": "### Instruction:",
            "completion": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "sciphi-textbooks-are-all-you-need": {
        "path": "emrgnt-cmplxty/sciphi-textbooks-are-all-you-need",
        "mappings": {
            "formatted_prompt": "### Instruction:",
            "completion": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "open-phi-textbooks": {
        "path": "open-phi/textbooks",
        "mappings": {"markdown": ""},
        "tokenizer": "simple_tokenize",
    },
    "programming-books-llama": {
        "path": "open-phi/programming_books_llama",
        "mappings": {"markdown": ""},
        "tokenizer": "simple_tokenize",
    },
    "open-orca": {
        "path": "Open-Orca/OpenOrca",
        "mappings": {
            "system_prompt": "### System:",
            "question": "### Question:",
            "response": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "tiny-stories": {
        "path": "roneneldan/TinyStories",
        "mappings": {"text": ""},
        "tokenizer": "simple_tokenize",
    },
    "tiny-codes": {
        "path": "nampdn-ai/tiny-codes",
        "mappings": {
            "prompt": "### Instruction:",
            "response": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "tiny-orca": {
        "path": "nampdn-ai/tiny-orca-textbooks",
        "mappings": {
            "prompt": "### System:",
            "question": "### Question:",
            "textbook": "### Context:",
            "response": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "tiny-textbooks": {
        "path": "nampdn-ai/tiny-textbooks",
        "mappings": {
            "text": "### Instruction:\nWrite a lesson based on the following content:",
            "textbook": "### Response:",
        },
        "tokenizer": "simple_tokenize",
    },
    "meta-math": {
        "path": "meta-math/MetaMathQA",
        "mappings": {
            "query": "### Question:",
            "response": "### Answer:",
        },
        "tokenizer": "simple_tokenize",
    },
    "evol-instruct": {
        "path": "nickrosh/Evol-Instruct-Code-80k-v1",
        "mappings": {
            "instruction": "### Instruction:",
            "output": "### Output:",
        },
        "tokenizer": "simple_tokenize",
    },
}
