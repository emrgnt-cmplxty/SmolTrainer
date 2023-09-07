"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from smol_trainer.config import Mode
from smol_trainer.model.moe import MoEGPT
from smol_trainer.nano_gpt.model import GPT, GPTConfig

# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "results"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
model_prefix = "checkpoint__mode_moe__n_layer_4__n_head_4__n_embd_128__n_experts_128__top_k_experts_16"
iter_num = "2000"
meta_path = "x"
exec(
    open("nano_gpt/configurator.py").read()
)  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = (
    "cuda" if "cuda" in device else "cpu"
)  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(
        out_dir, model_prefix, f"{model_prefix}__iter_num_{iter_num}.pt"
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(
        block_size=checkpoint["block_size"],
        n_layer=checkpoint["n_layer"],
        n_head=checkpoint["n_head"],
        n_embd=checkpoint["n_embd"],
        dropout=checkpoint["dropout"],
        bias=checkpoint["bias"],
    )
    model = (
        GPT(gptconf)
        if checkpoint["mode"].value == Mode.GPT.value
        else MoEGPT(
            gptconf, checkpoint["n_experts"], checkpoint["top_k_experts"]
        )
    )
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
print("checkpoint = ", checkpoint.keys())
# ok let's assume gpt-2 encodings by default
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(
                x, max_new_tokens, temperature=temperature, top_k=top_k
            )
            print(decode(y[0].tolist()))
            print("---------------")
