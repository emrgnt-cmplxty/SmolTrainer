# SmolTrainer

Welcome to SmolTrainer, an evolution of the popular nanoGPT repository, tailored to train additional models such as Llama, or with additional techniques like LoRA.

---

## Install and Setup

### Dependencies

The following are the primary dependencies for the project:

1. **python**: The version of Python used by the project.
   - Version: `>=3.10,<3.11`
2. **torch**: PyTorch, a popular deep learning framework.
   - Version: `2.0.0`
3. **numpy**: A library for numerical computations in Python.
   - Version: `^1.25.2`
4. **requests**: A Python library for making HTTP requests.
   - Version: `^2.31.0`
5. **tiktoken**: A library by OpenAI for tokenizing text.
   - Version: `^0.4.0`
6. **wandb**: Weights & Biases, a tool for machine learning experiment tracking.
   - Version: `^0.15.9`
7. **python-dotenv**: A library to read key-value pairs from a `.env` file and set them as environment variables.
   - Version: `^1.0.0`
8. **datasets**: A library by Hugging Face for easily accessing and processing various datasets.
   - Version: `^2.14.5`

```bash
# Clone the repository
git clone git@github.com:emrgnt-cmplxty/SmolTrainer.git && cd SmolTrainer

# Initialize git submodules
git submodule update --init

# Install poetry and the project
pip3 install poetry && poetry install

#  - If on A100 or newer - 
# Install PyTorch nightly 2.1.0 to use the latest Flash Attention.
# poetry run pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
# poetry run pip install 'flash-attn>=2.0.0.post1' --no-build-isolation
# Update smol_trainer/nano_gpt/model.py attention mechansim by hand to use flash_attn
# e.g. return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)


# Optional development tooling
# pre-commit install
```

```bash
# Prepare the training data
poetry run python smol_trainer/nano_gpt/data/shakespeare/prepare.py
mv smol_trainer/nano_gpt/data/shakespeare smol_trainer/data

# For the full training set, run the following -
# poetry run python smol_trainer/data/concoction/prepare.py
```

## Train

```bash
# Perform a local training run with GPT
export DATASET=shakespeare
export MODEL=gpt
poetry run python smol_trainer/runner.py  --eval-iters=20 --log-interval=1 --block-size=1024 --batch-size=12 --n-layer=12 --n-head=12 --n-embd=768 --max-iters=2000000 --lr-decay-iters=60000 --dropout=0.0 --model=$MODEL --dataset=$DATASET --gradient-accumulation-steps=1 --min-lr=1e-4 --beta2=0.99 --eval-interval=2500 --grad-clip=1 --compile=False --device=cpu

```

Great, now let's proceed onward to train the full UberSmol model. We assume that the training run occurs on 8x A100s.

```bash

## UberSmol-pico
poetry run torchrun --standalone --nproc_per_node=8 smol_trainer/runner.py --batch-size=8 --n-layer=12 --n-head=12 --n-embd=768 --dataset=concoction  --gradient-accumulation-steps=40 --wandb-log --eval-iters=250

## UberSmol-nano
poetry run torchrun --standalone --nproc_per_node=8 smol_trainer/runner.py --batch-size=8 --n-layer=24 --n-head=16 --n-embd=1024 --dataset=concoction  --gradient-accumulation-steps=40 --wandb-log --eval-iters=250

## UberSmol-micro
poetry run torchrun --standalone --nproc_per_node=8 smol_trainer/runner.py --batch-size=8 --n-layer=36 --n-head=20 --n-embd=1280 --dataset=concoction  --gradient-accumulation-steps=40 --wandb-log --eval-iters=500

## UberSmol-mili
poetry run torchrun --standalone --nproc_per_node=8 smol_trainer/runner.py --batch-size=8 --n-layer=48 --n-head=25 --n-embd=1600 --dataset=concoction  --gradient-accumulation-steps=40 --wandb-log --eval-iters=500
```

### Comments

* For additional monitoring, pass --wandb-log.

## Evaluate

```bash
# Run simple inference your model, defaults to iter_num=2000
poetry run python smol_trainer/inference.py --model_prefix=run_0_checkpoint__mode_moe__n_layer_12__n_head_4__n_embd_128__n_experts_8__top_k_experts_8
```

### TODO

- [ ] Add llama model to training workflow
- [ ] Add lora model to training workflow
- [ ] Implement llama2 model
- [ ] Add llama2 model to trainer
