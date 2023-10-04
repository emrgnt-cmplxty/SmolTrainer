# SmolTrainer

Welcome to SmolTrainer, an evolution of the popular nanoGPT repository, tailored to train additional models with as little overhead as possible.
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

# For the full training set, run the following -
# poetry run python smol_trainer/data/concoction/prepare.py

## Train

```bash
# Perform a local training run with GPT
export DATASET=shakespeare
export MODEL_NAME=pythia-70m
poetry run python smol_trainer/runner.py  --model-name=$MODEL_NAME --dataset=$DATASET --batch-size=8 --block-size=64 --eval-iters=250 --compile=False --device=cpu 
```

Great, now let's proceed onward to train the full UberSmol models.

```bash
poetry run python smol_trainer/runner.py --batch-size=8 --n-layer=12 --n-head=12 --n-embd=768 --dataset=concoction  --gradient-accumulation-steps=40 --wandb-log --eval-iters=250
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
