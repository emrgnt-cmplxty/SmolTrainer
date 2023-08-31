# BabyMoE Framework

## Install and Setup

```bash
# Clone the repository
git clone git@github.com:emrgnt-cmplxty/BabyMoE.git && cd BabyMoE

# Initialize git submodules
git submodule update --init

# Install poetry and the project
pip3 install poetry && poetry install
```

```bash
# Prepare the training data
poetry run python baby_moe/nano_gpt/data/shakespeare_char/prepare.py
```

## Train

```bash
# Perform a training run with GPT
export MODE=gpt
poetry run python baby_moe/train.py baby_moe/nano_gpt/config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --mode=$MODE

# Perform a training run with MoE
export MODE=moe
poetry run python baby_moe/train.py baby_moe/nano_gpt/config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --mode=$MODE

```
