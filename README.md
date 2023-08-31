# BabyMoE Framework

## Install and Setup

```bash
# Clone the repository
git clone git@github.com:emrgnt-cmplxty/BabyMoE.git && cd BabyMoE

# Initialize git submodules
git submodule update --init

# Install poetry and the project
pip3 install poetry && poetry install

# Optional development tooling
# pre-commit install
```

```bash
# Prepare the training data
poetry run python baby_moe/nano_gpt/data/shakespeare_char/prepare.py
```

## Train

```bash
# Perform a training run with GPT
export MODE=gpt
poetry run python baby_moe/train.py --device=cpu --compile=False --eval-iters=20 --log-interval=1 --block-size=64 --batch-size=12 --n-layer=4 --n-head=4 --n-embd=128 --max-iters=2000 --lr-decay-iters=2000 --dropout=0.0 --mode=$MODE --dataset=shakespeare_char --gradient-accumulation-steps=1 --min-lr=1e-4 --beta2=0.99

# Perform a training run with MoE
export MODE=moe
poetry run python baby_moe/train.py --device=cpu --compile=False --eval-iters=20 --log-interval=1 --block-size=64 --batch-size=12 --n-layer=4 --n-head=4 --n-embd=128 --max-iters=2000 --lr-decay-iters=2000 --dropout=0.0 --mode=$MODE --dataset=shakespeare_char --gradient-accumulation-steps=1 --min-lr=1e-4 --beta2=0.99 


```
