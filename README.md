# SmolTrainer

Welcome to SmolTrainer, an evolution of the popular nanoGPT repository, tailored to train additional models such as Llama, or with additional techniques like LoRA.

---

## Install and Setup

```bash
# Clone the repository
git clone git@github.com:emrgnt-cmplxty/SmolTrainer.git && cd SmolTrainer

# Initialize git submodules
git submodule update --init

# Install poetry and the project
pip3 install poetry && poetry install

# Optional development tooling
# pre-commit install
```

```bash
# Prepare the training data
poetry run python smol_trainer/data/concoction/prepare.py
```

## Train

```bash
# Perform a training run with GPT
export MODE=gpt
poetry run python smol_trainer/runner.py --device=cpu --compile=False --eval-iters=20 --log-interval=1 --block-size=64 --batch-size=12 --n-layer=4 --n-head=4 --n-embd=128 --max-iters=2000 --lr-decay-iters=60000 --dropout=0.0 --mode=$MODE --dataset=concoction --gradient-accumulation-steps=1 --min-lr=1e-4 --beta2=0.99 --n-experts=128 --top-k-experts=16  --eval-interval=50

# Perform a training run with MoE
export MODE=moe
poetry run python smol_trainer/runner.py --device=cpu --compile=False --eval-iters=20 --log-interval=1 --block-size=64 --batch-size=12 --n-layer=4 --n-head=4 --n-embd=128 --max-iters=2000 --lr-decay-iters=60000 --dropout=0.0 --mode=$MODE --dataset=concoction --gradient-accumulation-steps=1 --min-lr=1e-4 --beta2=0.99 --n-experts=128 --top-k-experts=16  --eval-interval=50

# Monitor your progress
poetry run tensorboard --logdir=results/
```

### Comments

* For additional monitoring, pass --wandb-log.

## Evaluate

```bash
# Run simple inference your model, defaults to iter_num=2000
poetry run python smol_trainer/inference.py --model_prefix=run_0_checkpoint__mode_moe__n_layer_12__n_head_4__n_embd_128__n_experts_8__top_k_experts_8
```
