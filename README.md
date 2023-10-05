# SmolTrainer

Welcome to SmolTrainer, an evolution of the popular nanoGPT repository, tailored to train additional models with as little overhead as possible.
---

## Install and Setup

```bash
# Clone the repository
git clone git@github.com:emrgnt-cmplxty/SmolTrainer.git && cd SmolTrainer

# Install poetry and the project
pip3 install poetry && poetry install

# Optional development tooling
# pre-commit install
```

Great, now let's proceed onward to train the full UberSmol models.

```bash
# Perform training run with pythia-410m
# place dataset in smol_trainer/data/open-platypus/[train,val].bin
export DATASET=open-platypus 
export MODEL_NAME=pythia-410m
poetry run python smol_trainer/runner.py \
    --model-name=$MODEL_NAME \
    --dataset=$DATASET \
    --batch-size=8 \
    --block-size=1024 \
    --eval-iters=250 \
    --compile=True \
    --device=cuda \
    --eval-interval=100 \
    --wandb-log \
    --run-name=run_$DATASET
``````