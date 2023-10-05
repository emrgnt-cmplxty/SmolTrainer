# Derived from https://github.com/Lightning-AI/lit-gpt/tree/main. Apache-2 License.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
from typing_extensions import Self


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelConfig:
    org: str = "Emergent-AGI"
    name: str = "SmolModel"
    # GPT Params
    block_size: int = 4096
    vocab_size: int = 32_000
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 768
    bias: bool = True

    # Unused
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError(
                    "The config needs to set the `intermediate_size`"
                )
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        if "condense_ratio" in json_kwargs:  # legacy name
            json_kwargs["rope_condense_ratio"] = json_kwargs.pop(
                "condense_ratio"
            )
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @property
    def mlp_class(self) -> Type:
        from smol_trainer import model

        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from smol_trainer.model import RMSNorm

            return RMSNorm
        return getattr(torch.nn, self._norm_class)


####################
# EleutherAI Pythia
####################
configs = [
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-70m",
        block_size=2048,
        n_layer=6,
        n_embd=512,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-160m",
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-410m",
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-1b",
        block_size=2048,
        n_embd=2048,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-1.4b",
        block_size=2048,
        n_layer=24,
        n_embd=2048,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-2.8b",
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-6.9b",
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
    ),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(
        org="EleutherAI",
        name="pythia-12b",
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
    ),
]
configs.append(
    dict(
        org="TEST",
        name="test-local",
        block_size=1024,
        n_layer=4,
        n_embd=128,
        n_head=4,
    )
)

name_to_config = {config["name"]: config for config in configs}
