import os
import torch
from typing import Dict, Tuple
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, AutoTokenizer
from smol_trainer.config import ModelConfig

import fire


class WeightMapping:
    """
    Handles the weight mapping from one model's state dict to another's.
    """

    def __init__(self, prefix="_orig_mod."):
        self.prefix = prefix
        self.weight_map = self._initialize_weight_map()

    def _initialize_weight_map(self) -> Dict[str, str]:
        return {
            self.prefix + "transformer.wte.weight": "gpt_neox.embed_in.weight",
            self.prefix
            + "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
            self.prefix
            + "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
            self.prefix
            + "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
            self.prefix
            + "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
            self.prefix
            + "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
            self.prefix
            + "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
            self.prefix
            + "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
            self.prefix
            + "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
            self.prefix
            + "transformer.h.{}.mlp.fc.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
            self.prefix
            + "transformer.h.{}.mlp.fc.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
            self.prefix
            + "transformer.h.{}.mlp.proj.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
            self.prefix
            + "transformer.h.{}.mlp.proj.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
            self.prefix + "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
            self.prefix + "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
            self.prefix + "lm_head.weight": "embed_out.weight",
        }

    def layer_template(self, layer_name: str, idx: int) -> Tuple[str, int]:
        """
        Split layer name and extract layer number.
        """
        split = layer_name.split(".")
        number = int(split[idx])
        split[idx] = "{}"
        from_name = ".".join(split)
        return from_name, number

    def map_weights(self, torch_weights: Dict, state_dict: Dict) -> None:
        """
        Map weights from the `torch_weights` to the `state_dict`.
        """
        for name, param in torch_weights.items():
            if "transformer.h" in name:
                from_name, number = self.layer_template(name, 3)
                to_name = self.weight_map[from_name].format(number)
            else:
                to_name = self.weight_map[name]

            param = self.load_param(param, name, None)
            print(f"overwriting {to_name}")
            state_dict[to_name] = param

    @staticmethod
    def load_param(param, name: str, dtype) -> torch.Tensor:
        """
        Load parameters, with optional dtype conversion.
        """
        if hasattr(param, "_load_tensor"):
            print(f"Loading {name!r} into RAM")
            param = param._load_tensor()
        if dtype is not None and dtype != param.dtype:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
            param = param.to(dtype)
        return param


class ModelHandler:
    """
    Handles the loading, transformation, and saving of models.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        result_dir: str,
        run_name: str,
        iter_num: int,
    ):
        self.iter_num = iter_num
        self.run_name = run_name

        self.result_dir = result_dir
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        # script file path
        script_file_path = os.path.dirname(os.path.realpath(__file__))
        self.results_dir = os.path.join(
            script_file_path,
            "..",
            result_dir,
            f"checkpoint_run_name_{run_name}__model_{model_name}",
        )

    def load_model(self) -> Dict:
        """
        Load the model's state dict from disk.
        """
        return torch.load(
            os.path.join(
                self.results_dir,
                f"checkpoint_run_name_{self.run_name}__model_{self.model_name}__iter_num_{self.iter_num}.pt",
            )
        )

    def convert_to_hf_model(self, model_name: str, state_dict: Dict) -> GPTNeoXForCausalLM:
        """
        Convert a PyTorch model's state dict to a Hugging Face GPTNeo model.
        """
        torch_config = ModelConfig.from_name(model_name)

        hf_config = GPTNeoXConfig(
            hidden_size=torch_config.n_embd,
            num_attention_heads=torch_config.n_head,
            num_hidden_layers=torch_config.n_layer,
            intermediate_size=torch_config.intermediate_size,
            max_position_embeddings=torch_config.block_size,
            rotary_pct=torch_config.rotary_percentage,
            vocab_size=torch_config.padded_vocab_size,
            use_parallel_residual=torch_config.parallel_residual,
            use_cache=False,
            torch_dtype="float16",
            hidden_act="gelu",
            initializer_range=0.02,
            layer_norm_eps=1e-05,
            rotary_emb_base=10000,
        )
        hf_model = GPTNeoXForCausalLM(hf_config)

        mapper = WeightMapping()
        mapper.map_weights(state_dict["model"], hf_model.state_dict())
        hf_model.load_state_dict(hf_model.state_dict())

        return hf_model

    def save_model_and_tokenizer(self, hf_model: GPTNeoXForCausalLM) -> None:
        """
        Save the Hugging Face model and tokenizer to disk.
        """
        hf_path = os.path.join(
            self.results_dir, f"{self.run_name}_{self.model_name}_{self.iter_num}_hf"
        )
        print(f"Saving model to {hf_path}")
        hf_model.save_pretrained(hf_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.save_pretrained(hf_path)


def main(
    iter_num: int,
    result_dir: str = "results",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    model_name: str = "pythia-410m",
    run_name: str = "run_open-orca",
):
    """
    Main function for CLI execution.
    """
    handler = ModelHandler(model_name, tokenizer_name, result_dir, run_name, iter_num)
    state_dict = handler.load_model()
    hf_model = handler.convert_to_hf_model(model_name, state_dict)
    handler.save_model_and_tokenizer(hf_model)


if __name__ == "__main__":
    fire.Fire(main)

# import os
# import torch
# from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
# from smol_trainer.config import ModelConfig
# from transformers import AutoTokenizer


# def copy_weights_to_state_dict(
#     lit_weights,
#     state_dict,
#     prefix="_orig_mod.",
# ) -> None:
#     weight_map = {
#         self.prefix + "transformer.wte.weight": "gpt_neox.embed_in.weight",
#         self.prefix + "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
#         self.prefix + "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
#         prefix
#         + "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
#         prefix
#         + "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
#         self.prefix + "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
#         self.prefix + "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
#         prefix
#         + "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
#         prefix
#         + "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
#         self.prefix + "transformer.h.{}.mlp.fc.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
#         self.prefix + "transformer.h.{}.mlp.fc.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
#         self.prefix + "transformer.h.{}.mlp.proj.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
#         self.prefix + "transformer.h.{}.mlp.proj.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
#         self.prefix + "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
#         self.prefix + "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
#         self.prefix + "lm_head.weight": "embed_out.weight",
#     }

#     for name, param in lit_weights.items():
#         if "transformer.h" in name:
#             from_name, number = layer_template(name, 3)
#             to_name = weight_map[from_name].format(number)
#         else:
#             to_name = weight_map[name]
#         param = load_param(param, name, None)
#         # if saver is not None:
#         # param = saver.store_early(param)
#         print(f"overwriting {to_name}")
#         state_dict[to_name] = param


# def layer_template(layer_name: str, idx: int):
#     split = layer_name.split(".")
#     number = int(split[idx])
#     split[idx] = "{}"
#     from_name = ".".join(split)
#     return from_name, number


# def load_param(param, name: str, dtype) -> torch.Tensor:
#     if hasattr(param, "_load_tensor"):
#         # support tensors loaded via `lazy_load()`
#         print(f"Loading {name!r} into RAM")
#         param = param._load_tensor()
#     if dtype is not None and dtype != param.dtype:
#         print(f"Converting {name!r} from {param.dtype} to {dtype}")
#         param = param.to(dtype)
#     return param


# if __name__ == "__main__":
#     # Load your local PyTorch model
#     result_dir = "results"
#     run_name = "run_open-orca"
#     model = "pythia-410m"
#     tokenizer = "mistralai/Mistral-7B-v0.1"
#     iter_num = 9100

#     results_dir = os.path.join(result_dir, f"checkpoint_run_name_{run_name}__model_{model}")

#     state_dict = torch.load(
#         os.path.join(
#             results_dir,
#             f"checkpoint_run_name_{run_name}__model_{model}__iter_num_{iter_num}.pt",
#         )
#     )

#     torch_config = ModelConfig.from_name(model)

#     hf_config = GPTNeoXConfig(
#         hidden_size=torch_config.n_embd,
#         num_attention_heads=torch_config.n_head,
#         num_hidden_layers=torch_config.n_layer,
#         intermediate_size=torch_config.intermediate_size,
#         max_position_embeddings=torch_config.block_size,
#         rotary_pct=torch_config.rotary_percentage,
#         vocab_size=torch_config.padded_vocab_size,
#         use_parallel_residual=torch_config.parallel_residual,
#         use_cache=False,
#         torch_dtype="float16",
#         hidden_act="gelu",
#         initializer_range=0.02,
#         layer_norm_eps=1e-05,
#         rotary_emb_base=10000,
#     )

#     hf_model = GPTNeoXForCausalLM(hf_config)

#     copy_weights_to_state_dict(state_dict["model"], hf_model.state_dict())
#     hf_model.load_state_dict(hf_model.state_dict())

#     # Save the Hugging Face model to the disk
#     hf_path = os.path.join(results_dir, f"{run_name}_{model}_{iter_num}_hf")
#     print(f"Saving model to {hf_path}")
#     hf_model.save_pretrained(hf_path)

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer)
#     tokenizer.save_pretrained(hf_path)
