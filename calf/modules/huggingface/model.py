from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from calf import HUGGINGFACE, device
from calf.utils.enumeration import Enumberation
from ..scalar_mix import ScalarMix
from .config import HuggingfaceConfig


@dataclass
class HuggingfaceModelType(Enumberation):
    CausalLM: str = "CausalLM"
    MaskedLM: str = "MaskedLM"


class HuggingfaceModel(torch.nn.Module):
    """
    Args:
        name (str):
            Path or name of the pretrained models registered, e.g.,
            ``'bert-base-cased'``.
        n_layers (int):
            The number of model layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the
            pretrained embedding model. Default: 0.
        mix_dropout (float):
            The dropout ratio of model layers. This value will be passed into the
            :class:`ScalarMix` layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the
            downstream task. Default: ``False``.
    """

    def __init__(
            self,
            model_name: str,
            model_type: str,
            torch_dtype: torch.dtype,
            revision: str = None,
            n_layers: int = 1,
            n_out: int = 0,
            n_context: int = 1024,
            mix_dropout: float = .0,
            finetune: bool = False,
            trust_remote_code: bool = False,
            **kwargs
    ) -> None:
        super().__init__()
        self.name = model_name.replace('/', '--')
        self.model_type = model_type
        self.config = HuggingfaceConfig(
            model_name,
            trust_remote_code=trust_remote_code
        )
        self.config.update_model_config(
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        # init backend model
        self.backend = self.init_model(
            model_name=model_name,
            model_type=model_type,
            cache_dir=str(HUGGINGFACE / "hub"),
            # config=self.config.model_config,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.backend = self.backend.requires_grad_(finetune)

        # unify model properties
        # revision
        self.revision = revision
        # n_layers
        self.n_layers = n_layers if n_layers else self.get_n_layers()
        # hidden size
        self.hidden_size = self.get_hidden_size()
        # output dimension
        self.n_out = n_out if n_out else self.hidden_size
        # dropout
        self.mix_dropout = mix_dropout
        # fixed
        self.finetune = finetune
        # max sequence length
        self.max_len = self.get_max_len(n_context) - 2

        # output processing model
        # scalar mix
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        # project
        if self.hidden_size == self.n_out:
            self.projection = torch.nn.Identity()
        else:
            self.projection = torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=self.n_out,
                bias=False
            )

    def get_n_layers(self) -> int:
        if hasattr(self.backend.config, "num_hidden_layers"):
            # pythia, llama, baichuan, mistral, falcon
            n_layers = self.backend.config.num_hidden_layers
        elif hasattr(self.backend.config, "n_layer"):
            # gpt2, gptj, cerebras
            n_layers = self.backend.config.n_layer
        elif hasattr(self.backend.config, "n_layers"):
            # mosaicml
            n_layers = self.backend.config.d_model
        else:
            # default
            n_layers = 0
        return n_layers

    def get_hidden_size(self) -> int:
        if hasattr(self.backend.config, "hidden_size"):
            # pythia, llama, baichuan, mistral, falcon
            hidden_size = self.backend.config.hidden_size
        elif hasattr(self.backend.config, "n_embd"):
            # gpt2, gptj, cerebras
            hidden_size = self.backend.config.n_embd
        elif hasattr(self.backend.config, "d_model"):
            # mosaicml
            hidden_size = self.backend.config.d_model
        else:
            # default
            hidden_size = 0
        return hidden_size

    def get_max_len(self, n_context: int) -> int:
        if hasattr(self.backend.config, "max_position_embeddings"):
            # pythia, llama, baichuan, mistral
            max_len = self.backend.config.max_position_embeddings
        elif hasattr(self.backend.config, "n_positions"):
            # gpt2, gptj, cerebras
            max_len = self.backend.config.n_positions
        elif hasattr(self.backend.config, "max_seq_len"):
            # mosaicml
            max_len = self.backend.config.max_seq_len
        elif hasattr(self.backend.config, "model_max_length"):
            # falcon
            max_len = self.backend.config.model_max_length
        else:
            # default
            max_len = n_context
        return max_len

    @staticmethod
    def init_model(
            model_name: str,
            model_type: str,
            torch_dtype: torch.dtype,
            cache_dir: str,
            # config=None,
            revision: str = "main",
            trust_remote_code: bool = False,
            **kwargs
    ):
        assert (
            model_type in [HuggingfaceModelType.CausalLM,
                           HuggingfaceModelType.MaskedLM]
        ), f"Unrecoginzed model type {model_type}"
        AutoModelClass = {
            HuggingfaceModelType.CausalLM: AutoModelForCausalLM,
            HuggingfaceModelType.MaskedLM: AutoModelForMaskedLM
        }[model_type]
        try:
            model = AutoModelClass.from_pretrained(
                model_name,
                # config=config,
                cache_dir=cache_dir,
                local_files_only=True,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        except Exception:
            model = AutoModelClass.from_pretrained(
                model_name,
                # config=config,
                cache_dir=cache_dir,
                local_files_only=False,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        return model

    def __call__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if attention_mask:
            attention_mask = attention_mask[:, :self.max_len].float()
        else:
            attention_mask = None
        x = self.backend(
            input_ids=input_ids[:, :self.max_len],
            attention_mask=attention_mask
        ).hidden_states
        # [batch_size, max_len, hidden_size]
        x = self.scalar_mix(x[-self.n_layers:])
        # [batch_size, n_tokens, hidden_size]
        return self.projection(x)


def get_model(
        model_name: str,
        model_type: str = None,
        revision: str = None,
        model_dtype: str = "float16",
        training: bool = False,
        trust_remote_code: bool = False,
        **kwargs
) -> HuggingfaceModel:
    model_type = {"causal": HuggingfaceModelType.CausalLM, "masked": HuggingfaceModelType.MaskedLM}[
        model_type if model_type else "causal"
    ]
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[model_dtype]
    model = HuggingfaceModel(
        model_name=model_name,
        model_type=model_type,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    model.train(mode=training)
    model.to(device=torch.device(device))
    return model