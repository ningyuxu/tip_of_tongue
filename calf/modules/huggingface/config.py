from transformers import AutoConfig
from transformers import GenerationConfig
from calf import HUGGINGFACE


class HuggingfaceConfig:

    def __init__(self, model_name: str, **kwargs) -> None:
        model_path = f"models--{model_name.replace('/', '--')}"

        self.config_path = HUGGINGFACE / "hub" / model_path / "config"
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.model_config_file = self.config_path / "config.json"
        self.model_config = None
        self.init_model_config(model_name, **kwargs)

        self.generation_config_file = self.config_path / "generation_config.json"
        self.generation_config = None
        self.init_generation_config(source="from_model_config", model_name=model_name)

    def init_model_config(self, model_name: str, **kwargs) -> None:
        if not self.model_config_file.is_file():
            self.model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_name,
                **kwargs
            )
            self.model_config.save_pretrained(
                save_directory=str(self.config_path)
            )
        else:
            self.model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=str(self.config_path),
                **kwargs
            )

    def update_model_config(self, **kwargs) -> None:
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=str(self.config_path),
            output_hidden_states=True,
            # output_attentions=True,
            return_dict=True,
            **kwargs,
        )

    def init_generation_config(
            self,
            source: str,
            model_name: str,
            **kwargs
    ) -> None:
        """
        Source can be `from_pretrained` or `from_model_config`
        """
        if not self.generation_config_file.is_file():
            if source == "from_pretrained":
                self.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name=model_name,
                    **kwargs
                )
            elif source == "from_model_config":
                self.generation_config = GenerationConfig.from_model_config(
                    model_config=self.model_config,
                )
            else:
                raise ValueError(
                    f"Unrecognized generation config source {source}"
                )
            self.generation_config.save_pretrained(
                save_directory=str(self.config_path)
            )
        else:
            self.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name=str(self.config_path),
                **kwargs
            )

    def update_generation_config(self, **kwargs) -> None:
        self.generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name=str(self.config_path),
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )