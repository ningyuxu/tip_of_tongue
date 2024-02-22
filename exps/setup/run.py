import sys
from omegaconf import DictConfig

from calf import logger
from calf.utils.log import log_line
from exps.utils.huggingface import load_tokenizer, load_model


def check_environment(cfg: DictConfig) -> None:
    component_name = cfg.get("component", "model")
    getattr(sys.modules[__name__], component_name)(cfg)


def model(cfg: DictConfig) -> None:
    for i, model_cfg in enumerate(cfg.models):
        logger.info(model_cfg.name)
        model = load_model(model_cfg, cfg.model_dtype)
        log_line(logger)
        logger.info(model.config.generation_config)
        logger.info(model.backend.generation_config)
        del model


def tokenizer(cfg: DictConfig) -> None:
    for i, model_cfg in enumerate(cfg.models):
        tokenizer = load_tokenizer(model_cfg, bos=True)
        sentence = "On the tip of tht tongue"
        result = tokenizer(sentence)
        segment_spans = result["segment_spans"]
        tokenized_text = [result["tokenized_text"][s[0]: s[1]] for s in segment_spans]
        input_ids = [result["input_ids"][s[0]: s[1]] for s in segment_spans]
        logger.info('')
        logger.info(model_cfg.name)
        logger.info(sentence)
        logger.info(tokenized_text)
        logger.info(input_ids)
        logger.info(segment_spans)