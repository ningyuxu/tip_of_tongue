from omegaconf import DictConfig
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from calf import device
from calf.modules import get_model, HuggingfaceModel, get_tokenizer, HuggingfaceTokenizer


def get_model_and_tokenizer(model_cfg: DictConfig, model_dtype: str, bos: bool = False) -> Tuple:
    model = load_model(model_cfg, model_dtype)
    tokenizer = load_tokenizer(model_cfg, bos)
    return model, tokenizer


def load_tokenizer(model_cfg: DictConfig, bos: bool = False) -> HuggingfaceTokenizer:
    """ Download and cache huggingface model """
    model_name = model_cfg.name
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    revision = model_cfg.get("revision", None)
    tokenizer = get_tokenizer(
        model_name=model_name,
        bos=bos,
        revision=revision,
        trust_remote_code=trust_remote_code
    )
    if bos and tokenizer.bos_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
        else:
            tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    return tokenizer


def load_model(model_cfg: DictConfig, model_dtype: str) -> HuggingfaceModel:
    """ Download and cache huggingface model """
    model_name = model_cfg.name
    revision = model_cfg.get("revision", None)
    n_context = model_cfg.get("n_context", 2048)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    model = get_model(
        model_name=model_name,
        revision=revision,
        model_dtype=model_dtype,
        n_context=n_context,
        trust_remote_code=trust_remote_code
    )
    model.config.update_generation_config()
    return model


def prompt_inference(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt_text: List,
        layer: int = -1,
        max_new_tokens: int = 1,
) -> Tuple:
    tokenize_result = tokenizer(prompt_text)
    input_ids = torch.stack([tokenize_result["input_ids"]]).to(device)
    segment_spans = tokenize_result["segment_spans"]

    if len(input_ids[0]) + max_new_tokens >= model.max_len:
        gen_tokens = None
        input_embeddings = None
        gen_embeddings = None
    else:
        with torch.no_grad():
            model.config.update_generation_config(max_new_tokens=max_new_tokens)
            output = model.backend.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.backend.eos_token_id,
                generation_config=model.config.generation_config
            )

            embeddings = output.hidden_states[0][layer][0].cpu().detach().numpy()
            input_embeddings = [embeddings[s[0]: s[1]] for s in segment_spans]

            gen_token_ids = output.sequences[0, len(input_ids[0]):]
            gen_token_ids = gen_token_ids.cpu().detach().numpy()
            gen_tokens = tokenizer.decode(gen_token_ids)

            gen_embeddings = [embeddings[-1]]  # first generate embedding
            gen_embeddings.extend([
                hs[layer][0].cpu().detach().numpy()[0]
                for hs in output.hidden_states[1:]
            ])
            gen_embeddings = np.array(gen_embeddings)
    return gen_tokens, input_embeddings, gen_embeddings


def answer_probability(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt_text: str,
        answer_text: Union[str, List],
) -> List:
    if type(answer_text) == str:
        prompt_answer = [prompt_text, answer_text]
    else:  # answer_text is list
        prompt_answer = [prompt_text] + answer_text

    tokenize_result = tokenizer(prompt_answer)
    input_ids = torch.stack([tokenize_result["input_ids"]]).to(device)
    segment_spans = [
        (s[0]-1 if s[0] > 1 else 0, s[1]-1 if s[1] > 1 else 0) for s in tokenize_result["segment_spans"]
    ]
    with torch.no_grad():
        output = model.backend(input_ids)
        probs = torch.log_softmax(output.logits[0], dim=-1).detach()
        probs = probs[:-1, :]
        inids = input_ids[0][1:].detach()
        answer_probs = torch.gather(probs, 1, inids[:, None]).squeeze(-1).cpu().numpy()
    probabilities = [answer_probs[s[0]: s[1]] for s in segment_spans][1:]
    return probabilities


def answer_probability_for_blimp(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        sentence: str,
) -> List:
    probs = answer_probability(
        model=model,
        tokenizer=tokenizer,
        prompt_text='',
        answer_text=sentence
    )
    probabilities = probs[1]
    return probabilities


def answer_probability_for_sgeval(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        sentence: str,
) -> List:
    words = sentence.split()
    probabilities = []
    with torch.no_grad():
        text = ''
        length = 0
        for word in words:
            text = f"{text} {word}" if text else word
            probs = answer_probability(
                model=model,
                tokenizer=tokenizer,
                prompt_text='',
                answer_text=text
            )
            probabilities.append(probs[1][length:])
            length = len(probs[1])
    return probabilities


def answer_probability_for_commqa(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt_text: str,
        answer_text: str,
) -> List:
    tokenize_prompt = tokenizer(prompt_text)
    prompt_length = len(tokenize_prompt["input_ids"])
    prompt_answer = f"{prompt_text} {answer_text}"
    probs = answer_probability(
        model=model,
        tokenizer=tokenizer,
        prompt_text='',
        answer_text=prompt_answer
    )
    probabilities = probs[0][prompt_length-1:]
    # answer_tokens = tokenizer(prompt_answer)["tokenized_text"][prompt_length:]
    # print()
    # print(prompt_answer)
    # print(answer_tokens)
    # l = np.array([float(len(t)) for t in answer_tokens])
    # print(l)
    # print(probabilities, np.sum(probabilities), np.mean(probabilities), np.sum(probabilities)/np.sum(l))
    return probabilities


def top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float("Inf")
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Nucleus filtering is described in Holtzman et al. (https://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p  # noqa
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt_text: str,
        num_samples: int = 1,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 1,
):
    context_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    length = len(context_tokens)
    if length + max_new_tokens >= model.max_len:
        gen_tokens = None
    else:
        context = torch.tensor(context_tokens, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model.backend(input_ids=generated)
                next_token_logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.)

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(num_samples):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= repetition_penalty

                filtered_logits = top_k_top_p_filtering(
                    logits=next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    softmax_logits = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(softmax_logits, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        gen_token_ids = generated[0, length:].cpu().detach().numpy()
        gen_tokens = tokenizer.decode(gen_token_ids)
    return gen_tokens


def sample_sequence_old(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt_text: List,
        greedy: bool = False,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 1,
) -> str:
    tokenize_result = tokenizer(prompt_text)
    input_ids = torch.stack([tokenize_result["input_ids"]]).to(device)
    if len(input_ids[0]) + max_new_tokens >= model.max_len:
        gen_tokens = None
    else:
        with torch.no_grad():
            if greedy:
                model.config.update_generation_config(max_new_tokens=max_new_tokens)
            else:
                model.config.update_generation_config(
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )
            output = model.backend.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.backend.eos_token_id,
                generation_config=model.config.generation_config
            )
            gen_token_ids = output.sequences[0, len(input_ids[0]):]
            gen_token_ids = gen_token_ids.cpu().detach().numpy()
            gen_tokens = tokenizer.decode(gen_token_ids)
    return gen_tokens
