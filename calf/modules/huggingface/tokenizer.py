import itertools
from typing import List, Union, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AddedToken
from calf import HUGGINGFACE


class HuggingfaceTokenizer:

    def __init__(
            self,
            model_name: str,
            revision: str = None,
            bos: bool = False,
            eos: bool = False,
            trust_remote_code: bool = False,
            **kwargs
    ) -> None:
        try:
            self.backend = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                cache_dir=str(HUGGINGFACE / "hub"),
                revision=revision,
                trust_remote_code=trust_remote_code,
                legacy=False,
                **kwargs
            )
        except Exception:
            self.backend = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                cache_dir=str(HUGGINGFACE / "hub"),
                revision=revision,
                trust_remote_code=trust_remote_code,
                legacy=False,
                **kwargs
            )
        self.name = model_name.replace('/', '--')
        self.revision = revision
        self.bos = bos
        self.eos = eos

    def __len__(self):
        return len(self.backend)

    def __call__(self, text: Union[str, List[str]]) -> Dict:
        """
        Tokenize the text.
        Args:
            text (`List[str]`):
                text is a list of segments to be analyzed.
        Returns:
            The tokenized result.
        """
        # unified text format
        text = [text] if isinstance(text, str) else text

        # tokenize text into pieces
        tokenized_text = [self.tokenize(w) for w in text]

        # add bos and eos if required
        if self.bos:
            tokenized_text = [[self.bos_token]] + tokenized_text
        if self.eos:
            tokenized_text = tokenized_text + [[self.eos_token]]

        # length (number of tokens)
        num_tokens = sum([len(w) for w in tokenized_text])

        # segment spans
        left_ids = np.cumsum([0] + [len(w) for w in tokenized_text])
        segment_spans = tuple(zip(left_ids[0:-1], left_ids[1:]))

        # flatten tokenized_text
        tokenized_text = list(
            itertools.chain.from_iterable([w for w in tokenized_text])
        )

        # input_ids
        input_ids = torch.tensor(
            self.backend.convert_tokens_to_ids(tokenized_text)
        ).long()

        # attention mask
        attention_mask = torch.ones(input_ids.size()).long()

        # special token mask
        special_token_mask = torch.zeros(input_ids.size()).long()
        if self.bos:
            special_token_mask[0] = 1
        if self.eos:
            special_token_mask[-1] = 1

        tokenize_result = {
            "tokenized_text": tokenized_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_token_mask": special_token_mask,
            "segment_spans": segment_spans,
            "num_tokens": num_tokens
        }
        return tokenize_result

    def add_special_tokens(
            self,
            special_tokens_dict: Dict[str, Union[str, AddedToken]]
    ) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder.
        If special tokens are NOT in the vocabulary, they are added to it (
        indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict (`Dict`):
                The dictionary containing special tokens. Keys should be in the
                list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``,
                ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
        Returns:
            `int`: Number of tokens added to the vocabulary.
        """
        return self.backend.add_special_tokens(special_tokens_dict)

    @property
    def pad_token(self):
        return self.backend.pad_token

    @property
    def pad_token_id(self):
        return self.backend.pad_token_id

    @property
    def unk_token(self):
        return self.backend.unk_token

    @property
    def unk_token_id(self):
        return self.backend.unk_token_id

    @property
    def bos_token(self):
        if self.backend._bos_token is not None:  # noqa
            return self.backend.bos_token
        else:
            return self.backend.cls_token

    @property
    def bos_token_id(self):
        if self.backend._bos_token is not None:  # noqa
            return self.backend.bos_token_id
        else:
            return self.backend.cls_token_id

    @property
    def eos_token(self):
        if self.backend._eos_token is not None:  # noqa
            return self.backend.eos_token
        else:
            return self.backend.sep_token

    @property
    def eos_token_id(self):
        if self.backend._eos_token is not None:  # noqa
            return self.backend.eos_token_id
        else:
            return self.backend.sep_token_id

    def encode(self, text: str, **kwargs) -> List:
        return self.backend.encode(text, **kwargs)

    def decode(self, ids: List, **kwargs) -> str:
        text = self.backend.decode(ids, **kwargs)
        return text

    def tokenize(self, text: str, **kwargs) -> List:
        return self.backend.tokenize(text, **kwargs)


def get_tokenizer(
        model_name: str,
        bos: bool = False,
        eos: bool = False,
        revision: str = None,
        trust_remote_code: bool = False,
        **kwargs
) -> HuggingfaceTokenizer:
    tokenizer = HuggingfaceTokenizer(
        model_name=model_name,
        bos=bos,
        eos=eos,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    return tokenizer