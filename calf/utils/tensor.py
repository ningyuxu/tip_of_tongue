from typing import List
import torch


def pad(tensors: List[torch.Tensor],
        padding_value: int = 0,
        total_length: int = None,
        padding_side: str = "right") -> torch.Tensor:
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == "left" else slice(0, i)
                       for i in tensor.size()]] = tensor
    return out_tensor
