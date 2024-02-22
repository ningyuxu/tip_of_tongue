from typing import Iterable
import torch


class ScalarMix(torch.nn.Module):
    """
    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float, torch.nn.Dropout):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjusts its softmax weight mass to 0 with
            the dropout probability (i.e., setting the unnormalized weight to -inf). This
            effectively redistributes the dropped probability mass to all other weights. Default: 0.
    """

    def __init__(self, n_layers: int, dropout: float = .0) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.weights = torch.nn.Parameter(torch.zeros(n_layers))
        self.gamma = torch.nn.Parameter(torch.tensor([1.0]))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tensors (Iterable[~torch.Tensor]):
                :math:`N` tensors to be mixed.
        Returns:
            The mixture of :math:`N` tensors.
        """
        return self.gamma * sum(
            w * h for w, h in zip(self.dropout(self.weights.softmax(-1)), tensors)
        )
