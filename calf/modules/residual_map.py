import torch
from .mlp import MLP


class ResidualMap(torch.nn.Module):
    def __init__(self,
                 n_in: int,
                 n_out: int = None,
                 hidden_size: int = 256,
                 dropout: float = 0.33) -> None:
        super().__init__()

        self.n_in = n_in
        self.n_out = n_in if n_out is None else n_out
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.mlp = MLP(n_in, hidden_size, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, n_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.mlp(x)
        r = self.fc(r)
        return r + x
