import torch
import torch.nn as nn


class Actor(nn.Module):
    """Predicts (mu, sigma) of the next realized volatility given inputs.

    Input: [u(k), s(k-1), s(k-2)]
        u(k) - position just chosen by the controller
        s(k-1), s(k-2) - recent realized volatilities

    Output: (mu, sigma), each a scalar.
        mu is the predicted next-bar realized volatility.
        sigma is enforced positive via softplus so a normal distribution
            can be sampled from it.

    Args:
        hidden_sizes: sizes of hidden layers, sigmoid activation per the paper.
        sigma_floor: added to softplus output so sigma never collapses to zero.
    """

    INPUT_SIZE = 3

    def __init__(self, hidden_sizes: tuple[int, ...] = (32, 32), sigma_floor: float = 1e-3):
        super().__init__()
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")

        layers: list[nn.Module] = []
        prev = self.INPUT_SIZE
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Sigmoid())
            prev = size
        self.trunk = nn.Sequential(*layers)
        self.head_mu = nn.Linear(prev, 1)
        self.head_sigma = nn.Linear(prev, 1)
        self.sigma_floor = sigma_floor

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.01, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        mu = self.head_mu(features).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.head_sigma(features)).squeeze(-1) + self.sigma_floor
        return mu, sigma
