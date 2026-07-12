import torch
import torch.nn as nn


class Critic(nn.Module):
    """Scalar state-value network.

    Estimates V(state), the expected cumulative reward from the given inputs.
    Used during training to compute the TD error that drives updates of both
    the actor (state-predictor) and the pid_net (controller gains).

    Input: [u(k), s(k-1), s(k-2)] -- same shape as Actor.
    Output: scalar V(state).

    Args:
        hidden_sizes: sizes of hidden layers, sigmoid activation per the paper.
    """

    INPUT_SIZE = 3

    def __init__(self, hidden_sizes: tuple[int, ...] = (32, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = self.INPUT_SIZE
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Sigmoid())
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.01, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
