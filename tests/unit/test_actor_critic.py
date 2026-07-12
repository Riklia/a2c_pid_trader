import pytest
import torch

from models.actor import Actor
from models.critic import Critic


def test_actor_output_shapes_are_scalar_per_sample():
    actor = Actor()
    x = torch.zeros(actor.INPUT_SIZE)
    mu, sigma = actor(x)
    assert mu.shape == torch.Size([])
    assert sigma.shape == torch.Size([])


def test_actor_batched_output_shapes():
    actor = Actor()
    batch = torch.zeros(5, actor.INPUT_SIZE)
    mu, sigma = actor(batch)
    assert mu.shape == (5,)
    assert sigma.shape == (5,)


def test_actor_sigma_is_always_positive():
    actor = Actor(sigma_floor=1e-3)
    for x in [torch.zeros(3), torch.ones(3) * 10, torch.ones(3) * -10, torch.randn(3)]:
        _, sigma = actor(x)
        assert sigma.item() > 0.0


def test_actor_sigma_stays_above_floor():
    actor = Actor(sigma_floor=0.5)
    for _ in range(20):
        _, sigma = actor(torch.randn(3))
        assert sigma.item() >= 0.5


def test_actor_rejects_nonpositive_sigma_floor():
    with pytest.raises(ValueError, match="sigma_floor must be positive"):
        Actor(sigma_floor=0.0)


def test_actor_supports_sampling_from_predicted_distribution():
    actor = Actor()
    mu, sigma = actor(torch.zeros(3))
    dist = torch.distributions.Normal(mu, sigma)
    sample = dist.rsample()  # rsample keeps gradients for backprop
    assert sample.shape == torch.Size([])


def test_critic_output_is_scalar():
    critic = Critic()
    v = critic(torch.zeros(critic.INPUT_SIZE))
    assert v.shape == torch.Size([])


def test_critic_batched_output_shape():
    critic = Critic()
    batch = torch.zeros(8, critic.INPUT_SIZE)
    v = critic(batch)
    assert v.shape == (8,)


def test_critic_near_zero_initialization():
    critic = Critic()
    v = critic(torch.zeros(critic.INPUT_SIZE))
    assert abs(v.item()) < 0.05


def test_critic_output_supports_backprop():
    critic = Critic()
    x = torch.randn(critic.INPUT_SIZE, requires_grad=False)
    v = critic(x)
    v.backward()
    # gradient should have flowed to at least one parameter
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())