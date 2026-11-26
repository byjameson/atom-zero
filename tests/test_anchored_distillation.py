"""Unit tests for the KL anchor used by the 1.7.0 training harness."""
import pytest
import torch

from research.finetune_copy_control import kl_anchor_loss, train_control


def test_kl_anchor_loss_is_zero_for_identical_distributions():
    logits = torch.randn(2, 4, 9, requires_grad=True)
    labels = torch.tensor([[0, 2, 4, 6], [1, 3, -100, -100]])
    loss = kl_anchor_loss(logits, logits.detach(), labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_anchor_loss_is_positive_and_backpropagates():
    candidate = torch.randn(2, 4, 9, requires_grad=True)
    reference = torch.randn(2, 4, 9)
    labels = torch.tensor([[0, 2, 4, 6], [1, 3, -100, -100]])
    loss = kl_anchor_loss(candidate, reference, labels)
    assert loss.item() > 0
    loss.backward()
    assert candidate.grad is not None
    assert float(candidate.grad.abs().sum()) > 0


def test_kl_anchor_loss_ignores_non_assistant_positions():
    candidate = torch.randn(1, 3, 7, requires_grad=True)
    changed = candidate.detach().clone()
    changed[0, 1, :] += 10.0
    labels = torch.tensor([[0, -100, 2]])
    first = kl_anchor_loss(candidate, candidate.detach(), labels)
    second = kl_anchor_loss(changed, candidate.detach(), labels)
    assert first.item() == pytest.approx(second.item(), abs=1e-6)


def test_kl_anchor_loss_handles_no_assistant_tokens():
    logits = torch.randn(1, 3, 7, requires_grad=True)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    loss = kl_anchor_loss(logits, logits.detach(), labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_anchor_loss_validates_shapes_and_temperature():
    logits = torch.randn(1, 3, 7)
    with pytest.raises(ValueError):
        kl_anchor_loss(logits, torch.randn(1, 4, 7),
                       torch.tensor([[0, 1, 2]]))
    with pytest.raises(ValueError):
        kl_anchor_loss(logits, logits, torch.tensor([[0, 1, 2]]),
                       temperature=0.0)


def test_train_control_rejects_anchor_without_reference():
    with pytest.raises(ValueError, match="anchor_beta > 0"):
        train_control(None, None, [{}], steps=1, lr=0.1,
                      anchor_beta=0.3)
