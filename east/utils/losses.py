from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class LossReduction:
    """Alias for loss reduction"""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if reduction == LossReduction.MEAN:
        loss = torch.mean(loss)
    elif reduction == LossReduction.SUM:
        loss = torch.sum(loss)
    elif reduction == LossReduction.NONE:
        return loss

    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "none",
    eps: float = 1e-5,
) -> torch.Tensor:
    inputs = F.softmax(inputs, dim=1)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    if inputs.shape != targets.shape:
        raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")

    # flatten prediction and label tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = torch.sum(inputs * targets)
    denominator = torch.sum(inputs) + torch.sum(targets)

    # calculate the dice loss
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1 - dice_score

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(inputs)
    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss


class DiceLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: Optional[float] = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * dice_loss(inputs, targets, weight=weight, reduction=self.reduction, eps=self.eps)

        return loss
