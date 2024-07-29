from typing import Literal, Optional

import torch
from torch import nn

from src.config.nat_pn.distributions import Posterior

Reduction = Literal["mean", "sum", "none"]


class BayesianLoss(nn.Module):
    """
    The Bayesian loss computes an uncertainty-aware loss based on the parameters of a conjugate
    prior of the target distribution.
    """

    def __init__(self,
                 entropy_weight: float,
                 log_prob_weight: float,
                 reduction: Reduction = "mean",
                 ):
        """
        Args:
            entropy_weight: The weight for the entropy regularizer.
            log_prob_weight: The weight for the log_prob regularizer.
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.entropy_weight = float(entropy_weight)
        self.log_prob_weight = float(log_prob_weight)
        self.reduction = reduction

    def forward(self, y_pred: Posterior, y_true: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            y_true: The true targets. Either indices for classes of a classification problem or
                the real target values. Must have the same batch shape as ``y_pred``.
            log_prob: The logarithm of the density of embeddings.

        Returns:
            The loss, processed according to ``self.reduction``.
        """
        nll = -y_pred.expected_log_likelihood(y_true)
        loss = nll
        if self.log_prob_weight != 0:
            log_prob_component = -self.log_prob_weight * log_prob
            loss = loss + log_prob_component
        if self.entropy_weight != 0:
            entropy_component = -self.entropy_weight * y_pred.entropy()
            loss += entropy_component

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LogMarginalLoss(nn.Module):
    def __init__(self, entropy_weight: float = 0.0, reduction: Reduction = "mean"):
        """
        Args:
            entropy_weight: The weight for the entropy regulaarizer.
            log_prob_weight: The weight for the log_prob regulaarizer.
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.entropy_weight = float(entropy_weight)
        self.reduction = reduction

    def forward(self, y_pred: Posterior, y_true: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            y_true: The true targets. Either indices for classes of a classification problem or
                the real target values. Must have the same batch shape as ``y_pred``.
            log_prob: The logarithm of the density of embeddings.

        Returns:
            The loss, processed according to ``self.reduction``.
        """
        a0 = y_pred.alpha.sum(-1)
        a_true = y_pred.alpha.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)
        nll = torch.log(a0) - torch.log(a_true)
        loss = nll
        if self.log_prob_weight > 0:
            log_prob_component = -self.log_prob_weight * log_prob
            loss += log_prob_component
        if self.entropy_weight > 0:
            entropy_component = -self.entropy_weight * y_pred.entropy()
            loss += entropy_component

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BrierLoss(nn.Module):
    def __init__(self, entropy_weight: float = 0.0, reduction: Reduction = "mean", log_prob_weight: float = 0.0):
        """
        Args:
            entropy_weight: The weight for the entropy regulaarizer.
            log_prob_weight: The weight for the log_prob regulaarizer.
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.entropy_weight = float(entropy_weight)
        self.log_prob_weight = float(log_prob_weight)
        self.reduction = reduction

    def forward(self, y_pred: Posterior, y_true: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            y_true: The true targets. Either indices for classes of a classification problem or
                the real target values. Must have the same batch shape as ``y_pred``.

        Returns:
            The loss, processed according to ``self.reduction``.
        """
        a0 = y_pred.alpha.sum(-1)
        a_true = y_pred.alpha.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)

        if self.log_prob_weight > 0:
            log_prob_component = -self.log_prob_weight * log_prob
        else:
            log_prob_component = torch.zeros_like(a0)

        sum_expected_square = torch.sum(
            y_pred.alpha * (y_pred.alpha + 1), dim=-1) / (a0 * (a0 + 1))
        brier_loss = 1 - 2 * (a_true / a0) + sum_expected_square
        loss = brier_loss + log_prob_component - self.entropy_weight * y_pred.entropy()

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def contrastive_loss(embeddings, labels, margin=5.0):
    """
    :param embeddings: Embeddings produced by the encoder
    :param labels: Ground truth labels for the embeddings
    :param margin: Margin for different class pairs
    :return: Contrastive loss for each instance in the batch
    """
    batch_size = embeddings.size(0)

    # Compute the pairwise distances between all embeddings in the batch
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

    # Compute the masks for same and different classes
    same_class_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    different_class_mask = 1 - same_class_mask

    # Compute the loss for same class pairs
    same_class_loss = torch.sum(
        same_class_mask * torch.pow(pairwise_distances, 2), dim=1)

    # Compute the loss for different class pairs
    different_class_loss = torch.sum(different_class_mask * torch.pow(
        torch.clamp(margin - pairwise_distances, min=0.0), 2), dim=1)

    # Calculate the final loss
    loss = same_class_loss + different_class_loss

    return loss / (batch_size - 1)
