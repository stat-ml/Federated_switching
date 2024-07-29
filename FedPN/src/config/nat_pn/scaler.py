import math
from typing import Literal

import torch
from torch import nn

from src.config.nat_pn.utils import clamp_preserve_gradients

CertaintyBudget = Literal["constant", "exp-half", "exp", "normal"]
"""
The certainty budget to distribute in the latent space of dimension ``H``:

- ``constant``: A certainty budget of 1, independent of the latent space's dimension.
- ``exp-half``: A certainty budget of ``exp(0.5 * H)``.
- ``exp``: A certainty budget of ``exp(H)``.
- ``normal``: A certainty budget that causes a multivariate normal distribution to yield the same
  probability at the origin at any dimension: ``exp(0.5 * log(4 * pi) * H)``.
"""


class EvidenceScaler(nn.Module):
    """
    Scaler for the evidence to distribute a certainty budget other than one in the latent space.
    """

    def __init__(self, dim: int, budget: CertaintyBudget):
        """
        Args:
            dim: The dimension of the latent space.
            budget: The budget to use.
        """
        super().__init__()
        if budget == "exp-half":
            self.log_scale = 0.5 * dim
        elif budget == "exp":
            self.log_scale = dim
        elif budget == "normal":
            self.log_scale = 0.5 * math.log(4 * math.pi) * dim
        else:
            self.log_scale = 0

    def forward(self, log_evidence: torch.Tensor, clamp_log_prob_value: bool = True) -> torch.Tensor:
        """
        Scales the evidence in the log space according to the certainty budget.

        Args:
            log_evidence: The log-evidence to scale.
            clamp_log_prob_value: If true, we clamp value of the result.

        Returns:
            The scaled and clamped evidence in the log-space.
        """
        if clamp_log_prob_value:
            return clamp_preserve_gradients(log_evidence + self.log_scale, lower=-50.0, upper=50.0)
        else:
            return log_evidence + self.log_scale
