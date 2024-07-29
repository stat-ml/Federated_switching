import torch
from torch import nn
import src.config.nat_pn.distributions as D
from ._base import Output
from copy import deepcopy


class CategoricalOutput(Output):
    """
    Categorical output with uniformative Dirichlet prior.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            dim: The dimension of the latent space.
            num_classes: The number of categories for the output distribution.
        """
        super().__init__()
        self.linear = deepcopy(model)
        num_classes = self.linear.out_features
        self.prior = D.DirichletPrior(
            num_categories=num_classes, evidence=num_classes)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        z = self.linear.forward(x)
        return D.Categorical(z.log_softmax(-1))
