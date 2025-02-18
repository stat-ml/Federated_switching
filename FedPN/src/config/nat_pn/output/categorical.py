import torch
from torch import nn
import src.config.nat_pn.distributions as D
from ._base import Output


class CategoricalOutput(Output):
    """
    Categorical output with uniformative Dirichlet prior.
    """

    def __init__(self, dim: int, num_classes: int):
        """
        Args:
            dim: The dimension of the latent space.
            num_classes: The number of categories for the output distribution.
        """
        super().__init__()
        # self.linear = nn.Linear(dim, num_classes)
        self.linear = nn.Sequential(
            nn.Linear(dim, 3 * dim),
            nn.ReLU(),
            nn.Linear(3 * dim, num_classes),
            )
        # self.linear = nn.Sequential(*make_list_n_layers(input_dim=dim,
        #                                                 hidden_dim=2 * dim,
        #                                                 n_hidden_layers=2,
        #                                                 output_dim=num_classes))
        self.prior = D.DirichletPrior(
            num_categories=num_classes, evidence=num_classes)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        z = self.linear.forward(x)
        return D.Categorical(z.log_softmax(-1))
