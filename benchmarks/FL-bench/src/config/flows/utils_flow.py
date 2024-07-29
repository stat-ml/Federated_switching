from typing import Optional, Union

import torch
from pyro.distributions.transforms import ComposeTransformModule, Radial
import math


def initialize_radial_flow(input_dim: int = 2, n_transforms: int = 2,
                           device: Optional[Union[str, torch.device]] = 'cpu') -> ComposeTransformModule:
    """
    The function creates stack of Radial flows for a given number of transformations
    :param input_dim:
    :param n_transforms:
    :param device:
    :return:
    """
    list_of_transforms = []
    for _ in range(n_transforms):
        list_of_transforms.append(Radial(input_dim=input_dim).to(device))
    composed_list = ComposeTransformModule(list_of_transforms)
    return composed_list


def process_flow_batch(
        local_flow: ComposeTransformModule,
        batch_embeddings: torch.Tensor) -> torch.Tensor:

    neutralized_embeddings = local_flow(batch_embeddings)

    # Compute log-probability
    const = neutralized_embeddings.shape[1] * math.log(2 * math.pi)
    norm = torch.einsum("...ij,...ij->...i",
                        neutralized_embeddings, neutralized_embeddings)
    normal_log_prob = -0.5 * (const + norm)

    log_prob = normal_log_prob + local_flow.log_abs_det_jacobian(
        batch_embeddings, neutralized_embeddings)
    return log_prob
