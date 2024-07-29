from collections import OrderedDict
from typing import Dict, List, Type, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.config.nat_pn.scaler import EvidenceScaler
from src.config.flows.utils_flow import initialize_radial_flow, initialize_realnvp_flow
from pyro.distributions.transforms import ComposeTransformModule
import src.config.nat_pn.distributions as D
from src.config.nat_pn.distributions import Posterior
from src.config.flows.utils_flow import process_flow_batch, process_gaussian_batch, GaussianFlow
from src.config.nat_pn.output.categorical import CategoricalOutput
from copy import deepcopy
import numpy as np


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.density_model: Optional[Union[ComposeTransformModule,
                                           GaussianFlow]] = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        self.need_all_features_flag = True
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            self.all_features.append(output)

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: torch.Tensor):
        out = self.classifier(F.relu(self.base(x)))
        if self.need_all_features_flag:
            self.all_features = []
        return out

    def get_final_features(self, x: torch.Tensor, detach=True):
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: torch.Tensor, detach=True):
        feature_list = None
        if self.need_all_features_flag:
            if len(self.dropout) > 0:
                for dropout in self.dropout:
                    dropout.eval()

            func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
            _ = self.base(x)
            feature_list = [func(feature) for feature in self.all_features]
            self.all_features = []

            if len(self.dropout) > 0:
                for dropout in self.dropout:
                    dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "medmnistS": (1, 1024, 11),
            "medmnistC": (1, 1024, 11),
            "medmnistA": (1, 1024, 11),
            "covid19": (3, 196736, 4),
            "fmnist": (1, 1024, 10),
            "emnist": (1, 1024, 62),
            "femnist": (1, 1, 62),
            "cifar10": (3, 1600, 10),
            "cinic10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
            "tiny_imagenet": (3, 3200, 200),
            "celeba": (3, 133824, 2),
            "svhn": (3, 1600, 10),
            "usps": (1, 800, 10),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 512),
            )
        )
        self.classifier = nn.Linear(512, config[dataset][2])


class LeNet5(DecoupledModel):
    def __init__(self, dataset: str, embedding_dim: int) -> None:
        super(LeNet5, self).__init__()
        config = {
            "mnist": (1, 256, 10),
            "noisy_mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "cinic10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "noisy_cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 6, 5),
                # bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                # bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, embedding_dim),
            )
        )

        self.classifier = nn.Linear(embedding_dim, config[dataset][2])


class TwoNN(DecoupledModel):
    def __init__(self, dataset):
        super(TwoNN, self).__init__()
        config = {
            "mnist": (784, 10),
            "medmnistS": (784, 11),
            "medmnistC": (784, 11),
            "medmnistA": (784, 11),
            "fmnist": (784, 10),
            "emnist": (784, 62),
            "femnist": (784, 62),
            "cifar10": (3072, 10),
            "cinic10": (3072, 10),
            "svhn": (3072, 10),
            "cifar100": (3072, 100),
            "usps": (1536, 10),
            "synthetic": (60, 10),  # default dimension and classes
            "toy_circle": (2, 10),
            "toy_noisy_3": (2, 3),
            "toy_noisy_5": (2, 5),
            "toy_noisy_10": (2, 10),
            "toy_noisy_15": (2, 15),
            "toy_noisy_20": (2, 20),
            "toy_noisy_30": (2, 30),
            "toy_noisy_40": (2, 40),
            "toy_noisy_50": (2, 50),
            "toy_noisy_75": (2, 75),
            "toy_noisy_100": (2, 100),
            "toy_noisy_150": (2, 150),
            "toy_noisy_200": (2, 200),
        }

        self.base = nn.Linear(config[dataset][0], 200)
        self.classifier = nn.Linear(200, config[dataset][1])
        self.activation = nn.ReLU()

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.base(x))
        x = self.classifier(x)
        return x

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x, detach=True):
        raise RuntimeError(
            "2NN has 0 Conv layer, so is unable to get all features.")


class MobileNetV2(DecoupledModel):
    def __init__(self, dataset):
        super(MobileNetV2, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[1].in_features, config[dataset]
        )

        self.base.classifier[1] = nn.Identity()


class ResNet18(DecoupledModel):
    def __init__(self, dataset, embedding_dim):
        super(ResNet18, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "noisy_cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.base.fc = nn.Linear(
            in_features=self.base.fc.in_features, out_features=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, config[dataset])
        # self.base.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x, detach)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        # NOTE: AlexNet does not support datasets with data size smaller than (64 x 64)
        config = {"covid19": 4, "celeba": 2, "tiny_imagenet": 200}

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, config[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


def marginalize_log_prob(
    local_embeddings: torch.Tensor,
        density_model: Union[nn.ModuleList, GaussianFlow],
        labels: torch.Tensor,
        labels_frequency: torch.Tensor,
) -> torch.Tensor:
    # Default behavior without labels, e.g., compute embeddings
    # Example: return self.embedding_layer(x)
    log_probs = torch.tensor(
        [], device=local_embeddings.device, dtype=torch.float32)

    for i, density_model_index in enumerate(labels):
        if isinstance(density_model, GaussianFlow):
            log_prob = process_gaussian_batch(gaussian=density_model, index=density_model_index,
                                              batch_embeddings=local_embeddings)
        else:
            log_prob = process_flow_batch(local_flow=density_model[density_model_index],
                                          batch_embeddings=local_embeddings)
        log_p_c = torch.log(labels_frequency[i]).to(
            local_embeddings.device)
        log_probs = torch.cat([log_probs, log_prob[None] + log_p_c], dim=0)
    log_probs = torch.logsumexp(log_probs, dim=0)
    return log_probs


def class_specific_log_prob(
        local_embeddings: torch.Tensor,
        current_labels: torch.Tensor,
        all_labels: torch.Tensor,
        density_model: nn.ModuleList,
):
    log_probs = torch.zeros_like(local_embeddings[:, 0])
    for i in all_labels:
        # Select samples with the current label
        label_indices = (current_labels == i).nonzero(as_tuple=True)[0]
        if label_indices.nelement() > 0:
            embeddings_for_label = local_embeddings[label_indices]
            if isinstance(density_model, GaussianFlow):
                log_prob = process_gaussian_batch(gaussian=density_model, index=i,
                                                batch_embeddings=embeddings_for_label)
            else:
                # Apply the corresponding flow
                log_prob = process_flow_batch(local_flow=density_model[i],
                                            batch_embeddings=embeddings_for_label)  # NOT DETACH HERE
            # Assign the transformed values to the output tensor
            log_probs[label_indices] = log_prob
    return log_probs


class NatPnModel(DecoupledModel):
    def __init__(
        self, dataset,
            backbone: str,
            stop_grad_logp: bool,
            stop_grad_embeddings: bool,
            embedding_dim: int,
            density_model_type: str,
            labels: Optional[torch.Tensor] = None,
            labels_frequency: Optional[torch.Tensor] = None,
    ):
        super(NatPnModel, self).__init__()
        self.embedding_dim = embedding_dim
        aux_model = MODEL_DICT[backbone](
            dataset=dataset, embedding_dim=self.embedding_dim)
        embeddings_dim = aux_model.classifier.in_features
        n_classes = aux_model.classifier.out_features
        if backbone != '2nn':
            self.base = deepcopy(aux_model.base)
        else:
            self.base = nn.Identity()
            embeddings_dim = 2

        self.stop_grad_logp = stop_grad_logp
        self.stop_grad_embeddings = stop_grad_embeddings
        self.classifier = CategoricalOutput(
            dim=embeddings_dim, num_classes=n_classes)
        self.scaler = EvidenceScaler(dim=embeddings_dim, budget="normal")
        self.register_buffer('labels',
                             labels if labels is not None else None)
        self.register_buffer('labels_frequency',
                             labels_frequency if labels_frequency is not None else None)

        if density_model_type == 'flow':
            self.density_model = nn.ModuleList(
                [initialize_radial_flow(input_dim=embeddings_dim, n_transforms=30) for _ in range(n_classes)])
        elif density_model_type == 'gaussian':
            self.density_model = GaussianFlow(
                dim=self.embedding_dim, n_classes=n_classes)

    def need_all_features(self):
        return

    def train_forward(
        self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            clamp: bool = True
    ) -> tuple[Posterior, torch.Tensor, torch.Tensor]:

        if self.base is not None:
            local_embeddings = self.base(x)
        else:
            local_embeddings = x

        if labels is None:
            log_probs = marginalize_log_prob(
                local_embeddings=local_embeddings,
                density_model=self.density_model,
                labels=self.labels,
                labels_frequency=self.labels_frequency,
            )
        else:
            log_probs = class_specific_log_prob(
                local_embeddings=local_embeddings, # try detach here?
                current_labels=labels,
                all_labels=self.labels,
                density_model=self.density_model,
            )

        if not clamp:
            log_prob_not_clamped = self.scaler.forward(
                log_probs.clone(), clamp_log_prob_value=False)

        log_prob_processed = self.scaler.forward(
            log_probs, clamp_log_prob_value=True)

        prediction = self.classifier(local_embeddings)
        sufficient_statistics = prediction.expected_sufficient_statistics()

        if self.stop_grad_logp:
            for p in self.density_model.parameters():
                p.requires_grad_(False)

            log_prob_processed_detached = class_specific_log_prob(
                local_embeddings=local_embeddings,
                current_labels=labels,
                all_labels=self.labels,
                density_model=self.density_model,
            )
            log_prob_processed_detached = self.scaler.forward(
                log_prob_processed_detached, clamp_log_prob_value=True)

            update = D.PosteriorUpdate(sufficient_statistics=sufficient_statistics,
                                       log_evidence=log_prob_processed_detached)

            for p in self.density_model.parameters():
                p.requires_grad_(True)
        else:
            update = D.PosteriorUpdate(sufficient_statistics=sufficient_statistics,
                                       log_evidence=log_prob_processed)

        y_pred = self.classifier.prior.update(update)

        if not clamp:
            log_prob_processed = log_prob_not_clamped

        return y_pred, log_prob_processed, local_embeddings

    def forward(self, x):
        return self.train_forward(x)[0].alpha.log()

    def get_all_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x, detach)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


class NatPnModelVanilla(DecoupledModel):
    def __init__(
        self, dataset,
            backbone: str,
            stop_grad_logp: bool,
            stop_grad_embeddings: bool,

    ):
        super(NatPnModelVanilla, self).__init__()

        aux_model = MODEL_DICT[backbone](dataset=dataset)
        embeddings_dim = aux_model.classifier.in_features
        n_classes = aux_model.classifier.out_features
        if backbone != '2nn':
            self.base = deepcopy(aux_model.base)
        else:
            self.base = nn.Identity()
            embeddings_dim = 2
        self.stop_grad_logp = stop_grad_logp
        self.stop_grad_embeddings = stop_grad_embeddings
        self.classifier = CategoricalOutput(
            dim=embeddings_dim, num_classes=n_classes)
        self.scaler = EvidenceScaler(dim=embeddings_dim, budget="normal")

        self.density_model = initialize_radial_flow(
            input_dim=embeddings_dim, n_transforms=30)

    def need_all_features(self):
        return

    def train_forward(
        self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            clamp: bool = True
    ) -> tuple[Posterior, torch.Tensor, torch.Tensor]:

        if self.base is not None:
            local_embeddings = self.base(x)
        else:
            local_embeddings = x

        log_probs = process_flow_batch(local_flow=self.density_model,
                                       batch_embeddings=local_embeddings)

        if not clamp:
            log_prob_not_clamped = self.scaler.forward(
                log_probs.clone(), clamp_log_prob_value=False)

        log_prob_processed = self.scaler.forward(
            log_probs, clamp_log_prob_value=True)

        prediction = self.classifier(local_embeddings)
        sufficient_statistics = prediction.expected_sufficient_statistics()

        if self.stop_grad_logp:
            update = D.PosteriorUpdate(sufficient_statistics=sufficient_statistics,
                                       log_evidence=log_prob_processed.detach())
        else:
            update = D.PosteriorUpdate(sufficient_statistics=sufficient_statistics,
                                       log_evidence=log_prob_processed)

        y_pred = self.classifier.prior.update(update)

        if not clamp:
            log_prob_processed = log_prob_not_clamped

        return y_pred, log_prob_processed, local_embeddings

    def forward(self, x):
        return self.train_forward(x)[0].alpha.log()

    def get_all_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x, detach)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


MODEL_DICT: Dict[str, Type[DecoupledModel]] = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "2nn": TwoNN,
    "mobile": MobileNetV2,
    "res18": ResNet18,
    "alex": AlexNet,
    "natpn": NatPnModel,
    "natpn_vanilla": NatPnModelVanilla,
}
