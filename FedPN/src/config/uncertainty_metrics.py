from scipy.special import digamma
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pickle
from torchvision.transforms import Compose, Normalize
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import torch
from src.config.models import NatPnModel
from copy import deepcopy
from typing import Optional
from tqdm import tqdm


def H(n):
    """Returns an approximate value of n-th harmonic number.

       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def load_dataset(dataset_name: str,
                 normalization_name: Optional[str] = None) -> tuple[list[list[int]], Subset, Subset]:
    if normalization_name is None:
        normalization_name = dataset_name
    with open(f'../data/{dataset_name}/partition.pkl', 'rb') as file:
        partition = pickle.load(file, )
    data_indices: list[list[int]] = partition["data_indices"]
    transform = Compose(
        [Normalize(MEAN[normalization_name], STD[normalization_name])]
    )

    dataset = DATASETS[dataset_name](
        root=f"../data/{dataset_name}",
        args=None,
        transform=transform,
        target_transform=None,
    )

    trainset: Subset = Subset(dataset, indices=[])
    testset: Subset = Subset(dataset, indices=[])

    return data_indices, trainset, testset


def load_dataloaders(
        client_id: int,
        data_indices: list[list[int]],
        trainset: Subset,
        testset: Subset,
        batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    trainset.indices = data_indices[client_id]["train"]
    trainloader = DataLoader(trainset, batch_size, shuffle=True)

    n_val = int(0.6 * len(data_indices[client_id]["test"]))
    np.random.seed(42)
    val_indices = np.random.choice(
        data_indices[client_id]["test"], size=n_val, replace=False, )
    cal_indices = np.array(
        [ind for ind in data_indices[client_id]["test"] if ind not in val_indices])

    testset.indices = val_indices
    testloader = DataLoader(testset, batch_size, shuffle=False)

    testset.indices = cal_indices
    calloader = DataLoader(testset, batch_size, shuffle=False)

    return trainloader, testloader, calloader


def load_model(
        dataset_name: str,
        backbone: str,
        density_model: str,
        embedding_dim: int,
        stopgrad: bool,
        index: int,
        all_params_dict: dict[int, torch.Tensor],
) -> NatPnModel:
    model = NatPnModel(dataset=dataset_name,
                       backbone=backbone,
                       density_model_type=density_model,
                       stop_grad_logp=stopgrad,
                       stop_grad_embeddings=stopgrad,
                       embedding_dim=embedding_dim,
                       )

    model.load_state_dict(all_params_dict[index], strict=False)
    if index == 'global':
        n_classes = len(model.density_model)
        model.labels = torch.arange(n_classes)
        model.labels_frequency = torch.ones(n_classes) / n_classes
    else:
        model.labels = all_params_dict[index]['labels']
        model.labels_frequency = all_params_dict[index]['labels_frequency']

    current_model = deepcopy(model)
    current_model.eval()
    return current_model


@torch.no_grad()
def choose_threshold(
        model: NatPnModel,
        calloader: DataLoader,
        device: str,
        alpha: float = 0.975,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mis = []  # Mutual Informations
    rmis = []  # Reverse Mutual Informations
    epkls = []  # Expected Pairwise Kullback Leiblers divergences
    entropies = []  # Entropies of Dirichlet
    # Expected entropy (expectation over Dirichlet of Categoricals)
    categorical_entropies = []
    log_probs = []  # Logarithms of embedding's density
    for x, y in calloader:
        x, y = x.to(device), y.to(device)
        y_pred, log_prob, local_embeddings = model.train_forward(
            x, clamp=False)
        alphas = y_pred.alpha.cpu().numpy()

        mi = mutual_information(alpha=alphas)
        rmi = reverse_mutual_information(alpha=alphas)
        epkl = expected_pairwise_kullback_leibler(alpha=alphas)
        entropy = y_pred.entropy().cpu().numpy()
        categorical_entropy = expected_entropy(alpha=alphas)

        mis.append(mi)
        rmis.append(rmi)
        epkls.append(epkl)
        entropies.append(entropy)
        log_probs.append(log_prob.cpu().numpy())
        categorical_entropies.append(categorical_entropy)

    mis = np.hstack(mis)
    rmis = np.hstack(rmis)
    epkls = np.hstack(epkls)
    entropies = np.hstack(entropies)
    log_probs = np.hstack(log_probs)
    categorical_entropies = np.hstack(categorical_entropies)

    threshold_mi = np.quantile(mis, alpha)
    threshold_rmi = np.quantile(rmis, alpha)
    threshold_epkl = np.quantile(epkls, alpha)
    threshold_entropy = np.quantile(entropies, alpha)
    threshold_categorical_entropy = np.quantile(categorical_entropies, alpha)
    threshold_log_prob = np.quantile(log_probs, 1 - alpha)
    thresholds = {
        "mi": threshold_mi,
        "rmi": threshold_rmi,
        "epkl": threshold_epkl,
        "entropy": threshold_entropy,
        "log_prob": threshold_log_prob,
        "categorical_entropy": threshold_categorical_entropy,
    }
    values = {
        "mi": mis,
        "rmi": rmis,
        "epkl": epkls,
        "entropy": entropies,
        "log_prob": log_probs,
        "categorical_entropy": categorical_entropies,
    }
    return thresholds, values


def mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)

    # exact
    exact = -1. * np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) - digamma(alpha + 1) + digamma(alpha_0 + 1)),
                         axis=1)
    # approximate
    diff = 1 - alpha.shape[1] + (1 / (1 + alpha_0[:, 0])) - \
        np.sum(1 / (alpha + 1), axis=1)
    approx = -1. * diff / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def reverse_mutual_information(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    exact = np.sum(alpha / alpha_0 * (np.log(alpha / alpha_0) -
                                      digamma(alpha) + digamma(alpha_0)), axis=1)

    approx = (alpha.shape[1] - 1) / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def expected_pairwise_kullback_leibler(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    exact = np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1) - digamma(alpha) + digamma(alpha_0)),
                   axis=1)

    diff = 2 * alpha.shape[1] - 2 - \
        (1. / (1 + alpha_0[:, 0])) + np.sum(1 / (alpha + 1), axis=1)
    approx = diff / (2 * alpha_0[:, 0])

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


def expected_entropy(alpha: np.ndarray):
    """_summary_

    Args:
        alpha (np.ndarray): size N_objects x K_classes
    """
    alpha_0 = np.sum(alpha, keepdims=True, axis=1)
    # exact
    exact = -np.sum(alpha / alpha_0 * (digamma(alpha + 1) - digamma(alpha_0 + 1)),
                   axis=1)

    # # approximation
    approx = -np.sum(alpha / alpha_0 *
                    (np.log(alpha) - np.log(alpha_0)), axis=-1)

    return np.where(alpha_0[:, 0] >= 10000, approx, exact)


@torch.no_grad()
def validate_ind_ood_accuracy(
        local_model, global_model,
        in_distribution_dataset,
    out_distribution_dataset,
        threshold_ind,
        threshold_ood,
        device,
        batch_size,
        measure: str,
):
    # Combine in-distribution and out-of-distribution data loaders
    combined_loader = torch.utils.data.ConcatDataset(
        [in_distribution_dataset, out_distribution_dataset])
    combined_loader = torch.utils.data.DataLoader(
        combined_loader, batch_size=batch_size, shuffle=False)

    # Create a tensor to keep track of the origin of each data point
    data_origin = torch.cat((torch.ones(len(in_distribution_dataset), device=device),
                             torch.zeros(len(out_distribution_dataset), device=device)))

    local_preds = 0
    global_preds = 0
    local_total = 0
    global_total = 0
    ood_total = 0
    ood_preds = 0

    if measure == 'entropy':
        threshold_ind = -threshold_ind
        threshold_ood = -threshold_ood

    for batch_idx, (images, labels) in enumerate(combined_loader):
        # Get uncertainty scores from local model
        images = images.to(device)
        labels = labels.to(device)

        origin_slice = data_origin[batch_idx *
                                   batch_size: (batch_idx + 1) * batch_size].clone()
        labels[origin_slice == 0] = -1.

        # import pdb
        # pdb.set_trace()
        origin_slice[~torch.isin(labels, local_model.labels)] = 0.

        y_pred_local, log_prob_local, _ = local_model.train_forward(
            images, clamp=False)

        if measure == 'log_prob':
            uncertainty_scores = log_prob_local
        elif measure == 'entropy':
            uncertainty_scores = -y_pred_local.entropy()
        else:
            raise ValueError(f"No such measure {measure}")

        in_dist_local = uncertainty_scores > threshold_ind
        local_preds += (in_dist_local == origin_slice).sum().item()
        local_total += batch_size

        origin_slice = data_origin[batch_idx *
                                   batch_size: (batch_idx + 1) * batch_size].clone()
        ood_local = uncertainty_scores <= threshold_ind
        y_pred_global, log_prob_global, _ = global_model.train_forward(
            images[ood_local], clamp=False)

        if measure == 'log_prob':
            uncertainty_scores = log_prob_global
        elif measure == 'entropy':
            uncertainty_scores = -y_pred_global.entropy()
        else:
            raise ValueError(f"No such measure {measure}")

        origin_slice = origin_slice[ood_local]
        in_dist_global = uncertainty_scores > threshold_ood
        global_preds += (in_dist_global == origin_slice).sum().item()
        global_total += len(in_dist_global)

        origin_slice = data_origin[batch_idx *
                                   batch_size: (batch_idx + 1) * batch_size].clone()
        ood_global = uncertainty_scores <= threshold_ood
        origin_slice = origin_slice[ood_local][ood_global]
        ood_preds += (1 - origin_slice).sum().item()
        ood_total += len(origin_slice)

    local_accuracy = local_preds / local_total if local_total > 0 else None
    global_accuracy = global_preds / global_total if global_total > 0 else None
    out_distribution_share = ood_preds / \
        ood_total if ood_total > 0 else None

    return local_accuracy, global_accuracy, out_distribution_share


@torch.no_grad()
def test_ind_ood_accuracy(device, measure: str, prefix="None"):
    resulting_dict = {}
    if measure == 'log_prob':
        quantiles = {
            'mnist': 1 - 0.3,
            'fmnist': 1 - 0.4,
            'medmnistA': 1 - 0.4,
            'medmnistC': 1 - 0.4,
            'medmnistS': 1 - 0.4,
            'cifar10': 1 - 0.4,
            'svhn': 1 - 0.4,
        }
        quantiles_ood = {
            'mnist': 1 - 0.3,
            'fmnist': 1 - 0.3,
            'medmnistA': 1 - 0.3,
            'medmnistC': 1 - 0.3,
            'medmnistS': 1 - 0.3,
            'cifar10': 1 - 0.5,
            'svhn': 1 - 0.5,
        }
    else:
        quantiles = {
            'mnist': 0.7,
            'fmnist': 0.65,
            'medmnistA': 0.6,
            'medmnistC': 0.6,
            'medmnistS': 0.6,
            'cifar10': 0.28,
            'svhn': 0.6,
        }
        quantiles_ood = {
            'mnist': 0.92,
            'fmnist': 0.65,
            'medmnistA': 0.65,
            'medmnistC': 0.65,
            'medmnistS': 0.65,
            'cifar10': 0.28,
            'svhn': 0.65,
        }

    for (ind_dataset_name, ood_dataset_name) in [
        # ('mnist', 'fmnist'),
        # ('fmnist', 'mnist'),
        # ('cifar10', 'svhn'),
        ('svhn', 'cifar10'),
        ('medmnistA', 'fmnist'),
        ('medmnistS', 'fmnist'),
        ('medmnistC', 'fmnist'),
    ]:
        resulting_dict[(ind_dataset_name, ood_dataset_name)] = []
        path = f"../out/FedAvg/{prefix}all_params_stopgrad_logp_{ind_dataset_name}_100_natpn.pt"
        all_params_dict = torch.load(path)
        backbone = 'res18' if ind_dataset_name in [
            'cifar10', 'svhn'] else 'lenet5'
        batch_size = 4000 if ind_dataset_name in ['cifar10', 'svhn'] else 25000
        stopgrad = True

        data_indices_ind, trainset_ind, testset_ind = load_dataset(
            dataset_name=ind_dataset_name)

        _, trainset_ood, _ = load_dataset(
            dataset_name=ood_dataset_name,
            normalization_name=ind_dataset_name
        )

        global_model = load_model(
            dataset_name=ind_dataset_name,
            backbone=backbone,
            stopgrad=stopgrad,
            index='global',
            all_params_dict=all_params_dict,
            # all_params_dict=torch.load("../out/FedAvg/centralized_all_params_stopgrad_logp_mnist_1_natpn.pt"),
        )

        global_model.stop_grad_embeddings = False
        global_model.to(device)
        global_model.eval()

        all_calibration_datasets = []
        for i in range(len(all_params_dict) - 1):
            _, _, calloader = load_dataloaders(
                client_id=i, data_indices=data_indices_ind,
                trainset=trainset_ind, testset=testset_ind, batch_size=batch_size,
            )
            aux_dataset = calloader.dataset
            aux_dataset.indices = calloader.dataset.indices
            all_calibration_datasets.append(aux_dataset)

        # Concatenate the datasets
        combined_dataset = ConcatDataset(all_calibration_datasets)
        calloader_global = torch.utils.data.DataLoader(dataset=combined_dataset,
                                                       shuffle=False, batch_size=batch_size,)

        threshold_dict_global, values = choose_threshold(
            model=global_model,
            calloader=calloader_global,
            device=device,
            alpha=quantiles_ood[ind_dataset_name],
        )

        ############################################################
        ############################################################
        outputs = []
        for i in tqdm(range(len(all_params_dict) - 1)):

            local_model = load_model(
                dataset_name=ind_dataset_name,
                backbone=backbone,
                stopgrad=stopgrad,
                index=i,
                all_params_dict=all_params_dict,
            )

            local_model.stop_grad_embeddings = False
            local_model.to(device)
            local_model.eval()

            _, _, calloader_ind = load_dataloaders(
                client_id=i, data_indices=data_indices_ind, trainset=trainset_ind,
                testset=testset_ind, batch_size=batch_size
            )

            threshold_dict, values = choose_threshold(
                model=local_model,
                calloader=calloader_ind,
                device=device,
                alpha=quantiles[ind_dataset_name],
            )

            out = validate_ind_ood_accuracy(
                local_model=local_model,
                global_model=global_model,
                in_distribution_dataset=trainset_ind.dataset,
                out_distribution_dataset=trainset_ood.dataset,
                threshold_ind=threshold_dict[measure],
                threshold_ood=threshold_dict_global[measure],
                device=device,
                batch_size=batch_size,
                measure=measure,
            )
            outputs.append(out)
            resulting_dict[(ind_dataset_name, ood_dataset_name)].append(out)

            # if i > 3:
            #     break

        print(f"{ind_dataset_name}, {ood_dataset_name}")
        local_acc = np.mean([a[0] if a[0] is not None else 0 for a in outputs])
        print(
            f"{local_acc} +/- {np.std([a[0] if a[0] is not None else 0 for a in outputs])}")

        global_acc = np.mean(
            [a[1] if a[1] is not None else 0 for a in outputs])
        print(
            f"{global_acc} +/- {np.std([a[1] if a[1] is not None else 0 for a in outputs])}")

        ood_acc = np.mean([a[2] if a[2] is not None else 0 for a in outputs])
        print(
            f"{ood_acc} +/- {np.std([a[2] if a[2] is not None else 0 for a in outputs])}")

    return resulting_dict
