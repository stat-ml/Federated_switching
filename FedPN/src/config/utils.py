import os
from collections import OrderedDict
from typing import List, Tuple, Union, Optional

import torch
import random
import numpy as np
from path import Path
from torch.utils.data import DataLoader
from src.config.nat_pn.loss import BayesianLoss, LogMarginalLoss
from src.config.models import NatPnModel
from src.config.uncertainty_metrics import (
    mutual_information,
    reverse_mutual_information,
    expected_entropy,
    expected_pairwise_kullback_leibler,
    load_dataloaders,
    load_dataset,
    load_model,
    choose_threshold,
)
from copy import deepcopy
import math
import csv
from tqdm.auto import tqdm

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
OUT_DIR = _PROJECT_DIR / "out"
TEMP_DIR = _PROJECT_DIR / "temp"


quantiles = {
    'mnist': 1 - 0.1,
    'fmnist': 1 - 0.3,
    'medmnistA': 1 - 0.4,
    'medmnistC': 1 - 0.4,
    'medmnistS': 1 - 0.4,
    'cifar10': 1 - 0.4,
    'svhn': 1 - 0.2,
}


def fix_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )



def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False, include_gaussian_flow=False,
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if include_gaussian_flow:
        keys.append('density_model.means')
        parameters.append(src.density_model.means)
        keys.append('density_model.stds')
        parameters.append(src.density_model.stds)

    if requires_name:
        return keys, parameters
    else:
        return parameters


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if isinstance(criterion, BayesianLoss) or isinstance(criterion, LogMarginalLoss):
            y_pred, log_prob, embeddings = model.train_forward(x)
            logits = y_pred.alpha.log()
            loss += criterion(y_pred, y, log_prob, embeddings).item()
        else:
            logits = model(x)
            loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
    return loss, correct, sample_num


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device=torch.device("cpu"),
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    sample_num = 0
    mean_log_prob = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, NatPnModel):
            y_pred, log_prob, _ = model.train_forward(x)
            # print(f"***************** logprob: {log_prob.cpu().mean()}")
            logits = y_pred.alpha
            mean_log_prob.append(log_prob.cpu().numpy())
        else:
            logits = model(x)
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)

    return correct / sample_num, np.mean(np.hstack(mean_log_prob))


@torch.no_grad()
def evaluate_switch(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float,
    uncertainty_measure: str,
    device=torch.device("cpu"),
    return_predictions: bool = False
) -> tuple[float, float, int]:
    local_model.eval()
    global_model.eval()
    correct_local = 0
    correct_global = 0
    correct_decision = 0
    if return_predictions:
        local_predictions = []
        global_predictions = []
        switch_predictions = []
        true_labels = []
        uncertainty_scores_local = []
        uncertainty_scores_global = []
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred_local, log_prob_local, _ = local_model.train_forward(
            x, clamp=False)
        y_pred_global, log_prob_global, _ = global_model.train_forward(x, clamp=False)

        alpha_local = y_pred_local.alpha
        alpha_global = y_pred_global.alpha

        threshold_torch = threshold * torch.ones_like(alpha_local[:, 0])

        if uncertainty_measure == 'mi':
            local_measure = mutual_information(alpha=alpha_local.cpu().numpy())
            global_measure = mutual_information(alpha=alpha_global.cpu().numpy())
        elif uncertainty_measure == 'rmi':
            local_measure = reverse_mutual_information(
                alpha=alpha_local.cpu().numpy())
            global_measure = reverse_mutual_information(
                alpha=alpha_global.cpu().numpy())
        elif uncertainty_measure == 'epkl':
            local_measure = expected_pairwise_kullback_leibler(
                alpha=alpha_local.cpu().numpy())
            global_measure = expected_pairwise_kullback_leibler(
                alpha=alpha_global.cpu().numpy())
        elif uncertainty_measure == 'entropy':
            local_measure = y_pred_local.entropy().cpu().numpy()
            global_measure = y_pred_global.entropy().cpu().numpy()
        elif uncertainty_measure == 'log_prob':
            local_measure = log_prob_local.cpu().numpy()
            global_measure = log_prob_global.cpu().numpy()
        elif uncertainty_measure == 'categorical_entropy':
            local_measure = expected_entropy(alpha=alpha_local.cpu().numpy())
            global_measure = expected_entropy(alpha=alpha_global.cpu().numpy())
        else:
            raise ValueError(
                f'No such uncertainty measure available! {uncertainty_measure}')

        local_measure = torch.tensor(
            local_measure, device=device) * torch.ones_like(alpha_local[:, 0])
        if uncertainty_measure != 'log_prob':
            alphas_after_decision = torch.where((local_measure < threshold_torch)[..., None],
                                                alpha_local, alpha_global)
        else:
            alphas_after_decision = torch.where((local_measure > threshold_torch)[..., None],
                                                alpha_local, alpha_global)

        pred_local = torch.argmax(alpha_local, -1)
        pred_global = torch.argmax(alpha_global, -1)
        pred_after_decision = torch.argmax(alphas_after_decision, -1)

        correct_local += (pred_local == y).sum().item()
        correct_global += (pred_global == y).sum().item()
        correct_decision += (pred_after_decision == y).sum().item()

        if return_predictions:
            switch_predictions.append(pred_after_decision.cpu())
            global_predictions.append(pred_global.cpu())
            local_predictions.append(pred_local.cpu())
            true_labels.append(y.cpu())
            uncertainty_scores_local.append(local_measure.cpu())
            uncertainty_scores_global.append(global_measure)

        sample_num += len(y)
    if return_predictions:
        return correct_decision, correct_local, correct_global, sample_num, local_predictions, global_predictions, switch_predictions, true_labels, uncertainty_scores_local, uncertainty_scores_global
    else:
        return correct_decision, correct_local, correct_global, sample_num


@torch.no_grad()
def validate_accuracy_per_client(
        dataset_name: str,
        backbone: str,
        density_model: str,
        stopgrad: bool,
        embedding_dim: int,
        criterion: torch.nn.Module,
        all_params_dict: dict[int, torch.Tensor],
        device: str,
        validate_only_classifier: bool = False,
) -> None:
    data_indices, trainset, testset = load_dataset(dataset_name=dataset_name)
    for index in ['global'] + [i for i in range(len(data_indices))]:
        print("model index is: ", index)
        current_accuracies_list = []

        current_model = load_model(
            dataset_name=dataset_name,
            backbone=backbone,
            density_model=density_model,
            embedding_dim=embedding_dim,
            stopgrad=stopgrad,
            index=index,
            all_params_dict=all_params_dict,
        )
        current_model.eval()
        current_model.to(device)

        for dataset_index in range(len(data_indices)):
            _, testloader, _ = load_dataloaders(
                client_id=dataset_index, data_indices=data_indices, trainset=trainset, testset=testset
            )
            test_loss, test_correct, test_sample_num = evaluate(
                model=current_model,
                dataloader=testloader,
                criterion=criterion,
                device=device,
            )
            if isinstance(current_model, NatPnModel) and validate_only_classifier:
                correct, overall = evaluate_only_classifier(
                    model=current_model, dataloader=testloader, device=device)
                print(
                    f"{dataset_index}: {test_correct / test_sample_num} \t Accuracy only classifier: {correct.sum() / overall}")
            else:
                print(f"{dataset_index}: {test_correct / test_sample_num}")
            current_accuracies_list.append(test_correct / test_sample_num)
        print(
            f"mean accuracy of {index} is {np.mean(current_accuracies_list)} +/- {np.std(current_accuracies_list)}")


@torch.no_grad()
def evaluate_only_classifier(
    model: NatPnModel,
    dataloader: DataLoader,
    device,
):
    all_correct = []
    n_samples = 0
    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        pred_raw = model.classifier(model.base(x))
        pred_logit = pred_raw.logits
        pred = pred_logit.argmax(-1).cpu().numpy()
        all_correct.append(pred == y.cpu().numpy())
        n_samples += len(y.cpu().numpy())
    return np.hstack(all_correct), n_samples


@torch.no_grad()
def reset_flow_and_classifier_params(state_dict: OrderedDict):
    new_state_dict = deepcopy(state_dict)
    for k, v in state_dict.items():
        if not (k.startswith("density_model") or k.startswith("classifier")):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = torch.nn.Parameter(torch.nn.init.kaiming_uniform(
                new_state_dict[k][None], a=math.sqrt(5))[0])
    return new_state_dict


def update_csv(filename, row_data):
    # Check if file exists
    try:
        with open(filename, 'x', newline='') as file:
            writer = csv.writer(file)
            # Write the headers to the CSV file
            writer.writerow([
                'dataset_name',
                'model_id',
                'dataset_id',
                'class_intersect_len',
                'centralized_model_acc',
                'local_model_acc',
                'global_model_acc',
                'switch_model_acc_mi',
                'switch_model_acc_rmi',
                'switch_model_acc_epkl',
                'switch_model_acc_entropy',
                'switch_model_acc_logprob',
                'switch_model_acc_catentropy',
            ])
    except FileExistsError:
        pass

    # Append new row to the CSV file
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


@torch.no_grad()
def make_inference_csv(
    dataset_name: str,
    path_to_saved_models: str,
    device: str,
    ALPHA_THRESHOLD: float = 0.95,
    path_to_centralized_model: Optional[str] = None,
):
    all_params_dict = torch.load(path_to_saved_models)
    backbone = 'res18' if dataset_name in ['cifar10', 'svhn'] else 'lenet5'

    if path_to_centralized_model is not None:
        centralized_dict = torch.load(path_to_centralized_model)
        centralized_model = load_model(
            dataset_name=dataset_name,
            backbone=backbone,
            stopgrad=True,
            index="global",
            all_params_dict=centralized_dict,
        )
        centralized_model.stop_grad_embeddings = False
        centralized_model.to(device)
        centralized_model.eval()
    else:
        centralized_model = None

    global_model = load_model(
        dataset_name=dataset_name,
        backbone=backbone,
        stopgrad=True,
        index="global",
        all_params_dict=all_params_dict,
    )
    global_model.stop_grad_embeddings = False
    global_model.to(device)
    global_model.eval()

    data_indices, trainset, testset = load_dataset(dataset_name=dataset_name)

    for model_id in tqdm(range(len(all_params_dict) - 1)):
        print("model index is: ", model_id)

        ind_labels = all_params_dict[model_id]['labels'].numpy()
        current_model = load_model(
            dataset_name=dataset_name,
            backbone=backbone,
            stopgrad=True,
            index=model_id,
            all_params_dict=all_params_dict,
        )
        current_model.stop_grad_embeddings = False
        current_model.to(device)
        current_model.eval()

        _, _, calloader = load_dataloaders(
            client_id=model_id, data_indices=data_indices, trainset=trainset, testset=testset
        )

        threshold_dict, _ = choose_threshold(
            model=current_model,
            calloader=calloader,
            device=device,
            alpha=ALPHA_THRESHOLD,
        )

        for dataset_id in range(len(all_params_dict) - 1):
            _, testloader, _ = load_dataloaders(
                client_id=dataset_id, data_indices=data_indices, trainset=trainset, testset=testset)
            other_labels = all_params_dict[dataset_id]['labels'].numpy(
            ).tolist()
            intersection_len = len(
                set(ind_labels).intersection(set(other_labels)))

            accuracies_of_different_uncertainties = []
            for uncertainty_measure, threshold in threshold_dict.items():
                correct_decision, correct_local, correct_global, sample_num = evaluate_switch(
                    local_model=current_model,
                    global_model=global_model,
                    dataloader=testloader,
                    threshold=threshold,
                    uncertainty_measure=uncertainty_measure,
                    device=device,
                )
                if centralized_model is not None:
                    centralized_acc, _ = evaluate_accuracy(
                        model=centralized_model,
                        dataloader=testloader,
                        device=device
                    )
                else:
                    centralized_acc = None

                global_accuracy = correct_global / sample_num
                local_model_accuracy = correct_local / sample_num
                accuracies_of_different_uncertainties.extend(
                    [correct_decision / sample_num])

            new_row = [
                dataset_name,
                model_id,
                dataset_id,
                intersection_len,
                centralized_acc,
                local_model_accuracy,
                global_accuracy,
                *accuracies_of_different_uncertainties
            ]  # replace with your function call
            update_csv(f'{dataset_name}.csv', new_row)


def save_models(args, algo, all_models_dict, model_name):
    text_stopgrad_logp = "_stopgrad_logp" if args.stop_grad_logp else ""
    text_stopgrad_embeddings = "_stopgrad_embeddings" if args.stop_grad_embeddings else ""
    if args.save_prefix is not None:
        args.save_prefix = args.save_prefix + '_'
    if args.dataset == 'toy_noisy':
        torch.save(all_models_dict, OUT_DIR / algo /
                    f"{args.save_prefix}all_params_{args.dataset_args['toy_noisy_classes']}{text_stopgrad_logp}{text_stopgrad_embeddings}_{args.seed}_{model_name}")
    else:
        torch.save(all_models_dict, OUT_DIR / algo /
                    f"{args.save_prefix}all_params{text_stopgrad_logp}{text_stopgrad_embeddings}_{model_name}")

    # save_path = os.path.join(save_root_path, run.id + "_" + run.name)
    # config["train_params"]["save_root_path"] = save_path

    # save_models(models_collection=trained_models, client_to_dataloaders=initialization.id2dl_dictionary,
    #             save_path=save_path,
    #             config=config)