from typing import Union
from sklearn.metrics import roc_auc_score
from src.config.utils import evaluate_switch
from src.config.uncertainty_metrics import (
    load_dataset,
    load_dataloaders,
    choose_threshold,
    load_model,
)

import matplotlib.pyplot as plt
import numpy as np


def validate_switch(
        model_index: Union[str, int],
        dataset_index: int,
        dataset_name: str,
        density_model: str,
        backbone: str,
        stopgrad: bool,
        embedding_dim: int,
        all_params_dict: dict,
        device: str,
        threshold: float,
        ) -> None:

    global_model = load_model(
        dataset_name=dataset_name,
        density_model=density_model,
        backbone=backbone,
        stopgrad=stopgrad,
        embedding_dim=embedding_dim,
        index='global',
        all_params_dict=all_params_dict,
    )

    local_model = load_model(
        dataset_name=dataset_name,
        backbone=backbone,
        density_model=density_model,
        stopgrad=stopgrad,
        embedding_dim=embedding_dim,
        index=model_index,
        all_params_dict=all_params_dict,
    )

    # Load corresponding calibration loader to choose thresholds
    data_indices, trainset, testset = load_dataset(dataset_name=dataset_name)
    _, _, calloader = load_dataloaders(
        client_id=model_index, data_indices=data_indices, trainset=trainset, testset=testset
    )

    threshold_dict, _ = choose_threshold(
        model=local_model,
        calloader=calloader,
        device=device,
        alpha=threshold,
    )

    # And now load another dataset (different index)

    _, testloader, _ = load_dataloaders(
        client_id=dataset_index, data_indices=data_indices, trainset=trainset, testset=testset
    )

    for uncertainty_measure, threshold in threshold_dict.items():
        correct_decision, correct_local, correct_global, sample_num = evaluate_switch(
            local_model=local_model,
            global_model=global_model,
            dataloader=testloader,
            threshold=threshold,
            uncertainty_measure=uncertainty_measure,
            device=device,
        )
        print("*" * 100)
        print(
            f"For model index {model_index} and test dataloader of client {dataset_index}")
        print(f"Global model accuracy is {correct_global / sample_num}")
        print(f"Local model accuracy is {correct_local / sample_num}")
        print(
            f"After decision with threshold of {uncertainty_measure} model accuracy is {correct_decision / sample_num}")

        print()
        print("*" * 100)


def plot_uncertainties_scores(
        model_index: int,
        ind_dataset_name: str,
        ood_dataset_index: int,
        ood_dataset_name: str,
        density_model: str,
        backbone: str,
        stopgrad: bool,
        embedding_dim: int,
        all_params_dict: dict,
        use_global_for_validation: bool,
        device: str,
        uncertainty_measure: str,
        threshold: str,
):

    global_model = load_model(
        dataset_name=ind_dataset_name,
        density_model=density_model,
        backbone=backbone,
        stopgrad=stopgrad,
        embedding_dim=embedding_dim,
        index='global',
        all_params_dict=all_params_dict,
    )

    local_model = load_model(
        dataset_name=ind_dataset_name,
        backbone=backbone,
        density_model=density_model,
        stopgrad=stopgrad,
        embedding_dim=embedding_dim,
        index=model_index,
        all_params_dict=all_params_dict,
    )

    data_indices, trainset, testset = load_dataset(
        dataset_name=ind_dataset_name,
        normalization_name=ind_dataset_name
    )

    ind_index = 0 if model_index == 'global' else model_index
    _, _, calloader = load_dataloaders(
        client_id=ind_index, data_indices=data_indices, trainset=trainset, testset=testset
    )

    data_indices_ood, trainset_ood, testset_ood = load_dataset(
        dataset_name=ood_dataset_name,
        normalization_name=ind_dataset_name
    )
    _, testloader_ood, _ = load_dataloaders(
        client_id=ood_dataset_index, data_indices=data_indices_ood, trainset=trainset_ood, testset=testset_ood
    )

    validation_model = global_model if use_global_for_validation else local_model

    _, values_ind = choose_threshold(
        model=validation_model,
        calloader=calloader,
        device=device,
        alpha=threshold,
    )

    _, values_ood = choose_threshold(
        model=validation_model,
        calloader=testloader_ood,
        device=device,
        alpha=threshold,
    )

    plt.close()
    plt.figure(dpi=150, figsize=(10, 8))

    key = uncertainty_measure
    bins = np.linspace(min(min(values_ind[key]), min(values_ood[key])),
                       max(max(values_ind[key]), max(values_ood[key])), num=100)
    if key in ['entropy', 'log_prob']:
        plt.hist(values_ind[key], bins=bins, label='InD')
    else:
        plt.hist(np.log(values_ind[key]), bins=bins, label='InD')

    if key in ['entropy', 'log_prob']:
        plt.hist(values_ood[key], bins=bins, label='OOD', alpha=0.7)
    else:
        plt.hist(np.log(values_ood[key]), bins=bins, label='OOD', alpha=0.7)

    if key == 'log_prob':
        class_0_scores = values_ood[key]
        class_1_scores = values_ind[key]
    else:
        class_0_scores = values_ind[key]
        class_1_scores = values_ood[key]

    scores = np.concatenate([class_0_scores, class_1_scores])
    labels = np.concatenate(
        [np.zeros_like(class_0_scores), np.ones_like(class_1_scores)])
    roc_auc = roc_auc_score(labels, scores)
    print(f'ROC AUC score: {roc_auc}')

    plt.legend()
    plt.tight_layout()
    plt.show()
