import sys
sys.path.insert(0, './FL-bench')
sys.path.insert(0, './FL-bench/src/server')

from src.config.utils import modified_evaluate, evaluate_switch
from src.config.models import NatPnModel
from src.server.fedrep import FedRepServer, get_fedrep_argparser
from src.server.ditto import DittoServer, get_ditto_argparser
from src.server.apfl import APFLServer, get_apfl_argparser
from src.client.apfl import MixedModel
from src.server.fedper import FedPerServer
from src.server.fedbn import FedBNServer
from src.server.fedbabu import FedBabuServer
from src.server.perfedavg import PerFedAvgServer, get_perfedavg_argparser
from src.server.fedavg import FedAvgServer, get_fedavg_argparser
from src.server.local import LocalServer
from src.server.fedpn import FedPNServer, get_fedpn_argparser
from src.config.uncertainty_metrics import choose_threshold
from tqdm.auto import tqdm
import csv
import pandas as pd
from copy import deepcopy
import torch
import numpy as np
import random


def get_unique_labels(dataloader):
    unique_labels = set()
    for _, labels in dataloader:
        unique_labels.update(labels.numpy())
    return unique_labels


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# "APFL", "Ditto", "FedPer", "FedRep", "FedBN", "FedBabu", "PerFedAvg", "FedAvg", "Local", "FedPN"
for method in ["FedPN"]:
    table_save_name = f'{method}_benchmark_compact_results.csv'

    header = [
        'dataset',
        'client_id',
        'ind_local_acc',
        'ood_local_acc',
        'mix_local_acc',
    ]

    # create and write the header to the csv file
    with open(table_save_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # 'cifar10', 'svhn', 'mnist', 'fmnist', 'medmnistA', 'medmnistC', 'medmnistS'
    for dataset_name in tqdm(['mnist']):
        set_seed(0)
        if method == "FedPer":
            args = get_fedavg_argparser().parse_args()
        elif method == "FedPN":
            args = get_fedpn_argparser().parse_args()
        elif method == "FedRep":
            args = get_fedrep_argparser().parse_args()
        elif method == 'Ditto':
            args = get_ditto_argparser().parse_args()
        elif method == 'FedBN':
            args = get_fedavg_argparser().parse_args()
        elif method == 'APFL':
            args = get_apfl_argparser().parse_args()
        elif method == 'FedBabu':
            args = get_fedavg_argparser().parse_args()
        elif method == 'PerFedAvg':
            args = get_perfedavg_argparser().parse_args()
        elif method == 'FedAvg':
            args = get_fedavg_argparser().parse_args()
        elif method == "Local":
            args = get_fedavg_argparser().parse_args()
        else:
            raise ValueError()

        if method in ['PerFedAvg', 'Local']:
            match method:
                case "PerFedAvg":
                    folder_name = "Per-FedAvg(FO)"
                case "Local":
                    folder_name = "Local-only"
        else:
            folder_name = method


        if method == "FedPN":
            model_name = f"{dataset_name}_10_natpn.pt"
            args.external_model_params_file = f"FL-bench/out/{folder_name}/{model_name}"
        else:
            args.external_model_params_file = f"FL-bench/out/{folder_name}/{dataset_name}_500_lenet5.pt"


        if method in ['FedAvg', 'FedBN']:
            args.finetune_epoch = 0
        else:
            args.finetune_epoch = 5

        args.dataset = dataset_name
        batch_size = 7000 if dataset_name in ['cifar10', 'svhn'] else 25000

        if method == "FedPer":
            model = FedPerServer(args=args)
        elif method == "FedRep":
            model = FedRepServer(args=args)
        elif method == "FedPN":
            model = FedPNServer(args=args)
        elif method == 'Ditto':
            model = DittoServer(args=args)
        elif method == 'FedBN':
            model = FedBNServer(args=args)
        elif method == 'APFL':
            model = APFLServer(args=args)
        elif method == 'FedBabu':
            model = FedBabuServer(args=args)
        elif method == 'PerFedAvg':
            model = PerFedAvgServer(args=args)
        elif method == 'FedAvg':
            model = FedAvgServer(args=args)
        elif method == 'Local':
            model = LocalServer(args=args)
        else:
            raise ValueError()

        ind_accuracies = []
        ood_accuracies = []
        mix_accuracies = []

        for client_id in range(args.dataset_args['client_num']):
            client_local_params = model.generate_client_params(client_id)
            model.trainer.client_id = client_id
            model.trainer.load_dataset()
            model.trainer.set_parameters(client_local_params)
            
            with torch.no_grad():
                # To put right batch norm
                model.trainer.finetune()

            if method == 'FedPN':
                global_model = NatPnModel(dataset_name).to(model.trainer.device,)
                global_model.load_state_dict(model.trainer.model.state_dict())
                global_model.eval()

            print("FINETUNING")
            model.trainer.finetune()
            model.trainer.dataset.enable_train_transform = False
            model.trainer.save_state()
            if method == 'FedPN':
                client_finetuned_local_params = model.generate_client_params(
                    client_id)
                model.trainer.set_parameters(client_finetuned_local_params)
                local_model = model.trainer.model
                local_model.eval()

            in_classes = get_unique_labels(model.trainer.trainloader)

            # create in-distribution and out-of-distribution datasets
            in_distribution_indices = [i for i, t in enumerate(
                model.trainer.testset.dataset.targets.numpy()) if t in in_classes]

            # Shuffle the array to ensure randomness
            np.random.shuffle(in_distribution_indices)

            # Calculate the index where to split the array
            split_idx = int(len(in_distribution_indices) * 0.4)
            # Split the array into two parts

            # first 40%
            calibration_indices = in_distribution_indices[:split_idx]
            # remaining 60%
            in_distribution_indices = in_distribution_indices[split_idx:]
            out_distribution_indices = [i for i, t in enumerate(
                model.trainer.testset.dataset.targets.numpy()) if t not in in_classes]

            in_distribution_dataset = torch.utils.data.Subset(
                model.trainer.testset.dataset, in_distribution_indices)
            calibration_dataset = torch.utils.data.Subset(
                model.trainer.testset.dataset, calibration_indices)
            out_distribution_dataset = torch.utils.data.Subset(
                model.trainer.testset.dataset, out_distribution_indices)

            ind_loader = torch.utils.data.DataLoader(
                in_distribution_dataset, batch_size=batch_size, shuffle=False)
            cal_loader = torch.utils.data.DataLoader(
                calibration_dataset, batch_size=batch_size, shuffle=False)
            ood_loader = torch.utils.data.DataLoader(
                out_distribution_dataset, batch_size=batch_size, shuffle=False)

            if method in ['Ditto', 'APFL']:
                match method:
                    case 'Ditto':
                        eval_model = model.trainer.pers_model
                    case 'APFL':
                        eval_model = MixedModel(model.trainer.local_model,
                                                model.trainer.model,
                                                alpha=model.trainer.alpha)
            else:
                eval_model = model.trainer.model
            eval_model.eval()

            if method == 'FedPN':
                threshold, values = choose_threshold(
                    model=local_model,
                    calloader=cal_loader,
                    device=model.trainer.device,
                    alpha=0.7,
                )
                ind_correct_decision, ind_correct_local, ind_correct_global, ind_sample_num = evaluate_switch(
                    local_model=local_model,
                    global_model=global_model,
                    dataloader=ind_loader,
                    threshold=threshold['entropy'],
                    device=model.trainer.device,
                    return_predictions=False
                )

                ood_correct_decision, ood_correct_local, ood_correct_global, ood_sample_num, ood_local_predictions, ood_global_predictions, ood_switch_predictions, ood_true_labels, _, _ = evaluate_switch(
                    local_model=local_model,
                    global_model=global_model,
                    dataloader=ood_loader,
                    threshold=threshold['entropy'],
                    device=model.trainer.device,
                    return_predictions=True
                )
                # This is SWITCH accuracies!
                ind_accuracies.append(ind_correct_decision / ind_sample_num)
                ood_accuracies.append(ood_correct_decision / ood_sample_num)

                mix_answers = torch.hstack(ood_switch_predictions)
                mix_correct = ind_correct_decision
                true_labels = torch.hstack(ood_true_labels)

            else:
                _, ind_correct_local, ind_sample_num, _, _ = modified_evaluate(
                    model=eval_model, dataloader=ind_loader, device=model.trainer.device)
                _, ood_correct_local, ood_sample_num, ood_local_predictions, ood_true_labels = modified_evaluate(
                    model=eval_model, dataloader=ood_loader, device=model.trainer.device)
                true_labels = torch.tensor(ood_true_labels)
                mix_answers = torch.tensor(ood_local_predictions)
                mix_correct = ind_correct_local

                ind_accuracies.append(ind_correct_local / ind_sample_num)
                ood_accuracies.append(ood_correct_local / ood_sample_num)

            mix_samples = ind_sample_num

            all_labels = torch.unique(true_labels)
            sample_to_pick = int(ind_sample_num / len(all_labels))

            for l in all_labels:
                selected_predictions = (mix_answers == true_labels)[
                    true_labels == l][:sample_to_pick]
                mix_correct += selected_predictions.sum().cpu().item()
                mix_samples += sample_to_pick

            mix_accuracies.append(mix_correct / mix_samples)

            new_row = [
                dataset_name,
                client_id,
                ind_accuracies[-1],
                ood_accuracies[-1],
                mix_accuracies[-1],
            ]
            with open(table_save_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_row)

    comp_results = pd.read_csv(table_save_name)
    grouped_df = comp_results.groupby('dataset')[
        [col for col in comp_results.columns if col not in ['client_id', 'dataset']]].mean()
    print(grouped_df)
