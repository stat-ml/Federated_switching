import pickle
import sys
import json
import os
import random
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional
from tqdm.auto import tqdm
import numpy as np

import optuna

import torch
from path import Path
from rich.console import Console
from rich.progress import track

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()

sys.path.append(_PROJECT_DIR)

from src.config.utils import evaluate_accuracy, evaluate_only_classifier
from src.config.nat_pn.loss import BayesianLoss
from src.client.fedavg import FedAvgClient
from src.config.args import get_fedavg_argparser
from src.config.models import MODEL_DICT
from src.config.utils import OUT_DIR, fix_random_seed, trainable_params, reset_flow_and_classifier_params


class FedAvgServer:
    def __init__(
        self,
        algo: str = "FedAvg",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
        trial: Optional[optuna.Trial]=None,
        optimizer_name: Optional[str]=None
    ):
        if optimizer_name is None:
            self.optimizer_name = "Adam"
        else:
            self.optimizer_name = optimizer_name
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo
        self.trial = trial
        self.unique_model = unique_model
        fix_random_seed(self.args.seed)
        with open(_PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)

        # get client party info
        self.train_clients: List[int] = None
        self.test_clients: List[int] = None
        self.client_num_in_total: int = None
        try:
            partition_path = _PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients = partition["separation"]["train"]
        self.test_clients = partition["separation"]["test"]
        self.client_num_in_total = partition["separation"]["total"]

        # init model(s) parameters
        self.device = torch.device(
            "cuda" if self.args.server_cuda and torch.cuda.is_available() else "cpu"
        )
        self.reinitialize_model()
        self.model.check_avaliability()
        self.trainable_params_name, init_trainable_params = trainable_params(
            self.model, requires_name=True
        )
        # client_trainable_params is for pFL, which outputs exclusive model per client
        # global_params_dict is for regular FL, which outputs a single global model
        if self.unique_model:
            self.client_trainable_params: List[List[torch.Tensor]] = [
                deepcopy(init_trainable_params) for _ in self.train_clients
            ]
        self.global_params_dict: OrderedDict[str, torch.nn.Parameter] = OrderedDict(
            zip(self.trainable_params_name, deepcopy(init_trainable_params))
        )

        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            random.sample(
                self.train_clients, int(
                    self.client_num_in_total * self.args.join_ratio)
            )
            for _ in range(self.args.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0

        # variables for logging
        if self.args.visible:
            from visdom import Visdom

            self.viz = Visdom()
            self.viz_win_name = (
                f"{self.algo}"
                + f"_{self.args.dataset}"
                + f"_{self.args.global_epoch}"
                + f"_{self.args.local_epoch}"
            )
        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
        }
        self.logger = Console(record=self.args.save_log,
                              log_path=False, log_time=False)
        self.test_results: Dict[int, Dict[str, str]] = {}
        self.train_progress_bar = track(
            range(self.args.global_epoch), "[bold green]Training..."
        )

        self.logger.log("=" * 20, "ALGORITHM:", self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))

        # init trainer
        self.trainer = None
        if default_trainer:
            self.trainer = FedAvgClient(
                deepcopy(self.model), self.args, self.logger, optimizer_name=self.optimizer_name)

    def reinitialize_model(self,):
        if self.args.model == "natpn":
            if self.args.dataset == "toy_noisy":
                dataset = self.args.dataset + \
                    f"_{self.args.dataset_args['toy_noisy_classes']}"
            else:
                dataset = self.args.dataset
            self.dataset_name = dataset

            self.model = MODEL_DICT[self.args.model](dataset=self.dataset_name,
                                                     backbone=self.args.nat_pn_backbone,
                                                     stop_grad_logp=self.args.stop_grad_logp,
                                                     stop_grad_embeddings=self.args.stop_grad_embeddings,
                                                     ).to(self.device)
        else:
            self.model = MODEL_DICT[self.args.model](
                self.args.dataset).to(self.device)

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            delta_cache = []
            weight_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                delta, weight, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                self.trainer.model.labels = None
                self.trainer.model.labels_frequency = None

                delta_cache.append(delta)
                weight_cache.append(weight)

            self.aggregate(delta_cache, weight_cache)

            if self.trial is not None and (E + 1) % 5 == 0:
                self.trainer.set_parameters(self.global_params_dict)

                n_classes = len(self.trainer.model.density_model)
                self.trainer.model.labels = torch.arange(n_classes)
                self.trainer.model.labels_frequency = torch.ones(n_classes) / n_classes

                accuracies = []
                for cl_id in range(self.client_num_in_total):
                    self.trainer.client_id = cl_id
                    self.trainer.load_dataset()
                    accuracy = evaluate_accuracy(model=self.trainer.model,
                                                  dataloader=self.trainer.testloader,
                                                    device=self.device)[0]
                    accuracies.append(accuracy)
                self.trial.report(np.mean(accuracies), E)

                # Handle pruning based on the intermediate value.
                if self.trial.should_prune():
                    raise optuna.TrialPruned()


    def test(self):
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        num_samples = []
        for client_id in self.test_clients:
            client_local_params = self.generate_client_params(client_id)
            stats = self.trainer.test(client_id, client_local_params)

            correct_before.append(stats["before"]["test_correct"])
            correct_after.append(stats["after"]["test_correct"])
            loss_before.append(stats["before"]["test_loss"])
            loss_after.append(stats["after"]["test_loss"])
            num_samples.append(stats["before"]["test_size"])

        loss_before = torch.tensor(loss_before)
        loss_after = torch.tensor(loss_after)
        correct_before = torch.tensor(correct_before)
        correct_after = torch.tensor(correct_after)
        num_samples = torch.tensor(num_samples)

        self.test_results[self.current_epoch + 1] = {
            "loss": "{:.4f} -> {:.4f}".format(
                loss_before.sum() / num_samples.sum(),
                loss_after.sum() / num_samples.sum(),
            ),
            "accuracy": "{:.2f}% -> {:.2f}%".format(
                correct_before.sum() / num_samples.sum() * 100,
                correct_after.sum() / num_samples.sum() * 100,
            ),
        }

    @torch.no_grad()
    def update_client_params(self, client_params_cache: List[List[torch.nn.Parameter]]):
        if self.unique_model:
            for i, client_id in enumerate(self.selected_clients):
                self.client_trainable_params[client_id] = [
                    param.detach().to(self.device) for param in client_params_cache[i]
                ]
        else:
            raise RuntimeError(
                "FL system don't preserve params for each client (unique_model = False)."
            )

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name,
                    self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    @torch.no_grad()
    def aggregate(self, delta_cache: List[List[torch.Tensor]], weight_cache: List[int]):
        weights = torch.tensor(
            weight_cache, device=self.device) / sum(weight_cache)
        delta_list = [list(delta.values()) for delta in delta_cache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]

        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.data -= diff.to(self.device)

    def check_convergence(self):
        for label, metric in self.metrics.items():
            if len(metric) > 0:
                self.logger.log(f"Convergence ({label}):")
                acc_range = [90.0, 80.0, 70.0, 60.0,
                             50.0, 40.0, 30.0, 20.0, 10.0]
                min_acc_idx = 10
                max_acc = 0
                for E, acc in enumerate(metric):
                    for i, target in enumerate(acc_range):
                        if acc >= target and acc > max_acc:
                            self.logger.log(
                                "{} achieved {}%({:.2f}%) at epoch: {}".format(
                                    self.algo, target, acc, E
                                )
                            )
                            max_acc = acc
                            min_acc_idx = i
                            break
                    acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        for label in ["train", "test"]:
            # In the `user` split, there is no test data held by train clients, so plotting is unnecessary.
            if (label == "train" and self.args.eval_train) or (
                label == "test"
                and self.args.eval_test
                and self.args.dataset_args["split"] != "user"
            ):
                correct_before = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"][
                            f"{label}_correct"
                        ]
                        for c in self.selected_clients
                    ]
                )
                correct_after = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["after"][
                            f"{label}_correct"
                        ]
                        for c in self.selected_clients
                    ]
                )
                num_samples = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"][
                            f"{label}_size"
                        ]
                        for c in self.selected_clients
                    ]
                )

                acc_before = (
                    correct_before.sum(dim=-1, keepdim=True) /
                    num_samples.sum() * 100.0
                ).item()
                acc_after = (
                    correct_after.sum(dim=-1, keepdim=True) /
                    num_samples.sum() * 100.0
                ).item()
                self.metrics[f"{label}_before"].append(acc_before)
                self.metrics[f"{label}_after"].append(acc_after)

                if self.args.visible:
                    self.viz.line(
                        [acc_before],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(before)",
                        opts=dict(
                            title=self.viz_win_name,
                            xlabel="Communication Rounds",
                            ylabel="Accuracy",
                        ),
                    )
                    self.viz.line(
                        [acc_after],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(after)",
                    )

    def run(self):
        if self.trainer is None:
            raise RuntimeError(
                "Specify your unique trainer or set `default_trainer` as True."
            )

        if self.args.visible:
            self.viz.close(win=self.viz_win_name)

        self.train()

        self.logger.log(
            "=" * 20, self.algo, "TEST RESULTS:", "=" * 20, self.test_results
        )

        self.check_convergence()

        # save log files
        if not os.path.isdir(OUT_DIR / self.algo) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(OUT_DIR / self.algo, exist_ok=True)

        if self.args.save_log:
            self.logger.save_text(OUT_DIR / self.algo /
                                  f"{self.args.dataset}_log.html")

        if self.args.save_fig:
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "test_before": "solid",
                "test_after": "solid",
                "train_before": "dotted",
                "train_after": "dotted",
            }
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    plt.plot(acc, label=label, ls=linestyle[label])
            plt.title(f"{self.algo}_{self.args.dataset}")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                OUT_DIR / self.algo / f"{self.args.dataset}.jpeg", bbox_inches="tight"
            )
        if self.args.save_metrics:
            import pandas as pd
            import numpy as np

            accuracies = []
            labels = []
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    accuracies.append(np.array(acc).T)
                    labels.append(label)
            pd.DataFrame(np.stack(accuracies, axis=1), columns=labels).to_csv(
                OUT_DIR / self.algo / f"{self.args.dataset}_acc_metrics.csv",
                index=False,
            )
        # save trained model(s)
        if self.args.save_model:
            model_name = (
                f"{self.args.dataset}_{self.args.global_epoch}_{self.args.model}.pt"
            )
            if self.unique_model:
                torch.save(
                    self.client_trainable_params, OUT_DIR / self.algo / model_name
                )
            else:
                if self.args.model != "natpn":
                    torch.save(self.global_params_dict,
                               OUT_DIR / self.algo / model_name)
                # torch.save(self.model.state_dict(), OUT_DIR / self.algo / model_name)

            # self.post_training_and_save(model_name)

    def post_training_and_save(self, model_name,):
        all_models_dict: dict[str | int, torch.nn.ParameterDict] = {}

        self.trainer.set_parameters(self.global_params_dict)
        global_model = self.trainer.model
        all_models_dict["global"] = deepcopy(
            global_model.state_dict(keep_vars=True))


        self.trainer.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.trainer.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.trainer.reset_optimizers("Adam")
        self.trainer.is_federated_stage = False

        if self.args.finetune_in_the_end > 0:
            self.trainer.criterion = BayesianLoss(
                entropy_weight=0.0,
                log_prob_weight=self.args.loss_log_prob_weight,
                embeddings_weight=0.0,
            )
            self.trainer.local_epoch = self.args.finetune_in_the_end
            for client_id in tqdm(range(self.client_num_in_total)):
                client_local_params = self.generate_client_params(client_id)
                client_local_params = reset_flow_and_classifier_params(
                    client_local_params)
                new_params, _, _ = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    return_diff=False,
                    verbose=False,
                    train_base_model=False,
                )
                all_models_dict[client_id] = deepcopy(
                    self.trainer.model.state_dict(keep_vars=True))
        text_stopgrad_logp = "_stopgrad_logp" if self.args.stop_grad_logp else ""
        text_stopgrad_embeddings = "_stopgrad_embeddings" if self.args.stop_grad_embeddings else ""
        if self.args.save_prefix is not None:
            self.args.save_prefix = self.args.save_prefix + '_'
        if self.args.dataset == 'toy_noisy':
            torch.save(all_models_dict, OUT_DIR / self.algo /
                       f"{self.args.save_prefix}all_params_{self.args.dataset_args['toy_noisy_classes']}{text_stopgrad_logp}{text_stopgrad_embeddings}_{self.args.seed}_{model_name}")
        else:
            torch.save(all_models_dict, OUT_DIR / self.algo /
                       f"{self.args.save_prefix}all_params{text_stopgrad_logp}{text_stopgrad_embeddings}_{model_name}")


def objective(trial: optuna.Trial):
    args = get_fedavg_argparser().parse_args()

    # Categorical parameter
    args.stop_grad_logp = trial.suggest_categorical("stop_grad_logp", [True, False])
    args.batch_size = trial.suggest_categorical("batch_size", [64, 128])

    args.optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "SGD"])
    args.local_lr = trial.suggest_categorical("local_lr", [0.01, 0.001, 0.0001, 0.00001,])
    args.loss_log_prob_weight = trial.suggest_categorical("loss_log_prob_weight", [0.01, 0.001, 0.0001, 0.00001,])
    args.loss_entropy_weight = trial.suggest_categorical("loss_entropy_weight", [0.0, 0.01, 0.001, 0.0001, 0.00001,])


    server = FedAvgServer(args=args, trial=trial)
    server.run()

    labels, counts = torch.unique(server.trainer.trainset.dataset.targets[server.trainer.trainset.indices], return_counts=True)
    server.trainer.model.labels = labels
    server.trainer.model.labels_frequency = counts / counts.sum()
    server.trainer.set_parameters(server.global_params_dict)

    n_classes = len(server.trainer.model.density_model)
    server.trainer.model.labels = torch.arange(n_classes)
    server.trainer.model.labels_frequency = torch.ones(n_classes) / n_classes

    accuracies = []
    for cl_id in range(server.client_num_in_total):
        server.trainer.client_id = cl_id
        server.trainer.load_dataset()
        accuracy = evaluate_accuracy(model=server.trainer.model,
                                        dataloader=server.trainer.testloader,
                                        device=server.device)[0]
        accuracies.append(accuracy)

    return np.mean(accuracies)



if __name__ == "__main__":
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(objective, timeout=60 * 60 * 8)

    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)