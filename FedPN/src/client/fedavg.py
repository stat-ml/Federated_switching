import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import wandb
import torch
from path import Path
from rich.console import Console
import numpy as np
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()

from src.config.utils import trainable_params, evaluate
from src.config.models import DecoupledModel
from src.config.nat_pn.loss import BayesianLoss, LogMarginalLoss
from src.config.flows.utils_flow import GaussianFlow
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(self, model: DecoupledModel, args: Namespace, logger: Console, optimizer_name: Optional[str]=None):
        self.args = args
        self.device = torch.device(
            "cuda" if self.args.client_cuda and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None

        # load dataset and clients' data indices
        try:
            partition_path = _PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        transform = Compose(
            [Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
        )
        # transform = None
        target_transform = None

        self.dataset = DATASETS[self.args.dataset](
            root=_PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            transform=transform,
            target_transform=target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        self.local_lr = self.args.local_lr
        if self.args.loss_name == "bayessian":
            self.criterion = BayesianLoss(
                entropy_weight=self.args.loss_entropy_weight,
                log_prob_weight=self.args.loss_log_prob_weight,
                embeddings_weight=self.args.loss_embeddings_weight,
                                          )
        elif self.args.loss_name == "marginal_ll":
            self.criterion = LogMarginalLoss(
                entropy_weight=self.args.loss_entropy_weight,
                log_prob_weight=self.args.loss_log_prob_weight,
                embeddings_weight=self.args.loss_embeddings_weight,
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.reset_optimizers(optimizer_name) # SGD
        

    def reset_optimizers(self, name):
        self.opt_state_dict = {}
        if name == "Adam":
            self.optimizer = getattr(torch.optim, name)(
                trainable_params(self.model),
                self.local_lr,
            )
        else:
            self.optimizer = getattr(torch.optim, name)(
                trainable_params(self.model),
                self.local_lr,
                self.args.momentum,
                self.args.weight_decay,
            )
        # self.scheduler = MultiStepLR(optimizer=self.optimizer,
        #                               milestones=np.linspace(0, self.args.global_epoch, 5), gamma=0.5)

    def load_dataset(self):
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size, shuffle=True,)
        self.testloader = DataLoader(self.testset, self.args.batch_size, shuffle=False,)

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        # before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            # after = self.evaluate()
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                )

        eval_stats = {"before": before, "after": after}
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.nn.Parameter]):
        personal_parameters = self.init_personal_params_dict
        if self.client_id in self.personal_params_dict.keys():
            personal_parameters = self.personal_params_dict[self.client_id]
        if self.client_id in self.opt_state_dict.keys():
            self.optimizer.load_state_dict(self.opt_state_dict[self.client_id])
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters at the same layers
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        return_diff=True,
        verbose=False,
        train_base_model=True,
    ) -> Tuple[List[torch.nn.Parameter], int, Dict]:
        self.client_id = client_id
        self.load_dataset()
        labels, counts = torch.unique(self.trainset.dataset.targets[self.trainset.indices], return_counts=True)
        self.model.labels = labels
        self.model.labels_frequency = counts / counts.sum()

        self.set_parameters(new_parameters)
        if not train_base_model:
            for p in self.model.base.parameters():
                p.requires_grad_(False)
        eval_stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model,
                                                          include_gaussian_flow=isinstance(self.model.density_model, GaussianFlow))
            ):
                delta[name] = p0.to(self.device) - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                deepcopy(trainable_params(self.model, include_gaussian_flow=isinstance(self.model.density_model, GaussianFlow))),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        self.model.train()

        if isinstance(self.model.density_model, GaussianFlow):
            collected_embeddings = torch.tensor([], device=self.device, dtype=torch.float32)
            collected_labels = torch.tensor([], device=self.device, dtype=torch.float32)

        for _ in range(self.local_epoch):
            all_nlls = []
            all_logprobs = []
            all_losses = []
            for batch_num, (x, y) in enumerate(self.trainloader):
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if isinstance(self.criterion, BayesianLoss) or isinstance(self.criterion, LogMarginalLoss):
                    y_pred, log_prob, embeddings = self.model.train_forward(x, labels=y)
                    loss = self.criterion(y_pred, y, log_prob, embeddings)
                else:
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if isinstance(self.model.density_model, GaussianFlow):
                    collected_embeddings = torch.cat([collected_embeddings, embeddings.detach()], dim=0)
                    collected_labels = torch.cat([collected_labels, y.detach()], dim=0)
                    self.model.density_model.update_params(embeddings=collected_embeddings, labels=collected_labels)

                if self.args.use_wandb:
                    with torch.no_grad():
                        nll = -y_pred.expected_log_likelihood(y)
                        all_nlls.append(nll.mean().cpu().item())
                        all_logprobs.append(log_prob.mean().cpu().item())
                        all_losses.append(loss.cpu().item())

                if self.args.local_batchs > -1 and batch_num >= self.args.local_batchs:
                    break
            if self.args.use_wandb:
                wandb.log({f"Mean logprob of client {self.client_id}": np.mean(all_logprobs)})
                wandb.log({f"Mean nll of client {self.client_id}": np.mean(all_nlls)})
                wandb.log({f"Mean loss of client {self.client_id}": np.mean(all_losses)})

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, Dict[str, float]]:
        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0

        if self.args.loss_name == "bayessian":
            criterion = BayesianLoss(
                entropy_weight=self.args.loss_entropy_weight,
                log_prob_weight=self.args.loss_log_prob_weight,
                reduction="sum"
                                          )
        elif self.args.loss_name == "marginal_ll":
            criterion = LogMarginalLoss(
                entropy_weight=self.args.loss_entropy_weight,
                log_prob_weight=self.args.loss_log_prob_weight,
                reduction="sum"
                                          )
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(self.device)

        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num = evaluate(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evaluate(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.nn.Parameter]
    ):
        self.client_id = client_id
        self.load_dataset()
        labels, counts = torch.unique(self.trainset.dataset.targets[self.trainset.indices], return_counts=True)
        self.model.labels = labels
        self.model.labels_frequency = counts / counts.sum()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

    def finetune(self):
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)

                if isinstance(self.criterion, BayesianLoss):
                    y_pred, log_prob, _ = self.model.train_forward(x)
                    loss = self.criterion(y_pred, y, log_prob)
                else:
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
