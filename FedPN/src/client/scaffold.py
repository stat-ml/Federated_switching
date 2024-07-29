from collections import OrderedDict
from typing import Dict, Iterator, List

import torch

from fedavg import FedAvgClient
from src.config.utils import trainable_params
from src.config.nat_pn.loss import BayesianLoss
from copy import deepcopy


class SCAFFOLDClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)
        self.c_local: Dict[List[torch.Tensor]] = {}
        self.c_global: List[torch.Tensor] = []
        self.iter_trainloader: Iterator[torch.utils.data.DataLoader] = None

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        c_global: List[torch.Tensor],
        return_diff=True,
        verbose=False,
        train_base_model=True,
    ):
        self.client_id = client_id
        self.load_dataset()
        self.iter_trainloader = iter(self.trainloader)
        self.set_parameters(new_parameters)
        if not train_base_model:
            skip_params = 0
            for p in self.model.base.parameters():
                p.requires_grad_(False)
                skip_params += 1
            c_global = c_global[skip_params:]
            self.c_local[self.client_id] = self.c_local[self.client_id][skip_params:]

        self.c_global = c_global
        if self.client_id not in self.c_local.keys():
            self.c_local[self.client_id] = [
                torch.zeros_like(c, device=self.device) for c in c_global
            ]

        stats = self.train_and_log(verbose=verbose)

        if not return_diff:
            return deepcopy(trainable_params(self.model)), None, None
        
        # update local control variate
        with torch.no_grad():

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for x, y_i in zip(new_parameters.values(), trainable_params(self.model)):
                y_delta.append(y_i - x)

            # compute c_plus
            coef = 1 / (self.local_epoch * self.args.local_lr)
            for c, c_i, x, y_i in zip(
                self.c_global,
                self.c_local[self.client_id],
                new_parameters.values(),
                trainable_params(self.model),
            ):
                c_plus.append(c_i - c + coef * (x - y_i))

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local[self.client_id]):
                c_delta.append(c_p - c_l)

            self.c_local[self.client_id] = c_plus

        return y_delta, c_delta, stats

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            if isinstance(self.criterion, BayesianLoss):
                y_pred, log_prob, _ = self.model.train_forward(x)
                loss = self.criterion(y_pred, y, log_prob)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            self.optimizer.zero_grad()
            loss.backward()
            for param, c, c_i in zip(
                trainable_params(self.model),
                self.c_global,
                self.c_local[self.client_id],
            ):
                param.grad.data += c - c_i
            self.optimizer.step()

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
