from copy import deepcopy
from typing import List
from argparse import Namespace

import torch
from tqdm.auto import tqdm

from fedavg import FedAvgServer
from src.client.scaffold import SCAFFOLDClient
from src.config.args import get_scaffold_argparser
from src.config.utils import trainable_params, reset_flow_and_classifier_params
from src.config.utils import OUT_DIR


class SCAFFOLDServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "SCAFFOLD",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_scaffold_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = SCAFFOLDClient(deepcopy(self.model), self.args, self.logger)
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            y_delta_cache = []
            c_delta_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                (
                    y_delta,
                    c_delta,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    c_global=self.c_global,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                y_delta_cache.append(y_delta)
                c_delta_cache.append(c_delta)

            self.aggregate(y_delta_cache, c_delta_cache)

            self.log_info()

    @torch.no_grad()
    def aggregate(
        self,
        y_delta_cache: List[List[torch.Tensor]],
        c_delta_cache: List[List[torch.Tensor]],
    ):
        for param, y_delta in zip(
            trainable_params(self.global_params_dict), zip(*y_delta_cache)
        ):
            x_delta = torch.stack(y_delta, dim=-1).mean(dim=-1).to(self.device)
            param.data += self.args.global_lr * x_delta.to(self.device)

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_cache)):
            c_delta = torch.stack(c_delta, dim=-1).sum(dim=-1).to(self.device)
            c_global.data += (1 / self.client_num_in_total) * c_delta.data

    def post_training_and_save(self, model_name):
        all_models_dict: dict[str | int, torch.nn.ParameterDict] = {}
        all_models_dict["global"] = deepcopy(self.global_params_dict)

        if self.args.finetune_in_the_end > 0:
            self.trainer.local_epoch = self.args.finetune_in_the_end
            for client_id in tqdm(range(self.client_num_in_total)):
                client_local_params = self.generate_client_params(client_id)
                client_local_params = reset_flow_and_classifier_params(client_local_params)
                new_params, _, _ = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    c_global=self.c_global,
                    return_diff=False,
                    verbose=False,
                    train_base_model=False,
                )
                all_models_dict[client_id] = deepcopy(self.trainer.model.state_dict(keep_vars=True))
        text_stopgrad_logp = "_stopgrad_logp" if self.args.stop_grad_logp else ""
        text_stopgrad_embeddings = "_stopgrad_embeddings" if self.args.stop_grad_embeddings else ""
        if self.args.dataset == 'toy_noisy':
            torch.save(all_models_dict, OUT_DIR / self.algo /
                   f"all_params_{self.args.dataset_args['toy_noisy_classes']}{text_stopgrad_logp}{text_stopgrad_embeddings}_{self.args.seed}_{model_name}")
        else:
            torch.save(all_models_dict, OUT_DIR / self.algo /
                    f"all_params{text_stopgrad_logp}{text_stopgrad_embeddings}_{model_name}")


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
