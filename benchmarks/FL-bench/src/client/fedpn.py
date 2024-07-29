from fedavg import FedAvgClient
from src.config.nat_pn.loss import BayesianLoss
import torch
import wandb
import numpy as np
from src.config.flows.utils_flow import process_flow_batch


class FedPNClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.personal_params_name = [
            name for name in self.model.state_dict().keys() if (("classifier" in name) or ("density_model" in name))
        ]
        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }
        self.criterion = BayesianLoss(
            entropy_weight=self.args.loss_entropy_weight,
            log_prob_weight=self.args.loss_log_prob_weight,
        ).to(self.device)

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            all_nlls = []
            all_logprobs = []
            all_losses = []

            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)

                y_pred, log_prob, _ = self.model.train_forward(x)
                loss = self.criterion(y_pred, y, log_prob)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.args.use_wandb:
                    with torch.no_grad():
                        nll = -y_pred.expected_log_likelihood(y)
                        all_nlls.append(nll.mean().cpu().item())
                        all_logprobs.append(log_prob.mean().cpu().item())
                        all_losses.append(loss.cpu().item())
            if self.args.use_wandb:
                wandb.log(
                    {f"Mean logprob of client {self.client_id}": np.mean(all_logprobs)})
                wandb.log(
                    {f"Mean nll of client {self.client_id}": np.mean(all_nlls)})
                wandb.log(
                    {f"Mean loss of client {self.client_id}": np.mean(all_losses)})

    def finetune(self):
        self.model.train()

        # fine-tune the classifier and density models only
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                y_pred, log_prob, _ = self.model.train_forward(x)
                loss = self.criterion(y_pred, y, log_prob)

                self.optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                for name, param in self.model.named_parameters():
                    if name not in self.personal_params_name:
                        if param.grad is not None:
                            param.grad.zero_()
                self.optimizer.step()

                
            print(np.mean(np.hstack(log_prob.cpu().detach().numpy())))
