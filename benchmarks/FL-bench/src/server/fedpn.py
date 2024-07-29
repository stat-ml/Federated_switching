from argparse import ArgumentParser, Namespace
from copy import deepcopy
import argparse

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedpn import FedPNClient
import wandb


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_fedpn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--loss_entropy_weight", type=float, default=0.0)
    parser.add_argument("--loss_log_prob_weight", type=float, default=1.0)
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    return parser


class FedPNServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedPN",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedpn_argparser().parse_args()
        args.model = 'natpn'
        args.save_model = 1
        args.global_epoch = 10
        args.local_epoch = 1
        args.join_ratio = 1.0
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedPNClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = FedPNServer()
    if server.args.use_wandb:
        run = wandb.init(project="FedPN_benchbark", config=server.args)
    server.run()
