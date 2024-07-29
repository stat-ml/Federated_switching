import json
import os
import pickle
import random
from argparse import ArgumentParser

import torch
import numpy as np
from path import Path

from datasets import DATASETS
from partition import dirichlet, iid_partition, randomly_assign_classes, allocate_shards
from util import prune_args, generate_synthetic_data, process_celeba, process_femnist, generate_toy_aleatoric_data

_CURRENT_DIR = Path(__file__).parent.abspath()


def main(args):
    dataset_root = _CURRENT_DIR.parent / args.dataset

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    partition = {"separation": None, "data_indices": None}

    if args.dataset == "femnist":
        partition, stats = process_femnist(args)
    elif args.dataset == "celeba":
        partition, stats = process_celeba(args)
    elif args.dataset == "synthetic":
        partition, stats = generate_synthetic_data(args)
    elif args.dataset == "toy_aleatoric":
        partition, stats = generate_toy_aleatoric_data(args)
    else:  # MEDMNIST, COVID, MNIST, CIFAR10, ...
        ori_dataset = DATASETS[args.dataset](dataset_root, args)

        if not args.iid:
            if args.alpha > 0:  # Dirichlet(alpha)
                partition, stats = dirichlet(
                    ori_dataset=ori_dataset,
                    num_clients=args.client_num_in_total,
                    alpha=args.alpha,
                    least_samples=args.least_samples,
                )
            elif args.classes != 0:  # randomly assign classes
                args.classes = max(1, min(args.classes, len(ori_dataset.classes)))
                partition, stats = randomly_assign_classes(
                    ori_dataset=ori_dataset,
                    num_clients=args.client_num_in_total,
                    num_classes=args.classes,
                )
            elif args.shards > 0:  # allocate shards
                partition, stats = allocate_shards(
                    ori_dataset=ori_dataset,
                    num_clients=args.client_num_in_total,
                    num_shards=args.shards,
                )
            else:
                raise RuntimeError(
                    "Please set arbitrary one arg from [--alpha, --classes, --shards] to split the dataset."
                )

        else:  # iid partition
            partition, stats = iid_partition(
                ori_dataset=ori_dataset, num_clients=args.client_num_in_total
            )

    if partition["separation"] is None:
        if args.split == "user":
            train_clients_num = int(args.client_num_in_total * args.fraction)
            clients_4_train = list(range(train_clients_num))
            clients_4_test = list(range(train_clients_num, args.client_num_in_total))
        else:
            clients_4_train = list(range(args.client_num_in_total))
            clients_4_test = list(range(args.client_num_in_total))

        partition["separation"] = {
            "train": clients_4_train,
            "test": clients_4_test,
            "total": args.client_num_in_total,
        }

    if args.dataset not in ["femnist", "celeba", "toy_aleatoric"]:
        for client_id, idx in enumerate(partition["data_indices"]):
            if args.split == "sample":
                num_train_samples = int(len(idx) * args.fraction)
                np.random.shuffle(idx)
                idx_train, idx_test = idx[:num_train_samples], idx[num_train_samples:]
                partition["data_indices"][client_id] = {
                    "train": idx_train,
                    "test": idx_test,
                }
            else:
                if client_id in clients_4_train:
                    partition["data_indices"][client_id] = {"train": idx, "test": []}
                else:
                    partition["data_indices"][client_id] = {"train": [], "test": idx}

    with open(_CURRENT_DIR.parent / args.dataset / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
        json.dump(stats, f)

    with open(_CURRENT_DIR.parent / args.dataset / "args.json", "w") as f:
        json.dump(prune_args(args), f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "toy_circle",
            "toy_noisy",
            "toy_aleatoric",
            "noisy_mnist",
            "noisy_cifar100",
        ],
        default="cifar10",
    )
    parser.add_argument("--iid", type=int, default=0)
    parser.add_argument("-cn", "--client_num_in_total", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("--fraction", type=float, default=0.5)
    # For random assigning classes only
    parser.add_argument("-c", "--classes", type=int, default=0)
    # For toy noisy only
    parser.add_argument("--toy_noisy_classes", type=int, default=0)

    parser.add_argument("--toy_aleatoric_side", type=float, default=10)
    parser.add_argument("--toy_aleatoric_num_points", type=int, default=1000)
    parser.add_argument("--toy_aleatoric_std_dev", type=float, default=0.5)
    # For allocate shards only
    parser.add_argument("-s", "--shards", type=int, default=0)
    # For dirichlet distribution only
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument("-ls", "--least_samples", type=int, default=40)

    # For synthetic data only
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--dimension", type=int, default=60)

    # For CIFAR-100 only
    parser.add_argument("--super_class", type=int, default=0)

    # For EMNIST only
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    args = parser.parse_args()
    main(args)
