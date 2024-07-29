from argparse import Namespace
from typing import List

import torch
import numpy as np
import torchvision
import os
import pandas as pd
from path import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
import math
from sklearn.datasets import make_blobs


def generate_blob_centers_on_circle(radius: float, n_gaussians: int) -> np.ndarray:
    """
    The function generates uniform centers on a circle of a given radius
    :param radius:
    :param n_gaussians:
    :return:
    """
    alphas = np.linspace(0., 2 * np.pi, n_gaussians + 1)[1:]
    centers = []
    for alpha in alphas:
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        centers.append([x, y])
    centers = np.vstack(centers).reshape((-1, 1, 2))
    return centers


def sample_from_gaussians_on_circle(radius: float, n_gaussians: int) -> tuple[np.ndarray, np.ndarray]:
    centers = generate_blob_centers_on_circle(
        radius=radius, n_gaussians=n_gaussians)
    all_samples = []
    all_labels = []
    for i in range(n_gaussians):
        samples = make_blobs(
            n_samples=2000, centers=centers[i], cluster_std=0.1,)
        labels = np.ones_like(samples[1]) * i

        all_samples.append(samples[0])
        all_labels.append(labels)
    return np.vstack(all_samples), np.hstack(all_labels)


def generate_toy_circle(args):
    data, targets = sample_from_gaussians_on_circle(
        radius=1.0,
        n_gaussians=2,
    )

    os.makedirs(args.dataset, exist_ok=True)
    with open(f"../{args.dataset}/data.npy", 'wb') as f:
        np.save(f, data)
    with open(f"../{args.dataset}/targets.npy", 'wb') as f:
        np.save(f, targets)


def generate_toy_noisy(args):

    all_samples = []
    all_labels = []
    centers = np.array([
        [-1., 0.],
        [1., 0.],
        [0., 0.],
    ], dtype=np.float32)
    for i, center in enumerate(centers):
        N = 20000 # if i != 2 else 5000 + int(1000 * (1 + args.toy_noisy_classes / 200))
        print(N)
        samples = make_blobs(
            n_samples=N, centers=center.reshape(1, -1), cluster_std=0.1,)

        all_samples.append(samples[0])
        if i == len(centers) - 1:
            labels = np.random.randint(
                low=0, high=args.toy_noisy_classes, size=samples[0].shape[0])
        else:
            labels = np.ones_like(samples[1]) * i

        all_labels.append(labels)

    data = np.vstack(all_samples)
    targets = np.hstack(all_labels)

    os.makedirs(args.dataset, exist_ok=True)
    with open(f"../{args.dataset}/data_{args.toy_noisy_classes}.npy", 'wb') as f:
        np.save(f, data)
    with open(f"../{args.dataset}/targets_{args.toy_noisy_classes}.npy", 'wb') as f:
        np.save(f, targets)


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.transform = None
        self.target_transform = None
        self.classes: List = None

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.targets)


class ToyCircle(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)

        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            generate_toy_circle(args)

        with open(f'{root}/data.npy', 'rb') as f:
            data = np.load(f)
        with open(f'{root}/targets.npy', 'rb') as f:
            targets = np.load(f)

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))
        self.transform = None
        self.target_transform = target_transform


class ToyNoisy(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)

        if hasattr(args, 'toy_noisy_classes'):
            n_classes = args.toy_noisy_classes
        else:
            n_classes = args['toy_noisy_classes']

        if not os.path.isfile(root / f"data_{n_classes}.npy") or not os.path.isfile(
            root / f"targets_{n_classes}.npy"
        ):
            generate_toy_noisy(args)

        with open(f"{root}/data_{n_classes}.npy", 'rb') as f:
            data = np.load(f)
        with open(f"{root}/targets_{n_classes}.npy", 'rb') as f:
            targets = np.load(f)

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))
        self.transform = None
        self.target_transform = target_transform


class FEMNIST(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.transform = transform
        self.target_transform = target_transform


class Synthetic(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))
        self.transform = transform
        self.target_transform = target_transform


class CelebA(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).permute([0, -1, 1, 2]).float()
        self.targets = torch.from_numpy(targets).long()
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [0, 1]


class MedMNIST(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy")
                         ).float().unsqueeze(1)
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(11))


class COVID19(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy"))
            .permute([0, -1, 1, 2])
            .float()
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [0, 1, 2, 3]


class USPS(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(
            root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(
            root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.transform = transform
        self.target_transform = target_transform


class SVHN(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(
            root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(
            root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.transform = transform
        self.target_transform = target_transform


class MNIST(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.MNIST(
            root, True, transform, target_transform, download=True
        )
        test_part = torchvision.datasets.MNIST(
            root, False, transform, target_transform)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform


class NoisyMNIST(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.MNIST(
            root, True, transform, target_transform, download=True
        )
        test_part = torchvision.datasets.MNIST(
            root, False, transform, target_transform)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        # Apply label noise for specific classes
        noisy_classes = [5, 6, 7, 8, 9]  # Classes to introduce label noise
        noisy_indices = torch.tensor([
            i for i in range(len(train_targets)) if train_targets[i] in noisy_classes
        ])
        noisy_targets = torch.randint(min(noisy_classes), max(noisy_classes) + 1, size=noisy_indices.size())
        train_targets[noisy_indices] = noisy_targets

        # Apply label noise for specific classes
        noisy_indices = torch.tensor([
            i for i in range(len(test_targets)) if test_targets[i] in noisy_classes
        ])
        noisy_targets = torch.randint(min(noisy_classes), max(noisy_classes) + 1, size=noisy_indices.size())
        test_targets[noisy_indices] = noisy_targets


        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform


class NoisyCIFAR100(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(
            root, True, transform, target_transform, download=True
        )
        test_part = torchvision.datasets.CIFAR100(
            root, False, transform, target_transform)
        train_data = torch.Tensor(
            train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        # Apply label noise for specific classes
        noisy_classes = [i for i in range(10, 100)]  # Classes to introduce label noise
        noisy_indices = torch.tensor([
            i for i in range(len(train_targets)) if train_targets[i] in noisy_classes
        ])
        noisy_targets = torch.randint(min(noisy_classes), max(noisy_classes) + 1, size=noisy_indices.size())
        train_targets[noisy_indices] = noisy_targets

        # Apply label noise for specific classes
        noisy_indices = torch.tensor([
            i for i in range(len(test_targets)) if test_targets[i] in noisy_classes
        ])
        noisy_targets = torch.randint(min(noisy_classes), max(noisy_classes) + 1, size=noisy_indices.size())
        test_targets[noisy_indices] = noisy_targets


        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform
        


class FashionMNIST(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.FashionMNIST(
            root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(
            root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform


class EMNIST(BaseDataset):
    def __init__(self, root, args, transform=None, target_transform=None):
        super().__init__()
        split = None
        if isinstance(args, Namespace):
            split = args.emnist_split
        elif isinstance(args, dict):
            split = args["emnist_split"]
        train_part = torchvision.datasets.EMNIST(
            root, split=split, train=True, download=True
        )
        test_part = torchvision.datasets.EMNIST(
            root, split=split, train=False, download=True
        )
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform


class CIFAR10(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=False) # True
        test_part = torchvision.datasets.CIFAR10(root, False, download=False) # True
        train_data = torch.Tensor(
            train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform


class CIFAR100(BaseDataset):
    def __init__(self, root, args, transform=None, target_transform=None):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(
            train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform
        super_class = None
        if isinstance(args, Namespace):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in self.targets:
                new_targets.append(mapping[self.classes[cls]])
            self.targets = torch.tensor(new_targets, dtype=torch.long)


class TinyImagenet(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = pd.read_table(
            root / "raw/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()

        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            mapping = dict(zip(self.classes, list(range(len(self.classes)))))
            data = []
            targets = []
            for cls in os.listdir(root / "raw" / "train"):
                for img_name in os.listdir(root / "raw" / "train" / cls / "images"):
                    img = pil_to_tensor(
                        Image.open(root / "raw" / "train" /
                                   cls / "images" / img_name)
                    ).float()
                    if img.shape[0] == 1:
                        img = torch.expand_copy(img, [3, 64, 64])
                    data.append(img)
                    targets.append(mapping[cls])

            table = pd.read_table(
                root / "raw/val/val_annotations.txt",
                sep="\t",
                engine="python",
                header=None,
            )
            test_classes = dict(zip(table[0].tolist(), table[1].tolist()))
            for img_name in os.listdir(root / "raw" / "val" / "images"):
                img = pil_to_tensor(
                    Image.open(root / "raw" / "val" / "images" / img_name)
                ).float()
                if img.shape[0] == 1:
                    img = torch.expand_copy(img, [3, 64, 64])
                data.append(img)
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long),
                       root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.transform = transform
        self.target_transform = target_transform


class CINIC10(BaseDataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            data = []
            targets = []
            mapping = dict(zip(self.classes, range(10)))
            for folder in ["test", "train", "valid"]:
                for cls in os.listdir(Path(root) / "raw" / folder):
                    for img_name in os.listdir(root / "raw" / folder / cls):
                        img = pil_to_tensor(
                            Image.open(root / "raw" / folder / cls / img_name)
                        ).float()
                        if img.shape[0] == 1:
                            img = torch.expand_copy(img, [3, 32, 32])
                        data.append(img)
                        targets.append(mapping[cls])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long),
                       root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.transform = transform
        self.target_transform = target_transform


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "emnist": EMNIST,
    "fmnist": FashionMNIST,
    "femnist": FEMNIST,
    "medmnistS": MedMNIST,
    "medmnistC": MedMNIST,
    "medmnistA": MedMNIST,
    "covid19": COVID19,
    "celeba": CelebA,
    "synthetic": Synthetic,
    "svhn": SVHN,
    "usps": USPS,
    "tiny_imagenet": TinyImagenet,
    "cinic10": CINIC10,
    "toy_circle": ToyCircle,
    "toy_noisy": ToyNoisy,
    "noisy_mnist": NoisyMNIST,
    "noisy_cifar100": NoisyCIFAR100,
}
