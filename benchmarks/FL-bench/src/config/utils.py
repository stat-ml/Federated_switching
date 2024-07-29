import os
import random
from copy import deepcopy
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import pynvml
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console

from data.utils.datasets import BaseDataset

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = PROJECT_DIR / "out"
TEMP_DIR = PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_device(use_cuda: bool) -> torch.device:
    """Dynamically select the vacant CUDA device for running FL experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    # This function is modified by the `get_best_gpu()` in https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/functional.py
    # Shout out to FedLab, which is an incredible FL framework!
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters


def vectorize(
    src: Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    """For evaluating the `model` over `dataloader` and return the result calculated by `criterion`.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Tuple[float, float, int]: [metric, correct, sample num]
    """
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
    return loss, correct, sample_num


def count_labels(
    dataset: BaseDataset, indices: List[int] = None, min_value=0
) -> List[int]:
    """For counting number of labels in `dataset.targets`.

    Args:
        dataset (BaseDataset): Target dataset.
        indices (List[int]): the subset indices. Defaults to all indices of `dataset` if not specified.
        min_value (int, optional): The minimum value for each label. Defaults to 0.

    Returns:
        List[int]: The number of each label.
    """
    if indices is None:
        indices = list(range(len(dataset.targets)))
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()


@torch.no_grad()
def modified_evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    """For evaluating the `model` over `dataloader` and return the result calculated by `criterion`.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Tuple[float, float, int]: [metric, correct, sample num]
    """
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    predictions = []
    labels = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)

        predictions.append(pred.cpu().numpy())
        labels.append(y.cpu().numpy())
    return loss, correct, sample_num, np.hstack(predictions), np.hstack(labels)


@torch.no_grad()
def evaluate_switch(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float,
    device=torch.device("cpu"),
    return_predictions: bool = False
) -> tuple[float, float, int]:
    local_model.eval()
    global_model.eval()
    correct_local = 0
    correct_global = 0
    correct_decision = 0
    if return_predictions:
        local_predictions = []
        global_predictions = []
        switch_predictions = []
        true_labels = []
        uncertainty_scores_local = []
        uncertainty_scores_global = []
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred_local, _, _ = local_model.train_forward(x)
        y_pred_global, _, _ = global_model.train_forward(x)

        alpha_local = y_pred_local.alpha
        alpha_global = y_pred_global.alpha

        threshold_torch = threshold * torch.ones_like(alpha_local[:, 0])

        local_measure = y_pred_local.entropy().cpu().numpy()
        global_measure = y_pred_global.entropy().cpu().numpy()

        local_measure = torch.tensor(
            local_measure, device=device) * torch.ones_like(alpha_local[:, 0])

        alphas_after_decision = torch.where((local_measure < threshold_torch)[..., None],
                                            alpha_local, alpha_global)

        pred_local = torch.argmax(alpha_local, -1)
        pred_global = torch.argmax(alpha_global, -1)
        pred_after_decision = torch.argmax(alphas_after_decision, -1)

        correct_local += (pred_local == y).sum().item()
        correct_global += (pred_global == y).sum().item()
        correct_decision += (pred_after_decision == y).sum().item()

        if return_predictions:
            switch_predictions.append(pred_after_decision.cpu())
            global_predictions.append(pred_global.cpu())
            local_predictions.append(pred_local.cpu())
            true_labels.append(y.cpu())
            uncertainty_scores_local.append(local_measure.cpu())
            uncertainty_scores_global.append(global_measure)

        sample_num += len(y)
    if return_predictions:
        return correct_decision, correct_local, correct_global, sample_num, local_predictions, global_predictions, switch_predictions, true_labels, uncertainty_scores_local, uncertainty_scores_global
    else:
        return correct_decision, correct_local, correct_global, sample_num
