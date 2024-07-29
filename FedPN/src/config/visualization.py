from src.config.models import NatPnModel
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_density_slice_2d(
        slice_coord: float,
        y_fixed: bool,
        limits: list[float],
        n_steps: int,
        models: list[NatPnModel],
        device,
) -> None:
    if y_fixed:
        np_points = np.array([[x, slice_coord]
                             for x in np.linspace(limits[0], limits[1], n_steps)])
    else:
        np_points = np.array([[slice_coord, x]
                             for x in np.linspace(limits[0], limits[1], n_steps)])

    points = torch.tensor(np_points, dtype=torch.float32)

    log_probs_all = []
    with torch.no_grad():
        for model in models:
            log_probs = []

            input_point = points.to(device)
            _, log_probs, _ = model.train_forward(input_point)

            log_probs_all.append(log_probs[None])

    log_probs_all = np.vstack(log_probs_all)

    probs_mean = np.mean(np.exp(log_probs_all), axis=0)
    probs_std = np.std(np.exp(log_probs_all), axis=0)


    plt.close()
    plt.figure(figsize=(10, 8), dpi=150)

    if y_fixed:
        plt.plot(np_points[:, 0], probs_mean, label="Mean")
        plt.fill_between(np_points[:, 0], np.maximum(0, probs_mean - probs_std),
                          probs_mean + probs_std, alpha=0.2, label="Std dev")
    else:
        plt.plot(np_points[:, 1], probs_mean, label="Mean")
        plt.fill_between(np_points[:, 1], np.maximum(0, probs_mean - probs_std),
                         probs_mean + probs_std, alpha=0.2, label="Std dev")

    plt.xlim(tuple(limits))
    plt.legend()
    plt.tight_layout()
    plt.show()
