import itertools
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import wandb
from PIL import Image as PILImage

from energies import BaseEnergy
from utils.particle_system import interatomic_distance


def visualize(
    energy: BaseEnergy,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    suffix: str = "",
) -> dict:
    try:
        out_dict = energy.visualize(samples, weights=weights)
    except NotImplementedError:
        warnings.warn(
            f"Warning: {energy.__class__.__name__} is not supported for visualization."
            + " Skipping..."
        )
        return {}

    plt.close("all")
    return {k.replace("visualization", f"visualization{suffix}"): v for k, v in out_dict.items()}


### Helper functions for vizualization ###


def sliced_log_reward(x: torch.Tensor, energy: BaseEnergy, dims: tuple) -> torch.Tensor:
    _x = torch.zeros((x.shape[0], energy.ndim))
    _x[:, dims] = x
    return energy.log_reward(_x.to(energy.device)).detach().cpu()


def fig_to_image(fig):
    import numpy as np
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return PILImage.fromarray(rgba).convert("RGB")


def viz_2d_slice(
    energy: BaseEnergy,
    dims: tuple,
    xs: torch.Tensor,
    weights: torch.Tensor | None = None,
    lim=3.0,
    alpha=0.8,
    n_contour_levels=50,
    grid_width_n_points=200,
    clamp_min=-1000.0,
) -> dict:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    x_points_dim1 = torch.linspace(-lim, lim, grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = sliced_log_reward(x_points, energy, dims)
    log_p_x = torch.clamp_min(log_p_x, clamp_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels, zorder=2)

    xs = torch.clamp(xs[:, dims].detach().cpu(), -lim, lim)
    weights = weights.detach().cpu() if weights is not None else None

    # weights are used for the size of the markers
    ax.scatter(
        xs[:, 0],
        xs[:, 1],
        alpha=alpha,
        marker="o",
        s=weights[: len(xs)] * len(xs) * 5 if weights is not None else 5,
        zorder=1,
    )

    return {f"visualization/contour{dims[0]}{dims[1]}": wandb.Image(fig_to_image(fig))}


def viz_energy_hist(energy: BaseEnergy, xs: torch.Tensor) -> dict:
    xs_logr = energy.log_reward(xs)
    gt_xs, gt_xs_logr = energy.cached_sample(xs.shape[0])

    xs, gt_xs = xs.cpu(), gt_xs.cpu()
    xs_logr, gt_xs_logr = xs_logr.cpu(), gt_xs_logr.cpu()

    min_energy = (-gt_xs_logr).min()
    max_energy = (-gt_xs_logr).max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for _logr, color in zip([xs_logr, gt_xs_logr], ["r", "g"]):
        ax.hist(
            torch.clamp(-_logr, min=min_energy, max=max_energy),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    ax.set_xlabel("Energy")
    ax.legend(["generated data", "test data"])
    ax.grid(True)

    return {"visualization/energy_hist": wandb.Image(fig_to_image(fig))}


def viz_interatomic_dist_hist(energy: BaseEnergy, xs: torch.Tensor) -> dict:
    assert energy.is_particle_system
    gt_xs, _ = energy.cached_sample(xs.shape[0])

    if hasattr(energy, "transform"):
        xs, _ = energy.transform(xs)
        gt_xs, _ = energy.transform(gt_xs)

    xs, gt_xs = xs.cpu(), gt_xs.cpu()
    n_particles = xs.shape[1] // 3

    dist_xs = interatomic_distance(xs, n_particles, 3, True).view(-1)
    dist_gt_xs = interatomic_distance(gt_xs, n_particles, 3, True).view(-1)

    bins = 100
    min_dist = 0
    max_dist = 6

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for _xs, color in zip([dist_xs, dist_gt_xs], ["r", "g"]):
        ax.hist(
            torch.clamp(_xs, min=min_dist, max=max_dist),
            bins=bins,
            alpha=0.5,
            density=True,
            histtype="step",
            color=color,
            linewidth=4,
        )
    ax.set_xlabel("Interatomic Distances")
    ax.legend(["generated data", "test data"])
    ax.grid(True)

    return {"visualization/interatomic_dist_hist": wandb.Image(fig_to_image(fig))}
