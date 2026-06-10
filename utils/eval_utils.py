import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import ot as pot
import torch
from ott.geometry import pointcloud

# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn

from utils.misc_utils import logmeanexp

MIN_VAR_EST = 1e-8


# # Credit: https://github.com/anonymous3141/SCLD/blob/master/eval/optimal_transport.py
# @jax.jit
# def compute_OT(
#     gt_samples: jax.Array,
#     model_samples: jax.Array,
#     a: jax.Array,
#     b: jax.Array,
#     epsilon: float = 1e-3,
#     entropy_reg: bool = True,
# ) -> jax.Array:
#     """
#     Entropy regularized optimal transport cost (see https://ott-jax.readthedocs.io/en/latest/tutorials/point_clouds.html)

#     Args:
#         gt_samples: Ground truth samples
#         model_samples: Model samples
#         a: Source distribution weights
#         b: Target distribution weights
#         epsilon: Entropy regularization parameter (static)
#         entropy_reg: Whether to use entropy regularization (static)
#     """
#     geom = pointcloud.PointCloud(gt_samples, model_samples, epsilon=epsilon)
#     ot_prob = linear_problem.LinearProblem(geom, a=a, b=b)
#     solver = sinkhorn.Sinkhorn()
#     ot = solver(ot_prob)

#     # More JAX-friendly way to handle the cost computation
#     reg_cost = ot.reg_ot_cost
#     unreg_cost = jnp.sum(ot.matrix * ot.geom.cost_matrix)
#     return jnp.where(entropy_reg, reg_cost, unreg_cost)  # type: ignore


# # Create a specialized version with static arguments
# compute_OT_static = jax.jit(compute_OT, static_argnames=("epsilon", "entropy_reg"))


def wasserstein(
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "emd",
    power: int = 2,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix

    a = pot.unif(x0.shape[0]) if weights is None else weights
    b = pot.unif(x1.shape[0])
    if x0.ndim > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.ndim > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    if method == "emd":
        M = pot.dist(x0, x1, metric="euclidean" if power == 1 else "sqeuclidean")
        ret = cast(float, pot.emd2(a, b, M, numItermax=10_000_000))
        ret = math.sqrt(ret)  # To make it consistent with previous works
    elif method == "sinkhorn":
        # ret = compute_OT_static(
        #     gt_samples=jnp.array(x0),
        #     model_samples=jnp.array(x1),
        #     a=jnp.array(a),
        #     b=jnp.array(b),
        #     epsilon=1e-3,
        #     entropy_reg=True,
        # )
        # ret = float(ret)

        from ott.tools import sinkhorn_divergence

        class SD:
            def __init__(self, gt_samples, epsilon=1e-3):
                self.groundtruth = gt_samples
                self.epsilon = epsilon

            def compute_SD(self, model_samples):
                """
                Entropy regularized debiased optimal transport (Sinkhorn divergence - SD) cost (see https://ott-jax.readthedocs.io/en/latest/tutorials/point_clouds.html)
                """

                geom = pointcloud.PointCloud(self.groundtruth, model_samples, epsilon=1e-3)

                sd, _ = sinkhorn_divergence.sinkhorn_divergence(
                    geom,
                    x=geom.x,
                    y=geom.y,
                )

                return sd

        ret = SD(jnp.array(x0), epsilon=1e-3).compute_SD(jnp.array(x1))
        ret = float(ret)

    else:
        raise ValueError(f"Unknown method: {method}")

    return ret


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = torch.zeros_like(exponent)
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=MIN_VAR_EST))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est


def vector_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def distribution_distance_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    weights: torch.Tensor | None = None,
):
    """
    computes distances between distributions.
    pred: [batch, dims] tensor
    true: [batch, dims] tensor
    """
    metrics = {}

    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    weights_np = weights.cpu().numpy() if weights is not None else None

    w1 = wasserstein(pred_np, true_np, weights=weights_np, power=1)
    w2 = wasserstein(pred_np, true_np, weights=weights_np, power=2)
    sinkhorn = wasserstein(pred_np, true_np, weights=weights_np, method="sinkhorn")
    metrics.update({"1-Wasserstein": w1, "2-Wasserstein": w2, "Sinkhorn": sinkhorn})

    return metrics
    # if weights is not None:
    #     return metrics

    # mmd_linear = linear_mmd2(pred, true).item()
    # mmd_poly = poly_mmd2(pred, true, d=2, alpha=1.0, c=2.0).item()
    # mmd_rbf = mix_rbf_mmd2(pred, true, sigma_list=[0.01, 0.1, 1, 10, 100]).item()

    # mean_mse, mean_l2, mean_l1 = vector_distances(torch.mean(pred, dim=0), torch.mean(true, dim=0))
    # median_mse, median_l2, median_l1 = vector_distances(
    #     torch.median(pred, dim=0)[0], torch.median(true, dim=0)[0]
    # )

    # metrics.update(
    #     {
    #         "Linear_MMD": mmd_linear,
    #         "Poly_MMD": mmd_poly,
    #         "RBF_MMD": mmd_rbf,
    #         "Mean_MSE": mean_mse,
    #         "Mean_L2": mean_l2,
    #         "Mean_L1": mean_l1,
    #         "Median_MSE": median_mse,
    #         "Median_L2": median_l2,
    #         "Median_L1": median_l1,
    #     }
    # )
    # return metrics


def density_metrics(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_rewards: torch.Tensor,
    init_log_probs: torch.Tensor,
    log_Z: torch.Tensor,
    gt_log_pfs: torch.Tensor | None = None,
    gt_log_pbs: torch.Tensor | None = None,
    gt_log_rewards: torch.Tensor | None = None,
    gt_init_log_probs: torch.Tensor | None = None,
    gt_log_Z: float | None = None,
) -> dict:
    log_weights = log_rewards + log_pbs.sum(-1) - (log_pfs.sum(-1) + init_log_probs)
    iw_elbo = logmeanexp(log_weights).item()
    log_Z_learned = log_Z.item()
    elbo = log_weights.mean().item()
    if gt_log_rewards is not None:
        assert (gt_log_pfs is not None) and (gt_log_pbs is not None)
        eubo = (
            (gt_log_rewards + gt_log_pbs.sum(-1) - (gt_log_pfs.sum(-1) + gt_init_log_probs))
            .mean()
            .item()
        )
    else:
        eubo = float("nan")
    ess = 1.0 / (log_weights.softmax(0) ** 2).sum().item()
    metrics = {
        "log_Z_learned": log_Z_learned,
        "elbo": elbo,
        "eubo": eubo,
        "eubo-elbo": eubo - elbo,
        "iw_elbo": iw_elbo,
        "Δ_elbo": (gt_log_Z - elbo) if gt_log_Z is not None else float("nan"),
        "Δ_eubo": (gt_log_Z - eubo) if gt_log_Z is not None else float("nan"),
        "Δ_iw_elbo": (gt_log_Z - iw_elbo) if gt_log_Z is not None else float("nan"),
        "ess(%)": ess / log_pfs.shape[0] * 100,
    }
    return metrics
