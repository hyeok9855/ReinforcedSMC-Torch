import math
from typing import cast

# import jax
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
                sd, _ = sinkhorn_divergence.sinkhorn_divergence(geom, x=geom.x, y=geom.y)
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


def wasserstein2_squared(
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray | None = None,
    num_iter_max: int = 10_000_000,
) -> tuple[float, bool]:
    """Exact empirical W2^2: the optimal transport cost under a squared-euclidean
    ground metric. Returns ``(w2_squared, converged)``.

    This is 1step_energy_sampler's headline `w2_squared` (their
    `study_metrics.wasserstein_squared_checked`, recorded in each target's
    `complete.json`). Their `metrics.wasserstein2` and our `wasserstein(power=2)`
    both return the *square root* of this quantity -- verified equal to
    float64 precision -- so `2-Wasserstein == sqrt(w2_squared)` exactly.

    `ot.emd2` silently returns a suboptimal value when it exhausts its iteration
    budget, so the convergence flag is reported alongside rather than discarded.
    Their protocol caps at 5e6 iterations; we allow 1e7, which can only help a
    solve that would otherwise hit the cap.
    """
    if x0.ndim > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.ndim > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    a = pot.unif(x0.shape[0]) if weights is None else weights
    b = pot.unif(x1.shape[0])
    M = pot.dist(x0, x1, metric="sqeuclidean")
    value, log = pot.emd2(a, b, M, numItermax=num_iter_max, log=True)
    converged = log.get("result_code") == 1 and log.get("warning") is None
    return float(value), bool(converged)


# ===================================================================
# MMD estimators ported from 1step_energy_sampler
# (energy_sampler/metrics.py::compute_mmd and
#  energy_sampler/metrics_jax.py::_build_mmd_median)
# ===================================================================


def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact squared euclidean distances, [n, m]. mm-based cdist loses too many
    digits in float64 for the median heuristic to be reproducible."""
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").pow(2)


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Biased MMD^2 with an RBF kernel and pooled median-heuristic bandwidth.

    The default metric of 1step_energy_sampler (configs/metrics/default.yaml).
    Bandwidth is the median *squared* pairwise distance over a pooled subsample
    of at most 512 points from each of x and y. Returns ``(mmd2, bandwidth)``.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan"), float("nan")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must both be [N, D] with matching D (got {tuple(x.shape)}, {tuple(y.shape)})"
        )

    with torch.no_grad():
        x, y = x.cpu(), y.cpu()
        sub = min(512, x.shape[0], y.shape[0])
        z = torch.cat([x[:sub], y[:sub]], dim=0)
        dists_sq = torch.pdist(z).pow(2)
        bw = max(torch.median(dists_sq).item() ** 0.5, 1e-3)
        denom = 2 * bw**2
        k_xx = torch.exp(-_pdist2(x, x) / denom).mean()
        k_yy = torch.exp(-_pdist2(y, y) / denom).mean()
        k_xy = torch.exp(-_pdist2(x, y) / denom).mean()
        mmd2 = float((k_xx + k_yy - 2 * k_xy).item())
    return mmd2, bw


def reference_mmd_squared(x, y, bandwidth, block_size=1024):
    """Biased RBF MMD squared with a fixed bandwidth and float64 block reduction."""
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    x, y = x.detach().double(), y.detach().double()

    def average(a, b):
        total = torch.zeros((), dtype=torch.float64, device=a.device)
        for i in range(0, len(a), block_size):
            for j in range(0, len(b), block_size):
                total += torch.exp(
                    -torch.cdist(a[i : i + block_size], b[j : j + block_size]).square()
                    / (2 * bandwidth**2)
                ).sum()
        return total / (len(a) * len(b))

    with torch.no_grad():
        value = float((average(x, x) + average(y, y) - 2 * average(x, y)).item())
    if not math.isfinite(value) or value < -1e-10:
        raise FloatingPointError(f"Invalid reference-bandwidth MMD squared: {value}")
    return max(0.0, value)


def mmd_median(x: torch.Tensor, y: torch.Tensor) -> float:
    """MMD under the VSM reporting protocol (mmdfuse ``mmd_median``, gaussian/L2).

    Port of 1step_energy_sampler's jax implementation, kept paste-faithful to
    the quirks that make the number comparable with published VSM tables:

      * bandwidth = median over the upper triangle of the pooled [X; Y] L2
        distance matrix **including the diagonal zeros**;
      * K_XX / K_YY are summed **with** their unit diagonal but normalised by
        n(n-1), which biases the estimate up by 1/(n-1);
      * the result is ``sqrt(max(1e-20, .))`` -- an MMD, not an MMD^2.

    Computed in float64 to match their ``jax_enable_x64`` setting.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must both be [N, D] with matching D (got {tuple(x.shape)}, {tuple(y.shape)})"
        )
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    with torch.no_grad():
        x = x.detach().double()
        y = y.detach().double()
        z = torch.cat([x, y], dim=0)
        d_zz = _pdist2(z, z).clamp_min(0).sqrt()
        iu = torch.triu_indices(d_zz.shape[0], d_zz.shape[0], offset=0, device=d_zz.device)
        bandwidth = torch.median(d_zz[iu[0], iu[1]])

        def gauss(d2: torch.Tensor) -> torch.Tensor:
            return torch.exp(-d2 / (2 * bandwidth**2))

        k_xx = gauss(_pdist2(x, x))
        k_yy = gauss(_pdist2(y, y))
        k_xy = gauss(_pdist2(x, y))

        mmd2 = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2 * k_xy.mean()
        return float(torch.sqrt(torch.clamp(mmd2, min=1e-20)).item())


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
    mmd_max_samples: int = 2048,
    mmd_bandwidth: float | None = None,
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
    w2_squared, w2_converged = wasserstein2_squared(pred_np, true_np, weights=weights_np)
    sinkhorn = wasserstein(pred_np, true_np, weights=weights_np, method="sinkhorn")
    metrics.update(
        {
            "1-Wasserstein": w1,
            "2-Wasserstein": math.sqrt(w2_squared),
            "w2_squared": w2_squared,
            "w2_converged": w2_converged,
            "Sinkhorn": sinkhorn,
        }
    )

    # Neither MMD estimator takes sample weights, so they are only meaningful
    # for unweighted (already-resampled) particles.
    if weights is None:
        n_mmd = min(mmd_max_samples, pred.shape[0], true.shape[0])
        pred_m, true_m = pred[:n_mmd], true[:n_mmd]
        mmd2, mmd_bw = compute_mmd(pred_m, true_m)
        metrics.update(
            {
                "MMD2": mmd2,
                "MMD2_bandwidth": mmd_bw,
                "MMD_median": mmd_median(pred_m, true_m),
                "MMD_n": n_mmd,
            }
        )
        if mmd_bandwidth is not None:
            # Uses all samples, not the mmd_max_samples subset: the fixed
            # bandwidth is only comparable against the pinned n=10_000 protocol.
            metrics["MMD2_reference"] = reference_mmd_squared(pred, true, mmd_bandwidth)
            metrics["MMD2_reference_n"] = min(pred.shape[0], true.shape[0])

    return metrics
    # if weights is not None:
    #     return metrics

    # mmd_linear = linear_mmd2(pred, true).item()
    # mmd_poly = poly_mmd2(pred, true, d=2, alpha=1.0, c=2.0).item()
    # mmd_rbf = mix_rbf_mmd2(pred, true, sigma_list=[0.01, 0.1, 1, 10, 100]).item()

    # mean_mse, mean_l2, mean_l1 = (
    #     vector_distances(torch.mean(pred, dim=0), torch.mean(true, dim=0))
    # )
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
    gt_log_pfs: torch.Tensor | None = None,
    gt_log_pbs: torch.Tensor | None = None,
    gt_log_rewards: torch.Tensor | None = None,
    gt_init_log_probs: torch.Tensor | None = None,
    gt_log_Z: float | None = None,
) -> dict:
    log_weights = log_rewards + log_pbs.sum(-1) - (log_pfs.sum(-1) + init_log_probs)
    iw_elbo = logmeanexp(log_weights).item()
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
        "elbo": elbo,
        "eubo": eubo,
        "eubo-elbo": eubo - elbo,
        "iw_elbo": iw_elbo,
        "Δ_elbo": (gt_log_Z - elbo) if gt_log_Z is not None else float("nan"),
        "Δ_eubo": (eubo - gt_log_Z) if gt_log_Z is not None else float("nan"),
        "Δ_iw_elbo": (gt_log_Z - iw_elbo) if gt_log_Z is not None else float("nan"),
        "ess(%)": ess / log_pfs.shape[0] * 100,
    }
    return metrics
