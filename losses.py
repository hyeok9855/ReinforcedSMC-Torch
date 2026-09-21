import numpy as np
import torch
import torch.nn.functional as F

LOGR_FLOOR = -1e10


def tb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    logZ_huber_delta: float = 0.0,
) -> torch.Tensor:
    log_iws = log_pfs.sum(-1) - log_fs[:, -1] - log_pbs.sum(-1)
    if logZ_huber_delta <= 0.0:
        return (log_iws + log_fs[:, 0]) ** 2

    policy_discrepancy = log_iws + log_fs[:, 0].detach()
    policy_losses = policy_discrepancy**2

    logZ_discrepancy = log_iws.detach() + log_fs[:, 0]
    logZ_losses = 2.0 * F.huber_loss(
        logZ_discrepancy,
        torch.zeros_like(logZ_discrepancy),
        delta=logZ_huber_delta,
        reduction="none",
    )
    return policy_losses + logZ_losses


def logvar_loss(
    log_pfs: torch.Tensor,  # (bs, T)
    log_pbs: torch.Tensor,  # (bs, T)
    log_r: torch.Tensor,  # (bs,)
    init_log_probs: torch.Tensor,
) -> torch.Tensor:
    rnd = log_r + log_pbs.sum(-1) - (init_log_probs + log_pfs.sum(-1))  # (bs,)
    return (rnd - rnd.mean(dim=0, keepdim=True)) ** 2  # (bs,)


def db_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, :-1] + log_pfs - log_fs[:, 1:] - log_pbs
    return (db_discrepancy**2).mean(-1)


def subtb_lambda_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    coef_matrix: torch.Tensor,  # (T+1, T+1)
) -> torch.Tensor:
    diff_logp = log_pfs - log_pbs  # (bs, T)
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)  # (bs, T+1, T+1)
    A2 = log_fs.unsqueeze(2) - log_fs.unsqueeze(1) + A1  # (bs, T+1, T+1)
    A2 = torch.triu(A2, diagonal=1) ** 2
    subtb_losses = (A2 * coef_matrix.unsqueeze(0)).sum((1, 2))
    return subtb_losses


def subtb_chunk_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    bs, T = log_pfs.shape
    n_chunks = T // chunk_size

    log_pfs_over_pbs = log_pfs - log_pbs
    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs[:, chunk_size::chunk_size]
    )
    log_pfs_over_pbs_cumsum = torch.flip(
        torch.cumsum(torch.flip(log_pfs_over_pbs, dims=[-1]), dim=-1), dims=[-1]
    )
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size] + log_pfs_over_pbs_cumsum[:, ::chunk_size] - log_fs[:, [-1]]
    ) / torch.arange(n_chunks, 0, -1, device=log_fs.device).unsqueeze(0)

    subtb_discrepancy = torch.cat([subtb_discrepancy1, subtb_discrepancy2], dim=-1)
    return (subtb_discrepancy**2).mean(-1)


def tb_subtb_loss(  # TB + SubTB (chunk)
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    chunk_size: int,
    subtb_weight: float = 1.0,
    logZ_huber_delta: float = 0.0,
) -> torch.Tensor:
    # TB for log_Z, log_pfs, and log_pbs
    tb_losses = tb_loss(log_pfs, log_pbs, log_fs, logZ_huber_delta)
    # SubTB for log_fs (intermediate flows)
    log_fs[:, 0] = log_fs[:, 0].detach()
    subtb_losses = subtb_chunk_loss(log_pfs.detach(), log_pbs.detach(), log_fs, chunk_size)
    return subtb_weight * subtb_losses + tb_losses


def get_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    init_log_probs: torch.Tensor,
    log_Z: torch.Tensor,
    invtemp: float = 1.0,
    subtb_coef_matrix: torch.Tensor | None = None,
    subtb_chunk_size: int = 0,
    subtb_weight: float = 1.0,
    logZ_huber_delta: float = 0.0,  # only affects the TB term (`tb` and `tb-subtb`)
    ndim: int | None = None,
) -> torch.Tensor:
    # Avoid in-place mutation
    first_col = (log_Z + init_log_probs).unsqueeze(1)
    middle_cols = log_fs[:, 1:-1]
    last_col = torch.nan_to_num(
        log_fs[:, -1] * invtemp, nan=LOGR_FLOOR, neginf=LOGR_FLOOR, posinf=-LOGR_FLOOR
    ).unsqueeze(1)
    log_fs = torch.cat([first_col, middle_cols, last_col], dim=1)

    if loss_type == "tb":
        losses = tb_loss(log_pfs, log_pbs, log_fs, logZ_huber_delta)
    elif loss_type == "logvar":
        losses = logvar_loss(log_pfs, log_pbs, log_fs[:, -1], init_log_probs)
    elif loss_type == "db":
        losses = db_loss(log_pfs, log_pbs, log_fs)
    elif loss_type == "subtb":
        if subtb_chunk_size > 0:  # Chunk-based subtb
            losses = subtb_chunk_loss(log_pfs, log_pbs, log_fs, subtb_chunk_size)
        else:
            assert subtb_coef_matrix is not None
            losses = subtb_lambda_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix)
    elif loss_type == "tb-subtb":
        losses = tb_subtb_loss(
            log_pfs, log_pbs, log_fs, subtb_chunk_size, subtb_weight, logZ_huber_delta
        )
    elif loss_type == "rev_kl":
        assert ndim is not None
        losses = (1 / ndim) * ((log_pfs.sum(-1) + init_log_probs) - log_pbs.sum(-1) - log_fs[:, -1])
    else:
        raise ValueError(f"Invalid training loss: {loss_type}")

    return losses


def cal_subtb_coef_matrix(lamda: float, T: int) -> torch.Tensor:
    """
    diff_matrix: (T+1, T+1)
     0,  1,  2, ...,   T
    -1,  0,  1, ..., T-1
    -2, -1,  0, .... T-2
    ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    assert lamda >= 0
    if lamda == 0:  # DB
        ones = torch.ones(T + 1, T + 1)
        coef = torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2)
        coef = coef / T
    elif lamda == float("inf"):  # TB if lambda is inf
        coef = torch.zeros(T + 1, T + 1)
        coef[0, -1] = 1.0
    else:
        range_vals = torch.arange(T + 1)
        diff_matrix = range_vals - range_vals.view(-1, 1)
        B = np.log(lamda) * diff_matrix
        B[diff_matrix <= 0] = -np.inf
        log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
        coef = torch.exp(B - log_total_lambda)
    return coef
