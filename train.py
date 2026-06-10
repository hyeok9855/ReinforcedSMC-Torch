import argparse
import os

import torch
import wandb
from tqdm import trange

from buffers import TerminalStateBuffer
from energies import get_energy
from mcmcs import MALA, MD
from models import GFN
from models.modules import get_module
from trainer import Trainer
from utils.misc_utils import get_name, set_seed
from utils.sampling_utils import get_sampling_func
from utils.train_utils import get_gfn_optimizer


def train(args):
    if "SLURM_PROCID" in os.environ:
        args.seed += int(os.environ["SLURM_PROCID"])
    set_seed(args.seed)

    if args.precision == "double":
        torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    energy = get_energy(args, device, seed=args.seed, n_threads=args.n_threads)
    exp_name = get_name(args)

    wandb.init(
        project=f"GFN-Diffusion-{args.energy_name}-{energy.ndim}d",
        config=args.__dict__,
        name=exp_name,
        tags=[f"seed{args.seed}"],
        mode="disabled" if args.disable_wandb else "online",
    )

    #########################
    # Initialize components #
    #########################

    module = get_module(args, energy)

    gfn_model = GFN(
        energy=energy,
        module=module,
        num_steps=args.num_steps,
        reference_process=args.reference_process,
        # --- Pinned Brownian Args --- #
        t_scale=args.t_scale,
        # --- OU Args --- #
        init_std=args.init_std,
        noise_scale=args.noise_scale,
        # --- SubTB Args --- #
        partial_energy=args.partial_energy,
        learn_beta=args.learn_beta,
        device=device,
    ).to(device)

    gfn_optimizer, gfn_scheduler = get_gfn_optimizer(
        gfn_model,
        lr_fwd=args.lr_fwd,
        lr_bwd=args.lr_bwd,
        lr_flow=args.lr_flow,
        lr_logZ=args.lr_logZ,
        lr_beta=args.lr_beta,
        lr_lgv=args.lr_lgv,
        use_weight_decay=args.use_weight_decay,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        milestones=[int(args.epochs * m) for m in args.milestones],
        gamma=args.gamma,
    )

    buffer = mcmc = None
    if args.use_buffer:
        buffer = TerminalStateBuffer(
            args.buffer_size,
            device,
            prioritization=args.prioritization,
            sampling_func=get_sampling_func(args.buffer_sampling, args.rank_k),
            logr_lb=args.logr_lb,
            target_ess=args.buffer_target_ess,
        )

        if args.mcmc_type != "none":
            mcmc_args = {
                "energy": energy,
                "n_steps": args.mcmc_n_steps,
                "burn_in": args.mcmc_burn_in,
                "thinning": args.mcmc_thinning,
                "step_size": args.mcmc_step_size,
                "gamma": args.mcmc_gamma,  # For MD
            }
            if args.mcmc_type == "md":
                mcmc = MD(**mcmc_args)
            elif args.mcmc_type == "mala":
                mcmc = MALA(**mcmc_args)
            else:
                raise ValueError(f"Invalid MCMC type: {args.mcmc_type}")

    trainer = Trainer(
        energy=energy,
        gfn_model=gfn_model,
        optimizer=gfn_optimizer,
        scheduler=gfn_scheduler,
        clip_grad_norm=args.clip_grad_norm,
        loss_type=args.loss_type,
        subtb_lambda=args.subtb_lambda,
        subtb_chunk_size=args.subtb_chunk_size,
        n_epochs=args.epochs,
        bwd_to_fwd_ratio=args.bwd_to_fwd_ratio,
        buffer=buffer,
        prefill_epochs=args.prefill_epochs,
        batch_size=args.batch_size,
        smc=args.smc,
        smc_sampling_func=get_sampling_func(args.smc_sampling),
        smc_resample_threshold=args.smc_resample_threshold,
        smc_target_ess=args.smc_target_ess,
        smc_freq=args.smc_freq,
        mcmc=mcmc,
        mcmc_freq=args.mcmc_freq,
        mcmc_batch_size=args.mcmc_batch_size,
        invtemp=args.invtemp,
        invtemp_anneal=args.invtemp_anneal,
        init_log_Z=args.init_log_Z,
        eval_batch_size=args.eval_batch_size,
        plot_gt=args.plot_gt,
        plot_t_idx=args.plot_t_idx,
    )

    ######################
    # Main training loop #
    ######################

    pbar = trange(args.epochs, desc="[Train]", dynamic_ncols=True)
    eubo_cache = elbo_cache = ess_cache = float("nan")
    for it in pbar:
        metrics = dict()

        ### Eval and plot###
        if it % args.eval_freq == 0:
            metrics.update(
                trainer.eval_and_plot(
                    data_size=args.eval_data_size,
                    full_eval=True if (it % args.full_eval_freq == 0 and args.full_eval) else False,
                    plot=args.plot if it % args.plot_freq == 0 else False,
                )
            )
            eubo_cache = metrics["eval/eubo"]
            elbo_cache = metrics["eval/elbo"]
            ess_cache = metrics["eval/ess(%)"]

        ### Train ###
        metrics["train/loss"] = trainer.train_step(it)
        pbar.set_postfix(
            {
                "Loss": metrics["train/loss"],
                "EUBO": eubo_cache,
                "ELBO": elbo_cache,
                "ESS": ess_cache,
            }
        )

        ### Log ###
        wandb.log(metrics, step=it)

    ### Final eval and plot ###
    final_metrics = trainer.eval_and_plot(
        data_size=args.final_eval_data_size,
        full_eval=True if args.full_eval else False,
        final_eval=True,
        plot=args.plot,
    )
    wandb.log(final_metrics, step=args.epochs)
    desc = ""
    if final_metrics.get("final_eval/eubo-elbo") is not None:
        desc += f"{'EUBO-ELBO':<10}: {final_metrics['final_eval/eubo-elbo']:.3f}\n"
    else:
        desc += f"{'ELBO':<10}: {final_metrics['final_eval/elbo']:.3f}\n"
    if final_metrics.get("final_eval/Sinkhorn") is not None:
        desc += f"{'Sinkhorn':<10}: {final_metrics['final_eval/Sinkhorn']:.3f}\n"
    if final_metrics.get("final_eval/ess(%)") is not None:
        desc += f"{'ESS':<10}: {final_metrics['final_eval/ess(%)']:.3f}\n"
    print(f"===============\n[Final results]\n{desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_name",
        type=str,
        default="gmm40",
        choices=(
            "25gmm",
            "gmm40",
            "student_t_mixture",
            "manywell",
            "funnel",
            "lgcp",
            "lj13",
            "lj55",
            "aldp",
        ),
    )
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--n_threads", type=int, default=16)  # only for ALDP
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="tb",
        choices=("tb", "logvar", "db", "subtb", "tb-subtb", "rev_kl", "mle"),
    )
    parser.add_argument("--subtb_lambda", type=float, default=2.0)
    parser.add_argument("--subtb_chunk_size", type=int, default=4)

    parser.add_argument("--lr_fwd", type=float, default=1e-3)
    parser.add_argument("--lr_bwd", type=float, default=None)
    parser.add_argument("--lr_logZ", type=float, default=1e-1)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--lr_beta", type=float, default=1e-1)
    parser.add_argument("--lr_lgv", type=float, default=1e-3)
    parser.add_argument("--use_weight_decay", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--gamma", type=float, default=0.3)

    parser.add_argument("--bwd_to_fwd_ratio", type=float, default=2.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--eval_batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10000)

    parser.add_argument("--module", type=str, default="pismlp", choices=("pismlp", "mlp", "ddsmlp"))
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--init_log_Z", type=str, default="0.0")  # "iw_elbo", "elbo" or float
    parser.add_argument("--precision", type=str, default="float", choices=("float", "double"))

    ################################################################
    ### Diffusion process
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument(
        "--reference_process",
        type=str,
        default="pinned_brownian",
        choices=("ou", "pinned_brownian"),
    )
    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument("--init_std", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=6.0)
    ################################################################

    ################################################################
    ### MLP parameters
    parser.add_argument("--hidden_dim", type=int, default=64)
    # parser.add_argument("--s_emb_dim", type=int, default=64)
    # parser.add_argument("--t_emb_dim", type=int, default=64)
    # parser.add_argument("--harmonics_dim", type=int, default=64)
    parser.add_argument("--joint_layers", type=int, default=2)
    parser.add_argument("--no_zero_init", action="store_false", dest="zero_init")
    parser.add_argument("--share_embeddings", action="store_true", default=False)
    parser.add_argument("--flow_hidden_dim", type=int, default=64)
    # parser.add_argument("--flow_s_emb_dim", type=int, default=64)
    # parser.add_argument("--flow_t_emb_dim", type=int, default=64)
    # parser.add_argument("--flow_harmonics_dim", type=int, default=64)
    parser.add_argument("--flow_layers", type=int, default=2)
    parser.add_argument("--lp", action="store_true", default=False)
    parser.add_argument("--lp_scaling_per_dimension", action="store_true", default=False)
    parser.add_argument("--lgv_layers", type=int, default=3)
    parser.add_argument("--no_clipping", action="store_false", dest="clipping")
    parser.add_argument("--out_clip", type=float, default=1e4)
    parser.add_argument("--lgv_clip", type=float, default=1e2)
    parser.add_argument("--learn_pb", action="store_true", default=False)
    parser.add_argument("--pb_scale_range", type=float, default=0.1)
    parser.add_argument("--learn_variance", action="store_true", default=False)
    parser.add_argument("--log_var_range", type=float, default=4.0)

    parser.add_argument("--no_partial_energy", action="store_false", dest="partial_energy")
    parser.add_argument("--no_learn_beta", action="store_false", dest="learn_beta")
    ################################################################

    ################################################################
    ### For replay buffer
    parser.add_argument("--use_buffer", action="store_true", default=False)
    parser.add_argument("--buffer_size", type=int, default=-1)  # 100 * batch_size by default
    # prioritization
    parser.add_argument(
        "--prioritization",
        type=str,
        default="none",
        choices=("none", "reward", "loss", "iw", "normalized_iw"),
    )
    # buffer sampling strategy  # TODO: support percentile-based sampling
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="systematic",
        choices=("multinomial", "stratified", "systematic", "rank"),
    )
    # low rank_k give steep priorization in rank-based replay sampling
    parser.add_argument("--rank_k", type=float, default=1e-2)
    # logr_lb for filtering out samples with extremely low reward values for numerical stability
    parser.add_argument("--logr_lb", type=float, default=-1e5)
    # prefill to wait before starting to sample from buffer
    parser.add_argument("--prefill_epochs", type=int, default=-1)
    # Adaptive tempering for buffer
    parser.add_argument("--buffer_target_ess", type=float, default=0.0)  # 0.0 has no effect
    ################################################################

    ################################################################
    ### For SMC
    parser.add_argument("--smc", action="store_true", default=False)
    parser.add_argument(
        "--smc_sampling",
        type=str,
        default="systematic",
        choices=("multinomial", "stratified", "systematic", "rank"),
    )
    parser.add_argument("--smc_resample_threshold", type=float, default=0.2)
    parser.add_argument("--smc_target_ess", type=float, default=0.05)
    parser.add_argument("--smc_freq", type=int, default=1)
    ################################################################

    ################################################################
    ### For MCMC
    parser.add_argument("--mcmc_type", type=str, default="none", choices=("none", "md", "mala"))
    parser.add_argument("--mcmc_freq", type=int, default=100)
    parser.add_argument("--mcmc_batch_size", type=int, default=100)
    parser.add_argument("--mcmc_n_steps", type=int, default=1000)
    parser.add_argument("--mcmc_burn_in", type=int, default=100)
    parser.add_argument("--mcmc_thinning", type=int, default=1)
    parser.add_argument("--mcmc_step_size", type=float, default=0.001)
    parser.add_argument("--mcmc_gamma", type=float, default=1.0)  # for MD
    ################################################################

    ################################################################
    ### Inverse temperature of the energy
    parser.add_argument("--invtemp", type=float, default=1.0)
    parser.add_argument("--no_invtemp_anneal", action="store_false", dest="invtemp_anneal")
    ################################################################

    ################################################################
    ### Eval & Plot
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_data_size", type=int, default=2000)
    parser.add_argument("--final_eval_data_size", type=int, default=2000)
    parser.add_argument("--no_full_eval", action="store_false", dest="full_eval")
    parser.add_argument("--full_eval_freq", type=float, default=0.1)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    parser.add_argument("--plot_freq", type=float, default=0.1)
    parser.add_argument("--no_plot_gt", action="store_false", dest="plot_gt")
    parser.add_argument("--plot_t_idx", type=int, nargs="+", default=[])
    ################################################################

    args = parser.parse_args()

    try:
        args.init_log_Z = float(args.init_log_Z)
    except ValueError:
        assert args.init_log_Z in ["iw_elbo", "elbo"]

    args.loss_type_str = args.loss_type
    if args.loss_type in ["db", "subtb", "tb-subtb"]:
        args.conditional_flow_model = True
        if args.partial_energy:
            args.loss_type_str = "fl-" + args.loss_type_str
        if args.loss_type == "subtb":
            if args.subtb_chunk_size > 0:
                assert args.num_steps % args.subtb_chunk_size == 0
                args.loss_type_str += f"-chunksize{args.subtb_chunk_size}"
            else:
                args.loss_type_str += f"-lambda{args.subtb_lambda}"
    else:
        args.conditional_flow_model = False
        args.partial_energy = False
        args.learn_beta = False

    if args.learn_pb:
        args.loss_type_str += "-learnpb"

    if args.lr_bwd is None:
        args.lr_bwd = args.lr_fwd

    if args.loss_type == "mle" or args.loss_type == "rev_kl":
        args.use_buffer = False

    if args.buffer_size == -1:
        args.buffer_size = 100 * args.batch_size

    if args.prefill_epochs == -1:
        args.prefill_epochs = int(min(100, args.buffer_size / args.batch_size))

    if args.full_eval_freq < 1:
        args.full_eval_freq = args.full_eval_freq * args.epochs
    if args.plot_freq < 1:
        args.plot_freq = args.plot_freq * args.epochs
    args.full_eval_freq = int(args.full_eval_freq)
    args.plot_freq = int(args.plot_freq)

    assert args.plot_freq % args.eval_freq == 0

    train(args)
