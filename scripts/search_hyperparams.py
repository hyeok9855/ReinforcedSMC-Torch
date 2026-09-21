#!/usr/bin/env python
import argparse
import itertools
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define hyperparameter grid values
BWD_TO_FWD_RATIOS = [2.0, 3.0]
CLIP_GRAD_NORMS = [0.1]
INIT_LOG_ZS = ["iw_elbo"]
MODULE = ["pismlp", "ddsmlp"]
NUM_STEPS = [128, 256]
LR_FWD = [0.0005]

ALL_PARAMS = [BWD_TO_FWD_RATIOS, CLIP_GRAD_NORMS, INIT_LOG_ZS, MODULE, NUM_STEPS, LR_FWD]


def run_job(params, gpu_queue, epochs, dry_run):
    bwd_to_fwd, clip_grad_norm, init_log_Z, module, num_steps, lr_fwd, seed = params

    lr_flow = lr_fwd
    lr_logZ = lr_fwd * 100
    lr_beta = lr_fwd * 100
    clip_logZ_grad_norm_ratio = 0.0  # off: the per-sample Huber (logZ_huber_delta) replaces it

    # Get an available GPU from the queue
    gpu_id = gpu_queue.get()

    # Construct train command
    cmd = [
        "uv",
        "run",
        "python",
        "train.py",
        "--energy_name",
        "aldp",
        "--init_log_Z",
        init_log_Z,
        "--epochs",
        str(epochs),
        "--hidden_dim",
        "1024",
        "--joint_layers",
        "3",
        "--flow_hidden_dim",
        "512",
        "--batch_size",
        "400",
        "--eval_data_size",
        "2000",
        "--final_eval_data_size",
        "100000",
        "--no_full_eval",
        "--module",
        str(module),
        "--reference_process",
        "ou",
        "--init_std",
        "1.0",
        "--t_scale",
        "1.0",
        "--num_steps",
        str(num_steps),
        "--use_scheduler",
        "--milestones",
        "0.5",
        "0.75",
        "--gamma",
        "0.3",
        "--loss_type",
        "tb-subtb",
        "--subtb_n_chunks",
        "4",
        "--use_buffer",
        "--prefill_epochs",
        "100" if not dry_run else "1",
        "--prioritization",
        "iw",
        "--buffer_target_ess",
        "0.05",
        "--buffer_sampling",
        "systematic",
        "--smc",
        "--smc_target_ess",
        "0.05",
        "--smc_resample_threshold",
        "0.2",
        "--smc_sampling",
        "systematic",
        "--smc_every",
        "3",
        "--buffer_size",
        "400000",
        "--bwd_to_fwd_ratio",
        str(bwd_to_fwd),
        "--lr_fwd",
        str(lr_fwd),
        "--lr_flow",
        str(lr_flow),
        "--lr_logZ",
        str(lr_logZ),
        "--lr_beta",
        str(lr_beta),
        "--clip_grad_norm",
        str(clip_grad_norm),
        "--clip_logZ_grad_norm_ratio",
        str(clip_logZ_grad_norm_ratio),
        "--seed",
        str(seed),
    ]

    if dry_run:
        cmd.append("--disable_wandb")
        cmd.append("--eval_freq")
        cmd.append("1")
        cmd.append("--full_eval_freq")
        cmd.append("1")
        cmd.append("--plot_freq")
        cmd.append("1")
        cmd.append("--final_eval_data_size")
        cmd.append("100")

    job_name = f"ratio{bwd_to_fwd}_clip{clip_grad_norm}_init{init_log_Z}_module{module}_steps{num_steps}_lr{lr_fwd}"  # noqa: E501
    cmd.extend(["--exp_name", f"aldp_search_{job_name}"])

    # Create logs directory
    log_dir = "search_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{job_name}_seed{seed}.log")

    print(f"[LAUNCH] {job_name} on GPU {gpu_id} (Logs: {log_file})")

    # Configure environment with CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"[SUCCESS] {job_name} completed successfully on GPU {gpu_id}")
        else:
            print(f"[FAILURE] {job_name} failed with exit code {result.returncode} on GPU {gpu_id}")
    except Exception as e:
        print(f"[ERROR] {job_name} encountered an error: {e}")
    finally:
        # Release the GPU back to the queue
        gpu_queue.put(gpu_id)


def main(args):
    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    # Determine the parameter combinations to run
    if args.dry_run:
        # Run a single config for verification
        combinations = [tuple(p[0] for p in ALL_PARAMS) + (seed_list[0],)]
        epochs = 2
        print("--- RUNNING IN DRY-RUN MODE (1 configuration, 2 epochs) ---")
    else:
        combinations = list(itertools.product(*ALL_PARAMS, seed_list))
        epochs = args.epochs

    jobs_per_gpu = max(1, args.jobs_per_gpu)
    max_workers = len(gpu_list) * jobs_per_gpu

    print(f"Total jobs to run: {len(combinations)}")
    print(f"GPUs available: {gpu_list}")
    print(f"Concurrent jobs per GPU: {jobs_per_gpu} (max {max_workers} parallel jobs)")

    # Each GPU ID is enqueued jobs_per_gpu times so up to that many jobs can share a GPU.
    gpu_queue = queue.Queue()
    for _ in range(jobs_per_gpu):
        for g in gpu_list:
            gpu_queue.put(g)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for params in combinations:
            futures.append(executor.submit(run_job, params, gpu_queue, epochs, args.dry_run))

        # Wait for all to complete
        for future in futures:
            future.result()

    print("All jobs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALDP Hyperparameter Grid Search")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs to run on")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=2,
        help="Maximum number of concurrent experiments per GPU",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run a single fast test job (1 epoch, no wandb)"
    )

    args = parser.parse_args()

    main(args)
