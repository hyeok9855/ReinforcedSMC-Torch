#! /bin/bash
# Launcher for the benchmark grid: every METHOD x TARGET x SEED below, spread over $GPUS.
# To change what runs, edit the block right below -- there are no command-line arguments.
#
#   bash scripts/benchmark.sh          # launch everything
#   DRY=1 bash scripts/benchmark.sh    # print the commands instead of running them
#
# Each run logs to $LOG_DIR/<tag>.log and, on success, touches $LOG_DIR/<tag>.done, which is
# skipped next time. So re-running after a crash resumes instead of starting over.
#
# Evaluation note: models are scored against ground truth from energy.cached_sample(), i.e. fresh
# draws from the analytic target seeded by --seed. This is NOT the pinned 1step_energy_sampler
# protocol -- utils/reference_utils.py can load energies/data/references/<T>/seed_<S>/reference.npy,
# but nothing imports it and those .npy files (git-lfs objects from a private repo) are absent here.
set -u
export PYTHONWARNINGS="ignore"
export TORCH_COMPILE=1

# ---------------------------------------------------------------- what to run
METHODS="smc_buf smc_iwbuf ls_buf ls_iwbuf"
TARGETS="gmm40_d2 gmm40_d2_40 gmm40_d10 gmm40_d10_40 funnel_d10 manywell_d32 manywell_d64 nice_mnist nice_fashion"
SEEDS="0 1 2 3 4"

GPUS=(0 1 2 3)
JOBS_PER_GPU=2
LOG_DIR=runs
SAVE_DIR=checkpoints
# -----------------------------------------------------------------------------

# per-target energy, dimension and initial scale (init_std matches the JAX configs)
target_spec() {
    case "$1" in
        gmm40_d2)     ENERGY=gmm40;              NDIM=2;   INIT_STD=20.0 ;;
        gmm40_d2_40)  ENERGY=gmm40;              NDIM=2;   INIT_STD=40.0 ;;
        gmm40_d10)    ENERGY=gmm40;              NDIM=10;  INIT_STD=20.0 ;;
        gmm40_d10_40) ENERGY=gmm40;              NDIM=10;  INIT_STD=40.0 ;;
        funnel_d10)   ENERGY=funnel;             NDIM=10;  INIT_STD=1.0  ;;
        manywell_d32) ENERGY=manywell;           NDIM=32;  INIT_STD=1.0  ;;
        manywell_d64) ENERGY=manywell;           NDIM=64;  INIT_STD=1.0  ;;
        nice_mnist)   ENERGY=nice_mnist;         NDIM=196; INIT_STD=5.0  ;;
        nice_fashion) ENERGY=nice_fashion_mnist; NDIM=784; INIT_STD=5.0  ;;
        *) echo "Unknown target: $1" >&2; exit 1 ;;
    esac
}

mkdir -p "$LOG_DIR"
i=0

for m in $METHODS; do
    for t in $TARGETS; do
        target_spec "$t"

        ARGS="--energy_name $ENERGY --ndim $NDIM --init_std $INIT_STD"
        ARGS="$ARGS --reference_process ou --module ddsmlp --lp --num_steps 128"
        ARGS="$ARGS --batch_size 2000 --use_buffer"
        # Pinned rather than left to the entry points, which default it differently (0.0 in
        # train.py, 0.1 in train_ls.py) and would otherwise confound the SMC and LS arms.
        ARGS="$ARGS --clip_logZ_grad_norm_ratio 0.0"
        ARGS="$ARGS --exp_name benchmark"

        # bwd_to_fwd_ratio is what balances the NFE budget across arms: SMC draws ~1.67
        # rollouts/iter (1 fwd + 2x(smc + bwd) per 3 iters) and LS 1.0, so 25000 x 1.67 ~= 40000.
        SMC_LOSS="--loss_type tb-subtb --flow_hidden_dim 256 --subtb_n_chunks 32 --bwd_to_fwd_ratio 2.0 --epochs 25000 --smc"
        LS_LOSS="--loss_type tb --bwd_to_fwd_ratio 1.0 --epochs 40000"

        case "$m" in
            smc_buf)   ENTRY=train.py;    LOSS="$SMC_LOSS --prioritization none" ;;
            smc_iwbuf) ENTRY=train.py;    LOSS="$SMC_LOSS --prioritization iw"   ;;
            ls_buf)    ENTRY=train_ls.py; LOSS="$LS_LOSS --prioritization none"  ;;
            ls_iwbuf)  ENTRY=train_ls.py; LOSS="$LS_LOSS --prioritization iw"    ;;
            *) echo "Unknown method: $m" >&2; exit 1 ;;
        esac

        for s in $SEEDS; do
            tag="${m}_${t}_seed${s}"
            [ -f "$LOG_DIR/$tag.done" ] && continue
            gpu=${GPUS[$((i % ${#GPUS[@]}))]}
            i=$((i + 1))
            echo "CUDA_VISIBLE_DEVICES=$gpu python $ENTRY $ARGS $LOSS --seed $s --save_dir $SAVE_DIR" \
                 "> $LOG_DIR/$tag.log 2>&1 && touch $LOG_DIR/$tag.done || echo '[FAIL] $tag'"
        done
    done
done | {
    if [ "${DRY:-0}" = 1 ]; then
        cat
    else
        xargs -P $(( ${#GPUS[@]} * JOBS_PER_GPU )) -I CMD bash -c CMD
    fi
}
