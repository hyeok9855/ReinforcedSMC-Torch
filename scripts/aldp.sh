#! /bin/bash

ALG=${1}  # pis, tb, tb_buf, tb_iwbuf, tb_buf_mcmc, tb_iwbuf_mcmc, tb-subtb_buf_smc, tb-subtb_iwbuf_smc, tb-subtb_buf_smc_mcmc, tb-subtb_iwbuf_smc_mcmc
SEED=${2}

COMMON_ARGS="--energy_name aldp --init_log_Z iw_elbo --epochs 40000"
COMMON_ARGS="$COMMON_ARGS --hidden_dim 1024 --flow_hidden_dim 256 --batch_size 400 --eval_data_size 2000 --final_eval_data_size 100000 --no_full_eval"
COMMON_ARGS="$COMMON_ARGS --reference_process ou --init_std 1.0 --t_scale 1.0 --num_steps 128"
COMMON_ARGS="$COMMON_ARGS --lr_fwd 0.0005 --lr_flow 0.0005 --lr_logZ 0.05 --lr_beta 0.05 --use_scheduler --milestones 0.5 0.75 --gamma 0.3"
COMMON_ARGS="$COMMON_ARGS --exp_name aldp"

BUFFER_ARGS="--use_buffer --buffer_size 400000 --bwd_to_fwd_ratio 3.0 --prefill_epochs 100"
IWBUFFER_ARGS="$BUFFER_ARGS --prioritization iw --buffer_target_ess 0.05 --buffer_sampling systematic"
MCMC_ARGS="--mcmc_type md --mcmc_freq 100 --mcmc_n_steps 500 --mcmc_batch_size 200 --mcmc_burn_in 0 --mcmc_step_size 0.001 --mcmc_gamma 2.5 --mcmc_thinning 50"
SMC_ARGS="--smc --epochs 20000 --smc_target_ess 0.05 --smc_resample_threshold 0.2 --smc_sampling systematic"

if [ "$ALG" = "pis" ]; then  # Path Integral Sampler with reverse-KL loss
    ARGS="$COMMON_ARGS --loss_type pis"
elif [ "$ALG" = "tb" ]; then  # On-policy training with Trajectory Balance (TB) loss
    ARGS="$COMMON_ARGS --loss_type tb"
elif [ "$ALG" = "tb_buf" ]; then  # TB with unweighted buffer
    ARGS="$COMMON_ARGS --loss_type tb $BUFFER_ARGS"
elif [ "$ALG" = "tb_iwbuf" ]; then  # TB with importance-weighted buffer
    ARGS="$COMMON_ARGS --loss_type tb $IWBUFFER_ARGS"
elif [ "$ALG" = "tb_buf_mcmc" ]; then  # TB with unweighted buffer and MCMC buffer augmentation
    ARGS="$COMMON_ARGS --loss_type tb $BUFFER_ARGS $MCMC_ARGS"
elif [ "$ALG" = "tb_iwbuf_mcmc" ]; then  # TB with importance-weighted buffer and MCMC buffer augmentation
    ARGS="$COMMON_ARGS --loss_type tb $IWBUFFER_ARGS $MCMC_ARGS"
elif [ "$ALG" = "tb-subtb_buf_smc" ]; then  # TB/SubTB combined loss with unweighted buffer and SMC
    ARGS="$COMMON_ARGS --loss_type tb-subtb --subtb_chunk_size 32 $BUFFER_ARGS $SMC_ARGS"
elif [ "$ALG" = "tb-subtb_iwbuf_smc" ]; then  # TB/SubTB combined loss with importance-weighted buffer and SMC
    ARGS="$COMMON_ARGS --loss_type tb-subtb --subtb_chunk_size 32 $IWBUFFER_ARGS $SMC_ARGS"
elif [ "$ALG" = "tb-subtb_buf_smc_mcmc" ]; then  # TB/SubTB combined loss with unweighted buffer, SMC, and MCMC buffer augmentation
    ARGS="$COMMON_ARGS --loss_type tb-subtb --subtb_chunk_size 32 $BUFFER_ARGS $SMC_ARGS $MCMC_ARGS"
elif [ "$ALG" = "tb-subtb_iwbuf_smc_mcmc" ]; then  # TB/SubTB combined loss with importance-weighted buffer, SMC, and MCMC buffer augmentation
    ARGS="$COMMON_ARGS --loss_type tb-subtb --subtb_chunk_size 32 $BUFFER_ARGS $SMC_ARGS $MCMC_ARGS --smc_freq 2"  # set smc_freq to 2 to make NFE comparable
else  # Invalid algorithm
    echo "Invalid algorithm: $ALG"
    exit 1
fi

python train.py $ARGS --seed $SEED